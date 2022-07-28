import os 
from os.path import join as ospj
import time 
import datetime
from munch  import Munch

# Munch library 설명
'''
속성 스타일 액세스를 제공하는 Python 사전(자바스크립트 개체)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

## core 파일
from core.model import build_model # main.py
from core.checkpoint import CheckpointIO # checkpoint.py
from core.data_loader import InputFetcher # data_loader.py
import core.utils as utils # utils.py

## metrics파일 
from metrics.eval import calculate_metrics # eval.py

## Solver 클래스
class Solver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args ## arguments
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') ## 장치 설정

        self.nets, self.nets_ema = build_model(args)
        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        for name, module in self.nets_ema.items():
            setattr(self, name + '_ema', module)

        ## *args, **kargs 설명
        '''
        함수의 파라미터 순서 : 일반변수, *변수, **변수
        *변수 -> 여러개가 argument로 들어올 때, 함수 내부에서는 해당 변수를  'tuple'로 처리
        **변수 -> 키워드='' 로 입력할 경우에 그것을 각각 키와 값으로 가져오는 '딕셔너리'로 처리
        '''
        ## 훈련
        if args.mode == 'train':
            self.optims = Munch() ## dictionary로 저장
            ## network 설정
            for net in self.nets.keys():
                if net == 'fan': ## FAN Network면 
                    continue ## 아래 코드들 실행 하지 않고 for문 계속
               
                ## FAN Network가 아니면 Adam 사용
                self.optims[net] = torch.optim.Adam(
                    params=self.nets[net].parameters(),
                    lr=args.f_lr if net == 'mapping_network' else args.lr, ## mapping 1e-6, 아니면 1e-4
                    betas=[args.beta1, args.beta2], ## 0.0, 0.99
                    weight_decay=args.weight_decay) ## 1e-4

            ## checkpoint
            self.ckptios = [
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets.ckpt'), data_parallel=True, **self.nets),
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), data_parallel=True, **self.nets_ema),
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_optims.ckpt'), **self.optims)]
        ## 훈련 모드 아니면
        else:
            self.ckptios = [CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), data_parallel=True, **self.nets_ema)]

        self.to(self.device)
        ## bias 초기화
        for name, network in self.named_children():
            # Do not initialize the FAN parameters
            if ('ema' not in name) and ('fan' not in name):
                print('Initializing %s...' % name)
                network.apply(utils.he_init) ## nn.Conv, nn.Linear의 bias초기화

    ## checkpoint 저장
    def _save_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.save(step)

    ## checkpoint 로드
    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)

    ## 기울기 초기화
    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    ## 훈련
    def train(self, loaders):
        args = self.args
        nets = self.nets
        nets_ema = self.nets_ema
        optims = self.optims

        # fetch random validation images for debugging
        ## img 가져오기
        ## dataloader.py
        fetcher = InputFetcher(loaders.src, loaders.ref, args.latent_dim, 'train')
        '''
        fetcher
        : x, y, y', x_ref, x_ref2, z_trg, z_trg2
        ''' 
        fetcher_val = InputFetcher(loaders.val, None, args.latent_dim, 'val')
        '''
        fetcher_val
        : x, y, x_ref, y_ref
        '''
        inputs_val = next(fetcher_val) ## next(반복가능한객체, 기본값). 이터레이터에서 값을 차례대로 꺼냄.
        
        ## 훈련 다시 시작
        # resume training if necessary 
        if args.resume_iter > 0:
            self._load_checkpoint(args.resume_iter)

        # remember the initial value of ds weight
        initial_lambda_ds = args.lambda_ds #1

        ## 훈련 시작
        print('Start training...')
        start_time = time.time()
        
        for i in range(args.resume_iter, args.total_iters): # 0 ~ 999999 반복
            # fetch images and labels
            inputs = next(fetcher) 
            '''
            fetcher
            : x, y, y', x_ref, x_ref2, z_trg, z_trg2
            ''' 
            
            ### 입력 요소들
            '''
            x : 원본이미지
            y : 이미지 x의 도메인
            y' : 타깃 도메인
            s' : Fy'(z), 타깃도메인에 대한 스타일 벡터
            s" : Ey(x), 이미지 x에서 도메인 y에 대해 추출한 스타일 벡터
            Dy : 도메인 y에 대한 Discriminator
            '''
            ## 스타일 인코더 입력 이미지와 대응하는 도메인 y에 대해 스타일 코드 생성
            x_real, y_org = inputs.x_src, inputs.y_src  # x, y
            ## 매핑 네트워크: 도메인 y와 잠재벡터 z가 주어질 경우 스타일 코드 생성
            x_ref, x_ref2, y_trg = inputs.x_ref, inputs.x_ref2, inputs.y_ref ## 참조 이미지, y'
            z_trg, z_trg2 = inputs.z_trg, inputs.z_trg2 ## 랜덤 잠재벡터

            ## 원본 이미지 mask
            masks = nets.fan.get_heatmap(x_real) if args.w_hpf > 0 else None

            ### 판별자 훈련
            ## Adverserial Loss
            ## 원본과 생성 이미지 비교 : x, y, y', z
            d_loss, d_losses_latent = compute_d_loss( 
                nets, args, x_real, y_org, y_trg, z_trg=z_trg, masks=masks) 
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            ## x, y, y', x'
            d_loss, d_losses_ref = compute_d_loss(
                nets, args, x_real, y_org, y_trg, x_ref=x_ref, masks=masks) ## x, y, y',z
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            ## 생성자 훈련
            ## x, y, y', z=[z1,z2]
            g_loss, g_losses_latent = compute_g_loss(
                nets, args, x_real, y_org, y_trg, z_trgs=[z_trg, z_trg2], masks=masks)
            self._reset_grad()
            g_loss.backward()
            optims.generator.step() ## G(x, Fy'(z1)), G(x, Fy'(z2))
            optims.mapping_network.step() ## Fy'()
            optims.style_encoder.step() ## Ey()

            ## x, y, y', x'=[x'1, x'2]
            g_loss, g_losses_ref = compute_g_loss(
                nets, args, x_real, y_org, y_trg, x_refs=[x_ref, x_ref2], masks=masks)
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()

            # compute moving average of network parameters
            moving_average(nets.generator, nets_ema.generator, beta=0.999)
            moving_average(nets.mapping_network, nets_ema.mapping_network, beta=0.999)
            moving_average(nets.style_encoder, nets_ema.style_encoder, beta=0.999)

            # decay weight for diversity sensitive loss
            if args.lambda_ds > 0:
                args.lambda_ds -= (initial_lambda_ds / args.ds_iter)

            # print out log info
            if (i+1) % args.print_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i+1, args.total_iters)
                all_losses = dict()
                for loss, prefix in zip([d_losses_latent, d_losses_ref, g_losses_latent, g_losses_ref],
                                        ['D/latent_', 'D/ref_', 'G/latent_', 'G/ref_']):
                    for key, value in loss.items():
                        all_losses[prefix + key] = value
                all_losses['G/lambda_ds'] = args.lambda_ds
                log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
                print(log)

            # generate images for debugging
            if (i+1) % args.sample_every == 0:
                os.makedirs(args.sample_dir, exist_ok=True)
                utils.debug_image(nets_ema, args, inputs=inputs_val, step=i+1)

            # save model checkpoints
            if (i+1) % args.save_every == 0:
                self._save_checkpoint(step=i+1)

            ## GAN 평가방법
            '''
            1. Quality : 실제 이미지와 유사한 이미지가 생성되는가
            2. Diversity : 여러 noise vector에 걸쳐 다양한 이미지가 생성되는가

            metrics
            1. Inception Score, IS
                ImageNet pretrained model인 inception v3로 GAN 측정
                1)조건부 확률로 생성된 이미지 x에 대해 어떤 클래스에 속할지 예측하는 것
                2)주변확률
                고품질 이미지를 생성할수록 하나의 클래스에만 속할 확률이 높아 엔트로피가 낮음.
            => 다양한 이미지를 생성한다면 엔트로피가 uniform하게 나옴. 높은 엔트로피.

            2. Frechet Inception distance, FID
                ImageNet을 pretrain된 inception v3 모델필요.
                feature extractor로 사용하여 실제 이미지와 생성된 이미지 사이 activation map 추출.
                Activation map으로 multivariate gaussian distribution 구할 수 있음.
                실제와 생성 이미지의 확률 분포 사이의 W-2 distance를 구해 FID 구함
            => FID가 낮으면 high-quality이미지 생성

            3. Learned Perceptual Image Patch Similarity, LPIPS
                AlexNEt, VGG, SqueezeNet 사용
                유사도를 사람의 인식에 기반하여 측정
            => 낮은 LPIPS는 두 이미지가 perceptually smilar하다고 봄
            '''
            # compute FID and LPIPS if necessary
            if (i+1) % args.eval_every == 0:
                calculate_metrics(nets_ema, args, i+1, mode='latent')
                calculate_metrics(nets_ema, args, i+1, mode='reference')

    @torch.no_grad()
    def sample(self, loaders):
        args = self.args
        nets_ema = self.nets_ema
        os.makedirs(args.result_dir, exist_ok=True)
        self._load_checkpoint(args.resume_iter)

        src = next(InputFetcher(loaders.src, None, args.latent_dim, 'test'))
        ref = next(InputFetcher(loaders.ref, None, args.latent_dim, 'test'))

        fname = ospj(args.result_dir, 'reference.jpg')
        print('Working on {}...'.format(fname))
        utils.translate_using_reference(nets_ema, args, src.x, ref.x, ref.y, fname)

        fname = ospj(args.result_dir, 'video_ref.mp4')
        print('Working on {}...'.format(fname))
        utils.video_ref(nets_ema, args, src.x, ref.x, ref.y, fname)

    @torch.no_grad()
    def evaluate(self):
        args = self.args
        nets_ema = self.nets_ema
        resume_iter = args.resume_iter
        self._load_checkpoint(args.resume_iter)
        calculate_metrics(nets_ema, args, step=resume_iter, mode='latent')
        calculate_metrics(nets_ema, args, step=resume_iter, mode='reference')

## discriminator loss 
## x, y, y', z, x'
def compute_d_loss(nets, args, x_real, y_org, y_trg, z_trg=None, x_ref=None, masks=None):
    assert (z_trg is None) != (x_ref is None) ## 잠재벡터와 참조 이미지가 같으면 안됨
    
    # 진짜 이미지
    x_real.requires_grad_() 
    out = nets.discriminator(x_real, y_org) ## x, y 
    loss_real = adv_loss(out, 1) ## 1
    loss_reg = r1_reg(out, x_real) ## 생성된 이미지와 실제 이미지 간 loss

    
    with torch.no_grad():
        if z_trg is not None:
            ## 매핑 네트워크
            s_trg = nets.mapping_network(z_trg, y_trg) ## 잠재벡터 z, 타깃 도메인 s'
        else:  # x_ref is not None
            ## 스타일 인코더
            s_trg = nets.style_encoder(x_ref, y_trg) # 참조 이미지 x', 타깃도메인 s"
    
    ## 판별자 adverserial loss : Ex,y[logDy(x)] + Ex,y',z[log(1-Dy'(g(x,s')))]    : log1+log1=0
        x_fake = nets.generator(x_real, s_trg, masks=masks) ## G(x, s') 
    out = nets.discriminator(x_fake, y_trg)  ## 가짜 이미지, 타깃 도메인 D(G(x,s'),y')
    loss_fake = adv_loss(out, 0) ## 가짜 이미지를 가짜라고 판별
    
    ## 전체 objective Function
    loss = loss_real + loss_fake + args.lambda_reg * loss_reg ##  256
    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item())

## generator loss
## x, y, y', z, x'
def compute_g_loss(nets, args, x_real, y_org, y_trg, z_trgs=None, x_refs=None, masks=None):
    assert (z_trgs is None) != (x_refs is None)
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    if x_refs is not None:
        x_ref, x_ref2 = x_refs

    # Mapping z, y',   Style En x',y' 
    if z_trgs is not None:
        s_trg = nets.mapping_network(z_trg, y_trg)
    else:  # x_ref is not None
        s_trg = nets.style_encoder(x_ref, y_trg)

    ## 생성자 adversarial loss : Ex,y',z[log(1-Dy'(g(x,s')))] : log0 = -infinity
    x_fake = nets.generator(x_real, s_trg, masks=masks) ## G(x, s') 
    out = nets.discriminator(x_fake, y_trg) ## 가짜 이미지, 타깃 도메인 D(G(x,s'),y')
    loss_adv = adv_loss(out, 1) ## 가짜 이미지를 진짜로 속임

    # style reconstruction loss : Ex,y',z[||s'-Ey'(G(x,s'))||1]
    s_pred = nets.style_encoder(x_fake, y_trg) ## Ey'(G(x,s'), y') 
    loss_sty = torch.mean(torch.abs(s_pred - s_trg)) ## Ex,y',z[||s'-Ey'(G(x,s'))||1]
    '''
    abs() : 절댓값
    torch.mean() : 모든 요소의 평균값을 반환
    '''
    
    # Mapping z2, y,   Style En x'2, y'
    if z_trgs is not None:
        s_trg2 = nets.mapping_network(z_trg2, y_trg)
    else:  # x_ref is not None
        s_trg2 = nets.style_encoder(x_ref2, y_trg)
    # diversity sensitive loss : Ex,y',z1,z2[||G(x,s'1)-G(x,s'2)||1]
    x_fake2 = nets.generator(x_real, s_trg2, masks=masks)
    x_fake2 = x_fake2.detach() # 연산 기록으로 부터 분리하여 이후 연산들이 추적되는 것을 방지
    loss_ds = torch.mean(torch.abs(x_fake - x_fake2))

    # cycle-consistency loss : Ex,y,y',z[||x-G(G(x,s'),Ey(s,y))||1]
    masks = nets.fan.get_heatmap(x_fake) if args.w_hpf > 0 else None
    s_org = nets.style_encoder(x_real, y_org) ## Ey(x, y)
    x_rec = nets.generator(x_fake, s_org, masks=masks) ## G(G(x,s'),Ey(s,y))
    loss_cyc = torch.mean(torch.abs(x_rec - x_real)) 

    ## 전체 objective Function
    loss = loss_adv + args.lambda_sty * loss_sty \
        - args.lambda_ds * loss_ds + args.lambda_cyc * loss_cyc  ## 1, 1, 1, 1
    return loss, Munch(adv=loss_adv.item(),
                       sty=loss_sty.item(),
                       ds=loss_ds.item(),
                       cyc=loss_cyc.item())

def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)

'''
full_like(input,full_value)
- input : 출력 텐서의 크기를 결정
- full_value : 출력 텐서를 채울 숫자

binary_cross_entropy_with_logits(input, target)
- 타깃 로짓과 입력 로짓 간의 binary cross entropy를 측정하는 기능
'''
## adverserial loss : Ex,y[logDy(x)] + Ex,y',z[log(1-Dy'(g(x,s')))]
def adv_loss(logits, target):
    assert target in [1, 0] ## 참 또는 거짓
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    ## BCE(x) = -1/N sigma(yi * log(h(xi;0))) + (1-yi)log(1-h(xi;seta))
    return loss

## https://github.com/Yangyangii/GAN-Tutorial
def r1_reg(d_out, x_in): ## CelebA에 많이 하는 r1 regularization
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg



