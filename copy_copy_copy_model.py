'''
StarGAN2 core 모델 파일 정리
'''

import copy
import math

from munch import Munch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.wing import FAN

# U-Net의 왼쪽 
## ResNet Encoder
class ResBlk(nn.Module):
    # 초기 정의
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2), normalize=False, downsample=False):
        super().__init__()
        self.actv =actv # activation
        self.normalize = normalize # normalize
        self.downsample =downsample # U-Net의 Encoder부분에서 다운샘플링
        self.learned_sc = dim_in != dim_out # learned skip connection : 입출력 차원이 다르면 배우자.
        self._build_weights(dim_in, dim_out)
    
    # 가중치 설정    
    def _build_weights(self, dim_in, dim_out):
        # 두개의 컨볼루션으로 구성됨.
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1) # 1번째 컨볼루션 층 
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1) # 2번째 컨볼루션 층
        # 정규화
        if self.normalize:
            # 정규화는 Instance로 한다.
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True) 
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
            # 입력과 출력의 차원이 다르면 컨볼루션 
            if self.learned_sc:
                self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)
    
    # Skip Connection
    def _shortcut(self, x):
        if self.learned_sc :
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x,2)
        return x
   
    # Residual Block
    def residual(self,x):
        #정규화1
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        # 다운 샘플링
        if self.downsample:
            x=F.avg_pool2d(x,2)
        #정규화2
        if self.normalize:
            x= self.norm2(x)
        x = self.actv(x)            
        x = self.conv2(x)
        return x
    
    # 순방향
    def forward(self, x): # 원본 이미지
        x = self._shortcut(x) + self._residual(x) #  ResNet 기반 U-Net형성
        return x / math.sqrt(2) #unit variance
    
# U-Net의 오른쪽
## ResNet Decoder     
class AdainResBlk(nn.Module):
    # 초기 정의
    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0, actv=nn.LeakyReLU(0.2),upsample=False):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = upsample # ResDlk 다운 샘플링-> AdainResBlk 업 샘플링
        self.learned_sc = dim_in != dim_out # learned skip connection : 입출력 차원이 다르면 배우자.
        self._build_weights(dim_in, dim_out, style_dim)
    
    # 가중치 설정    
    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        # skip connection
        if self.learned_sc : 
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)
   
    # Residual Block의 shortcut
    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest') # 다시
        if self.learned_sc:
            x= self.conv1x1(x)    
        return x
    
    # Residual Block 
    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest') # 다시
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x
    # 순방향
    def forward(self, x, s): # 원본 이미지, 스타일
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2) # 
        return out

# HighPass Filter
## https://www.geeksforgeeks.org/implement-photoshop-high-pass-filter-hpf-using-opencv-in-python/
## 저주파를 감쇠하여 이미지를 선명하게 하는 데 사용
class HighPass(nn.Module):
    def __init__(self, w_hpf, device):
        super(HighPass, self).__init__()
        self.register_biffer('filter', 
                             torch.tensor([[-1, -1, -1],
                                           [-1, 8., -1],
                                           [-1, -1, -1]]))
    def forward(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, filter, padding=1, groups=x.size(1))

# 생성자
class Generator(nn.Module):
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512, w_hpf=1):
        super().__init__()
        dim_in = 2**14 // img_size
        self.img_size = img_size
        self.from_rgb = nn.Conv2d(3, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList() 
        self.decode = nn.ModuleList()
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 3, 1, 1, 0))
        
        # down/up-sampling blocks
        repeat_num = int(np.log2(img_size)) - 4
        if w_hpf > 0:
            repeat_num += 1
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            ## ResNet Encoder : Downsampling 이미지의 context 포착
            self.encode.append(ResBlk(dim_in, dim_out, normalize=True, downsample=True))
            ## ResNet Decoder : Upsampling  정확한 localization
            self.decode.insert(0, AdainResBlk(dim_out, dim_in, style_dim, w_hpf=w_hpf, upsample=True))
            dim_in = dim_out

        # bottleneck blocks
        for _ in range(2):
            self.encode.append(ResBlk(dim_out, dim_out, normalize=True))
            self.decode.insert(0, AdainResBlk(dim_out, dim_out, style_dim, w_hpf=w_hpf))
        
        if w_hpf > 0:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
            self.hpf = HighPass(w_hpf, device)
            
    def forward(self, x, s, masks=None):
        x = self.from_rgb(x)
        cache = {}
        for block in self.encode:
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                cache[x.size(2)] = x
            x = block(x)
        for block in self.decode:
            x = block(x, s)
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                mask = masks[0] if x.size(2) in [32] else masks[1]
                mask = F.interpolate(mask, size=x.size(2), mode='bilinear')
                x = x + self.hpf(mask * cache[x.size(2)])
        return self.to_rgb(x)

# Mapping Network 
# Style Code 생성기
class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=16, style_dim=64, num_domains=2):
        super.__init__()
        layers = []
        layers += [nn.Linear(latent_dim, 512)]
        layers += [nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(512,512)]
            layers += [nn.ReLU()]
        self.shared = nn.Sequential(*layers)
        
        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Sequential(nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512,512),
                                            nn.ReLU(),
                                            nn.Linear(512,512),
                                            nn.ReLU(),
                                            nn.Linear(512, style_dim))]
    def forward(self, z, y):  ## 잠재 벡터z, 타깃 도메인 y'
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1) # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y] # (batch, style_dim)
        return s ## list로 반환

# Style Encoder
# Style Code 생성기
class StyleEncoder(nn.Module):
    def __init__(self, img_size=256, style_dim=64, num_domains=2, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]
        
        repeat_num = int(np.log2(img_size))-2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out
        
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        self.shared = nn.Seuquential(*blocks)
        
        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Linear(dim_out, style_dim)]

    def forward(self, x, y):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]
        return s
    
## 판별기
class Discriminator(nn.Module)    :
    def __init__(self, img_size=256, num_domains=2, max_conv_dim=512):
       super().__init__()
       dim_in = 2**14 // img_size
       blocks = []
       blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]
       
       repeat_num = int(np.llog2(img_size)) - 2
       for _ in range(repeat_num):
           dim_out = min(dim_in*2, max_conv_dim)
           blocks += [ResBlk(dim_in, dim_out, downsample=True)]
           dim_in = dim_out
        
       blocks += [nn.LeakyReLU(0.2)]
       blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)] 
       blocks += [nn.LeakyReLU(0.2)]
       blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)] 
       self.main = nn.Sequential(*blocks)
    
    def forward(self, x, y):
        out = self.main(x)
        out = out.view(out.size(0), -1)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        out = out[idx, y]
        return out 


### 모델 생성    
def build_model(args):
    generator = nn.DataParallel(Generator(args.img_size, args.style_dim, w_hpf=args.w_hpf))
    mapping_network = nn.DataParallel(MappingNetwork(args.latent_dim, args.style_dim, args.num_domains))
    style_encoder = nn.DataParallel(StyleEncoder(args.img_size, args.style_dim, args.num_domains))    
    discriminator = nn.DataParallel(Discriminator(args.img_size, args.num_domians))
    generator_ema = copy.deepcopy(generator)
    mapping_network_ema = copy.deepcopy(mapping_network)
    style_encoder_ema = copy.deepcopy(style_encoder)
    
    
    nets = Munch(generator=generator,
                 mapping_network=mapping_network,
                 style_encoder=style_encoder,
                 discriminator=discriminator)
    nets_ema = Munch(generator=generator_ema,
                     mapping_network=mapping_network_ema,
                     style_encoder=style_encoder_ema)

    if args.w_hpf > 0:
        fan = nn.DataParallel(FAN(fname_pretrained=args.wing_path).eval())
        fan.get_heatmap = fan.module.get_heatmap
        nets.fan = fan
        nets_ema.fan = fan

    return nets, nets_ema
