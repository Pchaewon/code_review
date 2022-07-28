from pathlib import Path ##
from itertools import chain ##
import os
import random

from sklearn.utils import compute_class_weight
'''
[compute_class_weight(class_weight, * , class, y)]
1. class_weight : dict, 'balanced' or None
'balanced' 면, class weights는 n_samples/(n_classes*np.bincount(y))
Dictionary가 주어지면, key는 클래스 value는 해당 클래스의 가중치
None이면, 클래스 가중치는 균일

2. class : ndarray
원래 클래스 레이블 y_org와 함께 np.unique(y_org)에 의해 제공된 데이터에서 발생하는 클래스 배열

3. y : array-like of shape(n_sampels, )
샘플당 원래 클래스 라벨들의 배열
====
[returns]
1. class_weight_vect : ndarray of shape(n_class,)
i번째 클래스에 대한 class weight vect[i] 가중치 배열
'''

from munch import Munch
from PIL import Image ## Python Imaging Library
import numpy as np

import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler ##주어진 확률(가중치)로 샘플링
from torchvision import transforms
from torchvision.datasets import ImageFolder ##


'''
dname: directory name
fname: file name
chain() : *(asterick/star syntax)로 각 튜플을 풀어서 붙임.
Path() : 경로를 Path 객체로 생성. 상대/절대 상관없음.
glob() : 와일드 카드 문자로 파일 이름 패턴 지정
rglob() : recursive globbing
'''

## list directory :  파일 이름 출력과 묶어서 list 만들기
def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.'+ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames

## Dataset 기본 설정 : x
class DefaultDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples = listdir(root) # file 열고 list
        self.samples.sort() # 정렬
        self.transform = transform
        self.targets = None 
        
    def __getitem__(self, index):
        fname = self.samples[index] # file 읽은 list를 입력된 index에 따라 넣기
        img = Image.open(fname).convert('RGB') # RGB로 변환
        ## transform 사용하기
        if self.transform is not None:
            img = self.transform(img)
        return img
    
    # list 길이 반환
    def __len__(self):
        return len(self.samples)

## 참조 데이더셋 설정 :  x_ref, x_ref2, y_trg
class ReferenceDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples, self.targets = self._make_dataset(root)
        self.transform = transform
    
    '''
    os.listdir() :지정한 디렉토리 내 모든 파일과 디렉토리 리스트를 반환.
    random.sample(): sequence에서 지정한 숫자만큼의 요소들을 랜덤으로 뽑아 리스트로 반환해주는 함수
    zip(): 양측에 있는 데이터를 하나씩 차례로 짝 지어줌.
    '''
    def _make_dataset(self, root):
        domains = os.listdir(root)
        fnames, fnames2, labels = [], [], []
        for idx, domain in enumerate(sorted(domains)):
            class_dir = os.path.join(root, domain) ## 경로 + 도메인 
            cls_fnames = listdir(class_dir) ## class file name
            fnames += cls_fnames 
            fnames2 += random.sample(cls_fnames, len(cls_fnames)) ## random sample해서 넣기
            labels += [idx] * len(cls_fnames) 
        return list(zip(fnames, fnames2)), labels
    
    def __getitem__(self, index):
        fname, fname2 = self.samples[index]
        label = self.targets[index]
        img = Image.open(fname).convert('RGB') # original img file
        img2 = Image.open(fname2).convert('RGB') #random sample img file
        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)
        return img, img2, label

'''
bincount() : ndarray 요소들 숫자세기
NotImplementedError : "아직 구현하지 않은 부분입니다" 라는 오류를 강제로 발생
lambda : [매개변수:표현식] 형태
'''

'''
[DataLoader(): 데이터 셋에 대해 반복 가능한 python]
1. 중요 인수
- map style dataset : __getitem__(), __len__()구현하는 데이터 셋. 인덱스/키에서 데이터 샘플까지의 map을 나타냄.
  i번째 이미지와 해당 레이블 dataset[idx]로 읽을 수 있음.
- iterable style dataset : __iter__()구현. 데이터 샘플에 대한 반복을 나타내는 하위 클래스 인스턴스
  iter(dtataset)이로 호출
2. 매개변수 
- shuffle : True, 모든 epoch에서 데이터를 다시 섞도록 설정
- sampler : 데이터 세트에서 샘플을 추출하는 전략
- batch_sampler : 인덱스 배치를 한번에 반환. batch_size, shuffle, sampler, drop_last와 상호배타적
- num_workers : 데이터 로드에 사용할 하위 프로세스 수
- collate_fn : 샘플 목록을 병합하여 Tensor의 미니 배치 형성
- pin_memory : True인 경우 데이터 로더는 Tensor를 반환하기 전 device/CUDA 고정 메모리에 복사
- drop_last : True 데이터 셋 크기가 배치크기로 나눌 수 없는 경우 마지막 불완전한 배치를 삭제하도록 설정
'''
def _make_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights)) ## 가중치 설정

## train set load
def get_train_loader(root, which='source', img_size=256, 
                     batch_size=8, prob=0.5, num_workers=4):
    print('Preparing DataLoader to fetch %s images'
          'during the training phase...' % which) 
    
    crop = transforms.RandomResizedCrop(
        img_size, scale=[0.8, 1.0], ratio=[0.9, 1.1]
    )
    random_crop = transforms.Lambda(
        lambda x: crop(x) if random.random() < prob else x
    )

    transform = transforms.Compose([
        random_crop,
        transforms.Resize([img_size, img_size]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    if which == 'source':
        dataset = ImageFolder(root, transform) ## return tuple(img, class) : x, y_org
    elif which == 'reference':
        dataset = ReferenceDataset(root, transform) ## x_ref, x_ref2, y_trg
    else : 
        raise NotImplementedError
    
    sampler = _make_balanced_sampler(dataset.targets) ## y_org, y_trg labels
    return data.DataLoader(dataset = dataset,
                           batch_size = batch_size,
                           sampler = sampler,
                           num_workers = num_workers,
                           pin_memory = True,
                           drop_last = True)

## eval(valid) set load
def get_eval_loader(root, img_size=256, batch_size=32, 
                    imagenet_normalize=True, shuffle=True,
                    num_workers=4, drop_last=False):
    print('Preparing DataLoader for the evaluation phase...')
    if imagenet_normalize:
        height, width = 299, 299
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        height, width = img_size, img_size
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.Resize([height, width]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    dataset = DefaultDataset(root, transform=transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=drop_last)

## test set load    
def get_test_loader(root, img_size=256, batch_size=32, 
                    shuffle=True, num_workers=4):
    print('Preparing DataLoader for the generation phase...')
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    dataset = ImageFolder(root, transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True)


'''
https://dojang.io/mod/page/view.php?id=2408
[내장함수]
next() : next(반복가능한객체, 기본값). 이터레이터에서 값을 차례대로 꺼냄.
iter() : iter(호출가능한객체, 반복을끝낼값). 반복 가능한 객체에서 이터레이터를 반환.
'''    
class InputFetcher:
    def __init__(self, loader, loader_ref=True, latent_dim=16, mode=''):
        self.loader = loader
        self.loader_ref = loader_ref
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode
    
    ## x, y    
    def _fetch_inputs(self):
        try:
            x, y = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader)
            x, y = next(self.iter)
        return x, y
    
    ## x_ref, x_ref2, y'
    def _fetch_refs(self):
        try:
            x, x2, y = next(self.iter_ref)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader_ref)
            x, x2, y = next(self.iter_ref)
        return x, x2, y
    
    def __next__(self):
        x, y = self._fetch_inputs() ## source
        
        ## train
        if self.mode == 'train':
            x_ref, x_ref2, y_ref = self._fetch_refs() ## reference
            ## latent trarget
            ## 잠재 벡터 : 랜덤하게 뽑는다
            z_trg = torch.randn(x.size(0), self.latent_dim)
            z_trg2 = torch.randn(x.size(0), self.latent_dim)
            ## input dictionary
            inputs = Munch(x_src=x, y_src=y, y_ref=y_ref,
                           x_ref=x_ref, x_ref2=x_ref2,
                           z_trg=z_trg, z_trg2=z_trg2)
        ## validation
        elif self.mode == 'val':
            x_ref, y_ref = self._fetch_inputs()
            inputs = Munch(x_src=x, y_src=y,
                           x_ref=x_ref, y_ref=y_ref)
        ## test
        elif self.mode == 'test':
            inputs = Munch(x=x, y=y)
        else:
            raise NotImplementedError

        ## Dictionary key, value 반환
        return Munch({k: v.to(self.device)
                      for k, v in inputs.items()})
            