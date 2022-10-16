import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
from glob import glob
#os.environ['CUDA_VISIBLE_DEVICES']='3'
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision import transforms
import torch 
from torch import nn
from PIL import Image
import open_clip
from collections.abc import Iterable
import logging
import datetime
from tqdm import tqdm
import torch.nn.functional as F


import glob
import os
import pandas as pd
from tqdm import tqdm
import torchvision
import albumentations as A


RESULT_DIR = 'result/'
RANDOM_SEED = 0
BATCH_SIZE = 32
EPOCHS = 3
INIT_LR = 0.02 / 80.
WEIGHT_DECAY = 1e-4
MILESTONES = [1]#[1]#
COMMENT = 'stage9_vith_280_pro10k_backbone'

def get_files_from_dir(dir):
    if not os.path.exists(dir):
        return ''

    file_paths = []

    for root, directories, files in os.walk(dir):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)

    return file_paths






############# initial DDP ###############

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")
rank = dist.get_rank()
print(f"Start running basic DDP example on rank {rank}.")
torch.cuda.set_device(rank)





train_csv = pd.read_csv('../products10k/train.csv')
train_csv.name = train_csv.name.map(lambda x: '../products10k/train/' + x)

file_list = train_csv['name'].values
label_list = train_csv['class'].values

classes_num = np.unique(label_list).shape[0]
##############################################
#        datasets and augmentation           #
##############################################
class shopeeDataset(Dataset):
    def __init__(self, img_path, label_list, transform, train_aug):
        self.img_path = img_path
        self.label_list = label_list
        self.transform = transform
        self.train_aug = train_aug
    def __len__(self,):
        return len(self.img_path)
    def __getitem__(self, index):
        img = self.img_path[index]
        label = self.label_list[index]
        img = plt.imread(img) #/ 255.
        
        if img.dtype == 'float32' or img.dtype == 'float64' or img.dtype == 'float16':
            img = img * 255
            img = img.astype('uint8')
            if img.shape[2] == 4:
                img = img[:,:,:3]
                
                
        #img = img.transpose(2,0,1)
        #print(img.shape)
        #img = Image.fromarray(img)
        
        img = self.train_aug(image=img)["image"]
        img = self.transform(img)
        return img, label

transform_train = transforms.Compose([
     transforms.ToPILImage(),
     #transforms.Resize((280,280)),
     #transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                                            std=[0.26862954, 0.26130258, 0.27577711])
 ])
transform_val = transforms.Compose([
     transforms.ToPILImage(),
     transforms.Resize((224,224)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                                            std=[0.26862954, 0.26130258, 0.27577711])
 ])

train_aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ImageCompression(quality_lower=99, quality_upper=100),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.7),
        A.Resize(280, 280),
        A.Cutout(max_h_size=int(280 * 0.4), max_w_size=int(280 * 0.4), num_holes=1, p=0.5),
    ])

datasets_train = shopeeDataset(file_list, label_list, transform_train, train_aug)

train_sampler = torch.utils.data.distributed.DistributedSampler(datasets_train)

dataloader_train = torch.utils.data.DataLoader(datasets_train, batch_size=BATCH_SIZE,\
                                               shuffle=False, num_workers=32, pin_memory=True,\
                                               drop_last=False, sampler = train_sampler)

print(len(dataloader_train))

##############################################
#             Model Preparation              #
##############################################
def set_freeze_by_idxs(model, idxs, freeze=True):
    if not isinstance(idxs, Iterable):
        idxs = [idxs]
    num_child = len(list(model.children()))
    idxs = tuple(map(lambda idx: num_child + idx if idx < 0 else idx, idxs))
    for idx, child in enumerate(model.children()):
        if idx not in idxs:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze

class AddMarginProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super(AddMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'



class add_layer_model(nn.Module):
    def __init__(self,backbone):
        super(add_layer_model, self).__init__()
        self.backbone = backbone
        self.fc1 = nn.Linear(1024,64)
        #self.bn = nn.BatchNorm1d(64)
        self.fc2 = AddMarginProduct(64,classes_num, s=30, m=0.65)# nn.Linear(64, classes_num, bias = False) #AddMarginProduct(64, classes_num, s=30, m=0.35)
        self.drpout = nn.Dropout(0.2)
    def forward(self, x, label):
        x = self.backbone(x)
        x = self.drpout(x)
        x = self.fc1(x)
        #x = self.bn(x)
        x = self.fc2(x, label)
        return x

    
backbone, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14-280')
weight_backbone = torch.load(  'vit-h-14-laion2b_s32b_b79k.pth'     )
'''
weight_clear = weight_backbone
positional_embedding = weight_clear['backbone.positional_embedding']
pos_embed_before = positional_embedding[:1,:]
pos_embed_after = positional_embedding[1:,:]
print(pos_embed_after.shape)
pos_embed_after = pos_embed_after.view(1,16,16,1024).permute(0,3,1,2)
pos_embed_after = torch.nn.functional.interpolate(pos_embed_after, size=(20,20), mode='bicubic') # 1，1024, 24,24
pos_embed_after = pos_embed_after.permute(0,2,3,1).view(20*20,1024)
pos_embed = torch.cat([pos_embed_before, pos_embed_after])
weight_clear['backbone.positional_embedding'] = pos_embed
'''

#backbone.load_state_dict(weight_backbone)
backbone = backbone.visual


model = add_layer_model(backbone)

### load pretrained weight

weight_try = torch.load(  os.path.join('result','stage8_vith_280_pro10k_last2layer.pth')     )
#weight_try = torch.load(  'vit-h-14-laion2b_s32b_b79k.pth'     )
weight_clear = {}
for i in weight_try.items():
    weight_clear[i[0].split('module.')[-1]] = i[1]

    
    

#weight_try.popitem('fc2.weight')
'''
positional_embedding = weight_clear['backbone.positional_embedding']
pos_embed_before = positional_embedding[:1,:]
pos_embed_after = positional_embedding[1:,:]
print(pos_embed_after.shape)
pos_embed_after = pos_embed_after.view(1,16,16,1280).permute(0,3,1,2)
pos_embed_after = torch.nn.functional.interpolate(pos_embed_after, size=(20,20), mode='bicubic') # 1，1024, 24,24
pos_embed_after = pos_embed_after.permute(0,2,3,1).view(20*20,1280)
pos_embed = torch.cat([pos_embed_before, pos_embed_after])
weight_clear['backbone.positional_embedding'] = pos_embed
'''
'''




weight_add_forlabel = torch.randn( (len(dataset_ali.classes),64) )
weight_add_forlabel = weight_add_forlabel.to(weight_clear['fc2.weight'].device)
nn.init.xavier_uniform_(weight_add_forlabel)
weight_clear['fc2.weight'] = torch.cat([weight_clear['fc2.weight'],weight_add_forlabel],0)
'''

model.load_state_dict(weight_clear , strict=True)
model.cuda()
# freeze backbone
set_freeze_by_idxs(model,[1,2])
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])







##############################################
#        Optimizer, Loss, Scheduler,...       #
##############################################
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),\
                            lr=INIT_LR,\
                            weight_decay=WEIGHT_DECAY)
criterian = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, MILESTONES, gamma=0.1, last_epoch=- 1, verbose=False)


##############################################
#                 Training                   #
##############################################


def train_one_epoch(dataloader, model, criterian): # return loss and acc1
    loss_total = 0
    for batch, (X, y) in tqdm(enumerate(dataloader)):
        # Compute prediction and loss
        X = X.cuda()
        y = y.cuda()
        pred = model(X, y)
        loss = criterian(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_total += loss.item()
        
        if (batch +1)%100 == 0 and rank == 0:
            print(f'Loss : {loss/batch}')
    return loss_total/batch


# we don't need validation for now
def val_one_epoch(model, criterian):
    raise NotImplementedError
ep_global = 0
def train():
    logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s:%(message)s',
                        filename= f'logs/{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}-{COMMENT}.txt',
                        level='INFO',
                        filemode='a'
    )
    #model.cuda()
    model.train()
    loss_total_train = 0
    acc1_total_train = 0
    
    for ep in tqdm(range(EPOCHS)):
        print(f'epoch {ep}')
        dataloader_train.sampler.set_epoch(ep)
        ep_global = ep
        loss = train_one_epoch(dataloader_train, model, criterian)
        # we don't need validation for now
        # val_one_epoch(model, criterian)
        #loss_total_train += loss
        #acc1_total_train += acc1
        if rank == 0:
            logging.info(f"epoch {ep}: train loss {loss}")
        scheduler.step()
    if rank == 0:
        logging.info(f"training done. {EPOCHS} epochs.")
    model.eval()
    if rank == 0:
        torch.save(model.state_dict(), os.path.join(RESULT_DIR, f'{COMMENT}.pth'))

train()