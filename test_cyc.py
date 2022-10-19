import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
import sys
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from datasets1 import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.autograd as autograd
from models import *


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=159)
parser.add_argument("--n_epochs", type=int, default=150)
parser.add_argument("--dataset_name", type=str, default="./data_drive")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=0.0002)
parser.add_argument("--b1", type=float, default=0.5)
parser.add_argument("--b2", type=float, default=0.999)
parser.add_argument("--decay_epoch", type=int, default=50)
parser.add_argument("--n_cpu", type=int, default=16)
parser.add_argument("--img_height", type=int, default=256)
parser.add_argument("--img_width", type=int, default=256)
parser.add_argument("--channels", type=int, default=3)
parser.add_argument("--sample_interval", type=int, default=1)
parser.add_argument("--checkpoint_interval", type=int, default=-1)
parser.add_argument("--n_residual_blocks", type=int, default=9)
parser.add_argument("--lambda_cyc", type=float, default=10.0)
parser.add_argument("--lambda_id", type=float, default=5.0)
opt = parser.parse_args()
print(opt)

# Create sample and checkpoint directories
os.makedirs("./%s" % opt.dataset_name, exist_ok=True)
os.makedirs("./%s" % "CycleGAN_model_test", exist_ok=True)
os.makedirs("./%s" % "images_test/", exist_ok=True)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()


cuda = torch.cuda.is_available()

if torch.cuda.is_available():
    print(torch.cuda.is_available())

    device = torch.device('cuda')
else:
    device = torch.device('cpu')


input_shape = (opt.channels, opt.img_height, opt.img_width)

# Initialize generator and discriminator
G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)
G_BA = GeneratorResNet(input_shape, opt.n_residual_blocks)
D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)
D_A1 = Discriminator(input_shape)
D_B1 = Discriminator(input_shape)

from torch.nn import DataParallel

if cuda:
    G_AB = DataParallel(G_AB).to(device)
    G_BA = DataParallel(G_BA).to(device)
    D_A = DataParallel(D_A).to(device)
    D_B = DataParallel(D_B).to(device)
    criterion_GAN.to(device)
    criterion_cycle.to(device)
    criterion_identity.to(device)

G_AB.load_state_dict(torch.load("./CycleGAN_model_drive/G_man_woman_%d.pth" % (opt.epoch)))
G_BA.load_state_dict(torch.load("./CycleGAN_model_drive/G_woman_man_%d.pth" % (opt.epoch)))
D_A.load_state_dict(torch.load("./CycleGAN_model_drive/D_man_%d.pth" % (opt.epoch)))
D_B.load_state_dict(torch.load("./CycleGAN_model_drive/D_woman_%d.pth" % (opt.epoch)))


Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Image transformations
transforms_ = [
    transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC),
    transforms.RandomCrop((opt.img_height, opt.img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

transforms_1 = [
    transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

transforms_2 = [
    transforms.Resize(int(opt.img_height), Image.BICUBIC),
    transforms.RandomCrop((opt.img_height, opt.img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
"""
# Training data loader
dataloader = DataLoader(

    Dataset("./%s" % opt.dataset_name, transforms_=transforms_, unaligned=True),
    batch_size=opt.batch_size,
    shuffle=True
)
"""


transform_to_image = transforms.Compose(
        [
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.ToPILImage(),
        ]
    )




#print(len(dataloader))

# Test data loader
val_dataloader = DataLoader(
    Dataset("./%s" % opt.dataset_name, transforms_=transforms_, unaligned=True, mode="test"),
    batch_size=1,
    shuffle=False
)




def sample_images(epoch):
    #os.makedirs("./CycleGAN_model/%s/%s" % (opt.dataset_name,epoch), exist_ok=True)
    """Saves a generated sample from the test set"""
    with torch.no_grad():
        for i, imgs in enumerate(val_dataloader):
            os.makedirs("./images_test/%d/" % epoch, exist_ok=True)

            G_AB.eval()
            G_BA.eval()
            real_A = Variable(imgs["A"].type(Tensor))
            fake_B = G_AB(real_A)
            real_B = Variable(imgs["B"].type(Tensor))
            fake_A = G_BA(real_B)
            # Arange images along x-axis
            #real_A = make_grid(real_A, nrow=5, normalize=True)
            #real_B = make_grid(real_B, nrow=5, normalize=True)
            #fake_A = make_grid(fake_A, nrow=5, normalize=True)
            fake_B = make_grid(fake_B, nrow=1, normalize=True)
            # Arange images along y-axis
            #image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
            save_image(fake_A, "./images_test/"+str(epoch)+"/real/"+str(i).zfill(5)+".png")
            #output = fake_B.detach().cpu()[0]
            #output = transform_to_image(output)
            #output.save("./images_test/gta/f_gta"+str(i).zfill(5)+".png")


#real_A = Variable(batch["A"].type(Tensor))
#        real_B = Variable(batch["B"].type(Tensor))

        # Adversarial ground truths
#        valid =  Variable(Tensor(np.ones((real_A.size(0), *D_A1.output_shape))), requires_grad=False)
#        fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A1.output_shape))), requires_grad=False)


for i in range(0,1):
    sample_images(i)
