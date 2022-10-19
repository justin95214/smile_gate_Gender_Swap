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

from datasets import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.autograd as autograd
from models import *


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=149)
parser.add_argument("--n_epochs", type=int, default=500)
parser.add_argument("--dataset_name", type=str, default="gender_dataset_256")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=0.0002)
parser.add_argument("--b1", type=float, default=0.5)
parser.add_argument("--b2", type=float, default=0.999)
parser.add_argument("--decay_epoch", type=int, default=50)
parser.add_argument("--n_cpu", type=int, default=16)
parser.add_argument("--img_height", type=int, default=128)
parser.add_argument("--img_width", type=int, default=128)
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
os.makedirs("./%s" % "CycleGAN_model", exist_ok=True)
os.makedirs("./%s" % "images/", exist_ok=True)

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

if opt.epoch != 0:
    # Load pretrained models
    G_AB.load_state_dict(torch.load("./CycleGAN_model/G_man_woman_%d.pth" % (opt.epoch)))
    G_BA.load_state_dict(torch.load("./CycleGAN_model/G_woman_man_%d.pth" % (opt.epoch)))
    D_A.load_state_dict(torch.load("./CycleGAN_model/D_man_%d.pth" % (opt.epoch)))
    D_B.load_state_dict(torch.load("./CycleGAN_model/D_woman_%d.pth" % (opt.epoch)))
else:
    # Initialize weights
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)

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

# Training data loader
dataloader = DataLoader(

    Dataset("./%s" % opt.dataset_name, transforms_=transforms_, unaligned=True),
    batch_size=opt.batch_size,
    shuffle=True
)

print(len(dataloader))

# Test data loader
val_dataloader = DataLoader(
    Dataset("./%s" % opt.dataset_name, transforms_=transforms_, unaligned=True, mode="test"),
    batch_size=5,
    shuffle=False
)


def calc_gradient_penalty(netD, real_data, generated_data):
    # GP strength
    LAMBDA = 10

    b_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(b_size, 1, 1, 1)
    alpha = alpha.expand_as(real_data)
    alpha = alpha.cuda()

    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = Variable(interpolated, requires_grad=True)
    interpolated = interpolated.cuda()

    # Calculate probability of interpolated examples
    prob_interpolated = netD(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch.Tensor.grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
                           create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(b_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return LAMBDA * ((gradients_norm - 1) ** 2).mean()



def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(16, 1, 8, 8)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty*10



def sample_images(epoch):
    #os.makedirs("./CycleGAN_model/%s/%s" % (opt.dataset_name,epoch), exist_ok=True)
    """Saves a generated sample from the test set"""
    with torch.no_grad():
        for i, imgs in enumerate(val_dataloader):
            os.makedirs("./images/%d/" % epoch, exist_ok=True)

            G_AB.eval()
            G_BA.eval()
            real_A = Variable(imgs["A"].type(Tensor))
            fake_B = G_AB(real_A)
            real_B = Variable(imgs["B"].type(Tensor))
            fake_A = G_BA(real_B)
            # Arange images along x-axis
            real_A = make_grid(real_A, nrow=5, normalize=True)
            real_B = make_grid(real_B, nrow=5, normalize=True)
            fake_A = make_grid(fake_A, nrow=5, normalize=True)
            fake_B = make_grid(fake_B, nrow=5, normalize=True)
            # Arange images along y-axis
            image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
            save_image(image_grid, "./images/%s/%s.png" % (epoch, i))

# ----------
#  Training
# ----------
print("train _start")

loss_hist = {'gen': [],
             'dis': [],
            'adv': [],
             'cycle': [],
            'identity': []
             }


prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):

        # Set model input
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))

        # Adversarial ground truths
        valid =  Variable(Tensor(np.ones((real_A.size(0), *D_A1.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A1.output_shape))), requires_grad=False)


        # ------------------
        #  Train Generators
        # ------------------

        G_AB.train()
        G_BA.train()

        optimizer_G.zero_grad()

        # Identity loss
        loss_id_A = criterion_identity(G_BA(real_A), real_A)
        loss_id_B = criterion_identity(G_AB(real_B), real_B)

        loss_identity = (loss_id_A + loss_id_B) / 2

        # GAN loss
        fake_B = G_AB(real_A)
        loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
        fake_A = G_BA(real_B)
        loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        # Cycle loss
        recov_A = G_BA(fake_B)
        loss_cycle_A = criterion_cycle(recov_A, real_A)
        recov_B = G_AB(fake_A)
        loss_cycle_B = criterion_cycle(recov_B, real_B)

        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # Total loss
        loss_G = loss_GAN + opt.lambda_cyc * loss_cycle + opt.lambda_id * loss_identity

        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator A
        # -----------------------

        optimizer_D_A.zero_grad()
        ############ criterion_GAN
        # Real loss
        loss_real = criterion_GAN(D_A(real_A), valid)
        # Fake loss (on batch of previously generated samples)
        fake_A_ = fake_A_buffer.push_and_pop(fake_A)
        loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
        # Total loss
        loss_D_A = (loss_real + loss_fake) / 2

        #fake_A_ = fake_A_buffer.push_and_pop(fake_A)
        #grad_penalty_A =  compute_gradient_penalty(D_A, fake_A_.data, real_A.data)
        #loss_D_A = torch.mean(D_A(fake_A_)) - torch.mean(D_A(real_A))


        loss_D_A.backward()
        optimizer_D_A.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------

        optimizer_D_B.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D_B(real_B), valid)
        # Fake loss (on batch of previously generated samples)
        fake_B_ = fake_B_buffer.push_and_pop(fake_B)
        loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2

        #fake_B_ = fake_B_buffer.push_and_pop(fake_B)
        #grad_penalty_B =  compute_gradient_penalty(D_B, fake_B_.data, real_B.data)
        #loss_D_B = torch.mean(D_B(fake_B_)) - torch.mean(D_B(real_B))


        loss_D_B.backward()
        optimizer_D_B.step()

        loss_D = (loss_D_A + loss_D_B) / 2

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        loss_hist['gen'].append(loss_G.item())
        loss_hist['dis'].append(loss_D.item())
        loss_hist['adv'].append(loss_GAN.item())
        loss_hist['cycle'].append(loss_cycle.item())
        loss_hist['identity'].append(loss_identity.item())

        print(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),
                loss_GAN.item(),
                loss_cycle.item(),
                loss_identity.item(),
                time_left,
            )
        )


    if batches_done % opt.sample_interval == 0:
        sample_images(batches_done)


    # loss history
    plt.figure(figsize=(10, 5))
    plt.title('Loss Progress')
    plt.plot(loss_hist['gen'], label='Gen. Loss')
    plt.plot(loss_hist['dis'], label='Dis. Loss')
    plt.plot(loss_hist['adv'], label='adv. Loss')
    plt.plot(loss_hist['cycle'], label='cycle. Loss')
    plt.plot(loss_hist['identity'], label='identity. Loss')

    plt.xlabel('batch count')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("./CycleGAN_model/%s_Cyc_%d_hist.png" % (opt.dataset_name, epoch))

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # Save model checkpoints
    torch.save(G_AB.state_dict(), "./CycleGAN_model/G_man_woman_%d.pth" % (epoch))
    torch.save(G_BA.state_dict(), "./CycleGAN_model/G_woman_man_%d.pth" % (epoch))
    torch.save(D_A.state_dict(), "./CycleGAN_model/D_man_%d.pth" % (epoch))
    torch.save(D_B.state_dict(), "./CycleGAN_model/D_woman_%d.pth" % (epoch))