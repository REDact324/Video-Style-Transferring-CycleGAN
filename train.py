import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch
import itertools
import tensorboardX

from src.model import Discriminator, Generator
from src.util.buffer import ReplayBuffer
from src.util.util import LambdaLR, weights_init_normal
from src.dataset import ImageDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batchsize = 1
size = 256
lr = 2e-4
n_epochs = 200
epoch = 0
decay_epoch = 100

# Networks
net_GAtoB = Generator().to(device)
net_GBtoA = Generator().to(device)
net_DA = Discriminator().to(device)
net_DB = Discriminator().to(device)

net_GAtoB.apply(weights_init_normal)
net_GBtoA.apply(weights_init_normal)
net_DA.apply(weights_init_normal)
net_DB.apply(weights_init_normal)

# Loss
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()
criterion_tamporal = torch.nn.L1Loss()

# Optimizer & LR
opt_G = torch.optim.Adam(itertools.chain(net_GAtoB.parameters(), net_GBtoA.parameters()), lr=lr, betas=(0.5, 0.999))
opt_DA = torch.optim.Adam(net_DA.parameters(), lr=lr, betas=(0.5, 0.999))
opt_DB = torch.optim.Adam(net_DB.parameters(), lr=lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=LambdaLR(n_epochs,epoch, decay_epoch).step)
lr_scheduler_DA = torch.optim.lr_scheduler.LambdaLR(opt_DA, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)
lr_scheduler_DB = torch.optim.lr_scheduler.LambdaLR(opt_DB, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)

data_root = './data/train'
input_A = torch.ones(1, 3, 256, 256, dtype=torch.float).to(device)
input_B = torch.ones(1, 3, 256, 256, dtype=torch.float).to(device)

label_real = torch.ones([1], requires_grad=False, dtype=torch.float).to(device)
label_fake = torch.zeros([1], requires_grad=False, dtype=torch.float).to(device)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

log_path = './log/'
writer = tensorboardX.SummaryWriter(log_path)

transforms_ = [
    transforms.Resize(int(size * 1.12), Image.BICUBIC),
    transforms.RandomCrop(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
              ]

dataloader = DataLoader(ImageDataset(data_root, transforms_), batch_size=batchsize, shuffle=True, num_workers=8)
step = 0

for epoch in range(epoch, n_epochs):
  for i, batch in enumerate(dataloader):
    real_A1 = torch.tensor(input_A.copy_(batch['A'][0]), dtype=torch.float).to(device)
    real_A2 = torch.tensor(input_A.copy_(batch['A'][1]), dtype=torch.float).to(device)
    real_B = torch.tensor(input_B.copy_(batch['B']), dtype=torch.float).to(device)

    opt_G.zero_grad()

    same_B = net_GAtoB(real_B)
    loss_identity_B = criterion_identity(same_B, real_B) * 5.0

    same_A1 = net_GBtoA(real_A1)
    same_A2 = net_GBtoA(real_A2)
    loss_identity_A1 = criterion_identity(same_A1, real_A1)
    loss_identity_A2 = criterion_identity(same_A2, real_A2)
    loss_identity_A = (loss_identity_A1 + loss_identity_A2) * 2.5

    fake_B1 = net_GAtoB(real_A1)
    fake_B2 = net_GAtoB(real_A2)
    pred_fake1 = net_DA(fake_B1)
    pred_fake2 = net_DA(fake_B2)
    loss_GAN_A2B1 = criterion_GAN(pred_fake1, label_real)
    loss_GAN_A2B2 = criterion_GAN(pred_fake2, label_real)
    loss_GAN_A2B = (loss_GAN_A2B1 + loss_GAN_A2B2) * 0.5

    fake_A = net_GBtoA(real_B)
    pred_fake = net_DB(fake_A)
    loss_GAN_B2A = criterion_GAN(pred_fake, label_real)

    # cycle loss
    recovered_A1 = net_GBtoA(fake_B1)
    recovered_A2 = net_GBtoA(fake_B2)
    loss_cycle_ABA1 = criterion_cycle(recovered_A1, real_A1)
    loss_cycle_ABA2 = criterion_cycle(recovered_A2, real_A2)
    loss_cycle_ABA = (loss_cycle_ABA1 + loss_cycle_ABA2) * 5.0

    recovered_B = net_GAtoB(fake_A)
    loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0

    # Temporal loss
    diff_real = real_A1 - real_A2
    diff_fake = fake_B1 - fake_B2
    loss_temporal = criterion_tamporal(diff_real, diff_fake) * 1e-3

    # total loss
    loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + loss_temporal
    loss_G.backward()
    opt_G.step()

    #####################

    # DA
    opt_DA.zero_grad()

    pred_real1 = net_DA(real_A1)
    pred_real2 = net_DA(real_A2)
    loss_D_real1 = criterion_GAN(pred_real1, label_real)
    loss_D_real2 = criterion_GAN(pred_real2, label_real)
    loss_D_real = (loss_D_real1 + loss_D_real2) * 0.5

    fake_A = fake_A_buffer.push_and_pop(fake_A)
    pred_fake = net_DA(fake_A.detach())
    loss_D_fake = criterion_GAN(pred_fake, label_fake)

    loss_DA = (loss_D_real + loss_D_fake) * 0.5
    loss_DA.backward()
    opt_DA.step()

    # DB
    opt_DB.zero_grad()

    pred_real = net_DB(real_B)
    loss_D_real = criterion_GAN(pred_real, label_real)

    fake_B1 = fake_B_buffer.push_and_pop(fake_B1)
    fake_B2 = fake_B_buffer.push_and_pop(fake_B2)

    pred_fake1 = net_DB(fake_B1.detach())
    pred_fake2 = net_DB(fake_B2.detach())
    loss_D_fake1 = criterion_GAN(pred_fake1, label_fake)
    loss_D_fake2 = criterion_GAN(pred_fake2, label_fake)
    loss_D_fake = (loss_D_fake1 + loss_D_fake2) * 0.5

    loss_DB = (loss_D_real + loss_D_fake) * 0.5
    loss_DB.backward()
    opt_DB.step()


    print("epoch:{}, batch:{}, loss_G:{}, loss_DA:{}, loss_DB:{}".format(
        epoch, i, loss_G, loss_DA, loss_DB))

    writer.add_scalar('loss_G', loss_G, global_step=step+1)
    writer.add_scalar('loss_identity', loss_identity_A+loss_identity_B, global_step=step+1)
    writer.add_scalar('loss_GAN', loss_GAN_A2B+loss_GAN_B2A, global_step=step+1)
    writer.add_scalar('loss_cycle', loss_cycle_ABA+loss_cycle_BAB, global_step=step+1)
    writer.add_scalar('loss_temporal', loss_temporal, global_step=step+1)
    writer.add_scalar('loss_DA', loss_DA, global_step=step+1)
    writer.add_scalar('loss_DB', loss_DB, global_step=step+1)

    step += 1

  lr_scheduler_G.step()
  lr_scheduler_DA.step()
  lr_scheduler_DB.step()

  torch.save(net_GAtoB.state_dict(), './models/net_GAtoB.pth')
  torch.save(net_GBtoA.state_dict(), './models/net_GBtoA.pth')
  torch.save(net_DA.state_dict(), './models/net_DA.pth')
  torch.save(net_DB.state_dict(), './models/net_DB.pth')
