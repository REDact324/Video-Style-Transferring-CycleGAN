import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image
import torch

from src.model import Generator
from src.dataset import ImageDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batchsize = 1

net_GAtoB = Generator().to(device)
net_GBtoA = Generator().to(device)

size = 256

net_GAtoB.load_state_dict(torch.load('./models/net_GAtoB.pth'))
net_GBtoA.load_state_dict(torch.load('./models/net_GBtoA.pth'))

net_GAtoB.eval()
net_GBtoA.eval()

input_A = torch.ones(1, 3, 256, 256, dtype=torch.float).to(device)
input_B = torch.ones(1, 3, 256, 256, dtype=torch.float).to(device)

transforms_ = [
    transforms.Resize(int(size * 1.12), Image.BICUBIC),
    transforms.RandomCrop(size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
               ]

data_root = './data/test/'
dataloader = dataloader = DataLoader(ImageDataset(data_root, transforms_, 'test'), batch_size=batchsize, shuffle=False, num_workers=8)

if not os.path.exists('./results/A'):
  os.makedirs('./results/A')
if not os.path.exists('./results/B'):
  os.makedirs('./results/B')

for i, batch in enumerate(dataloader):
  real_A = torch.tensor(input_A.copy_(batch['A']), dtype=torch.float).to(device)
  real_B = torch.tensor(input_B.copy_(batch['B']), dtype=torch.float).to(device)

  fake_B = 0.5 * (net_GAtoB(real_A).data + 1.0)
  fake_A = 0.5 * (net_GBtoA(real_B).data + 1.0)

  save_image(fake_A, './results/A/%04d.png' % (i + 1))
  save_image(fake_B, './results/B/%04d.png' % (i + 1))
