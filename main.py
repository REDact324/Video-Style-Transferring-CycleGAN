import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image
import torch
import itertools
import tensorboardX

from src.model import Discriminatror, Generator
from src.util.buffer import ReplayBuffer
from src.util.util import LambdaLR, weights_init_normal
from src.dataset import ImageDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net_GAtoB = Generator().to(device)
net_GBtoA = Generator().to(device)

size = 256

net_GAtoB.load_state_dict(torch.load('/content/models/net_GAtoB.pth'))
net_GBtoA.load_state_dict(torch.load('/content/models/net_GBtoA.pth'))

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

image_path = '/content/IMG_5342.JPG'
image = Image.open(image_path).convert('RGB')

image = transforms.Compose(transforms_)(image)

if not os.path.exists('/content/results/A'):
    os.makedirs('/content/results/A')

real = torch.tensor(input_A.copy_(image), dtype=torch.float).to(device)
fake_A = 0.5 * (net_GBtoA(real).data + 1.0)

save_image(fake_A, '/content/results/A/IMG_5342.png')
