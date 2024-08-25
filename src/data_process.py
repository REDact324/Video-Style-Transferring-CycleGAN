import glob
import random
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import os

# Build the dictionary of the images and labels in the dataset
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode='train'):
        self.transform = transforms.Compose(transforms_)

        self.paths_A = os.path.join(root, mode, "content/*")
        self.paths_B = os.path.join(root, mode, "style/*")

        self.list_A = glob.glob(self.paths_A)
        self.list_B = glob.glob(self.paths_B)

    def __getitem__(self, index1):
        image_path_A1 = self.list_A[index1 % len(self.list_A)]
        image_path_A2 = self.list_A[(index1+1) % len(self.list_A)]
        image_path_B = self.list_B[index1 % len(self.list_B)]

        image_A1 = Image.open(image_path_A1).convert('RGB')
        image_A2 = Image.open(image_path_A2).convert('RGB')
        image_B = Image.open(image_path_B).convert('RGB')

        item_A1 = self.transform(image_A1)
        item_A2 = self.transform(image_A2)
        item_B = self.transform(image_B)

        return {'A': [item_A1, item_A2], 'B': item_B}

    def __len__(self):
        return max(len(self.list_A), len(self.list_B))
