
import os
import random
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms as transforms
from PIL import Image
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
import torch
from gen_adv_x32 import CNN
import shutil
from torchvision.datasets import ImageFolder

class MixUp(object):
    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def __call__(self, batch):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        x, y = batch
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


transform32 = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5071, 0.4865, 0.4409],[0.1942, 0.1918, 0.1958]),
            ])

transform224 = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])

dataset = ImageFolder('~/dataset/imagenet_lt/train',transform=transform224)
# dataset = ImageFolder('~/dataset/places-lt/train',transform=transform224)

task = "imagenet_lt"

if not os.path.exists(f'behaviour_dataset/{task}'):
    os.makedirs(f'behaviour_dataset/{task}')
else:
    shutil.rmtree(f'behaviour_dataset/{task}')
    os.makedirs(f'behaviour_dataset/{task}')
    

target = [x[1] for x in dataset.samples]
# 顺序得固定下来

merge_y = []
for i,y in enumerate(target):
    label = np.array([y,y,1,0])
    merge_y.append(label)
        
numpy_merge_y = np.stack(merge_y)     
np.savez_compressed(f"behaviour_dataset/{task}/y_merge.npz",numpy_merge_y)



            
