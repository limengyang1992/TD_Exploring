
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

class REMNIST(torchvision.datasets.MNIST):
    def __init__(self,root: str,transform=None, train: bool =True):
        super().__init__(root=root,train=train,transform=transform)

    def __getitem__(self, index: int):

        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode="L").convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

class REFashionMNIST(torchvision.datasets.FashionMNIST):
    def __init__(self,root: str,transform=None, train: bool =True):
        super().__init__(root=root,train=train,transform=transform,download=True)

    def __getitem__(self, index: int):

        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode="L").convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
    
# 
# dataset = torchvision.datasets.SVHN(root='/mnt/st_data/dataset',  transform=transform32, download=True)
# dataset = REMNIST(root='/mnt/st_data/dataset',  transform=transform32) 
# dataset = REFashionMNIST(root='/mnt/st_data/dataset', transform=transform32)
# dataset = torchvision.datasets.CIFAR10(root='/mnt/st_data/dataset',  transform=transform32, download=True)
dataset = torchvision.datasets.CIFAR100(root='/mnt/st_data/dataset',  transform=transform32, download=True)

task = "CIFAR100"

if not os.path.exists(f'behaviour_dataset/{task}'):
    os.makedirs(f'behaviour_dataset/{task}')
else:
    shutil.rmtree(f'behaviour_dataset/{task}')
    os.makedirs(f'behaviour_dataset/{task}')
    
# net = torch.load(f"behaviour_dataset/net_AT/net_{task}.pt").eval()
images  = dataset.data

if task == "SVHN":
    target = dataset.labels
else:
    target = dataset.targets

# 顺序得固定下来
length = len(dataset)
max_index = length
num_chiose = int(0.1*length)
random_list = random.sample(range(max_index),num_chiose)
adver_list = []
cutout_list = []
mixup_list = []
noise_list = []

merge_y = []
for i,y in enumerate(target):
    if i in adver_list:
        label = np.array([y,y,1,1])
        merge_y.append(label)
        
    elif i in cutout_list:
        label = np.array([y,y,1,2])
        merge_y.append(label)
                
    elif i in mixup_list:
        y2 = target[i+1]
        label = np.array([y,y2,0.5,3])
        merge_y.append(label)
                
    elif i in noise_list:
        noise_label = random.sample([t for t in range(max(target)) if t!=y],1)[0]
        label = np.array([noise_label,y,1,4])
        merge_y.append(label)
                
    else:
        label = np.array([y,y,1,0])
        merge_y.append(label)
        
numpy_merge_y = np.stack(merge_y)     
np.savez_compressed(f"behaviour_dataset/{task}/y_merge.npz",numpy_merge_y)

    
for i in adver_list:
    net.eval()
    try:
        if task == "SVHN":
            x = transform32(Image.fromarray(np.transpose(images[i], (1, 2, 0))))
        else:
            x = transform32(Image.fromarray(images[i]))
    except:
        x = transform32(Image.fromarray(images[i].numpy(), mode="L").convert("RGB"))
    x_fgm = fast_gradient_method(net, x.cuda(), 0.2, np.inf)
    torch.save(x_fgm,f"behaviour_dataset/{task}/adver_{i}.pt")

# cut = Cutout(4, 8)
# for i in cutout_list:
#     try:
#         if task == "SVHN":
#             x = transform32(Image.fromarray(np.transpose(images[i], (1, 2, 0))))
#         else:
#             x = transform32(Image.fromarray(images[i]))
#     except:
#         x = transform32(Image.fromarray(images[i].numpy(), mode="L").convert("RGB"))
#     x_cutout = cut(x)
#     torch.save(x_cutout,f"behaviour_dataset/{task}/cutout_{i}.pt")

# for i,(x1,x2) in enumerate(zip(mixup_list,mixup_list_shift)):
#     try:
#         if task == "SVHN":
#             img1 = transform32(Image.fromarray(np.transpose(images[x1], (1, 2, 0))))
#             img2 = transform32(Image.fromarray(np.transpose(images[x2], (1, 2, 0))))
#         else:
#             img1 = transform32(Image.fromarray(images[x1]))
#             img2 = transform32(Image.fromarray(images[x2]))
#     except:
#         img1 = transform32(Image.fromarray(images[x1].numpy(), mode="L").convert("RGB"))
#         img2 = transform32(Image.fromarray(images[x2].numpy(), mode="L").convert("RGB"))
        

#     img = 0.5*img1 + 0.5*img2
#     torch.save(img,f"behaviour_dataset/{task}/mixup_{x1}.pt")

            
