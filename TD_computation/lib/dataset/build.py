

import os
import random
import numpy as np
from PIL import Image
import glob
import torch
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms as transforms

from torchtoolbox.transform import Cutout


# Image statistics
RGB_statistics = {
    'iNaturalist18': {
        'mean': [0.466, 0.471, 0.380],
        'std': [0.195, 0.194, 0.192]
    },
    'default': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
}


def get_data_transform(split, rgb_mean, rbg_std, key='default'):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]) if key == 'iNaturalist18' else transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ])
    }
    return data_transforms[split]


class LT_Dataset(torch.utils.data.Dataset):
    
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.fakes = []
        self.transform = transform
        print("--------------------------------------------")
        print(root)
        with open(txt) as f:
            lines = f.readlines()
            for line in lines:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
                self.fakes.append(int(line.split()[2]))
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]
        fake = self.fakes[index]
        merge = [fake,label,1,0]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        return index,sample, merge



def load_data(data_root,txt, phase, batch_size=128, shuffle=True,num_workers=16):

    print('Loading data from %s' % txt)
    key = 'default'

    rgb_mean, rgb_std = RGB_statistics[key]['mean'], RGB_statistics[key]['std']

    if phase not in ['train', 'val']:
        transform = get_data_transform('test', rgb_mean, rgb_std, key)
    else:
        transform = get_data_transform(phase, rgb_mean, rgb_std, key)

    print('Use data transformation:', transform)

    set_ = LT_Dataset(data_root, txt, transform)
    data_loader = DataLoader(dataset=set_,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=True)

    return data_loader



# class Cutout1(object):
#     """Randomly mask out one or more patches from an image.
#     Args:
#         n_holes (int): Number of patches to cut out of each image.
#         length (int): The length (in pixels) of each square patch.
#     """
#     def __init__(self, n_holes=16, length=1):
#         self.n_holes = n_holes
#         self.length = length

#     def __call__(self, img):
#         """
#         Args:
#             img (Tensor): Tensor image of size (C, H, W).
#         Returns:
#             Tensor: Image with n_holes of dimension length x length cut out of it.
#         """
#         h = img.size(1)
#         w = img.size(2)

#         mask = np.ones((h, w), np.float32)

#         for n in range(self.n_holes):
#             y = np.random.randint(h)
#             x = np.random.randint(w)

#             y1 = np.clip(y - self.length // 2, 0, h)
#             y2 = np.clip(y + self.length // 2, 0, h)
#             x1 = np.clip(x - self.length // 2, 0, w)
#             x2 = np.clip(x + self.length // 2, 0, w)

#             mask[y1:y2, x1:x2] = 0.

#         mask = torch.from_numpy(mask)
#         mask = mask.expand_as(img)
#         img = img * mask

#         return img


# 图像预处理步骤
transform32 = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5]),
            ])
transform32_val = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5]),
            ])
transform32_cutout = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                Cutout(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5]),
            ])

transform224 = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])

transform224_val = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

transform224_cutout = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                Cutout(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])

class REMnist(torchvision.datasets.MNIST):
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

class REFashionMnist(torchvision.datasets.FashionMNIST):
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
    
class REkMnist(torchvision.datasets.KMNIST):
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
    
class REMNIST(torchvision.datasets.MNIST):
    def __init__(self,root: str,task: str,transform=None, train: bool =True):
        super().__init__(root=root,train=train,transform=transform,download=True)
        self.task = task
        self.targets = torch.from_numpy(np.load(f"behaviour_dataset/{self.task}/y_merge.npz")["arr_0"])
        self.replace_dcit = self.get_replace_dict()
    

    def get_replace_dict(self):
        replace_path = glob.glob(f"behaviour_dataset/{self.task}/*.pt")
        tuples = [os.path.split(x)[1].split(".")[0].split("_")[::-1] for x in replace_path]
        replace = {}
        for (key,value) in tuples:
            img_path = f"behaviour_dataset/{self.task}/{value}_{key}.pt"
            img = torch.load(img_path).detach().to(self.targets.device)
            replace[key] = img
        return replace

    def __getitem__(self, index: int):
        target = self.targets[index]
        img = Image.fromarray(self.data[index].numpy(), mode="L").convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if str(index) in self.replace_dcit.keys():
            img = self.replace_dcit[str(index)]
        return index,img, target


class REKMNIST(torchvision.datasets.KMNIST):
    def __init__(self,root: str,task: str, transform=None, train: bool =True):
        super().__init__(root=root,train=train,transform=transform,download=True)
        self.task = task
        self.targets = torch.from_numpy(np.load(f"behaviour_dataset/{self.task}/y_merge.npz")["arr_0"])
        self.replace_dcit = self.get_replace_dict()
    

    def get_replace_dict(self):
        replace_path = glob.glob(f"behaviour_dataset/{self.task}/*.pt")
        tuples = [os.path.split(x)[1].split(".")[0].split("_")[::-1] for x in replace_path]
        replace = {}
        for (key,value) in tuples:
            img_path = f"behaviour_dataset/{self.task}/{value}_{key}.pt"
            img = torch.load(img_path).detach().to(self.targets.device)
            replace[key] = img
        return replace

    def __getitem__(self, index: int):
        target = self.targets[index]
        img = Image.fromarray(self.data[index].numpy(), mode="L").convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if str(index) in self.replace_dcit.keys():
            img = self.replace_dcit[str(index)]
        return index,img, target
    
    
class REFashionMNIST(torchvision.datasets.FashionMNIST):
    def __init__(self,root: str,task: str, transform=None, train: bool =True):
        super().__init__(root=root,train=train,transform=transform,download=True)
        self.task = task
        self.targets = torch.from_numpy(np.load(f"behaviour_dataset/{self.task}/y_merge.npz")["arr_0"])
        self.replace_dcit = self.get_replace_dict()
    

    def get_replace_dict(self):
        replace_path = glob.glob(f"behaviour_dataset/{self.task}/*.pt")
        tuples = [os.path.split(x)[1].split(".")[0].split("_")[::-1] for x in replace_path]
        replace = {}
        for (key,value) in tuples:
            img_path = f"behaviour_dataset/{self.task}/{value}_{key}.pt"
            img = torch.load(img_path).detach().to(self.targets.device)
            replace[key] = img
        return replace

    def __getitem__(self, index: int):
        target = self.targets[index]
        img = Image.fromarray(self.data[index].numpy(), mode="L").convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if str(index) in self.replace_dcit.keys():
            img = self.replace_dcit[str(index)]
        return index,img, target
    
class RECIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self,root: str,task: str,transform=None, train: bool =True):
        super().__init__(root=root,train=train,transform=transform,download=True)
        self.task = task
        self.targets = torch.from_numpy(np.load(f"behaviour_dataset/{self.task}/y_merge.npz")["arr_0"])
        self.replace_dcit = self.get_replace_dict()
    

    def get_replace_dict(self):
        replace_path = glob.glob(f"behaviour_dataset/{self.task}/*.pt")
        tuples = [os.path.split(x)[1].split(".")[0].split("_")[::-1] for x in replace_path]
        replace = {}
        for (key,value) in tuples:
            img_path = f"behaviour_dataset/{self.task}/{value}_{key}.pt"
            img = torch.load(img_path).detach().to(self.targets.device)
            replace[key] = img
        return replace

    def __getitem__(self, index: int):
        target = self.targets[index]
        img = Image.fromarray(self.data[index])
        if self.transform is not None:
            img = self.transform(img)
        if str(index) in self.replace_dcit.keys():
            img = self.replace_dcit[str(index)]
        return index, img, target
    
class RECIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self,root: str,task: str,transform=None, train: bool =True):
        super().__init__(root=root,train=train,transform=transform,download=True)
        self.task = task
        self.targets = torch.from_numpy(np.load(f"behaviour_dataset/{self.task}/y_merge.npz")["arr_0"])
        self.replace_dcit = self.get_replace_dict()

    def get_replace_dict(self):
        replace_path = glob.glob(f"behaviour_dataset/{self.task}/*.pt")
        tuples = [os.path.split(x)[1].split(".")[0].split("_")[::-1] for x in replace_path]
        replace = {}
        for (key,value) in tuples:
            img_path = f"behaviour_dataset/{self.task}/{value}_{key}.pt"
            img = torch.load(img_path).detach().to(self.targets.device)
            replace[key] = img
        return replace

    def __getitem__(self, index: int):
        target = self.targets[index]
        img = Image.fromarray(self.data[index])
        if self.transform is not None:
            img = self.transform(img)
        if str(index) in self.replace_dcit.keys():
            img = self.replace_dcit[str(index)]
        return index, img, target
    
class RESVHN(torchvision.datasets.SVHN):
    def __init__(self,root: str,task: str,transform=None, train: bool =True):
        super().__init__(root=root,transform=transform,download=True)
        self.task = task
        self.targets = torch.from_numpy(np.load(f"behaviour_dataset/{self.task}/y_merge.npz")["arr_0"])
        self.replace_dcit = self.get_replace_dict()
    

    def get_replace_dict(self):
        replace_path = glob.glob(f"behaviour_dataset/{self.task}/*.pt")
        tuples = [os.path.split(x)[1].split(".")[0].split("_")[::-1] for x in replace_path]
        replace = {}
        for (key,value) in tuples:
            img_path = f"behaviour_dataset/{self.task}/{value}_{key}.pt"
            img = torch.load(img_path).detach().to(self.targets.device)
            replace[key] = img
        return replace

    def __getitem__(self, index: int):
        target = self.targets[index]
        img = Image.fromarray(np.transpose(self.data[index], (1, 2, 0)))
        if self.transform is not None:
            img = self.transform(img)
        if str(index) in self.replace_dcit.keys():
            img = self.replace_dcit[str(index)]
        return index,img, target
    
    
class REFlowers102(torchvision.datasets.Flowers102):
    def __init__(self,root: str,task: str,transform=None, train: bool =True):
        super().__init__(root=root,transform=transform,download=True)
        self.task = task
        self.targets = torch.from_numpy(np.load(f"behaviour_dataset/{self.task}/y_merge.npz")["arr_0"])
        self.replace_dcit = self.get_replace_dict()
    

    def get_replace_dict(self):
        replace_path = glob.glob(f"behaviour_dataset/{self.task}/*.pt")
        tuples = [os.path.split(x)[1].split(".")[0].split("_")[::-1] for x in replace_path]
        replace = {}
        for (key,value) in tuples:
            img_path = f"behaviour_dataset/{self.task}/{value}_{key}.pt"
            img = torch.load(img_path).detach().to(self.targets.device)
            replace[key] = img
        return replace

    def __getitem__(self, index: int):
        target = self.targets[index]
        img = Image.open(self._image_files[index]).convert("RGB")
        
        if self.transform is not None:
            img = self.transform(img)
        if str(index) in self.replace_dcit.keys():
            img = self.replace_dcit[str(index)]
        return index,img, target
    

class REOxfordIIITPet(torchvision.datasets.OxfordIIITPet):
    def __init__(self,root: str,task: str,transform=None, train: bool =True):
        super().__init__(root=root,transform=transform,download=True)
        self.task = task
        self.targets = torch.from_numpy(np.load(f"behaviour_dataset/{self.task}/y_merge.npz")["arr_0"])
        self.replace_dcit = self.get_replace_dict()
    

    def get_replace_dict(self):
        replace_path = glob.glob(f"behaviour_dataset/{self.task}/*.pt")
        tuples = [os.path.split(x)[1].split(".")[0].split("_")[::-1] for x in replace_path]
        replace = {}
        for (key,value) in tuples:
            img_path = f"behaviour_dataset/{self.task}/{value}_{key}.pt"
            img = torch.load(img_path).detach().to(self.targets.device)
            replace[key] = img
        return replace

    def __getitem__(self, index: int):
        target = self.targets[index]
        img = Image.open(self._images[index]).convert("RGB")
        
        if self.transform is not None:
            img = self.transform(img)
        if str(index) in self.replace_dcit.keys():
            img = self.replace_dcit[str(index)]
        return index,img, target
    

class REStanfordCars(torchvision.datasets.StanfordCars):
    def __init__(self,root: str,task: str,transform=None, train: bool =True):
        super().__init__(root=root,transform=transform,download=True)
        self.task = task
        self.targets = torch.from_numpy(np.load(f"behaviour_dataset/{self.task}/y_merge.npz")["arr_0"])
        self.replace_dcit = self.get_replace_dict()
    

    def get_replace_dict(self):
        replace_path = glob.glob(f"behaviour_dataset/{self.task}/*.pt")
        tuples = [os.path.split(x)[1].split(".")[0].split("_")[::-1] for x in replace_path]
        replace = {}
        for (key,value) in tuples:
            img_path = f"behaviour_dataset/{self.task}/{value}_{key}.pt"
            img = torch.load(img_path).detach().to(self.targets.device)
            replace[key] = img
        return replace

    def __getitem__(self, index: int):
        target = self.targets[index]
        img = Image.open(self._samples[index][0]).convert("RGB")
        
        if self.transform is not None:
            img = self.transform(img)
        if str(index) in self.replace_dcit.keys():
            img = self.replace_dcit[str(index)]
        return index,img, target
   
class REImageFolder(ImageFolder):
    def __init__(self,root: str,transform=None):
        super().__init__(root=root,transform=transform)
        self.task = self.root.split("/")[-2]
        self.targets = torch.from_numpy(np.load(f"behaviour_dataset/{self.task}/y_merge.npz")["arr_0"])

    def __getitem__(self, index: int):

        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        target = self.targets[index]

        return index,sample, target 
    
    
def build_dataloader(args):
    if "MNIST" in args.dataset and  "FashionMNIST" not in args.dataset and  "KMNIST" not in args.dataset:
        train_dataset = REMNIST(root='/mnt/st_data/dataset',task = args.dataset, transform=transform32) 
        val_dataset = REMnist(root='/mnt/st_data/dataset',train=False, transform=transform32_val)
        num_classes = 10
    elif "KMNIST" in args.dataset:
        train_dataset = REKMNIST(root='/mnt/st_data/dataset',task = args.dataset, transform=transform32)
        val_dataset = REkMnist(root='/mnt/st_data/dataset',train=False, transform=transform32_val)
        num_classes = 10
    elif "FashionMNIST" in args.dataset:
        train_dataset = REFashionMNIST(root='/mnt/st_data/dataset',task = args.dataset, transform=transform32)
        val_dataset = REFashionMnist(root='/mnt/st_data/dataset',train=False, transform=transform32_val)
        num_classes = 10
    elif "CIFAR10" in args.dataset and  "CIFAR100" not in args.dataset:
        train_dataset = RECIFAR10(root='/mnt/st_data/dataset',task = args.dataset, transform=transform32)
        val_dataset = torchvision.datasets.CIFAR10(root='/mnt/st_data/dataset',train=False, transform=transform32_val)
        num_classes = 10
    elif "CIFAR100" in args.dataset:   
        train_dataset = RECIFAR100(root='/mnt/st_data/dataset',task = args.dataset, transform=transform32) 
        val_dataset = torchvision.datasets.CIFAR100(root='/mnt/st_data/dataset',train=False, transform=transform32_val)
        num_classes = 100
    elif "SVHN" in args.dataset:    
        train_dataset = RESVHN(root='/mnt/st_data/dataset',task = args.dataset, transform=transform32) 
        val_dataset = torchvision.datasets.SVHN(root='/mnt/st_data/dataset',split="test", transform=transform32_val,download=True)
        num_classes = 10
    elif "Flowers102" in args.dataset:
        train_dataset = REFlowers102(root='/mnt/st_data/dataset',task = args.dataset, transform=transform224) 
        val_dataset = torchvision.datasets.Flowers102(root='/mnt/st_data/dataset',split="test", transform=transform224_val,download=True)
        num_classes = 102
    elif "OxfordIIITPet" in args.dataset:    
        train_dataset = REOxfordIIITPet(root='/mnt/st_data/dataset',task = args.dataset, transform=transform224) 
        val_dataset = torchvision.datasets.OxfordIIITPet(root='/mnt/st_data/dataset',split="test", transform=transform224_val,download=True)
        num_classes = 37
    elif "StanfordCars" in args.dataset:    
        train_dataset = REStanfordCars(root='/mnt/st_data/dataset',task = args.dataset, transform=transform224) 
        val_dataset = torchvision.datasets.StanfordCars(root='/mnt/st_data/dataset',split="test", transform=transform224_val,download=True)
        num_classes = 196
    elif args.dataset=="imagenet_lt":
        train_dataset = REImageFolder('~/dataset/imagenet_lt/train',transform=transform224)
        val_dataset = ImageFolder('~/dataset/imagenet_lt/val',transform=transform224_val)
        num_classes = 1000
    elif args.dataset=="places-lt":
        train_dataset = REImageFolder('~/dataset/places-lt/train',transform=transform224)
        val_dataset = ImageFolder('~/dataset/places-lt/val',transform=transform224_val)
        num_classes = 365

    elif args.dataset=="clothing_flip_0.2":
        # train_dataset = REImageFolder('~/dataset/places-lt/train',transform=transform224)
        # val_dataset = ImageFolder('~/dataset/places-lt/val',transform=transform224_val)
        num_classes = 14
        train_loader = load_data(data_root="/home/kunyu/clothing1m",txt="labels/train_flip_0.2.txt",shuffle=True, phase="train")
        val_loader = load_data(data_root="/home/kunyu/clothing1m",txt="labels/test.txt", shuffle=False, phase="test")
        _ = 0
        return train_loader,val_loader,_,_,num_classes

    elif args.dataset=="clothing_flip_0.4":
        # train_dataset = REImageFolder('~/dataset/places-lt/train',transform=transform224)
        # val_dataset = ImageFolder('~/dataset/places-lt/val',transform=transform224_val)
        num_classes = 14
        train_loader = load_data(data_root="/home/kunyu/clothing1m",txt="labels/train_flip_0.4.txt",shuffle=True, phase="train")
        val_loader = load_data(data_root="/home/kunyu/clothing1m",txt="labels/test.txt", shuffle=False, phase="test")
        _ = 0
        return train_loader,val_loader,_,_,num_classes

    elif args.dataset=="clothing_sym_0.2":
        # train_dataset = REImageFolder('~/dataset/places-lt/train',transform=transform224)
        # val_dataset = ImageFolder('~/dataset/places-lt/val',transform=transform224_val)
        num_classes = 14
        train_loader = load_data(data_root="/home/kunyu/clothing1m",txt="labels/train_sym_0.2.txt",shuffle=True, phase="train")
        val_loader = load_data(data_root="/home/kunyu/clothing1m",txt="labels/test.txt", shuffle=False, phase="test")
        _ = 0
        return train_loader,val_loader,_,_,num_classes
    
    elif args.dataset=="clothing_sym_0.4":
        # train_dataset = REImageFolder('~/dataset/places-lt/train',transform=transform224)
        # val_dataset = ImageFolder('~/dataset/places-lt/val',transform=transform224_val)
        num_classes = 14
        train_loader = load_data(data_root="/home/kunyu/clothing1m",txt="labels/train_sym_0.4.txt",shuffle=True, phase="train")
        val_loader = load_data(data_root="/home/kunyu/clothing1m",txt="labels/test.txt", shuffle=False, phase="test")
        _ = 0
        return train_loader,val_loader,_,_,num_classes
     
    elif args.dataset=="clothing_noise":
        # train_dataset = REImageFolder('~/dataset/places-lt/train',transform=transform224)
        # val_dataset = ImageFolder('~/dataset/places-lt/val',transform=transform224_val)
        num_classes = 14
        train_loader = load_data(data_root="/home/kunyu/clothing1m",txt="labels/train_noise.txt",shuffle=True, phase="train")
        val_loader = load_data(data_root="/home/kunyu/clothing1m",txt="labels/test.txt", shuffle=False, phase="test")
        _ = 0
        return train_loader,val_loader,_,_,num_classes      
        
    print(args.dataset,torch.sum(train_dataset.targets[:,0] == train_dataset.targets[:,1])/len(train_dataset.targets[:,1]))
    train_loader = DataLoader(dataset=train_dataset,batch_size=args.bs,shuffle=True,num_workers=16,pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset,batch_size=128,shuffle=True,num_workers=16,pin_memory=True)
    _ = DataLoader(dataset=train_dataset,batch_size=128,shuffle=False,num_workers=16,pin_memory=True)
    print(f"train_loader number: {len(train_dataset)} val_loader number: {len(val_dataset)}")
    return train_loader,val_loader,_,_,num_classes


if __name__ == "__main__":

    # from pprint import pprint
    # pprint(os.listdir("behaviour_dataset"))
 
    tasks = ['clothing_noise',
    'clothing_flip_0.2',
    'clothing_flip_0.4',
    'clothing_sym_0.2',
    'clothing_sym_0.4',
]
    
       
    # tasks = ['FashionMNIST_clean',
    # 'MNIST_flip_0.1',
    # 'Flowers102_flip_0.1',
    # 'OxfordIIITPet_flip_0.2',
    # 'Flowers102_sym_0.2',
    # 'imagenet_lt',
    # 'KMNIST_sym_0.1',
    # 'KMNIST_sym_0.2',
    # 'KMNIST_flip_0.1',
    # 'KMNIST_flip_0.2',
    # # 'FashionMNIST_at_0.2',
    # 'CIFAR100_flip_0.2',
    # # 'StanfordCars_at_0.1',
    # # 'MNIST_at_0.1',
    # # 'Flowers102_at_0.1',
    # 'places-lt',
    # # 'SVHN_at_0.15',
    # 'FashionMNIST_sym_0.4',
    # 'StanfordCars_sym_0.3',
    # 'StanfordCars_clean',
    # 'CIFAR100_clean',
    # 'OxfordIIITPet_sym_0.1',
    # 'SVHN_flip_0.2',
    # 'CIFAR100_sym_0.2']
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='MNIST_clean', help='select model')
    parser.add_argument('--bs', default=128, help='select model')
    args = parser.parse_args()

    for task in tasks:
        print(f"++++++++++++++++++++++++++++++++++++++++{task}")
        
        args.dataset = task
        
        build_dataloader(args)