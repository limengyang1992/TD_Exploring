

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


# 图像预处理步骤
transform32 = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
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

def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

def default_loader(path: str) :

    return pil_loader(path)


class REMNIST(torchvision.datasets.MNIST):
    def __init__(self,root: str,transform=None, train: bool =True):
        super().__init__(root=root,train=train,transform=transform)
        self.task = self.__class__.__name__.replace("RE","")
        self.targets = torch.from_numpy(np.load(f"behaviour_dataset/{self.task}/y_merge.npz")["arr_0"])
        self.replace_dcit = self.get_replace_dict()

    def __len__(self) -> int:
        return int(len(self.data)*0.5)
    

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
        return img, target
    
class REFashionMNIST(torchvision.datasets.FashionMNIST):
    def __init__(self,root: str,transform=None, train: bool =True):
        super().__init__(root=root,train=train,transform=transform)
        self.task = self.__class__.__name__.replace("RE","")
        self.targets = torch.from_numpy(np.load(f"behaviour_dataset/{self.task}/y_merge.npz")["arr_0"])
        self.replace_dcit = self.get_replace_dict()

    def __len__(self) -> int:
        return int(len(self.data)*0.5)
    

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
        return img, target
    
class RECIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self,root: str,transform=None, train: bool =True):
        super().__init__(root=root,train=train,transform=transform)
        self.task = self.__class__.__name__.replace("RE","")
        self.targets = torch.from_numpy(np.load(f"behaviour_dataset/{self.task}/y_merge.npz")["arr_0"])
        self.replace_dcit = self.get_replace_dict()

    def __len__(self) -> int:
        return int(len(self.data)*0.5)
    

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
        return img, target
    
class RECIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self,root: str,transform=None, train: bool =True):
        super().__init__(root=root,train=train,transform=transform)
        self.task = self.__class__.__name__.replace("RE","")
        self.targets = torch.from_numpy(np.load(f"behaviour_dataset/{self.task}/y_merge.npz")["arr_0"])
        self.replace_dcit = self.get_replace_dict()

    def __len__(self) -> int:
        return int(len(self.data)*0.5)
    

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
        return img, target
    
class RESVHN(torchvision.datasets.SVHN):
    def __init__(self,root: str,transform=None, train: bool =True):
        super().__init__(root=root,transform=transform)
        self.task = self.__class__.__name__.replace("RE","")
        self.targets = torch.from_numpy(np.load(f"behaviour_dataset/{self.task}/y_merge.npz")["arr_0"])
        self.replace_dcit = self.get_replace_dict()

    def __len__(self) -> int:
        return int(len(self.data)*0.5)
    

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
        return img, target
    
    
class REFlowers102(torchvision.datasets.Flowers102):
    def __init__(self,root: str,transform=None, train: bool =True):
        super().__init__(root=root,transform=transform)
        self.task = self.__class__.__name__.replace("RE","")
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
        return img, target
    

class REOxfordIIITPet(torchvision.datasets.OxfordIIITPet):
    def __init__(self,root: str,transform=None, train: bool =True):
        super().__init__(root=root,transform=transform)
        self.task = self.__class__.__name__.replace("RE","")
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
        return img, target
    

class REStanfordCars(torchvision.datasets.StanfordCars):
    def __init__(self,root: str,transform=None, train: bool =True):
        super().__init__(root=root,transform=transform)
        self.task = self.__class__.__name__.replace("RE","")
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
        return img, target
    
class REImageFolder(ImageFolder):
    def __init__(self,root: str,transform=None):
        super().__init__(root=root,transform=transform)

    def __getitem__(self, index: int):

        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return index,sample, target
    
def build_dataloader(args):
    if args.dataset=="MNIST":
        train_dataset = REMNIST(root='/mnt/st_data/dataset', transform=transform32) 
        val_dataset = torchvision.datasets.MNIST(root='/mnt/st_data/dataset',train=False, transform=transform32)
    elif args.dataset=="FashionMNIST":
        train_dataset = REFashionMNIST(root='/mnt/st_data/dataset', transform=transform32)
        val_dataset = torchvision.datasets.FashionMNIST(root='/mnt/st_data/dataset',train=False, transform=transform32)
    elif args.dataset=="CIFAR10":
        train_dataset = RECIFAR10(root='/mnt/st_data/dataset', transform=transform32)
        val_dataset = torchvision.datasets.CIFAR10(root='/mnt/st_data/dataset',train=False, transform=transform32)
    elif args.dataset=="CIFAR100":   
        train_dataset = RECIFAR100(root='/mnt/st_data/dataset', transform=transform32) 
        val_dataset = torchvision.datasets.CIFAR100(root='/mnt/st_data/dataset',train=False, transform=transform32)
    elif args.dataset=="SVHN":    
        train_dataset = RESVHN(root='/mnt/st_data/dataset', transform=transform32) 
        val_dataset = torchvision.datasets.SVHN(root='/mnt/st_data/dataset',split="test", transform=transform32,download=True)
    elif args.dataset=="Flowers102":
        train_dataset = REFlowers102(root='/mnt/st_data/dataset', transform=transform224) 
        val_dataset = torchvision.datasets.Flowers102(root='/mnt/st_data/dataset',split="test", transform=transform32,download=True)
    elif args.dataset=="OxfordIIITPet":    
        train_dataset = REOxfordIIITPet(root='/mnt/st_data/dataset', transform=transform224) 
        val_dataset = torchvision.datasets.OxfordIIITPet(root='/mnt/st_data/dataset',split="test", transform=transform32,download=True)
    elif args.dataset=="StanfordCars":    
        train_dataset = REStanfordCars(root='/mnt/st_data/dataset', transform=transform224) 
        val_dataset = torchvision.datasets.StanfordCars(root='/mnt/st_data/dataset',split="test", transform=transform32,download=True)
    elif args.dataset=="imagenet_lt":
        train_dataset = REImageFolder('/mnt/st_data/dataset/imagenet_lt/train-tiny',transform=transform224)
        val_dataset = ImageFolder('/mnt/st_data/dataset/imagenet_lt/val',transform=transform224)
    elif args.dataset=="places-lt":
        train_dataset = REImageFolder('/mnt/st_data/dataset/places-lt/train',transform=transform224)
        val_dataset = ImageFolder('/mnt/st_data/dataset/places-lt/val',transform=transform224)

    train_loader = DataLoader(dataset=train_dataset,batch_size=128,shuffle=True,num_workers=16,pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset,batch_size=128,shuffle=True,num_workers=16,pin_memory=True)
    uncertain_loader = DataLoader(dataset=train_dataset,batch_size=128,shuffle=False,num_workers=16,pin_memory=True)
    print(f"train_loader number: {len(train_dataset)} val_loader number: {len(val_dataset)}")
    
    for x in train_loader:
        print(x)
    
    return train_loader,val_loader,uncertain_loader


if __name__ == "__main__":

    # if args.dataset=="MNIST":
    # elif args.dataset=="FashionMNIST":
    # elif args.dataset=="CIFAR10":
    # elif args.dataset=="CIFAR100":   
    # elif args.dataset=="SVHN":    
    # elif args.dataset=="Flowers102":
    # elif args.dataset=="OxfordIIITPet":    
    # elif args.dataset=="StanfordCars":    
    # elif args.dataset=="imagenet_lt":
    # elif args.dataset=="places-lt":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='imagenet_lt', help='select model')
    args = parser.parse_args()

    build_dataloader(args)