from absl import app, flags
from easydict import EasyDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)

FLAGS = flags.FLAGS

class CNN(torch.nn.Module):
    """Basic CNN architecture."""

    def __init__(self, num_classes, in_channels=3):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 8, 1)
        self.conv2 = nn.Conv2d(64, 128, 6, 2)
        self.conv3 = nn.Conv2d(128, 128, 5, 2)
        self.fc = nn.Linear(128 * 3 * 3, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128 * 3 * 3)
        x = self.fc(x)
        return x

transform32 = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5]),
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
    
dataset_MNIST = REMNIST(root='/mnt/st_data/dataset',  transform=transform32)          
dataset_CIFAR10 = torchvision.datasets.CIFAR10(root='/mnt/st_data/dataset',  transform=transform32, download=True)
dataset_CIFAR100 = torchvision.datasets.CIFAR100(root='/mnt/st_data/dataset',  transform=transform32, download=True)
dataset_FashionMNIST = REFashionMNIST(root='/mnt/st_data/dataset', transform=transform32)
dataset_SVHN = torchvision.datasets.SVHN(root='/mnt/st_data/dataset',  transform=transform32, download=True)



def main(_):
    # Load training and test data
    if FLAGS.dataset == "MNIST":
        loader = DataLoader(dataset=dataset_MNIST,batch_size=128,shuffle=True,num_workers=8)
        num_classes = 10
    elif FLAGS.dataset == "CIFAR10":
        loader = DataLoader(dataset=dataset_CIFAR10,batch_size=128,shuffle=True,num_workers=8)
        num_classes = 10
    elif FLAGS.dataset == "CIFAR100":
        loader = DataLoader(dataset=dataset_CIFAR100,batch_size=128,shuffle=True,num_workers=8)
        num_classes = 100
    elif FLAGS.dataset == "FashionMNIST":
        loader = DataLoader(dataset=dataset_FashionMNIST,batch_size=128,shuffle=True,num_workers=8)
        num_classes = len(dataset_FashionMNIST.classes)
    elif FLAGS.dataset == "SVHN":
        loader = DataLoader(dataset=dataset_SVHN,batch_size=128,shuffle=True,num_workers=8)
        num_classes = 10

    net = CNN(num_classes= num_classes)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        net = net.cuda()
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    # Train vanilla model
    net.train()
    for epoch in range(5):
        loss = 0.0
        for x,y in loader:
            x, y = x.to(device), y.to(device)
            x = projected_gradient_descent(net, x, FLAGS.eps, 0.01, 20, np.inf)
            optimizer.zero_grad()
            loss = loss_fn(net(x), y)
            loss.backward()
            optimizer.step()
            loss += loss.item()
        print(f"epoch: {epoch}, train loss: {loss}")
        torch.save(net,f"behaviour_dataset/net_AT/net_{FLAGS.dataset}.pt")


if __name__ == "__main__":
    flags.DEFINE_string("dataset", "FashionMNIST", "Number of epochs.")
    flags.DEFINE_float("eps", 0.2, "Total epsilon for FGM and PGD attacks.")

    app.run(main)