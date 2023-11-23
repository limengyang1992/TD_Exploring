# 我们这里还是对MNIST进行处理，初始的MNIST是 28 * 28，我们把它处理成 96 * 96 的torch.Tensor的格式
from torchvision import transforms as transforms
import torchvision
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
 
# 图像预处理步骤
transform = transforms.Compose([
    transforms.Resize(96), # 缩放到 96 * 96 大小
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5)) # 归一化
])
 
DOWNLOAD = True
BATCH_SIZE = 32
 
# train_dataset = torchvision.datasets.MNIST(root='/mnt/st_data/dataset', train=True, transform=transform, download=DOWNLOAD)
 
# train_dataset = torchvision.datasets.CIFAR10(root='/mnt/st_data/dataset', train=True, transform=transform, download=DOWNLOAD)

# train_dataset = torchvision.datasets.CIFAR100(root='/mnt/st_data/dataset', train=True, transform=transform, download=DOWNLOAD)

# train_dataset = torchvision.datasets.FashionMNIST(root='/mnt/st_data/dataset', train=True, transform=transform, download=DOWNLOAD)

# train_dataset = torchvision.datasets.SVHN(root='/mnt/st_data/dataset', split="train", transform=transform, download=DOWNLOAD)

# train_dataset = torchvision.datasets.StanfordCars(root='/mnt/st_data/dataset', split="train", transform=transform, download=DOWNLOAD)
# train_dataset = torchvision.datasets.OxfordIIITPet(root='/mnt/st_data/dataset', split="trainval", transform=transform, download=DOWNLOAD)



train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True)
 
print(len(train_dataset))
print(len(train_loader))


