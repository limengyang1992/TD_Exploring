# 我们这里还是对MNIST进行处理，初始的MNIST是 28 * 28，我们把它处理成 96 * 96 的torch.Tensor的格式
from torchvision import transforms as transforms
import torchvision
from torch.utils.data import DataLoader

from torchvision.datasets import CIFAR10
 
# 图像预处理步骤
transform = transforms.Compose([
    transforms.Resize(96), # 缩放到 96 * 96 大小
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5)) # 归一化
])
 
DOWNLOAD = True
BATCH_SIZE = 32


class ReCIFAR10(CIFAR10):
    def __init__(self,root: str,train: bool = True,download: bool = False, transform=None,fast: bool =True):
        super().__init__(root, train, transform)
        if fast:
            # fast train using ratio% images
            ratio = 0.5
            total_num = len(self.data)
            choice_num = int(total_num * ratio)
            print(f'FAST MODE: Choice num/Total num: {choice_num}/{total_num}')
            self.data = self.data[:choice_num]
            self.targets = self.targets[:choice_num]
            
            # 选择探针数据，进行标签替换
            
            
            # 选择探针数据，进行样本替换



train_dataset = ReCIFAR10(root='/mnt/st_data/dataset', train=False, transform=transform, download=DOWNLOAD)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True)

# 顺序得固定下来

print(len(train_dataset))
print(len(train_loader))
print(train_dataset._labels[:100])
print(len(train_dataset._labels))
