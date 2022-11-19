import torchvision.datasets as datasets
from torch.utils.data import DataLoader,Dataset
import torch
import numpy as np
from .transforms import build_transforms

from typing import Any, Callable, Optional, Tuple
from torchvision.datasets.vision import VisionDataset
from PIL import Image


class CIFAR(Dataset):
   
    def __init__(
        self,
        data_path,
        targets_path,
        transform):

        self.data = np.load(data_path)
        self.data = (255*(self.data/2+0.5)).astype(np.uint8).clip(min=0, max=255)
        self.data = self.data.transpose((0, 2, 3, 1)) 
        self.targets = np.load(targets_path)
        self.transform = transform


    def __getitem__(self, index):

        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return index, img, target

    def __len__(self) -> int:
        return len(self.data)

    

def build_dataset(type='train',
                  name='cifar10',
                  root='~/data',
                  args=None,
                  fast=False):
    assert name in ['cifar10', 'cifar100']
    assert type in ['train', 'val']

    dataset_type = None

    if name == 'cifar10':
        if type == 'train':
            dataset_type = CIFAR(
                data_path=f"behaviour_dataset/{name}_x_merge.npy",
                targets_path=f"behaviour_dataset/{name}_y_merge.npy",
                transform=build_transforms('cifar10', 'train', args=args),
            )
        elif type == 'val':
            dataset_type = datasets.CIFAR10(
                root=root,
                train=False,
                download=True,
                transform=build_transforms('cifar10', 'val', args=args),
            )

    elif name == 'cifar100':
        if type == 'train':
            dataset_type = CIFAR(
                data_path=f"behaviour_dataset/{name}_x_merge.npy",
                targets_path=f"behaviour_dataset/{name}_y_merge.npy",
                transform=build_transforms('cifar100', 'train', args=args),
            )
        elif type == 'val':
            dataset_type = datasets.CIFAR100(
                root=root,
                train=False,
                download=True,
                transform=build_transforms('cifar100', 'val', args=args),
            )
    else:
        raise 'Type Error: {} Not Supported'.format(name)

    if fast:
        # fast train using ratio% images
        ratio = 0.3
        total_num = len(dataset_type.targets)
        choice_num = int(total_num * ratio)
        print(f'FAST MODE: Choice num/Total num: {choice_num}/{total_num}')

        dataset_type.data = dataset_type.data[:choice_num]
        dataset_type.targets = dataset_type.targets[:choice_num]

    print('DATASET:', len(dataset_type))

    return dataset_type


def build_dataloader(name='cifar10', type='train', args=None):
    assert type in ['train', 'val']
    assert name in ['cifar10', 'cifar100']
    if name == 'cifar10':
        if type == 'train':
            dataloader_type = DataLoader(
                build_dataset('train',
                              'cifar10',
                              args.root,
                              args=args,
                              fast=args.fast),
                batch_size=args.bs,
                shuffle=True,
                num_workers=args.nw,
                pin_memory=True,
            )
        elif type == 'val':
            dataloader_type = DataLoader(
                build_dataset('val',
                              'cifar10',
                              args.root,
                              args=args,
                              fast=args.fast),
                batch_size=args.bs,
                shuffle=False,
                num_workers=args.nw,
                pin_memory=True,
            )
    elif name == 'cifar100':
        if type == 'train':
            dataloader_type = DataLoader(
                build_dataset('train',
                              'cifar100',
                              args.root,
                              args=args,
                              fast=args.fast),
                batch_size=args.bs,
                shuffle=True,
                num_workers=args.nw,
                pin_memory=True,
            )
        elif type == 'val':
            dataloader_type = DataLoader(
                build_dataset('val',
                              'cifar100',
                              args.root,
                              args=args,
                              fast=args.fast),
                batch_size=args.bs,
                shuffle=False,
                num_workers=args.nw,
                pin_memory=True,
            )
    else:
        raise 'Type Error: {} Not Supported'.format(name)

    return dataloader_type



# def build_dataloader(name='cifar10', type='train', args=None):
#     assert type in ['train', 'val']
#     assert name in ['cifar10', 'cifar100']
    
#     train_x = torch.from_numpy(np.load(f"behaviour_dataset/{name}_x_merge.npy"))
#     train_y = torch.from_numpy(np.load(f"behaviour_dataset/{name}_y_merge.npy"))
#     train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
#     if name == 'cifar10':
#         if type == 'train':
#             dataloader_type = DataLoader(
#                 train_dataset,
#                 batch_size=args.bs,
#                 shuffle=True,
#                 num_workers=args.nw,
#                 pin_memory=True,
#             )
#         elif type == 'val':
#             dataloader_type = DataLoader(
#                 build_dataset('val',
#                               'cifar10',
#                               args.root,
#                               args=args,
#                               fast=args.fast),
#                 batch_size=args.bs,
#                 shuffle=False,
#                 num_workers=args.nw,
#                 pin_memory=True,
#             )
#     elif name == 'cifar100':
#         if type == 'train':
#             dataloader_type = DataLoader(
#                 train_dataset,
#                 batch_size=args.bs,
#                 shuffle=True,
#                 num_workers=args.nw,
#                 pin_memory=True,
#             )
#         elif type == 'val':
#             dataloader_type = DataLoader(
#                 build_dataset('val',
#                               'cifar100',
#                               args.root,
#                               args=args,
#                               fast=args.fast),
#                 batch_size=args.bs,
#                 shuffle=False,
#                 num_workers=args.nw,
#                 pin_memory=True,
#             )
#     else:
#         raise 'Type Error: {} Not Supported'.format(name)

#     return dataloader_type
