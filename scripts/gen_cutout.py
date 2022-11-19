from easydict import EasyDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


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


def build_transforms(name='cifar10', type='train', cutout=False):
    assert type in ['train', 'val']
    assert name in ['cifar10', 'cifar100']
    transform_type = None

    if type == 'train':
        base_transform = [
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
        ]

        if name == 'cifar10':
            post_transform = [
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                # transforms.Normalize((0.4914, 0.4822, 0.4465),
                #                      (0.2023, 0.1994, 0.2010)),
            ]
        elif name == 'cifar100':
            post_transform = [
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ]

        if cutout:
            post_transform.append(Cutout(1, 4))

        transform_type = transforms.Compose([*base_transform, *post_transform])

    elif type == 'val':
        if name == 'cifar10':
            transform_type = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ])
        elif name == 'cifar100':
            transform_type = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ])
    else:
        raise 'Type Error in transforms'

    return transform_type


def cifar_dataset(name):
    """Load training and test data."""
    
 
    train_transforms = build_transforms(name, type='train', cutout=False)
    test_transforms = build_transforms(name, type='val', cutout=False)
    train_transforms_cutout = build_transforms(name, type='train', cutout=True)
    if name=="cifar10":
        train_dataset = torchvision.datasets.CIFAR10(
            root="../data", train=True, transform=train_transforms, download=True
        )
        train_dataset_cutout = torchvision.datasets.CIFAR10(
            root="../data", train=True, transform=train_transforms_cutout, download=True
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root="../data", train=False, transform=test_transforms, download=True
        )
    else:
        train_dataset = torchvision.datasets.CIFAR100(
            root="../data", train=True, transform=train_transforms, download=True
        )
        train_dataset_cutout = torchvision.datasets.CIFAR100(
            root="../data", train=True, transform=train_transforms_cutout, download=True
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root="../data", train=False, transform=test_transforms, download=True
        )       
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=False, num_workers=2
    )
    train_loader_cutout = torch.utils.data.DataLoader(
        train_dataset_cutout, batch_size=128, shuffle=False, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=2
    )
    return EasyDict(train=train_loader, test=test_loader, cutout=train_loader_cutout)


def main(name):
    # Load training and test data
    data = cifar_dataset(name)
    # Instantiate model, loss, and optimizer for training

    init_x,init_y = [],[]
    for x, y in data.train:
        init_x.append(x)
        init_y.append(y)
        
    cutout = []
    for x, y in data.cutout:
        cutout.append(x)

                
    mixup,mixup_label= [],[] 
    for x, y in data.train:
        idx = torch.randperm(x.size(0))
        input_a, input_b = x, x[idx]
        target_a, target_b = y, y[idx]
        l = np.random.beta(1.0,1.0)
        mixed_input = l * input_a + (1 - l) * input_b
        mixup.append(mixed_input)
        mixup_label.append(torch.stack([target_a.float(),target_b.float(),torch.ones_like(target_a)*l],dim=1))
        
        
    np.save(f"behaviour_dataset/{name}_x_init.npy",torch.cat(init_x).cpu().detach().numpy())
    np.save(f"behaviour_dataset/{name}_y_init.npy",torch.cat(init_y).cpu().detach().numpy())

    np.save(f"behaviour_dataset/{name}_x_cutout.npy",torch.cat(cutout).cpu().detach().numpy())
    np.save(f"behaviour_dataset/{name}_x_mixup.npy",torch.cat(mixup).cpu().detach().numpy())
    np.save(f"behaviour_dataset/{name}_y_mixup.npy",torch.cat(mixup_label).cpu().detach().numpy())
    

if __name__ == "__main__":

    main("cifar100")