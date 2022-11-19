from absl import app, flags
from easydict import EasyDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms

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


def cifar_dataset(dataset):
    """Load training and test data."""
    
 
    train_transforms = build_transforms(dataset, type='train', cutout=False)
    test_transforms = build_transforms(dataset, type='val', cutout=False)
    if dataset=="cifar10":
        train_dataset = torchvision.datasets.CIFAR10(
            root="../data", train=True, transform=train_transforms, download=True
        )

        test_dataset = torchvision.datasets.CIFAR10(
            root="../data", train=False, transform=test_transforms, download=True
        )
    else:
        train_dataset = torchvision.datasets.CIFAR100(
            root="../data", train=True, transform=train_transforms, download=True
        )

        test_dataset = torchvision.datasets.CIFAR100(
            root="../data", train=False, transform=test_transforms, download=True
        )
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=2
    )
    train_loader_adver = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=False, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=2
    )
    return EasyDict(train=train_loader, test=test_loader, adver=train_loader_adver)




def main(_):
    # Load training and test data
    data = cifar_dataset(FLAGS.dataset)

    # Instantiate model, loss, and optimizer for training
    net = CNN(num_classes= 10 if FLAGS.dataset=="cifar10" else 100)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        net = net.cuda()
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    # Train vanilla model
    net.train()
    for epoch in range(1, FLAGS.nb_epochs + 1):
        train_loss = 0.0
        for x, y in data.train:
            x, y = x.to(device), y.to(device)
            if FLAGS.adv_train:
                x = projected_gradient_descent(net, x, FLAGS.eps, 0.01, 40, np.inf)
            optimizer.zero_grad()
            loss = loss_fn(net(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(
            "epoch: {}/{}, train loss: {:.3f}".format(
                epoch, FLAGS.nb_epochs, train_loss
            )
        )

    # Evaluate on clean and adversarial data
    net.eval()
    report = EasyDict(nb_test=0, correct=0, correct_fgm=0, correct_pgd=0)
    
    for x, y in data.test:
        x, y = x.to(device), y.to(device)
        x_fgm = fast_gradient_method(net, x, FLAGS.eps, np.inf)
        x_pgd = projected_gradient_descent(net, x, FLAGS.eps, 0.01, 20, np.inf)
        _, y_pred = net(x).max(1)  # model prediction on clean examples
        _, y_pred_fgm = net(x_fgm).max(
            1
        )  # model prediction on FGM adversarial examples
        _, y_pred_pgd = net(x_pgd).max(
            1
        )  # model prediction on PGD adversarial examples
        report.nb_test += y.size(0)
        report.correct += y_pred.eq(y).sum().item()
        report.correct_fgm += y_pred_fgm.eq(y).sum().item()
        report.correct_pgd += y_pred_pgd.eq(y).sum().item()
        
    adver = []
    for x, y in data.adver:
        x, y = x.to(device), y.to(device)
        x_pgd = projected_gradient_descent(net, x, FLAGS.eps, 0.01, 20, np.inf)
        adver.append(x_pgd)
    np.save(f"behaviour_dataset/{FLAGS.dataset}_x_adver.npy",torch.cat(adver).cpu().detach().numpy())
    
    print(
        "test acc on clean examples (%): {:.3f}".format(
            report.correct / report.nb_test * 100.0
        )
    )
    print(
        "test acc on FGM adversarial examples (%): {:.3f}".format(
            report.correct_fgm / report.nb_test * 100.0
        )
    )
    print(
        "test acc on PGD adversarial examples (%): {:.3f}".format(
            report.correct_pgd / report.nb_test * 100.0
        )
    )


if __name__ == "__main__":
    flags.DEFINE_integer("nb_epochs", 8, "Number of epochs.")
    flags.DEFINE_string("dataset", "cifar100", "Number of epochs.")
    flags.DEFINE_float("eps", 0.3, "Total epsilon for FGM and PGD attacks.")
    flags.DEFINE_bool(
        "adv_train", False, "Use adversarial training (on PGD adversarial examples)."
    )

    app.run(main)