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
from torchvision.models import resnet18

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)

FLAGS = flags.FLAGS

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

dataset_StanfordCars = torchvision.datasets.StanfordCars(root='/mnt/st_data/dataset', transform=transform224, download=True)
dataset_OxfordIIITPet = torchvision.datasets.OxfordIIITPet(root='/mnt/st_data/dataset', transform=transform224, download=True)
dataset_Flowers102 = torchvision.datasets.Flowers102(root='/mnt/st_data/dataset',  transform=transform224, download=True)

def main(_):
    # Load training and test data
    if FLAGS.dataset == "StanfordCars":
        loader = DataLoader(dataset=dataset_StanfordCars,batch_size=128,shuffle=True,num_workers=8)
        num_classes = len(dataset_StanfordCars.classes)
    elif FLAGS.dataset == "OxfordIIITPet":
        loader = DataLoader(dataset=dataset_OxfordIIITPet,batch_size=128,shuffle=True,num_workers=8)
        num_classes = len(dataset_OxfordIIITPet.classes)
    elif FLAGS.dataset == "Flowers102":
        loader = DataLoader(dataset=dataset_Flowers102,batch_size=128,shuffle=True,num_workers=8)
        num_classes = 102


        
    net = resnet18(num_classes= num_classes)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        net = net.cuda()
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    # Train vanilla model
    net.train()
    for epoch in range(3):
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
    flags.DEFINE_string("dataset", "StanfordCars", "Number of epochs.")
    flags.DEFINE_float("eps", 0.2, "Total epsilon for FGM and PGD attacks.")
    flags.DEFINE_bool(
        "adv_train", False, "Use adversarial training (on PGD adversarial examples)."
    )

    app.run(main)