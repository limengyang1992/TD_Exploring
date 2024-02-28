import os

import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
# from pl_bolts.datamodules import CIFAR10DataModule
# from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch.optim.lr_scheduler import OneCycleLR,MultiStepLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy
from torchvision import transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader

seed_everything(7)




PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64
NUM_WORKERS = int(os.cpu_count() / 2)



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


def create_model():
    model = torchvision.models.resnet18(pretrained=False, num_classes=47)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model


class LitResnet(LightningModule):
    def __init__(self, lr=0.05):
        super().__init__()

        self.save_hyperparameters()
        self.model = create_model()

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y,task="multiclass", num_classes=47)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        steps_per_epoch = 45000 // BATCH_SIZE
        scheduler_dict = {
            "scheduler":MultiStepLR(optimizer, milestones=[30,80], gamma=0.1),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
    



class REEMNIST(torchvision.datasets.EMNIST):
    def __init__(self,root: str,transform=None, train: bool =True):
        super().__init__(root=root,train=train,transform=transform,split="bymerge",download=True)

    def __getitem__(self, index: int):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode="L").convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    


train_dataset = REEMNIST(root='/mnt/st_data/dataset', transform=transform32) 
val_dataset = REEMNIST(root='/mnt/st_data/dataset',train=False, transform=transform32_val)
train_loader = DataLoader(dataset=train_dataset,batch_size=128,shuffle=True,num_workers=16,pin_memory=True)
val_loader = DataLoader(dataset=val_dataset,batch_size=128,shuffle=True,num_workers=16,pin_memory=True)

model = LitResnet(lr=0.05)

trainer = Trainer(
    max_epochs=120,
    accelerator="auto",
    gpus=[3] if torch.cuda.is_available() else None,  # limiting got iPython runs
    logger=CSVLogger(save_dir="logs/"),
    callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10)],
)


trainer.fit(model, train_dataloaders=train_loader,val_dataloaders=val_loader)
trainer.test(model, val_dataloaders=val_loader)