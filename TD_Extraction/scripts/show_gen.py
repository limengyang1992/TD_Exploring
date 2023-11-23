



import numpy as np
import torch
from PIL import Image

data = torch.load("behaviour_dataset/CIFAR100/cutout_17914.pt").detach().cpu().numpy()

data = (255*(data/2+0.5)).astype(np.uint8).clip(min=0, max=255)


Image.fromarray(np.transpose(data, (1, 2, 0))).save("123.png")