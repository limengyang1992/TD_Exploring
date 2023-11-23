
import numpy as np
import os
import glob
from pprint import pprint

# total_paths = glob.glob(os.path.join("/home/kunyu/exps","*/feature_block/*.npy"))

# pprint(len(total_paths))

data = np.load("behaviour_dataset/cifar100_y_flip_0.6.npz")["arr_0"]

print(data)