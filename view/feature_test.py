import numpy as np
import os

feat = np.load("exps/e-gpu0_m-mobilenetv2_d-cifar10__12M_05D_05H__21/feature/total_feat_0.npy")


for task in os.listdir("exps")[::-1]:
        print(f"current task : {task}")