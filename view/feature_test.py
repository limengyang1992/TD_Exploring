import numpy as np
import os

feat = np.load("exps/e-gpu0_m-wideresnet40_10_d-cifar10__12M_12D_12H__24/feature/total_feat_0.npy")

print(feat.shape)
# for task in os.listdir("exps")[::-1]:
#         print(f"current task : {task}")