import os
import time
import glob
import numpy as np
import pandas as pd


for task in os.listdir("exps"):
    epochs = {}
    for i in range(200):
        print("epoch",i)
        p = f"exps/{task}/feature_scale/feature_scale_{i}.npy"
        epochs[i] = np.load(p)
        
    for i in range(50000):
        print("save",i)
        total = []
        for epoch,data in epochs.items():
            total.append(data[i][np.newaxis,:])
        data = np.concatenate(total, axis=0)
        save_dir = f"/home/kunyu/exps/sequence_wise/{task}"
        save_path = f"{save_dir}/sequence_{i}.npy"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(save_path,data)
        
    
        
    
    
