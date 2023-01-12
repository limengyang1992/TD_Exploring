
import os
import time
import glob
import numpy as np
import pandas as pd

# total_paths = glob.glob(os.path.join("exps","*/feature_block/*.npy"))
# total_index = np.random.permutation(len(total_paths)*100)

total_paths = pd.read_csv("paths.csv",header=None).iloc[:,1]
total_index = np.load("total_index.npy")

total_index_1 = total_index[:25000000]
total_index_2 = total_index[25000000:50000000]
total_index_3 = total_index[50000000:75000000]
total_index_4 = total_index[75000000:]


choose_index = total_index_4
choose_start_path = "epoch_75w_100w"


for i in range(int(len(choose_index)/2000)):
    start,end = i*2000,(i+1)*2000
    split = choose_index[start:end]
    total = []
    for j,num in enumerate(split):
        path = total_paths[num//100]
        index = num%100
        epoch = int(path.split("_")[-2])
        NOs = int(path.split("_")[-1].split(".")[0])+index
        data = np.load(path)[index]
        merge_data = np.concatenate([np.array([epoch,NOs]),data], axis=0)
        total.append(merge_data[np.newaxis,:])
    result = np.concatenate(total, axis=0)
    save_path = os.path.join("exps","epoch-wise-dataset",f"{choose_start_path}_{i}.npy")
    np.save(save_path,result)
    print(i,result.shape)
    
