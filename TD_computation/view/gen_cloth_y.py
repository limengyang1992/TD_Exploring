import numpy as np
import random


dataset = "clothing"
types = "noise"
rate = ""


merge_x,merge_y = [],[]
with open(f"labels/train_{types}_{rate}.txt") as f:
    lines = f.readlines()
    for line in lines:
        p,a,b = line.strip().split(" ")
        label = np.array([int(b),int(a),1,0])
        merge_y.append(label)
        
        
numpy_merge_y = np.stack(merge_y)    


task_path = f"behaviour_dataset/{dataset}_{types}_{str(rate)}"
import os
if not os.path.exists(task_path):
    os.makedirs(task_path)
    
     
np.savez_compressed(f"{task_path}/y_merge.npz",numpy_merge_y)
    