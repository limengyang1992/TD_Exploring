import numpy as np
import random


dataset = "KMNIST"
num_classes = 10
type = "flip"
# type = "sym"
rate = 0.1
print(rate,type)

data_y_init= np.load(f"behaviour_dataset/{dataset}/y_merge.npz")["arr_0"][:,:1]
flip_dict = [random.sample([t for t in range(num_classes) if t!=y],1)[0]  for y in range(num_classes)]

num_dataset = len(data_y_init)
num_chiose = int(rate*len(data_y_init))
noise_list = random.sample(range(num_dataset),num_chiose)

merge_x,merge_y = [],[]
for i,y in enumerate(data_y_init):
    y=y[0]
    
    if i in noise_list:
        if type =="sym":
            noise_label = random.sample([t for t in range(num_classes) if t!=y],1)[0]
        else:
            noise_label = flip_dict[y]
        label = np.array([noise_label,y,1,4])
        merge_y.append(label)
                
    else:
        label = np.array([y,y,1,0])
        merge_y.append(label)
        
numpy_merge_y = np.stack(merge_y)    


task_path = f"behaviour_dataset/{dataset}_{type}_{str(rate)}"
import os
if not os.path.exists(task_path):
    os.makedirs(task_path)
    
     
np.savez_compressed(f"{task_path}/y_merge.npz",numpy_merge_y)
    