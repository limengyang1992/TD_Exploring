import numpy as np
import random


dataset = "cifar100"
num_classes = 10 if dataset=="cifar10" else 100

data_x_init = np.load(f"behaviour_dataset/{dataset}_x_init.npy")
data_y_init= np.load(f"behaviour_dataset/{dataset}_y_init.npy")
data_x_adver = np.load(f"behaviour_dataset/{dataset}_x_adver.npy")
data_x_cutout = np.load(f"behaviour_dataset/{dataset}_x_cutout.npy")
data_x_mixup = np.load(f"behaviour_dataset/{dataset}_x_mixup.npy")
data_y_mixup = np.load(f"behaviour_dataset/{dataset}_y_mixup.npy")

assert len(data_x_init) == len(data_y_init)
assert len(data_x_init) == len(data_x_adver)
assert len(data_x_init) == len(data_x_cutout)
assert len(data_x_init) == len(data_x_mixup)
assert len(data_x_init) == len(data_y_mixup)

num_dataset = len(data_x_init)
num_chiose = int(0.001*len(data_x_adver))

random_list = random.sample(range(num_dataset),num_chiose)

adver_list = random_list[:12]
cutout_list = random_list[12:24]
mixup_list = random_list[24:36]
noise_list = random_list[36:]


merge_x,merge_y = [],[]
for i,(x,y) in enumerate(zip(data_x_init,data_y_init)):
    
    if i in adver_list:
        merge_x.append(data_x_adver[i])
        label = np.array([y,y,1,1])
        merge_y.append(label)
        
    elif i in cutout_list:
        merge_x.append(data_x_cutout[i])
        label = np.array([y,y,1,2])
        merge_y.append(label)
                
    elif i in mixup_list:
        merge_x.append(data_x_mixup[i])
        y = list(data_y_mixup[i])
        label = np.array(y+[3])
        merge_y.append(label)
                
    elif i in noise_list:
        merge_x.append(x)
        noise_label = random.sample([t for t in range(num_classes) if t!=y],1)[0]
        label = np.array([noise_label,y,1,4])
        merge_y.append(label)
                
    else:
        merge_x.append(x)
        label = np.array([y,y,1,0])
        merge_y.append(label)
        
        
       
numpy_merge_x = np.stack(merge_x)         
numpy_merge_y = np.stack(merge_y)     

np.savez_compressed(f"behaviour_dataset/{dataset}_x_merge.npz",numpy_merge_x)
np.savez_compressed(f"behaviour_dataset/{dataset}_y_merge.npz",numpy_merge_y)
    