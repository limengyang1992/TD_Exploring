import numpy as np
import random


for dataset in ["cifar100","cifar10"]:
    num_classes = 10 if dataset=="cifar10" else 100

    data_x_init = np.load(f"behaviour_dataset/{dataset}_x_init.npy")
    data_y_init= np.load(f"behaviour_dataset/{dataset}_y_init.npy")
    assert len(data_x_init) == len(data_y_init)
    flip_dict = [random.sample([t for t in range(num_classes) if t!=y],1)[0]  for y in range(num_classes)]

    for rate in [0.2,0.4,0.6]:
        for type in ["flip","sym"]:
            print(rate,type)
            num_dataset = len(data_x_init)
            num_chiose = int(rate*len(data_x_init))
            noise_list = random.sample(range(num_dataset),num_chiose)

            merge_x,merge_y = [],[]
            for i,(x,y) in enumerate(zip(data_x_init,data_y_init)):
                
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
            np.savez_compressed(f"behaviour_dataset/{dataset}_y_{type}_{str(rate)}.npz",numpy_merge_y)
                