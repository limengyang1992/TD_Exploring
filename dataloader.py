

import os
import time
import glob
import numpy as np
from torch.utils.data import Dataset,DataLoader

class EpochDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
 
    def __len__(self):
        return len(self.samples)
 
    def __getitem__(self, index):
        paths = self.samples[index]
        inputs1 =  np.load(paths)[:,2:-1]
        target1 =  np.load(paths)[:,-1]
        target2= np.load(paths)[:,0]
        return inputs1,target1,target2
    
    
start = time.time()   
samples = glob.glob("/home/kunyu/exps/*/*/*.npy")
total_numberl = len(samples)
print(f"loading succsee. total number:{total_numberl} total time: {time.time()-start}")

split_index_train = int(total_numberl*0.8)
split_index_test = int(total_numberl*0.9)
print(f"split_index_train:{split_index_train} split_index_test: {split_index_test}")


train_dataset = EpochDataset(samples[:split_index_train])
valid_dataset = EpochDataset(samples[split_index_train:split_index_test])
test_dataset = EpochDataset(samples[split_index_test:])

train_dataloader = DataLoader(train_dataset, batch_size=200, shuffle=True,num_workers=20)
valid_dataloader = DataLoader(valid_dataset, batch_size=200, shuffle=True,num_workers=20)
test_dataloader = DataLoader(test_dataset, batch_size=200, shuffle=True,num_workers=20)



if __name__ == "__main__":
    
    import time 
    for epoch in range(2):
        start = time.time()
        for i,x in enumerate(train_dataloader):
            print(i,"time spend:",time.time()-start)
            start = time.time()
