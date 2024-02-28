
import os
import random
import glob
import numpy as np
from torch.utils.data import Dataset,DataLoader

np.random.seed(42)
random.seed(42) # n就是你想设置的随机种子


class EpochDataset(Dataset):
    def __init__(self, samples, train_mode):
        self.samples = samples
        self.p = 0.5
        self.sigma = 0.01
        self.train_mode = train_mode
        
    def __len__(self):
        return len(self.samples)
 
    def __getitem__(self, index):
        paths,label = self.samples[index]
        ts =  np.load(paths)
        if self.train_mode:
            return self.transform(ts),label
        else:
            return ts,label,paths

    def transform(self, x):
        return self.jitter(self.shift(self.scale(x)))

    def jitter(self, x):
        if random.random() > self.p:
            return x
        return x + (np.random.randn(x.shape[0],x.shape[1]) * self.sigma)

    def scale(self, x):
        if random.random() > self.p:
            return x
        return x * (np.random.randn(x.shape[-1]) * self.sigma + 1)

    def shift(self, x):
        if random.random() > self.p:
            return x
        return x + (np.random.randn(x.shape[-1]) * self.sigma)


def random_dic(dicts):
    dict_key_ls = list(dicts.keys())
    random.shuffle(dict_key_ls)
    new_dic = {}
    for key in dict_key_ls:
        new_dic[key] = dicts.get(key)
    return new_dic
    
    
def dataloader_finetune(batch_size):
    
    dirs_label = {}
    for d in os.listdir("behaviour_dataset"):
        dirs_label[d] = np.load(f"behaviour_dataset/{d}/y_merge.npz")["arr_0"]

    samples = glob.glob(f"/home/kunyu/exps/sequence_wise/*/sequence_*.npy")
    labels = []
    for sample in samples:
        sample = sample.replace("places-lt","places_lt")
        index = int(os.path.split(sample)[-1].split("_")[1].split(".")[0])
        task = os.path.split(sample)[0].split("_d-")[1].split("__")[0]
        label = dirs_label[f"{task}"][index][-1]
        labels.append(label)

    datasets = dict(zip(samples,labels))

    choose_0= dict((k,v)  for (k,v) in datasets.items() if v==0 and random.random()>0.7)
    choose_1= dict((k,v)  for (k,v) in datasets.items() if v==1)
    choose_2= dict((k,v)  for (k,v) in datasets.items() if v==2)
    choose_3= dict((k,v)  for (k,v) in datasets.items() if v==3)
    choose_4= dict((k,v)  for (k,v) in datasets.items() if v==4)


    choose_0 = random_dic(choose_0)
    choose_1 = random_dic(choose_1)
    choose_2 = random_dic(choose_2)
    choose_3 = random_dic(choose_3)
    choose_4 = random_dic(choose_4)

    choose_0_train = list(choose_0.keys())[5000:]
    choose_0_test = list(choose_0.keys())[:5000]

    choose_1_train = list(choose_1.keys())[5000:]
    choose_1_test = list(choose_1.keys())[:5000]

    choose_2_train = list(choose_2.keys())[5000:]
    choose_2_test = list(choose_2.keys())[:5000]

    choose_3_train = list(choose_3.keys())[5000:]
    choose_3_test = list(choose_3.keys())[:5000]

    choose_4_train = list(choose_4.keys())[5000:]
    choose_4_test = list(choose_4.keys())[:5000]

    train_x = choose_0_train + choose_1_train*4 + choose_2_train*3 + choose_3_train*4  + choose_4_train*2
    train_y = [0]*len(choose_0_train) +[1]*len(choose_1_train)*4 +[2]*len(choose_2_train)*3 +[3]*len(choose_3_train)*4 + [4]*len(choose_4_train) *2

    test_x = choose_0_test + choose_1_test+ choose_2_test+ choose_3_test+ choose_4_test
    test_y = [0]*5000 + [1]*5000 +[2]*5000 +[3]*5000 +[4]*5000

    train_set = [(x,y) for (x,y) in zip(train_x,train_y) ]
    test_set = [(x,y) for (x,y) in zip(test_x,test_y)]

    
    train_dataset = EpochDataset(train_set,train_mode=True)
    test_dataset = EpochDataset(test_set,train_mode=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=64)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=20)
    
    return train_loader,test_loader



if __name__ == '__main__':
    pass