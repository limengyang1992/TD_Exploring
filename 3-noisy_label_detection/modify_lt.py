import os
import glob
import numpy as np
import pandas as pd

"""
imagenet_lt  400>(23996)  <60(13109)
place_lt    1000>(20741)   <100 (10440)
长尾标签直接更改
"""
# d="imagenet_lt"
# d_class = 1000

d="places_lt"
d_class = 365

data = np.load(f"behaviour_dataset/{d}/y_merge_init.npz")["arr_0"]

dict_label = {}

total = list(data[:,0])
for i in range(d_class):
    dict_label[i]=total.count(i)
    
print(f"total_num {sum(dict_label.values())},total_len {len(total)}")
new_label = []
for label in data[:,0]:
    count = dict_label[int(label)]
    if d=="imagenet_lt":
        if count>400: new_label.append(2)
        elif count<60: new_label.append(3)
        else: new_label.append(0)
    elif d=="places_lt":
        if count>1000: new_label.append(2)
        elif count<100: new_label.append(3)
        else: new_label.append(0)

data[:,3] = np.array(new_label)

np.savez(f"behaviour_dataset/{d}/y_merge.npz",data)

        
    







