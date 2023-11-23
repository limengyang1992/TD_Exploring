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


pd.DataFrame(dict_label.values()).to_excel(f"{d}.xlsx")
        
    







