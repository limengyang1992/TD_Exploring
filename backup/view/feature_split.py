import glob
import os
import numpy as np
import time
from multiprocessing import Pool


epoch_piece = 20      
root_path = glob.glob(os.path.join("exps","*/feature_scale"))

# 加载数据目录
for task in os.listdir("exps"):
    save_dir = os.path.join("/home/kunyu/exps",task,"feature_block")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(save_dir)
        
        
def split_epoch(path,epoch,piece):
    data = np.load(os.path.join(path,f"feature_scale_{epoch}.npy"))
    replace_path = path.replace("feature_scale","feature_block")
    save_path = os.path.join("/home/kunyu",replace_path,f"feature_scale_{epoch}_{piece*epoch_piece}.npy")
    np.save(save_path,data[piece*epoch_piece:(piece+1)*epoch_piece])
        

pools = []
start  = time.time()
p = Pool(64)
for path in root_path:
    for epoch in range(200):
        for piece in range(50000//epoch_piece):
            pools.append(p.apply_async(split_epoch, args=(path,epoch,piece)))
p.close()
p.join()
print("spend",time.time()-start)
            