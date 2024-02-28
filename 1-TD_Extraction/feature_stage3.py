
import os
import glob
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler


# 加载预处理方法
scaler = StandardScaler()

# 加载数据目录
root_path = glob.glob(os.path.join("exps","*/feature_total"))

# # 迭代，epoch维度
for t,path in enumerate(root_path):
    print(path)
    for epoch in range(120):
        npy_path = os.path.join(path,f"total_feature_{epoch}.npy")
        assert os.path.exists(npy_path)
        data = np.load(npy_path)
        data = data[~np.isnan(data).any(axis=1)]
        data = data[~np.isinf(data).any(axis=1)]
        print(t,epoch,data.shape,np.count_nonzero(np.isnan(data)),np.count_nonzero(np.isinf(data)))
        scaler.partial_fit(data[:,:-1])

    pkl_filename = "behaviour_dataset/scaler.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(scaler, file)
    
