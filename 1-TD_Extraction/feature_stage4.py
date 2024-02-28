
import os
import glob
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler


def fill_ndarray(data):
    for i in range(data.shape[1]):
        tmp_col = data[:, i]  # 当前的一列
        nan_num = np.count_nonzero(tmp_col != tmp_col)  # 当前一列不为nan的array
        if nan_num != 0:  # 说明这一列有nan
            tmp_not_nan_clo = tmp_col[tmp_col == tmp_col]  # 当前列中不是nan的array
            # 选中当前为nan的位置,赋其值为不为nan元素的均值
            tmp_col[np.isnan(tmp_col)] = tmp_not_nan_clo.mean()
    return data

    
# Load from file
pkl_filename = "behaviour_dataset/scaler.pkl"
with open(pkl_filename, 'rb') as file:
    scaler_model = pickle.load(file)

# 加载数据目录
root_path = glob.glob(os.path.join("exps","*/feature_total"))

for task in os.listdir("exps"):
    if "cifar100" in task:continue
    if "01M_29D_10H__75" in task:continue
    feature_path = os.path.join("exps",task,"feature_total")
    save_dir = os.path.join("exps",task,"feature_scale")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for t in range(0,200):
        print(task,t)
        current = os.path.join(feature_path,f"total_feature_{t}.npy")
        numpy_current = np.load(current)
        numpy_current[np.isinf(numpy_current)] = np.nan
        numpy_current = fill_ndarray(numpy_current)
        left = scaler_model.transform(numpy_current[:,:-1])
        total = np.concatenate([left,numpy_current[:,-1][:, np.newaxis]], axis=1)
        save_path = os.path.join(save_dir,f"feature_scale_{t}.npy")
        np.save(save_path,total)
        
        
        
