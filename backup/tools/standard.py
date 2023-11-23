from sklearn.preprocessing import StandardScaler
import glob
import os
import numpy as np

def fill_ndarray(data):
    for i in range(data.shape[1]):
        tmp_col = data[:, i]  # 当前的一列
        nan_num = np.count_nonzero(tmp_col != tmp_col)  # 当前一列不为nan的array
        if nan_num != 0:  # 说明这一列有nan
            tmp_not_nan_clo = tmp_col[tmp_col == tmp_col]  # 当前列中不是nan的array
            # 选中当前为nan的位置,赋其值为不为nan元素的均值
            tmp_col[np.isnan(tmp_col)] = tmp_not_nan_clo.mean()
    return data


# 加载预处理方法
scaler = StandardScaler()

root_path = glob.glob(os.path.join("exps","*/feature_total"))
# 迭代，epoch维度
for path in root_path:
    for epoch in range(5):
        data = np.load(os.path.join(path,f"total_feature_{epoch}.npy"))
        data = data[~np.isnan(data).any(axis=1)]
        data = data[~np.isinf(data).any(axis=1)]
        print(data.shape,np.count_nonzero(np.isnan(data)),np.count_nonzero(np.isinf(data)))
        scaler.partial_fit(data)
   
print("+++++++++++++++++++++")     
for path in root_path:
    for epoch in range(10):
        data = np.load(os.path.join(path,f"total_feature_{epoch}.npy"))
        data[np.isinf(data)] = np.nan
        data = fill_ndarray(data)
        scaler.transform(data)
        print(data.shape,np.count_nonzero(np.isnan(data)),np.count_nonzero(np.isinf(data)))
        
        
import pickle
pkl_filename = "pickle_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(scaler, file)
# Load from file
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)
# Calculate the accuracy score and predict target values
