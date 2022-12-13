
import os
import glob
import numpy as np

if __name__ == "__main__":
    
    for task in os.listdir("exps"):
        feature_path = os.path.join("exps",task,"feature")
        save_dir = os.path.join("exps",task,"feature_total")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        for t in range(0,200):
            current = os.path.join(feature_path,f"total_feat_{t}.npy")
            numpy_current = np.load(current)
            numpy_rate1 = np.ones((50000,84))
            numpy_rate2 = np.ones((50000,84))

            if t>0:
                current_last_1 = os.path.join(feature_path,f"total_feat_{t-1}.npy")
                numpy_current_last_1 = np.load(current_last_1)
                numpy_rate1 = numpy_current_last_1[:,:84]/numpy_current[:,:84]
            if t>1:
                current_last_2 = os.path.join(feature_path,f"total_feat_{t-2}.npy")
                numpy_current_last_2 = np.load(current_last_2)
                numpy_rate2 = numpy_current_last_2[:,:84]/numpy_current[:,:84]
            
            
            total = np.concatenate([numpy_rate1,numpy_rate2,numpy_current[:,:84],numpy_current[:,87:]], axis=1)
            print(t,total.shape)
            save_path = os.path.join(save_dir,f"total_feature_{t}.npy")
            np.save(save_path,total)
