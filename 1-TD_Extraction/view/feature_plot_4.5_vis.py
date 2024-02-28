

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
plt.rcParams['figure.figsize']=(14.4,25.6)
plt.subplots_adjust(wspace = 0, hspace =1)

def sample(new_list):
    original_list = [x for x in range(len(new_list))]
    random.shuffle(original_list)
    selected_elements = original_list[:120]
    selected_elements.sort()
    res = [new_list[x] for x in selected_elements]
    return res


def plot_save(index,type,epoch=200):
    task = "e-gpu0_m-ResNet34_d-cifar100_t-sym_r-0.4__01M_29D_16H__09"
    feats = [np.load(f"exps/{task}/feature/total_feat_{t}.npy")[index] for t in range(epoch)]
    feats_plot = [list(x[9:]) + list(x[:9]) for x in feats]

    total_x = []
    for col in range(84):
        x = [feats_plot[x][col] for x in range(epoch)]
        x = sample(x)
        total_x.append(x)


    
    return total_x
    

def choose_type(type,label_df):
    df_type = label_df[label_df["type"] == type]
    df_index = list(df_type.index)
    return df_index



dataset_label = np.load(r"behaviour_dataset/cifar100_y_sym_0.4.npz")["arr_0"]
# dataset_label = np.load(r"behaviour_dataset/cifar100_y_adver")["arr_0"]
label_df = pd.DataFrame(dataset_label, columns=[ "label", "label_p", "lam", "type"])
list_init = choose_type(0,label_df)[:20]
list_adver = choose_type(1,label_df)[:10]
list_cutout = choose_type(2,label_df)[:10]
list_mixup = choose_type(3,label_df)[:10]
list_noise = choose_type(4,label_df)[:10]
other = list_noise + list_mixup + list_cutout +list_adver
plot_init = [x for x in list_init if x not in other]


for t in plot_init:
    total_x = plot_save(index=t,type="init")
    # 保持total_x为npy
    total_x = np.array(total_x)
    np.save(f"sample_{t}.npy",total_x)
    # 加载 total_x
    np.load(f"sample_{t}.npy")
    print(t)

    
    
# for t in list_adver:
#     plot_save(index=t,type="adversarial")
    
# for t in list_cutout:
#     plot_save(index=t,type="cutout")
    
# for t in list_mixup:
#     plot_save(index=t,type="mixup")
    
# for t in list_noise:
#     plot_save(index=t,type="noise")