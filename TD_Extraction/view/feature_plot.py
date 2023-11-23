

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(14.4,25.6)
plt.subplots_adjust(wspace = 0, hspace =1)


def plot_save(index,type,epoch=200):
    task = "e-gpu0_m-resnet20_d-cifar10__12M_02D_22H__43"
    feats = [np.load(f"exps/{task}/feat/total_feat_{t}.npy")[index] for t in range(epoch)]
    feats_plot = [list(x[9:]) + list(x[:9]) for x in feats]

    for col in range(84):
        x = [feats_plot[x][col] for x in range(epoch)]
        y = range(epoch)
        plt.subplot(12, 7, col+1)
        plt.plot(y,x)
        left,right = "",""
        if col//7==0:
            left = "logit"
        elif col//7==1:
            left = "pos"
        elif col//7==2:
            left = "loss"
        elif col//7==3:
            left = "grad"
        elif col//7==4:
            left = "margin"
        elif col//7==5:
            left = "entropy"
        elif col//7==6:
            left = "uncerm"
        elif col//7==7:
            left = "uncerd"
        elif col//7==8:
            left = "KL"
        elif col//7==9:
            left = "cos"        
        elif col//7==10:
            left = "cor"                
            
        if col%7==0 and col//7<11:
            right = "-cla"
        elif col%7==1 and col//7<11:
            right = "-nei"
        elif col%7==2 and col//7<11:
            right = "-nei-cla"
        elif col%7==3 and col//7<11:
            right = "-nei-tot"
        elif col%7==4 and col//7<11:
            right = "-tot"
        elif col%7==5 and col//7<11:
            right = "-cla-tot"
        elif col%7==6 and col//7<11:
            right = "-%"
        
        res = left+right
        if col==76:
            res = "loss"

        if col==77:
            res = "grad_norm"
        if col==78:
            res = "logit_v"
            
        if col==79:
            res = "possible"
            
        if col==80:
            res = "entropy"
            
        if col==81:
            res = "self_uncer_data"
            
        if col==82:
            res = "self_uncer_model"
            
        if col==83:
            res = "self_margin"
            
        plt.title(res)

    plt.suptitle("resnet-32 total feature")
    plt.savefig(f"{type}_{index}.png")
    plt.clf()
    

def choose_type(type,label_df):
    df_type = label_df[label_df["type"] == type]
    df_index = list(df_type.index)
    return df_index



dataset_label = np.load(r"exps/cifar10_y_merge.npz")["arr_0"]
label_df = pd.DataFrame(dataset_label, columns=[ "label", "label_p", "lam", "type"])
list_init = choose_type(0,label_df)[:20]
list_adver = choose_type(1,label_df)[:10]
list_cutout = choose_type(2,label_df)[:10]
list_mixup = choose_type(3,label_df)[:10]
list_noise = choose_type(4,label_df)[:10]
other = list_noise + list_mixup + list_cutout +list_adver
plot_init = [x for x in list_init if x not in other][:10]


for t in plot_init:
    plot_save(index=t,type="init")
    
for t in list_adver:
    plot_save(index=t,type="adversarial")
    
for t in list_cutout:
    plot_save(index=t,type="cutout")
    
for t in list_mixup:
    plot_save(index=t,type="mixup")
    
for t in list_noise:
    plot_save(index=t,type="noise")