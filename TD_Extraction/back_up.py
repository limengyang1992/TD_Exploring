
import numpy as np
import os
import glob

epoch = int(120*0.1)
flag = -9   #grad(大到小)
# flag = -10  #loss(大到小)
# flag = -11  #margin(小到大)



dirs_label = {}
for d in os.listdir("behaviour_dataset"):
    if "KMNIST_" not in d: continue
    dirs_label[d] = np.load(f"behaviour_dataset/{d}/y_merge.npz")["arr_0"]
    current = None
    for t in os.listdir("/home/kunyu/exps/sequence_wise_task/"):
        if d in t:
            current = t
            break
    print(f"+++++++++++++++++++++++++++++++{current}")
    labels = []
    samples_loss_1,samples_grad_1,samples_margin_1= [],[],[]
    samples_loss_2,samples_grad_2,samples_margin_2= [],[],[]
    samples_loss_3,samples_grad_3,samples_margin_3= [],[],[]
    
    for index in range(50000):
        sample = f"/home/kunyu/exps/sequence_wise_task/{current}/sequence_{index}.npy"
        task = os.path.split(sample)[0].split("_d-")[1].split("__")[0]
        label = dirs_label[f"{task}"][index][-1]
        labels.append(label)
        data  = np.load(sample)
        samples_loss_1.append(data[7][-10])
        samples_grad_1.append(data[7][-9])
        samples_margin_1.append(data[7][-11])
        
        samples_loss_2.append(data[10][-10])
        samples_grad_2.append(data[10][-9])
        samples_margin_2.append(data[10][-11])
        

    labels = [1 if x==4 else 0 for x in labels]
    number = int(float(sample.split("__")[-3].split("_")[-1])*50000)
    rate_loss_1 = sum([labels[x] for x in np.argsort(samples_loss_1)[::-1]][:number])/number
    rate_margin_1 = sum([labels[x] for x in np.argsort(samples_margin_1)][:number])/number
    rate_grad_1 = sum([labels[x] for x in np.argsort(samples_grad_1)][:number])/number
    print(rate_loss_1,rate_margin_1,rate_grad_1)
    
    
    rate_loss_2= sum([labels[x] for x in np.argsort(samples_loss_2)[::-1]][:number])/number
    rate_margin_2 = sum([labels[x] for x in np.argsort(samples_margin_2)][:number])/number
    rate_grad_2 = sum([labels[x] for x in np.argsort(samples_grad_2)][:number])/number
    print(rate_loss_2,rate_margin_2,rate_grad_2)
    
    

    
    # 去除epoch的loss等、
    # 排序指标，取前多少个，定标签
    # 计算精度f1