
import os
import glob 
import time
import torch
import numpy as np

class CosineSimilarity(torch.nn.Module):
    def __init__(self):
        super(CosineSimilarity, self).__init__()

    def forward(self, x1, x2):
        x2 = x2.t()
        x = x1.mm(x2)
    
        x1_frobenius = x1.norm(dim=1).unsqueeze(0).t()
        x2_frobenins = x2.norm(dim=0).unsqueeze(0)
        x_frobenins = x1_frobenius.mm(x2_frobenins)
    
        final = x.mul(1/x_frobenins)
        return final


def sim_epoch(model,data,topk,batch,device):
    value_list,indec_list = [],[]
    index,feat= data[:,:1],data[:,1:]
    x1 = torch.from_numpy(feat).to(device)
    ind = [int(x[0]) for x in list(index)]
    ind_dict = dict(zip(range(len(ind)),ind))
    foo_number = int(1+len(index)/batch)
    for i in range(foo_number):
        start = i*batch
        end = min((i+1)*batch,len(index))
        final_value = model(x1[start:end], x1)
        value, indec = torch.topk(final_value, 
                                  topk, 
                                  dim=1, 
                                  largest=True, 
                                  sorted=True)
        value_list.append(value)
        indec_list.append(indec)
    
    numpy_value = torch.cat(value_list).cpu().detach().numpy()
    numpy_indec = torch.cat(indec_list).cpu().detach().numpy()
    total = np.concatenate([numpy_indec,numpy_value],axis=1)
    return total

device = torch.device("cuda:0")
model = CosineSimilarity().to(device)

def compute_feat_similarity(feat_paths,topk=10,batch=1000,device=torch.device("cuda")):
    for i,path in enumerate(feat_paths):
        save_path = path.replace("logit_","topk_")
        print(f"total epoch: {len(feat_paths)}, current eopch: {i}")
        ids = os.path.split(path)[1].split("_")[1].split(".")[0]
        data  = np.load(path)
        total = sim_epoch(model,data,topk,batch,device)
        np.save(save_path,total)
        
         
if __name__ == "__main__":
    import os
    import glob
    feat_paths = glob.glob(os.path.join("exps/e-gpu1_m-ResNet18_d-CIFAR10__02M_17D_17H__70/runs", "logit_*.npy"))
    compute_feat_similarity(feat_paths)
    
    # feat_paths = glob.glob(os.path.join("exps/e-convnext_tiny_m-convnext_tiny_d-cifar100__12M_02D_16H__42/runs","feat_*.npy"))
    # save_path = "out"
    # compute_feat_similarity(feat_paths,topk=2)


        
        