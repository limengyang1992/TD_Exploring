
import os
import glob 
import time
import torch
import numpy as np

class CosineSimilarityTest(torch.nn.Module):
    def __init__(self):
        super(CosineSimilarityTest, self).__init__()

    def forward(self, x1, x2):
        x2 = x2.t()
        x = x1.mm(x2)
    
        x1_frobenius = x1.norm(dim=1).unsqueeze(0).t()
        x2_frobenins = x2.norm(dim=0).unsqueeze(0)
        x_frobenins = x1_frobenius.mm(x2_frobenins)
    
        final = x.mul(1/x_frobenins)
        return final


def sim_epoch(data,batch,device):
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
                                  10, 
                                  dim=1, 
                                  largest=True, 
                                  sorted=True)
        value_list.append(value)
        indec_list.append(indec)
    
    numpy_value = torch.cat(value_list).cpu().detach().numpy()
    numpy_indec = torch.cat(indec_list).cpu().detach().numpy()
    numpy_indec_new = np.vectorize(ind_dict.get)(numpy_indec)
    total = np.concatenate([numpy_indec_new,numpy_value],axis=1)
    
    return total
    
if __name__ == "__main__":

    total = glob.glob(r"exps/e-gpu0_m-resnet20_d-cifar10__11M_19D_09H__14/runs/feat_*.npy")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CosineSimilarityTest().to(device)
    
    results = np.zeros((200,50000,20))
    for i,path in enumerate(total):
        # if i>10:continue
        start = time.time()
        ids = os.path.split(path)[1].split("_")[1].split(".")[0]
        data  = np.load(path)
        total = sim_epoch(data,5000,device)
        results[int(ids),:,:] = total
        print(i,time.time()-start)
    np.savez_compressed(f"out/totalz",results)
        
        