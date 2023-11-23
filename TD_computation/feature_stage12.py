
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_task_name(task):
    t = task.replace("e-gpu1_m-ResNet18_224_d-","").replace("e-gpu1_m-ResNet18_d-","")
    t = t.split("__")[0]
    return t


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
        
def jaccard(x,y):
    return len(list(set(x) & set(y)))/len(set(list(x)+list(y)))


def last_epoch(task,epoch,sub,label_one):
    last_logit = np.load(f"exps/{task}/runs/logit_{epoch-sub}.npy")
    last_index = np.load(f"exps/{task}/runs/topk_{epoch-sub}.npy")
    last_sorted_id = np.argsort([x[0] for x in list(last_logit[:,:1].astype("int"))])
    last_sorted_index = np.array([last_index[x][1:10] for x in last_sorted_id]).astype("int")
    last_sorted_class = [[label_one[t] for t in x] for x in last_sorted_index]
    sorted_label =list(np.argmax(last_logit[last_sorted_id][:,1:],axis=1))
    sorted_forget = [int(sorted_label[i]==label_one[i]) for i in range(len(sorted_label))]
    return last_sorted_index,last_sorted_class,sorted_forget
        

def extrace_feature(task,epoch,topk=10):
    save_dir = os.path.join("exps", task,"feature")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir,f"total_feat_{epoch}")
    total_feature = []
    t = get_task_name(task)
    if "MNIST" in task and "FashionMNIST" not in  task and "KMNIST" not in  task:
        
        label = np.load(f"behaviour_dataset/{t}/y_merge.npz")["arr_0"]
        label_one = [int(x[0]) for x in label[:,:1]]
        num_classes = 10
    elif "FashionMNIST" in task:
        label = np.load(f"behaviour_dataset/{t}/y_merge.npz")["arr_0"]
        label_one = [int(x[0]) for x in label[:,:1]]
        num_classes = 10
    elif "KMNIST" in task:
        label = np.load(f"behaviour_dataset/{t}/y_merge.npz")["arr_0"]
        label_one = [int(x[0]) for x in label[:,:1]]
        num_classes = 10
    elif "CIFAR10" in task and "CIFAR100" not in  task:
        label = np.load(f"behaviour_dataset/{t}/y_merge.npz")["arr_0"]
        label_one = [int(x[0]) for x in label[:,:1]]
        num_classes = 10
    elif "CIFAR100" in task:
        label = np.load(f"behaviour_dataset/{t}/y_merge.npz")["arr_0"]
        label_one = [int(x[0]) for x in label[:,:1]]
        num_classes = 100
    elif "SVHN" in task: 
        label = np.load(f"behaviour_dataset/{t}/y_merge.npz")["arr_0"]
        label_one = [int(x[0]) for x in label[:,:1]]
        num_classes = 10
    elif "Flowers102" in task:
        label = np.load(f"behaviour_dataset/{t}/y_merge.npz")["arr_0"]
        label_one = [int(x[0]) for x in label[:,:1]]
        num_classes = 102
    elif "OxfordIIITPet" in task:
        label = np.load(f"behaviour_dataset/{t}/y_merge.npz")["arr_0"]
        label_one = [int(x[0]) for x in label[:,:1]]
        num_classes = 37
    elif "StanfordCars" in task:
        label = np.load(f"behaviour_dataset/{t}/y_merge.npz")["arr_0"]
        label_one = [int(x[0]) for x in label[:,:1]]
        num_classes = 196
    elif "imagenet_lt" in task:
        label = np.load(f"behaviour_dataset/{t}/y_merge.npz")["arr_0"]
        label_one = [int(x[0]) for x in label[:,:1]]
        num_classes = 1000
    elif "places-lt" in task:
        label = np.load(f"behaviour_dataset/{t}/y_merge.npz")["arr_0"]
        label_one = [int(x[0]) for x in label[:,:1]]
        num_classes = 365
    

    IE_logit = np.load(f"exps/{task}/runs/logit_{epoch}.npy")
    IE_topk = np.load(f"exps/{task}/runs/topk_{epoch}.npy")
    
    if True in np.isnan(IE_topk):
        IE_topk[np.isnan(IE_topk)] = 0.0
        
    sorted_id = np.argsort([x[0] for x in list(IE_logit[:,:1].astype("int"))])
    IE_logit_sorted = np.array([IE_logit[x][1:] for x in sorted_id])
    IE_topk_sorted = np.array([IE_topk[x] for x in sorted_id])
    # IE_uncer_data = -1*np.load(f"exps/{task}/runs/uncert_data_{epoch}.npy")
    # IE_uncer_model = -1*np.load(f"exps/{task}/runs/uncert_model_{epoch}.npy")

    IE_logit_tensor = torch.from_numpy(IE_logit_sorted).cuda()
    IE_topk_tensor = torch.from_numpy(IE_topk_sorted).cuda()
    # uncer_data_tensor = torch.from_numpy(IE_uncer_data).cuda()
    # uncer_model_tensor = torch.from_numpy(IE_uncer_model).cuda()
    
    label = label[:IE_logit_tensor.shape[0]]
    target = torch.from_numpy(label).cuda()
    target_a = target[:,:1].view(-1).long().cuda()
    target_b = target[:,1:2].view(-1).long().cuda()
    target_l = target[:,2:3].view(-1).float().cuda()
    labels_one_hot_a = F.one_hot(target_a,num_classes=num_classes).float().cuda()
    labels_one_hot_b = F.one_hot(target_b,num_classes=num_classes).float().cuda()
    logit_possible = F.softmax(IE_logit_tensor,dim=1)
    IE_topk_sorted_class = [[label_one[t] for t in x] for x in IE_topk_sorted[:,0:9].astype("int")]
    
    #遗忘状态
    sorted_label =list(np.argmax(IE_logit_sorted,axis=1))
    sorted_forget = [int(sorted_label[i]==label_one[i]) for i in range(len(sorted_label))]
    
    
    ########################################### history ######################################################       
    jaccard_index_1 = torch.from_numpy(np.array([0.0]*len(sorted_id))).cuda()
    jaccard_class_1 = torch.from_numpy(np.array([0.0]*len(sorted_id))).cuda()
    jaccard_index_2 = torch.from_numpy(np.array([0.0]*len(sorted_id))).cuda()
    jaccard_class_2 = torch.from_numpy(np.array([0.0]*len(sorted_id))).cuda()
    forget_status = torch.from_numpy(np.array([0.0]*len(sorted_id))).cuda()
    if epoch>0:
        last1_sorted_index,last1_sorted_class,last1_sorted_forget = last_epoch(task,epoch,1,label_one)
        jaccard_index_1 = [jaccard(IE_topk_sorted[:,0:9][t],last1_sorted_index[t]) for t in range(len(sorted_id))]
        jaccard_class_1 = [jaccard(last1_sorted_class[t],IE_topk_sorted_class[t]) for t in range(len(sorted_id))]
        forget_status = [last1_sorted_forget[t]*1+sorted_forget[t]*2   for t in range(len(sorted_id))]
        jaccard_index_1 = torch.from_numpy(np.array(jaccard_index_1)).cuda()
        jaccard_class_1 = torch.from_numpy(np.array(jaccard_class_1)).cuda()
        forget_status = torch.from_numpy(np.array(forget_status)).cuda()
    ## Jasscard距离  index  last2
    if epoch>1:
        last2_sorted_index,last2_sorted_class,last2_sorted_forget = last_epoch(task,epoch,2,label_one)
        jaccard_index_2 = [jaccard(IE_topk_sorted[:,0:9][t],last2_sorted_index[t]) for t in range(len(sorted_id))]
        jaccard_class_2 = [jaccard(IE_topk_sorted_class[t],last2_sorted_class[t]) for t in range(len(sorted_id))]
        forget_status = [last2_sorted_forget[t]*1+sorted_forget[t]*2   for t in range(len(sorted_id))]
        jaccard_index_2 = torch.from_numpy(np.array(jaccard_index_2)).cuda()
        jaccard_class_2 = torch.from_numpy(np.array(jaccard_class_2)).cuda()
        forget_status = torch.from_numpy(np.array(forget_status)).cuda()


    ####################################### self ############################################################

    # labels_one_hot_a = labels_one_hot_a[:IE_logit_tensor.shape[0]]
    # labels_one_hot_b = labels_one_hot_b[:IE_logit_tensor.shape[0]]
    # target_a = target_a[:IE_logit_tensor.shape[0]]
    # target_b = target_b[:IE_logit_tensor.shape[0]]
    # target_l = target_l[:IE_logit_tensor.shape[0]]
    # feature 0 logit_value
    logit_left = target_l*torch.sum(IE_logit_tensor * labels_one_hot_a,dim=1)
    logit_right = (1 - target_l)*torch.sum(IE_logit_tensor * labels_one_hot_b,dim=1)
    self_logit_value = logit_left + logit_right
    # feature 1 possible
    possible_left = target_l*torch.sum(logit_possible * labels_one_hot_a,dim=1)
    possible_right = (1 - target_l)*torch.sum(logit_possible * labels_one_hot_b,dim=1)
    self_possible = possible_left + possible_right
    # feature 2 loss
    loss_left = target_l.unsqueeze(-1)*F.cross_entropy(IE_logit_tensor, target_a,reduction='none').unsqueeze(-1)
    loss_right = (1 - target_l.unsqueeze(-1)) * F.cross_entropy(IE_logit_tensor, target_b,reduction='none').unsqueeze(-1)
    self_loss = (loss_left+loss_right).squeeze(-1)
    # # feature 3 grad
    self_grad = 1- logit_possible
    self_grad_norm = torch.norm(self_grad,dim=1)    
    # feature 4 margin
    others_max =(logit_possible[labels_one_hot_a!=1].reshape(logit_possible.size(0),-1)).max(dim=1).values
    self_margin =  self_possible - others_max
    # feature 5 entropy
    self_entropy = -1*torch.sum(F.softmax(IE_logit_tensor,dim=1)*F.log_softmax(IE_logit_tensor,dim=1),dim=1)
    # feature 6 uncer_model
    # self_uncer_model = torch.mean(uncer_model_tensor,dim=1)
    # feature 7 uncer_data
    # self_uncer_data = torch.mean(uncer_data_tensor,dim=1)
    # feature 8 是否正确
    self_flag = (torch.argmax(IE_logit_tensor,dim=1) == target_a)*1.0
    IE_possible_tensor = F.softmax(IE_logit_tensor,dim=1)

    
    ############################################## self.class #####################################################    

    class_logit_value = torch.cat([torch.mean(self_logit_value[target_a==x]).unsqueeze(-1) for x in range(num_classes)])
    class_possible = torch.cat([torch.mean(self_possible[target_a==x]).unsqueeze(-1) for x in range(num_classes)])
    class_loss= torch.cat([torch.mean(self_loss[target_a==x]).unsqueeze(-1) for x in range(num_classes)])
    class_grad =  [torch.mean(self_grad[target_a==x],dim=0) for x in range(num_classes)]
    class_grad_norm= torch.cat([torch.mean(self_grad_norm[target_a==x]).unsqueeze(-1) for x in range(num_classes)])
    class_margin= torch.cat([torch.mean(self_margin[target_a==x]).unsqueeze(-1) for x in range(num_classes)])
    class_entropy= torch.cat([torch.mean(self_entropy[target_a==x]).unsqueeze(-1) for x in range(num_classes)])
    # class_uncer_model= torch.cat([torch.mean(self_uncer_model[target_a==x]).unsqueeze(-1) for x in range(num_classes)])
    # class_uncer_data= torch.cat([torch.mean(self_uncer_data[target_a==x]).unsqueeze(-1) for x in range(num_classes)])
    class_flag= torch.cat([torch.mean(self_flag[target_a==x]).unsqueeze(-1) for x in range(num_classes)])

    ############################################### self.total ####################################################

    total_logit_value = torch.mean(class_logit_value)
    total_possible= torch.mean(class_possible)
    total_loss= torch.mean(class_loss)
    total_grad = torch.mean(torch.cat(class_grad).view(num_classes,num_classes),dim=0)
    total_grad_norm= sum(class_grad_norm)
    total_margin= torch.mean(class_margin)
    total_entropy= torch.mean(class_entropy)
    # total_uncer_model= torch.mean(class_uncer_model)
    # total_uncer_data= torch.mean(class_uncer_data)
    total_flag = torch.mean(self_flag)

    ################################################## self.neighborhood ###########################################

    self_neighb_logit_value = torch.mean(torch.cat([self_logit_value[IE_topk_tensor[:,x:x+1].long()] for x in range(1,topk)],dim=1),dim=1)
    self_neighb_possible= torch.mean(torch.cat([self_possible[IE_topk_tensor[:,x:x+1].long()] for x in range(1,topk)],dim=1),dim=1)
    self_neighb_loss= torch.mean(torch.cat([self_loss[IE_topk_tensor[:,x:x+1].long()] for x in range(1,topk)],dim=1),dim=1)
    self_neighb_grad_norm= torch.mean(torch.cat([self_grad_norm[IE_topk_tensor[:,x:x+1].long()] for x in range(1,topk)],dim=1),dim=1)
    self_neighb_margin= torch.mean(torch.cat([self_margin[IE_topk_tensor[:,x:x+1].long()] for x in range(1,topk)],dim=1),dim=1)
    self_neighb_entropy= torch.mean(torch.cat([self_entropy[IE_topk_tensor[:,x:x+1].long()] for x in range(1,topk)],dim=1),dim=1)
    # self_neighb_uncer_model= torch.mean(torch.cat([self_uncer_model[IE_topk_tensor[:,x:x+1].long()] for x in range(1,topk)],dim=1),dim=1)
    # self_neighb_uncer_data= torch.mean(torch.cat([self_uncer_data[IE_topk_tensor[:,x:x+1].long()] for x in range(1,topk)],dim=1),dim=1)


    ################################################## feature1:  rate  ###########################################

    # feature 0 logit_value
    feat1 = self_logit_value/class_logit_value[target_a]
    feat2 = self_logit_value/self_neighb_logit_value
    feat3 = self_neighb_logit_value/class_logit_value[target_a]
    feat4 = self_neighb_logit_value/total_logit_value
    feat5 = self_logit_value/total_logit_value
    feat6 = class_logit_value[target_a]/total_logit_value
    feat7 = self_logit_value.sort().indices.sort().indices/len(self_logit_value)

    # feature 1 possible
    feat8 = self_possible/class_possible[target_a]
    feat9 = self_possible/self_neighb_possible
    feat10 = self_neighb_possible/class_possible[target_a]
    feat11 = self_neighb_possible/total_possible
    feat12 = self_possible/total_possible
    feat13 = class_possible[target_a]/total_possible
    feat14 = self_possible.sort().indices.sort().indices/len(self_possible)

    # feature 2 loss
    feat15 = self_loss/class_loss[target_a]
    feat16 = self_loss/self_neighb_loss
    feat17 = self_neighb_loss/class_loss[target_a]
    feat18 = self_neighb_loss/total_loss
    feat19 = self_loss/total_loss
    feat20 = class_loss[target_a]/total_loss
    feat21 = self_loss.sort().indices.sort().indices/len(self_loss)

    # feature 3 grad
    feat22 = self_grad_norm/class_grad_norm[target_a]
    feat23 = self_grad_norm/self_neighb_grad_norm
    feat24 = self_neighb_grad_norm/class_grad_norm[target_a]
    feat25 = self_neighb_grad_norm/total_grad_norm
    feat26 = self_grad_norm/total_grad_norm
    feat27 = class_grad_norm[target_a]/total_grad_norm
    feat28= self_grad_norm.sort().indices.sort().indices/len(self_grad_norm)


    # feature 4 margin
    feat29 = self_margin/class_margin[target_a]
    feat30 = self_margin/self_neighb_margin
    feat31 = self_neighb_margin/class_margin[target_a]
    feat32 = self_neighb_margin/total_margin
    feat33 = self_margin/total_margin
    feat34 = class_margin[target_a]/total_margin
    feat35 = self_margin.sort().indices.sort().indices/len(self_margin)

    # feature 5 entropy
    feat36 = self_entropy/class_entropy[target_a]
    feat37 = self_entropy/self_neighb_entropy
    feat38 = self_neighb_entropy/class_entropy[target_a]
    feat39 = self_neighb_entropy/total_entropy
    feat40 = self_entropy/total_entropy
    feat41 = class_entropy[target_a]/total_entropy
    feat42 = self_entropy.sort().indices.sort().indices/len(self_entropy)

    # feature 6 uncer_model
    # feat43 = self_uncer_model/class_uncer_model[target_a]
    # feat44 = self_uncer_model/self_neighb_uncer_model
    # feat45 = self_neighb_uncer_model/class_uncer_model[target_a]
    # feat46 = self_neighb_uncer_model/total_uncer_model
    # feat47 = self_uncer_model/total_uncer_model
    # feat48 = class_uncer_model[target_a]/total_uncer_model
    # feat49 = self_uncer_model.sort().indices.sort().indices/len(self_uncer_model)

    # # feature 7 uncer_data
    # feat50 = self_uncer_data/class_uncer_data[target_a]
    # feat51 = self_uncer_data/self_neighb_uncer_data
    # feat52 = self_neighb_uncer_data/class_uncer_data[target_a]
    # feat53 = self_neighb_uncer_data/total_uncer_data
    # feat54 = self_uncer_data/total_uncer_data
    # feat55 = class_uncer_data[target_a]/total_uncer_data
    # feat56 = self_uncer_data.sort().indices.sort().indices/len(self_uncer_data)

    ############################################## feature2: KL散度 #####################################################

    # 邻域平均KL距离
    neighb_possible_center = [torch.mean(IE_possible_tensor[x],dim=0) for x in IE_topk_tensor[:,:9].long()]  # 每个样本类别
    ###XXXXXX
    neighb_possible_kl = torch.cat([F.kl_div(neighb_possible_center[t],IE_possible_tensor[t]).unsqueeze(0) for t in range(len(IE_possible_tensor))])
    feat57 = neighb_possible_kl
    # 类别平均KL距离
    class_possible_center = [torch.mean(IE_possible_tensor[target_a==x],dim=0) for x in range(num_classes)]  # 每个样本类别
    ###XXXXXX
    class_possible_kl = torch.cat([F.kl_div(class_possible_center[target_a[t]],IE_possible_tensor[t]).unsqueeze(0) for t in range(len(IE_possible_tensor))])
    feat58 = class_possible_kl
    # 全局平均KL距离
    # 全局一致性
    total_possible_center = torch.mean(IE_possible_tensor,dim=0)
    total_possible_kl = torch.cat([F.kl_div(total_possible_center,IE_possible_tensor[t]).unsqueeze(0) for t in range(len(IE_possible_tensor))])
    feat59 = total_possible_kl
    # 全局百分位
    feat60 = total_possible_kl.sort().indices.sort().indices/len(total_possible_kl)
    # 邻域-类别KL距离
    feat61 = neighb_possible_kl/class_possible_kl
    # 类别-全局KL距离
    feat62 = class_possible_kl/total_possible_kl
    # 邻域-全局KL距离
    feat63 = neighb_possible_kl/total_possible_kl


    ############################################## feature3: 余弦距离 #####################################################

    # 邻域平均距离
    self_neighb_dis= torch.mean(IE_topk_tensor[:,10:],dim=1)
    feat64 = self_neighb_dis
    # 类别平均距离
    class_neighb_dis = torch.cat([torch.mean(self_neighb_dis[target_a==x]).unsqueeze(-1) for x in range(num_classes)])
    feat65 = self_neighb_dis/class_neighb_dis[target_a]
    # 全局平均距离
    total_neighb_dis = torch.mean(self_neighb_dis)
    feat66 = self_neighb_dis/total_neighb_dis
    # 全局百分位
    feat67 = feat66.sort().indices.sort().indices/len(feat66)
    # 邻域-类别距离
    feat68 = self_neighb_dis/class_neighb_dis[target_a]
    # 类别-全局距离
    feat69 = class_neighb_dis[target_a]/total_neighb_dis
    # 邻域-全局距离
    feat70 = self_neighb_dis/total_neighb_dis


    ############################################# feature4: grad一致性 ####################################################################

    # 邻域一致性
    cos_model = CosineSimilarity()
    neighb_grad= [self_grad[x] for x in IE_topk_tensor[:,:3].long()]  # 每个样本邻域
    neighb_grad_cor = torch.cat([torch.mean(cos_model(self_grad[t].unsqueeze(0),neighb_grad[t]),dim=1) for t in range(len(self_grad))])
    feat71 = neighb_grad_cor
    # neighb_grad_center = [torch.mean(self_grad[x],dim=0) for x in IE_topk_tensor[:,:9].long()]  # 每个样本邻域
    # neighb_grad_cor = torch.cat([F.cosine_similarity(self_grad[t],neighb_grad_center[t],dim=0).unsqueeze(0) for t in range(len(self_grad))])
    # 类别一致性
    class_grad_center = [torch.mean(self_grad[target_a==x],dim=0) for x in range(num_classes)]  # 每个样本类别
    class_grad_cor = torch.cat([F.cosine_similarity(self_grad[t],class_grad_center[target_a[t]],dim=0).unsqueeze(0) for t in range(len(self_grad))])
    feat72 = class_grad_cor
    # 全局一致性
    total_grad_center = torch.mean(self_grad,dim=0)
    total_grad_cor = torch.cat([F.cosine_similarity(self_grad[t],total_grad_center,dim=0).unsqueeze(0) for t in range(len(self_grad))])
    feat73 = class_grad_cor

    # 邻域-类别一致性
    feat74 = neighb_grad_cor/class_grad_cor
    # 邻域-全局一致性
    feat75 = neighb_grad_cor/total_grad_cor
    # 类别-全局一致性
    feat76 = class_grad_cor/total_grad_cor
    ############################################# feature5: 补充特征 ####################################################################
    #类别精度与全局精度比值			
    acc_class_total = class_flag[target_a]/total_flag
    
    ############################################# 特征拼接 ##############################################################################
    total_feature.append(feat1)
    total_feature.append(feat2)
    total_feature.append(feat3)
    total_feature.append(feat4)
    total_feature.append(feat5)
    total_feature.append(feat6)
    total_feature.append(feat7)
    total_feature.append(feat8)
    total_feature.append(feat9)
    total_feature.append(feat10)
    
    total_feature.append(feat11)
    total_feature.append(feat12)
    total_feature.append(feat13)
    total_feature.append(feat14)
    total_feature.append(feat15)
    total_feature.append(feat16)
    total_feature.append(feat17)
    total_feature.append(feat18)
    total_feature.append(feat19)
    total_feature.append(feat20)

    total_feature.append(feat21)
    total_feature.append(feat22)
    total_feature.append(feat23)
    total_feature.append(feat24)
    total_feature.append(feat25)
    total_feature.append(feat26)
    total_feature.append(feat27)
    total_feature.append(feat28)
    total_feature.append(feat29)
    total_feature.append(feat30)
    
    total_feature.append(feat31)
    total_feature.append(feat32)
    total_feature.append(feat33)
    total_feature.append(feat34)
    total_feature.append(feat35)
    total_feature.append(feat36)
    total_feature.append(feat37)
    total_feature.append(feat38)
    total_feature.append(feat39)
    total_feature.append(feat40)
    
    total_feature.append(feat41)
    total_feature.append(feat42)
    # total_feature.append(feat43)
    # total_feature.append(feat44)
    # total_feature.append(feat45)
    # total_feature.append(feat46)
    # total_feature.append(feat47)
    # total_feature.append(feat48)
    # total_feature.append(feat49)
    # total_feature.append(feat50)
    
    # total_feature.append(feat51)
    # total_feature.append(feat52)
    # total_feature.append(feat53)
    # total_feature.append(feat54)
    # total_feature.append(feat55)
    # total_feature.append(feat56)
    total_feature.append(feat57)
    total_feature.append(feat58)
    total_feature.append(feat59)
    total_feature.append(feat60)
    
    total_feature.append(feat61)
    total_feature.append(feat62)
    total_feature.append(feat63)
    total_feature.append(feat64)
    total_feature.append(feat65)
    total_feature.append(feat66)
    total_feature.append(feat67)
    total_feature.append(feat68)
    total_feature.append(feat69)
    total_feature.append(feat70)
    
    total_feature.append(feat71)
    total_feature.append(feat72)
    total_feature.append(feat73)
    total_feature.append(feat74)
    total_feature.append(feat75)
    total_feature.append(feat76)
    
    #标签
    total_feature.append(self_possible)
    total_feature.append(self_entropy)
    # total_feature.append(self_uncer_data)
    # total_feature.append(self_uncer_model)
    total_feature.append(self_margin)
    
    total_feature.append(self_loss)
    total_feature.append(self_grad_norm)
    total_feature.append(self_logit_value)
    
    #以上84标签可以和历史对比
    total_feature.append(self_flag)
    total_feature.append(acc_class_total)
    total_feature.append(jaccard_index_1)
    total_feature.append(jaccard_index_2)
    total_feature.append(jaccard_class_1)
    total_feature.append(jaccard_class_2)
    total_feature.append(forget_status)
    # total_feature.extend([eval(f"feat{x}") for x in range(1,78)])
    total_feature_reshape = torch.cat(total_feature).view(75,-1).T
    total_feature_reshape = total_feature_reshape.cpu().numpy()
    np.save(save_path,total_feature_reshape)



if __name__ == "__main__":
    
    for task in os.listdir("exps"):
        # if  "cifar" in task.lower(): continue
        # if  "mnist" in task.lower(): continue
        if  "__01" not in task.lower(): continue
        for epoch in range(28,120):
            print(f"current task : {task} current epoch: {epoch}")
            extrace_feature(task,epoch)