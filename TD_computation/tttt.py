# from lib.utils.similarity import compute_feat_similarity
# from glob import glob
# import os
# feat_root = f"exps-cloth/e-gpu1_m-ResNet18_224_d-clothing_noise__03M_01D_14H__61/runs"
# feat_paths = glob(os.path.join(feat_root, "logit_*.npy"))
# # device = model.device
# compute_feat_similarity(feat_paths)
import torch
import torchvision


net = torchvision.models.densenet.densenet121()
net = torchvision.models.vgg16()
x = torch.randn(2, 3, 224, 224)
y = net(x)
print(y.size())
