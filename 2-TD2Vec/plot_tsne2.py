from sklearn.manifold import TSNE
from sklearn.datasets import load_iris,load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import numpy as np
import random
# plt.rcParams['font.sans-serif'] = ['Times New Roman']


digits = np.load("analysis_test/feat.npy")
target = [0]*5000 + [1]*5000 +[2]*5000 +[3]*5000 +[4]*5000 



for i in range(10):

    choose = random.sample(range(25000),3500)
    choose_t = [target[x] for x in choose]
    choose_d = digits[choose]

    X_tsne = TSNE(n_components=2,perplexity=10).fit_transform(choose_d)
    ckpt_dir="images"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    plt.figure(figsize=(10, 10))
    plt.subplot()
    plt.scatter([X_tsne[:, 0][i] for i,x in enumerate(choose_t) if x==0], [X_tsne[:, 1][i] for i,x in enumerate(choose_t) if x==0],label="BC")
    plt.scatter([X_tsne[:, 0][i] for i,x in enumerate(choose_t) if x==1], [X_tsne[:, 1][i] for i,x in enumerate(choose_t) if x==1],label="NY")
    plt.scatter([X_tsne[:, 0][i] for i,x in enumerate(choose_t) if x==2], [X_tsne[:, 1][i] for i,x in enumerate(choose_t) if x==2],label="HC")
    plt.scatter([X_tsne[:, 0][i] for i,x in enumerate(choose_t) if x==3], [X_tsne[:, 1][i] for i,x in enumerate(choose_t) if x==3],label="TC")
    plt.scatter([X_tsne[:, 0][i] for i,x in enumerate(choose_t) if x==4], [X_tsne[:, 1][i] for i,x in enumerate(choose_t) if x==4],label="AD")

    plt.yticks(size = 14)
    plt.xticks(size = 14)

    plt.legend(title='',loc = 0,prop = {'size':16})
    plt.savefig(f'images/iccv_tsne-pca-{i}.png', dpi=120)


