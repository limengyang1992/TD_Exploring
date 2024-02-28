
import torch
import torch.nn as nn
import torch.nn.functional as F

from seq2vec import TS2Vec


class Attention(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.tanh = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x,mask):
        H0 = self.tanh(x)
        H1 = torch.matmul(H0, self.w.unsqueeze(1))
        H1 = H1.squeeze(2)
        H1 = H1.masked_fill(mask == 0, -1e6)
        H2 = nn.functional.softmax(H1, dim=1)
        alpha = H2.unsqueeze(2)
        att_hidden = torch.sum(x * alpha, 1)
        return att_hidden,H2


class TD2VEC(nn.Module):
    def __init__(self,ts2vec_pt=None,output_dims=100,class_num=5):
        super(TD2VEC, self).__init__()
        self.ts2vec = TS2Vec(output_dims=output_dims)
        # if ts2vec_pt is not None:
        #     self.ts2vec.load(ts2vec_pt)
        self.encoder = self.ts2vec._net
        self.attention = Attention(output_dims)
        self.fc = nn.Linear(output_dims, class_num)

    def forward(self, x):
        mask = (x[:,:,0] != 0)
        x = self.encoder(x)
        attn,score = self.attention(x,mask)
        out = self.fc(attn)
        return out,score,attn


class TS2VEC(nn.Module):
    def __init__(self,ts2vec_pt=None, hidden_dim=100,class_num=5):
        super(TS2VEC, self).__init__()
        self.ts2vec = TS2Vec()
        if ts2vec_pt is not None:
            self.ts2vec.load(ts2vec_pt)
        self.encoder = self.ts2vec._net

    def forward(self, x):
        x = self.encoder(x)
        return x
    
    

if __name__ == '__main__':
    net = TD2VEC('ckpt/92.pt').cuda()
    x = torch.randn(64, 120, 211).cuda()
    y = net(x)