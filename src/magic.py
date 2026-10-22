import torch, torch.nn as nn, torch.nn.functional as F
import dgl

class MagicEncoder(nn.Module):
    def __init__(self,in_dim=5,hid=128,layers=3,heads=4):
        super().__init__()
        self.layers=nn.ModuleList([nn.Linear(in_dim,hid)])
        for _ in range(layers-1):
            self.layers.append(nn.Linear(hid,hid))
        self.attn=nn.Linear(hid,1)
    def forward(self,g,x):
        h=x
        for lin in self.layers:
            h=F.relu(lin(h))
            g.ndata['h']=h
            g.ndata['a']=self.attn(h)
            g.update_all(dgl.function.u_mul_v('h','a','m'),dgl.function.mean('m','h'))
            h=g.ndata['h']
        return h
