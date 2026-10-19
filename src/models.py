import torch, torch.nn as nn, torch.nn.functional as F
import dgl

class FlashGNN(nn.Module):
    def __init__(self, in_dim, hid=128, layers=3):
        super().__init__()
        self.layers=nn.ModuleList()
        self.layers.append(nn.Linear(in_dim,hid))
        for _ in range(layers-1):
            self.layers.append(nn.Linear(hid,hid))
    def forward(self,g,x):
        h=x
        for i,lin in enumerate(self.layers):
            h=F.relu(lin(h))
            g.ndata['h']=h
            g.update_all(dgl.function.copy_u('h','m'),dgl.function.mean('m','h'))
            h=g.ndata['h']
        return h

class MagicGNN(nn.Module):
    def __init__(self, in_dim, hid=128, layers=3):
        super().__init__()
        self.layers=nn.ModuleList()
        self.layers.append(nn.Linear(in_dim,hid))
        for _ in range(layers-1):
            self.layers.append(nn.Linear(hid,hid))
        self.att=nn.Linear(hid,1)
    def forward(self,g,x):
        h=x
        for lin in self.layers:
            h=F.relu(lin(h))
            g.ndata['h']=h
            g.ndata['a']=self.att(h)
            g.update_all(dgl.function.u_mul_v('h','a','m'),dgl.function.mean('m','h'))
            h=g.ndata['h']
        return h

class NodeHead(nn.Module):
    def __init__(self,hid=128):
        super().__init__()
        self.lin=nn.Linear(hid,1)
    def forward(self,h):
        return self.lin(h).squeeze(-1)
