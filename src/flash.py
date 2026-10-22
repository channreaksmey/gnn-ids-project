import torch, torch.nn as nn, torch.nn.functional as F
import dgl

class FlashEncoder(nn.Module):
    def __init__(self,in_dim=5,hid=128,layers=3):
        super().__init__()
        self.layers=nn.ModuleList([nn.Linear(in_dim,hid)])
        for _ in range(layers-1):
            self.layers.append(nn.Linear(hid,hid))
    def forward(self,g,x):
        h=x
        for lin in self.layers:
            h=F.relu(lin(h))
            g.ndata['h']=h
            g.update_all(dgl.function.copy_u('h','m'),dgl.function.mean('m','h'))
            h=g.ndata['h']
        return h
