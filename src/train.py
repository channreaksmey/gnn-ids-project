import os, sys, pickle, glob
sys.path.insert(0,'.')
import torch, torch.nn as nn
import dgl
from src.data import load_graphs
from src.models import FlashGNN, MagicGNN, NodeHead

def main():
    graphs=load_graphs('data')
    # graph-level labels: train graphs benign(0), test graphs malicious(1)
    labels=[(g.ndata['type']==4).float() for _,g in graphs]
    device='cpu'
    for name,Model in [('FLASH',FlashGNN),('MAGIC',MagicGNN)]:
        enc=Model(5,128,3); node_head=NodeHead(128)
        opt=torch.optim.Adam(list(enc.parameters())+list(node_head.parameters()),lr=1e-3)
        for epoch in range(2):
            for (n,g),y in zip(graphs,labels):
                x=g.ndata['attr']
                h=enc(g,x)
                node_score=node_head(h).squeeze(-1)
                loss=nn.BCEWithLogitsLoss()(node_score, y)
                opt.zero_grad(); loss.backward(); opt.step()
            print(name,'epoch',epoch,'loss',float(loss))
        torch.save({'enc':enc.state_dict(),'node_head':node_head.state_dict()}, f'out/{name}.pt')
        print('saved',name)

if __name__=='__main__':
    os.makedirs('out',exist_ok=True)
    main()
