import os, sys, pickle, glob
sys.path.insert(0,'.')
import torch, torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from src.data import load_graphs
from src.models import FlashGNN, MagicGNN

def main():
    graphs=load_graphs('data')
    test=[(n,g) for n,g in graphs if 'test' in n]
    for name,Model in [('FLASH',FlashGNN),('MAGIC',MagicGNN)]:
        ckpt=torch.load(f'out/{name}.pt')
        enc=Model(5,128,3); enc.load_state_dict(ckpt['enc'])
        from src.models import NodeHead
        head=NodeHead(128); head.load_state_dict(ckpt['node_head'])
        enc.eval(); head.eval()
        for n,g in test:
            with torch.no_grad():
                h=enc(g,g.ndata['attr'])
                scores=torch.sigmoid(head(h).squeeze(-1))
            y=(g.ndata['type']==4).int()
            pred=(scores>0.5).int()
            p,r,f,_=precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
            try:
                auc=roc_auc_score(y,scores)
            except Exception:
                auc=float('nan')
            print(name, 'P',p,'R',r,'F1',f,'AUC',auc)

if __name__=='__main__':
    main()
