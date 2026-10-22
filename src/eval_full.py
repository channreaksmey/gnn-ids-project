import sys; sys.path.insert(0,'.')
import torch, torch.nn as nn
from sklearn.metrics import (precision_recall_fscore_support, roc_auc_score,
                             average_precision_score, confusion_matrix)
from src.data_v2 import load_graphs, get_node_labels, get_features
from src.flash_full import FlashFull
from src.magic_full import MagicFull

def evaluate(ModelClass, name):
    graphs=load_graphs('data')
    test=[(n,g) for n,g in graphs if 'test' in n]
    enc=ModelClass(); head=nn.Linear(128,1)
    try:
        ck=torch.load(f'out/{name}_full.pt')
        enc.load_state_dict(ck['enc']); head.load_state_dict(ck['head'])
    except:
        pass
    enc.eval(); head.eval()
    for n,g in test:
        with torch.no_grad():
            h=enc(g,get_features(g))
            s=torch.sigmoid(head(h).squeeze(-1))
        y=get_node_labels(g).int()
        pred=(s>0.5).int()
        p,r,f,_=precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
        auc=roc_auc_score(y,s) if y.sum()>0 else float('nan')
        pr=average_precision_score(y,s) if y.sum()>0 else float('nan')
        tn,fp,fn,tp=confusion_matrix(y,pred).ravel() if len(confusion_matrix(y,pred).ravel())==4 else (0,0,0,0)
        print(f'{name}: P={p:.4f} R={r:.4f} F1={f:.4f} AUC={auc:.4f} PR={pr:.4f} TP={tp} FP={fp} FN={fn} TN={tn}')

if __name__=='__main__':
    evaluate(FlashFull,'FLASH')
    evaluate(MagicFull,'MAGIC')
