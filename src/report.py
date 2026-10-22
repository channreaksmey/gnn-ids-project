import sys; sys.path.insert(0,'.')
import torch, torch.nn as nn, os
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_recall_curve, roc_curve, roc_auc_score)
from src.data_v2 import load_graphs, get_node_labels, get_features
from src.flash_full import FlashFull
from src.magic_full import MagicFull

def plot(name, ModelClass):
    graphs=load_graphs('data')
    test=[(n,g) for n,g in graphs if 'test' in n][0][1]
    enc=ModelClass(); head=nn.Linear(128,1)
    ck=torch.load(f'out/{name}_full.pt'); enc.load_state_dict(ck['enc']); head.load_state_dict(ck['head'])
    enc.eval(); head.eval()
    with torch.no_grad():
        s=torch.sigmoid(head(enc(test,get_features(test))).squeeze(-1))
    y=get_node_labels(test).int().numpy(); s=s.numpy()
    fpr,tpr,_=roc_curve(y,s); auc=roc_auc_score(y,s)
    plt.figure(); plt.plot(fpr,tpr,label=f'{name} AUC={auc:.3f}'); plt.legend(); plt.title('ROC'); plt.savefig(f'out/{name}_roc.png'); plt.close()
    prec,rec,_=precision_recall_curve(y,s)
    plt.figure(); plt.plot(rec,prec); plt.title('PR'); plt.savefig(f'out/{name}_pr.png'); plt.close()
    print('saved',name)

if __name__=='__main__':
    os.makedirs('out',exist_ok=True)
    plot('FLASH',FlashFull); plot('MAGIC',MagicFull)
