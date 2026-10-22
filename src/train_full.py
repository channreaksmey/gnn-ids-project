import sys; sys.path.insert(0,'.')
import torch, torch.nn as nn
from sklearn.model_selection import KFold
from src.data_v2 import load_graphs, get_node_labels, get_features
from src.flash_full import FlashFull
from src.magic_full import MagicFull

def train_cv(ModelClass, name, epochs=5, lr=1e-3, n_splits=4):
    graphs=load_graphs('data')
    # use train graphs for CV
    train_graphs=[(n,g) for n,g in graphs if 'train' in n]
    kf=KFold(n_splits=min(n_splits,len(train_graphs)),shuffle=True,random_state=42)
    for fold,(tr,va) in enumerate(kf.split(train_graphs)):
        enc=ModelClass(); head=nn.Linear(128,1)
        opt=torch.optim.Adam(list(enc.parameters())+list(head.parameters()),lr=lr)
        loss_fn=nn.BCEWithLogitsLoss()
        for ep in range(epochs):
            for i in tr:
                _,g=train_graphs[i]
                y=get_node_labels(g); x=get_features(g)
                h=enc(g,x); s=head(h).squeeze(-1)
                l=loss_fn(s,y); opt.zero_grad(); l.backward(); opt.step()
        print(f'{name} fold {fold} done')
    # final train on all train
    enc=ModelClass(); head=nn.Linear(128,1)
    opt=torch.optim.Adam(list(enc.parameters())+list(head.parameters()),lr=lr)
    for ep in range(epochs):
        for _,g in train_graphs:
            y=get_node_labels(g); x=get_features(g)
            h=enc(g,x); s=head(h).squeeze(-1)
            l=loss_fn(s,y); opt.zero_grad(); l.backward(); opt.step()
    import os; os.makedirs('out',exist_ok=True)
    torch.save({'enc':enc.state_dict(),'head':head.state_dict()},f'out/{name}_full.pt')
    print(f'saved out/{name}_full.pt')

if __name__=='__main__':
    train_cv(FlashFull,'FLASH')
    train_cv(MagicFull,'MAGIC')
