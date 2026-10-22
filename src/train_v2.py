import sys; sys.path.insert(0,'.')
import torch, torch.nn as nn
from src.data_v2 import load_graphs, get_node_labels, get_features
from src.flash import FlashEncoder
from src.magic import MagicEncoder

def train(ModelClass, name, epochs=10, lr=1e-3):
    graphs=load_graphs('data')
    enc=ModelClass(); head=nn.Linear(128,1)
    opt=torch.optim.Adam(list(enc.parameters())+list(head.parameters()),lr=lr)
    loss_fn=nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        total=0
        for n,g in graphs:
            y=get_node_labels(g)
            x=get_features(g)
            h=enc(g,x)
            s=head(h).squeeze(-1)
            l=loss_fn(s,y)
            opt.zero_grad(); l.backward(); opt.step()
            total+=l.item()
        print(name,ep,total/len(graphs))

if __name__=='__main__':
    train(FlashEncoder,'FLASH',epochs=5)
    train(MagicEncoder,'MAGIC',epochs=5)
