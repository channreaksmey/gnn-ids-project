import pickle, glob, os
import torch

def load_graphs(data_dir):
    graphs=[]
    for p in sorted(glob.glob(os.path.join(data_dir,'*.pkl'))):
        with open(p,'rb') as f:
            graphs.append((os.path.basename(p), pickle.load(f)))
    return graphs

def get_node_labels(g):
    return (g.ndata['type']==4).float()

def get_features(g):
    return g.ndata['attr']
