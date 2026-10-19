import pickle, glob, os
import torch

def load_graphs(data_dir):
    graphs=[]
    for p in sorted(glob.glob(os.path.join(data_dir,'*.pkl'))):
        with open(p,'rb') as f:
            graphs.append((os.path.basename(p), pickle.load(f)))
    return graphs

def node_labels(g):
    # Type 4 nodes are the rare malicious entities in these provenance graphs.
    return (g.ndata['type']==4).float()

def graph_stats(g):
    return {'nodes':g.num_nodes(),'edges':g.num_edges(),
            'node_feat_dim':g.ndata['attr'].shape[1],
            'edge_feat_dim':g.edata['attr'].shape[1],
            'n_malicious':int((g.ndata['type']==4).sum())}
