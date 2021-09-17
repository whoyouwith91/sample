import torch
from torch import nn
from torch_geometric.nn.conv import GCNConv, SAGEConv, GraphConv, GATConv, GINConv, GINEConv, EdgeConv, NNConv, PNAConv
from supergat_conv import SuperGATConv
import torch.nn.functional as F
from layers import *

def get_gnn(config):
    name = config['gnn_type']

    if name == 'gcn':
        return gcn_conv
    if name == 'sage':
        return sage_conv(config)
    if name == 'graphconv':
        return graph_conv(config)
    if name == 'resgatedgraphconv':
        return res_gated_graph_conv(config)
    if name == 'gatconv':
        return gat_conv(config)
    if name == 'ginconv':
        return gin_conv(config)
    if name == 'gineconv':
        return gine_conv(config)
    if name == 'edgeconv':
        return edge_conv(config)
    if name == 'supergat':
        return sgat_conv(config)
    if name == 'pnaconv':
        return pna_conv(config)
    if name == 'nnconv':
        return nn_conv(config)
    if name == 'dmpnn':
        return DMPNN(config)
    if name == 'dnn':
        return DNN(config)
    
class DNN():
    def __init__(self, config):
        super(DNN, self).__init__()
        self.emb_dim = config['emb_dim']

    def model(self):
        return Linear(self.emb_dim, self.emb_dim)

class gcn_conv():
    # https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv
    def __init__(self, config):
        super(gcn_conv, self).__init__()
        self.emb_dim = config['emb_dim'] 
        self.aggr = config['aggregate'] # default is add
        
    def model(self):
        # input: x(-1, emb_dim), edge index
        # return: x(-1, emb_dim)
        return GCNConv(in_channels=self.emb_dim, out_channels=self.emb_dim, aggr=self.aggr)

class sage_conv():
    # https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SAGEConv
    def __init__(self, config):
        super(sage_conv, self).__init__()
        self.emb_dim = config['emb_dim']
        self.aggr = config['aggregate'] # default is mean
        
    def model(self):
        # input: x(-1, emb_dim), edge index
        # return: x(-1, emb_dim)
        return SAGEConv(in_channels=self.emb_dim, out_channels=self.emb_dim, aggr=self.aggr)

class graph_conv():
    #https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GraphConv
    def __init__(self, config):
        super(graph_conv, self).__init__()
        self.emb_dim = config['emb_dim']
        self.aggr = config['aggregate'] # default is add
        
    def model(self):
        # input: x(-1, emb_dim), edge index 
        # return: x(-1, emb_dim)
        return GraphConv(in_channels=self.emb_dim, out_channels=self.emb_dim, aggr=self.aggr)

class res_gated_graph_conv():
    #https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.ResGatedGraphConv
    def __init__(self, config):
        super(res_gated_graph_conv, self).__init__()
        self.emb_dim = config['emb_dim']
        self.aggr = config['aggregate'] # default is add
        
    def model(self):
        # input: x(-1, emb_dim), edge index
        # # return: x(-1, emb_dim)
        return ResGatedGraphConv(in_channels=self.emb_dim, out_channels=self.emb_dim, aggr=self.aggr)

class gat_conv():
    #https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATConv
    def __init__(self, config):
        super(gat_conv, self).__init__()
        self.emb_dim = config['emb_dim']
        self.aggr = config['aggregate'] # default is add
        
    def model(self):
        # input: x(-1, emb_dim), edge index 
        # return: x(-1, emb_dim)
        return GATConv(in_channels=self.emb_dim, out_channels=self.emb_dim, aggr=self.aggr) # n_heads

class gin_conv():
    #https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GINConv
    def __init__(self, config):
        super(gin_conv, self).__init__()
        self.emb_dim = config['emb_dim']
        self.aggr = config['aggregate'] # default is add
        self.nn = nn.Sequential(nn.Linear(self.emb_dim, self.emb_dim), nn.ReLU())
        
    def model(self):
        # input: x(-1, emb_dim), edge index 
        # return: x(-1, emb_dim)
        return GINConv(nn=self.nn, train_eps=True)

class gine_conv():
    # https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GINEConv
    def __init__(self, config):
        super(gine_conv, self).__init__()
        self.emb_dim = config['emb_dim']
        self.aggr = config['aggregate'] # default is add 
        self.nn = nn.Sequential(nn.Linear(self.emb_dim, self.emb_dim), nn.ReLU())
        
    def model(self):
        # input: x(-1, emb_dim), edge(-1, emb_dim), edge index
        # edge need be converted from initial num bond feature to emb_dim before being fed 
        # return x(-1, emb_dim)
        return GINEConv(nn=self.nn, train_eps=True)
