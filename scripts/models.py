import torch
from torch import nn
from torch_geometric.nn.glob.glob import global_add_pool
from torch_scatter import scatter_mean
from torch_geometric.utils import add_self_loops, degree, softmax
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
from helper import *
from layers import *
from gnns import *
from PhysDimeNet import PhysDimeNet
from torch_geometric.nn.norm import PairNorm
import time, sys

def get_model(config):
    name = config['model']
    if name == None:
        raise ValueError('Please specify one model you want to work on!')
    if name == '1-GNN':
        return GNN_1(config)
    if name == '1-2-GNN':
        return GNN_1_2(config)
    if name == '1-efgs-GNN':
        return GNN_1_EFGS(config)
    if name == '1-interaction-GNN':
        if config['dataset'] in ['solWithWater', 'solWithWater_calc/ALL', 'logpWithWater', 'logpWithWater_calc/ALL']:
            if config['interaction_simpler']:
                return GNN_1_WithWater_simpler(config)
            else:
                return GNN_1_WithWater(config)
    if name == '1-2-GNN_dropout':
        return knn_dropout
    if name == '1-2-GNN_swag':
        return knn_swag
    if name == 'physnet':
        return PhysDimeNet(**config)
    
class GNN(torch.nn.Module):
    """
    Basic GNN unit modoule.
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat
    Output:
        node representations
    """
    def __init__(self, config):
        super(GNN, self).__init__()
        self.config = config
        self.num_layer = config['num_layer']
        self.emb_dim = config['emb_dim']
        self.drop_ratio = config['drop_ratio']
        self.JK = config['JK']
        self.gnn_type = config['gnn_type']
        self.gradCam = config['gradCam']
        self.bn = config['bn']
        self.act_fn = activation_func(config)

        if self.num_layer < 1:
            raise ValueError("Number of GNN layers must be no less than 1.")
        if self.config['gnn_type'] in ['gineconv', 'pnaconv', 'nnconv']:
            self.linear_b = Linear(self.config['num_bond_features'], self.emb_dim)
        
        # define graph conv layers
        if self.gnn_type in ['dmpnn']:
            self.gnns = get_gnn(self.config) # already contains multiple layers
        else:
            self.linear_x = Linear(self.config['num_atom_features'], self.emb_dim)
            self.gnns = nn.ModuleList()
            for _ in range(self.num_layer):
                self.gnns.append(get_gnn(self.config).model())
        
        if config['pooling'] == 'edge': # for edge pooling only
            self.pool = PoolingFN(self.config)
            assert len(self.pool) == self.num_layer
        
        ###List of batchnorms
        if self.bn and self.gnn_type not in ['dmpnn']:
            self.batch_norms = nn.ModuleList()
            for _ in range(self.num_layer):
                self.batch_norms.append(nn.BatchNorm1d(self.emb_dim))

        #self.pair_norm = PairNorm()
        self.gradients = None # for GradCAM     

    ## hook for the gradients of the activations GradCAM
    def activations_hook(self, grad):
        self.gradients = grad
    
    # method for the gradient extraction GradCAM
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return x

    #def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        elif len(argv) == 5:
            f_atoms, f_bonds, a2b, b2a, b2revb = argv[0], argv[1], argv[2], argv[3], argv[4]
        else:
            raise ValueError("unmatched number of arguments.")

        if self.gnn_type == 'dmpnn':
            node_representation = self.gnns(f_atoms, f_bonds, a2b, b2a, b2revb)
        else:
            x = self.linear_x(x) # first linear on atoms 
            h_list = [x]
            #x = F.relu(self.linear_x(x))
            if self.config['gnn_type'] in ['gineconv', 'pnaconv', 'nnconv']:
                edge_attr = self.linear_b(edge_attr.float()) # first linear on bonds 
                #edge_attr = F.relu(self.linear_b(edge_attr.float())) # first linear on bonds 

            for layer in range(self.num_layer):
                if self.config['residual_connect']: # adding residual connection
                    residual = h_list[layer] 
                if self.config['gnn_type'] in ['gineconv', 'pnaconv', 'nnconv']:
                    h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
                elif self.config['gnn_type'] in ['dnn']:
                    h = self.gnns[layer](h_list[layer])
                else:
                    h = self.gnns[layer](h_list[layer], edge_index)
                
                ### in order of Skip >> BN >> ReLU
                if self.config['residual_connect']:
                    h += residual
                if self.bn:
                    h = self.batch_norms[layer](h)
                
                #h = self.pair_norm(h, data.batch)
                if layer == self.num_layer - 1:
                    #remove relu for the last layer
                    h = F.dropout(h, self.drop_ratio, training = self.training)
                    self.last_conv = self.get_activations(h)
                else:
                    h = self.act_fn(h)
                    h = F.dropout(h, self.drop_ratio, training = self.training)
                if self.config['pooling'] == 'edge':
                    h, edge_index, batch, _ = self.pool[layer](h, edge_index, batch=batch)
                h_list.append(h)
                if self.gradCam and layer == self.num_layer - 1:
                    h.register_hook(self.activations_hook)
            
            ### Different implementations of Jk-concat
            if self.JK == "concat":
                node_representation = torch.cat(h_list, dim = 1)
            elif self.JK == "last":
                node_representation = h_list[-1]
            elif self.JK == "max":
                h_list = [h.unsqueeze_(0) for h in h_list]
                node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
                #print(node_representation.shape)
            elif self.JK == "sum":
                h_list = [h.unsqueeze_(0) for h in h_list]
                node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)
                #print(node_representation.shape)
        
        if self.config['pooling'] == 'edge':
            return node_representation, batch
        else:
            return node_representation