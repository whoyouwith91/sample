from typing import Union, Tuple, Callable
from torch_geometric.typing import OptTensor, OptPairTensor, Adj, Size
from torch.nn import Parameter

import numpy as np
import torch
import torch.nn as nn
from torch.nn import GRU, LSTM
import torch.nn.functional as F
from torch.nn import ModuleList, Sequential, Linear, ReLU, LeakyReLU, ELU, Tanh
from torch_scatter import scatter_mean
from torch_scatter import scatter
from typing import Optional, List, Dict
from torch_geometric.typing import Adj, OptTensor

from torch_geometric.nn import MessagePassing
from torch_geometric.nn import EdgePooling, global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set, TopKPooling, SAGPooling
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
from k_gnn import avg_pool, add_pool, max_pool
from helper import *

class NNDropout(nn.Module):
    def __init__(self, weight_regularizer, dropout_regularizer, init_min=0.1, init_max=0.1):
        super(NNDropout, self).__init__()
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        
        init_min = np.log(init_min) - np.log(1. - init_min)
        init_max = np.log(init_max) - np.log(1. - init_max)
        
        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min, init_max)) # 
    
    def _concrete_dropout(self, x, p):
        # This is like reparameterization tricks. 
        eps = 1e-7
        temp = 0.1 
        # Concrete distribution relaxation of the Bernoulli random variable
        unif_noise = torch.rand_like(x)

        drop_prob = (torch.log(p + eps)
                    - torch.log(1 - p + eps)
                    + torch.log(unif_noise + eps)
                    - torch.log(1 - unif_noise + eps))
        
        drop_prob = torch.sigmoid(drop_prob / temp)
        random_tensor = 1 - drop_prob
        retain_prob = 1 - p
        
        x  = torch.mul(x, random_tensor)
        x /= retain_prob
        
        return x

    def forward(self, data, layer):
        #p = torch.sigmoid(self.p_logit) # this is the drop out probablity, trainable. 
        p = torch.scalar_tensor(0.1)
        out = layer(self._concrete_dropout(data, p))
        x = out
        
        sum_of_square = 0
        for param in layer.parameters():
            sum_of_square += torch.sum(torch.pow(param, 2))
        
        weights_regularizer = self.weight_regularizer * sum_of_square / (1 - p)
        
        dropout_regularizer = p * torch.log(p)
        dropout_regularizer += (1. - p) * torch.log(1. - p)
        
        input_dimensionality = x[0].numel() # Number of elements of first item in batch
        #print(input_dimensionality)
        dropout_regularizer *= self.dropout_regularizer * input_dimensionality
        
        regularization = weights_regularizer + dropout_regularizer
        return out, regularization          

def index_select_ND(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Selects the message features from source corresponding to the atom or bond indices in index.
    :param source: A tensor of shape (num_bonds, hidden_size) containing message features.
    :param index: A tensor of shape (num_atoms/num_bonds, max_num_bonds) containing the atom or bond
    indices to select from source.
    :return: A tensor of shape (num_atoms/num_bonds, max_num_bonds, hidden_size) containing the message
    features corresponding to the atoms/bonds specified in index.
    """
    index_size = index.size()  # (num_atoms/num_bonds, max_num_bonds)
    suffix_dim = source.size()[1:]  # (hidden_size,)
    final_size = index_size + suffix_dim  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    target = source.index_select(dim=0, index=index.view(-1))  # (num_atoms/num_bonds * max_num_bonds, hidden_size)
    target = target.view(final_size)  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    return target