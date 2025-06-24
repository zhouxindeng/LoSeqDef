import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Graph Neural Network for User Collaborative Filtering
class GNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GNNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            
    def forward(self, features, adj):
        # features: [batch_size, num_nodes, in_dim]
        # adj: [batch_size, num_nodes, num_nodes]
        support = torch.matmul(features, self.weight)  # [batch_size, num_nodes, out_dim]
        output = torch.bmm(adj, support)  # [batch_size, num_nodes, out_dim]
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class UserGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout):
        super(UserGNN, self).__init__()
        self.n_layers = n_layers
        self.dropout = dropout
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(GNNLayer(input_dim, hidden_dim))
        for _ in range(n_layers - 2):
            self.gnn_layers.append(GNNLayer(hidden_dim, hidden_dim))
        self.gnn_layers.append(GNNLayer(hidden_dim, output_dim))
        
        # Layer normalization and dropout
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers-1)] + [nn.LayerNorm(output_dim)])
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, features, adj):
        # features: [batch_size, num_nodes, input_dim]
        # adj: [batch_size, num_nodes, num_nodes]
        h = features
        for i, layer in enumerate(self.gnn_layers):
            h = layer(h, adj)
            h = self.layer_norms[i](h)
            if i < self.n_layers - 1:
                h = F.relu(h)
                h = self.dropout_layer(h)
        return h