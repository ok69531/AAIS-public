# source code from https://github.com/snap-stanford/ogb/tree/master/examples/graphproppred/mol

import enum
from multiprocessing.sharedctypes import Value
import torch
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.utils import degree

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


### GCN convolution
class GCNConv(MessagePassing):
    def __init__(self, emb_dim, aggr = 'add'):
        super(GCNConv, self).__init__()
        
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)
        self.aggr = aggr
    
    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)
        
        row, col = edge_index
        
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        return self.propagate(edge_index, x = x, edge_attr = edge_embedding, norm = norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1, 1)
    
    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)
    
    def update(self, aggr_out):
        return aggr_out


### GIN convolution
class GINConv(MessagePassing):
    def __init__(self, emb_dim, aggr = 'add'):
        super(GINConv, self).__init__()
        
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.aggr = aggr
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)
        
    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x = x, edge_attr = edge_embedding))
        
        return out
    
    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)
    
    def update(self, aggr_out):
        return aggr_out


### GNN to generate node embedding
class GNN(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, drop_ratio = 0.3, JK = 'last', residual = False, gnn_type = 'gcn'):
        super(GNN, self).__init__()
        
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        
        ### add residual connection or not
        self.residual = residual
        
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        
        self.atom_encoder = AtomEncoder(emb_dim)
        
        ### List of GNNs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == 'gcn':
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == 'gin':
                self.gnns.append(GINConv(emb_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))
        
        ### List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
        
    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("Unmatched number of arguments.")
        
        if x.dtype == torch.float32:
            x = x
        elif x.dtype == torch.int64:
            x = self.atom_encoder(x)
        
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            
            if layer == self.num_layer - 1:
                # remove relu for the las layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            
            if self.residual:
                h += h_list[layer]
                
            h_list.append(h)
        
        ### Different implementations of JK-concat
        if self.JK == 'last':
            node_representation = h_list[-1]
        elif self.JK == 'sum':
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]
        
        return node_representation


### Virtual GNN to generate node embedding
class GNNVirtual(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, drop_ratio = 0.3, JK = 'last', residual = False, gnn_type = 'gcn'):
        super(GNNVirtual, self).__init__()
        
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        
        ### add residual connection or not
        self.residual = residual
        
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        
        self.atom_encoder = AtomEncoder(emb_dim)
        
        ### set the initiail virtual node embedding to 0
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)
        
        ### List of GNNs
        self.gnns = torch.nn.ModuleList()
        for layer in range(self.num_layer):
            if gnn_type == 'gcn':
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == 'gin':
                self.gnns.append(GINConv(emb_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))
        
        ### List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(self.num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
        
        ### List of MLPs to transform virtual node at ever layer
        self.mlp_virtual_list = torch.nn.ModuleList()
        for layer in range(self.num_layer - 1):
            self.mlp_virtual_list.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim),
                                                             torch.nn.BatchNorm1d(2*emb_dim),
                                                             torch.nn.ReLU(),
                                                             torch.nn.Linear(2*emb_dim, emb_dim),
                                                             torch.nn.BatchNorm1d(emb_dim),
                                                             torch.nn.ReLU()))
        
    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("Unmatched number of arguments.")
        
        if x.dtype == torch.float32:
            x = x
        elif x.dtype == torch.int64:
            x = self.atom_encoder(x)
        virtualnode_embedding = self.virtualnode_embedding(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))
        
        h_list = [x]
        for layer in range(self.num_layer):
            ### add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]
            
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            
            if layer == self.num_layer - 1:
                # remove relu for the las layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            
            if self.residual:
                h = h + h_list[layer]
                
            h_list.append(h)
            
            ### update the virtual nodes
            if layer < self.num_layer - 1:
                virtualnode_embedding_tmp = global_add_pool(h_list[layer], batch) + virtualnode_embedding
                ### transform virtual nodes using MLP
                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(self.mlp_virtual_list[layer](virtualnode_embedding_tmp), self.drop_ratio, training = self.training)
                else:
                    virtualnode_embedding = F.dropout(self.mlp_virtual_list[layer](virtualnode_embedding_tmp), self.drop_ratio, training = self.training)
        
        ### Different implementations of JK-concat
        if self.JK == 'last':
            node_representation = h_list[-1]
        elif self.JK == 'sum':
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]
        
        return node_representation


class GNNGraphPred(torch.nn.Module):
    def __init__(self, num_tasks, num_layer, emb_dim, gnn_type, graph_pooling = 'mean', drop_ratio = 0.3, JK = 'last', virtual_node = False, residual = False):
        super(GNNGraphPred, self).__init__()
        
        self.num_tasks = num_tasks
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.graph_pooling = graph_pooling
        
        if self.num_layer < 2:
            raise ValueError('Number of GNN layers must be greater than 1.')
        
        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn = GNNVirtual(num_layer, emb_dim, drop_ratio = drop_ratio, JK = JK, residual = residual, gnn_type = gnn_type)
        else:
            self.gnn = GNN(num_layer, emb_dim, drop_ratio = drop_ratio, JK = JK, residual = residual, gnn_type = gnn_type)
        
        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == 'sum':
            self.pool = global_add_pool
        elif self.graph_pooling == 'mean':
            self.pool = global_mean_pool
        elif self.graph_pooling == 'max':
            self.pool = global_max_pool
        elif self.graph_pooling == 'attention':
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim),
                                                                      torch.nn.BatchNorm1d(2*emb_dim),
                                                                      torch.nn.ReLU(),
                                                                      torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == 'set2set':
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError('Invalid graph pooling type.')
        
        if graph_pooling == 'set2set':
            self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)
            
    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("Unmatched number of arguments.")
        
        node_representation = self.gnn(x, edge_index, edge_attr, batch)
        graph_representation = self.pool(node_representation, batch)
        
        return self.graph_pred_linear(graph_representation)