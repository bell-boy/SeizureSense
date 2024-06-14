import torch
import numpy as np
from torch import nn
import scipy as sp
from torch_geometric.utils import dense_to_sparse

class GraphLearner(nn.Module):
    def __init__(self, depth, threshold=0.1):
        super().__init__()
        self.depth = depth
        self.threshold= threshold
        self.Q = nn.Linear(depth,depth, bias=False)
        self.K = nn.Linear(depth,depth, bias=False)

    def next(self,tensor):
        queries = self.Q(tensor)
        keys = torch.transpose(self.K(tensor), -1,-2)

        # this matrix multiplication results in the shape of the adj. matrix
        adj = torch.matmul(queries,keys) / np.sqrt(self.depth) 
        
        # return the softmax of the learned adjacency matrix
        adj = torch.softmax(adj,dim=-1)

        # Average the adj matrix with the transpose so that the graph will be bidirectional
        # e.g the connection from [1,2] should be the same as [2,1]
        adj = (adj + adj.transpose(-1,-2))/2

        # Set values below specified threshold to 0
        adj[adj<=self.threshold] = 0

        edge_indexes, edge_weights = dense_to_sparse(adj)
        return edge_indexes, edge_weights