"""
GNN models with auxiliary tasks for FireGNN framework.
Combines main classification with auxiliary learning objectives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch_geometric.data import Data


class AuxiliaryGCN(nn.Module):
    """
    GCN with auxiliary tasks.
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_classes, num_layers=2, dropout=0.5):
        super(AuxiliaryGCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_classes = num_classes
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # Batch normalization
        self.bns = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Main classification head
        self.classifier = nn.Linear(hidden_channels, out_channels)
        
        # Auxiliary task heads
        self.sim_predictor = nn.Linear(hidden_channels, num_classes)
        self.homophily_predictor = nn.Linear(hidden_channels, 1)
        self.entropy_predictor = nn.Linear(hidden_channels, 1)
        
    def forward(self, x, edge_index, edge_attr=None):
        # GCN forward pass
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index, edge_attr)
        
        # Main classification
        main_out = self.classifier(x)
        
        # Auxiliary predictions
        sim_out = self.sim_predictor(x)
        homophily_out = self.homophily_predictor(x)
        entropy_out = self.entropy_predictor(x)
        
        return (F.log_softmax(main_out, dim=1), 
                F.softmax(sim_out, dim=1),
                torch.sigmoid(homophily_out).squeeze(-1),
                torch.sigmoid(entropy_out).squeeze(-1))


class AuxiliaryGAT(nn.Module):
    """
    GAT with auxiliary tasks.
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_classes, num_layers=2, heads=8, dropout=0.5, negative_slope=0.2):
        super(AuxiliaryGAT, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.num_classes = num_classes
        
        # GAT layers
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, 
                                 dropout=dropout, negative_slope=negative_slope))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, 
                                    heads=heads, dropout=dropout, 
                                    negative_slope=negative_slope))
        
        if num_layers > 1:
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, 
                                    heads=1, dropout=dropout, 
                                    negative_slope=negative_slope))
        
        # Batch normalization
        self.bns = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
        
        # Main classification head
        self.classifier = nn.Linear(hidden_channels, out_channels)
        
        # Auxiliary task heads
        self.sim_predictor = nn.Linear(hidden_channels, num_classes)
        self.homophily_predictor = nn.Linear(hidden_channels, 1)
        self.entropy_predictor = nn.Linear(hidden_channels, 1)
        
    def forward(self, x, edge_index, edge_attr=None):
        # GAT forward pass
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index, edge_attr)
        
        # Main classification
        main_out = self.classifier(x)
        
        # Auxiliary predictions
        sim_out = self.sim_predictor(x)
        homophily_out = self.homophily_predictor(x)
        entropy_out = self.entropy_predictor(x)
        
        return (F.log_softmax(main_out, dim=1), 
                F.softmax(sim_out, dim=1),
                torch.sigmoid(homophily_out).squeeze(-1),
                torch.sigmoid(entropy_out).squeeze(-1))


class MLP(nn.Module):
    """
    Multi-layer perceptron for GIN.
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        
        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
        
        if num_layers > 1:
            self.lins.append(nn.Linear(hidden_channels, out_channels))
        
        # Batch normalization
        self.bns = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.bns.append(nn.BatchNorm1d(hidden_channels))
    
    def forward(self, x):
        for i in range(self.num_layers - 1):
            x = self.lins[i](x)
            x = self.bns[i](x)
            x = F.relu(x)
        
        x = self.lins[-1](x)
        return x


class AuxiliaryGIN(nn.Module):
    """
    GIN with auxiliary tasks.
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_classes, num_layers=2, dropout=0.5):
        super(AuxiliaryGIN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_classes = num_classes
        
        # GIN layers
        self.convs = nn.ModuleList()
        self.convs.append(GINConv(MLP(in_channels, hidden_channels, hidden_channels)))
        
        for _ in range(num_layers - 2):
            self.convs.append(GINConv(MLP(hidden_channels, hidden_channels, hidden_channels)))
        
        if num_layers > 1:
            self.convs.append(GINConv(MLP(hidden_channels, hidden_channels, hidden_channels)))
        
        # Batch normalization
        self.bns = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Main classification head
        self.classifier = nn.Linear(hidden_channels, out_channels)
        
        # Auxiliary task heads
        self.sim_predictor = nn.Linear(hidden_channels, num_classes)
        self.homophily_predictor = nn.Linear(hidden_channels, 1)
        self.entropy_predictor = nn.Linear(hidden_channels, 1)
        
    def forward(self, x, edge_index, edge_attr=None):
        # GIN forward pass
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index, edge_attr)
        
        # Main classification
        main_out = self.classifier(x)
        
        # Auxiliary predictions
        sim_out = self.sim_predictor(x)
        homophily_out = self.homophily_predictor(x)
        entropy_out = self.entropy_predictor(x)
        
        return (F.log_softmax(main_out, dim=1), 
                F.softmax(sim_out, dim=1),
                torch.sigmoid(homophily_out).squeeze(-1),
                torch.sigmoid(entropy_out).squeeze(-1))


def get_auxiliary_model(model_type, in_channels, hidden_channels, out_channels, num_classes, **kwargs):
    """
    Factory function to create auxiliary task models.
    
    Args:
        model_type: Type of model ('gcn', 'gat', 'gin')
        in_channels: Input feature dimension
        hidden_channels: Hidden layer dimension
        out_channels: Output dimension (number of classes)
        num_classes: Number of classes for auxiliary tasks
        **kwargs: Additional arguments for model initialization
        
    Returns:
        nn.Module: Initialized auxiliary model
    """
    model_type = model_type.lower()
    
    if model_type == 'gcn':
        return AuxiliaryGCN(in_channels, hidden_channels, out_channels, num_classes, **kwargs)
    elif model_type == 'gat':
        return AuxiliaryGAT(in_channels, hidden_channels, out_channels, num_classes, **kwargs)
    elif model_type == 'gin':
        return AuxiliaryGIN(in_channels, hidden_channels, out_channels, num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}") 