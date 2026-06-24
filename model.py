import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, AttentionalAggregation

class GCNModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(GCNModel, self).__init__()
        self.linear_input = nn.Linear(input_size, hidden_sizes[0])
        self.conv_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.conv_layers.append(GCNConv(hidden_sizes[i], hidden_sizes[i + 1]))
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.dropout_input = nn.Dropout(0.3)

    def forward(self, x, edge_index, batch):
        x = F.leaky_relu(self.linear_input(x), negative_slope=0.4)
        x = self.dropout_input(x)
        for conv_layer in self.conv_layers:
            x = F.leaky_relu(conv_layer(x, edge_index), negative_slope=0.4)
        x = global_mean_pool(x, batch)
        return self.output_layer(x)

class GATModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, num_attention_heads=4):
        super(GATModel, self).__init__()
        self.input_linear = nn.Linear(input_size, hidden_sizes[0])
        self.gnn_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.gnn_layers.append(
                GATConv(hidden_sizes[i], hidden_sizes[i + 1], heads=num_attention_heads, concat=False)
            )
        self.attention_pool = AttentionalAggregation(
            gate_nn=nn.Sequential(nn.Linear(hidden_sizes[-1], 128), nn.ReLU(), nn.Linear(128, 1))
        )
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index, batch):
        x = F.leaky_relu(self.input_linear(x), negative_slope=0.1)
        x = self.dropout(x)
        for gnn_layer in self.gnn_layers:
            x = F.leaky_relu(gnn_layer(x, edge_index), negative_slope=0.1)
        graph_embedding = self.attention_pool(x, batch)
        return self.output_layer(graph_embedding)

class HybridGNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, num_attention_heads=4):
        super(HybridGNN, self).__init__()
        self.input_linear = nn.Linear(input_size, hidden_sizes[0])
        
        # Hybrid layers: GCN -> GAT
        self.gcn_conv = GCNConv(hidden_sizes[0], hidden_sizes[1])
        self.gat_conv = GATConv(hidden_sizes[1], hidden_sizes[1] if len(hidden_sizes) < 3 else hidden_sizes[2], heads=num_attention_heads, concat=False)
        
        pool_dim = hidden_sizes[1] if len(hidden_sizes) < 3 else hidden_sizes[2]
        
        self.attention_pool = AttentionalAggregation(
            gate_nn=nn.Sequential(nn.Linear(pool_dim, 128), nn.ReLU(), nn.Linear(128, 1))
        )
        self.output_layer = nn.Linear(pool_dim, output_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index, batch):
        x = F.leaky_relu(self.input_linear(x), negative_slope=0.1)
        x = self.dropout(x)
        x = F.leaky_relu(self.gcn_conv(x, edge_index), negative_slope=0.1)
        x = F.leaky_relu(self.gat_conv(x, edge_index), negative_slope=0.1)
        graph_embedding = self.attention_pool(x, batch)
        return self.output_layer(graph_embedding)

class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLPModel, self).__init__()
        self.input_linear = nn.Linear(input_size, hidden_sizes[0])
        self.hidden_linear = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.output_layer = nn.Linear(hidden_sizes[1], output_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index, batch):
        # Ignore edge_index for MLP
        x = F.leaky_relu(self.input_linear(x), negative_slope=0.1)
        x = self.dropout(x)
        x = F.leaky_relu(self.hidden_linear(x), negative_slope=0.1)
        graph_embedding = global_mean_pool(x, batch)
        return self.output_layer(graph_embedding)

def get_model(model_name, input_size, hidden_sizes, output_size):
    if model_name.lower() == "gcn":
        return GCNModel(input_size, hidden_sizes, output_size)
    elif model_name.lower() == "gat":
        return GATModel(input_size, hidden_sizes, output_size)
    elif model_name.lower() == "mlp":
        return MLPModel(input_size, hidden_sizes, output_size)
    elif model_name.lower() in ["hybrid", "rarelabelgnn", "deepgreengo"]:
        return HybridGNN(input_size, hidden_sizes, output_size)
    else:
        raise ValueError(f"Unknown model type: {model_name}")
