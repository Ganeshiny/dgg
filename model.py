import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, global_mean_pool, AttentionalAggregation

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
    """DeepGreenGO main architecture: GCN → GATv2 with BatchNorm and LayerNorm.
    
    Follows the manuscript description:
    - Input projection: Linear → LeakyReLU → BatchNorm → Dropout
    - GCN layer with LayerNorm after aggregation
    - GATv2 layer (more expressive attention than GAT v1)
    - AttentionalAggregation pooling → prediction head
    """
    def __init__(self, input_size, hidden_sizes, output_size, num_attention_heads=4):
        super(HybridGNN, self).__init__()
        self.input_linear = nn.Linear(input_size, hidden_sizes[0])
        self.input_bn     = nn.BatchNorm1d(hidden_sizes[0])
        
        # GCN layer
        self.gcn_conv = GCNConv(hidden_sizes[0], hidden_sizes[1])
        self.gcn_ln   = nn.LayerNorm(hidden_sizes[1])
        
        # GATv2 layer (GATv2Conv is more expressive than GATConv/v1)
        gat_out_dim = hidden_sizes[1] if len(hidden_sizes) < 3 else hidden_sizes[2]
        self.gat_conv = GATv2Conv(hidden_sizes[1], gat_out_dim,
                                   heads=num_attention_heads, concat=False)
        self.gat_ln   = nn.LayerNorm(gat_out_dim)

        self.attention_pool = AttentionalAggregation(
            gate_nn=nn.Sequential(nn.Linear(gat_out_dim, 128), nn.ReLU(), nn.Linear(128, 1))
        )
        
        # Prediction head: Linear → LeakyReLU → BatchNorm → Dropout → Linear
        self.head_linear1 = nn.Linear(gat_out_dim, gat_out_dim)
        self.head_bn      = nn.BatchNorm1d(gat_out_dim)
        self.head_dropout = nn.Dropout(0.3)
        self.output_layer = nn.Linear(gat_out_dim, output_size)
        
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index, batch):
        # Input transformation
        x = self.input_linear(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.input_bn(x)
        x = self.dropout(x)
        
        # GCN layer with LayerNorm
        x = self.gcn_conv(x, edge_index)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.gcn_ln(x)
        
        # GATv2 layer with LayerNorm
        x = self.gat_conv(x, edge_index)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.gat_ln(x)
        
        # Global pooling
        graph_embedding = self.attention_pool(x, batch)
        
        # Prediction head
        out = F.leaky_relu(self.head_linear1(graph_embedding), negative_slope=0.1)
        out = self.head_bn(out)
        out = self.head_dropout(out)
        return self.output_layer(out)

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
