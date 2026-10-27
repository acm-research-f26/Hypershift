import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import *
from supervisor import Supervisor


class Model(nn.Module):
    def __init__(self, snapshot, num_stocks, num_features, d_model, num_heads, sequence_len, num_edges, dropout=0.5, fusion_mode="sum"):
        super(Model, self).__init__()

        self.snapshot = snapshot
        self.num_stocks = num_stocks
        self.num_features = num_features
        self.num_heads = num_heads
        self.d_model = d_model
        self.sequence_len = sequence_len
        self.num_edges = num_edges
        self.fusion_mode = fusion_mode
        
        if self.snapshot is not None:
            self.num_snapshot = len(snapshot)
            self.params_of_snapshot = nn.Parameter(torch.Tensor(self.num_snapshot))
            nn.init.uniform_(self.params_of_snapshot, 0, 0.99)

        self.feature_embedding = nn.Linear(num_features, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.individual_attn = IntraStockAttention(d_model, num_heads, dropout)
        self.intensity_func = IntensityFunction(d_model)
        self.mixing_layer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )
        self.temporal_attn = TemporalAttention(d_model, dropout)
        self.dropout = nn.Dropout(dropout)

        self.whconv1 = WaveletHypergraphConvolution(d_model, num_stocks, K1=2, K2=2, approx=False)
        self.whconv2 = WaveletHypergraphConvolution(d_model, num_stocks, K1=2, K2=2, approx=False)

        self.dynamic_graph_layer = DynamicGraphLearning(d_model=d_model, num_edges=num_edges)

        if self.fusion_mode == "sum":
            self.beta = nn.Parameter(torch.tensor([0.33,0.33,0.33]))

        elif self.fusion_mode == "adaptive":
            self.repre_selection = RepresentationSelection(d_model=d_model)

        self.mlp = MLP(d_model, fusion_mode=self.fusion_mode)

    def forward(self, x):
        """
        inputs shape: (S, T, F)
        S: the number of stocks
        T: length of sequence
        F: the number of features
        D: dimension of embeddings
        """
        # Feature Embedding and Positional Encoding
        x = self.feature_embedding(x) # (S, T, D)
        x = self.positional_encoding(x) # (S, T, D)

        # Individual Stock Attention
        x = self.individual_attn(x) # (S, T, D)
        intensity = self.intensity_func(x) # (S, T, D)
        x = torch.cat([x, intensity], dim=-1) # (S, T, 2D)
        x = self.mixing_layer(x) # (S, T, D)

        # Temporal Attention
        x, _ = self.temporal_attn(x) # (S, D)

        # Wavelet Hypergraph Convolution
        repre = []
        for snapshot_idx in range(self.num_snapshot):
            hyper_repre_1 = F.leaky_relu(self.whconv1(x, self.snapshot, snapshot_idx), 0.1) # (S, D)
            hyper_repre_1 = self.dropout(hyper_repre_1)
            hyper_repre_2 = F.leaky_relu(self.whconv2(hyper_repre_1, self.snapshot, snapshot_idx), 0.1) # (S, D)
            repre.append(hyper_repre_2)

        hyper_repre = torch.zeros_like(hyper_repre_2)
        for snapshot_idx in range(self.num_snapshot):
            hyper_repre += repre[snapshot_idx] * self.params_of_snapshot[snapshot_idx] # (S, D)

        # Dynamic Graph Representation
        dynamic_repre, _ = self.dynamic_graph_layer(x) # (S, D)

        if self.fusion_mode == "sum": # Final Model

            weights = torch.softmax(self.beta, dim=0)
            fused_representation = (
                weights[0] * x +
                weights[1] * dynamic_repre +
                weights[2] * hyper_repre
            ) # (S, D)
        
        elif self.fusion_mode == "cat":

            fused_representation = torch.cat([x, dynamic_repre, hyper_repre], dim=-1) # (S, 3D)

        elif self.fusion_mode == "adaptive":
            
            fused_representation = self.repre_selection(x, dynamic_repre, hyper_repre) # (S, D)

        # Final Prediction
        x = self.mlp(fused_representation) # (S, D or 3D) -> (S)

        return x # (S)


class Models(Supervisor):
    def __init__(self, config, **kwargs) -> None:
        super(Models, self).__init__(config, **kwargs)

        self.num_stocks = config['model']['num_stocks']
        self.num_features = config['model']['num_features']
        self.d_model = config['model']['d_model']
        self.num_heads = config['model']['num_heads']
        self.dropout = config['model']['dropout']
        self.sequence_len = config['model']['sequence_len']
        self.snapshot = None
        self.num_edges = config['model']['num_edges']
        self.fusion_mode = config['model']['fusion_mode']

        self.init_model()

    def init_model(self):

        print("Model Initialization...")

        self.model = Model(snapshot=self.snapshot,
                           num_stocks=self.num_stocks,
                           num_features=self.num_features,
                           d_model=self.d_model,
                           num_heads=self.num_heads,
                           sequence_len=self.sequence_len,
                           num_edges=self.num_edges,
                           dropout=self.dropout,
                           fusion_mode=self.fusion_mode)
        
        super(Models, self).init_model()

    def _snapshot(self, snapshot):

        self.snapshot = snapshot
        self.init_model()
