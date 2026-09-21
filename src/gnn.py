from __future__ import annotations
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv

# Each node begins with its input features: $h_i^{(0)} = x_i$
class GNN(nn.Module):
    
    def __init__(self, in_channels: int, hidden_channels: int = 32, out_channels: int = 1, dropout: float = 0.2) -> None:
        super().__init__()
        if min(in_channels, hidden_channels, out_channels) <= 0:
            raise ValueError("All channel dimensions must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0,1).")
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.head = nn.Linear(hidden_channels, out_channels)
        self.reset_parameters()

    # The regression head predicts one return per node, i.e. $\hat{y}_i = w^\top h_i^{(2)} + b$
    # encode returns $H^{(2)} \in \mathbb{R}^{N \times 32}$
    # We then return $\hat{Y} = H^{(2)}W^{(\text{head})} + \mathbb{1}(b^{\text{(head)}})^T$
    def forward(self, x: Tensor, edge_idx: Tensor, edge_weight: Tensor | None = None) -> Tensor:
        return self.head(self.encode(x,edge_idx, edge_weight))

    # Recall $$H^{(\ell + 1)} = \sigma\left(\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}H^{(\ell)}W^{(\ell)}\right)$$
    
    # Behind the scenes, GCNConv already (conceptually since it doesn't actually physically create the matrices in memory) does 
    # $ \tilde{A} = A + I $ 
    # $ \tilde{D}_{ii} = \sum_{j}{\tilde{A}_{ij}} $
    # and then
    # $S = \tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}$
    # $M = SX$
    # $Z = MW + \mathbb{1}(b^{(0)})^{T}$
    # and then that gets assigned to x in line 1. Thus the message passing is abstracted away.

    def encode(self, x: Tensor, edge_idx: Tensor, edge_weight: Tensor | None = None) -> Tensor:
        x = self.conv1(x, edge_idx, edge_weight=edge_weight) # $Z^{(0)} = SH^{(0)}W^{(0)} + \mathbb{1}(b^{(0)})^{T}$
        x = F.relu(x) # $H^{(1)} = ReLU(Z^{(0)})$
        x = F.dropout(x, p=self.dropout, training=self.training) # $\overline{H}^{(1)} = \text{dropout}(H^{(1)})$
        x = self.conv2(x, edge_idx, edge_weight=edge_weight) # $Z^{(1)} = S\overline{H}^{(1)}W^{(1)}+\mathbb{1}(b^{(1)})^{T} $
        x = F.relu(x) # $ H^{(2)} = ReLU(Z^{(1)}) $
        return F.dropout(x, p=self.dropout, training=self.training) # $\text{dropout}(H^{(2)})$

    def reset_parameters(self) -> None:
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.head.reset_parameters()
    