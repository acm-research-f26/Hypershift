import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):

        return x + self.pe[:x.shape[1], :]

class IntraStockAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.5):
        super(IntraStockAttention, self).__init__()

        assert d_model % num_heads == 0, f'd_model{d_model} cannot fully divided by num_heads{num_heads}!'
        self.d_model = d_model
        self.num_heads = num_heads
        
        # input LayerNorm
        self.input_norm = nn.LayerNorm(d_model, eps=1e-5)

        # attention LayerNorm
        self.attn_norm = nn.LayerNorm(d_model, eps=1e-5)

        # attention dropout
        if dropout > 0:
            self.attn_dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_heads)])

        # embedding for attention
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.head = nn.Linear(d_model, d_model, bias=False)

        # feed forward after attention
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout)
        )

    def reshape_to(self, x):

        self.num_stocks, self.sequence_len, _ = x.size() # (S, T, D)
        self.dim = self.d_model // self.num_heads # D/h

        x = x.reshape(self.num_stocks, self.sequence_len, self.num_heads, self.dim) # (S, T, h, D/h)
        x = x.permute(0,2,1,3) # (S, h, T, D/h)
        x = x.reshape(self.num_stocks*self.num_heads, self.sequence_len, self.dim) # (S*h, T, D/h)

        return x # (S*h, T, D/h)
    
    def reshape_from(self, x):

        """
        attn: (S*h, T, D/h)
        """
        x = x.reshape(self.num_stocks, self.num_heads, self.sequence_len, self.dim) # (S, h, T, D/h)
        x = x.permute(0,2,1,3) # (S, T, h, D/h)
        x = x.reshape(self.num_stocks, self.sequence_len, self.num_heads*self.dim) # (S, T, D)

        return x # (S, T, D)
    
    def generate_mask(self, input):

        num_stocks = input.shape[0]
        sequence_len = input.shape[1]

        mask = torch.triu(torch.ones((sequence_len, sequence_len), dtype=torch.int16, device=input.device), diagonal=1)
        mask = mask.bool().unsqueeze(0).expand(num_stocks, -1, -1)

        return mask # (S, T, T)

    def attention(self, query, key, value, mask=None):

        """
        query: (S*h, T, D/h) 
        key: (S*h, T, D/h) 
        value: (S*h, T, D/h) 
        """
        scaled = math.sqrt(self.d_model / self.num_heads)
        attn_score = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(scaled) # (S*h, T, T) 
        if mask is not None:
            attn_score = attn_score.masked_fill(mask, float('-inf'))
        attn_score = torch.softmax(attn_score, dim=-1)

        attn_score = attn_score.reshape(self.num_heads, self.num_stocks, self.sequence_len, self.sequence_len) # (h, S, T, T)
        attn_score_list = []
        for i in range(self.num_heads):
            attn_score_list.append(self.attn_dropout[i](attn_score[i]))
        attn_score = torch.stack(attn_score_list)  # (h, S, T, T)
        attn_score = attn_score.reshape(self.num_heads*self.num_stocks, self.sequence_len, self.sequence_len) # (S*h, T, T)
        
        attn_out = torch.matmul(attn_score, value) # (S*h, T, D/h)

        return attn_out # (S*h, T, D/h) 
    
    def forward(self, x, mask=None):
        """
        input shape: (S, T, D)
        """
        x = self.input_norm(x)
        q = self.query(x) # (S,T,D)
        k = self.key(x) # (S,T,D)
        v = self.value(x) # (S,T,D)

        # reshape to multi-head form
        q = self.reshape_to(q) # (S*h, T, D/h)
        k = self.reshape_to(k) # (S*h, T, D/h)
        v = self.reshape_to(v) # (S*h, T, D/h)

        mask = self.generate_mask(x) # (S, T, T)

        # multi-head self-attention
        if mask is not None:
            mask = mask.repeat(self.num_heads, 1, 1) # (S*h, T, T)

        attn = self.attention(q, k, v, mask) # (S*h, T, D/h)
        
        # reshape from multi-head form
        attn = self.reshape_from(attn) # (S, T, D)
        attn = self.head(attn) # (S, T, D)

        # feed forward after attention
        x = attn + x # residual connection
        x = self.attn_norm(x)
        x = self.ffn(x) + x # (S, T, D)

        return x # (S, T, D)

    
class IntensityFunction(nn.Module):
    def __init__(self, d_model):
        super(IntensityFunction, self).__init__()

        self.converge_linear = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU()
        )

        self.start_linear = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU()
        )

        self.decay_linear = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Softplus(beta=10.0)
        )

        self.softplus = nn.Softplus()
    
    def forward(self, x):

        num_stocks, sequence_len, d_model = x.shape

        mu = self.converge_linear(x) # (S, T, D)
        eta = self.start_linear(x) # (S, T, D)
        gamma = self.decay_linear(x) # (S, T, D)

        delta_t = torch.arange(sequence_len, device=x.device).unsqueeze(1) - torch.arange(sequence_len, device=x.device).unsqueeze(0) # (T, T)
        delta_t = delta_t.clamp(min=0).to(torch.float32)
        delta_t += torch.eye(sequence_len, device=x.device) * 1e-10
        delta_t = delta_t.unsqueeze(0).unsqueeze(-1).expand(num_stocks, -1, -1, d_model) # (S, T, T, D)

        time_decay = torch.exp(-gamma.unsqueeze(1) * delta_t) # (S, T, T, D)

        mask = torch.tril(torch.ones(sequence_len, sequence_len, device=x.device), diagonal=0).bool()
        time_decay = time_decay.masked_fill(~mask.unsqueeze(0).unsqueeze(-1), 0.0) # (S, T, T, D)

        influence_sum = time_decay.sum(dim=2) # (S, T, D)

        intensity = self.softplus(mu + (eta - mu) * influence_sum) # (S, T, D)

        return intensity
    

class TemporalAttention(nn.Module):
    def __init__(self, d_model, dropout=0.5):
        super(TemporalAttention, self).__init__()

        self.d_model = d_model

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product(self, q, k, v):

        scaled = math.sqrt(self.d_model)
        attn_score = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(scaled) # (S, 1, T)

        attn_score = torch.softmax(attn_score, dim=-1) 
        attn_score = self.dropout(attn_score)

        attn_out = torch.matmul(attn_score, v) # (S, 1, D)

        return attn_out, attn_score # (S, 1, D), (S, 1, T)
    
    def forward(self, x):

        """
        inputs shape: (S, T, D)
        """
        q = self.query(x)[:,-1:,:] # last vector of sequence # (S, 1, D)
        k = self.key(x) # (S, T, D)
        v = self.value(x) # (S, T, D)

        out, attn_score = self.scaled_dot_product(q, k, v) # (S, 1, D), (S, 1, T)
        attn_score = attn_score.squeeze(1) # (S, T)
        out = out.squeeze(1) # (S, D)
        out = self.layer_norm(out)
        
        return out, attn_score # (S, D), (S, T)


class WaveletHypergraphConvolution(nn.Module):
    def __init__(self, d_model, num_stocks, K1=2, K2=2, approx=False):
        super(WaveletHypergraphConvolution, self).__init__()

        self.d_model = d_model
        self.num_stocks = num_stocks
        self.K1 = K1 # the polynomial order of the approximation for Theta
        self.K2 = K2 # the polynomial order of the approximation for Theta_t
        self.approx = approx

        self.weight_matrix = nn.Parameter(torch.Tensor(self.d_model, self.d_model))
        self.diagonal_weight_filter = nn.Parameter(torch.Tensor(self.num_stocks))
        self.par = nn.Parameter(torch.Tensor(self.K1 + self.K2)) # learnable weights for each polynomially approximation by Stone-Weierstrass theorem
        
        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.weight_matrix)
        nn.init.uniform_(self.diagonal_weight_filter, 0.99, 1.01)
        nn.init.uniform_(self.par, 0, 0.99)

    def forward(self, x, snapshot, snapshot_idx):

        """
        inputs shape: (B, S, T*h)
        """

        diagonal_weight_filter = torch.diag(self.diagonal_weight_filter) 

        if self.approx:

            Theta = snapshot[snapshot_idx]['Theta'] # (S, S)
            Theta_t = Theta.transpose(0,1) # (S, S) 
            
            poly = 0
            Theta_mul = torch.eye(self.num_stocks)
            for i in range(self.K1+1): # 0~2
                poly += self.par[i] * Theta_mul
                Theta_mul = Theta_mul @ Theta

            poly_t = 0
            Theta_mul = torch.eye(self.num_stocks)
            for i in range(self.K1+1, self.K1+1+self.K2+1): # 3~5
                poly_t += self.par[i] * Theta_mul
                Theta_mul = Theta_mul @ Theta_t

            x = poly @ diagonal_weight_filter @ poly_t @ x @ self.weight_matrix

        else:
            wavelets = snapshot[snapshot_idx]['wavelets'].to(x.device) # (S, S)
            wavelets_inverse = snapshot[snapshot_idx]['wavelets_inv'].to(x.device) # (S, S)
            x = wavelets @ diagonal_weight_filter @ wavelets_inverse @ x @ self.weight_matrix
                # (S, S) @ (S, S) @ (S, S) @ (S, D) @ (D, D) -> (S, D)
        return x
   

class DynamicGraphLearning(nn.Module):
    def __init__(self, d_model, num_edges):
        super(DynamicGraphLearning, self).__init__()

        self.d_model = d_model
        self.num_edges = num_edges

        self.W = nn.Linear(d_model, num_edges, bias=False)
        self.U = nn.Linear(num_edges, num_edges, bias=False)

        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()

    def forward(self, embeddings):
        """
        Args:
            embeddings: Embeddings of all stocks (S, D)
        Returns:
            dynamic_representations: Representations for dynamic hypergraph (S, D)
        """

        # generate the dynamic incidence matrix
        dynamic_incidence = torch.softmax(self.W(embeddings), dim=-1) # (S, E)
        
        # calculate the dynamic hypergraph embedding
        hyperedge_embeddings = dynamic_incidence.T @ embeddings # (E, D)
        hyperedge_embeddings = self.activation(self.U.weight @ hyperedge_embeddings) + hyperedge_embeddings
        
        # dynamic hypergraph representations (with residual connections)
        dynamic_representations = self.activation(dynamic_incidence @ hyperedge_embeddings) # S x D
        dynamic_representations = self.norm(dynamic_representations + embeddings) # S x D

        return dynamic_representations, dynamic_incidence
    

class RepresentationSelection(nn.Module):
    def __init__(self, d_model):
        super(RepresentationSelection, self).__init__()

        # generate the selection weight
        self.linear1 = nn.Linear(int(3*d_model), 3)
        self.linear2 = nn.Linear(int(3*d_model), 3)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, stock_repre, dynamic_repre, hyper_repre):

        """
        stock_repre: outputs of temporal attention (S, D)
        dynamic_repre: outputs of dynamic graph learning (S, D)
        hyper_repre: outputs of wavelet hypergraph convolution (S, D)
        """

        stack_repre = torch.stack([stock_repre, dynamic_repre, hyper_repre], dim=-2) # (S, 3, D)
        flattend_repre = torch.flatten(stack_repre, -2) # (S, 3D)

        selelction_weight = (self.sigmoid(self.linear1(flattend_repre)) * self.linear2(flattend_repre)) # (S, 3)
        selelction_weight = torch.softmax(selelction_weight, dim=-1).unsqueeze(-2) # (S, 1, 3)
        fused_repre = torch.matmul(selelction_weight, stack_repre).squeeze(-2) # (S, D)

        return fused_repre # (S, D)



class MLP(nn.Module):
    def __init__(self, d_model, fusion_mode):
        super(MLP, self).__init__()

        if (fusion_mode == "sum") or (fusion_mode == "adaptive"):

            self.linear_hidden1 = nn.Linear(d_model, d_model)
            self.relu_hidden = nn.ReLU()
            self.norm_hidden = nn.LayerNorm(d_model)
            self.linear_hidden2 = nn.Linear(d_model, 1)

        elif fusion_mode == "cat":

            self.linear_hidden1 = nn.Linear(int(3*d_model), d_model)
            self.relu_hidden = nn.ReLU()
            self.norm_hidden = nn.LayerNorm(d_model)
            self.linear_hidden2 = nn.Linear(d_model, 1)

    def forward(self, x):

        """
        inputs shape: (S, D) or (S, 3D)
        """

        x = self.linear_hidden1(x) # (S, D)
        x = self.relu_hidden(x)
        x = self.norm_hidden(x)
        x = self.linear_hidden2(x).squeeze(-1) # (S)

        return x # (S)