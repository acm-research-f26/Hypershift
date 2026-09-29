"""Equation-mapped reconstruction of THINK, NOT certified author code.

Curvature is -1. Equation 7 in the printed PDF is ambiguous; we use standard
Mobius scalar multiplication for the scalar distance modulation in Eq.14.
Eq.10 uses the normalized direction of z, following the cited HNN++ source.
Attention is signed/un-normalized as Eq.15 writes it (no invented softmax).
Temporal blocks use non-overlapping windows, as the nK -> n shape in Eq.12.
"""
import math
import torch
from torch import nn


def project(x):
    norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return x * ((1-1e-5)/norm).clamp_max(1)


def exp0(x):
    norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return project(x * (norm.tanh()/norm))


def log0(x):
    x = project(x)
    norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return x * (norm.atanh()/norm)


def mobius_add(x, y):
    xx, yy = x.square().sum(-1,keepdim=True), y.square().sum(-1,keepdim=True)
    xy = (x*y).sum(-1,keepdim=True)
    return project(((1+2*xy+yy)*x + (1-xx)*y)/(1+2*xy+xx*yy).clamp_min(1e-12))


def distance(x, y):
    return 2 * mobius_add(-x,y).norm(dim=-1).clamp_max(1-1e-5).atanh()


def beta(n):
    return math.exp(math.lgamma(n/2)+math.lgamma(.5)-math.lgamma((n+1)/2))


def beta_concat(x):
    """Concatenate last two axes [K,C], using Eq.11 dimensional scaling."""
    k,c=x.shape[-2:]
    return exp0(log0(x).flatten(-2)* (beta(k*c)/beta(c)))


class PoincareFC(nn.Module):
    """Eq.9-10 / HNN++ unidirectional hyperbolic fully connected layer."""
    def __init__(self, ins, outs):
        super().__init__()
        self.z=nn.Parameter(torch.randn(ins,outs)/math.sqrt(2*ins*outs))
        self.r=nn.Parameter(torch.zeros(outs))

    def forward(self,x):
        x=project(x)
        zn=self.z.norm(dim=0).clamp_min(1e-12)
        lam=2/(1-x.square().sum(-1,keepdim=True)).clamp_min(1e-7)
        # HNN++ uses z / ||z|| inside the directional inner product.
        argument=lam*(x @ (self.z/zn))*torch.cosh(2*self.r) - (lam-1)*torch.sinh(2*self.r)
        v=2*zn*torch.asinh(argument)
        w=torch.sinh(v.clamp(-15,15))
        return project(w/(1+torch.sqrt(1+w.square().sum(-1,keepdim=True))))


class Temporal(nn.Module):
    def __init__(self,ins,outs,kernel,hyperbolic):
        super().__init__()
        self.kernel,self.hyperbolic=kernel,hyperbolic
        self.fc=PoincareFC(ins*kernel,outs) if hyperbolic else nn.Linear(ins*kernel,outs)

    def forward(self,x):
        # [B,T,N,C] -> [B,T/K,N,K,C]
        b,t,n,c=x.shape
        if t % self.kernel:
            raise ValueError('Temporal length must be divisible by kernel.')
        windows=x.reshape(b,t//self.kernel,self.kernel,n,c).permute(0,1,3,2,4)
        return self.fc(beta_concat(windows) if self.hyperbolic else windows.flatten(-2))


class HypergraphAttention(nn.Module):
    def __init__(self,hidden,nodes,groups,hyperbolic=True,use_distance=True):
        super().__init__()
        if not groups or any(not group for group in groups):
            raise ValueError('Groups must be nonempty.')
        pairs=[(node,e) for e,group in enumerate(groups) for node in sorted(group)]
        self.register_buffer('node_ids',torch.tensor([p[0] for p in pairs]))
        self.register_buffer('edge_ids',torch.tensor([p[1] for p in pairs]))
        self.register_buffer('edge_degree',torch.bincount(self.edge_ids,minlength=len(groups)).float())
        self.nodes,self.edges=nodes,len(groups)
        self.hyperbolic,self.use_distance=hyperbolic,use_distance
        self.fc=PoincareFC(hidden,hidden) if hyperbolic else nn.Linear(hidden,hidden)
        self.a=nn.Parameter(torch.randn(hidden,1)/math.sqrt(hidden))

    def forward(self,x):
        shape=x.shape
        u=x.reshape(-1,self.nodes,shape[-1]); b,_,c=u.shape
        if self.hyperbolic:
            u=project(u)
            lam=2/(1-u.square().sum(-1,keepdim=True)).clamp_min(1e-7)
            num=u.new_zeros(b,self.edges,c).index_add_(1,self.edge_ids,(lam*u)[:,self.node_ids])
            den=u.new_zeros(b,self.edges,1).index_add_(1,self.edge_ids,(lam-1)[:,self.node_ids])
            # Eq.13: 1/2 Mobius-multiply the normalized Lorentz sum.
            z=exp0(.5*log0(project(num/den.clamp_min(1e-12))))
            pair=mobius_add(u[:,self.node_ids],z[:,self.edge_ids])
            alpha=exp0(log0(pair) @ self.a)
            if self.use_distance:
                dist=distance(u[:,self.node_ids],z[:,self.edge_ids]).unsqueeze(-1)
                alpha=exp0(dist*log0(alpha))
            values=log0(self.fc(z))[:,self.edge_ids]*alpha
            out=u.new_zeros(b,self.nodes,c).index_add_(1,self.node_ids,values)
            out=exp0(torch.relu(out))
        else:
            z=u.new_zeros(b,self.edges,c).index_add_(1,self.edge_ids,u[:,self.node_ids])/self.edge_degree[None,:,None]
            alpha=(u[:,self.node_ids]+z[:,self.edge_ids]) @ self.a
            if self.use_distance:
                alpha=alpha*(u[:,self.node_ids]-z[:,self.edge_ids]).norm(dim=-1,keepdim=True)
            values=self.fc(z)[:,self.edge_ids]*alpha
            out=torch.relu(u.new_zeros(b,self.nodes,c).index_add_(1,self.node_ids,values))
        return out.reshape(*shape)


class ThinkReconstruction(nn.Module):
    """Eq.17: temporal -> hypergraph attention -> temporal -> tangent output.

Variants mirror the described architectural interventions. Settings unresolved
by the paper remain reconstruction choices, not a one-to-one reproduction.
"""
    def __init__(self,features,nodes,groups,hidden=16,kernels=(2,2),variant='think'):
        super().__init__()
        if variant not in ('think','euclidean_temporal','no_distance','euclidean'):
            raise ValueError(variant)
        self.hyper_temporal=variant in ('think','no_distance')
        self.hyper_spatial=variant!='euclidean'
        self.first=Temporal(features,hidden,kernels[0],self.hyper_temporal)
        self.spatial=HypergraphAttention(hidden,nodes,groups,self.hyper_spatial,variant!='no_distance')
        self.last=Temporal(hidden,1,kernels[1],self.hyper_temporal)

    def forward(self,x):
        u=self.first(exp0(x) if self.hyper_temporal else x)
        if self.hyper_spatial and not self.hyper_temporal:
            u=exp0(u)
        u=self.spatial(u)
        if self.hyper_spatial and not self.hyper_temporal:
            u=log0(u)
        y=self.last(u)
        if y.shape[1]!=1:
            raise ValueError('Input lookback must equal the product of kernels.')
        return (log0(y) if self.hyper_temporal else y)[:,0,:,0]
