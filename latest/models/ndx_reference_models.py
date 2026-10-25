"""Reviewed author neural networks with explicit NASDAQ input/membership adapters.

Author sources remain in a commit-pinned cache. These are transferred networks,
not replications of the papers' datasets, feature sets or published performance.
"""
import ast
import importlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import yaml

ROOT=Path(__file__).resolve().parent
REFERENCES=ROOT/'.cache/architecture_references_20261024'


def definitions(path,names,context):
    tree=ast.parse(path.read_text(encoding='utf8'))
    body=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    if {n.name for n in body}!=set(names):raise ValueError('Author definitions missing')
    scope=dict(context);exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),scope)
    return scope


def master_class():
    context=dict(torch=torch,nn=nn,math=math,Linear=nn.Linear,Dropout=nn.Dropout,LayerNorm=nn.LayerNorm)
    names=['PositionalEncoding','SAttention','TAttention','Gate','TemporalAttention','MASTER']
    return definitions(REFERENCES/'MASTER/master.py',names,context)['MASTER']


def dynamic_class():
    path=REFERENCES/'dynamic_hypergraph/src/framework/layer.py'
    spec=importlib.util.spec_from_file_location('ndx_author_dynamic_layers',path)
    layers=importlib.util.module_from_spec(spec);spec.loader.exec_module(layers)
    return definitions(path.parent/'model.py',['Model'],vars(layers))['Model']


def prism_components():
    # PyTorch's fused Transformer eval assumes LayerNorm; the author replaces
    # it with RMSNorm. Run the author's explicit forward arithmetic instead.
    torch.backends.mha.set_fastpath_enabled(False)
    root=REFERENCES/'PRISM_VQ'
    wheel=next((ROOT/'.cache/reference_dependencies_20261024').glob('einops-0.8.1-*.whl'))
    for path in [root,wheel]:
        if str(path) not in sys.path:sys.path.insert(0,str(path))
    vq=importlib.import_module('module.autoencoder').VQVAE
    loading=importlib.import_module('module.bidirectional').LoadingGenerator
    heads=definitions(root/'trainer/train_ypred.py',['LatentValueHead','ReturnPredictor'],dict(nn=nn,torch=torch))
    return vq,loading,heads


def prism_config():
    config=yaml.safe_load((REFERENCES/'PRISM_VQ/configs/config.yaml').read_text())
    config['vqvae']['num_features']=25
    config['vqvae']['seq_len']=16
    config['vqvae']['decoder']['initial_T']=4
    config['predictor']['num_features']=25
    config['predictor']['pred_len']=16
    config['predictor']['k']=4
    return config


def wavelet_snapshot(groups,nodes):
    """Author snapshot math, with reviewed train-only correlation hyperedges.

    This explicitly replaces the author's five sector/lead-lag snapshots and
    therefore is a restricted dynamic-hypergraph adaptation, not full SPF.
    """
    incidence=np.zeros((nodes,len(groups)),dtype=np.float32)
    for j,group in enumerate(groups):incidence[list(group),j]=1
    adjacency=torch.from_numpy(((incidence@incidence.T)>0).astype('float32'))
    degree=adjacency.sum(1);edge=adjacency.sum(0)
    theta=torch.diag(degree.pow(-.5))@adjacency@torch.diag(edge.reciprocal())@adjacency.T@torch.diag(degree.pow(-.5))
    eig,u=torch.linalg.eigh(torch.eye(nodes)-theta,UPLO='U')
    wave=u@torch.diag(torch.exp(-eig))@u.T;inverse=u@torch.diag(torch.exp(eig))@u.T
    wave[wave<1e-5]=0;inverse[inverse<1e-5]=0
    return {'wavelets':wave,'wavelets_inv':inverse}


class MasterAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.network=master_class()(25,256,4,2,.5,.5,25,35,5.)

    def forward(self,features,market,eligible):
        output=features.new_zeros(features.shape[0])
        if eligible.any():
            selected=torch.cat([features[eligible],market.expand(int(eligible.sum()),-1,-1)],dim=-1)
            output=output.masked_scatter(eligible,self.network(selected))
        return output,features.new_zeros(())


class DynamicAdapter(nn.Module):
    def __init__(self,nodes,snapshot):
        super().__init__()
        self.network=dynamic_class()([snapshot],nodes,25,32,4,16,64,.1,'sum')
        self.register_buffer('wavelets',snapshot['wavelets'].clone())
        self.register_buffer('wavelets_inv',snapshot['wavelets_inv'].clone())

    def forward(self,features,market,eligible):
        m=self.network;mask=eligible[:,None]
        clean=torch.where(eligible[:,None,None],features,torch.zeros_like(features))
        x=m.positional_encoding(m.feature_embedding(clean));x=m.individual_attn(x)
        x=m.mixing_layer(torch.cat([x,m.intensity_func(x)],dim=-1));x,_=m.temporal_attn(x)
        x=torch.where(mask,x,torch.zeros_like(x))
        # Mask both sides and intermediate layers, while keeping stable ID filters.
        pair=mask&mask.T
        snapshots=[dict(wavelets=self.wavelets*pair,wavelets_inv=self.wavelets_inv*pair)]
        h1=F.leaky_relu(m.whconv1(x,snapshots,0),.1);h1=m.dropout(h1)*mask
        h2=F.leaky_relu(m.whconv2(h1,snapshots,0),.1)*mask
        graph=h2*m.params_of_snapshot[0]
        dynamic,_=m.dynamic_graph_layer(x)
        weights=torch.softmax(m.beta,dim=0)
        output=m.mlp(weights[0]*x+weights[1]*dynamic+weights[2]*graph)
        return torch.where(eligible,output,torch.zeros_like(output)),features.new_zeros(())


class PrismAdapter(nn.Module):
    def __init__(self):
        super().__init__();vq,loading,heads=prism_components();self.config=prism_config()
        self.vq=vq(self.config)
        self.z_prior_norm=nn.LayerNorm(13)
        self.loadings=loading(self.config)
        self.latent_value_head=heads['LatentValueHead'](128,128)
        self.return_predictor=heads['ReturnPredictor'](13,128,True)
        self.stage_two=False

    def freeze_representation(self):
        self.stage_two=True
        for parameter in self.vq.parameters():parameter.requires_grad_(False)
        self.vq.eval()

    def train(self,mode=True):
        super().train(mode)
        # Author freeze_vqvae() explicitly requests eval. Preserve that across
        # outer train() calls: the quantizer has in-forward mutable state.
        if self.stage_two:self.vq.eval()
        return self

    def pretrain_loss(self,features,prior,future,eligible):
        return self.vq(features[eligible],prior.expand(len(features),-1)[eligible],future[eligible])[3]

    def forward(self,features,prior,eligible):
        selected=features[eligible];p=prior.expand(len(features),-1)[eligible]
        z=self.vq.spatial_encoder(self.vq.revin(selected,mode='norm'))
        z,_,_=self.vq.quantizer(z);z=z.detach()
        alpha,bp,bl,aux=self.loadings(selected,z)
        score=self.return_predictor(alpha,bp,bl,self.z_prior_norm(p),self.latent_value_head(z))
        output=features.new_zeros(len(features)).masked_scatter(eligible,score)
        return output,3*torch.log1p(F.relu(aux)/3)


def make_reference(kind,nodes,snapshot=None):
    if kind=='master':return MasterAdapter()
    if kind=='dynamic_restricted':return DynamicAdapter(nodes,snapshot)
    if kind=='prism':return PrismAdapter()
    raise ValueError(kind)
