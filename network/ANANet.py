from copy import deepcopy
from pathlib import Path
from numpy.core.numeric import outer
from torch.nn import functional as F
import torch
from torch import nn
import numpy as np
import os
import copy
from torch.nn.parameter import Parameter
def stack_hstar(hstars_list):
    b,_,n=hstars_list[0].shape
    H_star=torch.stack(hstars_list,dim=1).reshape(b,-1,n,1)
    return H_star
def l2_normalize(x,ratio=1.0,axis=1):
    norm=torch.unsqueeze(torch.clamp(torch.norm(x,2,axis),min=1e-6),axis)
    x=x/norm*ratio
    return x
def attention(query, key, value):#
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5#
    prob = torch.nn.functional.softmax(scores, dim=-1)#
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob #

def MLP(channels: list, do_bn=True):#
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)
class Encoder(nn.Module):
    def __init__(self,layers):
        super().__init__()
        self.encoder=MLP(layers)
        nn.init.constant_(self.encoder[-1].bias, 0.0)
    def forward(self,xs):
        return self.encoder(xs)

                
class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):#
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads#
        self.num_heads = num_heads

        self.w_qs = nn.Conv1d(d_model, self.dim * self.num_heads, kernel_size=1)
        self.w_ks = nn.Conv1d(d_model, self.dim * self.num_heads, kernel_size=1)
        self.w_vs = nn.Conv1d(d_model, self.dim * self.num_heads, kernel_size=1)
        self.fc = nn.Conv1d(d_model, self.dim * self.num_heads, kernel_size=1)

    def forward(self, query, key, value,sc_use=False):
        batch_dim = query.size(0)

        query = self.w_qs(query).view(batch_dim, self.dim, self.num_heads, -1)
        key   = self.w_ks(key).view(batch_dim, self.dim, self.num_heads, -1)
        value = self.w_ks(value).view(batch_dim, self.dim, self.num_heads, -1)

        x,prob= attention(query, key,value)
        return self.fc(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1)),prob

class ana_block(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.config=cfg
        
        channel_num=self.config['heads']*self.config['multifeature_dim']
        self.channel_num=channel_num
        self.atten=MultiHeadedAttention(self.config['heads'],self.channel_num)

        conv_channel_base=2

        if self.config['normalization_type']=='layer':
            self.layernorm = nn.LayerNorm(channel_num)
        
        self.merge=nn.Sequential()
        if self.config['hstars_use']:
            self.zoom=torch.nn.Parameter(2*torch.ones((1,self.config['heads'],1)), requires_grad=True)
            self.offset=torch.nn.Parameter(torch.ones((1,self.config['heads'],1)), requires_grad=True)
            self.neighborconv=MLP([self.config['heads']]+[32,64,128]+[self.config['houtdim']])  
            self.n_norm=nn.LayerNorm(self.config['houtdim'])
        self.conv=MLP([conv_channel_base*channel_num,channel_num,channel_num]) 
                    
    def forward(self,inputs):
        '''
        input: tensor b(hc)n
        output: tensor b(hc)n
        '''
        c_feat,prob=self.atten(inputs,inputs,inputs)
        
        H_star=self._sc_process(prob)

        if self.config['hstars_use']:
            confidence=torch.reciprocal(1+((-self.zoom*H_star.squeeze(-2)).exp()))
            n_feat=self.neighborconv(confidence)#1/(1+w*exp(-x))
            n_feat=self._normalization(n_feat,self.n_norm)
            inputs=inputs+n_feat


        final_feat=self.conv(torch.cat((inputs,c_feat),dim=1))
        
        final_feat=self._normalization(final_feat)
        
        return final_feat
        
    def _sc_process(self,prob):
        
        if self.config['hstars_type']==1:
            H_star=torch.sum(prob,dim=-2,keepdims=True)
            H_star=H_star-(H_star**2)/H_star.shape[-1]
            H_star=1.414*H_star
        
        elif self.config['hstars_type']==3:#sum-sum(^2)
            H_star=torch.sum(prob,dim=-2,keepdims=True)-torch.sum(prob**2,dim=-2,keepdims=True)
            
        elif self.config['hstars_type']==6:# sqrt(sum^2+sum(^2))
            L=torch.diag_embed(torch.sum(prob,dim=-2))-torch.einsum('bhmn,bhmk->bhnk',prob,prob)
            H_star=torch.sum(L**2,dim=-2,keepdims=True)
            H_star=torch.sqrt(H_star)
        
        return H_star
    def _normalization(self,inputs,normal_fn=None):
        '''
        input: tensor BCN
        output: tensor BCN
        '''
        if self.config['normalization_type']=='none':
            return inputs

        if normal_fn is None:
            if self.config['normalization_type']=='layer':
                c_desc=inputs.permute(0,2,1)
                c_desc=self.layernorm(c_desc)
                c_desc=c_desc.permute(0,2,1)
                return c_desc
            else:
                exit(1)
        else:
            if self.config['normalization_type']=='layer':
                c_desc=inputs.permute(0,2,1)
                c_desc=normal_fn(c_desc)
                c_desc=c_desc.permute(0,2,1)
                return c_desc
            else:
                exit(1)
        
  

class ANANet(nn.Module):
    default_config={ 
        'encoder':{
            'origin_dim':4,
            'parameter_encoder': [16,32,64,128],
            'parameter_dim':128,
        }
        ,
        'c_block':{
            'heads':4,
            'multifeature_dim':32,
            'hstars_type':1,
            'normalization_type':'layer',#'gainsinglehead','layer','none'
        },
       
        'iter_num':5,  
        }
    def __init__(self,**cfg):
        super().__init__()
        self.config={**self.default_config,**cfg}

        self.encoder=Encoder([self.config['encoder']['origin_dim']]+\
                              self.config['encoder']['parameter_encoder']+\
                              [self.config['encoder']['parameter_dim']])
        
        self.iter_layers=nn.ModuleList([
            ana_block(self.config['c_block'])
            for _ in range(self.config['iter_num'])])
        
        self.final_conv=nn.Sequential(nn.Conv1d(self.config['encoder']['parameter_dim'], 1, kernel_size=1))

        print(self.config)
    
    def forward(self,xs,return_intermediate=False):
        
        inputs=xs.transpose(1,2)#b4n
        c_desc=self.encoder(inputs)


            
        
        for iter_layer in self.iter_layers:
            c_desc=iter_layer(c_desc)
       
      
        out=self.final_conv(c_desc)


        out = out.view(out.size(0), -1)
        w = torch.tanh(out)
        w = torch.relu(w)
        return out, w
