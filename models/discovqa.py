import os

import torch
import torch.nn as nn

from models.swin_video import SwinVideo
from models.components.transformer import *

def create_pretrained_swint(pretrained_path=os.path.join(os.environ['SCRATCH_DIR'],
                                                         'hub/swin_tiny_patch244_window877_kinetics400_1k.pth')):
    model = SwinVideo()
    
    state_dict = torch.load(pretrained_path)['state_dict']

    model.load_state_dict(state_dict)

    model.cls_head = nn.Identity()

    del state_dict
    return model
    
class TemporalDistortionAware(nn.Module):
    def __init__(self, indim=4224, hiddim=1024, outdim=1) -> None:
        super(TemporalDistortionAware, self).__init__()

        self.l1 = nn.Linear(indim, hiddim)
        self.act = nn.GELU()
        self.l2 = nn.Linear(hiddim, outdim)

    def forward(self, x):
        return self.l2(self.act(self.l1(x)))
    
class TSF(nn.Module):
    def __init__(self, sr=8) -> None:
        super(TSF, self).__init__()
        self.sr = sr
    
    def forward(self, x):
        # input shape: [nbatch, sequence, features]
        seq_len = x.size(1) - (x.size(1)%self.sr)
        indices = torch.randint(0, self.sr, size=(seq_len//self.sr,)) + torch.arange(0, seq_len, self.sr)
        return x[:, indices]
    
class STDE(nn.Module):
    def __init__(self, sample_rate=8) -> None:
        super(STDE, self).__init__()

        self.swin_t = create_pretrained_swint()
        self.tsf = TSF(sample_rate)

    def forward(self, x):
        # input shape: [nbatch, channles=3, sequene, height, width  ]
        # output shape: [nbatch, channels, sequence, height, width], list([nbatch, sequence, channels])
        _, x = self.swin_t(x)
        # output shape: [nbatch, sequence, channels]
        x = torch.cat(x, dim=-1)
        # output shape: [nbatch, sequence, features]
        x1 = x - x.roll(-1, 1)
        x1[:,-1] = torch.zeros_like(x1[:,-1])
        # output shape: [nbatch, sequence, 2*features]
        return self.tsf(torch.cat((x, x1), dim=-1))
    
class TCT(nn.Module):
    def __init__(self, in_dim=4224, d_model=768, N=4, dropout=0.0, num_heads=1) -> None:
        super(TCT, self).__init__()

        # not mentioned in paper but exists in the Fig. 2
        self.l1 = nn.Linear(in_dim, d_model)
        self.l2 = nn.Linear(d_model, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        # transformer section
        self.encoder = ResidualTransformerEncoder(N, d_model, dropout)
        self.decoder = TransformerDecoder(d_model, num_heads, dropout=dropout)

    def forward(self, x):
        # input shape: [nbatch, sequence, features]
        # output shape: [nbatch, sequence, features]
        x = self.l2(self.dropout(self.act(self.l1(x))))
        t_pe = x
        # output shape: [nbatch, sequence, features]
        x = self.encoder(x)
        # output shape: [nbatch, sequence, 1]
        return self.decoder(t_pe, x)
    
class DisCoVQA(nn.Module):
    t_dim = 2112
    def __init__(self, d_model=512, num_heads=1, sample_rate=8, dropout=0.0) -> None:
        super(DisCoVQA, self).__init__()

        # STDE section in Fig 2. of the paper
        self.stde = STDE(sample_rate)
        # TCT section in Fig 2. of the paper
        self.tct = TCT(self.t_dim*2, d_model, 4, dropout, num_heads)
        # temporal distortion
        self.temp_dist = TemporalDistortionAware(self.t_dim*2)

    def forward(self, x):
        # input shape [nbatch, 3, sequence, height, width]
        # output shape [nbatch, sequence, features]
        x_d = self.stde(x)
        # output shape [nbatch, sequence]
        x = self.tct(x_d).squeeze(-1)
        # output shape [nbatch, sequence]
        x_d = self.temp_dist(x_d).squeeze(-1)
        # output shape [nbatch, sequence]
        x = (x_d * x) + x_d
        # output shape [nbatch,]
        return x.mean(dim=-1)
    

class VQEGSuggestion(nn.Module):
    def __init__(self, min_s, max_s) -> None:
        super(VQEGSuggestion, self).__init__()

        from torch.nn.parameter import Parameter

        self.min_s = Parameter(torch.tensor(min_s).float())
        self.max_s = Parameter(torch.tensor(max_s).float())
        self.mean_t = Parameter(torch.tensor(0).float())
        self.std_t = Parameter(torch.tensor(1).float())


    def forward(self, x):
        # input shape: [nbatch,]
        x = 1. + torch.exp((x-self.mean_t)/self.std_t)
        # output shape: [nbatch,]
        return self.min_s + ((self.max_s - self.min_s)/x)
    
class SigmoidReformulation(nn.Module):
    def __init__(self, min_s, max_s) -> None:
        super(SigmoidReformulation, self).__init__()

        from torch.nn.parameter import Parameter

        self.g1 = Parameter(torch.tensor(1).float())
        self.g2 = Parameter(torch.tensor(0).float())
        self.g3 = Parameter(torch.tensor(max_s-min_s).float())
        self.g4 = Parameter(torch.tensor(min_s).float())

    def forward(self, x):
        x = torch.sigmoid(self.g1*x+self.g2)
        return self.g3*x + self.g4
