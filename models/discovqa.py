import math

import torch
import torch.nn as nn

from models.swin_video import create_pretrained_swint
from models.components.transformer import *
    
class MLP(nn.Module):
    def __init__(self, indim, hiddim, outdim, dropout=0.1) -> None:
        super(MLP, self).__init__()

        self.l1 = nn.Linear(indim, hiddim)
        self.act = nn.GELU()
        self.dp0 = nn.Dropout(dropout)
        self.l2 = nn.Linear(hiddim, outdim)
        self.dp1 = nn.Dropout(dropout)

    def forward(self, x):
        return self.dp1(self.l2(self.dp0(self.act(self.l1(x)))))
    
class STDE(nn.Module):
    def __init__(self, pretrained_path='hub/swin_tiny_patch244_window877_kinetics400_1k.pth',
                 indim=4224, hiddim=2048, outdim=1, tsf_window=8) -> None:
        super(STDE, self).__init__()

        self.swin_t = create_pretrained_swint(pretrained_path)
        self.mlp = MLP(indim, hiddim, outdim)
        self.tsf_window = tsf_window

    def forward(self, x):
        # input shape: [nbatch, channles=3, sequene, height, width  ]
        # output shape: [nbatch, channels, sequence, height, width], list([nbatch, sequence, channels])
        _, x = self.swin_t(x)
        # output shape: [nbatch, sequence, channels]
        x = torch.cat(x, dim=-1)
        # output shape: [nbatch, sequence, features]
        x = self.temporal_difference(x)
        # output shape: [nbatch, sequence, 768]
        # The following line is not mentioned in the paper. But 
        # it is a workaround to make up for the final aggregation
        x = TCT.temporal_sampling_on_features(x, self.tsf_window)
        return self.mlp(x), x
    
    def temporal_difference(self, ten):
        diff = ten - ten.roll(-1, 1)
        diff[:, -1, :] = 0
        return torch.cat((ten, diff), dim=-1)
    
class TCT(nn.Module):
    def __init__(self, indim=4224, hiddim=2048, dmodel=768, outdim=16,
                 enc_layers=4, dec_layers=2, num_heads=12, tsf_window=8, 
                 dropout=0.1) -> None:
        super(TCT, self).__init__()

        # not mentioned in paper but exists in the Fig. 2
        self.mlp0 = MLP(indim, hiddim, dmodel, dropout)
        
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(dmodel, 
                                                                    num_heads, 
                                                                    hiddim, 
                                                                    dropout, 
                                                                    batch_first=True), 
                                         enc_layers)
        self.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(dmodel,
                                                                    num_heads,
                                                                    hiddim,
                                                                    dropout,
                                                                    batch_first=True),
                                         dec_layers)
        self.mlp1 = MLP(dmodel, hiddim, outdim, dropout)
        
        self.tsf_window = tsf_window

    def forward(self, x):
        # input shape: [nbatch, sequence, features]
        # x = self.temporal_sampling_on_features(x)
        Tpe = self.mlp0(x)
        
        Ten = Tpe + self.enc(Tpe)
        
        Tavg = self.temporal_average_pooling(Tpe)
        
        Tout = Ten + self.dec(Ten, Tavg)
        w = self.mlp1(Tout)
        return w
    
    @staticmethod
    def temporal_sampling_on_features(x, tsf_window=8):
        # According to the paper, TSF is only applied to input features of TCT 
        # and not to the output of STDE. However, due to descrapency in final 
        # aggregated features, we apply TSF to the output of STDE as well.
        # input shape: [nbatch, sequence, features]
        B, D, C = x.shape
        num_windows = math.ceil(D / tsf_window)

        # Create a range for each start of the window
        start_indices = torch.arange(0, D, step=tsf_window, device=x.device)
        
        # Generate random indices within each window
        random_offsets = torch.randint(0, tsf_window, size=(B, num_windows), device=x.device)
        # Ensure indices are unique and within bounds for each window segment
        indices = torch.clamp(start_indices + random_offsets, max=D - 1)

        # Indexing requires gathering along the sequence dimension
        sampled = torch.gather(x, 1, indices.unsqueeze(-1).expand(-1, -1, C))
        return sampled
    
    def temporal_average_pooling(self, x):
        return x.mean(dim=1, keepdim=True)
    
class DisCoVQA(nn.Module):
    t_dim = 2112
    def __init__(self, pretrained_path='hub/swin_tiny_patch244_window877_kinetics400_1k.pth', 
                 indim=4224, hiddim=2048, dmodel=768, outdim=1, max_s=100, min_s=0,
                 enc_layers=4, dec_layers=2, num_heads=12, tsf_window=8, 
                 dropout=0.1) -> None:
        super(DisCoVQA, self).__init__()

        # STDE section in Fig 2. of the paper
        self.stde = STDE(pretrained_path, indim, hiddim, outdim, tsf_window)
        # TCT section in Fig 2. of the paper
        self.tct = TCT(indim, hiddim, dmodel, outdim, enc_layers, dec_layers, 
                       num_heads, tsf_window, dropout)
        self.lin = SigmoidReformulation(min_s, max_s)

    def forward(self, x):
        # input shape [B, C=3, D, H, W]
        di, x = self.stde(x)
        wi = self.tct(x)

        q = torch.mean(di.squeeze()+di.squeeze()*wi.squeeze(), dim=1)
        return self.lin(q)
    

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
