import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
from layers.S4 import S4
import math
import numpy as np
from scipy import signal
from scipy import linalg as la
from scipy import special as ss
from utils import unroll
from utils.op import transition
import pdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, configs, N=256, N2=32):
        super(Model, self).__init__()
        self.configs = configs
        # self.modes = configs.modes
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.layers = configs.e_layers
        self.modes1 = min(32,self.pred_len//2)
    
        self.multiscale = [1, 2, 4]
        if configs.enc_in>800:
            self.multiscale =[1]
        self.d_model = configs.enc_in
        # self.decomps = [series_decomp(i) for i in self.multiscale]
        self.S4 = nn.ModuleList([S4(d_model=self.d_model,
            d_state=64,
            l_max=self.seq_len) for _ in range(self.layers)])
        self.output=nn.Linear(self.seq_len, self.pred_len)


    def forward(self, x_enc, x_mark_enc, x_dec_true, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        
        B, L, E = x_enc.shape
        seq_len = self.seq_len
        x_enc = x_enc.transpose(1,2)
        for i in range(0, self.layers):
            legt = self.S4[i]
            x_enc = legt(x_enc)[0]
        x_dec = self.output(x_enc).permute(0,2,1)
        if self.output_attention:
            return x_dec, None
        else:
            return x_dec  # [B, L, D]


if __name__ == '__main__':
    class Configs(object):
        ab = 2
        modes1 = 100
        seq_len = 96*8
        label_len = 0
        pred_len = 720
        output_attention = True
        enc_in = 7
        dec_in = 7
        d_model = 16
        embed = 'timeF'
        dropout = 0.05
        freq = 'h'
        factor = 1
        n_heads = 8
        d_ff = 16
        e_layers = 2
        d_layers = 1
        moving_avg = 25
        c_out = 1
        activation = 'gelu'
        wavelet = 0

    configs = Configs()
    model = Model(configs).to(device)

    enc = torch.randn([3, configs.seq_len, configs.enc_in]).cuda()
    enc_mark = torch.randn([3, configs.seq_len, 4]).cuda()

    dec = torch.randn([3, configs.label_len+configs.pred_len, configs.dec_in]).cuda()
    dec_mark = torch.randn([3, configs.label_len+configs.pred_len, 4]).cuda()
    out=model.forward(enc, enc_mark, dec, dec_mark)
    print('input shape',enc.shape)
    print('output shape',out[0].shape)
    a = 1