import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
#from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
#from layers.S4 import S4
import math
import numpy as np
from scipy import signal
from scipy import linalg as la
from scipy import special as ss
from utils import unroll
from utils.op import transition
#import pdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

class Model(nn.Module):
    def __init__(self, configs,size=25):
        super(Model,self).__init__()
        
        self.lstm = nn.LSTM(configs.enc_in, configs.d_model, configs.e_layers)
        self.mlp=nn.Linear(configs.d_model, configs.dec_in)
        self.configs = configs

    def forward(self, x,enc_mark, dec, dec_mark):
        x=x.transpose(0,1)
        #print(x.shape)
        output, (hn, cn) = self.lstm(x, (self.h0, self.c0))
        output = self.mlp(output)
        if self.configs.output_attention:
            return output.transpose(0,1),(hn,cn)
        return output.transpose(0,1)
    
    def init_hidden(self, BSIZE, LEN):
        self.h0=torch.randn(self.configs.e_layers, self.configs.batch_size,self.configs.d_model).cuda()
        self.c0=torch.randn(self.configs.e_layers, self.configs.batch_size,self.configs.d_model).cuda()

        return (self.h0,self.c0)



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
        batch_size = 3

    configs = Configs()
    model = Model(configs).to(device)
    LEN_TOTAL = configs.label_len + configs.seq_len
    hidden = model.init_hidden(configs.batch_size,LEN_TOTAL)

    enc = torch.randn([configs.batch_size, configs.seq_len, configs.enc_in])
    enc_mark = torch.randn([configs.batch_size, configs.seq_len, configs.enc_in])

    dec = torch.randn([configs.batch_size, configs.label_len+configs.pred_len, configs.dec_in])
    dec_mark = torch.randn([configs.batch_size, configs.label_len+configs.pred_len, configs.dec_in])
    out=model.forward(enc, enc_mark, dec, dec_mark)
    print('input shape',enc.shape)
    print('output shape',out[0].shape)
    a = 1