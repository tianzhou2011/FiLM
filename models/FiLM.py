# coding=utf-8
# author=maziqing
# email=maziqing.mzq@alibaba-inc.com
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import math
import numpy as np
from scipy import signal
from scipy import linalg as la
from scipy import special as ss
from utils import unroll
from utils.op import transition
import pickle
import pdb
from einops import rearrange, repeat
import opt_einsum as oe

contract = oe.contract

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TransposedLinear(nn.Module):
    """ Linear module on the second-to-last dimension """

    def __init__(self, d_input, d_output, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(d_output, d_input))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)) # nn.Linear default init
        # nn.init.kaiming_uniform_(self.weight, nonlinearity='linear') # should be equivalent

        if bias:
            self.bias = nn.Parameter(torch.empty(d_output, 1))
            bound = 1 / math.sqrt(d_input)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = 0.0

    def forward(self, x):
        return contract('... u l, v u -> ... v l', x, self.weight) + self.bias

def LinearActivation(
        d_input, d_output, bias=True,
        zero_bias_init=False,
        transposed=False,
        initializer=None,
        activation=None,
        activate=False, # Apply activation as part of this module
        weight_norm=False,
        **kwargs,
    ):
    """ Returns a linear nn.Module with control over axes order, initialization, and activation """

    # Construct core module
    linear_cls = TransposedLinear if transposed else nn.Linear
    if activation == 'glu': d_output *= 2
    linear = linear_cls(d_input, d_output, bias=bias, **kwargs)

    # Initialize weight
    if initializer is not None:
        get_initializer(initializer, activation)(linear.weight)

    # Initialize bias
    if bias and zero_bias_init:
        nn.init.zeros_(linear.bias)

    # Weight norm
    if weight_norm:
        linear = nn.utils.weight_norm(linear)

    if activate and activation is not None:
        activation = Activation(activation, dim=-2 if transposed else -1)
        linear = nn.Sequential(linear, activation)
    return linear


class DownPool(nn.Module):
    def __init__(self, d_input, expand, pool):
        super().__init__()
        self.d_output = d_input * expand
        self.pool = pool
        self.d_input = d_input * pool
        self.linear = LinearActivation(
            self.d_input,
            self.d_output,
            transposed=True,
            weight_norm=True,
        )

    def forward(self, x):
        #pdb.set_trace()
        x = rearrange(x, '... h (l s) -> ... (h s) l', s=self.pool)
        #print(x.shape)
        #print(self.d_input)
        #print(self.d_output)
        x = self.linear(x)
        #print(x.shape)
        return x

class UpPool(nn.Module):
    def __init__(self, d_input, expand, pool):
        super().__init__()
        self.d_output = d_input // expand
        self.pool = pool

        self.linear = LinearActivation(
            d_input,
            self.d_output * pool,
            transposed=True,
            weight_norm=True,
        )

    def forward(self, x, skip=None):
        x = self.linear(x)
        #print(x.shape)
        
        x = F.pad(x[..., :-1], (1, 0)) # Shift to ensure causality
        x = rearrange(x, '... (h s) l -> ... h (l s)', s=self.pool)
        #print(x.shape)
    
        if skip is not None:
            x = x + skip
        return x

class HiPPO_LegS(nn.Module):
    """ Vanilla HiPPO-LegS model (scale invariant instead of time invariant) """
    def __init__(self, N, max_length=1024, measure='legs', discretization='bilinear'):
        """
        max_length: maximum sequence length
        """
        super().__init__()
        self.N = N
        A, B = transition(measure, N)
        B = B.squeeze(-1)
        A_stacked = np.empty((max_length, N, N), dtype=A.dtype)
        B_stacked = np.empty((max_length, N), dtype=B.dtype)
        for t in range(1, max_length + 1):
            At = A / t
            Bt = B / t
            if discretization == 'forward':
                A_stacked[t - 1] = np.eye(N) + At
                B_stacked[t - 1] = Bt
            elif discretization == 'backward':
                A_stacked[t - 1] = la.solve_triangular(np.eye(N) - At, np.eye(N), lower=True)
                B_stacked[t - 1] = la.solve_triangular(np.eye(N) - At, Bt, lower=True)
            elif discretization == 'bilinear':
                A_stacked[t - 1] = la.solve_triangular(np.eye(N) - At / 2, np.eye(N) + At / 2, lower=True)
                B_stacked[t - 1] = la.solve_triangular(np.eye(N) - At / 2, Bt, lower=True)
            else: # ZOH
                A_stacked[t - 1] = la.expm(A * (math.log(t + 1) - math.log(t)))
                B_stacked[t - 1] = la.solve_triangular(A, A_stacked[t - 1] @ B - B, lower=True)
        self.A_stacked = torch.Tensor(A_stacked).to(device) # (max_length, N, N)
        self.B_stacked = torch.Tensor(B_stacked).to(device) # (max_length, N)
        # print("B_stacked shape", B_stacked.shape)

        vals = np.linspace(0.0, 1.0, max_length)
        self.eval_matrix = torch.Tensor((B[:, None] * ss.eval_legendre(np.arange(N)[:, None], 2 * vals - 1)).T).to(device)

    def forward(self, inputs, fast=False):
        """
        inputs : (length, ...)
        output : (length, ..., N) where N is the order of the HiPPO projection
        """

        L = inputs.shape[0]

        inputs = inputs.unsqueeze(-1)
        u = torch.transpose(inputs, 0, -2).to(device)
        u = u * self.B_stacked[:L]
        u = torch.transpose(u, 0, -2) # (length, ..., N)

        if fast:
            result = unroll.variable_unroll_matrix(self.A_stacked[:L], u)
        else:
            result = unroll.variable_unroll_matrix_sequential(self.A_stacked[:L], u)
        return result

    def reconstruct(self, c):
        a = self.eval_matrix @ c.unsqueeze(-1)
        return a.squeeze(-1)

class HiPPO_LegT(nn.Module):
    def __init__(self, N, dt=1.0, discretization='bilinear'):
        """
        N: the order of the HiPPO projection
        dt: discretization step size - should be roughly inverse to the length of the sequence
        """
        super(HiPPO_LegT,self).__init__()
        self.N = N
        A, B = transition('lmu', N)
        C = np.ones((1, N))
        D = np.zeros((1,))
        # dt, discretization options
        A, B, _, _, _ = signal.cont2discrete((A, B, C, D), dt=dt, method=discretization)

        B = B.squeeze(-1)

        self.register_buffer('A', torch.Tensor(A).to(device)) 
        self.register_buffer('B', torch.Tensor(B).to(device)) 
        vals = np.arange(0.0, 1.0, dt)
        self.register_buffer('eval_matrix',  torch.Tensor(
            ss.eval_legendre(np.arange(N)[:, None], 1 - 2 * vals).T).to(device))
    def forward(self, inputs):  # torch.Size([128, 1, 1]) -
        """
        inputs : (length, ...)
        output : (length, ..., N) where N is the order of the HiPPO projection
        """

        c = torch.zeros(inputs.shape[:-1] + tuple([self.N])).to(device)  # torch.Size([1, 256])
        cs = []
        for f in inputs.permute([-1, 0, 1]):
            f = f.unsqueeze(-1)
            # f: [1,1]
            new = f @ self.B.unsqueeze(0) # [B, D, H, 256]
            c = F.linear(c, self.A) + new
            # c = [1,256] * [256,256] + [1, 256]
            cs.append(c)
        return torch.stack(cs, dim=0)

    def reconstruct(self, c):
        a = (self.eval_matrix @ c.unsqueeze(-1)).squeeze(-1)
        return (self.eval_matrix @ c.unsqueeze(-1)).squeeze(-1)


################################################################
class SpectralConv1d_hierarchy(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d_hierarchy, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
        self.FNO_pre = SpectralConv1d_3d(in_channels,in_channels,modes1*2)
        self.FNO_post = SpectralConv1d_3d(out_channels,out_channels,modes1*2)

    def forward(self, x):
        x = x.permute([0,2,1])
        B, H, N = x.shape
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)
        x_ft = self.FNO_pre(x_ft)
        # Multiply relevant Fourier modes
        
        out_ft = torch.zeros(B,self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        a = x_ft[:, :, :self.modes1]
        out_ft[:, :, :self.modes1] = torch.einsum("bix,iox->box", a, self.weights1)

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1)).permute([0,2,1])
        return x
    

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels,seq_len, modes1,compression=0,ratio=0.5):
        super(SpectralConv1d, self).__init__()
            

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.compression = compression
        self.ratio = ratio 
        if modes1>10000:
            modes2=modes1-10000
            self.modes2 =min(modes2,seq_len//2)
            self.index0 = list(range(0, int(ratio*min(seq_len//2, modes2))))
            self.index1 = list(range(len(self.index0),self.modes2))
            np.random.shuffle(self.index1)
            self.index1 = self.index1[:min(seq_len//2,self.modes2)-int(ratio*min(seq_len//2, modes2))]
            self.index = self.index0+self.index1
            self.index.sort()
        elif modes1 > 1000:
            modes2=modes1-1000
            self.modes2 =min(modes2,seq_len//2)
            self.index = list(range(0, seq_len//2))
            np.random.shuffle(self.index)
            self.index = self.index[:self.modes2]
        else:
            self.modes2 =min(modes1,seq_len//2)
            self.index = list(range(0, self.modes2))

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, len(self.index), dtype=torch.cfloat))
        if self.compression > 0:
            print('compressed version')
            self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels,self.compression, len(self.index), dtype=torch.cfloat))
            self.weights2 = nn.Parameter(self.scale * torch.rand(self.compression,out_channels, dtype=torch.cfloat))
        #print(self.modes2)
        

    def forward(self, x):
        B, H,E, N = x.shape
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)
        #pdb.set_trace()
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(B,H, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        #a = x_ft[:, :,:, :self.modes1]
        #out_ft[:, :,:, :self.modes1] = torch.einsum("bjix,iox->bjox", a, self.weights1)
        if self.compression ==0:
            if self.modes1>1000:
                for wi, i in enumerate(self.index):
                    #print(self.index)
                    #print(out_ft.shape)
                    out_ft[:, :, :, wi] = torch.einsum('bji,io->bjo',(x_ft[:, :, :, i], self.weights1[:, :,wi]))
            else:
                a = x_ft[:, :,:, :self.modes2]
                out_ft[:, :,:, :self.modes2] = torch.einsum("bjix,iox->bjox", a, self.weights1)
        elif self.compression > 0:
            a = x_ft[:, :,:, :self.modes2]
            a = torch.einsum("bjix,ihx->bjhx", a, self.weights1)
            out_ft[:, :,:, :self.modes2] = torch.einsum("bjhx,ho->bjox", a, self.weights2)
        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x




class SpectralConv1d_3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d_3d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def forward(self, x):
        #x = x.permute([0,2,1])
        B, H, N = x.shape
       
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.fft(x)

        # Multiply relevant Fourier modes
        #pdb.set_trace()
        out_ft = torch.zeros(B,self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        a = x_ft[:, :, :self.modes1]
        out_ft[:, :, :self.modes1] = torch.einsum("bix,iox->box", a, self.weights1)

        # Return to physical space
        x = torch.fft.ifft(out_ft, n=x.size(-1))
        return x


class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, configs, N=512, N2=32):
        super(Model, self).__init__()
        self.configs = configs
        # self.modes = configs.modes
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        
        self.seq_len_all = self.seq_len+self.label_len
        
        self.output_attention = configs.output_attention
        self.layers = configs.e_layers
        self.modes1 = min(configs.modes1,self.pred_len//2)
        #self.modes1 = 32
        self.enc_in = configs.enc_in
        self.proj=False
        self.e_layers = configs.e_layers
        if self.configs.ours:
            #b, s, f means b, f
            self.affine_weight = nn.Parameter(torch.ones(1, 1, configs.enc_in))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, configs.enc_in))
        if configs.enc_in>1000:
            self.proj=True
            self.conv1 = nn.Conv1d(configs.enc_in,configs.d_model,1)
            self.conv2 = nn.Conv1d(configs.d_model,configs.dec_in,1)
            self.d_model = configs.d_model
            self.affine_weight = nn.Parameter(torch.ones(1, 1, configs.d_model))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, configs.d_model))

        # Decomp
        # kernel_size = configs.moving_avg
        # self.decomp = series_decomp(kernel_size)
        if self.configs.ab ==0:
            self.legt = nn.ModuleList([HiPPO_LegT(N=N, dt=1./configs.pred_len) for _ in range(configs.e_layers)])
            self.spec_conv_1 = nn.ModuleList([SpectralConv1d(in_channels=N, out_channels=N, modes1=self.modes1) for _ in range(configs.e_layers)])
        if self.configs.ab == 1:
            self.legt2 = HiPPO_LegT(N=N2, dt=1. / configs.pred_len)
            self.spec_conv_2 = SpectralConv1d(in_channels=N * N2, out_channels=N * N2, modes1=self.modes1)
            self.w1 = nn.Conv1d(N, N, 1)
            self.w2 = nn.Conv1d(N, N, 1)

        if self.configs.ab == 2:
            self.multiscale = [1,2,4]
            self.window_size=[256]
            self.legts = nn.ModuleList([HiPPO_LegT(N=n, dt=1./self.pred_len/i) for n in self.window_size for i in self.multiscale])
            self.spec_conv_1 = nn.ModuleList([SpectralConv1d(in_channels=n, out_channels=n,seq_len=min(self.pred_len,self.seq_len), modes1=configs.modes1,compression=configs.version,ratio=configs.ratio) for n in self.window_size for _ in range(len(self.multiscale))])               
            self.mlp = nn.Linear(len(self.multiscale)*len(self.window_size), 1)
        if self.configs.ab ==3:
            self.spec_conv_1 = SpectralConv1d_hierarchy(in_channels=self.d_model,out_channels=self.d_model,modes1=self.modes1)
            self.output_layer = nn.Conv1d(configs.seq_len,configs.pred_len,1)
        if self.configs.ab == 4:
            self.multiscale = [1, 2, 4]
            self.window_size=[256]
            self.legts = nn.ModuleList([HiPPO_LegS(N=n, max_length=self.pred_len*i) for n in self.window_size for i in self.multiscale])
            self.spec_conv_1 = nn.ModuleList([SpectralConv1d(in_channels=n, out_channels=n, modes1=self.modes1) for n in self.window_size for _ in range(len(self.multiscale))])
            self.mlp = nn.Linear(len(self.multiscale)*len(self.window_size), 1)
        if self.configs.ab ==5:
            self.multiscale = [1,2,4]
            self.downscale=[2,4]
            self.window_size=[256]
            self.legts = nn.ModuleList([HiPPO_LegT(N=256, dt=1./configs.pred_len) for _ in range(len(self.multiscale))])
            self.spec_conv_1 = nn.ModuleList([SpectralConv1d(in_channels=N, out_channels=N, modes1=self.modes1) for i in self.multiscale])
            self.upPool = nn.ModuleList([UpPool(self.enc_in*2,2,n) for n in self.downscale])
            self.downPool = nn.ModuleList([DownPool(self.enc_in,2,n) for n in self.downscale])
            self.mlp = nn.Linear(len(self.multiscale)*len(self.window_size), 1)
        if self.configs.ab == 6:
            self.multiscale = [1,2,4]
            self.window_size=[256]
            self.legts = nn.ModuleList([HiPPO_LegT(N=n, dt=1./self.pred_len/i) for n in self.window_size for i in self.multiscale])
            self.spec_conv_1 = nn.ModuleList([SpectralConv1d(in_channels=n, out_channels=n, modes1=self.modes1) for n in self.window_size for _ in range(len(self.multiscale))])    
            self.out_conv = nn.ModuleList([torch.nn.Conv1d(in_channels=2*min(self.seq_len,self.pred_len*i),out_channels=min(self.seq_len,self.pred_len*i),kernel_size=5,padding=2,groups=1) for  n in self.window_size for i in self.multiscale])
            self.mlp = nn.Linear(len(self.multiscale)*len(self.window_size), 1)
        if self.configs.ab == 7:
            self.multiscale = [1,2,4]
            self.window_size=[256]
            self.legts = nn.ModuleList([HiPPO_LegT(N=n, dt=1./self.pred_len/i) for _ in range(self.e_layers) for n in self.window_size for i in self.multiscale])
            self.spec_conv_1 = nn.ModuleList([SpectralConv1d(in_channels=n, out_channels=n, modes1=self.modes1) for _ in range(self.e_layers) for n in self.window_size for _ in range(len(self.multiscale))])
            self.mlp = nn.Linear(len(self.multiscale)*len(self.window_size), 1)

    def forward(self, x_enc, x_mark_enc, x_dec_true, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        if self.configs.ab == 0:
            if self.proj:
                x_enc = self.conv1(x_enc.transpose(1,2))
                x_enc = x_enc.transpose(1,2)
            B, L, E = x_enc.shape
            #print(x_enc.shape)
            #print(self.legt(x_enc.transpose(1, 2)).shape)
            x_enc=x_enc.transpose(1, 2)
            for i,_ in enumerate(self.legt):
                #print('x_enc shape',x_enc.shape)
                x_enc = self.legt[i](x_enc).squeeze().permute([1, 2,3,0])
                x_enc = self.spec_conv_1[i](x_enc)
                if i<self.layers-1:
                    x_enc = x_enc[:,:, -1, :]
                else:
                    x_enc = x_enc.transpose(2,3)[:,:, -1, :]
                #x_enc = x_dec @ (self.legt[-1].eval_matrix.T)
                #print('x_enc shape',x_enc.shape)
            #E, N, L = x_enc_c.shape
            
            #x_enc = x_enc[:, :,:, -self.pred_len:]
            #print('x_enc_c shape',x_enc_c.shape)
            #out1 = self.spec_conv_1(x_enc_c)
            #print('out1',out1.shape)
            #print('eval_matrix.T shape',self.legt.eval_matrix.T.shape)
            # out1 = F.gelu(out1 + self.w1(x_enc_c))
            #x_dec = x_enc.transpose(2, 3)[:,:, -1, :]
            #x_dec_c = out1.reshape
            x_dec = x_enc @ (self.legt[-1].eval_matrix.T)
            x_dec = x_dec.permute(0,2,1)
            if self.proj:
                x_dec = self.conv2(x_dec.transpose(1,2))
                x_dec = x_dec.transpose(1,2)
        elif self.configs.ab == 1:
            B, L, E = x_dec_true.shape
            x_enc = x_enc[:, -self.pred_len:, :]
            B, L, E = x_enc.shape
            x_enc_c = self.legt(x_enc.transpose(1, 2))
            x_enc_c = x_enc_c.squeeze().permute([1, 2, 0])
            # legt: [B, D, L] -> [L, B, D, 256]
            B, N, L = x_enc_c.shape
            x_enc_cc = self.legt2(x_enc_c)
            L, B, N, N2 = x_enc_cc.shape
            # legt2: [B, 256, L] -> [L, B, 256, 32]

            out1 = self.spec_conv_2(x_enc_cc.permute(1, 2, 3, 0).reshape([B, -1, L]))
            # spec_conv: [B, D, L] -> [B, D, L]

            x_dec_cc = out1.reshape([B, N, N2, -1])[:, :, :, -1] #.transpose(1, 2)
            B, N, N2 = x_dec_cc.shape
            x_dec_c = x_dec_cc @ (self.legt2.eval_matrix.T)
            x_dec = x_dec_c[:, :, -1] @ (self.legt.eval_matrix.T)
            # x_dec = x_dec
            x_dec = x_dec.unsqueeze(2)

        elif self.configs.ab == 2:
            return_data=[x_enc]
            if self.proj:
                x_enc = self.conv1(x_enc.transpose(1,2))
                x_enc = x_enc.transpose(1,2)
            if self.configs.ours:
                means = x_enc.mean(1, keepdim=True).detach()
                #mean
                x_enc = x_enc - means
                #var
                stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
                x_enc /= stdev
                # affine
                x_enc = x_enc * self.affine_weight + self.affine_bias
            B, L, E = x_enc.shape
            seq_len = self.seq_len
            x_decs = []
            jump_dist=0
            for i in range(0, len(self.multiscale)*len(self.window_size)):
                x_in_len = self.multiscale[i%len(self.multiscale)] * self.pred_len
                x_in = x_enc[:, -x_in_len:]
                legt = self.legts[i]
                x_in_c = legt(x_in.transpose(1, 2)).permute([1, 2,3, 0])[:,:,:,jump_dist:]
                out1 = self.spec_conv_1[i](x_in_c)
                if self.seq_len >= self.pred_len:
                    x_dec_c = out1.transpose(2, 3)[:,:, self.pred_len-1-jump_dist, :]
                else:
                    x_dec_c = out1.transpose(2, 3)[:,:, -1, :]
                x_dec = x_dec_c @ (legt.eval_matrix[-self.pred_len:,:].T)
                x_decs += [x_dec]
            return_data.append(x_in_c)
            return_data.append(out1)
            x_dec = self.mlp(torch.stack(x_decs, dim=-1)).squeeze(-1).permute(0,2,1)
            if self.configs.ours:
                x_dec = x_dec - self.affine_bias
                x_dec = x_dec / (self.affine_weight + 1e-10)
                x_dec = x_dec * stdev
                x_dec = x_dec + means
            if self.proj:
                x_dec = self.conv2(x_dec.transpose(1,2))
                x_dec = x_dec.transpose(1,2)
            return_data.append(x_dec)
        elif self.configs.ab ==3:
            if self.proj:
                x_enc = self.conv1(x_enc.transpose(1,2))
                x_enc = x_enc.transpose(1,2)
            B, L, E = x_enc.shape
            seq_len = self.seq_len
            x_dec = self.spec_conv_1(x_enc)
            if self.proj:
                x_dec = self.conv2(x_dec.transpose(1,2))
                #x_dec = x_dec.transpose(1,2)
            x_dec = self.output_layer(x_dec.transpose(1,2))
            #x_dec = x_dec.transpose(1,2)
        elif self.configs.ab == 4:
            if self.proj:
                x_enc = self.conv1(x_enc.transpose(1,2))
                x_enc = x_enc.transpose(1,2)
            B, L, E = x_enc.shape
            seq_len = self.seq_len
            x_decs = []
            for i in range(0, len(self.multiscale)*len(self.window_size)):
                x_in_len = self.multiscale[i%len(self.multiscale)] * self.pred_len
                x_in = x_enc[:, -x_in_len:]
                legt = self.legts[i]
                x_in_c = legt(x_in.transpose(1, 2))
                out1 = self.spec_conv_1[i](x_in_c.permute([0,1,3,2]))
                x_dec_c = out1.transpose(2, 3)[:,:, self.pred_len-1, :]
                x_dec = x_dec_c @ (legt.eval_matrix[-self.pred_len:,:].T)
                x_decs += [x_dec]
            x_dec = self.mlp(torch.stack(x_decs, dim=-1)).squeeze(-1).permute(0,2,1)
            if self.proj:
                x_dec = self.conv2(x_dec.transpose(1,2))
                x_dec = x_dec.transpose(1,2)
        elif self.configs.ab == 6:
            if self.proj:
                x_enc = self.conv1(x_enc.transpose(1,2))
                x_enc = x_enc.transpose(1,2)
            B, L, E = x_enc.shape
            seq_len = self.seq_len
            x_decs = []
            jump_dist=0
            for i in range(0, len(self.multiscale)*len(self.window_size)):
                x_in_len = self.multiscale[i%len(self.multiscale)] * self.pred_len
                x_in = x_enc[:, -x_in_len:]
                legt = self.legts[i]
                x_in_c = legt(x_in.transpose(1, 2)).permute([1, 2,3, 0])[:,:,:,jump_dist:]
                out1 = torch.cat([self.spec_conv_1[i](x_in_c),x_in_c],dim=-1)
                B, C,E,L = out1.shape
                out1 = self.out_conv[i](out1.view(B,C*E,-1).permute(0,2,1)).permute(0,2,1).view(B,C,E,-1)
                #print(out1.shape)
                if self.seq_len >= self.pred_len:
                    x_dec_c = out1.transpose(2, 3)[:,:, self.pred_len-1-jump_dist, :]
                    #x_append = x_in_c.transpose(2,3)[:,:, self.pred_len-1-jump_dist, :]
                else:
                    x_dec_c = out1.transpose(2, 3)[:,:, -1, :]
                x_dec = x_dec_c @ (legt.eval_matrix[-self.pred_len:,:].T)
                x_decs += [x_dec]
            x_dec = self.mlp(torch.stack(x_decs, dim=-1)).squeeze(-1).permute(0,2,1)
            if self.proj:
                x_dec = self.conv2(x_dec.transpose(1,2))
                x_dec = x_dec.transpose(1,2)
        elif self.configs.ab ==5:
            if self.proj:
                x_enc = self.conv1(x_enc.transpose(1,2))
                x_enc = x_enc.transpose(1,2)
            x_decs = []
            for i in range(0, len(self.multiscale)*len(self.window_size)):
                x_in_len = self.pred_len
                x_in = x_enc[:, -x_in_len:]
                #print('x_in.shape',x_in.shape)
                pool=False
                if i>0:
                #print(x_in.shape)
                    pool=True
                    x_in = self.downPool[i-1](x_in.permute([0,2,1])).permute([0,2,1])
                #print('x_in2.shape',x_in.shape)
                legt = self.legts[i]
                x_in_c = legt(x_in.transpose(1, 2)).permute([1, 2,3, 0])
                #
                out1 = self.spec_conv_1[i](x_in_c)
                if self.downscale[i-1] <=1:
                    x_dec_c = out1.transpose(2, 3)[:,:, self.pred_len-1, :]
                else:
                    x_dec_c = out1.transpose(2, 3)[:,:, -1, :]
                x_dec = x_dec_c @ (legt.eval_matrix[-self.pred_len:,:].T)
                if pool:
                    x_dec = x_dec_c @ (legt.eval_matrix[-(self.pred_len//self.downscale[i-1]):,:].T)
                    x_dec = self.upPool[i-1](x_dec)
                #print('x_dec shape',x_dec.shape)
                x_decs += [x_dec]
            x_dec = self.mlp(torch.stack(x_decs, dim=-1)).squeeze(-1).permute(0,2,1)
            if self.proj:
                x_dec = self.conv2(x_dec.transpose(1,2))
                x_dec = x_dec.transpose(1,2)
        elif self.configs.ab == 7:
            if self.proj:
                x_enc = self.conv1(x_enc.transpose(1,2))
                x_enc = x_enc.transpose(1,2)
            B, L, E = x_enc.shape
            seq_len = self.seq_len
            x_decs = []
            return_data=[x_enc]
            jump_dist=0
            for i in range(0, len(self.multiscale)*len(self.window_size)):
                for L in range(self.e_layers):
                    x_in_len = self.multiscale[i%len(self.multiscale)] * self.pred_len
                    x_in = x_enc[:, -x_in_len:]
                    #print('x_enc pre',x_enc.shape)
                    legt = self.legts[i]
                    x_in_c = legt(x_in.transpose(1, 2)).permute([1, 2,3, 0])[:,:,:,jump_dist:]
                    out1 = self.spec_conv_1[i](x_in_c)
                    if self.seq_len >= self.pred_len:
                        x_dec_c = out1.transpose(2, 3)[:,:, self.pred_len-1-jump_dist, :]
                    else:
                        x_dec_c = out1.transpose(2, 3)[:,:, -1, :]
                    if L < self.e_layers-1:
                        x_enc = x_dec_c @ (legt.eval_matrix[:,:].T)
                        x_enc = x_enc.permute(0,2,1)
                        #print('x_enc process',x_enc.shape)
                    else:
                        x_dec = x_dec_c @ (legt.eval_matrix[-self.pred_len:,:].T)
                x_decs += [x_dec]
            return_data.append(x_in_c)
            return_data.append(out1)
            x_dec = self.mlp(torch.stack(x_decs, dim=-1)).squeeze(-1).permute(0,2,1)
            if self.proj:
                x_dec = self.conv2(x_dec.transpose(1,2))
                x_dec = x_dec.transpose(1,2)
            
        if self.output_attention:
            return x_dec, return_data
        else:
            return x_dec  # [B, L, D]


if __name__ == '__main__':
    class Configs(object):
        ab = 2
        modes1 = 8
        seq_len = 336
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
        ours = False
        

    configs = Configs()
    model = Model(configs).to(device)

    enc = torch.randn([32, configs.seq_len, configs.enc_in]).cuda()
    enc_mark = torch.randn([32, configs.seq_len, 4]).cuda()

    dec = torch.randn([32, configs.label_len+configs.pred_len, configs.dec_in]).cuda()
    dec_mark = torch.randn([32, configs.label_len+configs.pred_len, 4]).cuda()
    out=model.forward(enc, enc_mark, dec, dec_mark)
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('model size',count_parameters(model)/(1024*1024))
    print('input shape',enc.shape)
    print('output shape',out[0].shape)
    a,b,c,d = out[1]
    print('input shape',a.shape)
    print('hippo shape',b.shape)
    print('processed hippo shape',c.shape)
    print('output shape',d.shape)
    # with open('data.pickle', 'wb') as f:
    # # Pickle the 'data' dictionary using the highest protocol available.
    #     pickle.dump(out[1], f, pickle.HIGHEST_PROTOCOL)
    # with open('data.pickle', 'rb') as f:
    # # The protocol version used is detected automatically, so we do not
    # # have to specify it.
    #     data = pickle.load(f)
    #     a,b,c = data
    # print('input shape',a.shape)
    # print('hippo shape',b.shape)
    # print('processed hippo shape',c.shape)
    # a = 1