import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.AE_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
from layers.FourierCorrelation import SpectralConv1d, SpectralConvCross1d
import numpy as np


class Model(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding1 = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding2 = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
        # self.encoder = Encoder(
        #     [
        #         EncoderLayer(
        #             AttentionLayer(
        #                 FullAttention(False, configs.factor, attention_dropout=configs.dropout,
        #                               output_attention=configs.output_attention), configs.d_model, configs.n_heads),
        #             configs.d_model,
        #             configs.d_ff,
        #             dropout=configs.dropout,
        #             activation=configs.activation
        #         ) 
        #     ],
        #     norm_layer=torch.nn.LayerNorm(configs.d_model)
        # )
        #Encoder
        if configs.ab == 0:
            decoder_self_att1 = FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention)
            decoder_cross_att1 = FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention)
            decoder_self_att2 = FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention)
            decoder_cross_att2 = FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention)
            
        elif configs.ab == 1:
            decoder_self_att1 = SpectralConv1d(in_channels=configs.d_model, out_channels=configs.d_model, seq_len=self.seq_len, modes1=configs.modes1)
            decoder_cross_att1 = SpectralConvCross1d(in_channels=configs.d_model, out_channels=configs.d_model,
                                                    seq_len_q=64 ,seq_len_kv=self.seq_len, modes1=configs.modes1)
            decoder_self_att2 = SpectralConv1d(in_channels=configs.d_model, out_channels=configs.d_model, seq_len=self.seq_len , modes1=configs.modes1)
            decoder_cross_att2 = SpectralConvCross1d(in_channels=configs.d_model, out_channels=configs.d_model,
                                                    seq_len_q=self.pred_len, seq_len_kv=64 , modes1=configs.modes1)
        elif configs.ab == 2:
            decoder_self_att1 = FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention)
            decoder_cross_att1 = FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention)
            decoder_self_att2 = SpectralConv1d(in_channels=configs.d_model, out_channels=configs.d_model, seq_len=self.seq_len , modes1=configs.modes1)
            decoder_cross_att2 = SpectralConvCross1d(in_channels=configs.d_model, out_channels=configs.d_model,
                                                    seq_len_q=self.pred_len, seq_len_kv=64 , modes1=configs.modes1)
        elif configs.ab == 3:
            decoder_self_att1 = SpectralConv1d(in_channels=configs.d_model, out_channels=configs.d_model, seq_len=self.seq_len, modes1=configs.modes1)
            decoder_cross_att1 = SpectralConvCross1d(in_channels=configs.d_model, out_channels=configs.d_model,
                                                    seq_len_q=64 ,seq_len_kv=self.seq_len, modes1=configs.modes1)
            decoder_self_att2 = FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention)
            decoder_cross_att2 = FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention)
            
            
        self.encoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        decoder_self_att1,
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        decoder_cross_att1,
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.d_model, bias=True)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        decoder_self_att2,
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        decoder_cross_att2,
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec1, x_mark_dec1,x_dec2, x_mark_dec2,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        #print('enc shape',enc_out.shape)
        dec1_out = self.dec_embedding1(x_dec1,x_mark_dec1)
        #print('dec1 out shape',dec1_out.shape)
        
        enc_out= self.encoder(dec1_out,enc_out,x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        #print('compress shape',enc_out.shape)

        dec2_out = self.dec_embedding2(x_dec2, x_mark_dec2)
        dec2_out = self.decoder(dec2_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        #print('out shape',dec2_out.shape)
        #raise Exception('aaaa')

        if self.output_attention:
            return dec2_out[:, -self.pred_len:, :], None
        else:
            return dec2_out[:, -self.pred_len:, :]  # [B, L, D]
