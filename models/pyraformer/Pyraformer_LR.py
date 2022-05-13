import torch
import torch.nn as nn
import sys
import os
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .Layers import EncoderLayer, Decoder, Predictor
from .Layers import Bottleneck_Construct, Conv_Construct, MaxPooling_Construct, AvgPooling_Construct
from .Layers import get_mask, get_subsequent_mask, refer_points, get_k_q, get_q_k
from .embed import DataEmbedding, CustomEmbedding

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(self, opt):
        super().__init__()

        self.d_model = opt.d_model
        self.model_type = 'Pyraformer'
        self.window_size = [4, 4, 4]
        self.truncate = False
        self.device  = torch.device("cuda")
        self.inner_size=3
        # if opt.decoder == 'attention':
        #     self.mask, self.all_size = get_mask(opt.input_size, opt.window_size, opt.inner_size, opt.device)
        # else:
        #     self.mask, self.all_size = get_mask(opt.input_size+1, opt.window_size, opt.inner_size, opt.device)
        # self.decoder_type = opt.decoder
        # if opt.decoder == 'FC':
        self.mask, self.all_size = get_mask(opt.seq_len, self.window_size, self.inner_size, self.device)
        self.indexes = refer_points(self.all_size, self.window_size, self.device)


        self.layers = nn.ModuleList([
            EncoderLayer(opt.d_model, 512, 4, 128,128, dropout=0.05, \
                normalize_before=False) for i in range(4)
            ])

        # if opt.embed_type == 'CustomEmbedding':
        #     self.enc_embedding = CustomEmbedding(opt.enc_in, opt.d_model, opt.covariate_size, opt.seq_num, opt.dropout)
        # else:
        self.enc_embedding = DataEmbedding(opt.enc_in, self.d_model, 0.05)

        self.conv_layers = eval('Bottleneck_Construct')(self.d_model, self.window_size, 128)

    def forward(self, x_enc, x_mark_enc):

        seq_enc = self.enc_embedding(x_enc, x_mark_enc)

        mask = self.mask.repeat(len(seq_enc), 1, 1).to(x_enc.device)
        seq_enc = self.conv_layers(seq_enc)

        for i in range(len(self.layers)):
            seq_enc, _ = self.layers[i](seq_enc, mask)

        #if self.decoder_type == 'FC':
        indexes = self.indexes.repeat(seq_enc.size(0), 1, 1, seq_enc.size(2)).to(seq_enc.device)
        indexes = indexes.view(seq_enc.size(0), -1, seq_enc.size(2))
        all_enc = torch.gather(seq_enc, 1, indexes)
        seq_enc = all_enc.view(seq_enc.size(0), self.all_size[0], -1)
        # elif self.decoder_type == 'attention' and self.truncate:
        #     seq_enc = seq_enc[:, :self.all_size[0]]

        return seq_enc


class Model(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(self, opt):
        super().__init__()

        self.predict_step = opt.pred_len
        self.d_model = 512
        self.input_size = opt.seq_len
        self.output_attention = opt.output_attention

        #self.decoder_type = opt.decoder
        self.channels = opt.enc_in

        self.encoder = Encoder(opt)
        # if opt.decoder == 'attention':
        #     mask = get_subsequent_mask(opt.input_size, opt.window_size, opt.predict_step, opt.truncate)
        #     self.decoder = Decoder(opt, mask)
        #     self.predictor = Predictor(opt.d_model, opt.enc_in)
        # elif opt.decoder == 'FC':
        self.predictor = Predictor(4 * self.d_model, opt.pred_len * opt.enc_in)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """
#         if self.decoder_type == 'attention':
#             enc_output = self.encoder(x_enc, x_mark_enc)
#             dec_enc = self.decoder(x_dec, x_mark_dec, enc_output)

#             if pretrain:
#                 dec_enc = torch.cat([enc_output[:, :self.input_size], dec_enc], dim=1)
#                 pred = self.predictor(dec_enc)
#             else:
#                 pred = self.predictor(dec_enc)
#         elif self.decoder_type == 'FC':
        enc_output = self.encoder(x_enc, x_mark_enc)[:, -1, :]
        pred = self.predictor(enc_output).view(enc_output.size(0), self.predict_step, -1)
        if self.output_attention:
            return x_dec, None
        else:
            return x_dec  # [B, L, D]

        #return pred

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
        d_model = 512
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