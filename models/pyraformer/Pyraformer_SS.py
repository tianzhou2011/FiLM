import torch
import torch.nn as nn
from .Layers import EncoderLayer, Predictor
from .Layers import Bottleneck_Construct
from .Layers import get_mask, refer_points, get_k_q, get_q_k
from .embed import SingleStepEmbedding


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(self, opt):
        super().__init__()

        self.d_model = opt.d_model
        self.window_size = opt.window_size
        self.num_heads = opt.n_head
        self.mask, self.all_size = get_mask(opt.input_size, opt.window_size, opt.inner_size, opt.device)
        self.indexes = refer_points(self.all_size, opt.window_size, opt.device)

        if opt.use_tvm:
            assert len(set(self.window_size)) == 1, "Only constant window size is supported."
            q_k_mask = get_q_k(opt.input_size, opt.inner_size, opt.window_size[0], opt.device)
            k_q_mask = get_k_q(q_k_mask)
            self.layers = nn.ModuleList([
                EncoderLayer(opt.d_model, opt.d_inner_hid, opt.n_head, opt.d_k, opt.d_v, dropout=opt.dropout, \
                    normalize_before=False, use_tvm=True, q_k_mask=q_k_mask, k_q_mask=k_q_mask) for i in range(opt.n_layer)
                ])
        else:
            self.layers = nn.ModuleList([
                EncoderLayer(opt.d_model, opt.d_inner_hid, opt.n_head, opt.d_k, opt.d_v, dropout=opt.dropout, \
                    normalize_before=False) for i in range(opt.n_layer)
                ])

        self.embedding = SingleStepEmbedding(opt.covariate_size, opt.num_seq, opt.d_model, opt.input_size, opt.device)

        self.conv_layers = Bottleneck_Construct(opt.d_model, opt.window_size, opt.d_k)

    def forward(self, sequence):

        seq_enc = self.embedding(sequence)
        mask = self.mask.repeat(len(seq_enc), self.num_heads, 1, 1).to(sequence.device)

        seq_enc = self.conv_layers(seq_enc)

        for i in range(len(self.layers)):
            seq_enc, _ = self.layers[i](seq_enc, mask)

        indexes = self.indexes.repeat(seq_enc.size(0), 1, 1, seq_enc.size(2)).to(seq_enc.device)
        indexes = indexes.view(seq_enc.size(0), -1, seq_enc.size(2))
        all_enc = torch.gather(seq_enc, 1, indexes)
        all_enc = all_enc.view(seq_enc.size(0), self.all_size[0], -1)

        return all_enc


class Model(nn.Module):

    def __init__(self, opt):
        super().__init__()

        self.encoder = Encoder(opt)

        # convert hidden vectors into two scalar
        self.mean_hidden = Predictor(4 * opt.d_model, 1)
        self.var_hidden = Predictor(4 * opt.d_model, 1)

        self.softplus = nn.Softplus()

    def forward(self, self, x_enc, x_mark_enc, x_dec_true, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_output = self.encoder(x_enc)

        mean_pre = self.mean_hidden(enc_output)
        var_hid = self.var_hidden(enc_output)
        var_pre = self.softplus(var_hid)
        mean_pre = self.softplus(mean_pre)

        return mean_pre.squeeze(2), var_pre.squeeze(2)

    def test(self, data, v):
        mu, sigma = self(data)

        sample_mu = mu[:, -1] * v
        sample_sigma = sigma[:, -1] * v
        return sample_mu, sample_sigma

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