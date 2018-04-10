import torch
import torch.nn as nn
import torch.nn.functional as F

from pyt.core_modules import MLP

def swish(inputs):
    return inputs * F.sigmoid(inputs)


class CausalConv1d(nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)

        self.left_padding = dilation * (kernel_size - 1)

    def forward(self, input):
        # import pdb; pdb.set_trace()
        x = F.pad(input.unsqueeze(2), (self.left_padding, 0, 0, 0)).squeeze(2)

        return super(CausalConv1d, self).forward(x)


class Encoder(nn.Module):
    def __init__(self, dim_input, dim_output, dim_hidden=8, act_fn=swish):
        super().__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.act_fn = act_fn

        self.c0 = nn.Conv1d(dim_input, dim_hidden, 1, dilation=1)
        self.b0 = nn.BatchNorm1d(dim_hidden)

        self.cc1 = CausalConv1d(dim_hidden, dim_hidden, 2, dilation=1)
        self.b1 = nn.BatchNorm1d(dim_hidden)
        # self.c1 = nn.Conv1d(dim_hidden, dim_hidden, 1, dilation=1)

        self.cc2 = CausalConv1d(dim_hidden, dim_hidden, 2, dilation=2)
        self.b2 = nn.BatchNorm1d(dim_hidden)
        # self.c2 = nn.Conv1d(dim_hidden, dim_hidden, 1, dilation=1)

        self.cc3 = CausalConv1d(dim_hidden, dim_hidden, 2, dilation=4)
        self.b3 = nn.BatchNorm1d(dim_hidden)
        # self.c3 = nn.Conv1d(dim_hidden, dim_hidden, 1, dilation=1)

        self.cc4 = CausalConv1d(dim_hidden, dim_hidden, 2, dilation=8)
        self.b4 = nn.BatchNorm1d(dim_hidden)
        # self.c4 = nn.Conv1d(dim_hidden, dim_hidden, 1, dilation=1)

        self.cc5 = CausalConv1d(dim_hidden, dim_hidden, 2, dilation=16)
        self.b5 = nn.BatchNorm1d(dim_hidden)

        self.readout = nn.Conv1d(dim_hidden, dim_output, 1, dilation=1)

    def forward(self, inputs):
        output = self.b0(self.c0(inputs))
        for li in range(1, 5 + 1):
            causalconv = getattr(self, 'cc' + str(li))
            bn = getattr(self, 'b' + str(li))
            # conv1 = getattr(self, 'c' + str(li))
            output = bn(self.act_fn(causalconv(output)))
        output = self.readout(output)
        return output


class Predictor(MLP):
    def __init__(self, dim_input, ch_output, len_output, dim_hidden=[], act_fn=swish, **kwargs):
        dim_output = ch_output * len_output
        super().__init__(dim_input, dim_output, dim_hidden, act_fn=swish, **kwargs)
        self.ch_output = ch_output
        self.len_output = len_output

    def forward(self, inputs):
        return super().forward(inputs[:, :, -1]).view(-1, self.ch_output, self.len_output)


class Model(nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, inputs):
        return self.classifier(self.encoder(inputs))

