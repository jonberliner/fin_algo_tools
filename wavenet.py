import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from layers import CausalConv1d


class One_Hot(nn.Module):
    def __init__(self, depth):
        super(One_Hot,self).__init__()
        self.depth = depth
        self.ones = torch.sparse.torch.eye(depth)

    def forward(self, X_in):
        X_in = X_in
        return Variable(self.ones.index_select(0,X_in.data))

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)


class WaveNet(nn.Module):
    def __init__(self, in_channels=256, n_residue=32, n_skip=512, dilation_depth=10, n_repeat=5, one_hot_input=False, n_out_per_input=1):
        # in_channels: audio quantization size
        # n_residue: residue channels
        # n_skip: skip channels
        # dilation_depth & n_repeat: dilation layer setup
        super(WaveNet, self).__init__()
        self.one_hot_input = one_hot_input
        self.dilation_depth = dilation_depth
        dilations = self.dilations = [2**i for i in range(dilation_depth)] * n_repeat
        self.one_hot = One_Hot(in_channels)
        self.from_input = nn.Conv1d(in_channels=in_channels, out_channels=n_residue, kernel_size=1)
        self.conv_sigmoid = nn.ModuleList([CausalConv1d(in_channels=n_residue, out_channels=n_residue, kernel_size=2, dilation=d)
                         for d in dilations])
        self.conv_tanh = nn.ModuleList([CausalConv1d(in_channels=n_residue, out_channels=n_residue, kernel_size=2, dilation=d)
                         for d in dilations])
        self.skip_scale = nn.ModuleList([nn.Conv1d(in_channels=n_residue, out_channels=n_skip, kernel_size=1)
                         for d in dilations])
        self.residue_scale = nn.ModuleList([nn.Conv1d(in_channels=n_residue, out_channels=n_residue, kernel_size=1)
                         for d in dilations])
        self.conv_post_1 = nn.Conv1d(in_channels=n_skip, out_channels=n_skip, kernel_size=1)
        self.conv_post_2 = nn.Conv1d(in_channels=n_skip, out_channels=in_channels * n_out_per_input, kernel_size=1)
        
    def forward(self, input):
        output = self.preprocess(input)
        skip_connections = [] # save for generation purposes
        for s, t, skip_scale, residue_scale in zip(self.conv_sigmoid, self.conv_tanh, self.skip_scale, self.residue_scale):
            output, skip = self.residue_forward(output, s, t, skip_scale, residue_scale)
            skip_connections.append(skip)
        # sum up skip connections
        output = sum([s[:,:,-output.size(2):] for s in skip_connections])
        output = self.postprocess(output)
        return output
    
    def preprocess(self, input):
        output = input
        if self.one_hot_input:
            output = self.one_hot(input).unsqueeze(0).transpose(1,2)
        output = self.from_input(output)
        return output
 
    def postprocess(self, input):
        output = F.elu(input)
        output = self.conv_post_1(output)
        output = F.elu(output)
        output = self.conv_post_2(output)  #.squeeze(0).transpose(0,1)
        return output
 
    def residue_forward(self, input, conv_sigmoid, conv_tanh, skip_scale, residue_scale):
        output = input
        output_sigmoid, output_tanh = conv_sigmoid(output), conv_tanh(output)
        output = F.sigmoid(output_sigmoid) * F.tanh(output_tanh)
        skip = skip_scale(output)
        output = residue_scale(output)
        # import pdb; pdb.set_trace()
        output = output + input[:,:,-output.size(2):]
        return output, skip
