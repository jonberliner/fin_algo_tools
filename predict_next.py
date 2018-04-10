import numpy as np
import pandas as pd
import os

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import Adam

from typing import List, Union, Dict

from pyt.core_modules import MLP, SNMLP
from pyt.util import vft, ft, lt, vlt, var_to_numpy
from layers import swish

# from synthetic_data import sin_data

DATA_DIR = '/Users/jsb/repos/eth_trading/data/'
# DATA_KEYS = ['PRICE', 'LOW', 'HIGH', 'OPEN', 'CLOSE', 'VOLUME']
DATA_KEYS = ['LOW', 'HIGH', 'OPEN', 'CLOSE', 'VOLUME']

CH_INPUT = len(DATA_KEYS)
CH_SKIP = 64
CH_RESID = 32

CH_TARGET = 1
LEN_INPUT = 256
OFFSET_TARGET = 1
LEN_TARGET = 1

all_data = pd.read_csv(os.path.join(DATA_DIR, 'prices_low_high_prev_month_subset.csv'))
data = all_data[DATA_KEYS]



#### WAVENET VERSION
from wavenet import WaveNet
encoder = WaveNet(in_channels=CH_INPUT,
                  n_residue=CH_RESID,
                  n_skip=CH_SKIP,
                  dilation_depth=8,
                  n_repeat=1,
                  n_out_per_input=3)

xs = vlt(data.values[:LEN_INPUT].T[None,:,:]).float()

hs = encoder(xs)

MIN_MAG = 2e-4  # empirically derived from distribution of diff(OPEN) / OPEN

# for making predict-next-step training set
def prep_batch(inputs_start: Union[int, List[int]], 
               inputs_length: int, 
               targets_offset: int=1,
               min_mag: float=MIN_MAG,
               ys_transform: str='None')\
              -> (torch.FloatTensor, torch.FloatTensor):

    assert ys_transform in ['', 'percent_change', 'categorical_percent_change']

    if type(inputs_start) is int:
        inputs_start = [inputs_start]

    inputs = list()
    targets = list()
    for i_start in inputs_start:
        inputs_end = i_start + inputs_length

        targets_start = i_start + targets_offset
        targets_end = targets_start + inputs_length

        xs = data[i_start:inputs_end].values  # len x ch
        ys = data[targets_start:targets_end].values  # len x ch

        if ys_transform in ['percent_change', 'categorical_percent_change']:
            # percent change
            ys = ((ys - xs) / xs)

        if ys_transform == 'categorical_percent_change':
            # quantise into three stripes of "going down", "not moving", and "going up"
            ys = torch.from_numpy(ys).float()
            sign_ys = ys.ge(0.).float().mul(2).sub(1)
            mag_ys = ys.abs()
            ys = mag_ys.ge(min_mag).float().mul(sign_ys)  # in {-1, 0, 1}
            ys = ys.add(1).long().numpy()  # make categories (in {0, 1, 2})

        inputs.append(xs.T)
        targets.append(ys.T)

    inputs = torch.from_numpy(np.stack(inputs)).float()
    targets = torch.from_numpy(np.stack(targets)).long()
    return (inputs, targets)


def forward(xs):
    bs = xs.shape[0]
    hs = encoder(xs)  # bs x ch_in * 3 x seqlen
    return hs.view(bs, -1, CH_INPUT, LEN_INPUT)\
             .permute(0, 2, 3, 1).contiguous()  # bs x ch_in x seqlen x nclass


    # sign_l = hs[:, :CH_INPUT, :]
    # mag_l = hs[:, CH_INPUT:, :]

    # return sign_l, mag_l

    # percent_resid = encoder(xs)
    # ys = F.softplus(xs + (xs * percent_resid))
    # ys.percent_resid = percent_resid
    # return ys


params = encoder.parameters()
# params = list(clf.parameters()) + list(encoder.parameters())
opt = Adam(params, lr=1e-2)  #, weight_decay=None)

mse = nn.MSELoss()
bcel = nn.BCEWithLogitsLoss()
cel = nn.CrossEntropyLoss()

MAX_SIGN_LOSS = nn.BCELoss()(Variable(torch.zeros(1)), Variable(torch.ones(1)))

def step(xs, ys, i_step):
    xs = Variable(xs)
    ys = Variable(ys).contiguous()
    opt.zero_grad()

    yhl = forward(xs)
    loss = cel(input=yhl.view(-1, 3), target=ys.view(-1))
    # # yhat = forward(xs)
    # sign_l, mag_l = forward(xs)

    # sign_t = ys.gt(0.).float()
    # mag_t = ys.abs()

    # # sign_losses = F.binary_cross_entropy_with_logits(input=sign_l, 
    # #                                                  target=sign_t,
    # #                                                  size_average=False,
    # #                                                  reduce=False)
    # # p_max_sign_loss = sign_losses.clamp(0., MAX_SIGN_LOSS)\
    # #                              .mul(1. / MAX_SIGN_LOSS)
    # # w_mags = 1. - p_max_sign_loss
    # # mag_losses = F.mse_loss(input=mag_l.abs(),
    # #                         target=mag_t,
    # #                         size_average=False,
    # #                         reduce=False)
    # # losses = w_mags * mag_losses

    # sign_loss = bcel(sign_l, sign_t)
    # mag_loss = mse(F.softplus(mag_l), mag_t)

    # loss = sign_loss + mag_loss
    loss.backward()
    opt.step()
    print(f'(step {i_step}) loss: {var_to_numpy(loss)[0]}', end='\r')

batcher = lambda i_start: prep_batch(i_start, LEN_INPUT, ys_transform='categorical_percent_change')
# batcher = lambda i_start: prep_batch(i_start, LEN_INPUT, OFFSET_TARGET, LEN_TARGET)

batch_size = 32
i_max_training = int(data.shape[0] * 0.7)
i_start_max_training = i_max_training - LEN_INPUT - 1
# i_start_max_training = i_max_training - LEN_INPUT - OFFSET_TARGET - LEN_TARGET

SEED = None
rng = np.random.RandomState(SEED)

N_STEP = 10000
for i_step in range(N_STEP):
    inputs_start = rng.randint(i_start_max_training, size=batch_size)
    batch_xs, batch_ys = batcher(inputs_start)
    step(batch_xs, batch_ys, i_step)

encoder.eval()
sign_l, mag_l = forward(Variable(batch_xs))

sign_h = F.sigmoid(sign_l).data.numpy() * 2 - 1
mag_h = mag_l.abs().data.numpy()

sign_t = batch_ys.gt(0.).float().numpy() * 2 - 1
mag_t = batch_ys.abs().numpy()

# _next_xh = forward(Variable(batch_xs))
next_xh = _next_xh.data.numpy()[0][0]
next_xt = batch_ys.numpy()[0][0]

p_resid_hat = var_to_numpy(_next_xh.percent_resid)[0][0]

resid_true = np.concatenate([[0], np.diff(next_xt)])
p_resid_true = resid_true / next_xt 
# def model(x):
#     return classifier(encoder(x))


# def conv_layer(dim_input, dim_output, **kwargs):


