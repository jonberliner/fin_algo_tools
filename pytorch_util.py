import torch.nn as nn

def freeze_params(params):
    for param in params:
        param.requires_grad = False


def unfreeze_params(params):
    for param in params:
        param.requires_grad = True


def constant_conv1d_kn(n):
    conv = nn.Conv1d(1, 5, n, bias=False)
    freeze_params(conv.parameters())
    conv.weight.fill_(1.)
    return conv

