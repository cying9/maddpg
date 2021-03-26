import torch
from torch import nn
from torch import functional as F


def get_device(disable_cuda=False):
    if (not disable_cuda) & torch.cuda.is_available():
        device = "cuda"
        print(f'Using CUDA ({torch.cuda.get_device_name(torch.cuda.current_device())})')
    else:
        print('Using CPU')
        device = "cpu"
    return torch.device(device)


def init_params(model, gain=1.0):
    for params in model.parameters():
        if len(params.shape) > 1:
            nn.init.xavier_uniform_(params.data, gain=gain)
