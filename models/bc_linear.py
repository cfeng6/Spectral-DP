import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as TF
import numpy as np
from torch.fft import fftn, ifftn, rfftn, irfftn
import math

from layer_config import DEVICE, SIGMA, FILTER_RATIO, LAYER_BOUND, TOTAL_BOUND
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# TOTAL_BOUND = 1.0
# SIGMA = 1.0
# FILTER_RATIO = 0.5
# LAYER_BOUND = TOTAL_BOUND


def BC_forward_ortho(w, x):
    """
    w is the weight vector with size (1,k) which k is the block size;
    x is the input vector with size (d,k), which d is the input dimension
    """
    fft_data = torch.fft.fft(w, norm='ortho') * torch.fft.fft(x, norm='backward')
    return torch.fft.ifft(fft_data, norm='ortho').float()


def L2_Clip(data, bound):
    norm = torch.norm(data, p=2) / bound
    norm[norm <= 1] = 1.
    data = data / norm
    return data


def PAD_noise_fc(w, x, k_value, sigma, bound, batch_size):
    noise_scale = sigma * TOTAL_BOUND
    fft_data = torch.fft.fft(w, norm="ortho") * torch.fft.fft(x, norm="backward")
    fft_data = L2_Clip(fft_data, bound)
    # noise = torch.normal(mean=0, std=noise_scale, size=fft_data[:, :, 0:k_value].size(), device=DEVICE)
    real_noise = torch.normal(mean=0, std=math.sqrt(0.5) * noise_scale, size=fft_data[:, :, 0:k_value].size(),
                              device=DEVICE)
    imag_noise = torch.normal(mean=0, std=math.sqrt(0.5) * noise_scale, size=fft_data[:, :, 0:k_value].size(),
                              device=DEVICE)

    noise = real_noise + 1j * imag_noise
    fft_data[:, :, 0:k_value] = fft_data[:, :, 0:k_value] + noise / batch_size
    fft_data[:, :, k_value:-1] = 0.

    return torch.fft.ifft(fft_data, norm="ortho").float()


class Linear_BC_func(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weights):
        ctx.weights = weights
        ctx.save_for_backward(input, weights)

        BS = input.size(0)
        out_channels = ctx.weights.size(0)
        in_channels = ctx.weights.size(1)
        k = ctx.weights.size(2)
        input = input.view(-1, in_channels, k)

        output = torch.zeros((BS, out_channels * k)).to(DEVICE)
        for n in range(BS):
            x_input = input[n, :, :].unsqueeze(0)
            y_outputx = BC_forward_ortho(ctx.weights, x_input)
            outputx = torch.sum(y_outputx, 1)
            output[n] = outputx.view(int(out_channels * k))
        return output.double()

    @staticmethod
    def backward(ctx, grad_output):
        input, weights = ctx.saved_tensors
        BS = input.size(0)
        out_channels = weights.size(0)
        in_channels = weights.size(1)
        k = weights.size(2)
        F, C, k = weights.size()
        input = input.view(-1, in_channels, k)

        k_value = math.ceil(k * FILTER_RATIO)

        grad_outs = grad_output.clone()
        grad_outs = grad_outs.view((BS, out_channels, k))
        grad_input = torch.zeros(BS, in_channels, k, device=DEVICE).double()
        grad_weight = torch.zeros((out_channels, in_channels, k), device=DEVICE).double()

        if ctx.needs_input_grad[0]:
            weight_s = torch.flip(torch.roll(weights, k - 1, 2), dims=[2])
            grad_input = BC_forward_ortho(weight_s, grad_outs.unsqueeze(2))
            grad_input = torch.sum(grad_input.double(), 1)
            grad_input = grad_input.view(BS, int(in_channels * k))

        if ctx.needs_input_grad[1]:
            input_f = torch.flip(torch.roll(input, -1, 2), dims=[2])
            grad_outs = grad_outs.permute(1, 0, 2)
            for n in range(BS):
                x_input_f = input_f[n, :, :].unsqueeze(0)
                grad_out = grad_outs[:, n, :].unsqueeze(1)
                grad_per_i = PAD_noise_fc(x_input_f, grad_out, k_value, SIGMA, LAYER_BOUND, BS)
                grad_weight += grad_per_i
        return grad_input, grad_weight


class Linear_BC(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k: int):
        super(Linear_BC, self).__init__()
        self.in_channels = in_channels // k
        self.out_channels = out_channels // k

        self.k = k
        self.weights = nn.Parameter(torch.empty(self.out_channels, self.in_channels, k))
        stdv = 1. / math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-stdv, stdv)
        self.bcfunc = Linear_BC_func()

    def forward(self, x):
        return self.bcfunc.apply(x, self.weights)