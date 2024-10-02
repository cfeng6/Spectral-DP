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

def _conv2dfft(img, kernel):
    cnvShape = (img.size(1) + kernel.size(1) - 1,img.size(2) + kernel.size(2) - 1, img.size(3) + kernel.size(3) - 1)
    img_fft = fftn(torch.flip(img.unsqueeze(1), dims=[2,3,4]), cnvShape, dim=(2, 3, 4), norm='ortho')
    kernel_fft = fftn(kernel.unsqueeze(0), cnvShape, dim=(2,3,4), norm='backward')
    XKhat = img_fft * kernel_fft
    fft_out = irfftn(XKhat, s=cnvShape, norm='ortho')
    fft_out = fft_out[:, :, :, kernel.size(2)-1:-(kernel.size(2)-1), kernel.size(3)-1:-(kernel.size(3)-1)]
    return fft_out.squeeze(2).to(DEVICE)


def _conv2dfft_single_noise(img, kernel, kernel_bound, sigma, batch_size):
    cnvShape = (img.size(1) + kernel.size(1) - 1, img.size(2) + kernel.size(2) - 1, img.size(3) + kernel.size(3) - 1)
    img_fft = fftn(torch.flip(img.unsqueeze(1), dims=[2, 3, 4]), cnvShape, dim=(2, 3, 4), norm='ortho')
    kernel_fft = fftn(kernel.unsqueeze(0), cnvShape, dim=(2, 3, 4), norm='backward')
    XKhat = img_fft * kernel_fft

    # Clipping
    norm = torch.norm(XKhat, p=2) / kernel_bound
    norm[norm <= 1] = 1.
    XKhat = XKhat / norm

    noise_k = math.ceil(XKhat.size(3) * FILTER_RATIO)
    noise_scale = sigma * TOTAL_BOUND
    # noise = torch.normal(mean=0, std=noise_scale, size=XKhat[:, :, :, 0:noise_k, 0:noise_k].size(), device=DEVICE)
    real_noise = torch.normal(mean=0, std=math.sqrt(0.5) * noise_scale,
                              size=XKhat[:, :, :, 0:noise_k, 0:noise_k].size(), device=DEVICE)
    imag_noise = torch.normal(mean=0, std=math.sqrt(0.5) * noise_scale,
                              size=XKhat[:, :, :, 0:noise_k, 0:noise_k].size(), device=DEVICE)
    noise = real_noise + 1j * imag_noise

    XKhat[:, :, :, 0:noise_k, 0:noise_k] = XKhat[:, :, :, 0:noise_k, 0:noise_k] + noise / batch_size
    XKhat[:, :, :, noise_k:-1, :] = 0.
    XKhat[:, :, :, :, noise_k:-1] = 0.

    fft_out = irfftn(XKhat, s=cnvShape, norm='ortho')
    fft_out = fft_out[:, :, :, kernel.size(2) - 1:-(kernel.size(2) - 1), kernel.size(3) - 1:-(kernel.size(3) - 1)]
    return fft_out.squeeze(2).to(DEVICE)

class conv2d_fft(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights):
        ctx.save_for_backward(input, weights)
        padding = 1
        stride = 1
        N, C, H, W = input.size()
        F, _, K, K = weights.size()

        H_out = int((H + 2 * padding - K) // stride + 1)
        W_out = int((H + 2 * padding - K) // stride + 1)

        output = torch.zeros((N, F, H_out, W_out))
        pad_widths = (padding, padding, padding, padding)
        input_pad = TF.pad(input=input, pad=pad_widths, mode='constant', value=0.)

        output = TF.conv2d(input_pad, weights, padding=0, stride=stride)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weights = ctx.saved_tensors
        padding = 1
        stride = 1
        N, F, H_out, W_out = grad_output.size()
        N, C, H, W = input.size()
        F, _, K, K = weights.size()

        pad_widths = (padding, padding, padding, padding)
        input_pad = TF.pad(input=input, pad=pad_widths, mode='constant', value=0.)
        grad_input_pad = torch.zeros((N, C, H + 2 * padding, W + 2 * padding))
        grad_input = torch.zeros_like(input)
        grad_weights = torch.zeros((C, F, K, K)).to(DEVICE)
        if ctx.needs_input_grad[0]:
            grad_input_pad = TF.conv_transpose2d(grad_output, weights)
            grad_input = grad_input_pad[:, :, padding:-padding, padding:-padding]
        if ctx.needs_input_grad[1]:
            input_pad = input_pad.permute(1, 0, 2, 3)
            grad_output = grad_output.permute(1, 0, 2, 3)
            for n in range(N):
                grad_weights += _conv2dfft_single_noise(input_pad[:, n, :, :].unsqueeze(1),
                                                        grad_output[:, n, :, :].unsqueeze(1), LAYER_BOUND, SIGMA, N)
            grad_weights = torch.flip(grad_weights.permute(1, 0, 2, 3), dims=[2, 3])
        return grad_input, grad_weights


class Conv2d_BC(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k: int):
        super(Conv2d_BC, self).__init__()
        self.k = k
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weights = nn.Parameter(torch.empty((self.out_channels, self.in_channels, k, k)))
        n = in_channels * k * k
        stdv = 1. / math.sqrt(n)
        self.weights.data.uniform_(-stdv, stdv)
        self.conv2d = conv2d_fft()

    def forward(self, x):
        return self.conv2d.apply(x, self.weights)
    
def _conv2dfft_single_noise_v2(img, kernel, kernel_bound, sigma, batch_size):
    cnvShape = (img.size(1) + kernel.size(1) - 1, img.size(2) + kernel.size(2) - 1, img.size(3) + kernel.size(3) - 1)
    img_fft = fftn(torch.flip(img.unsqueeze(1), dims=[2, 3, 4]), cnvShape, dim=(2, 3, 4), norm='ortho')
    kernel_fft = fftn(kernel.unsqueeze(0), cnvShape, dim=(2, 3, 4), norm='backward')
    XKhat = img_fft * kernel_fft

    # Clipping
    norm = torch.norm(XKhat, p=2) / kernel_bound
    norm[norm <= 1] = 1.
    XKhat = XKhat / norm

    noise_k = math.ceil(XKhat.size(3) * FILTER_RATIO)
    noise_scale = sigma * TOTAL_BOUND
    # noise = torch.normal(mean=0, std=noise_scale, size=XKhat[:, :, :, 0:noise_k, 0:noise_k].size(), device=DEVICE)
    real_noise = torch.normal(mean=0, std=math.sqrt(0.5) * noise_scale,
                              size=XKhat[:, :, :, 0:noise_k, 0:noise_k].size(), device=DEVICE)
    imag_noise = torch.normal(mean=0, std=math.sqrt(0.5) * noise_scale,
                              size=XKhat[:, :, :, 0:noise_k, 0:noise_k].size(), device=DEVICE)
    noise = real_noise + 1j * imag_noise

    XKhat[:, :, :, 0:noise_k, 0:noise_k] = XKhat[:, :, :, 0:noise_k, 0:noise_k] + noise / batch_size
    XKhat[:, :, :, noise_k:-1, :] = 0.
    XKhat[:, :, :, :, noise_k:-1] = 0.

    fft_out = irfftn(XKhat, s=cnvShape, norm='ortho')
    if kernel.size(2) - 1 == 0:
        fft_out = fft_out
    else:
        fft_out = fft_out[:, :, :, kernel.size(2) - 1:-(kernel.size(2) - 1), kernel.size(3) - 1:-(kernel.size(3) - 1)]
    return fft_out.squeeze(2).to(DEVICE)

class conv2d_fft_no_padding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights):
        ctx.save_for_backward(input, weights)
        padding = 0
        stride = 1
        N, C, H, W = input.size()
        F, _, K, K = weights.size()

        H_out = int((H + 2 * padding - K) // stride + 1)
        W_out = int((H + 2 * padding - K) // stride + 1)

        output = torch.zeros((N, F, H_out, W_out))
        input_pad = input
        output = TF.conv2d(input_pad, weights, padding=0, stride=stride)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weights = ctx.saved_tensors
        padding = 0
        
        N, F, H_out, W_out = grad_output.size()
        N, C, H, W = input.size()
        F, _, K, K = weights.size()

        input_pad = input
        grad_input_pad = torch.zeros((N, C, H + 2 * padding, W + 2 * padding))
        grad_input = torch.zeros_like(input)
        grad_weights = torch.zeros((C, F, K, K)).to(DEVICE)
        if ctx.needs_input_grad[0]:
            grad_input_pad = TF.conv_transpose2d(grad_output, weights)
            grad_input = grad_input_pad
        if ctx.needs_input_grad[1]:
            input_pad = input_pad.permute(1, 0, 2, 3)
            grad_output = grad_output.permute(1, 0, 2, 3)
            for n in range(N):
                grad_weights += _conv2dfft_single_noise_v2(input_pad[:, n, :, :].unsqueeze(1),
                                                        grad_output[:, n, :, :].unsqueeze(1), LAYER_BOUND, SIGMA, N)
            grad_weights = torch.flip(grad_weights.permute(1, 0, 2, 3), dims=[2, 3])
        return grad_input, grad_weights
    

class Conv2d_BC_no_padding(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k: int):
        super(Conv2d_BC_no_padding, self).__init__()
        self.k = k
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weights = nn.Parameter(torch.empty((self.out_channels, self.in_channels, k, k)))
        n = in_channels * k * k
        stdv = 1. / math.sqrt(n)
        self.weights.data.uniform_(-stdv, stdv)
        self.conv2d = conv2d_fft_no_padding()

    def forward(self, x):
        return self.conv2d.apply(x, self.weights)