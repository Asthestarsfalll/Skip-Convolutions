from typing import Tuple, Union

import spconv.pytorch as spconv
from spconv.pytorch import functional as Fsp
import torch
import torch.nn.functional as F
import torch.nn as nn


class NormGate(nn.Module):
    def __init__(
        self,
        threshold: float,
        in_channel: int,
        kernel_size: int = 3,
        stride: int = 1,
    ):
        super(NormGate, self).__init__()
        assert stride == 1
        self.sigmoid = nn.Sigmoid()
        self.threshold = threshold
        self.weight = torch.ones((1, in_channel, kernel_size, kernel_size))
        self.padding = (kernel_size + 1) // 2

    def forward(self, r, w):
        r = torch.abs(r)
        r = F.conv2d(r, self.weight, padding=self.padding)
        if w is not None:
            w = torch.abs(w).sum()
            r = r * w
        return torch.round(self.sigmoid(r-self.threshold))


class GumbelGate(nn.Module):
    def __init(
        self,
        in_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        threshold: float,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        tau: float = 1.
    ):
        super(GumbelGate, self).__init__()
        self.tau = tau
        self.conv = nn.Conv2d(
            in_channels,
            1,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        self.sigmoid()

    def forward(self, r):
        mask = self.sigmoid(self.conv(r))
        if self.training:
            b, _, h, w = mask.shape
            # TODO: verify correctness
            mask = F.gumbel_softmax(mask.flatten(1), tau=self.tau, hard=True)
            mask = mask.reshape(b, 1, h, w)
        else:
            mask = torch.round(mask)

        return mask


class BlockSample(nn.Module):
    def __init__(
        self,
        sample_size: int = 4
    ):
        super(BlockSample, self).__init__()
        self.downsample = nn.MaxPool2d(sample_size)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=sample_size)

    def forward(self, r):
        r = self.downsample(r)
        r = self.upsample(r)
        return r


class SkipConv2d(nn.Module):
    """
        Args:
            threshold: control the effect of skipgate
            gate_variant: the variants of skipgate
            use_blocksample: whether use blocksample
            sample_size: sample size of blocksample 
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        threshold: float,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        gate_variant: str = "Input",
        use_blocksample: bool = True,
        sample_size: int = 4
    ):
        super(SkipConv2d, self).__init__()
        assert gate_variant in ['Input', 'Output', 'Gumbel']
        self.variant = gate_variant
        self.spare_conv = spconv.SparseConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        if variant == 'Gumbel':
            self.skip_gate = GumbelGate(
                in_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias
            )
        else:
            self.skip_gate = NormGate(threshold, in_channel, 3, 1)
        self.sample = BlockSample(
            sample_size) if use_blocksample else nn.Identity()
        self.z_p = 0

    def forward(self, r):
        if variant == 'Input':
            mask = self.skip_gate(r)
        elif variant == 'Output':
            w = self.spare_conv.weight
            mask = self.skip_gate(r, w)
        r = mask * r
        r = self.sample(r)

        r = r.permute(0, 2, 3, 1)
        r = spconv.SparseConvTensor.from_dense(r)
        r = self.spare_conv(r)
        if self.z_p is None:
            self.z_p = r
        else:
            self.z_p = Fsp.sparse_add(r, self.z_p)
        return self.z_p

    def reset(self):
        self.z_p = None


# TODO: gate loss
# class GateLoss():

if __name__ == ' __main__':
    skipconv = SkipConv2d(in_channels=3, out_channels=16,
                          kernel_size=3, threshold=0.5)
    # must use cuda
    skipconv = skipconv.cuda()
    inputs = torch.randn(2, 3, 224, 224).cuda()
    inputs_p = torch.randn(2, 3, 224, 224).cuda()
    inp = inputs - inputs_p
    out = skipconv(inp)
    print(out.shape)
