# *************************************************************************************************** #
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from functools import partial

# 安装mmcv参考链接：https://mmcv.readthedocs.io/zh-cn/latest/get_started/installation.html#
from mmengine.model import BaseModule
from mmcv.cnn import ConvModule
# pip install torch_dct
import torch_dct as DCT
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from timm.models.layers import DropPath
from typing import Callable

# 参考文献[48]、[46]
'''
卷基层
'''
def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)

'''
BPC蓝图卷积
'''
class BPC(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size,stride=1, dilation=1):
        super(BPC, self).__init__()
        self.main = nn.Sequential(
            conv_layer(in_channels, out_channels, 1),
            conv_layer(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, groups=out_channels)
        )

    def forward(self, x):
        return self.main(x)

'''
DSC深度可分离卷积
'''
class DSC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DSC, self).__init__()
        self.main = nn.Sequential(
            conv_layer(in_channels, out_channels, 3, groups=out_channels),
            conv_layer(out_channels, out_channels, 1)
        )

    def forward(self, x):
        return self.main(x)

'''
Linear Layer的MLP
'''
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.permute(0, 3, 1, 2)  # 恢复 [B, C, H, W]
        return x

'''
借鉴[68]设计的小波变换
'''
# cA = F.interpolate(cA, scale_factor=2, mode='bicubic', align_corners=False)
def dwt_init(x):
    x01 = x[:, :, 0::2, :]
    x02 = x[:, :, 1::2, :]
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]

    min_height = min(x1.size(2), x2.size(2), x3.size(2), x4.size(2))
    min_width = min(x1.size(3), x2.size(3), x3.size(3), x4.size(3))

    x1 = x1[:, :, :min_height, :min_width]
    x2 = x2[:, :, :min_height, :min_width]
    x3 = x3[:, :, :min_height, :min_width]
    x4 = x4[:, :, :min_height, :min_width]

    x_LL = (x1 + x2 + x3 + x4) / 4
    x_HL = (-x1 - x2 + x3 + x4) / 4
    x_LH = (-x1 + x2 - x3 + x4) / 4
    x_HH = (x1 - x2 - x3 + x4) / 4

    return x_LL, x_HL, x_LH, x_HH
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)

def idwt_init(x_LL, x_HL, x_LH, x_HH):
    min_height = min(x_LL.size(2), x_HL.size(2), x_LH.size(2), x_HH.size(2))
    min_width = min(x_LL.size(3), x_HL.size(3), x_LH.size(3), x_HH.size(3))

    x_LL = x_LL[:, :, :min_height, :min_width]
    x_HL = x_HL[:, :, :min_height, :min_width]
    x_LH = x_LH[:, :, :min_height, :min_width]
    x_HH = x_HH[:, :, :min_height, :min_width]

    x1 = x_LL - x_HL - x_LH + x_HH
    x2 = x_LL - x_HL + x_LH - x_HH
    x3 = x_LL + x_HL - x_LH - x_HH
    x4 = x_LL + x_HL + x_LH + x_HH

    # 拼接回原始尺寸
    upper = torch.zeros(x1.size(0), x1.size(1), x1.size(2)*2, x1.size(3), device=x1.device, dtype=x1.dtype)
    lower = torch.zeros_like(upper)

    upper[:, :, 0::2, :] = x1
    upper[:, :, 1::2, :] = x2
    lower[:, :, 0::2, :] = x3
    lower[:, :, 1::2, :] = x4

    x = torch.zeros(x1.size(0), x1.size(1), upper.size(2), upper.size(3)*2, device=x1.device, dtype=x1.dtype)
    x[:, :, :, 0::2] = upper
    x[:, :, :, 1::2] = lower

    return x
class IDWT(nn.Module):
    def __init__(self):
        super(IDWT, self).__init__()
        self.requires_grad = False

    def forward(self, x_LL, x_HL, x_LH, x_HH):
        return idwt_init(x_LL, x_HL, x_LH, x_HH)

'''
(This is the official implementation of the AAAI2025 paper HS-FPN: High Frequency and Spatial Perception FPN for Tiny Object Detection.)
HFP (High Frequency Perception Module)
'''
# ------------------------------------------------------------------#
# Spatial Path of HFP
# Only p1&p2 use dct to extract high_frequency response
# ------------------------------------------------------------------#
class DctSpatialInteraction(BaseModule):
    def __init__(self,
                 in_channels,
                 ratio,
                 isdct=True,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(DctSpatialInteraction, self).__init__(init_cfg)
        self.ratio = ratio
        self.isdct = isdct  # true when in p1&p2 # false when in p3&p4
        if not self.isdct:
            self.spatial1x1 = nn.Sequential(
                *[ConvModule(in_channels, 1, kernel_size=1, bias=False)]
            )

        # 新增：高频噪声过滤的可学习参数
        self.noise_threshold = nn.Parameter(torch.tensor(0.1))  # 控制高频抑制强度

    def forward(self, x):
        x, x_HL, x_LH, x_HH = dwt_init(x)
        _, _, h0, w0 = x.size()
        if not self.isdct:
            return x * torch.sigmoid(self.spatial1x1(x))
        idct = DCT.dct_2d(x, norm='ortho')
        weight = self._compute_weight(h0, w0, self.ratio).to(x.device)

        # 新增：抑制过高频的噪声（基于阈值裁剪）
        idct = torch.where(torch.abs(idct) > self.noise_threshold, torch.tanh(idct), idct)

        weight = weight.view(1, h0, w0).expand_as(idct)
        dct = idct * weight  # filter out low-frequency features
        dct_ = DCT.idct_2d(dct, norm='ortho')  # generate spatial mask
        out = idwt_init(x * dct_, x_HL * dct_, x_LH * dct_, x_HH * dct_)
        return out

    def _compute_weight(self, h, w, ratio):
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0
        return weight

# ------------------------------------------------------------------#
# Channel Path of HFP
# Only p1&p2 use dct to extract high_frequency response
# ------------------------------------------------------------------#
class DctChannelInteraction(BaseModule):
    def __init__(self,
                 in_channels,
                 patch,
                 ratio,
                 isdct=True,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')
                 ):
        super(DctChannelInteraction, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.h = patch[0]
        self.w = patch[1]
        self.ratio = ratio
        self.isdct = isdct
        self.channel1x1 = nn.Sequential(
            *[ConvModule(in_channels, in_channels, kernel_size=1, groups=in_channels, bias=False)],
        )
        self.channel2x1 = nn.Sequential(
            *[ConvModule(in_channels, in_channels, kernel_size=1, groups=in_channels, bias=False)],
        )
        self.gelu = nn.GELU()

    def forward(self, x):
        n, c, h, w = x.size()
        if not self.isdct:  # true when in p1&p2 # false when in p3&p4
            amaxp = F.adaptive_max_pool2d(x, output_size=(1, 1))
            aavgp = F.adaptive_avg_pool2d(x, output_size=(1, 1))
            channel = self.channel1x1(self.gelu(amaxp)) + self.channel1x1(self.gelu(aavgp))  # 2025 03 15 szc
            return x * torch.sigmoid(self.channel2x1(channel))

        idct = DCT.dct_2d(x, norm='ortho')
        weight = self._compute_weight(h, w, self.ratio).to(x.device)
        weight = weight.view(1, h, w).expand_as(idct)
        dct = idct * weight  # filter out low-frequency features
        dct_ = DCT.idct_2d(dct, norm='ortho')

        amaxp = F.adaptive_max_pool2d(dct_, output_size=(self.h, self.w))
        aavgp = F.adaptive_avg_pool2d(dct_, output_size=(self.h, self.w))
        amaxp = torch.sum(self.gelu(amaxp), dim=[2, 3]).view(n, c, 1, 1)
        aavgp = torch.sum(self.gelu(aavgp), dim=[2, 3]).view(n, c, 1, 1)

        # channel = torch.cat([self.channel1x1(aavgp), self.channel1x1(amaxp)], dim = 1) # TODO: The values of aavgp and amaxp appear to be on different scales. Add is a better choice instead of concate.
        channel = self.channel1x1(amaxp) + self.channel1x1(aavgp)  # 2025 03 15 szc
        return x * torch.sigmoid(self.channel2x1(channel))

    def _compute_weight(self, h, w, ratio):
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0
        return weight

# 输入为尺寸为[B,C,H,W]
class HFP(BaseModule):
    def __init__(self,
                in_channels,
                ratio=(0.25, 0.25),           # None       (0.25, 0.25)
                patch = (8,8),   #(8,8),(16,16)
                isdct = True,
                init_cfg=dict(
                    type='Xavier', layer='Conv2d', distribution='uniform')):
        super(HFP, self).__init__(init_cfg)
        self.spatial = DctSpatialInteraction(in_channels, ratio=ratio, isdct = isdct)
        self.channel = DctChannelInteraction(in_channels, patch=patch, ratio=ratio, isdct = isdct)
        self.out =  nn.Sequential(
            *[ConvModule(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(in_channels, in_channels)]
            )
    def forward(self, x):
        spatial = self.spatial(x) # output of spatial path
        channel = self.channel(x) # output of channel path
        return self.out(spatial + channel)

'''
EMA注意力
'''
class EMA(nn.Module):  # 定义一个继承自 nn.Module 的 EMA 类
    def __init__(self, channels, c2=None, factor=8):  # 构造函数，初始化对象
        super(EMA, self).__init__()  # 调用父类的构造函数
        self.groups = factor  # 定义组的数量为 factor，默认值为 32
        assert channels // self.groups > 0  # 确保通道数可以被组数整除
        self.softmax = nn.Softmax(-1)  # 定义 softmax 层，用于最后一个维度
        self.agp = nn.AdaptiveAvgPool2d((1, 1))  # 定义自适应平均池化层，输出大小为 1x1
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 定义自适应平均池化层，只在宽度上池化
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 定义自适应平均池化层，只在高度上池化
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)  # 定义组归一化层
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1,
                                 padding=0)  # 定义 1x1 卷积层
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1,
                                 padding=1)  # 定义 3x3 卷积层

    def forward(self, x):  # 定义前向传播函数
        b, c, h, w = x.size()  # 获取输入张量的大小：批次、通道、高度和宽度
        group_x = x.reshape(b * self.groups, -1, h, w)  # 将输入张量重新形状为 (b * 组数, c // 组数, 高度, 宽度)
        x_h = self.pool_h(group_x)  # 在高度上进行池化
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)  # 在宽度上进行池化并交换维度
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))  # 将池化结果拼接并通过 1x1 卷积层
        x_h, x_w = torch.split(hw, [h, w], dim=2)  # 将卷积结果按高度和宽度分割
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())  # 进行组归一化，并结合高度和宽度的激活结果
        x2 = self.conv3x3(group_x)  # 通过 3x3 卷积层
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))  # 对 x1 进行池化、形状变换、并应用 softmax
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # 将 x2 重新形状为 (b * 组数, c // 组数, 高度 * 宽度)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))  # 对 x2 进行池化、形状变换、并应用 softmax
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # 将 x1 重新形状为 (b * 组数, c // 组数, 高度 * 宽度)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)  # 计算权重
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)  # 应用权重并将形状恢复为原始大小

'''
SS2D---参考[68]    ES2D
'''
# ********************************* SS2D ********************************* #
class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y
class CAB(nn.Module):
    def __init__(self, num_feat, is_light_sr= False, compress_ratio=3,squeeze_factor=30):
        super(CAB, self).__init__()
        if is_light_sr: # a larger compression ratio is used for light-SR
            compress_ratio = 6
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)
# 此SS2D用于block
class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out
# 此SS2D用于单独使用
class SS2D2(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        out = out.permute(0, 3, 1, 2)  # 恢复 [B, C, H, W]
        return out
# class VSSBlock(nn.Module):
#     def __init__(
#             self,
#             hidden_dim: int = 0,
#             drop_path: float = 0,
#             norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
#             attn_drop_rate: float = 0,
#             d_state: int = 16,
#             expand: float = 2.,
#             is_light_sr: bool = False,
#             **kwargs,
#     ):
#         super().__init__()
#         self.ln_1 = norm_layer(hidden_dim)
#         self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)
#         self.drop_path = DropPath(drop_path)
#         self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
#         self.spatial_scale = None  # 延迟初始化
#         self.conv_blk = CAB(hidden_dim,is_light_sr)
#         self.ln_2 = nn.LayerNorm(hidden_dim)
#         self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))
#         self.spatial_scale2 = None  # 延迟初始化
#
#     def forward(self, input):
#         # Input shape: [B, C, H, W]
#         B, C, H, W = input.shape
#         input = input.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
#
#         # 如果是第一次运行，初始化 spatial_scale
#         if self.spatial_scale is None:
#             self.spatial_scale = nn.Parameter(torch.ones(H, W))  # [H, W]
#             self.register_parameter("spatial_scale", self.spatial_scale)  # 注册为可学习参数
#         # 如果是第一次运行，初始化 spatial_scale
#         if self.spatial_scale2 is None:
#             self.spatial_scale2 = nn.Parameter(torch.ones(H, W))  # [H, W]
#             self.register_parameter("spatial_scale", self.spatial_scale2)  # 注册为可学习参数
#
#         x = self.ln_1(input)
#         x = input*self.skip_scale + self.drop_path(self.self_attention(x))
#         x = x*self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
#         # Convert back to [B, C, H, W]
#         x = x.permute(0, 3, 1, 2).contiguous()
#         return x

class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            is_light_sr: bool = False, **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state, expand=expand, dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))
        self.spatial_scale = None  # 延迟初始化
        self.conv_blk = CAB(hidden_dim, is_light_sr)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))
        self.spatial_scale2 = None  # 延迟初始化

    def forward(self, input):
        # Input shape: [B, C, H, W]
        B, C, H, W = input.shape
        input = input.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]

        # 初始化 spatial_scale（第一次运行时）
        if self.spatial_scale is None:
            self.spatial_scale = nn.Parameter(torch.ones(H, W))  # [H, W]
            self.register_parameter("spatial_scale", self.spatial_scale)  # 正确注册第一个参数

        # 初始化 spatial_scale2（第一次运行时）
        if self.spatial_scale2 is None:
            self.spatial_scale2 = nn.Parameter(torch.ones(H, W))  # [H, W]
            # 关键修复：使用不同的名称"spatial_scale2"注册第二个参数
            self.register_parameter("spatial_scale2", self.spatial_scale2)

        x = self.ln_1(input)
        x = input * self.skip_scale + self.drop_path(self.self_attention(x))
        x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3,
                                                                                                        1).contiguous()
        # Convert back to [B, C, H, W]
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    # 重写参数加载逻辑，处理延迟初始化的参数
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # 检查需要动态创建的参数是否在state_dict中
        spatial_scale_key = prefix + "spatial_scale"
        spatial_scale2_key = prefix + "spatial_scale2"

        # 如果state_dict中有spatial_scale，但当前模块中还没有，则创建它
        if spatial_scale_key in state_dict and self.spatial_scale is None:
            # 从state_dict中获取参数形状和值
            scale_value = state_dict[spatial_scale_key]
            self.spatial_scale = nn.Parameter(scale_value)
            self.register_parameter("spatial_scale", self.spatial_scale)

        # 处理spatial_scale2
        if spatial_scale2_key in state_dict and self.spatial_scale2 is None:
            scale2_value = state_dict[spatial_scale2_key]
            self.spatial_scale2 = nn.Parameter(scale2_value)
            self.register_parameter("spatial_scale2", self.spatial_scale2)

        # 调用父类方法完成剩余参数加载
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs
        )


# *********************************** 下面是自己定义的模块CRFFM和STFRM *********************************** #
class STFRM(nn.Module):
    def __init__(self, dim, mlp_ratio=4.):
        """
        Args:
            dim (int): 输入通道数 C
            mlp_ratio (float): MLP隐藏层扩展比例（默认4）
            drop (float): Dropout率
        """
        super().__init__()

        # MLP（通道维处理，保持空间结构）
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim
        )
        # 您的HFP模块（假设输入输出同尺寸）
        self.hfp = HFP(dim)
        # GELU激活
        self.act = nn.GELU()
        # 您的EMA模块（假设输入输出同尺寸）
        self.ema1 = EMA(dim)

        self.vss = VSSBlock(dim)  # VSSBlock   SS2D2
        self.ema2 = EMA(dim)

        # 新增：特征融合注意力（筛选有效特征）
        self.fusion_attn = nn.Sequential(
            conv_layer(2*dim, dim, kernel_size=1),  # 融合x1和x2的特征
            nn.Sigmoid()
        )


    def forward(self, x):
        """
        输入: [B, C, H, W]
        输出: [B, C, H, W]
        """
        # --- 第一次处理流 ---
        # 层归一化（需将通道维放到最后）
        x1 = self.hfp(x)
        x1 = self.mlp(x1)  # 假设您的Mlp已支持4D输入（见前文方案1）
        x1 = self.act(x1)
        x1 = self.ema1(x1)
        x2 = self.vss(x)
        x2 = self.ema2(x2)

        # 改进：通过注意力加权融合，而非直接相加
        fusion_weight = self.fusion_attn(torch.cat([x1, x2], dim=1))
        out = x1 * fusion_weight + x2 * (1 - fusion_weight)  # 动态选择更优特征
        return out

class CRFFM(nn.Module):
    def __init__(self, dim):
        """
        Args:
            dim (int): 输入总通道数（需被4整除）
            A_module (nn.Module): 您定义的A模块（若未提供则默认使用STFRM）
        """
        super().__init__()

        # 初始化4个A模块（可共享权重或独立）
        self.A1 = STFRM(dim)


    def forward(self, x):
        """
        输入: [B, C, H, W]
        输出: [B, C, H, W]
        """


        # 级联处理
        x = self.A1(x)  # x1 = A1(x1)

        return x

class DFE(nn.Module):
    def __init__(self, dim, num):
        """
        SFEM模块 - 由n个CRFFM模块串联组成的特征提取模块

        参数:
            num: 串联的CRFFM模块数量
            dim: 每个CRFFM的输入通道数
        """
        super(DFE, self).__init__()

        # 创建n个CRFFM模块的序列
        self.crffm_blocks = nn.Sequential()
        for i in range(num):
            # 初始化每个CRFFM模块（假设CRFFM是你已经定义好的类）
            crffm = CRFFM(dim)
            self.crffm_blocks.add_module(f'crffm_{i}', crffm)

        # 新增：残差缩放系数（稳定训练）
        self.res_scale = nn.Parameter(torch.tensor(0.1))  # 初始值0.1，可学习

    def forward(self, x):
        out = self.res_scale * self.crffm_blocks(x) + x
        return out

# 定义的OutBlock---轻量级
class OutBlock_lightweight(nn.Module):
    def __init__(self, in_channel, up_scale = 4):
        super(OutBlock_lightweight, self).__init__()
        self.up_scale = up_scale
        self.main = nn.Sequential(
            conv_layer(in_channel, 3*(self.up_scale**2), kernel_size=3),
            nn.PixelShuffle(self.up_scale),
            conv_layer(3, 3, kernel_size=3)
        )
    def forward(self, x):
        return self.main(x)

class OutBlock_lightweight2(nn.Module):
    def __init__(self, in_channel, up_scale = 4):
        super(OutBlock_lightweight2, self).__init__()
        self.up_scale = up_scale
        # 新增：上采样前的平滑滤波（抑制噪声）
        self.denoise_conv = nn.Sequential(
            conv_layer(in_channel, in_channel, kernel_size=3, groups=in_channel),  # 深度可分离卷积平滑
            nn.GELU()
        )
        if up_scale == 4:
            self.main = nn.Sequential(
                conv_layer(in_channel, in_channel*2, kernel_size=3),
                nn.PixelShuffle(2),
                conv_layer(in_channel//2, in_channel, kernel_size=3),
                nn.PixelShuffle(2),
                conv_layer(in_channel//4, 3, kernel_size=3)
            )
        else:
            self.main = nn.Sequential(
                conv_layer(in_channel, in_channel*2, kernel_size=3),
                nn.PixelShuffle(2),
                conv_layer(in_channel//2, 3, kernel_size=3)
            )
    def forward(self, x):
        x = self.denoise_conv(x)  # 上采样前先平滑噪声
        return self.main(x)



'''
MYMODEL
最终的模型
链式残差特征融合模块---Chained Residual Feature Fusion Module，CRFFM
小目标特征识别模块---Small target feature recognition module，STFRM
'''
class MYMODEL(nn.Module):
    def __init__(self, up_scale=2):
        super(MYMODEL, self).__init__()
        # 设置基础通道数
        self.up_scale = up_scale
        self.basechannles = 64
        self.modulenum = 4

        self.sfe = conv_layer(3, self.basechannles, kernel_size=3)
        self.dfe = DFE(self.basechannles,self.modulenum)
        self.rec = OutBlock_lightweight2(self.basechannles, self.up_scale)

    def forward(self, input):
        out = self.sfe(input)
        out = self.dfe(out)
        out = self.rec(out)
        return out