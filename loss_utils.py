#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


import lpips  # 导入LPIPS库
def lpips_loss(img1, img2, net='alex', device='cuda'):
    """
    LPIPS（感知相似性）损失函数

    参数:
        img1 (torch.Tensor): 输入图像1，形状为 [N, C, H, W]，范围建议 [0, 1] 或 [-1, 1]
        img2 (torch.Tensor): 输入图像2，形状与 img1 相同
        net (str): 使用的网络 backbone，可选 'alex' | 'vgg' | 'squeeze'
        device (str): 计算设备，如 'cuda' 或 'cpu'

    返回:
        torch.Tensor: 标量损失值
    """
    # 初始化LPIPS模型（单例模式，避免重复加载）
    loss_fn = lpips.LPIPS(net=net, verbose=False).to(device)

    # 确保输入范围在[-1, 1]（LPIPS的默认要求）
    if img1.max() > 1.0:
        img1 = img1 / 127.5 - 1.0  # 假设输入是[0, 255]
    if img2.max() > 1.0:
        img2 = img2 / 127.5 - 1.0

    # 计算LPIPS
    return loss_fn(img1, img2).mean()


from model.ours2 import dwt_init
def wavelet_loss(pred, target, mode='l1', weight_LL=1.0, weight_High=1.0):
    """
    小波域损失函数（函数式实现）

    参数:
        pred (Tensor): 预测图像 [N,C,H,W]
        target (Tensor): 目标图像 [N,C,H,W]
        mode (str): 'l1' 或 'l2' 损失
        weight_LL (float): 低频分量权重
        weight_High (float): 高频分量权重

    返回:
        Tensor: 标量损失值
    """
    # 小波分解
    pred_LL, pred_HL, pred_LH, pred_HH = dwt_init(pred)  # 直接调用你的 dwt_init 函数
    target_LL, target_HL, target_LH, target_HH = dwt_init(target)

    # 选择损失类型
    loss_fn = F.l1_loss if mode == 'l1' else F.mse_loss

    # 计算各子带损失
    loss_LL = loss_fn(pred_LL, target_LL)
    loss_HL = loss_fn(pred_HL, target_HL)
    loss_LH = loss_fn(pred_LH, target_LH)
    loss_HH = loss_fn(pred_HH, target_HH)

    # 加权合并
    return weight_LL * loss_LL + weight_High * (loss_HL + loss_LH + loss_HH)


def tv_loss(x, beta=0.5):
    """计算总变分损失，抑制噪声"""
    dh = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).sum()
    dw = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).sum()
    return (dh + dw) ** beta

