import torch
import torch.nn as nn
import torch.nn.functional as F



class AxialDW(nn.Module):
    def __init__(self, dim, mixer_kernel, dilation = 1):
        super().__init__()
        h, w = mixer_kernel
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding='same', groups = dim, dilation = dilation)
        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding='same', groups = dim, dilation = dilation)

    def forward(self, x):
        x = x + self.dw_h(x) + self.dw_w(x)
        return x

class Multi_scale_Axial(nn.Module):
    def __init__(self, dim, reduction = 8):
        super().__init__()
        # gc = dim // 4
        reduced = max(1, dim//reduction)
        self.pw1 = nn.Conv2d(dim, reduced, kernel_size=1)
        self.dw1 = AxialDW(reduced, mixer_kernel=(3, 3), dilation=1)     # 16
        self.dw2 = AxialDW(reduced, mixer_kernel=(3, 3), dilation=3)
        self.dw3 = AxialDW(reduced, mixer_kernel=(3, 3), dilation=5)

        self.bn = nn.BatchNorm2d(reduced)
        self.pw2 = nn.Conv2d(3*reduced, 1, kernel_size=1)
        self.act = nn.ReLU()

    def forward(self, x):
        x0 = x
        x = self.pw1(x)
        x = torch.cat([self.act(self.bn(self.dw1(x))), self.act(self.bn(self.dw2(x))), self.act(self.bn(self.dw3(x)))], 1)
        x = self.pw2(x)
        x = torch.sigmoid(x)
        return x * x0

class PCA_C(nn.Module):

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.dw = nn.Conv2d(channels, channels, kernel_size=kernel_size,
                            groups=channels, padding=padding, bias=False)
        self.pw = nn.Conv2d(channels, channels, kernel_size = 1) 
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.prob = nn.Softmax(dim=1)

    def forward(self, x):
        # x: (B, C, H, W)
        assert x.dim() == 4
        c = x.mean(dim=(2, 3))             # (B, C)
        x_conv = self.relu(self.bn(self.dw(x)))
        x_conv = self.pw(x_conv)
        #x_conv = self.dw(x)
        c_ = x_conv.mean(dim=(2, 3))       # (B, C)
        raise_ch = self.prob(c_ - c)       # (B, C)
        att_score = torch.sigmoid(c_ * (1.0 + raise_ch))  # (B, C)
        att_score = att_score.unsqueeze(-1).unsqueeze(-1) # (B, C, 1, 1)
        return x_conv, att_score


class PMAABlock(nn.Module):
    """TrippleAttention combining H-, W-, and C- axis attention.
    Input: x shape (B, C, H, W)
    """
    def __init__(self, gate_channels, height, width, reduction_ratio=16):
        super().__init__()
        self.gate_channels = gate_channels
        self.ChannelGateH = Multi_scale_Axial(height)
        self.ChannelGateW = Multi_scale_Axial(width)
        self.ChannelGateC = Multi_scale_Axial(gate_channels)

        self.ChannelGate_out = PCA_C(gate_channels)
    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x, attn = self.ChannelGate_out(x)

        x_perm_h = x.permute(0, 2, 1, 3).contiguous()           # (B, H, C, W)
        x_pool_h = self.ChannelGateH(x_perm_h)                  # (B, H, C, W)
        x_h = x_pool_h.permute(0, 2, 1, 3).contiguous()         # (B, C, H, W)

        x_perm_w = x.permute(0, 3, 1, 2).contiguous()           # (B, W, H, C)
        x_pool_w = self.ChannelGateW(x_perm_w)                  # (B, W, H, C)
        x_w = x_pool_w.permute(0, 2, 3, 1).contiguous()         # (B, C, H, W)

        # --- C branch (channel-wise) ---
        x_pool_c = self.ChannelGateC(x)                         # (B, C, H, W)
        # Q, K, V
        out = x_pool_c + x_h + x_w
        out = out * attn

        return out
