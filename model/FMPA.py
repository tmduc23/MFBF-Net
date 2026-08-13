import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class AxialDW(nn.Module):
    def __init__(self, dim, mixer_kernel, dilation = 1):
        super().__init__()
        h,w = mixer_kernel
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h,1), padding = 'same', groups= dim, dilation= dilation)
        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1,w), padding='same', groups=dim, dilation=dilation)

    def forward(self,x):
        return x + self.dw_h(x) + self.dw_w(x)



class Conv(nn.Module):
    def __init__(self, in_c, out_c, k):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=k, padding=k//2, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.conv(x)

class DWConv(nn.Module):
    def __init__(self, in_c, k):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=k, padding=k//2, groups=in_c, bias=False),
            nn.BatchNorm2d(in_c),
            nn.GELU()
        )
    def forward(self, x):
        return self.dw(x)

class SCSA(nn.Module):
    def __init__(self, dim, head_num, attn_drop_ratio: float = 0.,):
        super().__init__()
        self.dim = dim
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.scaler = self.head_dim ** -0.5

        self.avg = nn.AvgPool2d(kernel_size=(4, 4), stride=4)
        self.max = nn.MaxPool2d(kernel_size=(4, 4), stride=4)
        self.conv_d = nn.Identity()
        self.norm = nn.GroupNorm(1, dim)

        self.q = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=False, groups=dim)
        self.k = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=False, groups=dim)
        self.v = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=False, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop_ratio)

    
    def forward(self, x):
        

         # (B, H, W, C) -> (B, C, H * W) 
        x = self.norm(x)
        x = self.avg(x)
        residual = x

        B, C, H, W = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        N = H * W
        # (B, C, H, W) -> (B, head_num, head_dim, N)
        q = rearrange(q, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=self.head_num, head_dim=self.head_dim)
        k = rearrange(k, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=self.head_num, head_dim=self.head_dim)
        v = rearrange(v, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=self.head_num, head_dim=self.head_dim)

        attn = q @ k.transpose(-2, -1) * (N ** -0.5)
        attn = self.attn_drop(attn.softmax(dim=-1))

        attn = attn @ v 

        attn = rearrange(attn, 'b head_num head_dim (h w) -> b (head_num head_dim) h w', h=H, w = W)

        attn = attn.mean((2, 3), keepdim=True)
        # attn_max  = torch.amax(attn, dim=(2, 3), keepdim=True)        
        # attn = attn_mean + attn_max
        attn = torch.sigmoid(attn)

        return attn 

class CFGC(nn.Module):
    def __init__(self, in_c, out_c, head_num=1):
        super().__init__()
        # self.conv1 = DWConv(in_c, 3)
        # self.conv2 = DWConv(in_c, 7)
        # self.adw3 = AxialDW(in_c, mixer_kernel = (3,3))
        # self.adw7 = AxialDW(in_c, mixer_kernel = (7,7))
        self.dw3x3 = nn.Conv2d(in_c//2, in_c//2, kernel_size = 3, padding = 1, groups = in_c//2)
        self.dw7x7 = nn.Conv2d(in_c//2, in_c//2, kernel_size = 7, padding = 3, groups = in_c//2)

        self.map1 = SCSA(in_c//2, head_num=head_num)
        self.map2 = SCSA(in_c//2, head_num=head_num)

        self.block_1 = VSSBlock(in_c)
        # self.block_2 = VSSBlock(in_c//2)

        self.conv3 = nn.Conv2d(in_c, out_c, 1)
        self.ins_norm = nn.InstanceNorm2d(in_c, affine=True)
        self.act = nn.LeakyReLU(negative_slope=0.01)
        
        self.bn = nn.BatchNorm2d(in_c)
        self.re = nn.ReLU()
        self.scale = nn.Parameter(torch.ones(1))

        #self.axial = AxialDW(in_c, mixer_kernel=(3,3))


    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        
        #x = self.dw3x3(x)
        x = x.permute(0, 2, 3, 1)
        x_mamba = self.block_1(x)

        x_mamba = x_mamba.permute(0, 3, 1, 2)
        x = self.act(self.ins_norm(x_mamba)) + self.scale * residual 

        # 2 nhánh depthwise
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.dw3x3(x1)
        x2 = self.dw7x7(x2)

        # Attention 1
        c1 = self.map1(x1)
        x_1 = x1 * c1 

        # Attention 2
        c2 = self.map2(x2)
        x_2 = x2 * c2 

        # Concat 2 nhánh
        x_concat = torch.cat([x_1, x_2], dim=1) 
        #x_tong = x + x_1 + x_2

        # x_concat = x_concat.permute(0, 2, 3, 1)
        # x_mamba = self.block_1(x_concat)

        # x_mamba = x_mamba.permute(0, 3, 1, 2)
        #x_mamba = self.act(self.ins_norm(x_mamba)) + self.axial(residual)

        skip_in = x_concat

        skip = self.re(self.bn(skip_in)) 

        out = self.conv3(skip)

        return out, skip


class FMPABlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.cfgc = CFGC(in_c, out_c)
        self.down = nn.MaxPool2d(2)

    def forward(self, x):
        x, skip = self.cfgc(x)
        x = self.down(x)
        return x, skip