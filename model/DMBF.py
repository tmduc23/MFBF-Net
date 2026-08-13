import torch
import torch.nn as nn
import torch.nn.functional as F
from VSS import VSSBlock

class DMBFBlock(nn.Module):
    def __init__(self,
                 in_channels_t1: int = 16,
                 in_channels_t2: int = 32,
                 in_channels_t3: int = 64,
                 in_channels_t4: int = 128,
                 in_channels_t5: int = 256,
                 ):
        super().__init__()
        self.c1 = in_channels_t1
        self.c2 = in_channels_t2
        self.c3 = in_channels_t3
        self.c4 = in_channels_t4
        self.c5 = in_channels_t5
        self.total_c = self.c1 + self.c2 + self.c3 + self.c4 + self.c5

        c_mid = self.total_c // 4
        self.pw1 = nn.Conv2d(self.total_c, c_mid, 1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=min(4, max(1, c_mid // 4)), num_channels=c_mid, eps=1e-5, affine=True)
        self.act = nn.ReLU(inplace=True)

        # VSS: dim = c_mid//2 cho mỗi nhánh
        self.block = VSSBlock(hidden_dim=c_mid // 2)   

        # IN với eps lớn hơn một chút
        self.ins_norm = nn.InstanceNorm2d(c_mid, eps=1e-5, affine=True)
        self.act1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.pw2 = nn.Conv2d(c_mid, self.total_c, 1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=min(4, max(1, self.total_c // 8)),
                                num_channels=self.total_c, eps=1e-5, affine=True)
        
        self.norm = nn.LayerNorm(self.total_c//16)                         
        self.scale = nn.Parameter(torch.ones(1))
    @staticmethod
    def _resize(x, size):
        # Auto: downsample -> adaptive_avg_pool2d ; upsample -> bilinear interpolate
        h, w = x.shape[-2:]
        H, W = size
        if H <= h and W <= w:   # downsample
            return F.adaptive_avg_pool2d(x, (H, W))
        else:                   # upsample
            return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

    def _Rescale(self, x1, x2, x3, x4, x5):
        feature_list = [x1, x2, x3, x4, x5]
        n_sizes_list = [t.shape[-2:] for t in feature_list]            # [(H1,W1), (H2,W2), ...]
        H, W = x3.shape[-2:]                                           # normalize to size(x3), tầng trung gian
        x1 = self._resize(x1, (H, W))
        x2 = self._resize(x2, (H, W))
        x3 = self._resize(x3, (H, W))
        x4 = self._resize(x4, (H, W))
        x5 = self._resize(x5, (H, W))
        mamba_input = torch.cat([x1, x2, x3, x4, x5], dim=1)               # (B, total_c, H, W)
        # mamba_input = self.pw(mamba_input)
        return mamba_input, n_sizes_list, feature_list

    def _Scale_Original(self, mamba_out, original_sizes, feature_list):
        c1, c2, c3, c4, c5 = self.c1, self.c2, self.c3, self.c4, self.c5
        x1 = mamba_out[:, :c1, :, :]
        x2 = mamba_out[:, c1:c1+c2, :, :]
        x3 = mamba_out[:, c1+c2:c1+c2+c3, :, :]
        x4 = mamba_out[:, c1+c2+c3:c1+c2+c3+c4, :, :]
        x5 = mamba_out[:, c1+c2+c3+c4 : c1+c2+c3+c4+c5, :, :]

        x1_out = self._resize(x1, original_sizes[0])
        x2_out = self._resize(x2, original_sizes[1])
        x3_out = self._resize(x3, original_sizes[2])
        x4_out = self._resize(x4, original_sizes[3])
        x5_out = self._resize(x5, original_sizes[4])
        return x1_out, x2_out, x3_out, x4_out, x5_out


    def forward(self, x1, x2, x3, x4, x5):
        # 1) rescale & concat
        mamba_input, n_sizes_list, feature_list = self._Rescale(x1, x2, x3, x4, x5)   # (B,C,H,W)

        mamba_input = self.act(self.gn1(self.pw1(mamba_input)))            # B C/4 H W

        mamba_1, mamba_2 = torch.chunk(mamba_input, 2, dim=1)             # B C/8 H W
        res_1, res_2 = mamba_1, mamba_2

        mamba_1 = mamba_1.permute(0, 2, 3, 1)
        mamba_1 = self.block(mamba_1)
        mamba_1 = mamba_1.permute(0, 3, 1, 2)
        mamba_1 = mamba_1 + self.scale * res_1

        mamba_2 = mamba_2.permute(0, 2, 3, 1)
        mamba_2 = self.block(mamba_2)
        mamba_2 = mamba_2.permute(0, 3, 1, 2)
        mamba_2 = mamba_2 + self.scale * res_2

        fused = torch.cat([mamba_1, mamba_2], dim=1)
        fused = self.act1(self.ins_norm(fused))
        
        fused = self.act(self.gn2(self.pw2(fused)))
        # 4) split channels & resize back to original sizes
        x1_out_1, x2_out_1, x3_out_1, x4_out_1, x5_out_1 = self._Scale_Original(fused, n_sizes_list, feature_list)
        
        x1_out = x1_out_1 + x1
        x2_out = x2_out_1 + x2
        x3_out = x3_out_1 + x3
        x4_out = x4_out_1 + x4
        x5_out = x5_out_1 + x5

        return x1_out, x2_out, x3_out, x4_out, x5_out
