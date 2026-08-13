import torch
import torch.nn as nn
import torch.nn.functional as F
from model.FMPA import FMPABlock
from model.DMBF import DMBFBlock
from model.PMAA import PMAABlock

class AxialDW(nn.Module):
    def __init__(self, dim, mixer_kernel, dilation = 1):
        super().__init__()
        h, w = mixer_kernel
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding='same', groups = dim, dilation = dilation)
        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding='same', groups = dim, dilation = dilation)

    def forward(self, x):
        x = x + self.dw_h(x) + self.dw_w(x)
        return x

class Encoder_Axial(nn.Module):
      def __init__(self, in_c, out_c, mixer_kernel = (7,7)):
          super().__init__()
          self.adw = AxialDW(in_c, mixer_kernel = mixer_kernel )
          self.bn = nn.BatchNorm2d(in_c)
          self.pw = nn.Conv2d(in_c, out_c, kernel_size=1)
          self.down = nn.MaxPool2d((2,2))
          self.act = nn.ReLU()

      def forward(self, x):
          x = self.adw(x)
          skip = self.act(self.bn(x))
          x = self.pw(skip)
          x = self.down(x)
          return x, skip



class DecoderBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.pw = nn.Conv2d(in_c + skip_c, out_c, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU()
        self.adw = AxialDW(out_c, mixer_kernel=(3, 3))
        self.pw2 = nn.Conv2d(out_c, out_c , kernel_size = 1)

    def forward(self,x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.bn(self.pw(x))
        skip = x
        x = self.act((self.pw2(self.adw(x))))
        return x

class MFBF_Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.pw_in = nn.Conv2d(1, 16, kernel_size=3, padding = 1)
        #self.sk_in = SKConv_7(16, M=2, G=16, r=4, stride=1 ,L=32)
        self.pw1 = nn.Conv2d(16, 1, kernel_size=1)
        self.pw2 = nn.Conv2d(32, 1, kernel_size=1)
        self.pw3 = nn.Conv2d(64, 1, kernel_size=1)
        self.pw4 = nn.Conv2d(128, 1, kernel_size=1)
  
        """Encoder"""
        self.e1 = FMPABlock(16, 32)
        self.e2 = FMPABlock(32, 64)
        self.e3 = FMPABlock(64, 128)
        self.e4 = FMPABlock(128, 256)
        self.e5 = FMPABlock(256, 512)

        """Multi_fusion"""
        self.multi_fusion = DMBFBlock(16, 32, 64, 128, 256)

        # """Skip connection"""

        
        """Bottle Neck"""
        #self.b5 = SKUnit(512, 512, 512, M=2, G=16, r=2, stride=1, L=32)
        # self.b5 = BottleneckPCAPSA(256)
        self.b6 = PMAABlock(512, 8, 8)


        """Decoder"""
        self.d5 = DecoderBlock(512, 256, 256)
        self.d4 = DecoderBlock(256, 128, 128)
        self.d3 = DecoderBlock(128, 64, 64)
        self.d2 = DecoderBlock(64, 32, 32)
        self.d1 = DecoderBlock(32, 16, 16)
        self.conv_out = nn.Conv2d(4, 1, kernel_size=1)
        # self.out = OutBlock(3, 1)

    def forward(self, x):
        """Encoder"""
        H, W = x.shape[2:]
        x = self.pw_in(x)
        #x = self.sk_in(x)
        x, skip1 = self.e1(x)
        x, skip2 = self.e2(x)
        x, skip3 = self.e3(x)
        x, skip4 = self.e4(x)
        x, skip5 = self.e5(x)


        """Multi_fusion"""
        skip_1, skip_2, skip_3, skip_4, skip_5 = self.multi_fusion(skip1, skip2, skip3, skip4, skip5)


        """BottleNeck"""
        # x = torch.cat([xm, xa], dim = 1)
        x = self.b6(x)

        """Decoder"""

        x5 = self.d5(x, skip_5)
        x4 = self.d4(x5, skip_4)
        x3 = self.d3(x4, skip_3)
        x2 = self.d2(x3, skip_2)
        x1 = self.d1(x2, skip_1)


        x_in4 = self.pw4(x4)
        x_in3 = self.pw3(x3)
        x_in2 = self.pw2(x2)
        x_in1 = self.pw1(x1)

        x_in4 = F.interpolate(x_in4, size=(H, W), mode="bilinear", align_corners=False)
        x_in3 = F.interpolate(x_in3, size=(H, W), mode="bilinear", align_corners=False)
        x_in2 = F.interpolate(x_in2, size=(H, W), mode="bilinear", align_corners=False)
        x_in1 = F.interpolate(x_in1, size=(H, W), mode="bilinear", align_corners=False)

        decoder_output = [x_in1, x_in2, x_in3, x_in4, x5]
        x = torch.cat([x_in4, x_in3, x_in2, x_in1], dim=1)
        x = self.conv_out(x)
        return x
        