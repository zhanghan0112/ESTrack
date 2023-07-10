import torch.nn as nn
import torch
import torch.nn.functional as F

class adjust_z_cls(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(adjust_z_cls, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.downsample(x)
        return x

class adjust_z_loc(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(adjust_z_loc, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.downsample(x)
        return x

class adjust_x_cls(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(adjust_x_cls, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.downsample(x)
        return x

class adjust_x_loc(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(adjust_x_loc, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.downsample(x)
        return x

class AdjustLayer(nn.Module):
    def __init__(self, in_channels, out_channels, center_size=7):
        super(AdjustLayer, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            )
        self.center_size = center_size

    def forward(self, x):
        x = self.downsample(x)
        if x.size(3) < 20:
            l = (x.size(3) - self.center_size) // 2
            r = l + self.center_size
            x = x[:, :, l:r, l:r]
        return x

def xcorr_depthwise(x, kernel):
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch*channel, x.size(2), x.size(3))
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch*channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out

def xcorr_pixelwise(search, kernel):
    """pixelwise cross correlation
    """
    b, c, h, w = search.shape
    ker = kernel.reshape(b, c, -1).transpose(1, 2)
    feat = search.reshape(b, c, -1)
    corr = torch.matmul(ker, feat)
    corr = corr.reshape(*corr.shape[:2], h, w)
    return corr,h,w

# def attention(search,kernel):
#     b,c,h,w = search.shape
#     src=search.reshape(b,c,-1).transpose(1,2)
#     ker = kernel.reshape(b,c,-1)
#     dist = torch.matmul(src,ker) #(b,len(src),len(ker))
#     dist = torch.softmax(dist,dim=-1)

