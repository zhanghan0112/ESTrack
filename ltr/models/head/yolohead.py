import torch
import torch.nn as nn

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

#11.08 c1: 1280; c2: 256; e:0.2 c_:256 input:(1280,10,10)->256
class SPPCSPC(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.4, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))

class ELAN(nn.Module):
    def __init__(self,c1,c2):
        super(ELAN,self).__init__()
        self.cbs1=Conv(c1, c2)
        self.cbs2=Conv(c1, c2)
        self.cbs3=Conv(c2, c2//2, k=3)
        self.cbs4=Conv(c2//2, c2//2, k=3)
        self.cbs5=Conv(c2//2, c2//2, k=3)
        self.cbs6=Conv(c2//2, c2//2, k=3)
        c=(c2//2)*4+c2*2
        self.cbs7=Conv(c,c2)

    def forward(self,x):
        x1=self.cbs1(x)
        x2=self.cbs2(x)
        x3=self.cbs3(x2)
        x4=self.cbs4(x3)
        x5=self.cbs5(x4)
        x6=self.cbs6(x5)
        x7=torch.cat([x1,x2,x3,x4,x5,x6],dim=1)
        x8=self.cbs7(x7)
        return x8

#第4个satge的输入通道,中间通道：512,输出通道[1280,512,256]
class Yoloneck(nn.Module):
    def __init__(self,in_channel,mid_channel,out_channel):
        super(Yoloneck,self).__init__()
        self.sppcspc=SPPCSPC(in_channel,mid_channel)
        self.cbs1=Conv(mid_channel,out_channel)
        self.upsample=nn.Upsample(scale_factor=2)
        self.cbs2=Conv(192,out_channel)
        self.elan=ELAN(mid_channel,out_channel)

    def forward(self,x4,x3):
        x4_spp=self.sppcspc(x4) #(512,10,10)
        x4_cbs1=self.cbs1(x4_spp) #(256,10,10)
        x4_up=self.upsample(x4_cbs1) #(256,20,20)
        x3_cbs2=self.cbs2(x3) #(256,20,20)
        x_cat=torch.cat([x3_cbs2,x4_up], dim=1) #(512,20,20)
        x_elan=self.elan(x_cat) #(256,20,20)
        return x_elan

class Yoloneckv2(nn.Module):
    def __init__(self,in_channel,mid_channel,out_channel):
        super(Yoloneckv2,self).__init__()
        self.sppcspc=SPPCSPC(in_channel,mid_channel)
        self.cbs1=Conv(mid_channel,out_channel)
        # self.upsample=nn.Upsample(scale_factor=2)
        # self.cbs2=Conv(192,out_channel)
        # self.elan=ELAN(mid_channel,out_channel)

    def forward(self,x4):
        x4_spp=self.sppcspc(x4) #(512,10,10)
        x4_cbs1=self.cbs1(x4_spp) #(256,10,10)
        # x4_up=self.upsample(x4_cbs1) #(256,20,20)
        # x3_cbs2=self.cbs2(x3) #(256,20,20)
        # x_cat=torch.cat([x3_cbs2,x4_up], dim=1) #(512,20,20)
        # x_elan=self.elan(x_cat) #(256,20,20)
        return x4_cbs1

class RepConv(nn.Module):
    # Represented convolution
    # https://arxiv.org/abs/2101.03697

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, deploy=False):
        super(RepConv, self).__init__()

        self.deploy = deploy
        self.groups = g
        self.in_channels = c1
        self.out_channels = c2

        assert k == 3
        assert autopad(k, p) == 1

        padding_11 = autopad(k, p) - k // 2

        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        if deploy:
            self.rbr_reparam = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)

        else:
            self.rbr_identity = (nn.BatchNorm2d(num_features=c1) if c2 == c1 and s == 1 else None)

            self.rbr_dense = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2),
            )

            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d( c1, c2, 1, s, padding_11, groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2),
            )

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:   #train时走此路
            id_out = self.rbr_identity(inputs)

        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

class ImplicitA(nn.Module):
    def __init__(self, channel, mean=0., std=.02):
        super(ImplicitA, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit + x

class ImplicitM(nn.Module):
    def __init__(self, channel, mean=1., std=.02):
        super(ImplicitM, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit * x

class IDetect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export
    end2end = False
    include_nms = False
    concat = False

    def __init__(self, nc=1, channel=256, num_convs=4):  #anchors:torch.tensor([[12,16, 19,36, 40,28]]) ch:torch.tensor([])
        super(IDetect, self).__init__()
        # self.nc = nc  # number of classes
        self.no = nc + 4  # number of outputs per anchor
        self.grid_num = 400
        # self.nl = len(anchors)  # number of detection layers
        # self.na = len(anchors[0]) // 2  # number of anchors  #3
        # self.grid = [torch.zeros(1)] * self.nl  # init grid
        # a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        # self.register_buffer('anchors', a)  # shape(num_layer,num_anchor,2) (1,3,2)
        # self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2) (1,1,3,1,1,2)
        # self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv 多个检测层
        # self.ia = nn.ModuleList(ImplicitA(x) for x in ch)
        # self.im = nn.ModuleList(ImplicitM(self.no * self.na) for _ in ch)
        # self.m = nn.Conv2d(channel,self.no,1) #一个检测层
        cls_tower=[]
        for l in range(num_convs):
            cls_tower.append(
                nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
            )
            cls_tower.append(nn.GroupNorm(32, channel))
            cls_tower.append(nn.ReLU())
        self.m=nn.Conv2d(channel, 5, kernel_size=1, stride=1, padding=0)
        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.ia = ImplicitA(channel)
        self.im = ImplicitM(self.no)

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training = True
        x=self.m(self.cls_tower(self.ia(x)))
        x=self.im(x)
        bs, _, ny, nx = x.shape #(bs,6,20,20)
        x = x.view(bs, self.no, ny, nx).permute(0, 2, 3, 1).contiguous() #(bs,20,20,7)
        if not self.training:  # inference
            self.grid = self._make_grid(nx, ny).to(x.device)

            y = x.sigmoid()
            # y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[0]) * self.stride[0]  # xy [0:2]长宽的0.25
            # y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[0]  # wh [2:4]相对于anchor的长宽的平方根
            z.append(y.view(bs, -1, self.no))
        # for i in range(self.nl):
        #     x[i] = self.m[i](self.ia[i](x[i]))  # conv
        #     x[i] = self.im[i](x[i])
        #     bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        #     x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        #
        #     if not self.training:  # inference
        #         if self.grid[i].shape[2:4] != x[i].shape[2:4]:
        #             self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
        #
        #         y = x[i].sigmoid()
        #         y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
        #         y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
        #         z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        grid_xy=torch.stack([xv,yv],dim=-1)
        return grid_xy.view((1, ny*nx, 2)).float()

if __name__=='__main__':
    #sppcspc
    # model=SPPCSPC(1280,256)
    # pic=torch.randn(1,1280,20,20).cuda()
    # model.cuda()
    # output=model(pic)
    # print(output.shape)

    #neck
    # model=Yoloneck(1280,512,256)
    # x4 = torch.randn(1, 1280, 10, 10).cuda()
    # x3=torch.randn(1, 192, 20, 20).cuda()
    # model.cuda()
    # output = model(x4,x3)
    # print(output.shape)

    #repconv
    model=RepConv(256,256)
    x3 = torch.randn(1, 256, 20, 20).cuda()
    model.cuda()
    output = model(x3)
    print(output.shape)