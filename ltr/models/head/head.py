import torch.nn as nn
import torch
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers-1) #[hidden_dim,hidden_dim,hidden_dim...]
        self.layers = nn.ModuleList(nn.Linear(n,k) for n,k in zip([input_dim]+h, h+[output_dim]))

    def forward(self,x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers-1 else layer(x)
        return x


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = 1e-5

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()  # rsqrt(x): 1/sqrt(x), r: reciprocal
        bias = b - rm * scale
        return x * scale + bias

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1, freeze_bn=False):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=False))
    if freeze_bn:
        result.add_module('bn', FrozenBatchNorm2d(out_channels))
    else:
        result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, padding_mode='zeros', freeze_bn=False):
        super(RepVGGBlock, self).__init__()
        self.groups = groups
        self.in_channels = in_channels
        assert kernel_size == 3
        assert padding == 1
        padding_11 = padding - kernel_size // 2
        self.nonlinearity = nn.ReLU()
        self.se = nn.Identity()
        self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == \
        in_channels and  stride == 1 else None
        self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                 stride=stride, padding=padding, groups=groups, freeze_bn=freeze_bn)
        self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                               stride=stride, padding=padding_11, groups=groups, freeze_bn=freeze_bn)

    def forward(self, inputs):
        id_out = self.rbr_identity(inputs)
        return self.nonlinearity(self.se(self.rbr_dense(inputs)+self.rbr_1x1(inputs)+id_out))


class Corner_Predictor_Lite_Rep(nn.Module):
    def __init__(self,feat_sz=16, stride=16, inplanes=128, channel=128):
        super().__init__()
        self.feat_sz = feat_sz
        self.feat_len = feat_sz ** 2
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        self.conv_tower = nn.Sequential(RepVGGBlock(inplanes, channel, kernel_size=3, padding=1),
                                        RepVGGBlock(channel, channel, kernel_size=3, padding=1),
                                        nn.Conv2d(channel, 2, kernel_size=3, padding=1))
        with torch.no_grad():
            self.indice = (torch.arange(0, self.feat_sz).view(-1,1)+0.5) * stride
            self.coord_x = self.indice.repeat(self.feat_sz, 1).view((self.feat_sz*self.feat_sz,)).float().cuda()
            self.coord_y = self.indice.repeat(1,self.feat_sz).view((self.feat_sz*self.feat_sz)).float().cuda()

    def get_score_map(self, x):
        score_map = self.conv_tower(x) #b,2,h,w
        return score_map[:,0,:,:], score_map[:,1,:,:]

    def soft_argmax(self, score_map, return_dist=False, softmax=True):
        score_vec = score_map.view(-1, self.feat_len) #b,sz*sz
        prob_vec = nn.functional.softmax(score_vec, dim=1)
        exp_x = torch.sum((self.coord_x*prob_vec),dim=1)
        exp_y = torch.sum((self.coord_y*prob_vec),dim=1)
        if return_dist:
            if softmax:
                return exp_x, exp_y, prob_vec
            else:
                return exp_x, exp_y, score_vec
        else:
            return exp_x, exp_y

    def forward(self, x, return_dist=False, soft_max=True):
        score_map_tl, score_map_br = self.get_score_map(x)
        if return_dist:
            coorx_tl, coory_tl, prob_vec_tl = self.soft_argmax(score_map_tl, return_dist=True, softmax=soft_max)
            coorx_br, coory_br, prob_vec_br = self.soft_argmax(score_map_br, return_dist=True, softmax=soft_max)
            return torch.stack((coorx_br,coory_tl,coorx_br,coory_br),dim=1)/self.img_sz, prob_vec_tl,prob_vec_br
        else:
            coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
            coorx_br, coory_br = self.soft_argmax(score_map_br)
        return torch.stack((coorx_tl,coory_tl,coorx_br,coory_br),dim=1)/self.img_sz

def build_box_head(settings):
    if 'corner' in settings.head_type:
        stride = 16
        feat_sz = settings.search_sz // stride
        return Corner_Predictor_Lite_Rep(feat_sz, stride)
    elif settings.head_type == 'MLP':
        hidden_dim = settings.hidden_dim
        mlp_head = MLP(hidden_dim, hidden_dim, 4, 3)
        return mlp_head
    else:
        raise ValueError()