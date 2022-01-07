import torch
from torch import nn, einsum
from einops import rearrange

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x, message=''):
        # Do your print / debug stuff here
        print(message, " ", x.shape)
        return x

class LayerNorm(nn.Module):  # layernorm, but done in the channel dimension #1
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.BatchNorm2d(dim_in),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        padding = proj_kernel // 2
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)

        self.to_q = DepthWiseConv2d(dim, inner_dim, proj_kernel, padding = padding, stride = 1, bias = False)
        self.to_kv = DepthWiseConv2d(dim, inner_dim * 2, proj_kernel, padding = padding, stride = kv_proj_stride, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        shape = x.shape
        b, n, _, y, h = *shape, self.heads
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> (b h) (x y) d', h = h), (q, k, v))

        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, y = y)
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, depth, heads, dim_head = 64, mlp_mult = 4, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, proj_kernel = proj_kernel, kv_proj_stride = kv_proj_stride, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_mult, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_sample_mode):
        super(UpBlock, self).__init__()
        if up_sample_mode == 'conv_transpose':
            #self.up_sample = nn.ConvTranspose2d(in_channels-out_channels, in_channels-out_channels, kernel_size=2, stride=2)        
            self.up_sample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)        
        elif up_sample_mode == 'bilinear':
            self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            raise ValueError("Unsupported `up_sample_mode` (can take one of `conv_transpose` or `bilinear`)")
        self.double_conv = DoubleConv(in_channels, out_channels)

    #def forward(self, down_input, skip_input= torch.Tensor().to(device)):
    #    x = self.up_sample(down_input)
    #    x = torch.cat([x, skip_input], dim=1)
    #    return self.double_conv(x)
    
    def forward(self, down_input):        
        x = self.up_sample(down_input)        
        #x = torch.cat([x, skip_input], dim=1)
        return self.double_conv(x)

class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_rate, up_mode='bilinear'):
        super(UpSampleBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding='same'
        )
        # self.batch_norm = nn.BatchNorm2d(out_channels, eps=norm_rate)
        self.layer_norm = LayerNorm(out_channels, eps=norm_rate)
        self.activation = nn.ReLU()

        self.upsample = nn.Upsample(scale_factor=2, mode=up_mode, align_corners=True)

    def forward(self, x, skip=None):
        # print("DEBUG")

        # if skip != None:
        #     PrintLayer()(x, 'X debug')
        #     try:
        #         PrintLayer()(skip, 'SKIP debug')
        #     except AttributeError as err:
        #         print("Can't print layer")
        
        # if skip != None:
        #     x = torch.cat([x, skip], dim=1)
        
        x = self.upsample(x)
        x = self.conv(x)
        # x = self.batch_norm(x)
        x = self.layer_norm(x)
        return x

class UpBlockskip(nn.Module):
    def __init__(self, in_channels, out_channels, up_sample_mode):
        super(UpBlockskip, self).__init__()
        if up_sample_mode == 'conv_transpose':
            #self.up_sample = nn.ConvTranspose2d(in_channels-out_channels, in_channels-out_channels, kernel_size=2, stride=2)        
            self.up_sample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)        
        elif up_sample_mode == 'bilinear':
            self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            raise ValueError("Unsupported `up_sample_mode` (can take one of `conv_transpose` or `bilinear`)")
        self.double_conv = DoubleConv(in_channels, out_channels)

    #def forward(self, down_input, skip_input= torch.Tensor().to(device)):
    #    x = self.up_sample(down_input)
    #    x = torch.cat([x, skip_input], dim=1)
    #    return self.double_conv(x)
    
    def forward(self, down_input, skip_input):        
        x = torch.cat([down_input, skip_input], dim=1)
        x = self.up_sample(x)          
        return self.double_conv(x)

class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, strides=1):
        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=strides,
            padding='same'
        )

        activation = nn.Softmax(dim=1)
        super(SegmentationHead, self).__init__(
            conv, 
            activation
        )

class ConvolutionalBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, strides, padding):

        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=strides,
            padding=padding,
        )
        
        activation = nn.ReLU()
        bn = nn.BatchNorm2d(out_channels)
        # norm = LayerNorm(out_channels, eps=1e-4)

        super(ConvolutionalBlock, self).__init__(
            conv, 
            activation,
            bn 
            # norm
        )

class ConnectionComponents(nn.Module):
    def __init__(self, in_channels, out_channels, norm_rate, kernel_size=3, strides=1):
        super(ConnectionComponents, self).__init__()

        self.conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding='same'
        )

        self.conv_2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=strides,
            padding='same'
        )

        self.activation_1 = nn.ReLU()
        self.activation_2 = nn.ReLU()

        self.bach_norm_1 = nn.BatchNorm2d(1, eps=norm_rate)
        self.bach_norm_2 = nn.BatchNorm2d(out_channels, eps=norm_rate)
        self.bach_norm_3 = nn.BatchNorm2d(out_channels, eps=norm_rate)

    def forward(self, x):
        shortcut = x
        path_1 = self.conv_1(shortcut)
        path_1 = self.bach_norm_1(path_1)
        
        # conv 3x3
        path_2 = self.conv_2(x)
        path_2 = self.bach_norm_2(path_2)
        path_2 = self.activation_2(path_2)

        # add layer
        out = path_1 + path_2
        out = self.activation_1(out)
        out = self.bach_norm_3(out)
        return out

class EncoderDecoderConnections(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, norm_rate=1e-4):
        super(EncoderDecoderConnections, self).__init__()

        self.con_comp_1 = ConnectionComponents(
            in_channels=in_channels,
            out_channels=out_channels,
            norm_rate=norm_rate,
            kernel_size=kernel_size,
        )

        # self.con_comp_2 = ConnectionComponents(
        #     in_channels=in_channels,
        #     out_channels=out_channels,
        #     norm_rate=norm_rate,
        #     kernel_size=kernel_size,
        # )

        # self.con_comp_3 = ConnectionComponents(
        #     in_channels=in_channels,
        #     out_channels=out_channels,
        #     norm_rate=norm_rate,
        #     kernel_size=kernel_size,
        # )

        # self.con_comp_4 = ConnectionComponents(
        #     in_channels=in_channels,
        #     out_channels=out_channels,
        #     norm_rate=norm_rate,
        #     kernel_size=kernel_size,
        # )

    def forward(self, x):
        x = self.con_comp_1(x)
        # x = self.con_comp_2(x)
        # x = self.con_comp_3(x)
        # x = self.con_comp_4(x)
        return x