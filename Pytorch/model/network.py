from torch import nn
from .blocks import *

class CvTModified(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.conv_0 = ConvolutionalBlock(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            padding=1,
            strides=1
        )

        self.conv_1 = ConvolutionalBlock(
            in_channels=16,
            out_channels=config.transformers[0]['dim'],
            kernel_size=config.transformers[0]['proj_kernel'],
            padding=(config.transformers[0]['proj_kernel']//2),
            strides=config.transformers[0]['kv_proj_stride']
        )

        self.att_1 = Transformer(
            dim=config.transformers[0]['dim'],
            proj_kernel=config.transformers[0]['proj_kernel'],
            kv_proj_stride=config.transformers[0]['kv_proj_stride'],
            depth=config.transformers[0]['depth'],
            heads=config.transformers[0]['heads'],
            mlp_mult=config.transformers[0]['mlp_mult'],
            dropout=config.transformers[0]['dropout']
        )

        self.conv_2 = ConvolutionalBlock(
            in_channels=config.transformers[0]['dim'],
            out_channels=config.transformers[1]['dim'],
            kernel_size=config.transformers[1]['proj_kernel'],
            padding=(config.transformers[1]['proj_kernel']//2),
            strides=config.transformers[1]['kv_proj_stride']
        )

        self.att_2 = Transformer(
            dim=config.transformers[1]['dim'],
            proj_kernel=config.transformers[1]['proj_kernel'],
            kv_proj_stride=config.transformers[1]['kv_proj_stride'],
            depth=config.transformers[1]['depth'],
            heads=config.transformers[1]['heads'],
            mlp_mult=config.transformers[1]['mlp_mult'],
            dropout=config.transformers[1]['dropout']
        )

        self.conv_3 = ConvolutionalBlock(
            in_channels=config.transformers[1]['dim'],
            out_channels=config.transformers[2]['dim'],
            kernel_size=config.transformers[2]['proj_kernel'],
            padding=(config.transformers[2]['proj_kernel']//2),
            strides=config.transformers[2]['kv_proj_stride']
        )

        self.att_3 = Transformer(
            dim=config.transformers[2]['dim'],
            proj_kernel=config.transformers[2]['proj_kernel'],
            kv_proj_stride=config.transformers[2]['kv_proj_stride'],
            depth=config.transformers[2]['depth'],
            heads=config.transformers[2]['heads'],
            mlp_mult=config.transformers[2]['mlp_mult'],
            dropout=config.transformers[2]['dropout']
        )

        self.bottle_neck = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.up_1 = UpSampleBlock(96, 64, config.normalization_rate)

        self.up_2 = UpSampleBlock(96, 32, config.normalization_rate)

        self.up_3 = UpSampleBlock(64, 16, config.normalization_rate)

        self.up_4 = UpSampleBlock(32, 16, config.normalization_rate)

        self.seg_head = SegmentationHead(16, config.num_classes)

    
    def forward(self, x):

        # Input: N, 1, 256, 256 | output: N, 16, 256, 256
        x = self.conv_0(x)
        skip_0 = self.layerNorm0(x)
        # PrintLayer()(skip_0)

        # Input: N, 16, 256, 256 | output: N, 32, 128, 128
        skip_1 = self.conv_1(skip_0)
        x = self.att_1(skip_1)
        # PrintLayer()(x)

        # Input: N, 32, 128, 128 | output: N, 32, 64, 64
        skip_2 = self.conv_2(x)
        x = self.att_2(skip_2)
        # PrintLayer()(x)

        # Input: N, 32, 64, 64 | output: N, 32, 32, 32
        skip_3 = self.conv_3(x)
        x = self.att_3(skip_3)
        # PrintLayer()(x)

        # Input: N, 32, 32, 32 | output: N, 64, 16, 16
        x = self.bottle_neck(x)
        # PrintLayer()(x)

        # Input: N, 64, 16, 16 | output: N, 64, 32, 32
        x = self.up_1(x, skip_3)
        # PrintLayer()(x)

        # Input: N, 64, 32, 32 | output: N, 32, 64, 64
        x = self.up_2(x, skip_2)
        # PrintLayer()(x)

        # Input: N, 128, 64, 64 | output: N, 16, 128, 128
        x = self.up_3(x, skip_1)
        # PrintLayer()(x)

        # Input: N, 16, 128, 128 | output: N, 16, 256, 256
        x = self.up_4(x, skip_0)
        # PrintLayer()(x)

        # Input: N, 16, 256, 256 | output: N, 2, 256, 256
        x = self.seg_head(x)
        # PrintLayer()(x)

        return x

class CvT(nn.Module):
    def __init__(
            self,
            *,
            num_classes=2, s1_emb_dim=32, s1_emb_kernel=5, s1_emb_stride=2,
            s1_proj_kernel=3, s1_kv_proj_stride=2, s1_heads=1, s1_depth=1, s1_mlp_mult=4,
    
            s2_emb_dim=32, s2_emb_kernel=3, s2_emb_stride=2, s2_proj_kernel=3, s2_kv_proj_stride=2,
            s2_heads=1, s2_depth=2, s2_mlp_mult=4,
    
            s3_emb_dim=32, s3_emb_kernel=3, s3_emb_stride=2, s3_proj_kernel=3,
            s3_kv_proj_stride=2, s3_heads=6, s3_depth=10, s3_mlp_mult=4, dropout=0.,
            up_sample_mode='bilinear'):
        super().__init__()
        kwargs = dict(locals())

        dim_input = 3
        layers = []

        #Attention path
        self.att_convB1 = nn.Conv2d(dim_input,
                                    s1_emb_dim,
                                    kernel_size=s1_emb_kernel,
                                    padding=(s1_emb_kernel // 2),
                                    stride=s1_emb_stride)
        self.att_normB1 = LayerNorm(s1_emb_dim)
        self.att_tranB1 = Transformer(dim=s1_emb_dim,
                                      proj_kernel=s1_proj_kernel,
                                      kv_proj_stride=s1_kv_proj_stride,
                                      depth=s1_depth,
                                      heads=s1_heads,
                                      mlp_mult=s1_mlp_mult,
                                      dropout=dropout)

        self.att_convB2 = nn.Conv2d(s1_emb_dim,
                                    s2_emb_dim,
                                    kernel_size=s2_emb_kernel,
                                    padding=(s2_emb_kernel // 2),
                                    stride=s2_emb_stride)
        self.att_normB2 = LayerNorm(s2_emb_dim)
        self.att_tranB2 = Transformer(dim=s2_emb_dim,
                                      proj_kernel=s2_proj_kernel,
                                      kv_proj_stride=s2_kv_proj_stride,
                                      depth=s2_depth,
                                      heads=s2_heads,
                                      mlp_mult=s2_mlp_mult,
                                      dropout=dropout)

        self.att_convB3 = nn.Conv2d(s2_emb_dim,
                                    s3_emb_dim,
                                    kernel_size=s3_emb_kernel,
                                    padding=(s3_emb_kernel // 2),
                                    stride=s3_emb_stride)
        self.att_normB3 = LayerNorm(s3_emb_dim)
        self.att_tranB3 = Transformer(dim=s3_emb_dim,
                                      proj_kernel=s3_proj_kernel,
                                      kv_proj_stride=s3_kv_proj_stride,
                                      depth=s3_depth,
                                      heads=s3_heads,
                                      mlp_mult=s3_mlp_mult,
                                      dropout=dropout)

        self.double_conv = DoubleConv(s3_emb_dim, 512)

        #Upsampling path
        self.up_convB4 = UpBlock(512, 128, up_sample_mode)
        self.up_convB3 = UpBlockskip(128+32, 64, up_sample_mode)
        self.norm1     = LayerNorm(64)
        self.up_convB2 = UpBlockskip(64+32, 32, up_sample_mode)
        self.norm2     = LayerNorm(32)
        #self.up_convB1 = UpBlock(32, 16, up_sample_mode)

        # Output match
        self.conv_last = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # Input: N, 1, 256, 256 | output: N, 1, 128, 128
        x = self.att_convB1(x)
        PrintLayer()(x)
        skip_l1 = x
        
        # Input: N, 32, 128, 128 | output: N, 32, 128, 128
        x = self.att_normB1(x)
        PrintLayer()(x)
        
        # Input: N, 32, 128, 128 | output: N, 32, 128, 128
        x = self.att_tranB1(x)
        PrintLayer()(x)
        
        # Input: N, 32, 128, 128 | output: N, 32, 64, 64
        x = self.att_convB2(x)
        PrintLayer()(x)
        skip_l2 = x
        
        # Input: N, 32, 64, 64 | output: N, 32, 64, 64
        x = self.att_normB2(x)
        PrintLayer()(x)
        
        # Input: N, 32, 64, 64 | output: N, 32, 64, 64
        x = self.att_tranB2(x)
        PrintLayer()(x)
        
        # Input: N, 32, 64, 64 | output: N, 32, 32, 32
        x = self.att_convB3(x)
        PrintLayer()(x)
        
        # Input: N, 32, 32, 32 | output: N, 32, 32, 32
        x = self.att_normB3(x)
        PrintLayer()(x)
        
        # Input: N, 32, 32, 32 | output: N, 32, 32, 32
        x = self.att_tranB3(x)
        PrintLayer()(x)

        # Input: N, 32, 32, 32 | output: N, 512, 32, 32
        x = self.double_conv(x)
        PrintLayer()(x)
        
        # Input: N, 512, 32, 32 | output: N, 128, 64, 64
        x = self.up_convB4(x)
        PrintLayer()(x)

        # Input: N, 128, 64, 64 | output: N, 64, 128, 128
        x = self.up_convB3(x, skip_l2)
        x = self.norm1(x)
        PrintLayer()(x)

        # Input: N, 64, 128, 128 | output: N, 32, 256, 256
        x = self.up_convB2(x, skip_l1)
        x = self.norm2(x)
        PrintLayer()(x)
       
        # Input: N, 32, 256, 256 | output: N, 2, 256, 256
        x = self.conv_last(x)
        PrintLayer()(x)
        return x

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    from config import get_config
    image_size = 256
    channels = 3
    x = torch.Tensor(1, channels, image_size, image_size)

    config = get_config()
    model = CvTModified(config)#CvT()
    out = model(x)
    trainable_params, total_params = count_params(model)
    print(model)
    print("Trainable params: ", trainable_params, " total params: ", total_params)