import torch
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange
from model_package.resnet import resnet18

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential( # 两个全连接层变换
            nn.Linear(dim, hidden_dim), # 将14^14 线性变换到256维
            nn.GELU(), # 激活
            nn.Dropout(dropout),# drop
            nn.Linear(hidden_dim, dim), # 重新变换为14^14
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout=0.):
        super().__init__()
        # token_mix, 不是针对channel,而是针对token进行全连接,进行token的信息提取
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout), # 全连接层
            Rearrange('b d n -> b n d')
        )
        # 正常的channel信息提取
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        return x


class MLPMixer(nn.Module):

    def __init__(self, in_channels, dim, num_classes, patch_size,num_patch, depth, token_dim, channel_dim):
        super().__init__()

        # assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        # self.num_patch = (image_size // patch_size) ** 2  # (224/16)^2 = 14^2 = 196
        self.num_patch = num_patch  # 50*50=2500
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size),  # 用一个卷积核以patch的边长为stride扫过图片,即分块卷积
            Rearrange('b c h w -> b (h w) c'),  # [batch_size,h*w,c]
        )

        self.mixer_blocks = nn.ModuleList([])

        for _ in range(depth):
            # self.mixer_blocks.append(MixerBlock(dim, self.num_patch, token_dim, channel_dim))
            self.mixer_blocks.append(MixerBlock(dim, 12, token_dim, channel_dim))

        self.layer_norm = nn.LayerNorm(dim)
        # 全连接分类层
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, num_classes)
        )

        self.bone = BaseBone()

    def forward(self, x):
        x = self.bone(x)
        x = self.to_patch_embedding(x)

        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)

        x = self.layer_norm(x)

        x = x.mean(dim=1)

        return self.mlp_head(x)


class BasicBonv(torch.nn.Module):
    def __init__(self,in_ch,out_ch,ks,stride,padding):
        super(BasicBonv, self).__init__()
        self.conv = nn.Conv2d(in_ch,out_ch,ks,stride,padding)
        self.act = nn.ReLU(True)

    def forward(self,input):
        input = self.conv(input)
        input = self.act(input)
        return input

from torchvision.models._utils import IntermediateLayerGetter
class BaseBone(torch.nn.Module):
    def __init__(self):
        super(BaseBone, self).__init__()
        self.conv_1 = BasicBonv(3,36,3,2,1)
        self.conv_2 = BasicBonv(36,64,3,2,1)
        self.conv_3 = BasicBonv(64,96,3,1,1)
        # backbone = resnet18(True,num_classes=2)
        # return_layers = {'layer4': "0"}
        # self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
    def forward(self,input):
        inputs = []
        for img_fidx in range(12):
            img = input[:,img_fidx,...]
            inputs.append(self.conv_1(img))
        inputs = [self.conv_2(img) for img in inputs]
        inputs = [self.conv_3(img) for img in inputs]
        out = torch.cat(inputs,-1)
        return out



if __name__ == "__main__":
    img = torch.ones([1,12,3,224,224])
    basic_conv = BaseBone()

    out = basic_conv(img)

    model = MLPMixer(in_channels=96, num_patch=25*25, patch_size=25, num_classes=1000,
                     dim=512, depth=8, token_dim=256, channel_dim=2048)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)
    # model.cuda()
    out_img = model(out)

    print("Shape of out :", out_img.shape)  # [B, in_channels, image_size, image_size]
#
#     # [bs,100,1200,3]
#     # [bs,600*600,3]
#     # ks = 50, stride=50
#     # 600/50 = 12 , 12*12 = 144
