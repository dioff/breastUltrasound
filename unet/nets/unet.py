import torch
import torch.nn as nn

from nets.resnet import resnet50
from nets.vgg import VGG16


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        # 定义卷积层1
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        # 定义卷积层2
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        # 定义上采样层
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        # 定义ReLU激活函数
        self.relu   = nn.ReLU(inplace = True)

    def forward(self, inputs1, inputs2):
        # 将输入1和上采样后的输入2进行拼接
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        # 通过卷积层1
        outputs = self.conv1(outputs)
        # 通过ReLU激活函数
        outputs = self.relu(outputs)
        # 通过卷积层2
        outputs = self.conv2(outputs)
        # 通过ReLU激活函数
        outputs = self.relu(outputs)
        return outputs

class Unet(nn.Module):
    def __init__(self, num_classes = 21, pretrained = False, backbone = 'vgg'):
        super(Unet, self).__init__()
        # 判断backbone类型
        if backbone == 'vgg':
            # 使用VGG16作为backbone
            self.vgg    = VGG16(pretrained = pretrained)
            # 定义VGG16的输出通道数
            in_filters  = [192, 384, 768, 1024]
        elif backbone == "resnet50":
            # 使用resnet50作为backbone
            self.resnet = resnet50(pretrained = pretrained)
            # 定义resnet50的输出通道数
            in_filters  = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        # 定义Unet的输出通道数
        out_filters = [64, 128, 256, 512]

        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        if backbone == 'resnet50':
            # 定义resnet50的卷积层
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor = 2), 
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        # 定义最后的卷积层
        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

    def forward(self, inputs):
        if self.backbone == "vgg":
            # 使用VGG16作为backbone
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            # 使用resnet50作为backbone
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        # 上采样
        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        if self.up_conv != None:
            # 使用resnet50的卷积层
            up1 = self.up_conv(up1)

        # 最后的卷积层
        final = self.final(up1)
        
        return final

    def freeze_backbone(self):
        # 冻结backbone的参数
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        # 解冻backbone的参数
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True
