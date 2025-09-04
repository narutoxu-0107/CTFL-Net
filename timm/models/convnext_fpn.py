import torch
from torch import nn
from timm.custom_block.MCA import MCALayer
from timm.custom_block.MSCAAttention import MSCAAttention
from timm.custom_block.odconv import ODConv2d
from timm.custom_block.SCSA import SCSA
from timm.custom_block.WTConv import WTConv2d
from timm.custom_block.LSKblock import LSKblock
from timm.custom_block.DFAM import DynamicFilter
from timm.custom_block.MobileViTv2Attention import MobileViTv2Attention
from timm.custom_block.OutlookAtt import OutlookAttention
from timm.custom_block.ParNetAttention import ParNetAttention
from timm.custom_block.S2Attention import S2Attention
from timm.custom_block.ScConv import ScConv
from timm.custom_block.SKNet import SKAttention

# from network.backbone.resnet import resnet50
from timm.models._registry import register_model, generate_default_cfgs, register_model_deprecations
from timm.models._builder import resolve_pretrained_cfg
from fightingcv_attention.attention.CBAM import CBAMBlock
import timm
from timm.layers import ClassifierHead
# class resnet50_cbam(nn.Module):
#     def __init__(self):
#         super(resnet50_cbam, self).__init__()
#         model = resnet50(pretrained=False)
#         model.load_state_dict(
#             torch.load(r'/home/ubuntu/stu_data/xujiajing_projects/public_weights/ResNet/resnet50-19c8e357.pth'))
#         self.backbone = nn.Sequential(
#             model.conv1,
#             model.bn1,
#             model.relu,
#             model.maxpool,
#             model.layer1,
#             model.layer2,
#             model.layer3,
#             model.layer4
#         )
#         self.cbam = CBAMBlock(channel=2048)
#         self.avgpool = model.avgpool
#         self.fc = nn.Linear(2048, 1)
#     def forward(self, x):
#         x = self.backbone(x)
#         x = self.cbam(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x,1)
#         x = self.fc(x)
#         return x


import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models.resnet import resnet50

# FPN的卷积层（即横向连接层）
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        # 使用3x3卷积进行特征处理，并保持通道数不变
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        # 使用BatchNorm来加速训练和防止过拟合
        self.bn = nn.BatchNorm2d(out_channels)
        # 使用ReLU激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # 卷积 -> 批归一化 -> ReLU
        return self.relu(self.bn(self.conv(x)))

class Cat_CBAM(nn.Module):
    def __init__(self, channels):
        super(Cat_CBAM, self).__init__()
        self.cbam = CBAMBlock(channels)

    def forward(self, x1=None,x2=None):
        if x2 is not None:
            x = torch.cat([x1,x2],dim=1)
            x = self.cbam(x)
        else:
            x = self.cbam(x1)
        return x

# 定义FPN模块
# FPN中的卷积块定义
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# FPN模块定义，包含C2层
class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        # 1x1横向卷积层，用于调整每个输入特征图的通道数到一致
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1) for in_channels in in_channels_list
        ])
        # 3x3卷积进一步处理每个特征图
        self.output_convs = nn.ModuleList([
            ConvBlock(out_channels, out_channels) for _ in in_channels_list
        ])

    def forward(self, inputs):
        # C2, C3, C4, C5 特征图输入
        c2, c3, c4, c5 = inputs

        # 从最高层的C5开始处理
        p5 = self.lateral_convs[3](c5)
        p4 = self.lateral_convs[2](c4) + F.interpolate(p5, size=c4.shape[-2:], mode='nearest')
        p3 = self.lateral_convs[1](c3) + F.interpolate(p4, size=c3.shape[-2:], mode='nearest')
        p2 = self.lateral_convs[0](c2) + F.interpolate(p3, size=c2.shape[-2:], mode='nearest')

        # 通过3x3卷积进一步处理P2, P3, P4, P5
        p2 = self.output_convs[0](p2)
        p3 = self.output_convs[1](p3)
        p4 = self.output_convs[2](p4)
        p5 = self.output_convs[3](p5)

        return [p2, p3, p4, p5]

# ResNet50与FPN结合的实现，包含C2
class Convnext_tiny_FPN(nn.Module):
    def __init__(self, pretrained=True, num_classes=1000):
        super(Convnext_tiny_FPN, self).__init__()
        # 加载convnext_tiny预训练模型
        convnext = timm.create_model('convnext_tiny', pretrained=False, checkpoint_path='/home/ubuntu/stu_data/xujiajing_projects/public_weights/convnext/timm-convnext_tiny.in12k_ft_in1k.safetensors')

        # 取ResNet的前几层，分别对应C2, C3, C4, C5
        self.layer1 = nn.Sequential(
            convnext.stem,
            convnext.stages[0]
        )  # C2
        self.layer2 = convnext.stages[1]  # C3
        self.layer3 = convnext.stages[2]  # C4
        self.layer4 = convnext.stages[3]  # C5

        # FPN接收C2, C3, C4, C5特征图，输出通道数统一为256
        self.fpn = FPN([96, 192, 384, 768], 96)

        # Regression head for LAI prediction
        self.reg_head = nn.Conv2d(96 * 4, num_classes, kernel_size=1, stride=1, padding=0)

        # 最终分类层，使用全局平均池化后接全连接层
        self.classifier = nn.Linear(96 * 4, num_classes)

    def forward(self, x):
        # 得到ResNet的多层特征图
        print(x.shape)
        c2 = self.layer1(x)
        print(c2.shape)
        c3 = self.layer2(c2)
        print(c3.shape)
        c4 = self.layer3(c3)
        print(c4.shape)
        c5 = self.layer4(c4)
        print(c5.shape)

        # 送入FPN模块得到多尺度特征图
        features = self.fpn([c2, c3, c4, c5])

        # 对每个FPN输出的特征图进行全局平均池化并展平
        pooled_features = [F.adaptive_avg_pool2d(f, 1).view(f.size(0), -1) for f in features]

        # 将池化后的特征拼接在一起
        concat_features = torch.cat(pooled_features, dim=1)
        # """因为要加CBAM,所以换一种合并方式"""
        # multi_scale_features = torch.cat(features, dim=1)
        #
        # out = self.reg_head(multi_scale_features)

        # 通过分类层得到最终分类结果
        out = self.classifier(concat_features)

        return out

class FPN_CBAM(nn.Module):
    def __init__(self, in_channels_list, out_channels, cbam_list):
        super(FPN_CBAM, self).__init__()
        # 1x1横向卷积层，用于调整每个输入特征图的通道数到一致
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1) for in_channels in in_channels_list
        ])
        # 3x3卷积进一步处理每个特征图
        self.output_convs = nn.ModuleList([
            ConvBlock(c, out_channels) for c in cbam_list
        ])
        self.cat_cbam_layers = nn.ModuleList([
            Cat_CBAM(channels=inchs) for inchs in cbam_list
        ])

    def forward(self, inputs):
        # C2, C3, C4, C5 特征图输入
        c2, c3, c4, c5 = inputs

        # 从最高层的C5开始处理
        p5 = self.lateral_convs[3](c5)
        p5 = self.cat_cbam_layers[3](x1=p5)
        p4 = self.lateral_convs[2](c4)
        p4 = self.cat_cbam_layers[2](x1=F.interpolate(p5, size=c4.shape[-2:], mode='nearest'), x2=p4)
        p3 = self.lateral_convs[1](c3)
        p3 = self.cat_cbam_layers[1](x1=F.interpolate(p4, size=c3.shape[-2:], mode='nearest'), x2 = p3)
        p2 = self.lateral_convs[0](c2)
        p2 = self.cat_cbam_layers[0](x1 = F.interpolate(p3, size=c2.shape[-2:], mode='nearest'), x2 = p2)

        # 通过3x3卷积进一步处理P2, P3, P4, P5
        p2 = self.output_convs[0](p2)
        p3 = self.output_convs[1](p3)
        p4 = self.output_convs[2](p4)
        p5 = self.output_convs[3](p5)

        return [p2, p3, p4, p5]

# ResNet50与FPN结合的实现，包含C2
class Convnext_tiny_FPN_CBAM(nn.Module):
    def __init__(self, pretrained=True, num_classes=1000):
        super(Convnext_tiny_FPN_CBAM, self).__init__()
        # 加载convnext_tiny预训练模型
        convnext = timm.create_model('convnext_tiny', pretrained=False,
                                     checkpoint_path='/home/ubuntu/stu_data/xujiajing_projects/public_weights/convnext/timm-convnext_tiny.in12k_ft_in1k.safetensors')

        # 取ResNet的前几层，分别对应C2, C3, C4, C5
        self.layer1 = nn.Sequential(
            convnext.stem,
            convnext.stages[0]
        )  # C2
        self.layer2 = convnext.stages[1]  # C3
        self.layer3 = convnext.stages[2]  # C4
        self.layer4 = convnext.stages[3]  # C5

        # FPN接收C2, C3, C4, C5特征图，输出通道数统一为256
        self.fpn = FPN_CBAM([96, 192, 384, 768], 96, [96*4,96*3,96*2,96])

        # Regression head for LAI prediction
        self.reg_head = nn.Conv2d(96 * 4, num_classes, kernel_size=1, stride=1, padding=0)

        # 最终分类层，使用全局平均池化后接全连接层
        self.classifier = nn.Linear(96 * 4, num_classes)

    def forward(self, x):
        # 得到ResNet的多层特征图
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # 送入FPN模块得到多尺度特征图
        features = self.fpn([c2, c3, c4, c5])

        # 对每个FPN输出的特征图进行全局平均池化并展平
        pooled_features = [F.adaptive_avg_pool2d(f, 1).view(f.size(0), -1) for f in features]

        # 将池化后的特征拼接在一起
        concat_features = torch.cat(pooled_features, dim=1)
        # """因为要加CBAM,所以换一种合并方式"""
        # multi_scale_features = torch.cat(features, dim=1)
        #
        # out = self.reg_head(multi_scale_features)

        # 通过分类层得到最终分类结果
        out = self.classifier(concat_features)

        return out

class FPN_CBAM_WTconv(nn.Module):
    def __init__(self, in_channels_list, out_channels, cbam_list):
        super(FPN_CBAM_WTconv, self).__init__()
        # 1x1横向卷积层，用于调整每个输入特征图的通道数到一致
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1) for in_channels in in_channels_list
        ])
        # 3x3卷积进一步处理每个特征图
        self.output_convs = nn.ModuleList([
            ConvBlock(c, out_channels) for c in cbam_list
        ])
        self.cat_cbam_layers = nn.ModuleList([
            Cat_CBAM(channels=inchs) for inchs in cbam_list
        ])
        self.wt_convs = nn.ModuleList([
            WTConv2d(in_channels, in_channels) for in_channels in in_channels_list
        ])

    def forward(self, inputs):
        # C2, C3, C4, C5 特征图输入
        c2, c3, c4, c5 = inputs

        # 从最高层的C5开始处理
        p5 = self.lateral_convs[3](self.wt_convs[3](c5))
        p5 = self.cat_cbam_layers[3](x1=p5)
        p4 = self.lateral_convs[2](self.wt_convs[2](c4))
        p4 = self.cat_cbam_layers[2](x1=F.interpolate(p5, size=c4.shape[-2:], mode='nearest'), x2=p4)
        p3 = self.lateral_convs[1](self.wt_convs[1](c3))
        p3 = self.cat_cbam_layers[1](x1=F.interpolate(p4, size=c3.shape[-2:], mode='nearest'), x2 = p3)
        p2 = self.lateral_convs[0](self.wt_convs[0](c2))
        p2 = self.cat_cbam_layers[0](x1 = F.interpolate(p3, size=c2.shape[-2:], mode='nearest'), x2 = p2)

        # 通过3x3卷积进一步处理P2, P3, P4, P5
        p2 = self.output_convs[0](p2)
        p3 = self.output_convs[1](p3)
        p4 = self.output_convs[2](p4)
        p5 = self.output_convs[3](p5)

        return [p2, p3, p4, p5]

# ResNet50与FPN结合的实现，包含C2
class Convnext_tiny_FPN_CBAM_WTconv(nn.Module):
    def __init__(self, pretrained=True, num_classes=1):
        super(Convnext_tiny_FPN_CBAM_WTconv, self).__init__()
        # 加载convnext_tiny预训练模型
        convnext = timm.create_model('convnext_tiny', pretrained=False,
                                     checkpoint_path='/home/ubuntu/stu_data/xujiajing_projects/public_weights/convnext/timm-convnext_tiny.in12k_ft_in1k.safetensors')

        # 取ResNet的前几层，分别对应C2, C3, C4, C5
        self.layer1 = nn.Sequential(
            convnext.stem,
            convnext.stages[0]
        )  # C2
        self.layer2 = convnext.stages[1]  # C3
        self.layer3 = convnext.stages[2]  # C4
        self.layer4 = convnext.stages[3]  # C5

        # FPN接收C2, C3, C4, C5特征图，输出通道数统一为256
        self.fpn = FPN_CBAM_WTconv([96, 192, 384, 768], 96, [96 * 4, 96 * 3, 96 * 2, 96])

        # Regression head for LAI prediction
        self.reg_head = nn.Conv2d(96 * 4, num_classes, kernel_size=1, stride=1, padding=0)

        # 最终分类层，使用全局平均池化后接全连接层
        self.classifier = nn.Linear(96 * 4, num_classes)

    def forward(self, x):
        # 得到ResNet的多层特征图
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # 送入FPN模块得到多尺度特征图
        features = self.fpn([c2, c3, c4, c5])

        # 对每个FPN输出的特征图进行全局平均池化并展平
        pooled_features = [F.adaptive_avg_pool2d(f, 1).view(f.size(0), -1) for f in features]

        # 将池化后的特征拼接在一起
        concat_features = torch.cat(pooled_features, dim=1)
        # """因为要加CBAM,所以换一种合并方式"""
        # multi_scale_features = torch.cat(features, dim=1)
        #
        # out = self.reg_head(multi_scale_features)

        # 通过分类层得到最终分类结果
        out = self.classifier(concat_features)

        return out

class FPN_CBAM_WTconv_v1(nn.Module):
    def __init__(self, in_channels_list, out_channels, cbam_list):
        super(FPN_CBAM_WTconv_v1, self).__init__()
        # 1x1横向卷积层，用于调整每个输入特征图的通道数到一致
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1) for in_channels in in_channels_list
        ])
        # 3x3卷积进一步处理每个特征图
        self.output_convs = nn.ModuleList([
            ConvBlock(c, out_channels) for c in cbam_list
        ])
        self.cat_cbam_layers = nn.ModuleList([
            Cat_CBAM(channels=inchs) for inchs in cbam_list
        ])
        self.wt_convs = nn.ModuleList([
            WTConv2d(out_channels, out_channels) for in_channels in in_channels_list
        ])

    def forward(self, inputs):
        # C2, C3, C4, C5 特征图输入
        c2, c3, c4, c5 = inputs

        # 从最高层的C5开始处理
        p5 = self.lateral_convs[3](c5)
        p5 = self.cat_cbam_layers[3](x1=p5)
        p4 = self.lateral_convs[2](c4)
        p4 = self.cat_cbam_layers[2](x1=F.interpolate(p5, size=c4.shape[-2:], mode='nearest'), x2=p4)
        p3 = self.lateral_convs[1](c3)
        p3 = self.cat_cbam_layers[1](x1=F.interpolate(p4, size=c3.shape[-2:], mode='nearest'), x2 = p3)
        p2 = self.lateral_convs[0](c2)
        p2 = self.cat_cbam_layers[0](x1 = F.interpolate(p3, size=c2.shape[-2:], mode='nearest'), x2 = p2)

        # 通过3x3卷积进一步处理P2, P3, P4, P5
        p2 = self.wt_convs[0](self.output_convs[0](p2))
        p3 = self.wt_convs[1](self.output_convs[1](p3))
        p4 = self.wt_convs[2](self.output_convs[2](p4))
        p5 = self.wt_convs[3](self.output_convs[3](p5))

        return [p2, p3, p4, p5]

# ResNet50与FPN结合的实现，包含C2
class Convnext_tiny_FPN_CBAM_WTconv_v1(nn.Module):
    def __init__(self, pretrained=True, num_classes=1):
        super(Convnext_tiny_FPN_CBAM_WTconv_v1, self).__init__()
        # 加载convnext_tiny预训练模型
        convnext = timm.create_model('convnext_tiny', pretrained=False,
                                     checkpoint_path='/home/ubuntu/stu_data/xujiajing_projects/public_weights/convnext/timm-convnext_tiny.in12k_ft_in1k.safetensors')

        # 取ResNet的前几层，分别对应C2, C3, C4, C5
        self.layer1 = nn.Sequential(
            convnext.stem,
            convnext.stages[0]
        )  # C2
        self.layer2 = convnext.stages[1]  # C3
        self.layer3 = convnext.stages[2]  # C4
        self.layer4 = convnext.stages[3]  # C5

        # FPN接收C2, C3, C4, C5特征图，输出通道数统一为256
        self.fpn = FPN_CBAM_WTconv_v1([96, 192, 384, 768], 96, [96 * 4, 96 * 3, 96 * 2, 96])

        # Regression head for LAI prediction
        self.reg_head = nn.Conv2d(96 * 4, num_classes, kernel_size=1, stride=1, padding=0)

        # 最终分类层，使用全局平均池化后接全连接层
        self.classifier = nn.Linear(96 * 4, num_classes)

    def forward(self, x):
        # 得到ResNet的多层特征图
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # 送入FPN模块得到多尺度特征图
        features = self.fpn([c2, c3, c4, c5])

        # 对每个FPN输出的特征图进行全局平均池化并展平
        pooled_features = [F.adaptive_avg_pool2d(f, 1).view(f.size(0), -1) for f in features]

        # 将池化后的特征拼接在一起
        concat_features = torch.cat(pooled_features, dim=1)
        # """因为要加CBAM,所以换一种合并方式"""
        # multi_scale_features = torch.cat(features, dim=1)
        #
        # out = self.reg_head(multi_scale_features)

        # 通过分类层得到最终分类结果
        out = self.classifier(concat_features)

        return out

class FPN_CBAM_WTconv_v2(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN_CBAM_WTconv_v2, self).__init__()
        # 1x1横向卷积层，用于调整每个输入特征图的通道数到一致
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1) for in_channels in in_channels_list
        ])
        # 3x3卷积进一步处理每个特征图
        self.output_convs = nn.ModuleList([
            WTConv2d(out_channels,out_channels) for _ in in_channels_list
        ])

    def forward(self, inputs):
        # C2, C3, C4, C5 特征图输入
        c2, c3, c4, c5 = inputs

        # 从最高层的C5开始处理
        p5 = self.lateral_convs[3](c5)
        p4 = self.lateral_convs[2](c4) + F.interpolate(p5, size=c4.shape[-2:], mode='nearest')
        p3 = self.lateral_convs[1](c3) + F.interpolate(p4, size=c3.shape[-2:], mode='nearest')
        p2 = self.lateral_convs[0](c2) + F.interpolate(p3, size=c2.shape[-2:], mode='nearest')

        # 通过3x3卷积进一步处理P2, P3, P4, P5
        p2 = self.output_convs[0](p2)
        p3 = self.output_convs[1](p3)
        p4 = self.output_convs[2](p4)
        p5 = self.output_convs[3](p5)

        #上采样至p2尺寸
        p3 = F.interpolate(p3, size=p2.shape[-2:], mode='nearest')
        p4 = F.interpolate(p4, size=p2.shape[-2:], mode='nearest')
        p5 = F.interpolate(p5, size=p2.shape[-2:], mode='nearest')

        return [p2, p3, p4, p5]

# ResNet50与FPN结合的实现，包含C2
class Convnext_tiny_FPN_CBAM_WTconv_v2(nn.Module):
    def __init__(self, pretrained=True, num_classes=1000):
        super(Convnext_tiny_FPN_CBAM_WTconv_v2, self).__init__()
        # 加载convnext_tiny预训练模型
        convnext = timm.create_model('convnext_tiny', pretrained=False,
                                     checkpoint_path='/home/ubuntu/stu_data/xujiajing_projects/public_weights/convnext/timm-convnext_tiny.in12k_ft_in1k.safetensors')

        # 取ResNet的前几层，分别对应C2, C3, C4, C5
        self.layer1 = nn.Sequential(
            convnext.stem,
            convnext.stages[0]
        )  # C2
        self.layer2 = convnext.stages[1]  # C3
        self.layer3 = convnext.stages[2]  # C4
        self.layer4 = convnext.stages[3]  # C5

        # FPN接收C2, C3, C4, C5特征图，输出通道数统一为256
        self.fpn = FPN_CBAM_WTconv_v2([96, 192, 384, 768], 96)

        self.cbam = CBAMBlock(96*4)

        # Regression head for LAI prediction
        # self.reg_head = nn.Conv2d(96 * 4, num_classes, kernel_size=1, stride=1, padding=0)

        # 最终分类层，使用全局平均池化后接全连接层
        # self.classifier = nn.Linear(96 * 4, num_classes)
        self.head = ClassifierHead(
            96*4,
            num_classes,
            pool_type='avg',
            drop_rate=0.0,
        )


    def forward(self, x):
        # 得到ResNet的多层特征图
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # 送入FPN模块得到多尺度特征图
        features = self.fpn([c2, c3, c4, c5])

        # 对每个FPN输出的特征图进行全局平均池化并展平
        features = torch.cat(features,dim=1)

        concat_features = self.cbam(features)

        # 通过分类层得到最终分类结果
        out = self.head(concat_features)

        return out

class FPN_WTconv(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN_WTconv, self).__init__()
        # 1x1横向卷积层，用于调整每个输入特征图的通道数到一致
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1) for in_channels in in_channels_list
        ])
        # 3x3卷积进一步处理每个特征图
        self.output_convs = nn.ModuleList([
            WTConv2d(out_channels,out_channels) for _ in in_channels_list
        ])

    def forward(self, inputs):
        # C2, C3, C4, C5 特征图输入
        c2, c3, c4, c5 = inputs

        # 从最高层的C5开始处理
        p5 = self.lateral_convs[3](c5)
        p4 = self.lateral_convs[2](c4) + F.interpolate(p5, size=c4.shape[-2:], mode='nearest')
        p3 = self.lateral_convs[1](c3) + F.interpolate(p4, size=c3.shape[-2:], mode='nearest')
        p2 = self.lateral_convs[0](c2) + F.interpolate(p3, size=c2.shape[-2:], mode='nearest')

        # 通过3x3卷积进一步处理P2, P3, P4, P5
        p2 = self.output_convs[0](p2)
        p3 = self.output_convs[1](p3)
        p4 = self.output_convs[2](p4)
        p5 = self.output_convs[3](p5)

        return [p2, p3, p4, p5]

# ResNet50与FPN结合的实现，包含C2
class Convnext_tiny_FPN_WTconv(nn.Module):
    def __init__(self, pretrained=True, num_classes=1000):
        super(Convnext_tiny_FPN_WTconv, self).__init__()
        # 加载convnext_tiny预训练模型
        convnext = timm.create_model('convnext_tiny', pretrained=False,
                                     checkpoint_path='/home/ubuntu/stu_data/xujiajing_projects/public_weights/convnext/timm-convnext_tiny.in12k_ft_in1k.safetensors')

        # 取ResNet的前几层，分别对应C2, C3, C4, C5
        self.layer1 = nn.Sequential(
            convnext.stem,
            convnext.stages[0]
        )  # C2
        self.layer2 = convnext.stages[1]  # C3
        self.layer3 = convnext.stages[2]  # C4
        self.layer4 = convnext.stages[3]  # C5

        # FPN接收C2, C3, C4, C5特征图，输出通道数统一为256
        self.fpn = FPN_WTconv([96, 192, 384, 768], 96)

        # Regression head for LAI prediction
        self.reg_head = nn.Conv2d(96 * 4, num_classes, kernel_size=1, stride=1, padding=0)

        # 最终分类层，使用全局平均池化后接全连接层
        self.classifier = nn.Linear(96 * 4, num_classes)

    def forward(self, x):
        # 得到ResNet的多层特征图
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # 送入FPN模块得到多尺度特征图
        features = self.fpn([c2, c3, c4, c5])

        # 对每个FPN输出的特征图进行全局平均池化并展平
        pooled_features = [F.adaptive_avg_pool2d(f, 1).view(f.size(0), -1) for f in features]

        # 将池化后的特征拼接在一起
        concat_features = torch.cat(pooled_features, dim=1)
        # """因为要加CBAM,所以换一种合并方式"""
        # multi_scale_features = torch.cat(features, dim=1)
        #
        # out = self.reg_head(multi_scale_features)

        # 通过分类层得到最终分类结果
        out = self.classifier(concat_features)

        return out

# FPN模块定义，包含C2层
class FPN_Wtconv_v1(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN_Wtconv_v1, self).__init__()
        # 1x1横向卷积层，用于调整每个输入特征图的通道数到一致
        self.wtcoonvs = nn.ModuleList([
            WTConv2d(in_channels=in_channel,out_channels=in_channel) for in_channel in in_channels_list
        ])

        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1) for in_channels in in_channels_list
        ])
        # 3x3卷积进一步处理每个特征图
        self.output_convs = nn.ModuleList([
            ConvBlock(out_channels, out_channels) for _ in in_channels_list
        ])

    def forward(self, inputs):
        # C2, C3, C4, C5 特征图输入
        c2, c3, c4, c5 = inputs

        # 从最高层的C5开始处理
        p5 = self.lateral_convs[3](self.wtcoonvs[3](c5))
        p4 = self.lateral_convs[2](self.wtcoonvs[2](c4)) + F.interpolate(p5, size=c4.shape[-2:], mode='nearest')
        p3 = self.lateral_convs[1](self.wtcoonvs[1](c3)) + F.interpolate(p4, size=c3.shape[-2:], mode='nearest')
        p2 = self.lateral_convs[0](self.wtcoonvs[0](c2)) + F.interpolate(p3, size=c2.shape[-2:], mode='nearest')

        # 通过3x3卷积进一步处理P2, P3, P4, P5
        p2 = self.output_convs[0](p2)
        p3 = self.output_convs[1](p3)
        p4 = self.output_convs[2](p4)
        p5 = self.output_convs[3](p5)

        return [p2, p3, p4, p5]

# ResNet50与FPN结合的实现，包含C2
class Convnext_tiny_FPN_Wtconv_v1(nn.Module):
    def __init__(self, pretrained=True, num_classes=1000):
        super(Convnext_tiny_FPN_Wtconv_v1, self).__init__()
        # 加载convnext_tiny预训练模型
        convnext = timm.create_model('convnext_tiny', pretrained=False, checkpoint_path='/home/ubuntu/stu_data/xujiajing_projects/public_weights/convnext/timm-convnext_tiny.in12k_ft_in1k.safetensors')

        # 取ResNet的前几层，分别对应C2, C3, C4, C5
        self.layer1 = nn.Sequential(
            convnext.stem,
            convnext.stages[0]
        )  # C2
        self.layer2 = convnext.stages[1]  # C3
        self.layer3 = convnext.stages[2]  # C4
        self.layer4 = convnext.stages[3]  # C5

        # FPN接收C2, C3, C4, C5特征图，输出通道数统一为256
        self.fpn = FPN_Wtconv_v1([96, 192, 384, 768], 96)

        # Regression head for LAI prediction
        self.reg_head = nn.Conv2d(96 * 4, num_classes, kernel_size=1, stride=1, padding=0)

        # 最终分类层，使用全局平均池化后接全连接层
        self.classifier = nn.Linear(96 * 4, num_classes)

    def forward(self, x):
        # 得到ResNet的多层特征图
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # 送入FPN模块得到多尺度特征图
        features = self.fpn([c2, c3, c4, c5])

        # 对每个FPN输出的特征图进行全局平均池化并展平
        pooled_features = [F.adaptive_avg_pool2d(f, 1).view(f.size(0), -1) for f in features]

        # 将池化后的特征拼接在一起
        concat_features = torch.cat(pooled_features, dim=1)
        # """因为要加CBAM,所以换一种合并方式"""
        # multi_scale_features = torch.cat(features, dim=1)
        #
        # out = self.reg_head(multi_scale_features)

        # 通过分类层得到最终分类结果
        out = self.classifier(concat_features)

        return out

@register_model
def convnext_tiny_fpn(pretrained=False, **kwargs):
    pretrained_cfg = resolve_pretrained_cfg(
        "mobilevitv2_200",
        pretrained_cfg=None,
        pretrained_cfg_overlay=None
    )
    pretrained_cfg = pretrained_cfg.to_dict()
    model = Convnext_tiny_FPN(pretrained=True, num_classes=1)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg
    return model

@register_model
def convnext_tiny_fpn_cbam(pretrained=False, **kwargs):
    pretrained_cfg = resolve_pretrained_cfg(
        "mobilevitv2_200",
        pretrained_cfg=None,
        pretrained_cfg_overlay=None
    )
    pretrained_cfg = pretrained_cfg.to_dict()
    model = Convnext_tiny_FPN_CBAM(pretrained=True, num_classes=1)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg
    return model

@register_model
def convnext_tiny_fpn_cbam_wtconv(pretrained=False, **kwargs):
    pretrained_cfg = resolve_pretrained_cfg(
        "mobilevitv2_200",
        pretrained_cfg=None,
        pretrained_cfg_overlay=None
    )
    pretrained_cfg = pretrained_cfg.to_dict()
    model = Convnext_tiny_FPN_CBAM_WTconv(pretrained=True, num_classes=1)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg
    return model

@register_model
def convnext_tiny_fpn_cbam_wtconv_v1(pretrained=False, **kwargs):
    pretrained_cfg = resolve_pretrained_cfg(
        "mobilevitv2_200",
        pretrained_cfg=None,
        pretrained_cfg_overlay=None
    )
    pretrained_cfg = pretrained_cfg.to_dict()
    model = Convnext_tiny_FPN_CBAM_WTconv_v1(pretrained=True, num_classes=1)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg
    return model
@register_model
def convnext_tiny_fpn_cbam_wtconv_v2(pretrained=False, **kwargs):
    pretrained_cfg = resolve_pretrained_cfg(
        "mobilevitv2_200",
        pretrained_cfg=None,
        pretrained_cfg_overlay=None
    )
    pretrained_cfg = pretrained_cfg.to_dict()
    model = Convnext_tiny_FPN_CBAM_WTconv_v2(pretrained=True, num_classes=1)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg
    return model

#
@register_model
def convnext_tiny_fpn_wtconv(pretrained=False, **kwargs):
    pretrained_cfg = resolve_pretrained_cfg(
        "mobilevitv2_200",
        pretrained_cfg=None,
        pretrained_cfg_overlay=None
    )
    pretrained_cfg = pretrained_cfg.to_dict()
    model = Convnext_tiny_FPN_WTconv(pretrained=True, num_classes=1)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg
    return model

@register_model
def convnext_tiny_fpn_wtconv_v1(pretrained=False, **kwargs):
    pretrained_cfg = resolve_pretrained_cfg(
        "mobilevitv2_200",
        pretrained_cfg=None,
        pretrained_cfg_overlay=None
    )
    pretrained_cfg = pretrained_cfg.to_dict()
    model = Convnext_tiny_FPN_Wtconv_v1(pretrained=True, num_classes=1)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg
    return model


# 测试模型
if __name__ == "__main__":
    # 创建模型实例
    model = convnext_tiny_fpn_cbam_wtconv_v2(pretrained=True, num_classes=1)

    # 打印模型架构
    print(model)

    # 创建一个假输入（模拟224x224大小的RGB图片）
    input_tensor = torch.randn(16, 3, 224, 224)

    # 执行前向传播
    output = model(input_tensor)
    print(output.shape)  # 输出应该是 [1, 10]，表示分类结果
