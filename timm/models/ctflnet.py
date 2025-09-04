
# Copyright 2024 [Hainan University]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import timm
from timm.models._registry import register_model
from torch.nn import init
from timm.layers import ClassifierHead
import pywt
import pywt.data
import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F
import os



"""WTConv-Block"""
def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters

def wavelet_transform(x, filters):
    filters = filters.to(x.dtype)
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    filters = filters.to(x.dtype)
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


def wavelet_transform_init(filters):
    class WaveletTransform(Function):

        @staticmethod
        def forward(ctx, input):
            with torch.no_grad():
                x = wavelet_transform(input, filters)
            return x

        @staticmethod
        def backward(ctx, grad_output):
            grad = inverse_wavelet_transform(grad_output, filters)
            return grad, None

    return WaveletTransform().apply


def inverse_wavelet_transform_init(filters):
    class InverseWaveletTransform(Function):

        @staticmethod
        def forward(ctx, input):
            with torch.no_grad():
                x = inverse_wavelet_transform(input, filters)
            return x

        @staticmethod
        def backward(ctx, grad_output):
            grad = wavelet_transform(grad_output, filters)
            return grad, None

    return InverseWaveletTransform().apply


class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = wavelet_transform_init(self.wt_filter)
        self.iwt_function = inverse_wavelet_transform_init(self.iwt_filter)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,
                                   groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
                                                   groups=in_channels)
        else:
            self.do_stride = None



    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


"""CBAM-Block"""
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class CBAMBlock(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + residual


""""""
class WLGF(nn.Module):
    def __init__(self, channel, input_num=3, is_upsample=True, cbam_red=16, cbam_ker=7, is_subout_upsample=True,wt_levels=1):
        super(WLGF, self).__init__()
        self.channel = channel
        self.input_num = input_num
        self.is_upsample = is_upsample
        self.is_subout_sample = is_subout_upsample
        self.cbam = CBAMBlock(channel=channel * input_num, reduction=cbam_red, kernel_size=cbam_ker)
        self.conv = nn.Conv2d(input_num * channel, input_num, kernel_size=3, stride=1, padding=1)
        self.conv1x1 = nn.Conv2d(channel, channel // 2, kernel_size=1, stride=1, padding=0)
        self.wtconv = WTConv2d(channel, channel,wt_levels=wt_levels)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # 使用 Kaiming 正态分布初始化 Linear 层的权重
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # 将偏置初始化为 0
        elif isinstance(m, nn.LayerNorm):
            # 将 LayerNorm 的权重和偏置初始化
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.Linear)):
            # 使用 Kaiming 正态分布初始化 Conv2d 或 Linear 层的权重
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # 将偏置初始化为 0

    def forward(self, xl=None, x_fuse=None, xg=None):
        if self.input_num == 3:
            y = torch.cat([xl, x_fuse, xg], dim=1)
            y = self.cbam(y)
            w = self.conv(y)
            w = nn.Softmax(dim=1)(w)
            wl = w[:, :1, :, :]
            wf = w[:, 1:2, :, :]
            wg = w[:, 2:, :, :]
            xl = xl * wl
            x_fuse = x_fuse * wf
            xg = xg * wg
            x_sum = xl + x_fuse + xg
        elif self.input_num == 2:
            y = torch.cat([xl, xg], dim=1)
            y = self.cbam(y)
            w = self.conv(y)
            w = nn.Softmax(dim=1)(w)
            wl = w[:, :1, :, :]
            wg = w[:, 1:2, :, :]
            xl = xl * wl
            xg = xg * wg
            x_sum = xl + xg

        out = self.wtconv(x_sum)

        if self.is_upsample:    # main_out 是否需要上采样
            main_out = self.conv1x1(out)
            main_out = F.interpolate(main_out, size=main_out.size(2) * 2, mode='nearest')
            if self.is_subout_sample:   # sub_out 是否需要上采样
                if out.size(-2) % 7==0:
                    out = F.interpolate(out, size=56, mode='nearest')
                else:
                    out = F.interpolate(out, size=64, mode='nearest')
                return main_out, out
            else:
                return main_out, out
        else:
            return out

class CTFLNet(nn.Module):

    def __init__(self, num_classes, conv_dims=(96, 192, 384, 768), cbam_red=16,
                 cbam_ker=7, **kwargs):
        super().__init__()

        ###### Local Branch Setting #######
        # 获取项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        convnext_ckpt_path = os.path.join(project_root, 'ckpt', 'timm-convnext_tiny.in12k_ft_in1k.safetensors')

        convnext = timm.create_model('convnext_tiny', pretrained=False,
                                     checkpoint_path=convnext_ckpt_path,
                                     )
        self.stages1 = nn.Sequential(
            convnext.stem,
            convnext.stages[0]
        )
        self.stages2 = convnext.stages[1]  # C3
        self.stages3 = convnext.stages[2]  # C4
        self.stages4 = convnext.stages[3]  # C5


        self.head = ClassifierHead(
            sum(conv_dims),
            num_classes=num_classes,
            pool_type='avg'
        )

        ###### Global Branch Setting ######
        davit_ckpt_path = os.path.join(project_root, 'ckpt', 'timm-davit_tiny.msft_in1k.bin')
        davit = timm.create_model('davit_tiny', pretrained=False,
                                  checkpoint_path=davit_ckpt_path)
        self.layers1 = nn.Sequential(
            davit.stem,
            davit.stages[0]
        )
        self.layers2 = davit.stages[1]
        self.layers3 = davit.stages[2]
        self.layers4 = davit.stages[3]
        # self.num_layers = len(depths)
        # self.embed_dim = embed_dim
        # self.patch_norm = patch_norm

        # The channels of stage4 output feature matrix
        # self.num_features = conv_dims[0]

        # split image into non-overlapping patches
        # self.pos_drop = nn.Dropout(p=drop_rate)

        # self.norm = norm_layer(self.num_features)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.wlgf4 = WLGF(conv_dims[3], input_num=2, cbam_red=cbam_red, cbam_ker=cbam_ker,wt_levels=1)
        self.wlgf3 = WLGF(conv_dims[2], input_num=3, cbam_red=cbam_red, cbam_ker=cbam_ker,wt_levels=1)
        self.wlgf2 = WLGF(conv_dims[1], input_num=3, cbam_red=cbam_red, cbam_ker=cbam_ker,wt_levels=1)
        self.wlgf1 = WLGF(conv_dims[0], input_num=3, is_upsample=False, cbam_red=cbam_red, cbam_ker=cbam_ker,wt_levels=1)

        self.cbam = CBAMBlock(channel=sum(conv_dims), reduction=cbam_red, kernel_size=cbam_ker)

    def forward(self, imgs):
        ######  Global Branch ######
        x_s_1 = self.layers1(imgs)
        x_s_2 = self.layers2(x_s_1)
        x_s_3 = self.layers3(x_s_2)
        x_s_4 = self.layers4(x_s_3)

        ######  Local Branch ######
        x_c_1 = self.stages1(imgs)
        x_c_2 = self.stages2(x_c_1)
        x_c_3 = self.stages3(x_c_2)
        x_c_4 = self.stages4(x_c_3)

        ###### Hierachical Feature Fusion Path ######
        x_f_4, x_4 = self.wlgf4(xl=x_c_4, xg=x_s_4)
        x_f_3, x_3 = self.wlgf3(xl=x_c_3, x_fuse=x_f_4, xg=x_s_3)
        x_f_2, x_2 = self.wlgf2(xl=x_c_2, x_fuse=x_f_3, xg=x_s_2)
        x_f_1 = self.wlgf1(xl=x_c_1, x_fuse=x_f_2, xg=x_s_1)

        x_concat = torch.cat([x_f_1, x_2, x_3, x_4], dim=1)
        out = self.cbam(x_concat)
        out = self.head(out)
        return out
""""""







@register_model
def ctflnet(num_classes=1, pretrained=False, **kwargs):

    model= CTFLNet(depths=(2, 2, 2, 2),
                                     conv_depths=(2, 2, 2, 2),
                                     num_classes=num_classes,
                                     cbam_red=16,
                                     cbam_ker=7)

    return model

    return model




if __name__ == '__main__':
    model = ctflnet(num_classes=1)
    x = torch.randn(16, 3, 224, 224)
    print(model)
    print(model(x).shape)
