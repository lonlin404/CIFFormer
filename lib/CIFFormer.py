# 加入超分，使用的是train，4个输出
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.net import pvt_v2_b2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class MSBlock(nn.Module):
    def __init__(self, c_in):
        super(MSBlock, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(c_in, 32, (3, 3), stride=(1, 1), padding=1)
        self.conv1 = nn.Conv2d(32, 32, (3, 3), stride=(1, 1), padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 32, (3, 3), stride=(1, 1), padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, 32, (3, 3), stride=(1, 1), padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(4 * 32, 32, (1, 1))
        self.bn = nn.BatchNorm2d(32)

    def forward(self, x):
        conv = self.conv(x)
        relu = self.relu(conv)
        conv1 = self.conv1(relu)
        relu1 = self.relu1(conv1)
        add1 = relu1 + relu
        conv2 = self.conv2(add1)
        relu2 = self.relu2(conv2)
        add2 = add1 + relu2
        conv3 = self.conv3(add2)
        relu3 = self.relu3(conv3)
        # 需要将conv，relu1，relu2，relu3四个cat起来
        conv4 = self.conv4(torch.cat((relu, relu1, relu2, relu3), dim=1))
        conv4 = self.relu(conv4)
        out = conv4 + relu
        return out


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class CFM(nn.Module):
    def __init__(self, channel):
        super(CFM, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 上采样
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  # 上采样
        self.downsample = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)  # 下采样
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_upsample6 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv_upsample7 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample8 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample9 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample10 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(4 * channel, 3 * channel, 3, padding=1)
        self.conv_concat4 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)

    # x1,x2,x3,x4分别表示对应的x4,x3,x2,x1
    def forward(self, x1, x2, x3, x4):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        # print(x2_2.shape)  # [1, 64, 22, 22]
        x2_2 = self.conv_concat2(x2_2)
        # print(x2_2.shape)  # [1, 64, 22, 22]

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        # print(x3_2.shape)  # [1, 96, 44, 44]
        x3_2 = self.conv_concat4(x3_2)
        x3_2 = self.conv_upsample6(x3_2)
        # print(x3_2.shape)  # [1, 96, 44, 44]

        x4_1 = self.conv_upsample7(self.upsample2(x1))
        x4_2 = self.conv_upsample8(self.upsample(x2))
        x4_3 = self.conv_upsample9(x3)
        # print(x4_3.shape, x4_2.shape, x4_1.shape)
        x4_4 = self.conv_upsample10(self.downsample(x4)) * x4_3 * x4_2 * x4_1
        # print(x4_4.shape)  # [1, 32, 44, 44]

        x5 = torch.cat((x3_2, x4_4), 1)
        # print(x5.shape)  # [1, 128, 44, 44]
        x5 = self.conv_concat3(x5)
        # print(x5.shape) # [1, 96, 44, 44]
        x1 = self.conv4(x5)  # [1, 32, 44, 44]
        # print(x1.shape)

        return x1


class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output


class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_relu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_relu(output)

        return output


# CFP模块
class CFPModule(nn.Module):
    def __init__(self, nIn, d=1, KSize=3, dkSize=3):
        super().__init__()

        self.bn_relu_1 = BNPReLU(nIn)
        self.bn_relu_2 = BNPReLU(nIn)
        self.conv1x1_1 = Conv(nIn, nIn // 4, KSize, 1, padding=1, bn_acti=True)

        self.dconv_4_1 = Conv(nIn // 4, nIn // 16, (dkSize, dkSize), 1, padding=(1 * d + 1, 1 * d + 1),
                              dilation=(d + 1, d + 1), groups=nIn // 16, bn_acti=True)

        self.dconv_4_2 = Conv(nIn // 16, nIn // 16, (dkSize, dkSize), 1, padding=(1 * d + 1, 1 * d + 1),
                              dilation=(d + 1, d + 1), groups=nIn // 16, bn_acti=True)

        self.dconv_4_3 = Conv(nIn // 16, nIn // 8, (dkSize, dkSize), 1, padding=(1 * d + 1, 1 * d + 1),
                              dilation=(d + 1, d + 1), groups=nIn // 16, bn_acti=True)

        self.dconv_1_1 = Conv(nIn // 4, nIn // 16, (dkSize, dkSize), 1, padding=(1, 1),
                              dilation=(1, 1), groups=nIn // 16, bn_acti=True)

        self.dconv_1_2 = Conv(nIn // 16, nIn // 16, (dkSize, dkSize), 1, padding=(1, 1),
                              dilation=(1, 1), groups=nIn // 16, bn_acti=True)

        self.dconv_1_3 = Conv(nIn // 16, nIn // 8, (dkSize, dkSize), 1, padding=(1, 1),
                              dilation=(1, 1), groups=nIn // 16, bn_acti=True)

        self.dconv_2_1 = Conv(nIn // 4, nIn // 16, (dkSize, dkSize), 1, padding=(int(d / 4 + 1), int(d / 4 + 1)),
                              dilation=(int(d / 4 + 1), int(d / 4 + 1)), groups=nIn // 16, bn_acti=True)

        self.dconv_2_2 = Conv(nIn // 16, nIn // 16, (dkSize, dkSize), 1, padding=(int(d / 4 + 1), int(d / 4 + 1)),
                              dilation=(int(d / 4 + 1), int(d / 4 + 1)), groups=nIn // 16, bn_acti=True)

        self.dconv_2_3 = Conv(nIn // 16, nIn // 8, (dkSize, dkSize), 1, padding=(int(d / 4 + 1), int(d / 4 + 1)),
                              dilation=(int(d / 4 + 1), int(d / 4 + 1)), groups=nIn // 16, bn_acti=True)

        self.dconv_3_1 = Conv(nIn // 4, nIn // 16, (dkSize, dkSize), 1, padding=(int(d / 2 + 1), int(d / 2 + 1)),
                              dilation=(int(d / 2 + 1), int(d / 2 + 1)), groups=nIn // 16, bn_acti=True)

        self.dconv_3_2 = Conv(nIn // 16, nIn // 16, (dkSize, dkSize), 1, padding=(int(d / 2 + 1), int(d / 2 + 1)),
                              dilation=(int(d / 2 + 1), int(d / 2 + 1)), groups=nIn // 16, bn_acti=True)

        self.dconv_3_3 = Conv(nIn // 16, nIn // 8, (dkSize, dkSize), 1, padding=(int(d / 2 + 1), int(d / 2 + 1)),
                              dilation=(int(d / 2 + 1), int(d / 2 + 1)), groups=nIn // 16, bn_acti=True)

        self.conv1x1 = Conv(nIn, nIn, 1, 1, padding=0, bn_acti=False)

    def forward(self, input):
        inp = self.bn_relu_1(input)
        inp = self.conv1x1_1(inp)

        o1_1 = self.dconv_1_1(inp)
        o1_2 = self.dconv_1_2(o1_1)
        o1_3 = self.dconv_1_3(o1_2)

        o2_1 = self.dconv_2_1(inp)
        o2_2 = self.dconv_2_2(o2_1)
        o2_3 = self.dconv_2_3(o2_2)

        o3_1 = self.dconv_3_1(inp)
        o3_2 = self.dconv_3_2(o3_1)
        o3_3 = self.dconv_3_3(o3_2)

        o4_1 = self.dconv_4_1(inp)
        o4_2 = self.dconv_4_2(o4_1)
        o4_3 = self.dconv_4_3(o4_2)

        output_1 = torch.cat([o1_1, o1_2, o1_3], 1)
        output_2 = torch.cat([o2_1, o2_2, o2_3], 1)
        output_3 = torch.cat([o3_1, o3_2, o3_3], 1)
        output_4 = torch.cat([o4_1, o4_2, o4_3], 1)

        ad1 = output_1
        ad2 = ad1 + output_2
        ad3 = ad2 + output_3
        ad4 = ad3 + output_4
        output = torch.cat([ad1, ad2, ad3, ad4], 1)
        output = self.bn_relu_2(output)
        output = self.conv1x1(output)

        return output + input


# RA-RA模块
class self_attn(nn.Module):
    def __init__(self, in_channels, mode='hw'):
        super(self_attn, self).__init__()

        self.mode = mode

        self.query_conv = Conv(in_channels, in_channels // 8, kSize=(1, 1), stride=1, padding=0)
        self.key_conv = Conv(in_channels, in_channels // 8, kSize=(1, 1), stride=1, padding=0)
        self.value_conv = Conv(in_channels, in_channels, kSize=(1, 1), stride=1, padding=0)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channel, height, width = x.size()

        axis = 1
        if 'h' in self.mode:
            axis *= height
        if 'w' in self.mode:
            axis *= width

        view = (batch_size, -1, axis)

        projected_query = self.query_conv(x).view(*view).permute(0, 2, 1)
        projected_key = self.key_conv(x).view(*view)

        attention_map = torch.bmm(projected_query, projected_key)
        attention = self.sigmoid(attention_map)
        projected_value = self.value_conv(x).view(*view)

        out = torch.bmm(projected_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channel, height, width)

        out = self.gamma * out + x
        return out


class AA_kernel(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AA_kernel, self).__init__()
        self.conv0 = Conv(in_channel, out_channel, kSize=1, stride=1, padding=0)
        self.conv1 = Conv(out_channel, out_channel, kSize=(3, 3), stride=1, padding=1)
        self.Hattn = self_attn(out_channel, mode='h')
        self.Wattn = self_attn(out_channel, mode='w')

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)

        Hx = self.Hattn(x)
        Wx = self.Wattn(Hx)

        return Wx


"""
将pvtv2的模型结构进行了更改，更改为pool_pvt_1(将attention机制更改为pooling)
本模型采用骨架+分类器的结构进行构建(目前对骨架进行了微改，后期可以对分类器进行更改)
"""


class PolypPVT(nn.Module):
    def __init__(self, channel=32):
        super(PolypPVT, self).__init__()

        self.backbone = pvt_v2_b3()  # [64, 128, 320, 512]
        path = './pretrained_pth/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        # 通过预训练模型的参数初始化骨架网络
        self.backbone.load_state_dict(model_dict)

        self.Translayer1 = BasicConv2d(64, 64, 1)
        self.Translayer2 = BasicConv2d(128, 128, 1)
        self.Translayer3 = BasicConv2d(320, 320, 1)
        self.Translayer4 = BasicConv2d(512, 512, 1)

        self.Translayer2_0 = MSBlock(64)
        self.Translayer2_1 = MSBlock(128)
        self.Translayer3_1 = MSBlock(320)
        self.Translayer4_1 = MSBlock(512)

        self.CFM = CFM(channel)
        self.CFP3 = CFPModule(512, d=8)
        self.CFP2 = CFPModule(320, d=8)
        self.CFP1 = CFPModule(128, d=8)

        self.ra1_conv1 = Conv(128, 32, 3, 1, padding=1, bn_acti=True)
        self.ra1_conv2 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra1_conv3 = Conv(32, 1, 3, 1, padding=1, bn_acti=True)

        self.ra2_conv1 = Conv(320, 32, 3, 1, padding=1, bn_acti=True)
        self.ra2_conv2 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra2_conv3 = Conv(32, 1, 3, 1, padding=1, bn_acti=True)

        self.ra3_conv1 = Conv(512, 32, 3, 1, padding=1, bn_acti=True)
        self.ra3_conv2 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra3_conv3 = Conv(32, 1, 3, 1, padding=1, bn_acti=True)

        self.aa_kernel_1 = AA_kernel(128, 128)
        self.aa_kernel_2 = AA_kernel(320, 320)
        self.aa_kernel_3 = AA_kernel(512, 512)

        self.down05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.out_SAM = nn.Conv2d(channel, 1, 1)
        self.out_CFM = nn.Conv2d(channel, 1, 1)

    def forward(self, x):
        # backbone
        pvt = self.backbone(x)
        # print(pvt)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]

        x1_t = self.Translayer1(x1)
        x2_t = self.Translayer2(x2)
        x3_t = self.Translayer3(x3)
        x4_t = self.Translayer4(x4)

        # CFM
        x1_t = self.Translayer2_0(x1_t)
        x2_t = self.Translayer2_1(x2_t)
        x3_t = self.Translayer3_1(x3_t)
        x4_t = self.Translayer4_1(x4_t)
        cfm_feature = self.CFM(x4_t, x3_t, x2_t, x1_t)
        prediction1 = self.out_CFM(cfm_feature)
        lateral_map_1 = F.interpolate(prediction1, scale_factor=8, mode='bilinear')

        # CFP3
        decoder_2 = F.interpolate(prediction1, scale_factor=0.25, mode='bilinear')
        # print(decoder_2.shape)
        cfp_out_1 = self.CFP3(x4)
        # print(cfp_out_1.shape)
        # cfp_out_1 += x4
        decoder_2_ra = -1 * (torch.sigmoid(decoder_2)) + 1
        aa_atten_3 = self.aa_kernel_3(cfp_out_1)
        aa_atten_3 += cfp_out_1
        aa_atten_3_o = decoder_2_ra.expand(-1, 512, -1, -1).mul(aa_atten_3)

        ra_3 = self.ra3_conv1(aa_atten_3_o)
        ra_3 = self.ra3_conv2(ra_3)
        ra_3 = self.ra3_conv3(ra_3)

        x_3 = ra_3 + decoder_2  # 11*11
        # print(x_3.shape)
        lateral_map_2 = F.interpolate(x_3, scale_factor=32, mode='bilinear')  # [1, 1, 352, 352]

        # CFP2
        decoder_3 = F.interpolate(x_3, scale_factor=2, mode='bilinear')
        cfp_out_2 = self.CFP2(x3)
        # cfp_out_2 += x3
        decoder_3_ra = -1 * (torch.sigmoid(decoder_3)) + 1
        aa_atten_2 = self.aa_kernel_2(cfp_out_2)
        aa_atten_2 += cfp_out_2
        aa_atten_2_o = decoder_3_ra.expand(-1, 320, -1, -1).mul(aa_atten_2)

        ra_2 = self.ra2_conv1(aa_atten_2_o)
        ra_2 = self.ra2_conv2(ra_2)
        ra_2 = self.ra2_conv3(ra_2)

        x_2 = ra_2 + decoder_3
        lateral_map_3 = F.interpolate(x_2, scale_factor=16, mode='bilinear')

        # CFP1
        decoder_4 = F.interpolate(x_2, scale_factor=2, mode='bilinear')
        cfp_out_3 = self.CFP1(x2)
        # cfp_out_3 += x2
        decoder_4_ra = -1 * (torch.sigmoid(decoder_4)) + 1
        aa_atten_1 = self.aa_kernel_1(cfp_out_3)
        aa_atten_1 += cfp_out_3
        aa_atten_1_o = decoder_4_ra.expand(-1, 128, -1, -1).mul(aa_atten_1)

        ra_1 = self.ra1_conv1(aa_atten_1_o)
        ra_1 = self.ra1_conv2(ra_1)
        ra_1 = self.ra1_conv3(ra_1)

        x_1 = ra_1 + decoder_4
        lateral_map_4 = F.interpolate(x_1, scale_factor=8, mode='bilinear')
        # print(lateral_map_4.shape)
        return lateral_map_4, lateral_map_3, lateral_map_2, lateral_map_1


def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()  # 参数大小
        param_sum += param.nelement()  # 参数总量
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型总大小为：{:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)


if __name__ == '__main__':
    model = PolypPVT().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    lateral_map_4, lateral_map_3, lateral_map_2, lateral_map_1 = model(input_tensor)
    # print(prediction1.size(), prediction2.size())
    param = getModelSize(model)
    print(param)
