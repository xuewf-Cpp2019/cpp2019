# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F

class single_conv(nn.Module):
    '''(conv => BN => ReLU)'''
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        #    SELayer(out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        x = self.conv(x)
        return x



class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),

            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),

            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False) ,

            #SKConv(out_ch)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = nn.Sequential(

            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv_with_sk(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv_with_sk, self).__init__()
        self.conv = nn.Sequential(
            #SKConv(in_ch),
            single_conv(in_ch, out_ch),
            SKConv(out_ch),

        )

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv_with_sk_for_fea(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv_with_sk_for_fea, self).__init__()
        self.conv = nn.Sequential(
            #SKConv(in_ch),
            single_conv(in_ch, out_ch),
            SKConv_SA_get_feature(out_ch),

        )

    def forward(self, x):
        feas, fea_U1, fea_Attention_S, fea_U2, x = self.conv(x)
        return x

class inconv_with_se(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv_with_se, self).__init__()
        self.conv = nn.Sequential(
            single_conv(in_ch, out_ch),

            single_conv(out_ch, out_ch),
            SELayer(out_ch),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv_with_DIB_se(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv_with_DIB_se, self).__init__()
        self.conv3 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.convs = nn.ModuleList([])
        for i in range(4):
            self.convs.append(nn.Sequential(

                nn.Conv2d(out_ch, out_ch//4, kernel_size=3, stride=1, padding=i + 1, dilation=i+1),
                # 将channel分为G组，默认为1，即不分
                nn.BatchNorm2d(out_ch//4),
                nn.ReLU(inplace=False)
            ))

        self.CA = nn.Sequential(
            SELayer(out_ch),

        )
        '''
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False)
        )
        '''
    def forward(self, x):
        x = self.conv3(x)
        for i, conv in enumerate(self.convs):
            fea = conv(x)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)   #feas:4*4C*80*80
        x = self.CA(feas)
        #x = self.conv2(x)

        return x


class inconv_with_DIB_se_DIG(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv_with_DIB_se_DIG, self).__init__()
        self.conv3 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.convs = nn.ModuleList([])
        for i in range(4):
            self.convs.append(nn.Sequential(

                nn.Conv2d(out_ch, out_ch // 4, kernel_size=3, stride=1, padding=i + 1, dilation=i + 1),
                # 将channel分为G组，默认为1，即不分
                nn.BatchNorm2d(out_ch // 4),
                nn.ReLU(inplace=False)
            ))

        self.CA = nn.Sequential(
            SELayer(out_ch),

        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False)
        )


    def forward(self, x):
        x = self.conv3(x)
        for i, conv in enumerate(self.convs):
            fea = conv(x)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)  # feas:4*4C*80*80
        x,SA_x,no_SA_x = self.CA(feas)
        x = self.conv2(x)

        return x,SA_x,no_SA_x

class down_with_DIB_se(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_with_DIB_se, self).__init__()
        self.mpconv = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding= 1)
        self.convs = nn.ModuleList([])
        for i in range(4):
            self.convs.append(nn.Sequential(

                nn.Conv2d(out_ch, out_ch//4, kernel_size=3, stride=1, padding=i + 1, dilation=i+1),
                # 将channel分为G组，默认为1，即不分
                nn.BatchNorm2d(out_ch//4),
                nn.ReLU(inplace=False)
            ))

        self.CA = nn.Sequential(
            SELayer(out_ch),

        )
        '''
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False)
        )
        '''
    def forward(self, x):
        x = self.mpconv(x)
        x = self.conv3(x)
        for i, conv in enumerate(self.convs):
            fea = conv(x)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)   #feas:4*4C*80*80
        x = self.CA(feas)
        #x = self.conv2(x)
        return x

class down_with_DIB_se_DIG(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_with_DIB_se_DIG, self).__init__()
        self.mpconv = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding= 1)
        self.convs = nn.ModuleList([])
        for i in range(4):
            self.convs.append(nn.Sequential(

                nn.Conv2d(out_ch, out_ch//4, kernel_size=3, stride=1, padding=i + 1, dilation=i+1),
                # 将channel分为G组，默认为1，即不分
                nn.BatchNorm2d(out_ch//4),
                nn.ReLU(inplace=False)
            ))
        '''
        self.CA = nn.Sequential(
            SELayer(out_ch),

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False)
        )
        '''
    def forward(self, x):
        x = self.mpconv(x)
        x = self.conv3(x)
        for i, conv in enumerate(self.convs):
            fea = conv(x)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)   #feas:4*4C*80*80
        #x,SA_x,no_SA_x = self.CA(feas)
        #x = self.conv2(x)
        return feas#x,SA_x,no_SA_x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(

            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class fc(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(fc, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_ch, in_ch // 64, bias=True),
            #nn.ReLU(inplace=False)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_ch // 64, out_ch, bias=True),
            #nn.ReLU(inplace=False)
        )

    def forward(self, x):
        high_d = x.view(x.shape[0],-1)
        fcl_1 = self.fc1(high_d)
        fcl_2 = self.fc2(fcl_1)
        return fcl_2



class fc1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(fc1, self).__init__()

        self.fc_sqeeze = nn.Sequential(
            nn.Linear(in_ch, in_ch // 2,bias=False),
            nn.ReLU(inplace=False)
        )
        self.fc_decompress = nn.Sequential(
            nn.Linear(in_ch//2, out_ch, bias=False),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        x = self.fc_sqeeze(x)
        x = self.fc_decompress(x)
        return x.unsqueeze(dim=-1).unsqueeze(dim=-1)

class down_with_sk(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_with_sk, self).__init__()
        self.mpconv = nn.Sequential(
            #SELayer(in_ch),

            nn.MaxPool2d(2),
            single_conv(in_ch, out_ch),
            SKConv(out_ch),
            #single_conv(out_ch, out_ch)
            #double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class down_with_sk_for_fea(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_with_sk_for_fea, self).__init__()
        self.mpconv = nn.Sequential(
            #SELayer(in_ch),

            nn.MaxPool2d(2),
            single_conv(in_ch, out_ch),
            SKConv_SA_get_feature(out_ch),
            #single_conv(out_ch, out_ch)
            #double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        feas, fea_U1, fea_Attention_S, fea_U2, x = self.mpconv(x)
        return x

class down_with_se(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_with_se, self).__init__()
        self.mpconv = nn.Sequential(
            #SELayer(in_ch),

            nn.MaxPool2d(2),
            #double_conv(in_ch, out_ch),

            single_conv(in_ch, out_ch),

            single_conv(out_ch, out_ch),SELayer(out_ch),
            #double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x
class bottom_conv_sk(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(bottom_conv_sk, self).__init__()
        self.mpconv = nn.Sequential(
            #SELayer(in_ch),
            nn.MaxPool2d(2),
            #single_conv(in_ch, out_ch),
            SKConv(in_ch),
            single_conv(in_ch, out_ch)

        )

    def forward(self, x):
        x = self.mpconv(x)
        return x
'''
class deconv0(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(deconv0, self).__init__()
        self.vector2mat = ,

    def forward(self, x1):
        x1 = self.vector2mat(x1)
        return x1
'''
class deconv1(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):  # bilinear=True
        super(deconv1, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

    def forward(self, x1):
        x1 = self.deconv(x1)
        return x1


class bi_conv(nn.Module):
    def __init__(self, in_ch, out_ch):  # bilinear=True
        super(bi_conv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):  # bilinear=True
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,  #后期加入可训练的Upsampling
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  #
            #self.up = F.interpolate(input,scale_factor=2, mode='bilinear', align_corners=True) 如何改为这一句？
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv =  double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,          #左右添加dim
                        diffY // 2, diffY - diffY//2))         #上下添加dim

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)

        return x

class up_no_skip(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):  # bilinear=True
        super(up_no_skip, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  #
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1):
        x1 = self.up(x1)
        x = self.conv(x1)
        return x

class up_auto_c0(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):  # bilinear=True
        super(up_auto_c0, self).__init__()

        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.convs = double_conv(out_ch,out_ch)
    def forward(self, x1):
        x1 = self.up(x1)
        x1 = self.convs(x1)
        return x1

class up_with_se(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up_with_se, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,  #后期加入可训练的Upsampling
        #  but my machine do not have enough memory to handle all those weights
        #self.se = SELayer(in_ch)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  #
            # self.up = F.interpolate(input,scale_factor=2, mode='bilinear', align_corners=True) 如何改为这一句？
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.encoder_se = SELayer(in_ch // 2)
        self.conv = nn.Sequential(

        #    SELayer(in_ch),
            double_conv(in_ch, out_ch)

        )

    def forward(self, x1, x2):
      #  x1 = self.se(x1)
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        # pad长度必须是偶数，即下列(,,,,)括号内均为偶数
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,  # 左右添加dim
                        diffY // 2, diffY - diffY // 2))  # 上下添加dim

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x2 = self.encoder_se(x2)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)

        return x


class up_with_sk(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up_with_sk, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,  #后期加入可训练的Upsampling
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  #
            # self.up = F.interpolate(input,scale_factor=2, mode='bilinear', align_corners=True) 如何改为这一句？
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = nn.Sequential(

            single_conv(in_ch, out_ch),
            SKConv(out_ch),
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        # pad长度必须是偶数，即下列(,,,,)括号内均为偶数
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,  # 左右添加dim
                        diffY // 2, diffY - diffY // 2))  # 上下添加dim

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
    #    x2 = self.skconv(x2)
        x = torch.cat([x2, x1], dim=1)
        #x = self.skconv(x)
        x = self.conv(x)

        return x


class up_with_DIB_se(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up_with_DIB_se, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  #
            # self.up = F.interpolate(input,scale_factor=2, mode='bilinear', align_corners=True) 如何改为这一句？
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv3 = nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=1)
        self.convs = nn.ModuleList([])
        for i in range(4):
            self.convs.append(nn.Sequential(

                nn.Conv2d(out_ch, out_ch//4, kernel_size=3, stride=1, padding=i + 1, dilation=i + 1),
                # 将channel分为G组，默认为1，即不分
                nn.BatchNorm2d(out_ch//4),
                nn.ReLU(inplace=False)
            ))

        self.CA = nn.Sequential(
            SELayer(out_ch),

        )
        '''
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False)
        )

        '''


    def forward(self, x1, x2):

        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        # pad长度必须是偶数，即下列(,,,,)括号内均为偶数
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,  # 左右添加dim
                        diffY // 2, diffY - diffY // 2))  # 上下添加dim

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv3(x)
        for i, conv in enumerate(self.convs):
            fea = conv(x)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)   #feas:4*4C*80*80
        x = self.CA(feas)
        #x = self.conv2(x)
        return x


class up_with_DIB_se_DIG(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up_with_DIB_se_DIG, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,  #后期加入可训练的Upsampling
        #  but my machine do not have enough memory to handle all those weights
        #self.se = SELayer(in_ch)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  #
            # self.up = F.interpolate(input,scale_factor=2, mode='bilinear', align_corners=True) 如何改为这一句？
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv3 = nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=1)
        self.convs = nn.ModuleList([])
        for i in range(4):
            self.convs.append(nn.Sequential(

                nn.Conv2d(out_ch, out_ch//4, kernel_size=3, stride=1, padding=i + 1, dilation=i + 1),
                # 将channel分为G组，默认为1，即不分
                nn.BatchNorm2d(out_ch//4),
                nn.ReLU(inplace=False)
            ))

        self.CA = nn.Sequential(
            SELayer(out_ch),

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False)
        )




    def forward(self, x1, x2):

        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        # pad长度必须是偶数，即下列(,,,,)括号内均为偶数
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,  # 左右添加dim
                        diffY // 2, diffY - diffY // 2))  # 上下添加dim

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv3(x)
        for i, conv in enumerate(self.convs):
            fea = conv(x)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)   #feas:4*4C*80*80
        x,SA_x,no_SA_x = self.CA(feas)
        x = self.conv2(x)
        return x,SA_x,no_SA_x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):   #Reduction ratio may be change
        super(SELayer, self).__init__()

        self.conv_for_SA = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 3, padding=1),
            nn.BatchNorm2d(channel // reduction),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel // reduction, 1, 1),
            #nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)     #自适应均值池化，输出shape:(channels,1,1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.conv_for_SA(x)
        x2 = x1 * x
        b, c, _, _ = x.size()
        y = self.avg_pool(x2).view(b, c)             #y shape:(batchsize,channels)
        y = self.fc(y).view(b, c, 1, 1)             #channels trans
        return x * y.expand_as(x)   #将y扩展为x的大小


class SKConv(nn.Module):
    def __init__(self, features, WH=80, M=2, G=8, r=16, stride=1, L=8):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        #d = int(features / r)
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(

                nn.Conv2d(features, features, kernel_size=3, stride=stride, dilation=i+1, padding=i+1, groups=G), #将channel分为G组，默认为1，即不分
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=False)
            ))



        self.conv_for_SA = nn.Sequential(
            nn.Conv2d(features, features// r, 3, padding=1),
            nn.BatchNorm2d(features// r),
            nn.ReLU(inplace=False),
            nn.Conv2d(features// r, 1, 1),

            nn.Sigmoid(),
        )

        self.gap = nn.AvgPool2d(int(WH / stride))     #构建函数时确定channel，等的格式，forward时输入只有变量（图像）
        self.fc = nn.Linear(features, d, bias=False)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features, bias=False)
            )
        self.softmax = nn.Softmax(dim=1)   #在dim=1使用softmax

    def forward(self, x):
        b, c, _, _ = x.size()
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)   #feas:4*2*C*80*80
        fea_U = torch.sum(feas, dim=1)             #fea_U:4*C*80*80
        fea_Attention_S = self.conv_for_SA(fea_U)
        fea_U = fea_U * fea_Attention_S
        fea_s = self.gap(fea_U).squeeze(dim=-1).squeeze(dim=-1)  #batch_size * C

        fea_z = self.fc(fea_s)               #fea_z:b*d
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)     #vector:4*1*f
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)   #attention_vectors:4*2*f
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)   #attention_vectors:4*2*C*1*1
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v

class SKConv_SA_get_feature(nn.Module):
    def __init__(self, features, WH=80, M=2, G=32, r=16, stride=1, L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv_SA_get_feature, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(

                nn.Conv2d(features, features, kernel_size=3 + i * 2, stride=stride, padding=i+1, groups=G), #将channel分为G组，默认为1，即不分
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=False)
            ))

        self.conv_for_SA = nn.Sequential(
            nn.Conv2d(features, features // 4, 3, padding=1),
            nn.BatchNorm2d(features // 4),
            nn.ReLU(inplace=False),
            nn.Conv2d(features // 4, 1, 3, padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.gap = nn.AvgPool2d(int(WH / stride))     #构建函数时确定channel，等的格式，forward时输入只有变量（图像）
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)   #在dim=1使用softmax

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)   #feas:4*2*C*80*80
        fea_U1 = torch.sum(feas, dim=1)             #fea_U:4*C*80*80
        fea_Attention_S = self.conv_for_SA(fea_U1)
        fea_U2 = fea_U1 * fea_Attention_S
        fea_s = self.gap(fea_U2).squeeze_(dim=-1).squeeze_(dim=-1)

        fea_z = self.fc(fea_s)               #fea_z:4*d
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)     #vector:4*1*C
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)   #attention_vectors:4*2*C
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)   #attention_vectors:4*2*C*1*1
        fea_v = (feas * attention_vectors).sum(dim=1)
        return feas.cpu(), fea_U1.cpu(), fea_Attention_S.cpu(), fea_U2.cpu(), fea_v.cpu()