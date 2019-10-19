# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F
import torch
from .unet_parts import *
from torch.autograd import Variable
#cuda_device = 0

class No_skip_CNN(nn.Module):
    def __init__(self, n_channels, n_classes, high_represent_dim):
        super(No_skip_CNN, self).__init__()

        self.inc = inconv(n_channels, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 256)
        self.flat_fc = fc(int(256 * 256), high_represent_dim)
        self.deflat = nn.ConvTranspose2d(high_represent_dim, 256, (16, 16))
        self.conv = double_conv(256, 192)
        self.up1 = up_no_skip(192, 128)
        self.up2 = up_no_skip(128, 96)
        self.up3 = up_no_skip(96, 64)
        self.up4 = up_no_skip(64, 32)
        self.outc = outconv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        high_represent = self.flat_fc(x5)
        x_deflat = self.deflat(high_represent.unsqueeze(-1).unsqueeze(-1))
        f5_d = self.conv(x_deflat)
        x4_d = self.up1(f5_d)
        x3_d = self.up2(x4_d)
        x2_d = self.up3(x3_d)
        x1_d = self.up4(x2_d)
        x_out = self.outc(x1_d)
        return F.softmax(x_out, dim=1), high_represent  # F

class Unet_CNN(nn.Module):
    def __init__(self, n_channels, n_classes, high_represent_dim):
        super(Unet_CNN, self).__init__()

        self.inc = inconv(n_channels, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 256)
        self.flat_fc = fc(int(256 * 256), high_represent_dim)
        self.deflat = nn.ConvTranspose2d(high_represent_dim, 256, (16, 16))
        self.conv = bi_conv(512, 192)
        self.up1 = up(448, 160)
        self.up2 = up(288, 128)
        self.up3 = up(192, 96)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        high_represent = self.flat_fc(x5)
        x_deflat = self.deflat(high_represent.unsqueeze(-1).unsqueeze(-1))
        f5_d = self.conv(x_deflat, x5)
        x4_d = self.up1(f5_d, x4)
        x3_d = self.up2(x4_d, x3)
        x2_d = self.up3(x3_d, x2)
        x1_d = self.up4(x2_d, x1)
        x_out = self.outc(x1_d)
        return F.softmax(x_out, dim=1)  # F

class UNet_multi_class(nn.Module):
    def __init__(self, n_channels, n_classes):

        super(UNet_multi_class, self).__init__()

        self.inc = inconv(n_channels, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 512)
        self.down5 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 32)
        self.up5 = up(64, 32)
        self.outc = outconv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        x = self.outc(x)
        return F.softmax(x,dim=1)  #F
'''
class CNN_multi_class(nn.Module):
    def __init__(self, n_channels, n_classes, input_dim=64):

        super(CNN_multi_class, self).__init__()

        self.inc = inconv(n_channels, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 256)
        self.flat_fc = fc(102400,64) # output shape(batchsize * 64)
        self.biGRU = nn.GRU(input_size=input_dim,
                               hidden_size=input_dim,
                               num_layers=1,
                               batch_first=True,
                               bidirectional=True)
        self.fcl = fc1(128,256)
        self.deflat = nn.ConvTranspose2d(256, 256, (16,25))
        self.biconv = bi_conv(512, 192)
        self.up2 = up(448, 160)
        self.up3 = up(288, 128)
        self.up4 = up(192, 96)
        self.up5 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x,n_classes=4, input_dim=64): #x shape :(batch, seq_len, n_channels, 256, 400) input_dim = hidden_dim
        #x_mask = (torch.zeros(x.shape[0], x.shape[1], n_classes, x.shape[2], x.shape[3])).cuda
        x_mask_l = []
        feature_seq = Variable((torch.zeros(x.shape[0], x.shape[1], input_dim)).cuda())
        f1 = Variable(torch.zeros(x.shape[0], x.shape[1], 32, x.shape[2], x.shape[3]).cuda())
        f2 = Variable((torch.zeros(x.shape[0], x.shape[1], 64, x.shape[2] // 2, x.shape[3] // 2)).cuda())
        f3 = Variable((torch.zeros(x.shape[0], x.shape[1], 128, x.shape[2] // 4, x.shape[3] // 4)).cuda())
        f4 = Variable((torch.zeros(x.shape[0], x.shape[1], 256, x.shape[2] // 8, x.shape[3] // 8)).cuda())
        f5 = Variable((torch.zeros(x.shape[0], x.shape[1], 256, x.shape[2] // 16, x.shape[3] // 16)).cuda())  # batch * seq_len * 16 * 25
        x = x.unsqueeze(dim=2)
        for i in range(x.shape[1]):
            f1[:, i, ] = self.inc(x[:, i, ])
            f2[:, i, ] = self.down1(f1[:, i, ])
            f3[:, i, ] = self.down2(f2[:, i, ])
            f4[:, i, ] = self.down3(f3[:, i, ])
            f5[:, i, ] = self.down4(f4[:, i, ])
            feature_seq[:, i, ] = self.flat_fc(f5[:, i, ])
        h0 = (Variable(torch.randn(2, x.shape[0], input_dim))).cuda()
        feature_seq_GRU, h = self.biGRU(feature_seq, h0)  # feature_seq_GRU shape : (batch * seq_len * 128(bi-direction))
        x_fc = Variable((torch.zeros(x.shape[0], x.shape[1], 256, 1, 1)).cuda())
        x0 = Variable((torch.zeros(x.shape[0], x.shape[1], 256, 16, 25)).cuda())
        x1 = Variable((torch.zeros(x.shape[0], x.shape[1], 192, 16, 25)).cuda())
        x2 = Variable((torch.zeros(x.shape[0], x.shape[1], 160, 32, 50)).cuda())
        x3 = Variable((torch.zeros(x.shape[0], x.shape[1], 128, 64, 100)).cuda())
        x4 = Variable((torch.zeros(x.shape[0], x.shape[1], 96, 128, 200)).cuda())
        x5 = Variable((torch.zeros(x.shape[0], x.shape[1], 64, 256, 400)).cuda())
        x6 = Variable((torch.zeros(x.shape[0], x.shape[1], n_classes, 256, 400)).cuda())
        x7 = Variable((torch.zeros(x.shape[0], 2, n_classes, 256, 400)).cuda())

        for i in range(x.shape[1]):
            x_fc[:, i, ] = self.fcl(feature_seq_GRU[:, i, ])
            x0[:, i, ] = self.deflat(x_fc[:, i, ])
            x1[:, i, ] = self.biconv(x0[:, i, ], f5[:, i, ])
            x2[:, i, ] = self.up2(x1[:, i, ], f4[:, i, ])
            x3[:, i, ] = self.up3(x2[:, i, ], f3[:, i, ])
            x4[:, i, ] = self.up4(x3[:, i, ], f2[:, i, ])
            x5[:, i, ] = self.up5(x4[:, i, ], f1[:, i, ])
            x6[:, i, ] = self.outc(x5[:, i, ])

            if (i==0):
                x7[:, 0, ] = F.softmax(x6[:, i, ], dim=1)
            if (i==9):
                x7[:, 1, ] = F.softmax(x6[:, i, ], dim=1)
        return x7 #x_mask_l
'''
class encoder_image_sequences(nn.Module):
    def __init__(self, n_channels=1, high_represent_dim = 64):

        super(encoder_image_sequences, self).__init__()

        self.inc = inconv(n_channels, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 256)
        self.flat_fc = fc(int(256*256),high_represent_dim) # input shape(batchsize,16*25*256) output shape:(batchsize * high_represent_dim)

    def forward(self, x):
        f1 = self.inc(x)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        f4 = self.down3(f3)
        f5 = self.down4(f4)
        high_v = self.flat_fc(f5)
        return high_v,f1,f2,f3,f4,f5

class decoder_image_sequences(nn.Module):
    def __init__(self, n_classes = 4):

        super(decoder_image_sequences, self).__init__()
        self.deflat = nn.ConvTranspose2d(64, 256, (16,16))
        self.conv = bi_conv(512, 192)
        self.up1 = up(448, 160)
        self.up2 = up(288, 128)
        self.up3 = up(192, 96)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x, x1, x2, x3, x4, x5):

        x_deflat = self.deflat(x.unsqueeze(-1).unsqueeze(-1))
        f5_d = self.conv(x_deflat, x5)
        x4_d = self.up1(f5_d, x4)
        x3_d = self.up2(x4_d, x3)
        x2_d = self.up3(x3_d, x2)
        x1_d = self.up4(x2_d,x1)
        x_out = self.outc(x1_d)
        return F.softmax(x_out,dim=1)  #F

class bi_GRU(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, num_layers=1):
        super(bi_GRU, self).__init__()
        self.biGRU = nn.GRU(input_size=input_dim,
                               hidden_size=hidden_dim,
                               num_layers=num_layers,
                               batch_first=True,
                               bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim, bias=False),
            nn.ReLU(inplace=False)
        )
    def forward(self, x, seq_len=10, input_dim=64, hidden_dim=64, cuda_device=0):
        h0 = Variable(torch.randn(2, x.shape[0], hidden_dim)).cuda(cuda_device)
        bi_x, _ = self.biGRU(x, h0)  #feature_seq_GRU shape : (batch * seq_len * 128(bi-direction))
        feature_seq_GRU = self.fc(bi_x)
        return feature_seq_GRU

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv_with_DIB_se(n_channels, 64)
        self.down1 = down_with_DIB_se(64, 128)
        self.down2 = down_with_DIB_se(128, 256)
        self.down3 = down(256, 256)
        self.up1 = up(512, 128)
        self.up2 = up_with_DIB_se(256, 64)
        self.up3 = up_with_DIB_se(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        return torch.sigmoid(x)

class encoder(nn.Module):
    def __init__(self, n_channels):
        super(encoder, self).__init__()
        self.inc_label = inconv(n_channels, 64)
        self.down1_label = down(64, 128)
        self.down2_label = down(128, 256)
        self.down3_label = down(256, 512)
        self.fc1_label = fc1(512, 64)

        for p in self.parameters():
            p.require_grad = False


    def forward(self, x):
        x1 = self.inc_label(x)

        x2 = self.down1_label(x1)
        x3 = self.down2_label(x2)
        x4 = self.down3_label(x3)
        x_fc1 = self.fc1_label(x4)
        return x_fc1

class ACNN(nn.Module):  # 继承Module类，并实现forward方法
    def __init__(self, n_channels, n_classes):
        # nn.Module的子类函数必须在构造函数中执行父类的构造函数
        super(ACNN, self).__init__()       # 或写成nn.Module.__init__(self)
        #nn.Module.__init__(self)
        # nn.Conv2d返回的是一个Conv2d class的一个对象，该类中包含forward函数的实现
        # 当调用self.inc(input)的时候，就会调用该类的forward函数
        #self.inc = inconv(n_channels, 64)
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 256)
     #   self.down4 = down(512, 512)
        self.up1 = up(512, 128)
        self.up2 = up(256, 64)
        self.up3 = up(128, 64)
    #    self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)
        self.eocoder_label = encoder(1)
        '''
        self.inc_label = inconv(n_channels, 64)
        self.down1_label = down(64, 128)
        self.down2_label = down(128, 256)
        self.down3_label = down(256, 512)
        self.fc1_label = fc1(512, 64)
        '''


    def forward(self, x):
        x1 = self.inc(x)
        #x1 = self.sk(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
    #    x5 = self.down4(x4)
        x2_ = self.up1(x4, x3)
        x1_ = self.up2(x2_, x2)
        x_ = self.up3(x1_, x1)
        x0_ = self.outc(x_)
        x00 = torch.sigmoid(x0_)
        x__ = torch.tensor((x00 > 0.5),dtype=torch.float32).cuda()
        x_vector = self.eocoder_label(x__)
        '''
        x1__ = self.inc_label(x__)

        x2__ = self.down1_label(x1__)
        x3__ = self.down2_label(x2__)
        x4__ = self.down3_label(x3__)
        x_vector = self.fc1_label(x4__)
        '''
        return x00,x_vector
'''
class autoencoder(nn.Module):
    def __init__(self, n_channels, n_classes):
        # nn.Module的子类函数必须在构造函数中执行父类的构造函数
        super(autoencoder, self).__init__()       # 或写成nn.Module.__init__(self)

        self.inc_label = inconv(n_channels, 64)
        self.down1_label = down(64, 128)
        self.down2_label = down(128, 256)
        self.down3_label = down(256, 512)
        self.fc1_label = fc1(512,64)
        self.fc2_label = fc2(64,512)
        self.deconv_label = deconv0(512,512)
        self.up1_label = deconv1(512, 256)
        self.up2_label = deconv1(256, 128)
        self.up3_label = deconv1(128, 64)

        self.outc_label = outconv(64, n_classes)


    def forward(self, x):
        x1 = self.inc_label(x)

        x2 = self.down1_label(x1)
        x3 = self.down2_label(x2)
        x4 = self.down3_label(x3)
        x_fc1 = self.fc1_label(x4)
        x_fc2 = self.fc2_label(x_fc1)
        x3_ = self.deconv_label(x_fc2)

        x2_ = self.up1_label(x3_)
        x1_ = self.up2_label(x2_)
        x_ = self.up3_label(x1_)
        x = self.outc_label(x_)
        return torch.sigmoid(x)
'''
class autoencoder_c0(nn.Module):
    def __init__(self, n_channels, n_classes):
        # nn.Module的子类函数必须在构造函数中执行父类的构造函数
        super(autoencoder_c0, self).__init__()       # 或写成nn.Module.__init__(self)

        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up_auto_c0(512, 256)
        self.up2 = up_auto_c0(256, 128)
        self.up3 = up_auto_c0(128, 64)
        self.up4 = up_auto_c0(64, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.outc(x)
        return F.softmax(x, dim=1)  # F


'''
class UNet(nn.Module):  # 继承Module类，并实现forward方法
    def __init__(self, n_channels, n_classes):
        # nn.Module的子类函数必须在构造函数中执行父类的构造函数
        super(UNet, self).__init__()       # 或写成nn.Module.__init__(self)
        #nn.Module.__init__(self)
        # nn.Conv2d返回的是一个Conv2d class的一个对象，该类中包含forward函数的实现
        # 当调用self.inc(input)的时候，就会调用该类的forward函数
        #self.inc = inconv(n_channels, 64)
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
      #  self.down3 = down(256, 256)
     #   self.down4 = down(512, 512)
        self.up1 = up(384, 128)
        self.up2 = up(192, 64)
    #    self.up3 = up(128, 64)
    #    self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        #x1 = self.sk(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
     #   x4 = self.down3(x3)
    #    x5 = self.down4(x4)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
     #   x = self.up3(x, x1)
        x = self.outc(x)
        return torch.sigmoid(x)
'''

class UNet_for_fea(nn.Module):  # 继承Module类，并实现forward方法
    def __init__(self, n_channels, n_classes):
        # nn.Module的子类函数必须在构造函数中执行父类的构造函数
        super(UNet_for_fea, self).__init__()       # 或写成nn.Module.__init__(self)
        #nn.Module.__init__(self)
        # nn.Conv2d返回的是一个Conv2d class的一个对象，该类中包含forward函数的实现
        # 当调用self.inc(input)的时候，就会调用该类的forward函数
        self.inc = inconv(n_channels, 64)
        #
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        #self.down3 = bottom_conv_sk(256, 256)
        self.down3 = down(256, 256)
     #   self.down4 = down(512, 512)
        #self.up1 = up_with_sk(512, 128)
        #self.up2 = up_with_sk(256, 64)
        self.up1 = up(512, 128)
        self.up2 = up(256, 64)
        self.up3 = up(128, 64)
    #    self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        #x1 = self.sk(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
    #    x5 = self.down4(x4)
        x3_ = self.up1(x4, x3)
        x2_ = self.up2(x3_, x2)
        x1_ = self.up3(x2_, x1)
        x_ = self.outc(x1_)
        x_out = torch.sigmoid(x_)
        return x1.cpu(), x2.cpu(), x3.cpu(), x4.cpu(), x3_.cpu(), x2_.cpu(), x1_.cpu(), x_.cpu(), x_out.cpu()


class UNet_for_sk_fea(nn.Module):  # 继承Module类，并实现forward方法
    def __init__(self, n_channels, n_classes):
        # nn.Module的子类函数必须在构造函数中执行父类的构造函数
        super(UNet_for_sk_fea, self).__init__()       # 或写成nn.Module.__init__(self)
        #nn.Module.__init__(self)
        # nn.Conv2d返回的是一个Conv2d class的一个对象，该类中包含forward函数的实现
        # 当调用self.inc(input)的时候，就会调用该类的forward函数
        self.inc = inconv(n_channels, 64)
        #
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        #self.down3 = bottom_conv_sk(256, 256)
        self.down3 = down(256, 256)
     #   self.down4 = down(512, 512)
        #self.up1 = up_with_sk(512, 128)
        #self.up2 = up_with_sk(256, 64)
        self.up1 = up(512, 128)
        self.up2 = up(256, 64)
        self.up3 = up(128, 64)
    #    self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        #x1 = self.sk(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
    #    x5 = self.down4(x4)
        x3_ = self.up1(x4, x3)
        x2_ = self.up2(x3_, x2)
        x1_ = self.up3(x2_, x1)
        x_ = self.outc(x1_)
        x_out = torch.sigmoid(x_)
        return x1.cpu(), x2.cpu(), x3.cpu(), x4.cpu(), x3_.cpu(), x2_.cpu(), x1_.cpu(), x_.cpu(), x_out.cpu()