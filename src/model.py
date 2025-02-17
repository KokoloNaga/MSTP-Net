import torch
import torch.nn as nn


class Model(nn.Module):
    # downsize_nb_filters_factor=4 compare to DUNetV1V2_MM
    def __init__(self, n_channels, n_classes, downsize_nb_filters_factor=4):
        super(Model, self).__init__()
        # self.inc = deform_inconv(n_channels, 64 // downsize_nb_filters_factor)
        self.inc = inconv(n_channels, 64 // downsize_nb_filters_factor)
        self.down1 = down(64 // downsize_nb_filters_factor, 128 // downsize_nb_filters_factor)
        self.down2 = down(128 // downsize_nb_filters_factor, 256 // downsize_nb_filters_factor)
        # self.down3 = deform_down(256 // downsize_nb_filters_factor, 512 // downsize_nb_filters_factor)
        self.down3 = down(256 // downsize_nb_filters_factor, 512 // downsize_nb_filters_factor)
        self.down4 = down(512 // downsize_nb_filters_factor, 512 // downsize_nb_filters_factor)
        self.neck = nn.Sequential(
            *[double_deform_conv_T(512 // downsize_nb_filters_factor, 512 // downsize_nb_filters_factor) for _ in range(1)]
        )

        self.up1 = up(1024 // downsize_nb_filters_factor, 256 // downsize_nb_filters_factor, is_up=False)
        self.up2 = up(512 // downsize_nb_filters_factor, 128 // downsize_nb_filters_factor, is_up=False)
        self.up3 = up(256 // downsize_nb_filters_factor, 64 // downsize_nb_filters_factor)
        self.up4 = up(128 // downsize_nb_filters_factor, 64 // downsize_nb_filters_factor)

        self.up1_ = up_(1024 // downsize_nb_filters_factor, 256 // downsize_nb_filters_factor, is_up=False)
        self.up2_ = up_(512 // downsize_nb_filters_factor, 128 // downsize_nb_filters_factor, is_up=False)
        self.up3_ = up_(256 // downsize_nb_filters_factor, 64 // downsize_nb_filters_factor)
        self.up4_ = up_(128 // downsize_nb_filters_factor, 64 // downsize_nb_filters_factor)

        self.up1_2 = up_2(256 // downsize_nb_filters_factor)
        self.up2_2 = up_2(128 // downsize_nb_filters_factor)
        self.up3_2 = up_2(64 // downsize_nb_filters_factor)
        self.up4_2 = up_2(64 // downsize_nb_filters_factor, last=True)

        self.outc1 = nn.Conv2d(64 // downsize_nb_filters_factor, n_classes, 1)
        self.outc2 = nn.Conv2d(64 // downsize_nb_filters_factor, n_classes, 1)
        self.outc = nn.Conv2d(64 // downsize_nb_filters_factor+3, n_classes, 1)

    def forward(self, inp):

        x1 = self.inc(inp)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.neck(x5)

        xd1 = self.up1(x5, x4)
        xd2 = self.up2(xd1, x3)
        xd3 = self.up3(xd2, x2)
        xd4 = self.up4(xd3, x1)

        xc1 = self.up1_(x5, x4)
        xc2 = self.up2_(xc1, x3)
        xc3 = self.up3_(xc2, x2)
        xc4 = self.up4_(xc3, x1)

        o1 = self.outc1(xd4)
        o2 = self.outc2(xc4)

        xa1 = self.up1_2(x5, xd1, xc1)
        xa2 = self.up2_2(xa1, xd2, xc2)
        xa3 = self.up3_2(xa2, xd3, xc3)
        xa4 = self.up4_2(xa3, xd4, xc4)

        x = torch.cat([inp, xa4], dim=1)
        x = self.outc(x)

        return {'out': 0.5 * x + 0.25 * o1 + 0.25 * o2, 'out1': o1, 'out2': o2, 'out3': x}


