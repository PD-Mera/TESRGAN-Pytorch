import math
import torch
from torch import nn, sigmoid


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels = 64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels *
                              up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class Generator(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2)) - 1

        super(Generator, self).__init__()
        self.feBlock1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.resBlock1 = nn.Sequential(ResidualBlock(64, 64),
                                       ResidualBlock(64, 64),
                                       ResidualBlock(64, 64))
        self.upBlock1 = UpsampleBLock(64, 2)
        self.conv_edge = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv = nn.Conv2d(72, 64, kernel_size=3, padding=1)
        self.resBlock2 = nn.Sequential(ResidualBlock(64, 64),
                                       ResidualBlock(64, 64),
                                       ResidualBlock(64, 64))
        self.feBlock2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        upBlock2 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        upBlock2.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.upBlock2 = nn.Sequential(*upBlock2)
        self.output = nn.Tanh()



    def forward(self, x, edge):
        fe = self.feBlock1(x) # 64
        res1 = self.resBlock1(fe) # 64
        up1 = self.upBlock1(res1 + fe) # 64
        edge = self.conv_edge(edge)
        up1 = torch.cat((up1, edge), dim=1) # 72
        cov = self.conv(up1) # 64
        res2 = self.resBlock2(cov) # 64
        fe2 = self.feBlock2(res2 + cov) # 64
        up2 = self.upBlock2(fe2)

        return (self.output(up2)+1.0)/2.0  # or only tanh?


'''
class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            # input size. (3) x 512 x 640
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), bias=False),
            nn.LeakyReLU(0.2, True),
            # state size. (64) x 256 x 320
            nn.Conv2d(64, 64, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # state size. (128) x 128 x 160
            nn.Conv2d(128, 128, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # state size. (256) x 64 x 80
            nn.Conv2d(256, 256, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 32 x 40
            nn.Conv2d(512, 512, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1024, (3, 3), (1, 1), (1, 1), bias=False),
            nn.LeakyReLU(0.2, True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        out = self.features(x)
        out = self.classifier(out)
        return torch.sigmoid(out)
'''


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),         # @Thuan:

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3,  padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),   # @Thuan: return batch*128*1*1 tensor
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            # @Thuan: return batch*1*1*1 tensor
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        return self.net(x)
        # batch_size = x.size(0)
        # @Thuan: OK
        # return torch.sigmoid(self.net(x).view(batch_size))


class Discriminator_WGAN(nn.Module):
    def __init__(self, l=0.2):
        super(Discriminator_WGAN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(l, True),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(l, True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(l, True),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(l, True),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(l, True),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(l, True),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(l, True),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(l, True),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(l, True),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        #print ('D input size :' +  str(x.size()))
        y = self.net(x)
        #print ('D output size :' +  str(y.size()))
        return y.view(y.size()[0])


def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.randn(real_samples.size(0), 1, 1, 1)
    if torch.cuda.is_available():
        alpha = alpha.cuda()

    interpolates = (alpha * real_samples + ((1 - alpha)
                                            * fake_samples)).requires_grad_(True)

    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size())
    if torch.cuda.is_available():
        fake = fake.cuda()

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
