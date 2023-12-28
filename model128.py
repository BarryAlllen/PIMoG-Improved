import torch
import torch.nn as nn
import torch.nn.functional as F
from Noise_Layer import Identity, ScreenShooting

# 128x128 15bit

class ConvBNRelu(nn.Module):
    def __init__(self, channels_in, channels_out, stride=1):

        super(ConvBNRelu, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)

class SingleConv(nn.Module):
    def __init__(self, inchannel, outchannel, s):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=s, padding=1, bias=True),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, s):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=s, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if s != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=s, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, num_channels):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
                ConvBNRelu(3,num_channels),
                ConvBNRelu(num_channels,num_channels),
                ConvBNRelu(num_channels,num_channels),
                nn.AdaptiveAvgPool2d(output_size = (1,1))
                )
        self.linear = nn.Linear(num_channels,1)
    def forward(self,x):
        D = self.discriminator(x)
        print(f"D shape: {D.shape}")
        print(f"D: {D}")
        D.squeeze_(3).squeeze_(2)
        print(f"D squeeze3 & squeeze2 shape: {D.shape}")
        print(f"D: {D}")
        D = self.linear(D)
        return D

class DoubleConv(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class U_Net_Encoder_Diffusion(nn.Module):
    def __init__(self, inchannel=3, outchannel=3):
        super(U_Net_Encoder_Diffusion, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Globalpool = nn.MaxPool2d(kernel_size=4, stride=4)

        self.Conv1 = DoubleConv(inchannel, 16)
        self.Conv2 = DoubleConv(16, 32)
        self.Conv3 = DoubleConv(32, 64)

        self.Up4 = up_conv(64*3, 64)
        self.Conv7 = DoubleConv(64*3, 64)

        self.Up3 = up_conv(64, 32)
        self.Conv8 = DoubleConv(32*2+64, 32)

        self.Up2 = up_conv(32, 16)
        self.Conv9 = DoubleConv(16*2+64, 16)

        self.Conv_1x1 = nn.Conv2d(16, outchannel, kernel_size=1, stride=1, padding=0)
        self.linear = nn.Linear(15,256)
        self.Conv_message = DoubleConv(1,64)


    def forward(self, x, watermark): # 3x128x128 1x15
        x1 = self.Conv1(x) # 16x128x128

        x2 = self.Maxpool(x1) # 16x64x64
        x2 = self.Conv2(x2) # 32x64x64

        x3 = self.Maxpool(x2) # 32x32x32
        x3 = self.Conv3(x3) # 64x32x32

        x4 = self.Maxpool(x3) # 64x16x16

        x6 = self.Globalpool(x4) # 64x4x4
        x7 = x6.repeat(1,1,4,4) # 64x16x16`
        print(f"x7 shape: {x7.shape}")

        expanded_message = self.linear(watermark) # 1x256
        expanded_message = expanded_message.view(-1,1,16,16) # 1x16x16`
        expanded_message = self.Conv_message(expanded_message) # 64x16x16
        x4 = torch.cat((x4, x7, expanded_message), dim=1) # 192x16x16`
        print(f"x4 shape: {x4.shape}")

        d4 = self.Up4(x4) # 64x32x32
        expanded_message = self.linear(watermark) # 1x256
        expanded_message = expanded_message.view(-1,1,16,16) # 1x16x16`
        expanded_message = torch.nn.functional.interpolate(expanded_message,size=(d4.shape[2],d4.shape[3]),mode='bilinear') # 1x32x32`
        expanded_message = self.Conv_message(expanded_message) # 64x32x32
        d4 = torch.cat((x3, d4, expanded_message), dim=1) # 192x32x32`
        d4 = self.Conv7(d4) # 64x32x32

        d3 = self.Up3(d4) # 32x64x64
        expanded_message = self.linear(watermark) # 1x256
        expanded_message = expanded_message.view(-1,1,16,16) # 1x16x16`
        expanded_message = torch.nn.functional.interpolate(expanded_message,size=(d3.shape[2],d3.shape[3]),mode='bilinear') # 1x64x64`
        expanded_message = self.Conv_message(expanded_message) # 64x64x64
        d3 = torch.cat((x2, d3, expanded_message), dim=1) # 128x64x64`
        d3 = self.Conv8(d3) # 32x64x64

        d2 = self.Up2(d3) # 16x128x128
        expanded_message = self.linear(watermark) # 1x256
        expanded_message = expanded_message.view(-1,1,16,16) # 1x16x16`
        expanded_message = torch.nn.functional.interpolate(expanded_message,size=(d2.shape[2],d2.shape[3]),mode='bilinear') # 1x128x128`
        expanded_message = self.Conv_message(expanded_message) # 64x128x128
        d2 = torch.cat((x1, d2, expanded_message), dim=1) # 96x128x128`
        d2 = self.Conv9(d2) # 16x128x128

        out = self.Conv_1x1(d2) # 3x128x128

        return out

class Extractor(nn.Module):
    def __init__(self, inchannel=64):
        super(Extractor,self).__init__()
        self.layer1 = SingleConv(inchannel,64,1)
        self.layer2 = nn.Sequential(ResidualBlock(64,64,1), ResidualBlock(64,64,2))
        self.layer3 = nn.Sequential(ResidualBlock(64,64,1), ResidualBlock(64,64,2))
        self.layer4 = nn.Sequential(ResidualBlock(64,64,1), ResidualBlock(64,64,2))
        self.layer5 = nn.Conv2d(64,1,kernel_size=1,stride=1,padding=0,bias=False) # 1x1 conv
        self.linear = nn.Linear(256,15)
    def forward(self, x):
        out = self.layer1(x) # 64x128x128
        out = self.layer2(out) # 64x64x64
        out = self.layer3(out) # 64x32x32
        out = self.layer4(out) # 64x16x16
        out = self.layer5(out) # 1x16x16
        # print(f"out5 shape: {out.shape}")
        # print(f"out5: {out}")
        out.squeeze_(1)
        # print(f"out.squeeze_(1) shape: {out.shape}")
        # print(f"out.squeeze_(1): {out}")
        out = out.view(-1,1,256)
        # print(f"out.view(-1,1,256) shape: {out.shape}")
        # print(f"out.view(-1,1,256): {out}")
        out = self.linear(out)
        # print(f"linear(out) shape: {out.shape}")
        # print(f"linear(out): {out}")
        out.squeeze_(1)
        # print(f"out.squeeze_(1) shape: {out.shape}")
        # print(f"out.squeeze_(1): {out}")
        return out

		
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.extractor = Extractor()
        self.layer1 = nn.Sequential(
            SingleConv(3,64,1), # 64x128x128
            SingleConv(64,64,1), # 64x128x128
            SingleConv(64,64,1), # 64x128x128
            ResidualBlock(64,64,1), # 64x128x128
            ResidualBlock(64,64,1), # 64x128x128
            ResidualBlock(64,64,1), # 64x128x128
            )

    def forward(self, x):
        x1 = self.layer1(x)
        Message = self.extractor(x1)
        return Message

		
class Encoder_Decoder(nn.Module):
	def __init__(self, distortion):
		super(Encoder_Decoder, self).__init__()
		self.Encoder = U_Net_Encoder_Diffusion()
		self.Decoder = Decoder()
		self.distortion = distortion
		if distortion == 'Identity':
		    self.Noiser = Identity()
		elif distortion == 'ScreenShooting':
		    self.Noiser = ScreenShooting()

	def forward(self, x, m):
		Encoded_image = self.Encoder(x, m)
		Noised_image = self.Noiser(Encoded_image)
		Decoded_message = self.Decoder(Noised_image.float())
		return Encoded_image, Noised_image, Decoded_message

