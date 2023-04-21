import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm2d

#残差ブロック
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, int(in_channels/2), kernel_size=1, stride=1 ,padding=0) 
        self.bn1 = nn.BatchNorm2d(int(in_channels/2))
        self.lr1 = nn.LeakyReLU(0.1, True)
        self.conv2 = nn.Conv2d(int(in_channels/2), in_channels, kernel_size=3, stride=1 ,padding=1) ###
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.lr2 = nn.LeakyReLU(0.1, True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lr1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.lr2(out)      
        out = out + residual
        return out

class YOLOv3(nn.Module):
 def __init__(self,in_channel=3, class_n = 80):
   super(YOLOv3 , self).__init__()
   self.class_n = class_n

   #darknet-53
   self.first_block = nn.Sequential(
       nn.Conv2d(in_channel, 32, kernel_size=3, stride=1, padding=1),
       nn.BatchNorm2d(32),
       nn.LeakyReLU(negative_slope=0.1,inplace=True),
       nn.Conv2d(32, 64, 3, 2, 1),
       nn.BatchNorm2d(64),
       nn.LeakyReLU(0.1, True),   
   )
   
   self.res_block1 = ResidualBlock(64)
   self.conv1 = nn.Conv2d(64, 128, 3, 2, 1)
   
   self.res_block2 = nn.Sequential(
       ResidualBlock(128),
       ResidualBlock(128),
   )
   self.conv2 = nn.Conv2d(128, 256, 3, 2, 1)
   
   self.res_block3 = nn.Sequential(
       ResidualBlock(256),
       ResidualBlock(256),
       ResidualBlock(256),
       ResidualBlock(256),
       ResidualBlock(256),
       ResidualBlock(256),
       ResidualBlock(256),
       ResidualBlock(256),
   )       
   self.conv3 = nn.Conv2d(256, 512, 3, 2, 1)

   self.res_block4 = nn.Sequential(
       ResidualBlock(512),
       ResidualBlock(512),
       ResidualBlock(512),
       ResidualBlock(512),
       ResidualBlock(512),
       ResidualBlock(512),
       ResidualBlock(512),
       ResidualBlock(512),
   )   
   self.conv4 = nn.Conv2d(512, 1024, 3, 2, 1)

   self.res_block5 = nn.Sequential(
       ResidualBlock(1024),
       ResidualBlock(1024),
       ResidualBlock(1024),
       ResidualBlock(1024),
   )  

   self.ConvBlock = nn.Sequential(
       nn.Conv2d(1024, 512, 1, 1),
       nn.BatchNorm2d(512),
       nn.LeakyReLU(negative_slope=0.1,inplace=True),
       nn.Conv2d(512, 1024, 3, 1, 1),
       nn.BatchNorm2d(1024),
       nn.LeakyReLU(negative_slope=0.1,inplace=True),
  
       nn.Conv2d(1024, 512, 1, 1),
       nn.BatchNorm2d(512),
       nn.LeakyReLU(negative_slope=0.1,inplace=True),
       nn.Conv2d(512, 1024, 3, 1, 1),
       nn.BatchNorm2d(1024),
       nn.LeakyReLU(negative_slope=0.1,inplace=True),

       nn.Conv2d(1024, 512, 1, 1),
       nn.BatchNorm2d(512),
       nn.LeakyReLU(negative_slope=0.1,inplace=True),
       nn.Conv2d(512, 1024, 3, 1, 1),
       nn.BatchNorm2d(1024),
       nn.LeakyReLU(negative_slope=0.1,inplace=True),
   )

   #FPN構造
   #スケール3
   self.scale3_output = nn.Conv2d(1024, (3*(4+1+self.class_n)), 1, 1)

   #スケール2
   self.scale2_tmp = nn.Conv2d(1024, 256, 1, 1)
   self.scale2_UpTmp = nn.Upsample(scale_factor = 2)

   self.scale2_ConvBlock = nn.Sequential(
       nn.Conv2d(768, 256, 1, 1),
       nn.BatchNorm2d(256),
       nn.LeakyReLU(negative_slope=0.1,inplace=True),
       nn.Conv2d(256, 512, 3, 1, 1),
       nn.BatchNorm2d(512),
       nn.LeakyReLU(negative_slope=0.1,inplace=True),
  
       nn.Conv2d(512, 256, 1, 1),
       nn.BatchNorm2d(256),
       nn.LeakyReLU(negative_slope=0.1,inplace=True),
       nn.Conv2d(256, 512, 3, 1, 1),
       nn.BatchNorm2d(512),
       nn.LeakyReLU(negative_slope=0.1,inplace=True),

       nn.Conv2d(512, 256, 1, 1),
       nn.BatchNorm2d(256),
       nn.LeakyReLU(negative_slope=0.1,inplace=True),
       nn.Conv2d(256, 512, 3, 1, 1),
       nn.BatchNorm2d(512),
       nn.LeakyReLU(negative_slope=0.1,inplace=True),
   )
   self.scale2_output = nn.Conv2d(512, (3*(4+1+self.class_n)), 1, 1)

   #スケール1
   self.scale1_tmp = nn.Conv2d(512, 128, 1, 1)
   self.scale1_UpTmp = nn.Upsample(scale_factor= 2)

   self.scale1_ConvBlock = nn.Sequential(
       nn.Conv2d(384, 128, 1, 1),
       nn.BatchNorm2d(128),
       nn.LeakyReLU(negative_slope=0.1,inplace=True),
       nn.Conv2d(128, 256, 3, 1, 1),
       nn.BatchNorm2d(256),
       nn.LeakyReLU(negative_slope=0.1,inplace=True),
  
       nn.Conv2d(256, 128, 1, 1),
       nn.BatchNorm2d(128),
       nn.LeakyReLU(negative_slope=0.1,inplace=True),
       nn.Conv2d(128, 256, 3, 1, 1),
       nn.BatchNorm2d(256),
       nn.LeakyReLU(negative_slope=0.1,inplace=True),

       nn.Conv2d(256, 128, 1, 1),
       nn.BatchNorm2d(128),
       nn.LeakyReLU(negative_slope=0.1,inplace=True),
       nn.Conv2d(128, 256, 3, 1, 1),
       nn.BatchNorm2d(256),
       nn.LeakyReLU(negative_slope=0.1,inplace=True),
   )
   self.scale1_output = nn.Conv2d(256, (3*(4+1+self.class_n)), 1, 1)

   #Global Average Poolingとして用いる。 (darknetのネットワーク)
   #self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))

 def forward(self, x):
   x = self.first_block(x)
   #print(x.shape)
   x = self.res_block1(x)
   x = self.conv1(x)

   x = self.res_block2(x)
   x = self.conv2(x)

   x = self.res_block3(x)
   scale1 = x
   x = self.conv3(x)

   x = self.res_block4(x)
   scale2 = x
   x = self.conv4(x)

   x = self.res_block5(x)
   x = self.ConvBlock(x)
  #  for layer in self.conv_block:
  #    x = layer(x)
   scale2_2 = x
   #print(scale2_2.shape)
   scale3 = x

   scale3_out = self.scale3_output(scale3)

   scale2_tmp = self.scale2_tmp(scale2_2) #ただの畳込み
   scale2_up = self.scale2_UpTmp(scale2_tmp)

   x = torch.cat([scale2, scale2_up],dim=1)
   x = self.scale2_ConvBlock(x)
   scale2_out = self.scale2_output(x)

   scale1_tmp = self.scale1_tmp(x) #ただの畳込み
   scale1_up = self.scale1_UpTmp(scale1_tmp)

   #print(scale1.shape)
   #print(scale1_up.shape)

   x = torch.cat([scale1, scale1_up], dim=1)
   x = self.scale1_ConvBlock(x)
   scale1_out = self.scale1_output(x)

   return scale3_out , scale2_out, scale1_out

model = YOLOv3()
with torch.no_grad():
  #第1引数はバッチサイズ 2:チャンネル数 3,4:画像の高さと幅
  output = model(torch.zeros((1,3,416,416)))
for i in range(3):
  print(output[i].shape)