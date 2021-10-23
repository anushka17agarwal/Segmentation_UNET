import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import BatchNorm2d
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv= nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            
            

        )
    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self, in_channels= 3, out_channel= 1, features= [64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.ups= nn.ModuleList()
        self.downs = nn.ModuleList() 
        self.pool = nn.MaxPool2d(kernel_size= 2, stride=2)    



        #downpart

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels= feature

        #upsampling
        for feature in reversed(features):


            #upsampling
            self.ups.append(nn.ConvTranspose2d(
                feature*2, feature, kernel_size= 2, stride= 2
            ))

            #Convolution Operation
            self.ups.append(DoubleConv(feature*2, feature))


        #bottom neck

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channel, 1)
    
    
    #Here, X is the input
    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x= self.pool(x)
        x= self.bottleneck(x)
        skip_connections = skip_connections[::-1 ]

        #iterating ups list by a step of= 2

        for ind in range(0, len(self.ups), 2):
            x= self.ups[ind] (x)
            skip_connection = skip_connections[ind//2]
            concat_skip= torch.cat((skip_connection, x), dim= 1)
            x= self.ups[ind+1] (concat_skip)

        return self.final_conv(x)

def test():
    x= torch.randn((3, 1, 160, 160))
    model = UNET(in_channels=1, out_channel=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)

if __name__ == "__main__":
    test()