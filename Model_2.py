# PRILAGOJEN Unet (podoben kot iz članka)
# RAZLIKE:
#   - 2 vhoda, 1 izhod (v "bottlenecku" združimo output iz enkoderja in nizko-resolucijsko globinsko sliko)
#   - Output ima 1 channel

import torch
import torch.nn as nn

def double_conv(in_ch, out_ch): # input channels, output channels
    conv = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3),
        nn.ReLU(inplace=True), # za bolj efficeint memory usage
        nn.Conv2d(out_ch, out_ch, kernel_size=3),
        nn.ReLU(inplace=True)
    )
    return conv
    

def crop_img(tensor, target_tensor):
    target_size_x, target_size_y = target_tensor.size()[2:]  # Get target size in x and y dimensions
    tensor_size_x, tensor_size_y = tensor.size()[2:]  # Get tensor size in x and y dimensions
    
    # predpostavka: Target size < Tensor size
    delta_x = tensor_size_x - target_size_x
    delta_y = tensor_size_y - target_size_y
    
    delta_x_start = delta_x // 2
    delta_x_end = delta_x - delta_x_start
    
    delta_y_start = delta_y // 2
    delta_y_end = delta_y - delta_y_start
    
    return tensor[:, :, delta_x_start:(tensor_size_x - delta_x_end), delta_y_start:(tensor_size_y - delta_y_end)]

class UNet2(nn.Module):
    def __init__(self):
        super(UNet2, self).__init__()
        
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(3, 64)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128, 256)
        self.down_conv_4 = double_conv(256, 512)
        self.down_conv_5 = double_conv(512 + 1, 1024) # ( + 1) --> DRUGI INPUT
        
        self.up_trans_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_conv_1 = double_conv(1024, 512)
        self.up_trans_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_conv_2 = double_conv(512, 256)
        self.up_trans_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_conv_3 = double_conv(256, 128)
        self.up_trans_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up_conv_4 = double_conv(128, 64)
        
        # Zadnja konvolucija
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1) # PRI OUTPUTU RABIMO 1 KANAL
        
    def forward(self, rgb_image, depth_image_down): 
        
        # Encoder
        x1 = self.down_conv_1(rgb_image) #
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv_2(x2) #
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4) # 
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv_4(x6) #
        x8 = self.max_pool_2x2(x7)
        
        ################# 2nd INPUT #################
        d_size = x8.size()[2:]
        # depth_image_down je grayscale depth map z resolucijo npr. 10x10
        depth_image_down = nn.functional.interpolate(depth_image_down, size=d_size, mode='bilinear', align_corners=False)
        x0 = torch.cat([x8, depth_image_down], dim=1)
        #############################################
        
        x9 = self.down_conv_5(x0) 
        
        
        # Decoder
        x = self.up_trans_1(x9)
        y = crop_img(x7, x)
        x = self.up_conv_1(torch.cat([x, y], dim=1)) # concatenate with x7 (x7 cropamo na enako resolucijo)
        
        x = self.up_trans_2(x)
        y = crop_img(x5, x)
        x = self.up_conv_2(torch.cat([x, y], dim=1))
        
        x = self.up_trans_3(x)
        y = crop_img(x3, x)
        x = self.up_conv_3(torch.cat([x, y], dim=1))
        
        x = self.up_trans_4(x)
        y = crop_img(x1, x)
        x = self.up_conv_4(torch.cat([x, y], dim=1))
        
        x = self.out_conv(x)
        
        #print(x.size())
        return x

#rgb_input = torch.randn(1, 3, 106*2, 140*2)
#depth_input = torch.randn(1, 1, 10, 10)
#model = UNet2().to(device='cuda' if torch.cuda.is_available() else 'cpu')
#output1 = model.forward(rgb_input, depth_input)
#print("Output tensor size: ", output1.size())

"""NALETIMO NA TEZAVO, KER JE SLIKA PREVEC POMANJSANA! INPUT SLIKA NAJ IMA VECJE X IN Y DIMENZIJE IN ZATO TRENIRAJ NA MANJSEM SAMPLE SIZEU (NE 1500)!!"""