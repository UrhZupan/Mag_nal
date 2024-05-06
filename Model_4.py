# Same as Model_3.py
# Less layers

import torch
import torch.nn as nn

def double_conv(in_ch, out_ch):
    conv = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3),
        nn.ReLU(inplace=True)
    )
    return conv

def crop_img(tensor, target_tensor):
    target_size_x, target_size_y = target_tensor.size()[2:]
    tensor_size_x, tensor_size_y = tensor.size()[2:]
    
    # Assumption: Target_size < Tensor_size
    delta_x = tensor_size_x - target_size_x
    delta_y = tensor_size_y - target_size_y
    
    delta_x_start = delta_x // 2
    delta_x_end = delta_x - delta_x_start
    
    delta_y_start = delta_y // 2
    delta_y_end = delta_y - delta_y_start
    
    return tensor[:, :, delta_x_start:(tensor_size_x - delta_x_end), delta_y_start:(tensor_size_y - delta_y_end)]


class UNet4(nn.Module):
    def __init__(self):
        super(UNet4, self).__init__()
        
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(3, 64) # LATER CHANGE TO 1 -> GRAYSCALE
        self.down_conv_2 = double_conv(64 + 1, 128) # + 1 --> 2ND INPUT
        
        self.up_trans_1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up_conv_1 = double_conv(128, 64)
        
        # Last convolution
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)
        
    def forward(self, rgb_image, depth_image_down): 
        
        # Encoder
        x1 = self.down_conv_1(rgb_image) #
        x2 = self.max_pool_2x2(x1)
        
        ################# 2nd INPUT #################
        # Bottleneck
        d_size = x2.size()[2:]
        # depth_image_down je grayscale depth map z resolucijo npr. 10x10
        depth_image_down = nn.functional.interpolate(depth_image_down, size=d_size, mode='bilinear', align_corners=False)
        x0 = torch.cat([x2, depth_image_down], dim=1)
        #print(x0.size())
        #############################################
        
        x3 = self.down_conv_2(x0)
        
        # Decoder
        x = self.up_trans_1(x3)
        y = crop_img(x1, x)
        x = self.up_conv_1(torch.cat([x, y], dim=1))
        
        x = self.out_conv(x)
        
        return x


#rgb_input = torch.randn(1, 3, 427, 561)
#rgb_input = torch.randn(1, 3, 572, 572)
#rgb_input = torch.randn(1, 3, 106, 140)
#depth_input = torch.randn(1, 1, 10, 10)
#model = UNet4().to(device='cuda' if torch.cuda.is_available() else 'cpu')
#output1 = model.forward(rgb_input, depth_input)
#print("Output tensor size: ", output1.size())

#"""output size in pix: 90 x 124"""