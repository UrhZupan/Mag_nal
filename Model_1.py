# POENOSTAVLJEN UNet MODEL
# PREDLAGAL ChatGPT

import torch
import torch.nn as nn

class UNet1(nn.Module):
    def __init__(self):
        super(UNet1, self).__init__()
        
        # Za zacetek bomo dali 3 layerje
        # ENCODER
        self.encoder = nn.Sequential(
            
            nn.Conv2d(3, 64, 3, padding=1), # 1arg = input channels, 2 arg = output channels, 3 arg je kernel size
            nn.ReLU(inplace=True), 
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 1arg = kernel size (2x2), 2arg = stride (2)
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
            # DODAJAJ ALI ODVZEMAJ LAYERJE
        )
        
        # DECODER 
        self.decoder = nn.Sequential(
            
            nn.Conv2d(256 + 1, 512, 3, padding=1), # Tu skombiniramo depth image
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 2, stride=2), # brez skip connectiona (skočnih povezav)
            nn.ReLU(inplace=True), # Mora biti tu relu (?)  
            
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 2, stride=2),
            nn.ReLU(inplace= True),
            nn.ConvTranspose2d(128, 64, 2, stride=2), 
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1)  # Output depth image
        )
        
    def forward(self, rgb_image, depth_image_down):
        # Encoder
        x = self.encoder(rgb_image) # izvlecemo lastnosti (extract features) iz rgb slike
        
        # Upsample the down-sampled image
        depth_image_down = nn.functional.interpolate(depth_image_down, size=x.size()[2:], mode='bilinear', align_corners=False) # preveri size
        # interpoliramo tenzor, da bo enake velikosti kot rgb slika (size=x.size()[2:] --> x in y dimenziji slike, bilinearna interpolacija)
        
        # Concatenate depth image with encoder output (concatenate = 'združiti')
        #print("Tensor size after the encoder part: ", x.size())
        x = torch.cat([x, depth_image_down], dim=1) 
        #print("Tensor size after concatenating with 2nd input: ", x.size())
        
        # Decoder
        x = self.decoder(x)
        return x

# Initialize the model
#model = UNet1()
#rgb_input = torch.randn(1, 3, 561, 427)
#depth_input = torch.randn(1, 1, 10, 10)
#output1 = model.forward(rgb_input, depth_input)
#print("Output tensor size: ", output1.size())


# Ce bi hoteli uporabiti skip connection po vsakem layerju, spremenimo lahko forward metodo (kompleksnejsa metoda in bolj potratna)
"""
# Skip connectione delamo med vsakim layerjem

def forward(self, rgb_image, depth_image_down):
    # Encoder
    x1 = self.encoder[:4](rgb_image)  # First block: conv1, relu1, conv2, relu2
    x2 = self.encoder[4:9](x1)         # Second block: maxpool1, conv3, relu3, conv4, relu4
    x3 = self.encoder[9:](x2)          # Third block: maxpool2, conv5, relu5, conv6, relu6 (zadnjega maxpoola tu ne upostevamo)
    # Encoder zgleda vredu
    
    # Upsample depth image to match the spatial dimensions of x3
    depth_image_down = nn.functional.interpolate(depth_image_down, size=x3.size()[2:], mode='bilinear', align_corners=False)
    
    # Concatenate depth image with features from x3
    x3 = torch.cat([x3, depth_image], dim=1)
    
    # Decoder (skip connections BEFORE upconv - tako pravi ChatGPT)
    x = self.decoder[0:4](x3)          # First decoder block: conv7, relu7, conv8, relu8
    x = torch.cat([x, x2], dim=1)       # Skip connection 1
    x = self.decoder[4:9](x)           # Second decoder block: upconv1, relu9, conv9, relu9, conv10, relu10
    x = torch.cat([x, x1], dim=1)       # Skip connection 2
    x = self.decoder[9:](x)             # Third decoder block: upconv2, relu11, conv11
    return x
"""


# VPRASANJA
#   - Lahko "skip connection" po vsakem layerju manjka?
#   - ReLU po up-convolutionu (nn.ConvTranspose2d) JA/NE?
#   - Dodati/odvzeti layerje?
#   - Ali je kombinacija z down-samplean-o depth sliko vredu?