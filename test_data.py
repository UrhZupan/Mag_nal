import numpy
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt

from Dataset_separate import CustomDatasetRGB, CustomDatasetDepth

# For grayscale:
# transforms.Normalize((0,), (255,)) # Normalize between 0 and 255
# transforms.Normalize((0.5,), (0.5,))  # Normalize between -1 and 1

transform_high_res_rgb = transforms.Compose([
    transforms.Resize((427, 561)),
    transforms.ToTensor() # convert to tensor
    #, transforms.Normalize( (0.5,0.5,0.5),(0.5,0.5,0.5) )
    #, transforms.Normalize((0,), (255,)) 
])

# FIX OUTPUT DIMENSIONS --> Output tensor size:  torch.Size([1, 1, 228, 372])
transform_high_res_depth = transforms.Compose([
    transforms.Resize((228, 372)), # TRANSFORM TO THE SAME DIMENSIONS AS MODEL OUTPUT
    transforms.ToTensor() # convert to tensor
    #, transforms.Normalize( (0.5,0.5,0.5),(0.5,0.5,0.5) )
    #, transforms.Normalize((0,), (255,)) 
])

transform_low_res_depth = transforms.Compose([
    transforms.Resize((10, 10)),
    transforms.ToTensor() # convert to tensor
    #, transforms.Normalize( (0.5,0.5,0.5),(0.5,0.5,0.5) )
    #, transforms.Normalize((0,), (255,)) 
])

# Directories
rgb_directory = r'D:\FAKS\MAGISTRSKA_NALOGA\Training_data\rgb_data' + '\\'
depth_directory = r'd:\FAKS\MAGISTRSKA_NALOGA\Training_data\depth_data' + '\\'
depth_down_directory = r'd:\FAKS\MAGISTRSKA_NALOGA\Training_data\depth_down_data' + '\\'

# Datasets
rgb_dataset = CustomDatasetRGB(rgb_directory, transform=transform_high_res_rgb) # (561 x 427) pix, 3 channels
depth_dataset = CustomDatasetDepth(depth_directory, transform=transform_high_res_depth) # (561 x 427) pix , 1 channel
depth_down_dataset = CustomDatasetDepth(depth_down_directory, transform=transform_low_res_depth) # (10 x 10) pix, 1 channel

# Split to 'Train' and 'Test'
train_rgb_dataset, test_rgb_dataset = train_test_split(rgb_dataset, test_size = 0.05, random_state=1) # test_size = 0.05 --> 5% slik gre za testiranje, random_state = 45 --> consistent splitting
train_depth_dataset, test_depth_dataset = train_test_split(depth_dataset, test_size = 0.05, random_state=1) 
train_depth_down_dataset, test_depth_down_dataset = train_test_split(depth_down_dataset, test_size = 0.05, random_state=1) 

# DataLoader
train_rgb_loader = DataLoader(train_rgb_dataset, batch_size=32, shuffle=False) # change 'Batch Sizes' !
test_rgb_loader = DataLoader(test_rgb_dataset, batch_size=1, shuffle=False) 

train_depth_loader = DataLoader(train_depth_dataset, batch_size=32, shuffle=False) 
test_depth_loader = DataLoader(test_depth_dataset, batch_size=1, shuffle=False) 

train_depth_down_loader = DataLoader(train_depth_down_dataset, batch_size=32, shuffle=False)
test_depth_down_loader = DataLoader(test_depth_down_dataset, batch_size=1, shuffle=False) 


# Prikaz TESTING podatkov 
examples_rgb = iter(train_rgb_loader)
examples_depth = iter(train_depth_loader)
examples_depth_down = iter(train_depth_down_loader)

samples_rgb = next(examples_rgb)
samples_depth = next(examples_depth)
samples_depth_down = next(examples_depth_down)

# SHAPES
print("RGB shape: ", samples_rgb.shape, '\tDEPTH shape: ', samples_depth.shape, '\tDEPTH DOWN shape: ', samples_depth_down.shape)
#print("\nRGB shape [0]: ", samples_rgb[0].shape, '\tDEPTH shape [0]: ', samples_depth[0].shape, '\tDEPTH DOWN shape [0]: ', samples_depth_down[0].shape)
#print("\nRGB shape [0][0]: ", samples_rgb[0][0].shape, '\tDEPTH shape [0][0]: ', samples_depth[0][0].shape, '\tDEPTH DOWN shape [0][0]: ', samples_depth_down[0][0].shape)

for i in range(3):
    plt.subplot(1, 3, i+1)
    # samples.shape = torch.Size([32, 3, 561, 427])
    if (i == 0):
        plt.imshow(samples_rgb[18][0]) # 17. slika v batchu, 1. channel
    if (i == 1):
        plt.imshow(samples_depth[18][0], cmap='gray') 
    if (i == 2):
        plt.imshow(samples_depth_down[18][0], cmap='gray')
        
plt.show()