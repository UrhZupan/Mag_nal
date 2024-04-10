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
from Model_3 import UNet3 as UNet_model_3 # originalen ZMANJSAN UNet model
#from Model_2 import UNet2 as UNet_model_2 # originalen prirejen UNet model
#from Model_1 import UNet1 as UNet_model_1 # poenostavljen UNet model 

start = time.time()

# 0) PREP
# Device config 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper - parameters
num_epochs = 1
batch_size = 1
learning_rate = 0.001

# 1) DATASET

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
    transforms.Resize((384, 520)), # TRANSFORM TO THE SAME DIMENSIONS AS MODEL OUTPUT
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
train_rgb_loader = DataLoader(train_rgb_dataset, batch_size=4, shuffle=False) # change 'Batch Sizes' !
test_rgb_loader = DataLoader(test_rgb_dataset, batch_size=1, shuffle=False) 

train_depth_loader = DataLoader(train_depth_dataset, batch_size=4, shuffle=False) 
test_depth_loader = DataLoader(test_depth_dataset, batch_size=1, shuffle=False) 

train_depth_down_loader = DataLoader(train_depth_down_dataset, batch_size=4, shuffle=False)
test_depth_down_loader = DataLoader(test_depth_down_dataset, batch_size=1, shuffle=False) 

# Prikaz TESTING podatkov 
# GLEJ test_data.py


# 2) MODEL
model = UNet_model_3().to(device)


# 3) OPTIMIZER AND LOSS
criterion = nn.MSELoss() # Mean Squared Error
#criterion = nn.L1Loss() # Mean Absolute Error
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Adam
#optimizer = torch.optim.SGD() # SGD


# 4) TRAINING

if len(train_rgb_loader) == len(train_depth_loader) and len(train_depth_loader) == len(train_depth_down_loader):
    print("Proceed")
else:
    print("Error! - the lenght of DataLoaders isn't the same!")


n_total_steps = len(train_rgb_loader)
print("len(train_rgb_loader): ", n_total_steps)

for epoch in range(num_epochs):
    for i, (rgb, depth_down, depth_correct) in enumerate(zip(train_rgb_loader, train_depth_down_loader, train_depth_loader)):
        tries = 10
        
        # Just 5 tries
        if i > tries: 
            print("THE END ")
            break
        
        # rgb = input 1
        # depth_down = input 2
        # depth_correct = output used to compare with model output
        # i --> 43 
        rgb_input_1 = rgb.to(device) # 32 rgb slik oziroma 1 batch
        depth_down_input_2 = depth_down.to(device)
        depth_correct_output = depth_correct.to(device)
        #print(type(rgb_input_1))
        #print("RGB shape: ", rgb_input_1.shape, '\tDEPTH shape: ', depth_correct_output.shape, '\tDEPTH DOWN shape: ', depth_down_input_2.shape)
        
        # Forward pass
        outputs_predicted = model.forward(rgb_input_1, depth_down_input_2)
        loss = criterion(outputs_predicted, depth_correct_output)
        
        # Backward pass and optimizer
        optimizer.zero_grad() # empty the gradients
        loss.backward() # backpropagation
        optimizer.step() # update the parameters
        
        # Print the loss
        if (i+1) % 2 == 0: # CHANGE LATER
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')

        # Prikaz rezultatov v zadnjo
        if i == tries:
            # show the both outputs
            #fig, axes = plt.subplots(1, 2)
            print(outputs_predicted[1][0].shape, depth_correct_output[1][0].shape)
            # both shapes: torch.Size([4, 1, 384, 520])
            #axes[0].imshow(outputs_predicted[1][0], cmap='gray')
            #axes[0].set_title('Predicted')
            
            #axes[1].imshow(depth_correct_output[1][0], cmap='gray')
            #axes[1].set_title('Correct')
            
            #for ax in axes:
                #ax.axis('off')
            #plt.show()

# 5) TESTING
# To be continued

end = time.time()
print(f'{(end - start):.4f}, s')


# na strezniku lahko treniramo
# google collab
# hugging face --> online service

# ZMANJSEVANJE
# kvantizacija --> najbolj uporabna
# distilation
# federated learning
# techer student learning

# jan.pleterski@fs.uni-lj.si