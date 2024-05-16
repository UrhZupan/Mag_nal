import numpy as np
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
from Model_4 import UNet4 as UNet_model_4 # SE BOLJ ZMANJSAN UNet model
#from Model_3 import UNet3 as UNet_model_3 # originalen ZMANJSAN UNet model
#from Model_2 import UNet2 as UNet_model_2 # originalen prirejen UNet model
#from Model_1 import UNet1 as UNet_model_1 # poenostavljen UNet model 

start = time.time()

# 0) PREP
# Device config 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper - parameters
num_epochs = 1
batch_size_train = 4
learning_rate = 0.001

# 1) DATASET

print("\nDATA PREPARATION PHASE")
# For grayscale:
# transforms.Normalize((0,), (255,)) # Normalize between 0 and 255
# transforms.Normalize((0.5,), (0.5,))  # Normalize between -1 and 1 [? transforms.Normalize( (0.5,0.5,0.5),(0.5,0.5,0.5) )]

transform_high_res_rgb = transforms.Compose([
    transforms.Resize((106, 140)),
    transforms.ToTensor() # convert to tensor
    #, transforms.Normalize( (0.5,0.5,0.5),(0.5,0.5,0.5) )
    #, transforms.Normalize((0,), (255,)) 
])

# FIX OUTPUT DIMENSIONS 
# Model_3: Input tensor size: torch.Size([1, 3, 427, 561]) --> Output tensor size:  torch.Size([1, 1, 228, 372])
# Model_3: Input tensor size: torch.Size([1, 3, 106, 140]) --> Output tensor size:  torch.Size([1, 1, 64, 100])
transform_high_res_depth = transforms.Compose([
    transforms.Resize((90, 124)), # TRANSFORM TO THE SAME DIMENSIONS AS MODEL OUTPUT
    transforms.ToTensor() # convert to tensor
    #, transforms.Normalize((0.5,), (0.5,))
    #, transforms.Normalize((0,), (255,)) 
])

transform_low_res_depth = transforms.Compose([
    transforms.Resize((10, 10)),
    transforms.ToTensor() # convert to tensor
    #, transforms.Normalize((0.5,), (0.5,))
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
train_rgb_loader = DataLoader(train_rgb_dataset, batch_size=batch_size_train, shuffle=False) # change 'Batch Sizes' !
test_rgb_loader = DataLoader(test_rgb_dataset, batch_size=1, shuffle=False) 

train_depth_loader = DataLoader(train_depth_dataset, batch_size=batch_size_train, shuffle=False) 
test_depth_loader = DataLoader(test_depth_dataset, batch_size=1, shuffle=False) 

train_depth_down_loader = DataLoader(train_depth_down_dataset, batch_size=batch_size_train, shuffle=False)
test_depth_down_loader = DataLoader(test_depth_down_dataset, batch_size=1, shuffle=False) 

# Prikaz TESTING podatkov 
# GLEJ test_data.py

end_data_prep = time.time()
print(f'DATA PREPARATION duration: {(end_data_prep - start):.4f}, s')

# 2) MODEL
model = UNet_model_4().to(device)


# 3) OPTIMIZER AND LOSS
criterion = nn.MSELoss() # Mean Squared Error
#criterion = nn.L1Loss() # Mean Absolute Error
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Adam
#optimizer = torch.optim.SGD() # SGD


# 4) TRAINING

print("\nTRAINING PHASE")
if len(train_rgb_loader) == len(train_depth_loader) and len(train_depth_loader) == len(train_depth_down_loader):
    print("Proceed")
else:
    print("Error! - the lenght of DataLoaders isn't the same!")


def showResult(predicted, correct):
    # show the both outputs
    fig, axes = plt.subplots(1, 2)
    print(predicted[0][0].shape, correct[0][0].shape)
    
    axes[0].imshow(predicted[0][0], cmap='gray')
    axes[0].set_title('Predicted')
            
    axes[1].imshow(correct[0][0], cmap='gray')
    axes[1].set_title('Correct')
            
    for ax in axes:
        ax.axis('off')
    plt.show()



n_total_steps = len(train_rgb_loader)
print("len(train_rgb_loader): ", n_total_steps)
loss_plot = []
for epoch in range(num_epochs):
    for i, (rgb, depth_down, depth_correct) in enumerate(zip(train_rgb_loader, train_depth_down_loader, train_depth_loader)):
        tries = n_total_steps - 1
        
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
        loss_plot.append(loss.item()) # add losses to a list
        
        # Print the loss
        if (i+1) % 50 == 0: # CHANGE LATER
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')

        # Prikaz rezultatov v zadnjo
        if i == tries - 1:
            with torch.no_grad():
                # show the both outputs
                showResult(outputs_predicted, depth_correct_output )
                print("THE END OF TRAINING")
                break

end_training = time.time()
print(f'TRAINING duration: {(end_training - end_data_prep):.4f}, s')

plt.plot(loss_plot)
plt.ylabel("LOSS")
plt.xlabel("i")
plt.grid()
plt.show()

# 5) TESTING

print("\nTESTING PHASE")

def similarity_percentage(tensor1, tensor2, threshold=0):
    # Calculate absolute difference between corresponding pixels
    pixel_diff = torch.abs(tensor1 - tensor2)
    
    # Calculate percentage of pixels that meet the threshold
    num_pixels = tensor1.numel()
    num_similar_pixels = torch.sum(pixel_diff <= threshold)
    similarity_percentage = (num_similar_pixels / num_pixels) * 100
    
    return similarity_percentage.item()

accuracy_1 = []
accuracy_3 = []
accuracy_5 = []
with torch.no_grad():
    for i, (rgb_T, depth_down_T, depth_correct_T) in enumerate(zip(test_rgb_loader, test_depth_down_loader, test_depth_loader)):
        rgb_T = rgb_T.to(device)
        depth_down_T = depth_down_T.to(device)
        depth_correct_T = depth_correct_T.to(device)
        
        # Make forward pass WITHOUT CALCULATING THE GRADIENTS AND WITHOUT BACKWARD PASS
        depth_predicted_T = model.forward(rgb_T, depth_down_T)
        #if i==0:
            #print("PREDICTED shape: ", depth_predicted_T.shape, "\tCORRECT shape: ", depth_correct_T.shape)
        
        
        # Show some results
        #if i == 0:
            #print(depth_correct_T[0, 0, :2, :2])
            #print(depth_predicted_T[0, 0, :2, :2])
        
        # Calculate accuracy
        similarity_1 = similarity_percentage(depth_correct_T, depth_predicted_T, threshold=0.01) # 1% error allowed
        accuracy_1.append(similarity_1)
        similarity_3 = similarity_percentage(depth_correct_T, depth_predicted_T, threshold=0.03) # 3% error allowed
        accuracy_3.append(similarity_3)
        similarity_5 = similarity_percentage(depth_correct_T, depth_predicted_T, threshold=0.05) # 5% error allowed
        accuracy_5.append(similarity_5)


# Print accuracy
cnn_accuracy_1 = sum(accuracy_1) / len(accuracy_1)
cnn_accuracy_3 = sum(accuracy_3) / len(accuracy_3)
cnn_accuracy_5 = sum(accuracy_5) / len(accuracy_5)

print(f"Accuracy (1% error): {cnn_accuracy_1:.1f}%")
print(f"Accuracy (3% error): {cnn_accuracy_3:.1f}%")
print(f"Accuracy (5% error): {cnn_accuracy_5:.1f}%")
    
# Save the model under a name


end_testing = time.time()
print(f'TESTING duration: {(end_testing - end_training):.4f}, s')

end = time.time()
print(f'\nDURATION of whole process: {(end - start):.4f}, s')
print(f'BATCH SIZE: {batch_size_train}, LEARNING RATE: {learning_rate}')


# na strezniku lahko treniramo
# google collab
# hugging face --> online service

# ZMANJSEVANJE
# kvantizacija --> najbolj uporabna
# distilation
# federated learning
# techer student learning

# jan.pleterski@fs.uni-lj.si