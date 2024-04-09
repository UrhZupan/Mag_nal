import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image

# Device config 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DATASET
#   - 2 inputs: rgb (highres), depth (lowres - 10x10)
#   - 1 output: the output of our Model --> highres depth map
#   - the correct output, thate we use to compare with our model output

class CustomDatasetRGB(Dataset):
    def __init__(self, root_dir , transform=None):
        super(CustomDatasetRGB, self).__init__()
        self.root_dir = root_dir # directory of the rgb images
        self.images = os.listdir(root_dir) # List of files in the directory
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx): # kdaj in kako poklicemo to metodo (?)
        name = os.path.join(self.root_dir, self.images[idx]) # idx je index slike v mapi --> ustvarimo path za vsak image

        image = Image.open(name).convert("RGB") # for RGB images
                 
        if self.transform: # ali vrne PIL.Image objekt, ali pa torch.Tensor
            image = self.transform(image)
        
        return image

class CustomDatasetDepth(Dataset):
    def __init__(self, root_dir , transform=None):
        super(CustomDatasetDepth, self).__init__()
        self.root_dir = root_dir # directory of the rgb images
        self.images = os.listdir(root_dir) # List of files in the directory
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx): # kdaj in kako poklicemo to metodo (?)
        name = os.path.join(self.root_dir, self.images[idx]) # idx je index slike v mapi --> ustvarimo path za vsak image
        image = Image.open(name).convert("L") # for Depth images
            
        if self.transform: # ali vrne PIL.Image objekt, ali pa torch.Tensor
            image = self.transform(image)
        
        return image