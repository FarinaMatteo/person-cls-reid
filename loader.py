import numpy as np
import matplotlib.pyplot as plt
from numpy.core.shape_base import block
import torch 
from torchvision import transforms, datasets
import torchvision
import pandas as pd
import os,glob
from torchvision.io import image, read_image
import PIL
from pathlib import Path
import collections
import random
from split import *
from functions import *

def main():
    
    split()
    folder_label_test = "csv_files/train_label.csv"
    folder_images_test= "train_directory"
    transformation = 6
    test = get_dataset(folder_images_test, folder_label_test, transformation)

    #print_images(test, transformation, train=True)

    folder_label_val = "csv_files/validation_label.csv"
    folder_images_val= "validation_directory"
    transformation = 2
    val = get_dataset(folder_images_val, folder_label_val, transformation)

    
    #print_images(val, transformation , train=False)

        
    print("-------------")

    dataloader_test = torch.utils.data.DataLoader(test,batch_size=2, shuffle=True, collate_fn=collate_fn)
    dataloader_val = torch.utils.data.DataLoader(val,batch_size=2, shuffle=True, collate_fn=collate_fn)
   
    print("-------------")

    for idx, (label, images) in enumerate(dataloader_test):
        print("idx:", idx)
        print("label:", label)
        print("images:", images)
        
    print("-------------")

    for idx, (label, images) in enumerate(dataloader_val):
        print("idx:", idx)
        print("label:", label)
        print("images:", images)
        
  
if __name__ == "__main__":
    main()