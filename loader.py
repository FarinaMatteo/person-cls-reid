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
    transformation = 3
    test = get_dataset(folder_images_test, folder_label_test, transformation)

    folder_label_val = "csv_files/train_label.csv"
    folder_images_val= "train_directory"
    transformation = 0
    val = get_dataset(folder_images_val, folder_label_val, transformation)

    #print_image(test)
    
    print("-------------")
    #for item in dataloader:
        #train_features, train_labels = next(iter(dataloader))

        #print(f"Labels batch shape: {train_labels.size()}")
        #print(f"Label: {label}")
   # for i_batch, sample_batched in enumerate(dataloader):
        #print(i_batch) 
    dataloader_test = torch.utils.data.DataLoader(test,batch_size=36, shuffle=True, collate_fn=collate_fn)#collate_fn=lambda x: x )
    dataloader_val = torch.utils.data.DataLoader(val,batch_size=25, shuffle=True, collate_fn=collate_fn)#collate_fn=lambda x: x )
   
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
        
    # x = 0
    # xx = 0
    # for data in dataloader:
    #     print(data)
    #     print("label --> " + str(data['label']))
    #     print(data['image'])
    #     for item in data['image']:
    #         print(item)
    #         xx = xx+1
    #     x = x+1 
    # print("TOT ID " + str(x))
    # print(xx)
    
    # for data in data_loader:
    #     print("Data: ", data['label'])
    #     print("Image" ,data['image'])         

    
'''     figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1,cols * rows + 1):
        values = train_dataset[i]
        print(len(values['image']))
        img_plot = values['image'][1]
        figure.add_subplot(rows, cols, i)
        plt.axis("off")
        plt.imshow(np.transpose(img_plot.numpy(), (1, 2, 0)))
    
    plt.show() '''
  
if __name__ == "__main__":
    main()