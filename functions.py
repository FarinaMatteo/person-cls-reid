from typing import ItemsView
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.shape_base import block
import torch 
from torchvision import transforms, datasets
import torchvision
import pandas as pd
import os,glob
from torchvision.io import read_image
from pathlib import Path

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, img_dir, transform, target_transform=None):
        super(CustomImageDataset,self).__init__()
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        #print(self.img_labels.iloc[idx, 0])
        folder_path= str(str(self.img_dir) +"/" + str(self.img_labels.iloc[idx, 0]))
        image = []
        for filename in glob.glob(os.path.join(folder_path, '*.jpg')):
            with open(filename, 'r') as f:    
                image.append(read_image(filename))
                #break ##--> TO RETURN ONLY ONE IMAGE AND AVOID THE BATCH SIZE PROBLEM IN THE DATALOADER
        # label = {}
        # label['age'] = self.img_labels.iloc[idx, 1]
        # label['backpack'] = self.img_labels.iloc[idx, 2]
        # label['bag'] = self.img_labels.iloc[idx, 3]
        # label['handbag'] = self.img_labels.iloc[idx, 4]
        # label['clothes'] = self.img_labels.iloc[idx, 5]
        # label['down'] = self.img_labels.iloc[idx, 6]
        # label['up'] = self.img_labels.iloc[idx, 7]
        # label['hair'] = self.img_labels.iloc[idx, 8]
        # label['hat'] = self.img_labels.iloc[idx, 9]
        # label['gender'] = self.img_labels.iloc[idx, 10]
        # label['upblack'] = self.img_labels.iloc[idx, 11]
        # label['upwhite'] = self.img_labels.iloc[idx, 12]
        # label['upred'] = self.img_labels.iloc[idx, 13]
        # label['uppurple'] = self.img_labels.iloc[idx, 14]
        # label['upyellow'] = self.img_labels.iloc[idx, 15]
        # label['upgray'] = self.img_labels.iloc[idx, 16]
        # label['upblue'] = self.img_labels.iloc[idx, 17]
        # label['upgreen'] = self.img_labels.iloc[idx, 18]
        # label['downblack'] = self.img_labels.iloc[idx, 19]
        # label['downwhite'] = self.img_labels.iloc[idx, 20]
        # label['downpink'] = self.img_labels.iloc[idx, 21]
        # label['downpurple'] = self.img_labels.iloc[idx, 22]
        # label['downyellow'] = self.img_labels.iloc[idx, 23]
        # label['downgray'] = self.img_labels.iloc[idx, 24]
        # label['downblue'] = self.img_labels.iloc[idx, 25]
        # label['downgreen'] = self.img_labels.iloc[idx, 26]
        # label['downbrown'] = self.img_labels.iloc[idx, 27]


        label = []

        for i in range (1,28):
            label.append(self.img_labels.iloc[idx, i])
        
        if self.transform:
            image1 = image
            del image
            image = []
            #print("ci siamo")
            for item in image1:
                new = self.transform(item)#transforms.ToPILImage()(item).convert("RGB")
                image.append(new)
        if self.target_transform:
            label = self.target_transform(label)
       
        if(self.transform == None):
            #print("QUAAAA")
            sample = {"label": label, "image" : image}
        else:
            #print("QUIIIII")
            sample =  {"label": label, "image" : image}
        return sample


# END Functions -------------------------------------------------------------

def get_dataset(folder_images, folder_label, transformation):

    all_dataset = []
    train_dataset0 = CustomImageDataset(folder_label, folder_images, transform= None)
    all_dataset.append(train_dataset0)
    #Real lenght
    # k = 0
    # for i in range(len(train_dataset0)):  
    #     values  = train_dataset0[i]
    #     k = k + int(len(values['image']))
    # print("Final lenght --> " +str(k))

    ##FLIP
    if transformation > 0 :
        train_dataset1 = CustomImageDataset(folder_label, folder_images, transform= transforms.Compose([torchvision.transforms.RandomHorizontalFlip(p=1)]))
        transformation -= 1
        all_dataset.append(train_dataset1)

    ##RANDOM CROP
    if transformation > 0 :
        train_dataset2 = CustomImageDataset(folder_label, folder_images, transform=transforms.Compose([torchvision.transforms.RandomResizedCrop(size=(128,64),scale=(0.4,0.5),ratio=(0.75, 1.3333333333333333))]))
        transformation -= 1
        all_dataset.append(train_dataset2)

    #INFORMATION LOSS    
    if transformation > 0 :
        train_dataset3 = CustomImageDataset(folder_label, folder_images,transform=transforms.Compose([torchvision.transforms.RandomErasing(p=1, scale=(0.05, 0.13), ratio=(0.3, 3.3), value=0, inplace=False),torchvision.transforms.RandomErasing(p=1, scale=(0.05, 0.13), ratio=(0.3, 3.3), value=0, inplace=False)]))
        transformation -= 1
        all_dataset.append(train_dataset3)

    #ROTATION
    if transformation > 0 :
        train_dataset4 = CustomImageDataset(folder_label, folder_images,transform=transforms.Compose([torchvision.transforms.RandomRotation(15),torchvision.transforms.RandomResizedCrop(size=(128,64),scale=(0.2,0.7),ratio=(0.75, 1.3333333333333333))]))
        transformation -= 1
        all_dataset.append(train_dataset4)


    #CONTRAST --> to set    
    if transformation > 0 :
        train_dataset5 = CustomImageDataset(folder_label, folder_images,transform=transforms.Compose([torchvision.transforms.ColorJitter(contrast=3)]))
        transformation -= 1
        all_dataset.append(train_dataset5)

  
    ##Color shift --> to set
    if transformation > 0 :
        train_dataset6 = CustomImageDataset(folder_label, folder_images,transform=transforms.Compose([torchvision.transforms.ColorJitter(brightness=1.2, saturation=1.2)]))
        transformation -= 1
        all_dataset.append(train_dataset6)
   
    #NOISE
    ##missing

    increased_dataset = torch.utils.data.ConcatDataset(all_dataset)
    return increased_dataset

def print_images(set, transformation , train):
    k=0
    while k < len(set):
        values  = set[k]
        #print(len(values['image']))      
        #print(values['image'])
        for i in range(len(values['image'])):
            plt.imshow(np.transpose(values['image'][i].numpy(), (1, 2, 0)))
            plt.show()       
        if (train ==True):
            k = k + int((len(set) / transformation))
        else :
            if (transformation > 0):
                k = k + int((len(set) / transformation))
            else :
                k = k+1
        print("dataset" +str(k))
    
    plt.close('all')

def collate_fn(batch):
    #plot_batch(batch)
    images = [data['image'] for data in batch]
    labels = [data['label'] for data in batch]
    label = []
    for i in range(len(labels[0])):
        appo = []
        for item in labels:
            appo.append(int(item[i]))               
        appo = torch.Tensor(appo)
        label.append(appo) 
    image = []
    for item in images:
        image.append(item)
    return label,image

def plot_batch(set):
    k=0
    while k < len(set):
        values  = set[k]
        for i in range(len(values['image'])):
            plt.imshow(np.transpose(values['image'][i].numpy(), (1, 2, 0)))
            plt.show()       

        k = k +1    
    plt.close('all')