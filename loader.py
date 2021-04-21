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

from torchvision.transforms.transforms import Normalize, RandomSizedCrop, Scale

#Functions -------------------------------------------------------------

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
        print(self.img_labels.iloc[idx, 0])
        folder_path= str(str(self.img_dir) +"/" + str(self.img_labels.iloc[idx, 0]))
        image = []
        for filename in glob.glob(os.path.join(folder_path, '*.jpg')):
            with open(filename, 'r') as f:    
                image.append(read_image(filename))
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
        image1 = []

        for i in range (1,28):
            label.append(self.img_labels.iloc[idx, i])
        
        if self.transform:
            #print("ci siamo")
            for item in image:
                new = self.transform(item)#transforms.ToPILImage()(item).convert("RGB")
                image1.append(new)
        if self.target_transform:
            label = self.target_transform(label)
       
        if(self.transform == None):
            #print("QUAAAA")
            sample = {"label": label, "image" : image}
        else:
            sample =  {"label": label, "image" : image1}
        return sample


# END Functions -------------------------------------------------------------

def main():
    
    #custom
    train_dataset0 = CustomImageDataset("csv_files/train_label.csv", "train_directory", transform= None)

    #Real lenght
    # k = 0
    # for i in range(len(train_dataset0)):  
    #     values  = train_dataset0[i]
    #     k = k + int(len(values['image']))
    # print("Final lenght --> " +str(k))

    ##FLIP
    #train_dataset1 = CustomImageDataset("csv_files/train_label.csv", "train_directory", transform= transforms.Compose([torchvision.transforms.RandomHorizontalFlip(p=1)]))
    
    ##RANDOM CROP
    #train_dataset2 = CustomImageDataset("csv_files/train_label.csv", "train_directory", transform=transforms.Compose([torchvision.transforms.RandomSizedCrop(size=(128,64),scale=(0.4,0.5),ratio=(0.75, 1.3333333333333333))]))

    #INFORMATION LOSS
    train_dataset3 = CustomImageDataset("csv_files/train_label.csv", "train_directory",transform=transforms.Compose([torchvision.transforms.RandomErasing(p=1, scale=(0.05, 0.13), ratio=(0.3, 3.3), value=0, inplace=False),torchvision.transforms.RandomErasing(p=1, scale=(0.05, 0.13), ratio=(0.3, 3.3), value=0, inplace=False)]))

    #ROTATION
    #train_dataset4 = CustomImageDataset("csv_files/train_label.csv", "train_directory",transform=transforms.Compose([torchvision.transforms.RandomRotation(35),torchvision.transforms.RandomSizedCrop(size=(128,64),scale=(0.2,0.7),ratio=(0.75, 1.3333333333333333))]))

    #CONTRAST --> to set
    #train_dataset5 = CustomImageDataset("csv_files/train_label.csv", "train_directory",transform=transforms.Compose([torchvision.transforms.ColorJitter(contrast=5)]))
    
    ##Color shift --> to set
    #train_dataset6 = CustomImageDataset("csv_files/train_label.csv", "train_directory",transform=transforms.Compose([torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=1, hue=0.5)]))
    
    #NOISE
    ##missing

    increased_dataset = torch.utils.data.ConcatDataset([train_dataset0,train_dataset3])

    # k=0
    # while k < (len(increased_dataset)):
    #     #print(values['label'])
    #     values  = increased_dataset[k]
    #     print(len(values['image']))      
    #     print(values['image'])
    #     for i in range(len(values['image'])):
    #         plt.imshow(np.transpose(values['image'][i].numpy(), (1, 2, 0)))
    
    #         plt.show()       
    #         break     
    #     k = k + 20
    #     print("dataset" +str(k))
    
    data_loader = torch.utils.data.DataLoader(increased_dataset,batch_size=1,shuffle=True)
    x = 0
    xx = 0
    for data in data_loader:
        print(data['label'])
        print(data['image'])
        for item in data['image']:
            xx = xx+1
        x = x+1 
    print("TOT ID " + str(x))
    print(xx)
    
    # for data in data_loader:
    #     print("Data: ", data['label'])
    #     print("Image" ,data['image'])         


    #dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # figure = plt.figure(figsize=(8, 8))
    # cols, rows = 3, 3
    # for i in range(1,cols * rows + 1):
    #     values = train_dataset[i]
    #     print(len(values['image']))
    #     img_plot = values['image'][1]
    #     figure.add_subplot(rows, cols, i)
    #     plt.axis("off")
    #     plt.imshow(np.transpose(img_plot.numpy(), (1, 2, 0)))
    
    # plt.show()
  





  # train_transforms = transforms.Compose([transforms.RandomRotation(30),
  #                                       transforms.RandomResizedCrop(224),
  #                                       transforms.RandomHorizontalFlip(),
  #                                       transforms.ToTensor(),
  #                                       transforms.Normalize([0.5, 0.5, 0.5], 
  #                                                             [0.5, 0.5, 0.5])])
  
  # train_dataset = torchvision.datasets.ImageFolder('train_directory/', transform=train_transforms)
  # #validation_dataset = torchvision.datasets.ImageFolder('validation_directory/', transform=train_transforms)

  # list_train = []
  # df = pd.read_csv('train.csv')
  # list_train = df['id']
  # for item in list_train:
  #       print(item)

  # df1 = pd.read_csv('dataset/annotations_train.csv')
  # print("---------")
  # int_df = pd.merge(df, df1, how ='inner', on =['id'])
  # print("---------")
  # print(int_df)
  # int_df.to_csv('train_boh.csv',index=False)
  
  # dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
  # data_iter = iter(dataloader_train)
  # #dataloader_val= torch.utils.data.DataLoader(validation_dataset, batch_size=32, shuffle=True)

  
  # images, labels = next(data_iter)
  # #fig, axes = plt.subplots(figsize=(10,4), ncols=4)
  # # for ii in range(4):
  # #     ax = axes[ii]
  # # #     helper.imshow(images[ii], ax=ax, normalize=False)
  # #     plt.imshow(np.transpose(images[ii].numpy(), (1, 2, 0)))
  # #     print(ii)

  # # plt.show()
  # w=10
  # h=10
  # fig=plt.figure(figsize=(8, 8))
  # columns = 4
  # rows = 5
  # for i in range(1, columns*rows +1):
  #     fig.add_subplot(rows, columns, i)
  #     plt.imshow(np.transpose(images[i].numpy(), (1, 2, 0)))
  # plt.show()

  # plt.savefig('prova.png')

if __name__ == "__main__":
    main()