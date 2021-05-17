import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch 
from torchvision import transforms
import torchvision
import pandas as pd
import os,glob
from torchvision.io import read_image

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
        folder_path = str(str(self.img_dir) +"/" + str(self.img_labels.iloc[idx, 0]))
        image = []
        for filename in glob.glob(os.path.join(folder_path, '*.jpg')):
            with open(filename, 'r') as f:    
                tmp_image = read_image(filename)
                image.append(tmp_image/255)
                # break ##--> TO RETURN ONLY ONE IMAGE AND AVOID THE BATCH SIZE PROBLEM IN THE DATALOADER

        label = []
        for i in range(1,28):
            label.append(self.img_labels.iloc[idx, i]-1)
        
        if self.transform:
            image_tmp = image
            del image
            image = []
            for item in image_tmp:
                new = self.transform(item) # transforms.ToPILImage()(item).convert("RGB")
                image.append(new)
        
        if self.target_transform:
            label = self.target_transform(label)
    
        sample = {"label": label, "image" : image}
        return sample


# END Functions -------------------------------------------------------------

def get_dataset(folder_images, folder_label, transformations):

    # initialize list of transformations to apply
    transforms_list = []
    
    # FLIP
    if "flip" in transformations:
        transforms_list.append(torchvision.transforms.RandomHorizontalFlip())

    # INFORMATION LOSS    
    if "erasing" in transformations:
        transforms_list.append(torchvision.transforms.RandomErasing(p=0.1, scale=(0.05, 0.13), ratio=(0.3, 3.3), value=0, inplace=False))
        transforms_list.append(torchvision.transforms.RandomErasing(p=0.1, scale=(0.05, 0.13), ratio=(0.3, 3.3), value=0, inplace=False))

    # ROTATION
    if "rotation" in transformations:
        transforms_list.append(torchvision.transforms.RandomRotation(15))
  
    # COLOR JITTER
    if "color_jitter" in transformations:
        transforms_list.append(torchvision.transforms.ColorJitter(contrast=1.2, brightness=1.2, saturation=1.2))

    # add mandatory normalization transform
    transforms_list.append(torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    # NOISE
    # missing
   
    # generate the dataset and return it 
    return CustomImageDataset(folder_label, folder_images, transform=transforms.Compose(transforms_list))


def print_images(set, transformation , train):
    k = 0
    while k < len(set):
        values = set[k]
        # print(len(values['image']))      
        # print(values['image'])
        for i in range(len(values['image'])):
            plt.imshow(np.transpose(values['image'][i].numpy(), (1, 2, 0)))
            plt.show()       
        if train:
            k = k + int((len(set) / transformation))
        else:
            if (transformation > 0):
                k = k + int((len(set) / transformation))
            else:
                k = k+1
        print("dataset" +str(k))
    
    plt.close('all')


def collate_fn(batch, plot=False):
    if plot:
        plot_batch(batch)
    
    # list of lists
    images = [data['image'] for data in batch]
    # flat list
    labels = [data['label'] for data in batch]

    ret_images = []
    ret_labels = []
    for i, image_list in enumerate(images):
        # list of images for each id
        for image in image_list:
            if len(ret_images) == 0:
                ret_images.append(image.numpy())
                ret_images = np.array(ret_images)
            else:
                ret_images = np.concatenate((ret_images, image.unsqueeze(dim=0).numpy()))
            ret_labels.append(labels[i])

    ret_images = torch.from_numpy(ret_images)
    ret_labels = torch.from_numpy(np.array(ret_labels))
    return ret_images, ret_labels


def plot_batch(set):
    k=0
    while k < len(set):
        values = set[k]
        print(f"This batch item has {len(values['image'])} images")
        for i in range(len(values['image'])):
            cv2.imshow(f"Image {i} of batch item {k}", np.transpose(values['image'][i].numpy(), (1, 2, 0)))
            cv2.waitKey(0)
            # plt.imshow(np.transpose(values['image'][i].numpy(), (1, 2, 0)))
            # plt.show()       

        k = k+1    
    plt.close('all')
    cv2.destroyAllWindows()
