import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch 
from torchvision import transforms
import torchvision
import pandas as pd
import os
import glob
import shutil
from torchvision.io import read_image

def flatten_folder(folder, target=None):
    # append a trailing slash to the folder names if needed
    folder = folder if folder.endswith("/") else folder+"/"
    if not target: target = folder
    else: target = target if target.endswith("/") else target+"/"
    # retrieve the subtree of the current folder and start parsing it
    file_list = glob.glob(folder + "*")
    for f in file_list:
        # base case, directly copy the src file to the target location
        if os.path.isfile(f):
            shutil.move(src=f, dst=target+f"{os.path.basename(f)}")
        # more complicated scenario, start recursion keeping the same target
        elif os.path.isdir(f):
            flatten_folder(f, target)
            os.rmdir(f)

def group_folder(folder):
    folder = folder if folder.endswith("/") else folder+"/"
    file_list = glob.glob(folder + "*")
    ids = []
    for f in file_list:
        assert not os.path.isdir(f)
        cur_id = os.path.basename(f)[:4]
        # trim starting zeros
        while cur_id.startswith("0"):
            cur_id = cur_id[1:]
        # create new id directory if needed
        if cur_id not in ids:
            os.makedirs(os.path.join(folder, cur_id))
        
        ids.append(cur_id)
        # # move the current file in its directory group
        shutil.move(src=f, dst=os.path.join(folder, cur_id, os.path.basename(f)))

    

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, img_dir, transform, target_transform=None, batch_mode="id"):
        super(CustomImageDataset,self).__init__()
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.batch_mode = batch_mode
        if self.batch_mode == "img":
            self.img_list = os.listdir(self.img_dir)
        
    def __len__(self):
        if self.batch_mode == "id":
            return len(self.img_labels)
        elif self.batch_mode == "img":
            return len(self.img_list)

    def __getitem__(self, idx):
        # getitem based on :param idx: being someone's id. Thus, it should point to a folder containing
        # possibly more than one image.
        if self.batch_mode == "id":
            folder_path = str(str(self.img_dir) +"/" + str(self.img_labels.iloc[idx, 0]))
            image = []
            for filename in glob.glob(os.path.join(folder_path, '*.jpg')):
                with open(filename, 'r') as f:    
                    tmp_image = read_image(filename)
                    image.append(tmp_image/255)
                    # break ##--> TO RETURN ONLY ONE IMAGE AND AVOID THE BATCH SIZE PROBLEM IN THE DATALOADER

            # load annotation vector, excluding the id column
            label = []
            for i in range(1,len(self.img_labels.iloc[idx])):
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
        
        # otherwise, retrieve data by indexing images inside the provided folder
        else:
            # load the tensor image using cached files
            img_path = self.img_list[idx]
            img = read_image(os.path.join(self.img_dir, img_path))/255.0
            
            # load the annotations vector, excluding the id column
            person_id = os.path.basename(img_path)[:4]
            
            # shrink person id
            while person_id.startswith("0"): person_id = person_id[1:]
            person_id = int(person_id)
            
            # load the label based on the person id
            label = self.img_labels.loc[(self.img_labels["id"]==person_id)]
            label = label.to_numpy()
            label = torch.from_numpy(label).squeeze()[1:] - 1
            
            if self.transform:
                img = self.transform(img)
            
            if self.target_transform:
                label = self.target_transform(label)
            
            return img, label




# END Functions -------------------------------------------------------------

def get_dataset(folder_images, folder_label, transformations, batch_mode):

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
    return CustomImageDataset(folder_label, folder_images, transform=transforms.Compose(transforms_list), batch_mode=batch_mode)


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