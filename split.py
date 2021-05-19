
'''
Code which splits the Image-Dataset inyo train and validation sets.
The percentage of train and validation can be changed directly from the node thanks to the test and val GLOBAL variable.
The choice of which ID goes to one set or another is completely randomic due to the shuffle list.

'''

import os
import glob
import random
import shutil
import re
import pandas as pd
from multicolor import *

# DEFINE GLOBAL VARIABLES
def split(train_split=0.8, max_images=None):
    folder_path = 'dataset/train'

    # DELETE FOLDERS IF AVAILABLE
    try:
        shutil.rmtree('train_directory/')
    except FileNotFoundError as exc:
        print("No train directory found on your system, creating one.")

    try:
        shutil.rmtree('validation_directory/')
    except FileNotFoundError as exc:
        print("No validation directory found on your system, creating one.")

    # INITIALIZE NEW FOLDERS (for validation and train set)
    try:
        os.mkdir('train_directory/')
    except FileExistsError as exc:
        print(exc)

    try:
        os.mkdir('validation_directory/')
    except FileExistsError as exc:
        print(exc)

    # useless --> to remove 
    i=0

    random_list = []
    df = pd.DataFrame(columns=['id','name'])

    # # for each image inside the specified folder save the respective ID and NAME of the image eg: 0009_c4_0963839.jpg
    for filename in glob.glob(os.path.join(folder_path, '*.jpg')):
        with open(filename, 'r') as f:
            row = {}

            # keeping only the image name without the path
            filename = str(filename.replace("dataset/train/",""))
            row["name"]= filename

            # Keeping only the ID
            filename = re.sub(r'_.*',"",filename)

            # Add the id to the ID's list
            random_list.append(filename)
            
            row["id"] = filename
            
            # save each couple ID- NAME(image) into a DataFrame
            df = df.append(row, ignore_index=True)
        
        i += 1
        if max_images and i==max_images:
            print("Reached the {} limit, no further images will be placed in the dataset folder".format(max_images))
            break

    # Group each image based on the ID value
    df = df.groupby('id')['name'].apply(list).reset_index()

    # save the csv with the link between ID and Name
    df.to_csv('csv_files/file.csv', index=False)

    # Remove duplicate ID and shuffle the ID's list
    random_list = list(set(random_list))
    random.shuffle(random_list)

    # Counter to be sure we use ALL the images inside the Folder selected
    counter = 0

    # I select the test% of ID to put inside the train folder
    # ATTENTION!!! --> We are NOT splitting test% val% our images ---> we select test% of IDS INDEPENDETLY of how many images are related to each ID

    df1 = pd.DataFrame(columns=['id'])

    for i in range (0,int(len(random_list)*train_split)):
        
        # Looking for the row of the DataFrame which contains the selected ID
        l = df[df['id'] == random_list[i]].to_dict('records')[0]
        
        # # Add to a list of ID selected for the train set
        row = {}
        row['id']= int(random_list[i])
        df1 = df1.append(row, ignore_index=True)    

        # Create a new folder inside train_directory in order to put all the images related to the ID
        try:
            os.mkdir('train_directory/'+ str(int(random_list[i])))
        except FileExistsError as exc:
            print(exc)

        # # Adding ALL ID's images to its respective folder
        for key,value in l.items():
            if (key=='name'):
                l2 = list(value)
                for item in l2:                
                    new_item = folder_path+ "/"+item
                    try:
                        shutil.copy(new_item, 'train_directory/'+ str(int(random_list[i])))
                    except FileExistsError or FileNotFoundError as exc:
                        print(exc)
        counter = counter +1



    # create and save a new annotations.csv which is the result of the intersection between the Train IDs and the original annotations file
    # df1.to_csv('csv_files/train.csv', index=False)
    # df2 = pd.read_csv('csv_files/train.csv')

    df2 = pd.read_csv('dataset/annotations_train.csv')
    int_df = pd.merge(df2, df1, how='inner', on=['id'])
    int_df = multi_color(int_df)
    int_df.to_csv('csv_files/train_label.csv',index=False)


    # THE SAME IS REPEATED ALSO FOR THE VALIDATION
    df1 = pd.DataFrame()
    for i in range(counter,len(random_list)):
        l = df[df['id'] == random_list[i]].to_dict('records')[0]
        try:
            os.mkdir('validation_directory/'+ str(int(random_list[i])))
        except FileExistsError as exc:
            print(exc)
        row = {}
        row['id']= int(random_list[i])
        df1 = df1.append(row, ignore_index=True)    
        for key,value in l.items():
            if (key=='name'):
                l2 = list(value)
                for item in l2:
                    item = folder_path+ "/"+item
                    try:
                        shutil.copy(item, 'validation_directory/'+str(int(random_list[i])))
                    except FileExistsError or FileNotFoundError as exc:
                        print(exc)
        counter = counter +1

    df2 = pd.read_csv('dataset/annotations_train.csv')
    int_df = pd.merge(df2, df1, how ='inner', on =['id'])
    int_df = multi_color(int_df)
    int_df.to_csv('csv_files/validation_label.csv',index=False)


