from os import fchmod, read
import numpy as np
import numpy as np
from numpy.core.fromnumeric import shape
import torch
from kmeans_pytorch import kmeans
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
from functions import flatten_folder
import os
import re
import shutil
import glob
import random
from models.custom import DeepAttentionClassifier
from torchvision.io import read_image

val_dir = "validation_directory"
reid_queries_dir = "reid_queries"
reid_val_dir = "reid_validation"
flatten_folder(val_dir)


try:
    shutil.rmtree(reid_val_dir)
except FileNotFoundError as exc:
    print("No train directory found on your system, creating one.")

try:
    shutil.rmtree(reid_queries_dir)
except FileNotFoundError as exc:
    print("No validation directory found on your system, creating one.")

# INITIALIZE NEW FOLDERS (for validation and train set)
try:
    os.mkdir(reid_val_dir)
except FileExistsError as exc:
    print(exc)

try:
    os.mkdir(reid_queries_dir)
except FileExistsError as exc:
    print(exc)

file_list = glob.glob(val_dir + "/*.jpg")
random.shuffle(file_list)

for i in range(int(len(file_list)*0.75)):
    filepath = file_list[i]
    dst_path = os.path.join(reid_val_dir, os.path.basename(filepath))
    shutil.copy(filepath, dst_path)

for i in range(int(len(file_list)*0.75)+1, len(file_list)):
    filepath = file_list[i]
    dst_path = os.path.join(reid_queries_dir, os.path.basename(filepath))
    shutil.copy(filepath, dst_path)

queries = glob.glob(reid_queries_dir + "/*.jpg")
reid_images = glob.glob(reid_val_dir + "/*.jpg")
print(reid_images)
reid_identities = len(set([os.path.basename(filename).split("_")[0] for filename in reid_images]))
print("Number of identities:", reid_identities)

model = DeepAttentionClassifier(pretrained=False)
weights_path = "networks/model.pth"
# load the model
if torch.cuda.is_available():
    model.load_state_dict(torch.load(weights_path))
    model = model.to("cuda:0")
else:
    model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))
model.eval()

X = []
with torch.no_grad():
    # for each image produce the output vector
    for i, image_path in enumerate(reid_images):
        img_tensor = read_image(image_path)/255
        img_tensor = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])(img_tensor)
        img_tensor = img_tensor.unsqueeze(dim=0)
        feature_vec = model.encode(img_tensor).tolist()
        X.append(feature_vec)

print(len(X), len(X[0]))

# # # for each image inside the specified folder save the respective ID and NAME of the image eg: 0009_c4_0963839.jpg
# for filename in glob.glob("/Users/elia/Desktop/person-cls-reid/validation_directory/*.jpg"):
#     with open(filename, 'r') as f:
#         row = {}

#         # keeping only the image name without the path
#         filename = str(filename.replace("dataset/train/",""))
#         row["name"]= filename

#         # Keeping only the ID
#         filename = re.sub(r'_.*',"",filename)

#         # Add the id to the ID's list
#         random_list.append(filename)
        
#         row["id"] = filename
        
#         # save each couple ID- NAME(image) into a DataFrame
#         df = df.append(row, ignore_index=True)
    
#     i += 1
#     if max_images and i==max_images:
#         print("Reached the {} limit, no further images will be placed in the dataset folder".format(max_images))
#         break

'''
datasize = number of images
dims = number of attributes
num_cluster = number of cluster to create'''

data_size, dims, num_clusters = len(X), 512, reid_identities

'''
x = Matrix of images' attributes
Matrix x(NXM) 
--> N= images 
--> M= features''' 

# = np.random.random_integers(5, size=(data_size, dims))
#print(x)
'''
DataFrame where linked each row(image) with its name.jpeg'''

df = pd.DataFrame(data=X)
df["index"] = df.index
df["name_image"] = reid_images

x = torch.Tensor(X)

''' 
K-Means ALGORITHM'''

cluster_ids_x, cluster_centers = kmeans(
    X=x, num_clusters=num_clusters, distance='euclidean'
)

print((cluster_centers))
print(cluster_centers.shape)

# print(type(cluster_centers))
# print(type(cluster_ids_x))
#print(x.shape)
x = cluster_ids_x.tolist()
#print(x)
#print(len(x))

'''
Inizialing a DataFrame in order to merge it afterwards in order to keep track of 
the original images'''

df1 = pd.DataFrame(data=x, columns=["cluster"])
df1["index"] = df1.index
#print(df1)

'''Merge to retrieve the name of the images'''

mergedDf = df1.merge(df, left_on='index', right_on='index')
del mergedDf["index"]
print(mergedDf)

'''
image query '''
con = np.random.randint(5, size=25)

# similarities = cosine_similarity([con], cluster_centers)
# print(similarities, max(similarities[0]))

images = []
clusters = []
simm = []

''' cosine similary among the query attributes 
and each image in the matrix X'''
from sklearn.metrics.pairwise import cosine_similarity
for i in range (len(mergedDf)):
    ##retriving list of attribute of row i
    v = mergedDf.iloc[i,1:-1].to_list()

    ##retriving the cluster of attributes' row i
    cluster =  mergedDf.iloc[i,0]

    ##retriving the image's name
    image = mergedDf.iloc[i,-1]
    #print(v)
    #print(len(v))

    ##caltulating COSINE-SIMILARTY BETWEEN QUERY'S ATTRIBUTES AND IMAGES OF ROW I ATTRIBUTES
    diff = cosine_similarity([v], [con])
    diff = diff.tolist()
    f = diff[0][0]
    print(f)
    #print(type(diff))
    #print(f)

    images.append(image)
    simm.append(f)
    clusters.append(cluster)

'''creating the final DataFrame to match all together'''
final = pd.DataFrame()
final["simm"] = simm
final["cluster"] = clusters
final["image"] = images

final.to_csv("final.csv", index=False)

'''
Calculating the mean the cosine similarty between query image and images grouping by cluster'''
new_final = final
new_final = new_final.groupby(["cluster"])["simm"].mean().reset_index()
new_final.to_csv("final_1.csv", index=False)

'''searching the cluster with the higher mean'''
c = new_final["simm"].argmax()
print(type(c))
print("c-> "+ str(c))


'''retrive all the image of the cluster previously obtained'''

final["cluster"] = final["cluster"].apply(str)
im = final[final["cluster"] == str(c)]
im = im["image"].to_list()
print(im)
