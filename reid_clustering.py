import numpy as np
import torch
from kmeans_pytorch import kmeans
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
from functions import flatten_folder
import os
import shutil
import glob
import random
from models.custom import DeepAttentionClassifier
from torchvision.io import read_image
from split import split
from evaluator import Evaluator
from sklearn.cluster import AgglomerativeClustering

def ground_truth():
    queries_list = os.listdir("reid_queries/")
    reid_list = os.listdir("reid_gallery/")


    ground_truth = {}

    for query in queries_list:
        id = query[:4]
        list = []
        for reid in reid_list:
            if reid.startswith(id):
                ground_truth[query]=list
                ground_truth[query].append(reid)

    return ground_truth

def main(cluster_type="hierarchical", max_images=1000):
    
    split(max_images=max_images)

    val_dir = "validation_directory"
    reid_queries_dir = "reid_queries"
    reid_galley_dir = "reid_gallery"
    flatten_folder(val_dir)


    try:
        shutil.rmtree(reid_galley_dir+"/")
    except FileNotFoundError as exc:
        print("No gallery directory found on your system, creating one.")

    try:
        shutil.rmtree(reid_queries_dir+"/")
    except FileNotFoundError as exc:
        print("No queries directory found on your system, creating one.")

    # INITIALIZE NEW FOLDERS (for queries and gallery
    try:
        os.mkdir(reid_galley_dir)
    except FileExistsError as exc:
        print(exc)

    try:
        os.mkdir(reid_queries_dir)
    except FileExistsError as exc:
        print(exc)

    file_list = glob.glob(val_dir + "/*.jpg")
    random.shuffle(file_list)

    #SELECT QUERY IMAGES
    for i in range(int(len(file_list)*0.20)):
        filepath = file_list[i]
        dst_path = os.path.join(reid_queries_dir, os.path.basename(filepath))
        shutil.copy(filepath, dst_path)

    queries_list = os.listdir("reid_queries/")

    #SELECT OTHER IMAGES (FOR EACH QUERY IMAGES THERE)
    for i in range(int(len(file_list)*0.20)+1, len(file_list)):
        filepath = file_list[i]
        dst_path = os.path.join(reid_galley_dir, os.path.basename(filepath))
        for q in queries_list:
            if os.path.basename(file_list[i]).split("_")[0]==q[:4]:
                shutil.copy(filepath, dst_path)

    
    queries = glob.glob(reid_queries_dir + "/*.jpg")
    reid_images = glob.glob(reid_galley_dir + "/*.jpg")

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
            if torch.cuda.is_available():
                img_tensor = img_tensor.to("cuda:0")
            feature_vec = model.encode(img_tensor).tolist()
            X.append(feature_vec)

    print(len(X), len(X[0]))

    '''
    datasize = number of images
    dims = number of attributes
    num_cluster = number of cluster to create'''

    num_clusters = reid_identities

    '''
    x = Matrix of images' attributes
    Matrix x(NXM) 
    --> N= images 
    --> M= features

    DataFrame where linked each row(image) with its name.jpeg'''


    df = pd.DataFrame(data=X)
    df["index"] = df.index 
    df["name_image"] = reid_images

    if cluster_type=="hierarchical":
        cluster = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='ward')
        cluster.fit_predict(X)
        x = cluster.labels_

    elif cluster_type=="kmeans":
        x = torch.Tensor(X)
        cluster_ids_x, _ = kmeans( X=x, num_clusters=num_clusters, distance='euclidean', device="cpu")
        x = cluster_ids_x.tolist()

    '''
    Inizialing a DataFrame in order to merge it afterwards in order to keep track of 
    the original images'''

    df1 = pd.DataFrame(data=x, columns=["cluster"])
    df1["index"] = df1.index

    '''Merge to retrieve the name of the images'''

    mergedDf = df1.merge(df, left_on='index', right_on='index')
    del mergedDf["index"]
    print(mergedDf)

    '''
    image query '''

    accuracy_list = []
    preds = {}

    for query in queries:
        con = read_image(query)/255
        con = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])(con)
        with torch.no_grad():
            con = con.unsqueeze(dim=0)
            if torch.cuda.is_available():
                con = con.to("cuda:0")
            con = model.encode(con).tolist()
            if torch.cuda.is_available():
                con = con.to("cpu")
            

        images = []
        clusters = []
        simm = []

        ''' cosine similary among the query attributes 
        and each image in the matrix X'''

        for i in range (len(mergedDf)):
            ##retriving list of attribute of row i
            v = mergedDf.iloc[i,1:-1].to_list()

            ##retriving the cluster of attributes' row i
            cluster =  mergedDf.iloc[i,0]

            ##retriving the image's name
            image = mergedDf.iloc[i,-1]

            ##caltulating COSINE-SIMILARTY BETWEEN QUERY'S ATTRIBUTES AND IMAGES OF ROW I ATTRIBUTES
            diff = cosine_similarity([v], [con])
            f = diff[0][0]

            images.append(image)
            simm.append(f)
            clusters.append(cluster)

        '''creating the final DataFrame to match all together for each query image'''
        final = pd.DataFrame()
        final["simm"] = simm
        final["cluster"] = clusters
        final["image"] = images


        #final.to_csv("final.csv", index=False)

        '''
        Calculating the mean the cosine similarity between query image and images grouping by cluster'''
        new_final = final
        new_final = new_final.groupby(["cluster"])["simm"].mean().reset_index()
        #print(new_final)
        #new_final.to_csv("final_1.csv", index=False)

        '''searching the cluster with the higher mean'''
        c = new_final["simm"].argmax()
        #print("c-> "+ str(c))


        '''retrive all the image of the cluster previously obtained'''
        final["cluster"] = final["cluster"].apply(str)
        #print("final",final)
        im = final[final["cluster"] == str(c)]
        #print("im",im)
        im = im.sort_values("simm", ascending=False) #order images in the closest cluster
        #print("im after sort", im)
        im = im["image"].to_list()  
        
        im = [os.path.basename(filename) for filename in im]
        
        query_im = os.path.basename(query)

        query_idx = query_im.split("_")[0]
        
        preds[query_im] = im 

        counts = 0
        for image in im:
            idx = image.split("_")[0]
            if query_idx == idx: 
                counts+=1
        acc = counts/len(im)
        print("Accuracy for id {}: {:.2f}".format(query_idx, acc))
        accuracy_list.append(acc)


    gt = ground_truth()
    print(preds)
    print("\n\n",gt)

    map = Evaluator.evaluate_map(preds, gt)

    print(map)

    def avg_lst(lst):
        return sum(lst) / len(lst)

    print("Average accuracy with eucliedean distance: {:.2f}".format(avg_lst(accuracy_list)))

if __name__ == "__main__":
    main(cluster_type="hierarchical", max_images=500)