"""Run the person Re-ID pipeline of the project creating 'reid_test.txt'"""
import os
import glob
import torch
import pandas as pd
from kmeans_pytorch import kmeans
from torchvision import transforms
from torchvision.io import read_image
from models.custom import DeepAttentionClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

def vec2str(vec):
  vec = [str(item) for item in vec]
  return ", ".join(vec)

def main(cluster_type="hierarchical"):
    
    test_dir = "dataset/test"
    queries_dir = "dataset/queries/"
    
    test_list = glob.glob(test_dir + "/*.jpg")
    queries_list = glob.glob(queries_dir + "/*.jpg")

    model = DeepAttentionClassifier(pretrained=False)
    weights_path = "network/model.pth"

    # load the model
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(weights_path))
        model = model.to("cuda:0")
    else:
        model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))
    model.eval()

    i=0
    X = []
    with torch.no_grad():
        # for each image produce the feature vector
        for image_path in test_list:
            img_tensor = read_image(image_path)/255
            img_tensor = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])(img_tensor)
            img_tensor = img_tensor.unsqueeze(dim=0)
            i+=1
            print(i)
            if torch.cuda.is_available():
                img_tensor = img_tensor.to("cuda:0")
            feature_vec = model.encode(img_tensor).tolist()
            X.append(feature_vec)
   
    '''
    num_cluster = number of cluster to create '''

    #750 identities + 1 cluster for junk images
    num_clusters = 751   

    '''
    X = Matrix of images attributes (NXM) 
    --> N = number images 
    --> M = number of features  '''

    #Inizialing a DataFrame containing the feature vector for each image
    df = pd.DataFrame(data=X)
    df["index"] = df.index 
    im = [os.path.basename(filename) for filename in test_list]
    df["name_image"] = im


    #clustering
    if cluster_type=="hierarchical":
        cluster = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='ward')
        cluster.fit_predict(X)
        x = cluster.labels_

    elif cluster_type=="kmeans":
        x = torch.Tensor(X)
        cluster_ids_x, _ = kmeans(X=x, num_clusters=num_clusters, distance='euclidean', device="cpu")
        x = cluster_ids_x.tolist()


    
    #Inizialing a DataFrame in order to merge it afterwards in order to keep track of the original images
    df1 = pd.DataFrame(data=x, columns=["cluster"])
    df1["index"] = df1.index

    #Merge to retrieve the name of the images
    mergedDf = df1.merge(df, left_on='index', right_on='index')
    del mergedDf["index"]

    #compute the centroid for each cluster
    centroids = {}
    for i in range(num_clusters):
        cluster = mergedDf[mergedDf["cluster"]==i]
        del cluster["cluster"]
        del cluster["name_image"]
        cluster_centroids = cluster.mean(axis=0)
        cluster_centroids = cluster_centroids.to_list()
        centroids[str(i)] = cluster_centroids


    preds = {}
    for query in queries_list:
        # for each image in the gallery folder produce the feature vector
        query_features = read_image(query)/255
        query_features = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])(query_features)
        with torch.no_grad():
            query_features = query_features.unsqueeze(dim=0)
            if torch.cuda.is_available():
                query_features = query_features.to("cuda:0")
            query_features = model.encode(query_features).tolist()

        
        #cosine similary among the query attributes and each centroid, to select the best cluster
        max = 0
        for i in range(num_clusters):
            similarity = cosine_similarity([query_features], [centroids[str(i)]])
            similarity = similarity[0][0]

            if similarity > max:
                max = similarity
                best_cluster = i
        
        #images only in the best cluster
        selected_images = mergedDf[mergedDf["cluster"]==best_cluster]

        images = []
        simm = []

        for i in range (len(selected_images)):
            #retriving list of attribute of row i
            v = selected_images.iloc[i,1:-1].to_list()

            #retriving the image's name
            image = selected_images.iloc[i,-1]

            #caltulating cosine similarity between query attributes and cluster images
            similarity = cosine_similarity([v], [query_features])
            similarity = similarity[0][0]

            images.append(image)
            simm.append(similarity)


        #creating the final DataFrame
        final = pd.DataFrame()
        final["simm"] = simm
        final["cluster"] = best_cluster
        final["image"] = images

        #order images in the closest cluster
        final = final.sort_values(by="simm", ascending=False)
        im = final["image"].to_list()

        query_im = os.path.basename(query)

        #dictionary with predictions for each query image
        preds[query_im] = im 

    #print the final file with all the predictions
    with open('reid_test.txt', 'w') as f:
      for key in preds.keys():
        f.write(key + ": ")
        f.write(vec2str(preds[key]))
        f.write("\n")
    

if __name__ == "__main__":
    main(cluster_type="hierarchical")