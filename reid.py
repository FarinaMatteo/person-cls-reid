import os
import sys
from models.custom import DeepAttentionClassifier
import torch
import torchvision
import torch.nn as nn

def extract_features():
    # silly check
    #cwd = os.getcwd()
    #if "models" in cwd.split("/"):
    #    logging.error("Run this script from the root folder of the project.\nChange directory to 'person-cls-reid' and run 'python models/custom.py'.")
    #    sys.exit(0)

    # read images from 'train_directory'. 
    file_list = os.listdir("train_directory/")

    features = []
    for i in file_list:
        image = os.path.join("train_directory", i)
        img_tensor = torchvision.io.read_image(image)/255
        img_tensor = img_tensor.unsqueeze(dim=0)

        # initialize the classifier
        deep_classifier = DeepAttentionClassifier(pretrained=True)

        # perform feature extraction using the classifier, outputting a single 1d vector of 512 features.
        # NOTE: REMEMBER TO PUT THE MODEL IN EVALUATION MODE WHEN EXTRACTING EMBEDDINGS FROM TENSORS!
        deep_classifier.eval()
        embedding = deep_classifier.encode(img_tensor)
        features.append(embedding)  

    return features

extract_features()




