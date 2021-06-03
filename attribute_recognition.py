import os
import torch
import progressbar
import pandas as pd
import multiprocessing as mp
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from models.custom import DeepAttentionClassifier
from functions import get_transforms, ClassificationDataset

COLUMN_NAMES = ["id", "age", "backpack", "bag", "handbag", "clothes", "down", "up", "hair", "hat", "gender",
                "upblack", "upwhite", "upred", "uppurple", "upyellow", "upgray", "upblue", "upgreen", "upmulticolor",
                "downblack", "downwhite", "downpink", "downpurple", "downyellow", "downgray", "downblue", "downgreen", "downbrown", "downmulticolor"]

DATA_FOLDER = "dataset/test"


def inference(model_path="networks/model.pth", out="classification_test.csv", transformations=["flip", "erasing", "rotation", "color_jitter"]):
    """Docstring here, later"""
    
    model = DeepAttentionClassifier(pretrained=False)
    # load the model and place it on the correct device based on hardware availability
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
        model = model.to("cuda:0")
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    # put the model in evaluation mode
    model.eval()
    # get the transformations
    transformations = Compose(get_transforms(transformations))
    # instantiate the table to write on disk (further 'classification_test.csv')
    table = pd.DataFrame(columns=COLUMN_NAMES)
    # initialize the dataset and the dataloader
    dataset = ClassificationDataset(img_dir=DATA_FOLDER, transform=transformations)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=mp.cpu_count()//2)
    n_iterations = int(len(os.listdir("dataset/test"))/32 + 0.5)  # ceil approx
    bar = progressbar.ProgressBar(max_value=n_iterations)
    # start producing classification output
    with torch.no_grad():
        for i, (images, paths) in enumerate(dataloader):
            bar.update(i)
            # for each image produce the output vector
            preds = model.inference(images).tolist()
            # insert the filename in the 
            for j, pred in enumerate(preds):
                pred.insert(0, paths[j])
            # update the dataframe with the current batch predictions
            preds_df = pd.DataFrame(preds, columns=COLUMN_NAMES)
            table = pd.concat([table, preds_df], axis=0)
    # write the table on the filesystem
    table.to_csv(out, index=False)

if __name__ == "__main__":
    inference()