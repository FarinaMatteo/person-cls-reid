import os
import torch
import pandas as pd
from torchvision.io import read_image
from torchvision.transforms import Compose
from models.custom import DeepAttentionClassifier
from functions import get_transforms

COLUMN_NAMES = ["id", "age", "backpack", "bag", "handbag", "clothes", "down", "up", "hair", "hat", "gender",
                "upblack", "upwhite", "upred", "uppurple", "upyellow", "upgray", "upblue", "upgreen", "upmulticolor",
                "downblack", "downwhite", "downpink", "downpurple", "downyellow", "downgray", "downblue", "downgreen", "downbrown", "downmulticolor"]


def inference(model_path="networks/attention_net/model.pth", out="classification_test.csv", 
              transformations=["flip", "erasing", "rotation", "color_jitter"]):
    model = DeepAttentionClassifier(pretrained=False)
    # load the model
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
        model = model.to("cuda:0")
    else:
        model.load_state_dict(torch.load(model_path))

    model.eval()
    # get the transformations
    transformations = Compose(get_transforms(transformations))
    # instantiate the table to write on disk
    table = []
    # scan the test folder
    file_list = os.listdir("dataset/test")
    with torch.no_grad():
        for f in file_list[:150]:
            # load the tensor corresponding to the current image and transform it properly
            tensor =  transformations(read_image(os.path.join("dataset/test", f))/255).unsqueeze(dim=0)
            # for each image produce the output vector
            preds = model.inference(tensor).squeeze().tolist()
            preds.insert(0, f)
            # save the row inside the table
            table.append(preds)

    # write the table on the filesystem
    table = pd.DataFrame(table, columns=COLUMN_NAMES)
    table.to_csv(out, index=False)

if __name__ == "__main__":
    inference()