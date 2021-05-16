import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.io import image 
from split import *
from functions import *
from train import classification_train, initialize_alexnet, get_cost_fn, get_optimizer

def main(device="cuda:0"):
    folder_label_test = "csv_files/train_label.csv"
    folder_images_test= "train_directory"
    if not os.path.exists(folder_images_test) or len(os.listdir(folder_images_test))==0:
        split()
    
    transformation = 0
    test = get_dataset(folder_images_test, folder_label_test, transformation)

    # print_images(test, transformation, train=True)

    folder_label_val = "csv_files/validation_label.csv"
    folder_images_val= "validation_directory"
    transformation = 2
    val = get_dataset(folder_images_val, folder_label_val, transformation)

    
    # print_images(val, transformation , train=False)

        
    print("-------------")

    train_loader = torch.utils.data.DataLoader(test,batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val,batch_size=2, shuffle=True, collate_fn=collate_fn)
   
    print("-------------")

    # define the writer to monitor data with Tensorboard
    writer = SummaryWriter(log_dir="experiments/baseline")
    
    # initialize the network and place it on the correct device according to the system
    net = initialize_alexnet(num_classes=32)
    if device.startswith("cuda") and torch.cuda.is_available():
        net = net.to(device)

    # self explanatory
    optimizer = get_optimizer(net, lr=0.001, wd=1e-4, momentum=0.0009)
    cost_fn = get_cost_fn()

    # start the training pipeline
    classification_train(net, train_loader, val_loader, cost_fn, optimizer, writer, epochs=15)
        
    print("-------------")
    return
        
if __name__ == "__main__":
    main()