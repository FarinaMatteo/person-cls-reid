import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.io import image 
from split import *
from functions import *
from train import classification_train, initialize_alexnet, initialize_resnet50, initialize_resnet18, initialize_resnet34, get_optimizer


def main(network="alexnet", batch_size=16, device="cuda:0", exp_name="baseline"):
    # defining folders with data and files with annotations
    folder_labels_train = "csv_files/train_label.csv"
    folder_images_train = "train_directory"
    # ... also for the validation set
    folder_label_val = "csv_files/validation_label.csv"
    folder_images_val= "validation_directory"
    
    # creating the Train and Validation sets if needed
    print("=" * 50)
    if not os.path.exists(folder_images_train) or len(os.listdir(folder_images_train))==0:
        print("\nSplitting the whole dataset into Train and Validation sets.\n")
        split()
    else:
        print("\nTrain and Validation sets already exist. No need to recreate them.\n")
    
    # creating the datasets based on the output of the previous split(s)
    transform_list = ["flip", "erasing", "rotation", "color_jitter"]
    train = get_dataset(folder_images_train, folder_labels_train, transform_list)
    val = get_dataset(folder_images_val, folder_label_val, transform_list)

    # feedinf the respective dataloaders with the datasets    
    print("=" * 50)
    print("\nInitializing Train and Validation DataLoaders...\n")
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
   
    print("=" * 50)
    print("\nInitializing the network and starting the training pipeline...\n")
    # define the writer to monitor data with Tensorboard
    writer = SummaryWriter(log_dir="experiments/{}".format(exp_name))
    
    # initialize the network and place it on the correct device according to the system
    if network=="alexnet":
        net = initialize_alexnet(num_classes=32)
    elif network=="resnet50":
        net = initialize_resnet50(num_classes=32)
    elif network=="resnet18":
        net = initialize_resnet18(num_classes=32)
    elif network=="resnet34":
        net = initialize_resnet34(num_classes=32)
    
    if device.startswith("cuda") and torch.cuda.is_available():
        net = net.to(device)

    # self explanatory
    optimizer = get_optimizer(net, lr=0.001, wd=1e-4, momentum=0.0009)

    # start the training pipeline
    classification_train(net, train_loader, val_loader, optimizer, writer, save_path=f"networks/{exp_name}/model.pth")
        
    print("=" * 50)
    print("\nTraining finished. Launch 'tensorboard --logdir=experiments' to monitor the learning curves.")
    return
        
if __name__ == "__main__":
    main(network="resnet18", exp_name="resnet18_multiloss")
    main(network="resnet34", exp_name="resnet34_multiloss")
    main(network="resnet50", exp_name="resnet50_multiloss")