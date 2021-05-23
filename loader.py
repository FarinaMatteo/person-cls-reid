import os
import torch
from split import split
from functions import *
import multiprocessing as mp
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from train import classification_train, initialize_alexnet,\
                  initialize_resnet50, initialize_resnet18, initialize_resnet34, initialize_resnet101, \
                  initialize_densenet, initialize_attention_classifier, get_optimizer


def main(network="alexnet", batch_size=16, batch_mode="id", 
        train_mode="finetune", max_images=None, train_split=0.8, 
        norm_loss=True, avg_loss=False, device="cuda:0", exp_name="baseline"):
    
    assert batch_mode in ("id", "img")
    assert train_mode in ("finetune", "feature_extract")
    feature_extracting = not (train_mode == "finetune")

    # defining folders with data and files with annotations
    folder_labels_train = "csv_files/train_label.csv"
    folder_images_train = "train_directory"
    # ... also for the validation set
    folder_label_val = "csv_files/validation_label.csv"
    folder_images_val= "validation_directory"
    
    # creating the Train and Validation sets if needed
    print("=" * 100)
    print("\nSplitting the whole dataset into Train and Validation sets.")
    print("Your current configuration is:\n\tTrain split: {}\n\tMax Images: {}\n".format(train_split, max_images))
    split(train_split=train_split, max_images=max_images)

    # access elements by idx in the fs
    if batch_mode=="img":
        flatten_folder(folder_images_train)
        flatten_folder(folder_images_val)
    
    # creating the datasets based on the output of the previous split(s)
    transform_list = ["double", "flip", "erasing", "rotation", "color_jitter"]
    train = get_dataset(folder_images_train, folder_labels_train, transform_list, batch_mode)
    val = get_dataset(folder_images_val, folder_label_val, transform_list, batch_mode)

    # feeding the respective dataloaders with the datasets    
    print("=" * 100)
    print("\nInitializing Train and Validation DataLoaders...\n")
    if batch_mode == "id":
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=mp.cpu_count()//2)
        val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=mp.cpu_count()//2)
    elif batch_mode == "img":
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=mp.cpu_count()//2)
        val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=mp.cpu_count()//2)

    print("=" * 100)
    print("\nInitializing the network and starting the training pipeline...\n")
    # define the writer to monitor data with Tensorboard
    writer = SummaryWriter(log_dir="experiments/{}".format(exp_name))
    
    # initialize the network and place it on the correct device according to the system
    if network == "alexnet":
        net = initialize_alexnet(num_classes=32, feature_extracting=feature_extracting)
    elif network == "resnet50":
        net = initialize_resnet50(num_classes=32, feature_extracting=feature_extracting)
    elif network == "resnet18":
        net = initialize_resnet18(num_classes=32, feature_extracting=feature_extracting)
    elif network == "resnet34":
        net = initialize_resnet34(num_classes=32, feature_extracting=feature_extracting)
    elif network == "resnet101":
        net = initialize_resnet101(num_classes=32, feature_extracting=feature_extracting)
    elif network == "densenet":
        net = initialize_densenet(num_classes=32, feature_extracting=feature_extracting)
    elif network == "attentionnet":
        net = initialize_attention_classifier(pretrained=True, feature_extracting=feature_extracting)
    
    if device.startswith("cuda") and torch.cuda.is_available():
        net = net.to(device)

    # self explanatory
    optimizer = get_optimizer(net, lr=0.1, wd=1e-4, momentum=0.09, net_name=network)
    
    summary(net, (3, 128, 64))
    classification_train(net, train_loader, val_loader, optimizer, writer, \
                         norm_loss=norm_loss, avg_loss=avg_loss, epochs=100, save_path=f"networks/{exp_name}/model.pth", patience=5)
        
    print("=" * 100)
    print("\nTraining finished. Launch 'tensorboard --logdir=experiments' to monitor the learning curves.\n")
    return
        
if __name__ == "__main__":
    # main(network="resnet18", batch_mode="img", exp_name="resnet18_weighted_multiloss_img", batch_size=128, max_images=1000)
    # main(network="resnet34", batch_mode="img", exp_name="resnet34_weighted_multiloss_img", batch_size=128, max_images=1000)
    # main(network="resnet50", batch_mode="img", exp_name="resnet50_weighted_multiloss_img", batch_size=128, max_images=1000)
    # main(network="resnet101", batch_mode="img", exp_name="resnet101_weighted_multiloss_img", batch_size=128, max_images=1000)
    # main(network="densenet", batch_mode="img", exp_name="densenet_weighted_multiloss_img", batch_size=128, max_images=2000)
    # main(network="densenet", batch_mode="img", exp_name="densenet_weighted_multiloss_img_feature_extract", batch_size=128, max_images=2000, train_mode="feature_extract")
    main(network="attentionnet", batch_size=8, batch_mode="img", train_mode="feature_extract", max_images=10, exp_name="attention_net")



