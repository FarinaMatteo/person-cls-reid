import os
import torch
import torchvision
import torch.nn.functional as F
from functions import set_parameter_requires_grad
from models.custom import DeepAttentionClassifier


def initialize_alexnet(num_classes, feature_extracting=False):
    # load the pre-trained Alexnet
    alexnet = torchvision.models.alexnet(pretrained=True)

    # get the number of neurons in the penultimate layer
    in_features = alexnet.classifier[6].in_features

    # freeze the backbone if feature extracting
    if feature_extracting:
        set_parameter_requires_grad(model=alexnet)
    
    # re-initalize the output layer
    alexnet.classifier[6] = torch.nn.Linear(in_features=in_features, out_features=num_classes)

    return alexnet


def initialize_resnet50(num_classes, feature_extracting=False):
    # load the pre-trained resnet50
    resnet50 = torchvision.models.resnet50(pretrained=True)
    in_features = resnet50.fc.in_features
    # freeze the backbone if feature extracting
    if feature_extracting:
        set_parameter_requires_grad(model=resnet50)
    resnet50.fc = torch.nn.Linear(in_features=in_features, out_features=num_classes)
    return resnet50


def initialize_resnet18(num_classes, feature_extracting=False):
    # load the pre-trained resnet18
    resnet18 = torchvision.models.resnet18(pretrained=True)
    # freeze the backbone if feature extracting
    if feature_extracting:
        set_parameter_requires_grad(model=resnet18)
    in_features = resnet18.fc.in_features
    resnet18.fc = torch.nn.Linear(in_features=in_features, out_features=num_classes)
    return resnet18


def initialize_resnet101(num_classes, feature_extracting=False):
    # load the pre-trained resnet101
    resnet101 = torchvision.models.resnet101(pretrained=True)
    in_features = resnet101.fc.in_features
    # freeze the backbone if feature extracting
    if feature_extracting:
        set_parameter_requires_grad(model=resnet101)
    resnet101.fc = torch.nn.Linear(in_features=in_features, out_features=num_classes)
    return resnet101


def initialize_resnet34(num_classes, feature_extracting=False):
    # load the pre-trained resnet50
    resnet34 = torchvision.models.resnet34(pretrained=True)
    in_features = resnet34.fc.in_features
    # freeze the backbone if feature extracting
    if feature_extracting:
        set_parameter_requires_grad(model=resnet34)
    resnet34.fc = torch.nn.Linear(in_features=in_features, out_features=num_classes)
    return resnet34


def initialize_densenet(num_classes, feature_extracting=False):
    # load the pretrained network
    densenet = torchvision.models.densenet121(pretrained=True)
    in_features = densenet.classifier.in_features
    # freeze the backbone if feature extracting
    if feature_extracting:
        set_parameter_requires_grad(model=densenet)  
    densenet.classifier = torch.nn.Linear(in_features=in_features, out_features=num_classes)
    return densenet


def initialize_attention_classifier(pretrained=True, feature_extracting=True):
    classifier = DeepAttentionClassifier(pretrained=pretrained)
    if feature_extracting:
        set_parameter_requires_grad(model=classifier.backbone)
    return classifier


def get_optimizer(model, lr, wd, momentum, net_name, optim_name="SGD"):
    assert optim_name in ("SGD", "RMSprop", "Adam")
    assert net_name in ("alexnet", "resnet", "densenet", "attentionnet")

    # we will create two groups of weights, one for the newly initialized layer
    # and the other for rest of the layers of the network
    final_layer_weights = []
    rest_of_the_net_weights = []

    # we will iterate through the layers of the network
    for name, param in model.named_parameters():
        if (name.startswith('classifier.6') and net_name == "alexnet") \
            or (name.startswith("fc") and net_name.startswith("resnet")) \
            or (name.startswith("classifier") and net_name == "densenet") \
            or (name.startswith("backbone") and net_name == "attentionnet"):
            final_layer_weights.append(param)
        else:
            rest_of_the_net_weights.append(param)

    # so now we have divided the network weights into two groups.
    # We will train the final_layer_weights with learning_rate = lr
    # and rest_of_the_net_weights with learning_rate = lr / 10
    if optim_name == "SGD":
        return torch.optim.SGD([
            {'params': rest_of_the_net_weights},
            {'params': final_layer_weights, 'lr': lr}
        ], lr=lr/10, weight_decay=wd, momentum=momentum)
    elif optim_name == "Adam":
        return torch.optim.Adam([
            {'params': rest_of_the_net_weights},
            {'params': final_layer_weights, 'lr':lr}
        ], lr=lr/10, weight_decay=wd)
    elif optim_name == "RMSprop":
        return torch.optim.RMSprop([
            {'params': rest_of_the_net_weights},
            {'params': final_layer_weights, 'lr':lr}
        ], lr=lr/10, weight_decay=wd, momentum=momentum)
    

def train(net, loader, optimizer, norm_loss=True, avg_loss=False, device='cuda:0'):
    
    assert not (norm_loss and avg_loss), "Loss Normalization and Loss Averaging are mutually exclusive.\nPlease select at most one of them."
    
    num_samples = 0.0
    total_loss = 0.0
    total_up_acc = 0.
    total_down_acc = 0.
    total_age_acc = 0.
    total_rest_acc = 0.
    total_avg_acc = 0.

    # define loss functions to be used throughout the process
    ce_loss = torch.nn.CrossEntropyLoss()
    bce_loss = torch.nn.BCELoss()

    # select the device according to system hardware
    if device.startswith("cuda") and torch.cuda.is_available():
        device = 'cuda:0'
        ce_loss = ce_loss.to(device)
        bce_loss = bce_loss.to(device)
    else:
        device = 'cpu'

    # put the network in training mode
    net.train()

    for (images, labels) in loader:
                
        # move tensors onto gpu if needed
        if device.startswith("cuda"):
            images = images.to(device)
            labels = labels.to(device)
        else:
            images = images.cpu()
            labels = labels.cpu()

        # apply a forward pass
        preds = net(images)

        # compute the cross entropy loss for the first 4 predictions (age-related)
        age_preds = preds[:, :4]
        age_labels = labels[:, 0]
        age_loss = ce_loss(age_preds, age_labels)

        # compute losses for the first 9 independent features
        independent_labels = labels[:, 1:10].float()
        independent_preds = preds[:, 4:13]
        # normalize preds into 0-1 range with softmax
        independent_preds = torch.sigmoid(independent_preds)
        ind_loss = bce_loss(independent_preds, independent_labels)

        # compute losses for upper and lower body clothing color
        # upper body cross entropy loss
        up_labels = torch.argmax(labels[:, 10:19], dim=1)
        up_preds = preds[:, 13:22]
        up_ce_loss = ce_loss(up_preds, up_labels)

        # lower body cross entropy loss
        down_labels = torch.argmax(labels[:, 19:], dim=1)
        down_preds = preds[:, 22:]
        down_ce_loss = ce_loss(down_preds, down_labels)

        # normalize loss based on settings
        if norm_loss:
            loss = 0.375*up_ce_loss + 0.417*down_ce_loss + 0.041*ind_loss + 0.167*age_loss
        elif avg_loss:
            loss = (up_ce_loss + down_ce_loss + ind_loss + age_loss) / 4.0
        else:
            loss = up_ce_loss + down_ce_loss + ind_loss + age_loss 

        # backprop
        loss.backward()

        # optimize and reset
        optimizer.step()
        optimizer.zero_grad()

        # update stats
        num_samples += images.shape[0]
        total_loss += loss.item()
        # update stats for accuracies (cross-entropy losses)
        total_age_acc += torch.argmax(age_preds, dim=1).eq(age_labels).sum().item()
        total_up_acc += torch.argmax(up_preds, dim=1).eq(up_labels).sum().item()
        total_down_acc += torch.argmax(down_preds, dim=1).eq(down_labels).sum().item()
        # update stats for bc accuracy
        total_rest_acc += independent_preds.round().eq(independent_labels).sum().item()/9

    total_avg_acc = (((total_up_acc+total_down_acc+total_age_acc+total_rest_acc)/4)/num_samples)*100

    return total_loss/num_samples, total_up_acc/num_samples*100, total_down_acc/num_samples*100,\
        total_age_acc/num_samples*100, total_rest_acc/num_samples*100, total_avg_acc


def test(net, loader, norm_loss=True, avg_loss=False, device="cuda:0"):

    assert not (norm_loss and avg_loss), "Loss Normalization and Loss Averaging are mutually exclusive.\nPlease select at most one of them."
    
    num_samples = 0.0
    total_loss = 0.0
    total_up_acc = 0.
    total_down_acc = 0.
    total_age_acc = 0.
    total_rest_acc = 0.

    # define loss functions to be used throughout the process
    ce_loss = torch.nn.CrossEntropyLoss()
    bce_loss = torch.nn.BCELoss()

    # select the device according to system hardware
    if device.startswith("cuda") and torch.cuda.is_available():
        device = 'cuda:0'
        ce_loss = ce_loss.to(device)
        bce_loss = bce_loss.to(device)
    else:
        device = 'cpu'

    # put the network in evaluation mode
    net.eval()

    # set context manager to avoid gradients computation
    with torch.no_grad():
        for (images, labels) in loader:

            # move tensors onto gpu if needed
            if device.startswith("cuda"):
                images = images.to(device)
                labels = labels.to(device)
            else:
                images = images.cpu()
                labels = labels.cpu()

            # apply a forward pass
            preds = net(images)
            # compute the cross entropy loss for the first 4 predictions (age-related)
            age_preds = preds[:, :4]
            age_labels = labels[:, 0]
            age_loss = ce_loss(age_preds, age_labels)

            # compute losses for the first 9 independent features
            independent_labels = labels[:, 1:10].float()
            independent_preds = preds[:, 4:13]
            # normalize preds into 0-1 range with softmax
            independent_preds = torch.sigmoid(independent_preds)
            ind_loss = bce_loss(independent_preds, independent_labels)

            # compute losses for upper and lower body clothing color
            # upper body cross entropy loss
            up_labels = torch.argmax(labels[:, 10:19], dim=1)
            up_preds = F.softmax(preds[:, 13:22], dim=1)
            up_ce_loss = ce_loss(up_preds, up_labels)

            # lower body cross entropy loss
            down_labels = torch.argmax(labels[:, 19:], dim=1)
            down_preds = F.softmax(preds[:, 22:], dim=1)
            down_ce_loss = ce_loss(down_preds, down_labels)

            # normalize loss based on settings
            if norm_loss:
                loss = 0.375*up_ce_loss + 0.417*down_ce_loss + 0.041*ind_loss + 0.167*age_loss
            elif avg_loss:
                loss = (up_ce_loss + down_ce_loss + ind_loss + age_loss) / 4.0
            else:
                loss = up_ce_loss + down_ce_loss + ind_loss + age_loss 

            # update stats for loss
            num_samples += images.shape[0]
            total_loss += loss.item()
            # update stats for accuracies (cross-entropy losses)
            total_age_acc += torch.argmax(age_preds, dim=1).eq(age_labels).sum().item()
            total_up_acc += torch.argmax(up_preds, dim=1).eq(up_labels).sum().item()
            total_down_acc += torch.argmax(down_preds, dim=1).eq(down_labels).sum().item()
            # update stats for bc accuracy
            total_rest_acc += independent_preds.round().eq(independent_labels).sum().item()/9

    total_avg_acc = (((total_up_acc+total_down_acc+total_age_acc+total_rest_acc)/4)/num_samples)*100

    return total_loss/num_samples, total_up_acc/num_samples*100, total_down_acc/num_samples*100,\
        total_age_acc/num_samples*100, total_rest_acc/num_samples*100, total_avg_acc


def classification_train(net, tr_loader, val_loader, optimizer, writer,
                        scheduler_mode="min", norm_loss=True, avg_loss=False, 
                        epochs=30, save_path="networks/baseline.pth", patience=5):

    assert not (norm_loss and avg_loss), "Loss Normalization and Loss Averaging are mutually exclusive.\nPlease select at most one of them."
    assert scheduler_mode in ("min", "max"), "Scheduler mode must be set either to 'min' or 'max'."
    
    # define the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=scheduler_mode, patience=patience//2)
    
    # check the random behaviour before training
    training_loss, tr_up_acc, tr_down_acc, tr_age_acc, tr_rest_acc, tr_avg_acc = test(net, tr_loader, norm_loss=norm_loss, avg_loss=avg_loss)
    validation_loss, val_up_acc, val_down_acc, val_age_acc, val_rest_acc, val_avg_acc = test(net, val_loader, norm_loss=norm_loss, avg_loss=avg_loss)

    # print data and add data to tensorboard for further monitoring
    print("=" * 100, "\n")
    print("Values Before Training:")
    print("\tTraining Loss: {:.4f}".format(training_loss))
    print("\tTrain Up Accuracy: {:.2f}".format(tr_up_acc))
    print("\tTrain Down Accuracy: {:.2f}".format(tr_down_acc))
    print("\tTrain Age Accuracy: {:.2f}".format(tr_age_acc))
    print("\tTrain Rest Accuracy: {:.2f}".format(tr_rest_acc))
    print("\tTrain Avg Accuracy: {:.2f}\n".format(tr_avg_acc))
    print("\tValidation Loss: {:.4f}".format(validation_loss))
    print("\tValidation Up Accuracy: {:.2f}".format(val_up_acc))
    print("\tValidation Down Accuracy: {:.2f}".format(val_down_acc))
    print("\tValidation Age Accuracy: {:.2f}".format(val_age_acc))
    print("\tValidation Rest Accuracy: {:.2f}".format(val_rest_acc))
    print("\tValidation Avg Acc: {:.2f}\n".format(val_avg_acc))
    writer.add_scalar("Loss/training", training_loss, 0)
    writer.add_scalar("Loss/validation", validation_loss, 0)
    writer.add_scalar("Accuracy/training/upperbody_accuracy", tr_up_acc, 0)
    writer.add_scalar("Accuracy/training/lowerbody_accuracy", tr_down_acc, 0)
    writer.add_scalar("Accuracy/training/age_accuracy", tr_age_acc, 0)
    writer.add_scalar("Accuracy/training/binary_accuracy_avg", tr_rest_acc, 0)
    writer.add_scalar("Accuracy/training/average_accuracy", tr_avg_acc, 0)
    writer.add_scalar("Accuracy/validation/upperbody_accuracy", val_up_acc, 0)
    writer.add_scalar("Accuracy/validation/lowerbody_accuracy", val_down_acc, 0)
    writer.add_scalar("Accuracy/validation/age_accuracy", val_age_acc, 0)
    writer.add_scalar("Accuracy/validation/binary_accuracy_avg", val_rest_acc, 0)
    writer.add_scalar("Accuracy/validation/average_accuracy", val_avg_acc, 0)

    # set variables needed for the Early Stopping mechanism
    es_epochs = 0
    best_loss = 1_000_000_000_000
    
    # start the training loop, process the whole dataset 'epochs' times
    for e in range(epochs):
        
        # training and validation step
        training_loss, tr_up_acc, tr_down_acc, tr_age_acc, tr_rest_acc, tr_avg_acc = train(net, tr_loader, optimizer, norm_loss=norm_loss, avg_loss=avg_loss)
        validation_loss, val_up_acc, val_down_acc, val_age_acc, val_rest_acc, val_avg_acc = test(net, val_loader, norm_loss=norm_loss, avg_loss=avg_loss)
        
        # use the scheduler to update the learning rate after the training step
        if scheduler_mode == "min":
            scheduler.step(validation_loss)
        elif scheduler_mode == "max":
            scheduler.step(val_avg_acc)

        # still, print out data to monitor the training (and also confirm everything is still working and nothing has crashed...)
        print("=" * 100, "\n")
        print("Epoch {}/{}\n".format(e+1, epochs))
        print("Values After Training Epoch {}".format(e+1))
        print("\tTraining Loss: {:.4f}".format(training_loss))
        print("\tTrain Up Accuracy: {:.2f}".format(tr_up_acc))
        print("\tTrain Down Accuracy: {:.2f}".format(tr_down_acc))
        print("\tTrain Age Accuracy: {:.2f}".format(tr_age_acc))
        print("\tTrain Rest Accuracy: {:.2f}".format(tr_rest_acc))
        print("\tTrain Avg Accuracy: {:.2f}\n".format(tr_avg_acc))
        print("\tValidation Loss: {:.4f}".format(validation_loss))
        print("\tValidation Up Accuracy: {:.2f}".format(val_up_acc))
        print("\tValidation Down Accuracy: {:.2f}".format(val_down_acc))
        print("\tValidation Age Accuracy: {:.2f}".format(val_age_acc))
        print("\tValidation Rest Accuracy: {:.2f}".format(val_rest_acc))
        print("\tValidation Avg Acc: {:.2f}\n".format(val_avg_acc))
        writer.add_scalar("Loss/training", training_loss, e+1)
        writer.add_scalar("Loss/validation", validation_loss, e+1)
        writer.add_scalar("Accuracy/training/upperbody_accuracy", tr_up_acc, e+1)
        writer.add_scalar("Accuracy/training/lowerbody_accuracy", tr_down_acc, e+1)
        writer.add_scalar("Accuracy/training/age_accuracy", tr_age_acc, e+1)
        writer.add_scalar("Accuracy/training/binary_accuracy_avg", tr_rest_acc, e+1)
        writer.add_scalar("Accuracy/training/average_accuracy", tr_avg_acc, e+1)
        writer.add_scalar("Accuracy/validation/upperbody_accuracy", val_up_acc, e+1)
        writer.add_scalar("Accuracy/validation/lowerbody_accuracy", val_down_acc, e+1)
        writer.add_scalar("Accuracy/validation/age_accuracy", val_age_acc, e+1)
        writer.add_scalar("Accuracy/validation/binary_accuracy_avg", val_rest_acc, e+1)
        writer.add_scalar("Accuracy/validation/average_accuracy", val_avg_acc, e+1)

        # early stopping check, if no loss improvement with respect to the best loss value, update the counter
        if validation_loss >= best_loss:
            es_epochs += 1
        # otherwise, if loss has improved, update the best loss value and make sure to reset the counter!
        else:
            es_epochs = 0
            best_loss = validation_loss
            # save the best model according to the provided path
            if not os.path.exists(os.path.dirname(save_path)): 
                os.makedirs(os.path.dirname(save_path))
            print("\tModel Loss improved, saving at checkpoint at: {}".format(save_path))
            torch.save(net.state_dict(), save_path)

        # check against the patience and exit due to early stopping conditions
        if es_epochs > patience:
            print("\nEarly Stopping has been triggered. Exiting the training pipeline.\n")
            break

    return net


