import torch
import torchvision
from torch.nn import functional as F

def initialize_alexnet(num_classes):
  # load the pre-trained Alexnet
  alexnet = torchvision.models.alexnet(pretrained=True)
  
  # get the number of neurons in the penultimate layer
  in_features = alexnet.classifier[6].in_features
  
  # re-initalize the output layer
  alexnet.classifier[6] = torch.nn.Linear(in_features=in_features, 
                                          out_features=num_classes)
  
  return alexnet

def get_optimizer(model, lr, wd, momentum):
  
  # we will create two groups of weights, one for the newly initialized layer
  # and the other for rest of the layers of the network
  
  final_layer_weights = []
  rest_of_the_net_weights = []
  
  # we will iterate through the layers of the network
  for name, param in model.named_parameters():
    if name.startswith('classifier.6'):
      final_layer_weights.append(param)
    else:
      rest_of_the_net_weights.append(param)
  
  # so now we have divided the network weights into two groups.
  # We will train the final_layer_weights with learning_rate = lr
  # and rest_of_the_net_weights with learning_rate = lr / 10
  
  optimizer = torch.optim.Adam([
      {'params': rest_of_the_net_weights},
      {'params': final_layer_weights, 'lr': lr}
  ], lr=lr / 10, weight_decay=wd)
  
  return optimizer


def train(net, loader, optimizer, device='cuda:0'):
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
        
        # compute losses for the first 12 independent features
        independent_labels = labels[:, 1:10].float()
        independent_preds = preds[:, 4:13]
        # normalize preds into 0-1 range with softmax
        independent_preds = torch.sigmoid(independent_preds)
        ind_loss = bce_loss(independent_preds, independent_labels)

        # compute losses for upper and lower body clothing color
        ## upper body cross entropy loss 
        up_labels = torch.argmax(labels[:, 10:18], dim=1)
        up_preds = preds[:, 13:21]
        up_ce_loss = ce_loss(up_preds, up_labels)
        ## lower body cross entropy loss
        down_labels = torch.argmax(labels[:, 18:], dim=1)
        down_preds = preds[:, 21:]
        down_ce_loss = ce_loss(down_preds, down_labels)

        # compute overall loss
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
        total_rest_acc += independent_preds.round().eq(independent_labels).sum().item()

    return total_loss/num_samples, total_up_acc/num_samples*100, total_down_acc/num_samples*100, total_age_acc/num_samples*100, (total_rest_acc/(num_samples*9))*100


def test(net, loader, device="cuda:0"):
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
    else:
      device = 'cpu'
    
    # put the network in evaluation mode
    net.eval()
    
    # set context manager to avoid gradients computation
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
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
            
            # compute losses for the first 12 independent features
            independent_labels = labels[:, 1:10].float()
            independent_preds = preds[:, 4:13]
            # normalize preds into 0-1 range with softmax
            independent_preds = torch.sigmoid(independent_preds)
            ind_loss = bce_loss(independent_preds, independent_labels)

            # compute losses for upper and lower body clothing color
            ## upper body cross entropy loss 
            up_labels = torch.argmax(labels[:, 10:18], dim=1)
            up_preds = preds[:, 13:21]
            up_ce_loss = ce_loss(up_preds, up_labels)
            ## lower body cross entropy loss
            down_labels = torch.argmax(labels[:, 18:], dim=1)
            down_preds = preds[:, 21:]
            down_ce_loss = ce_loss(down_preds, down_labels)

            # compute overall loss
            loss = up_ce_loss + down_ce_loss + ind_loss + age_loss
            
            # update stats for loss
            num_samples += images.shape[0]
            total_loss += loss.item()
            # update stats for accuracies (cross-entropy losses)
            total_age_acc += torch.argmax(age_preds, dim=1).eq(age_labels).sum().item()
            total_up_acc += torch.argmax(up_preds, dim=1).eq(up_labels).sum().item()
            total_down_acc += torch.argmax(down_preds, dim=1).eq(down_labels).sum().item()
            # update stats for bc accuracy
            total_rest_acc += independent_preds.round().eq(independent_labels).sum().item()

    return total_loss/num_samples, total_up_acc/num_samples*100, total_down_acc/num_samples*100, total_age_acc/num_samples*100, (total_rest_acc/(num_samples*9))*100


def classification_train(net, tr_loader, val_loader, optimizer, writer, epochs=20, save_dir="networks/baseline.pth"):
    
    training_loss, tr_up_acc, tr_down_acc, tr_age_acc, tr_rest_acc = test(net, tr_loader)
    validation_loss, val_up_acc, val_down_acc, val_age_acc, val_rest_acc = test(net, val_loader)
    
    print("=" * 50, "\n")
    print("Values Before Training:")
    print("\tTraining Loss: {:.4f}".format(training_loss))
    print("\tTrain Up Accuracy: {:.2f}".format(tr_up_acc))
    print("\tTrain Down Accuracy: {:.2f}".format(tr_down_acc))
    print("\tTrain Age Accuracy: {:.2f}".format(tr_age_acc))
    print("\tTrain Rest Accuracy: {:.2f}\n".format(tr_rest_acc))
    print("\tValidation Loss: {:.4f}".format(validation_loss))
    print("\tValidation Up Accuracy: {:.2f}".format(val_up_acc))
    print("\tValidation Down Accuracy: {:.2f}".format(val_down_acc))
    print("\tValidation Age Accuracy: {:.2f}".format(val_age_acc))
    print("\tValidation Rest Accuracy: {:.2f}\n".format(val_rest_acc))
    writer.add_scalar("Loss/training", training_loss, 0)
    writer.add_scalar("Loss/validation", validation_loss, 0)
    writer.add_scalar("Accuracy/training/upperbody_accuracy", tr_up_acc, 0)
    writer.add_scalar("Accuracy/training/lowerbody_accuracy", tr_down_acc, 0)
    writer.add_scalar("Accuracy/training/age_accuracy", tr_age_acc, 0)
    writer.add_scalar("Accuracy/training/binary_accuracy_avg", tr_rest_acc, 0)
    writer.add_scalar("Accuracy/validation/upperbody_accuracy", val_up_acc, 0)
    writer.add_scalar("Accuracy/validation/lowerbody_accuracy", val_down_acc, 0)
    writer.add_scalar("Accuracy/validation/age_accuracy", val_age_acc, 0)
    writer.add_scalar("Accuracy/validation/binary_accuracy_avg", val_rest_acc, 0)
    
    for e in range(epochs):
        training_loss, tr_up_acc, tr_down_acc, tr_age_acc, tr_rest_acc = train(net, tr_loader, optimizer)
        validation_loss, val_up_acc, val_down_acc, val_age_acc, val_rest_acc = test(net, val_loader)

        print("=" * 50, "\n")
        print("Values After Training Epoch {}".format(e+1))
        print("\tTraining Loss: {:.4f}".format(training_loss))
        print("\tTrain Up Accuracy: {:.2f}".format(tr_up_acc))
        print("\tTrain Down Accuracy: {:.2f}".format(tr_down_acc))
        print("\tTrain Age Accuracy: {:.2f}".format(tr_age_acc))
        print("\tTrain Rest Accuracy: {:.2f}\n".format(tr_rest_acc))
        print("\tValidation Loss: {:.4f}".format(validation_loss))
        print("\tValidation Up Accuracy: {:.2f}".format(val_up_acc))
        print("\tValidation Down Accuracy: {:.2f}".format(val_down_acc))
        print("\tValidation Age Accuracy: {:.2f}".format(val_age_acc))
        print("\tValidation Rest Accuracy: {:.2f}\n".format(val_rest_acc))
        writer.add_scalar("Loss/training", training_loss, e+1)
        writer.add_scalar("Loss/validation", validation_loss, e+1)
        writer.add_scalar("Accuracy/training/upperbody_accuracy", tr_up_acc, e+1)
        writer.add_scalar("Accuracy/training/lowerbody_accuracy", tr_down_acc, e+1)
        writer.add_scalar("Accuracy/training/age_accuracy", tr_age_acc, e+1)
        writer.add_scalar("Accuracy/training/binary_accuracy_avg", tr_rest_acc, e+1)
        writer.add_scalar("Accuracy/validation/upperbody_accuracy", val_up_acc, e+1)
        writer.add_scalar("Accuracy/validation/lowerbody_accuracy", val_down_acc, e+1)
        writer.add_scalar("Accuracy/validation/age_accuracy", val_age_acc, e+1)
        writer.add_scalar("Accuracy/validation/binary_accuracy_avg", val_rest_acc, e+1)

    torch.save(net.state_dict(), save_dir)
    return net


