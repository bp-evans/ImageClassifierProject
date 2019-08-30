# A collection of functions to be used with 
import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torchvision.models as models
import torchvision
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import json
from workspace_utils import active_session

def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transform = transforms.Compose([transforms.RandomRotation(15),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomVerticalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    validation_transform = transforms.Compose([transforms.Resize(255),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])

    testing_transform = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform = train_transform)
    valid_dataset = datasets.ImageFolder(valid_dir, transform = validation_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform = testing_transform)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size = 32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size = 32, shuffle=True)
   
    return trainloader, validloader, testloader, train_dir, valid_dir, test_dir
        
def model_build(model_selection, learning_rate, hidden_units, power):
    input_units = 0
    # determine the pretrined unit type and set corresponding number of input units
    if model_selection == "alexnet":
        model = models.alexnet(pretrained=True)
        input_units = 9216
    elif model_selection == "vgg13":
        model = models.vgg16(pretrained=True)
        input_units = 25088
    elif model_selection == "densenet121":
        model = models.densenet121(pretrained=True)
        input_units = 1024
    else:
        model = models.alexnet(pretrained=True)
        input_units = 9216
    
    # Now the custom classifier for the model is built
    model.classifier = nn.Sequential(nn.Linear(input_units, hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.05),
                                     nn.Linear(hidden_units, 100),
                                     nn.ReLU(),
                                     nn.Dropout(0.05),
                                     nn.Linear(100, 80),
                                     nn.ReLU(),
                                     nn.Dropout(0.05),
                                     nn.Linear(80, 102),
                                     nn.LogSoftmax(dim=1))
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)
    
    if torch.cuda.is_available() and power == 'gpu':
        device = 'cuda'
    else:
        device = 'cpu'
     
    model.to(device);
    
    return model, optimizer, criterion, device

def model_train(model, optimizer, criterion, trainloader, validloader, epochs, device):
    with active_session():
        for e in range(epochs):
            accuracy = 0
            train_loss = 0
            for images, labels in trainloader:
                images, labels = images.to(device), labels.to(device)
                # Making sure gradients are reset on each run
                optimizer.zero_grad()
                # Running through the training pass
                nn_logProbs = model.forward(images)
                loss = criterion(nn_logProbs, labels)
                # The backprop
                loss.backward()
                # Stepping features in right dir
                optimizer.step()
                train_loss += loss.item()
            else:
                test_loss = 0
                with torch.no_grad():
                    model.eval()
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)
                        log_ps = model.forward(images)
                        batch_loss = criterion(log_ps, labels)
                        test_loss += batch_loss.item()
                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))
                print(f"Epoch: {e + 1}")
                print(f"Train Loss: {train_loss/len(trainloader):.3f}")
                print(f"Validation Loss: {test_loss/len(validloader):.3f}")
                print(f"Validation Accuracy: {accuracy/len(validloader):.3f}")
                print("-------------")
                model.train()
            
def save_checkpoint(saved, arch, hidden, learnrate, epoch, model, optimizer, hidden_units):
    model.cpu
    checkpoint = {'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'arch': arch,
                  'learn_rate': learnrate,
                  'hidden_units': hidden_units}
    torch.save(checkpoint, saved)
    
    
    
                                    
                                     
        
    
        
    
    