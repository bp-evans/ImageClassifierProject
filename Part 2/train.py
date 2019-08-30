import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torchvision.models as models
import torchvision
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import train_tools
import argparse
# Setting up all of the parser information
parser = argparse.ArgumentParser(description='train.py')
parser.add_argument('data_dir', nargs="*", action="store", default="./flowers")
parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
parser.add_argument('--arch', dest="arch", action="store", default="alexnet")
parser.add_argument('--learning_rate', dest="learning_rate", action="store", default="0.001")
parser.add_argument('--hidden_units', dest="hidden_units", action="store", default="512")
parser.add_argument('--epochs', dest="epochs", action="store", default=1)
parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")

# Getting info from the parser
parsed = parser.parse_args()
loc = parsed.data_dir
saved = parsed.save_dir
arch = parsed.arch
learn_rate = float(parsed.learning_rate)
hidden = int(parsed.hidden_units)
epochs = int(parsed.epochs)
gpu = parsed.gpu

# Printing out user defined inputs
print("Data Location: " + loc)
print("Save Location: " + saved)
print("Architecture: " + arch)
print("Learning Rate: " + str(learn_rate))
print("Hidden Units: " + str(hidden))
print("Epochs: " + str(epochs))
print("GPU: " + gpu)

# Load the data
trainloader, validloader, testloader, train_dir, valid_dir, test_dir = train_tools.load_data(loc)

# Actually build the model
model, optimizer, criterion, device = train_tools.model_build(arch, learn_rate, hidden, gpu)
# Display the model
print(model)

# Train the newley created model
train_tools.model_train(model, optimizer, criterion, trainloader, validloader, epochs, device)

# Save the trained model to a checkpoint
train_tools.save_checkpoint(saved, arch, hidden, learn_rate, epochs, model, optimizer, hidden)
print("---------------------------------------------------------")
print("Model has been trained and saved. Ready for predictions.")
