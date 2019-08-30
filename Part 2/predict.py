import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torchvision.models as models
import torchvision
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import predict_tools
import argparse
import json

# Set up the parser 
parser = argparse.ArgumentParser(description="predict.py")
parser.add_argument('img_path', nargs="*", default = "./flowers/test/1/image_06743.jpg", action="store")
parser.add_argument('--checkpoint_loc', dest = 'checkpoint', default = "./checkpoint.pth", action="store")
parser.add_argument('--top_k', dest = 'top_k', default = 5, action="store")
parser.add_argument('--categories', dest='cats', default = "cat_to_name.json", action="store")
parser.add_argument('--gpu', dest="gpu", default = "gpu", action="store")

# Set variables to argument fata
parsed = parser.parse_args()
img_path = parsed.img_path
load_loc = parsed.checkpoint
topk = int(parsed.top_k)
cats = parsed.cats
power = parsed.gpu

# Print user arguments
print('Image Path: ' + str(img_path))
print('Checkpoint Location: ' + str(load_loc))
print('Topk Shown: ' + str(topk))
print('Categories File: ' + str(cats))
print('Power Mode: ' + str(power))

# Get dic with index of classes to label names for predictions
with open(str(cats), 'r') as json_file:
    cat_to_name = json.load(json_file)

# Loading the checkpoint from the model
model = predict_tools.load_checkpoint(load_loc)
print(model)

# Get prediction pribabilities and classes
probs, classes = predict_tools.predict_image_class(img_path, model, topk, power, cat_to_name)
probs = np.array(probs)
classes = np.array(classes)
classes = classes + 1

# Display topk probs and classes
print(' ')
print('---------------------')
print('TopK Shown: ' + str(topk))

for i in range(topk):
    print('Top k ' + str(i + 1) + ': ')
    print(f"Probability: {probs[0][i]:.3f}")
    print("Flower: " + str(cat_to_name[str(classes[0][i])]))

# Display the final predcition of the model
print(' ')    
print('********************')
print('********************')
print("Model Prediction = " + str(cat_to_name[str(classes[0][0])]))

    
