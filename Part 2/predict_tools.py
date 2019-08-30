import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torchvision.models as models
import torchvision
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from PIL import Image

def load_checkpoint(path):
    checkpoint = torch.load(path)
    if checkpoint['arch'] == 'alexnet':
        input_units = 9216
        model = models.alexnet(pretrained=True)
    elif checkpoint['arch'] == 'vgg13':
        input_units = 25088
        model = models.vgg13(pretrained=True)
    elif checkpoint['arch'] == 'densenet121':
        input_units = 1024
        model = models.densenet121(pretrained=True)
    else:
        input_units = 9216
        model = models.alexnet(pretrained=True)
        
    model.classifier = nn.Sequential(nn.Linear(input_units, checkpoint['hidden_units']),
                                     nn.ReLU(),
                                     nn.Dropout(0.05),
                                     nn.Linear(checkpoint['hidden_units'], 100),
                                     nn.ReLU(),
                                     nn.Dropout(0.05),
                                     nn.Linear(100, 80),
                                     nn.ReLU(),
                                     nn.Dropout(0.05),
                                     nn.Linear(80, 102),
                                     nn.LogSoftmax(dim=1))
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

def process_image(img_path):
    im = Image.open(img_path)
    
    changes = transforms.Compose([transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485,0.456,0.406],
                                                       std=[0.229, 0.224, 0.225])])
    
    img_final_tensor = changes(im)
    
    return img_final_tensor
    
def predict_image_class(img_path, model, topk, power, cat_to_name):
    
    if torch.cuda.is_available() and power == 'gpu':
        device = 'cuda'
    else:
        device = 'cpu'
     
    model.to(device);
    
    image = process_image(img_path)
    image = image.unsqueeze(0)
    image = image.float()
    image = image.to(device)
    
    with torch.no_grad():
        log_probs = model.forward(image)
        ps = torch.exp(log_probs)
        probs, classes = ps.topk(topk, dim=1)
        
    return probs, classes
    