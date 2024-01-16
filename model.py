import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
from torch import optim
from torch import nn
from torch import tensor
import torchvision.models as models
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.autograd import Variable
import argparse
from collections import OrderedDict
import PIL
from PIL import Image
import seaborn as sns
import json
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.vgg16(pretrained=True)

# restricting actual feature detectors(parameters) of pretrained model to be updated
for param in model.parameters():
        param.requires_grad = False

# defining custom model with output layer having 102 neurons as we have 102 categories of flowers(prediction)         
my_classifier = nn.Sequential(nn.Linear(25088, 1588),
                                 nn.ReLU(),
                                 nn.Linear(1588, 488),
                                 nn.ReLU(),                                 
                                 nn.Linear(488, 102), 
                                 nn.LogSoftmax(dim=1))
# Freeze all parameters in the pre-trained model
for param in model.parameters():
    param.reuires_grad = False

# Import OrderedDict to define a custom classifier
from collections import OrderedDict

# Replace the default classifier in the VGG16 model with the custom classifier    
model.classifier = my_classifier
    
#defining criterion and optimizer
# Define the loss function (criterion) as Negative Log Likelihood Loss
criterion = nn.NLLLoss()
# Define the optimizer as Adam, which will update the custom classifier's parameters
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# Check if a CUDA-compatible GPU is available and set the device accordingly
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f'{device} is in use right now.\n')



#Save the checkpoint 
model.class_to_idx = train_data.class_to_idx
param_dict = {'architecture': model,
             'classifier': model.classifier,
             'class_to_idx': model.class_to_idx,
             'state_dict': model.state_dict()}

torch.save(param_dict, 'my_model.pth')



def load_checkpoint(model_path='my_model.pth'):
    # This function allows you to resume training, fine-tune, or make predictions with a pre-trained model.
    # Load the saved file
    checkpoint = torch.load("my_model.pth")
    
    # Download pretrained model
    model = models.vgg16(pretrained=True)
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters(): 
        param.requires_grad = False
    
    # Load stuff from checkpoint
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])

    
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    image = Image.open(image)
    # Get original dimensions
    orig_width, orig_height = image.size

    # resizes to crop shortest side to 256
    resize_size=[256, 256]
    
        
    image.thumbnail(size=resize_size)

    # Find pixels to crop on to create 224x224 image
    center = orig_width/4, orig_height/4
    left, top, right, bottom = center[0]-(244/2), center[1]-(244/2), center[0]+(244/2), center[1]+(244/2)
    test_image = image.crop((left, top, right, bottom))

    # Convert to numpy
    np_image = np.array(test_image)/255 # / 255 because imshow() expected integers are (0:1)

    # Normalize each color channel
    normalise_means = [0.485, 0.456, 0.406]
    normalise_std = [0.229, 0.224, 0.225]
    np_image = (np_image-normalise_means)/normalise_std
        
    # The color set to the first channel
    np_image = np_image.transpose(2, 0, 1)
    
    return np_image



# Make predictions using the deep learning model
def predict_class(image, model, idx_mapping, topk, device):
    pre_processed_image = preprocess_image(image).to(device)
    model.to(device)
    model.eval()

    with torch.no_grad():
        log_ps = model(pre_processed_image)
        ps = torch.exp(log_ps)
        top_ps, top_idx = ps.topk(topk, dim=1)
        list_ps = top_ps.tolist()[0]
        list_idx = top_idx.tolist()[0]
        classes = [idx_mapping[x] for x in list_idx]
    
    model.train()
    return list_ps, classes