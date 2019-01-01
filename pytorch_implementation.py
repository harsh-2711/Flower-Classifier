%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


########## For Google Colab ################

# http://pytorch.org/
from os.path import exists
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())
cuda_output = !ldconfig -p|grep cudart.so|sed -e 's/.*\.\([0-9]*\)\.\([0-9]*\)$/cu\1\2/'
accelerator = cuda_output[0] if exists('/dev/nvidia0') else 'cpu'

!pip install -q http://download.pytorch.org/whl/{accelerator}/torch-0.4.1-{platform}-linux_x86_64.whl torchvision
import torch

!pip install -U -q PyDrive

import tensorflow as tf
import timeit

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# Authenticate and create the PyDrive client.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

from os import path
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())

accelerator = 'cu80' if path.exists('/opt/bin/nvidia-smi') else 'cpu'

!pip install -q http://download.pytorch.org/whl/{accelerator}/torch-0.4.1-{platform}-linux_x86_64.whl torchvision
import torch
print(torch.__version__)
print(torch.cuda.is_available())

from google.colab import drive
drive.mount('/content/drive')

############## Till here ################

# Imports here
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import models
from torch.optim.lr_scheduler import StepLR

import cv2
import helper

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline

# convert data to a normalized torch.FloatTensor
train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

valid_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder('drive/My Drive/AI/flower_data/train', transform=train_transforms) # Change path
valid_data = datasets.ImageFolder('drive/My Drive/AI/flower_data/valid', transform=valid_transforms) # Change path

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainLoader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True) 
validLoader = torch.utils.data.DataLoader(valid_data, batch_size=10)

############ Not for Colab ##############

images,labels = next(iter(trainLoader))
img = images[0] / 2 + 0.485  # unnormalize
plt.imshow(np.transpose(img, (1, 2, 0)))
print(labels[0])
print(images.shape)

############ Till here ##################

model = models.densenet201(pretrained=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # For Colab

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1920, 1000)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(1000, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.classifier = classifier

criterion = nn.CrossEntropyLoss()

optimizer = optim.RMSprop(model.classifier.parameters(), lr=0.001)

# New variable
scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

model.to(device) # For Colab


############ Main Algo ################

n_epochs = 200 # Initial 30
steps = 0
every_step = 5
train_counter = 0
validation_counter = 0

valid_loss_min = np.Inf

for epoch in range(1, n_epochs+1):

    train_loss = 0.0
    valid_loss = 0.0
    
    steps += 1
    
    ###################
    # train the model #
    ###################
    model.train()
    for data, target in trainLoader:

      train_counter += 1
      print("Currect Training Batch: ", train_counter)

      data,target = data.to(device),target.to(device) # For Colab

      optimizer.zero_grad()

      output = model(data)
      loss = criterion(output, target)
      loss.backward()
      #optimizer.step()
      scheduler.step()

      train_loss += loss.item()
        
    ######################    
    # validate the model #
    ######################

    if steps % every_step == 0:
      model.eval()
      accuracy = 0
      for data, target in validLoader:

          validation_counter += 1
          print("Current Validating Batch: ", validation_counter)

          data,target = data.to(device),target.to(device) # For Colab

          output = model(data)
          loss = criterion(output, target)
          valid_loss += loss.item()

          # Calculate accuracy
          ps = torch.exp(output)
          top_p, top_class = ps.topk(1, dim=1)
          equals = top_class == target.view(*top_class.shape)
          accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

      # calculating average losses
      train_loss = train_loss/len(trainLoader.dataset)
      valid_loss = valid_loss/len(validLoader.dataset)
      final_accuracy = accuracy/len(validLoader.dataset)

      print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tAccuracy: {:.6f} '.format(
          epoch, train_loss, valid_loss, final_accuracy))

      # save model if validation loss has decreased
      if valid_loss <= valid_loss_min:
          print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
          valid_loss_min,
          valid_loss))
          torch.save(model.state_dict(), 'flower_102_model.pt')
          valid_loss_min = valid_loss


