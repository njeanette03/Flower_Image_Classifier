# import libraries
# import pytorch library
import torch
# import nn, optim, tensor modules
from torch import nn, optim, tensor
# import functional module
import torch.nn.functional as F
# import numpy library
import numpy as np
# import pytorch vision library
import torchvision
# import transforms, datasets, and models modules
from torchvision import transforms, datasets, models
# import json library
import json
# import matplotlib library
import matplotlib.pyplot as plt
# import PIL
import PIL
# import image module
from PIL import Image
# import seaborn
import seaborn as sns
# import argparse
import argparse


# function to define and get arguments
def get_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type = str, help = 'Provide data directory')
    parser.add_argument('--checkpoint', type = str, help = 'Provide checkpoint', default = 'checkpoint.pth')
    parser.add_argument('--arch', type = str, help = 'Model architecture', default = 'vgg16')
    parser.add_argument('--learn_rate', type = float, help = 'Model learning rate')
    parser.add_argument('--epochs', type = int, help = 'Number of epochs')
    parser.add_argument('--hidden_units', type = int, help = 'Set hidden unit')
    parser.add_argument('--gpu', action = 'store_true', help = 'Use GPU if available')
    
    return parser.parse_args()


# function to define and get transformations
def get_transforms(train_dir, valid_dir, test_dir):
    data_transforms = {
        # transforms for training set
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(size = 224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225]),
        ]),
        
        # transforms for validation set
        'valid': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(size = 224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225]),
        ]),
        
        # transforms for testing set
        'test': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(size = 224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225]),
        ])
    }
    
    # TODO: Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform = data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform = data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform = data_transforms['test'])
    }
    
    # define the dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size = 64, shuffle = True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size = 64, shuffle = True),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size = 64, shuffle = True)
    }
    
    return image_datasets, dataloaders


# function to load, train, and save network
def train_model(arch, model, device, hidden_units, dataloaders):
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     
    # load  pre-trained VGG network
    if arch == 'vgg16':
        model = models.vgg16(weights = torchvision.models.VGG16_Weights.DEFAULT)
        
        # build network
        # define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
        # freeze parameters
        for param in model.parameters():
            param.requires_grad = False
           
        if hidden_units:
            classifier = nn.Sequential(nn.Linear(25088,2048),
                                       nn.ReLU(),
                                       nn.Linear(2048, 256),
                                       nn.ReLU(),
                                       nn.Linear(256, 102),
                                       nn.LogSoftmax(dim = 1))
           
        else:
            classifier = nn.Sequential(nn.Linear(25088,2048),
                                       nn.ReLU(),
                                       nn.Linear(2048, 256),
                                       nn.ReLU(),
                                       nn.Linear(256, 102),
                                       nn.LogSoftmax(dim = 1))
       
    # attach new classifier to model
    model.classifier = classifier
    
    # train network# set model to device
    model = model.to(device)
    
    # track the loss and accuracy on the validation set
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = 0.001)
    
    # set training parameters
    epochs = 5
    steps = 0
    running_loss = 0
    print_every = 10
    
    for epoch in range(epochs):
        training_loss = 0
        validation_loss = 0
        accuracy = 0
        
        # iterate training data
        for inputs, labels in dataloaders['train']:
            steps += 1
            # move input and label tensors to device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # forward pass
            outputs = model.forward(inputs)
            
            # calculate loss
            loss = criterion(outputs, labels)
            
            # zeroing out accumlated gradients
            optimizer.zero_grad()
            
            # backpropgation
            loss.backward()
            
            # optimization step
            optimizer.step()
            
            # update running loss
            running_loss += loss.item()
            
            
            # iterate print_every and evaluate
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.inference_mode():
                    for inputs, labels in dataloaders['valid']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model.forward(inputs)
                        batch_loss = criterion(outputs, labels)
                        valid_loss += batch_loss.item()
                        
                        # calculate accuracy
                        ps = torch.exp(outputs)
                        top_p, top_class = ps.topk(1, dim = 1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()         
                   
                print(f"Epoch {epoch + 1} / {epochs}.. "
                      f"Training loss: {running_loss/len(dataloaders['train']):.3f}.. "
                      f"Validation loss: {valid_loss/len(dataloaders['valid']):.3f}.. "
                      f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}")
                
                running_loss = 0
                model.train()
               
    # save checkpoint
    # map classes of indices to training dataset
    model.class_to_idx = image_datasets['train'].class_to_idx
    
    checkpoint = {'model' : model,
                  'epoch' : epochs,
                  'model_state_dict' : model.state_dict(),
                  'class_to_idx' : model.class_to_idx,
                  'optimizer' : optimizer.state_dict()
                 }
    
    # save serialized object to disk
    torch.save(checkpoint, 'checkpoint.pth')