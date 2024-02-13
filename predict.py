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
    parser.add_argument('--image_path', type = str, help = 'Provide path to image directory')
    parser.add_argument('--load_checkpoint', type = str, help = 'Load checkpoint')
    parser.add_argument('--top_k', type = int, help = 'Top K', default = 5)
    parser.add_argument('--category_names', type = float, help = 'Map categories to names', default = 'cat_to_name.json')
    parser.add_argument('--gpu', action = 'store_true', help = 'Use GPU if available')
    
    return parser.parse_args()


# function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
  checkpoint = torch.load('checkpoint.pth')
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
  optimizer.load_state_dict(checkpoint['optimizer'])
  model.class_to_idx = checkpoint['class_to_idx']
  epoch = checkpoint['epoch']
  return model

# function for image preprocessing
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    # opens image
    img = PIL.Image.open(image)

    # original dimensions
    width, height = img.size

    # resize image to shortest size 256 pixels, keeping the aspect ratio
    if width < height:
        size=[256, 10000]
    else: 
        size=[10000, 256]
        
    img.thumbnail(size)

    # crop to the center 224x224 portion of image
    center = width/4, height/4
    left, bottom, right, top = center[0]-(244/2), center[1]-(244/2), center[0]+(244/2), center[1]+(244/2)
    img = img.crop((left, bottom, right, top))

    # convert image to a numpy array
    np_img = np.array(img) / 255

    # normalize image
    means = [0.485, 0.456, 0.406]
    sd = [0.229, 0.224, 0.225]
    np_img = (np_img - means) / sd

    # reorder dimensions
    np_img = np_img.transpose(2,0,1)

    return np_img


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # code to predict the class from an image file
    # set device and model to evaluate
    model.to('cuda')
    model.eval()
    
    # preprocess image
    img = process_image(image_path)
    
    # convert to tensor
    img = torch.from_numpy(img).type(torch.FloatTensor) 
    
    # batch size of 1
    img = img.unsqueeze(0)
    
    with torch.inference_mode():
        # move model to same device as image tensor
        output = model.forward(img.cuda())
    
    # convert log probabilities to probabilities
    probability = torch.exp(output).data
    
    # find the top k probabilities and class indices
    top_prob, top_indices = probability.topk(topk)
    
    top_prob = top_prob.cpu().numpy()
    top_indices = top_indices.cpu().numpy()
    
    # convert indices to classes labels
    idx_to_class = {idx: class_ for class_, idx in model.class_to_idx.items()}
    top_class = [idx_to_class[idx] for idx in top_indices[0]]
    
    return top_prob[0].tolist(), top_class

    # print prediction of flower name and probability
    if args.image_path:
        print('Predictions and probabilities:', list(zip(top_class, top_prob)))
    
    return top_class, top_prob
