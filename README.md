# Image_Classifier

For my final capstone for the AI programming in Python nanodegree at Udacity, I built and trained an image classifier to recognize different species of flowers on a Flower Dataset and then predicted new flower images. The Dataset contains 102 flower categories.

I developed the code in Python for an image classifier built with PyTorch, then converted into a command line applications: train.py, predict.py. The code is first implement in a Jupyter notebook.

## Walk through: The classifier

In Image_classifier_project.ipynb, the VGG network from torchvision.models pretrained models was used. It was loaded as a pre-trained network, based on which defined a new, untrained feed-forward network as a classifier, using ReLU activations and dropout. I then trained the classifier layers using backpropagation using the pre-trained network to get the features. The loss and accuracy on the validation set were tracked to determine the best hyperparameters.

### Preparation of Tensor data and label mappiing
To make sure my neural network trained properly, I organized the training images in folders names as their class name, training, testing, and validation folders, as follows:

```python
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
```


Note that the image folders names are not the actual names of the flowers, but rather numbers. Accordingly, Udacity provided a .json file, `cat_to_name.json`, which contained the mapping between flower names and folder labels, which my model will use to predict and return an index between 0 and 101, which corresponds to one of the folder labels (1-102). In turn, the folder labels (1-102) correspond to flower names that the .json file maps.

I then adapted my images to work with the pre-trained networks of the torchvision library. First, I defined transformations on the image data which included resizing these to 256x256 pixels. Subsequently, I created Torch Dataset objects using ImageFolder. This is done as follows:

```python
# loads the datasets using ImageFolder
image_datasets = {
    'train': datasets.ImageFolder(train_dir, transform = data_transforms['train']),
    'valid': datasets.ImageFolder(valid_dir, transform = data_transforms['valid']),
    'test': datasets.ImageFolder(test_dir, transform = data_transforms['test'])
}
```

Then, I created DataLoader objects to make sure I could work on my data. 

### Upload of the pre-trained model and preparation of the classifier

The fully-connected layer that I then trained on the flower images was as follows:

```python
classifier = nn.Sequential(nn.Linear(25088,2048),
                           nn.ReLU(),
                           nn.Linear(2048, 256),
                           nn.ReLU(),
                           nn.Linear(256, 102),
                           nn.LogSoftmax(dim = 1))
```

I chose to work with a highly accurate Convolutional Network, VGG16.

### Training and testing the network

To train the network, I set the hyperparameters for the training (i.e. epochs, learning rate, etc.). Initially, I wanted to choose 5 epochs for better accuracy but due to being abysmally slow and proved impossible to train, with each epoch being about 3-5 hours on Udacity's GPU server, I opted to work on Google Colab and chose to work with 5 epochs to avoid overfitting. The code loops through each epoch, trains 10 batches at at time, and tests the model's progress on the validation data. At the end, the training and validation metrics are printed.

The file developed on the Udacity portal is `Image_classifier_project.ipynb`. I copied below the training results for the last epoch:

```python
Epoch 5 / 5.. Training loss: 0.068.. Validation loss: 0.769.. Validation accuracy: 0.790
Epoch 5 / 5.. Training loss: 0.067.. Validation loss: 0.891.. Validation accuracy: 0.771
Epoch 5 / 5.. Training loss: 0.072.. Validation loss: 0.865.. Validation accuracy: 0.782
Epoch 5 / 5.. Training loss: 0.063.. Validation loss: 0.825.. Validation accuracy: 0.784
Epoch 5 / 5.. Training loss: 0.069.. Validation loss: 0.835.. Validation accuracy: 0.762
Epoch 5 / 5.. Training loss: 0.067.. Validation loss: 0.883.. Validation accuracy: 0.783
Epoch 5 / 5.. Training loss: 0.074.. Validation loss: 0.814.. Validation accuracy: 0.783
Epoch 5 / 5.. Training loss: 0.080.. Validation loss: 0.897.. Validation accuracy: 0.750
Epoch 5 / 5.. Training loss: 0.060.. Validation loss: 0.830.. Validation accuracy: 0.775
Epoch 5 / 5.. Training loss: 0.069.. Validation loss: 0.846.. Validation accuracy: 0.780
```

I set aside test data that the model had never been exposed to in order to see how it performed. The aim was to score about 70%. My results was 74.84%. Some thought to improve accuracy for the future is to increase epoch size and to change parameters to test for improvement of accuracy.


I then created two functions to save the checkpoint in a .pth file, and load it when necessary.

### Command line applications train.py and predict.py

I developed the code in Python for an image classifier built with PyTorch, then converted into a command line applications: train.py, predict.py. 

Following arguments mandatory or optional for train.py 

1.	'--data_dir', type = str, help = 'Provide data directory'
2.	'--checkpoint', type = str, help = 'Provide checkpoint', default = 'checkpoint.pth'
3.	'--arch', type = str, help = 'Model architecture', default = 'vgg16'
4.	'--learn_rate', type = float, help = 'Model learning rate'
5.	'--epochs', type = int, help = 'Number of epochs'
6.	'--hidden_units', type = int, help = 'Set hidden unit'
7.	'--gpu', action = 'store_true', help = 'Use GPU if available'

Following arguments mandatory or optional for predict.py

1.	'--image_path', type = str, help = 'Provide path to image directory'
2.	'--load_checkpoint', type = str, help = 'Load checkpoint'
3.	'--top_k', type = int, help = 'Top K', default = 5
4.	'--category_names', type = float, help = 'Map categories to names', default = 'cat_to_name.json'
5.	'--gpu', action = 'store_true', help = 'Use GPU if available'
