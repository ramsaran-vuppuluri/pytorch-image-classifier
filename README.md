# Image Classifier using PyTorch

### Introduction

In this project, we'll train an image classifier to recognize different species of flowers.  We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, and you can see a few examples below.

The project is broken down into multiple steps:

* Load and preprocess the image dataset
* Train the image classifier on your dataset
* Use the trained classifier to predict image content

We are going to use PyTorch & torchvision Deep Learning frameworks to load, train and predict the flower species.

### Project Structure:
    |
    |___ cat_to_name.json                           <- Dictionary mapping the integer encoded categories to the actual names of the flowers.
    |
    |___ flower_species_classifier_densenet121.pth  <- Pretrained densenet121 network is stored for predictions.
    |
    |___ Image Classifier Project.html              <- Jupyter notebook HTML snapshot.
    |
    |___ Image Classifier Project.ipynb             <- Jupyter notebook with source code.
    |
    |___ LICENSE                                    <- MIT distribution license.
    |
    |___ predict.py                                 <- Python script to predict flower species based on pretrained Deep Learning network.
    |
    |___ train.py                                   <- Python script to train deep learning network to training data.
    |
    |___ workspace-utils.py                         <- Python script to maintain GPU session for long running training sessions.
    
### Instructions:

1. Run the following command in the project root directory to train new network.
    
    python3 train.py [parameters]
    
    Following are the available parameters
    
    | Parameter | Default Value |Valid values  | Help  |
    |-----------|---------------|--------------|-------| 
    |--data_directory|flowers | | training data directory|
    |--arch| |alexnet, vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn,vgg19, vgg19_bn, resnet18, resnet34, resnet50, resnet101,densenet161, densenet201, inception_v3|Base pre trained network|
    |--drop_out|0.0| |Proportion of drop out in new classifier layers to address overfitting|
    |--learning_rate|0.001| | Learning rate for gradient descent|
    |--num_labels|102| | Learning rate for gradient descent|
    |--hidden_units|512| | Number of starting hidden units|
    |--epochs|5| | Number of iterations|
    |--gpu| | 'cpu', 'cuda'| Run mode|
    
2. Run the following command in the project root directory to used existing network to predict values

    python3 predict.py [parameters]
    
    | Parameter | Default Value |Valid values  | Help  |
    |-----------|---------------|--------------|-------|
    |--data_directory|flowers| |training data directory|
    |--path_to_image|/home/workspace/saved_model/| | Path to saved model|
    |--top_k| 1| | Number of highest values|
    |--category_names| cat_to_name.json | | Category name map|
    |--gpu| | 'cpu', 'cuda'| Run mode|
    
### Libraries

* matplotlib
* numpy
* pandas
* json
* PIL
* PyTorch
* torchvision 