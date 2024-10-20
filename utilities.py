#Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import argparse

#Convolutional neural network
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # First convolutional layer: 1 input channel (grayscale), 32 output channels, 5x5 kernel
        self.layer1 = nn.Sequential(
            #https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            #https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
            nn.BatchNorm2d(32),  # Batch normalization for faster convergence
            # https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
            nn.ReLU(),  # Activation function
            # https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
            nn.MaxPool2d(kernel_size=2, stride=2))  # Max pooling layer

        # Second convolutional layer: 32 input channels, 64 output channels, 5x5 kernel
        self.layer2 = nn.Sequential(
            #https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            #https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
            nn.BatchNorm2d(64),  # Batch normalization
            # https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
            nn.ReLU(),  # Activation function
            # https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
            nn.MaxPool2d(kernel_size=2, stride=2))  # Max pooling layer

        # Fully connected layer: input size 7*7*64, output size 1000
        self.fc1 = nn.Linear(7*7*64, 1000)
        # Fully connected layer: input size 1000, output size 10 (number of classes)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        # Forward pass through the first convolutional layer
        out = self.layer1(x)
        # Forward pass through the second convolutional layer
        out = self.layer2(out)
        # Flatten the output for the fully connected layer
        out = out.reshape(out.size(0), -1) # also look at https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html
        # Forward pass through the first fully connected layer
        out = self.fc1(out)
        # Forward pass through the second fully connected layer
        out = self.fc2(out)
        return out






#Plot Function
def plot(loss_list, acc_train, acc_test, save_path='./figure.png'):
    # Plot the loss & accuracy curves
    plt.figure(figsize=(10, 4))

    # Plot the training loss over iterations
    plt.subplot(1, 2, 1)
    plt.plot(loss_list)
    plt.xlabel('Iteration')

    plt.savefig(save_path)

    

    '''plt.ylabel('Loss')
    plt.title(f"Training Loss")

    # Plot the train & test accuracies over epochs
    plt.subplot(1, 2, 2)
    plt.plot(acc_train)
    plt.plot(acc_test)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f"Model Accuracy")
    plt.legend(['Training', 'Testing'])'''

# Argument parser to receive command-line arguments
def myargs():
    parser = argparse.ArgumentParser(description="MNIST Classification with PyTorch & GPU",
                                     epilog='** Version 1.0 **')
    parser.add_argument('-e', 
                        '--num_epochs', 
                        type=int, 
                        default=10, 
                        help='Number of epochs for training')

    parser.add_argument('-b', 
                        '--batch_size', 
                        type=int, 
                        default=1000,
                         help='Batch size for training')

    parser.add_argument('-l', 
                        '--learning_rate', 
                        type=float, 
                        default=0.001, 
                        help='Learning rate for training')
                        
    return parser.parse_args()


