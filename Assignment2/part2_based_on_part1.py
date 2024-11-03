# -*- coding: utf-8 -*-
"""Assignment_2_Part_1_Cifar10_vp1.ipynb

Purpose: Implement image classsification nn the cifar10
dataset using a pytorch implementation of a CNN architecture (LeNet5)

Pseudocode:
1) Set Pytorch metada
- seed
- tensorboard output (logging)
- whether to transfer to gpu (cuda)

2) Import the data
- download the data
- create the pytorch datasets
    scaling
- create pytorch dataloaders
    transforms
    batch size

3) Define the model architecture, loss and optimizer

4) Define Test and Training loop
    - Train:
        a. get next batch
        b. forward pass through model
        c. calculate loss
        d. backward pass from loss (calculates the gradient for each parameter)
        e. optimizer: performs weight updates
        f. Calculate accuracy, other stats
    - Test:
        a. Calculate loss, accuracy, other stats

5) Perform Training over multiple epochs:
    Each epoch:
    - call train loop
    - call test loop

"""

# Step 1: Pytorch and Training Metadata

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from pathlib import Path

# hyperparameters
batch_size = 128
epochs = 10
lr = 0.003
warmup_epochs = 3  # Number of warmup epochs
try_cuda = True
seed = 1000

# Architecture
num_classes = 10

# other parameters
logging_interval = 10  # how many batches to wait before logging
logging_dir = None
grayscale = True

# 1) setting up the logging

datetime_str = datetime.now().strftime('%b%d_%H-%M-%S')

if logging_dir is None:
    runs_dir = Path("./") / Path(f"runs/")
    runs_dir.mkdir(exist_ok=True)

    logging_dir = runs_dir / Path(f"{datetime_str}")

    logging_dir.mkdir(exist_ok=True)
    logging_dir = str(logging_dir.absolute())

writer = SummaryWriter(log_dir=logging_dir)

# deciding whether to send to the cpu or not if available
if torch.cuda.is_available() and try_cuda:
    cuda = True
    torch.cuda.manual_seed(seed)
else:
    cuda = False
    torch.manual_seed(seed)

"""# Step 2: Data Setup"""

# Transformations to apply to the data
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1) if grayscale else transforms.Lambda(lambda x: x),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Normalize((0.5,), (0.5,))
])

# Loading the dataset from the CIFAR10 folder structure
train_dataset = ImageFolder(root='./Assignment2/CIFAR10/Train', transform=transform)
test_dataset = ImageFolder(root='./Assignment2/CIFAR10/Test', transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

def check_data_loader_dim(loader):
    # Checking the dataset
    for images, labels in loader:
        print('Image batch dimensions:', images.shape)
        print('Image label dimensions:', labels.shape)
        break

check_data_loader_dim(train_loader)
check_data_loader_dim(test_loader)

"""# 3) Creating the Model"""

layer_1_n_filters = 32
layer_2_n_filters = 64
fc_1_n_nodes = 1024
kernel_size = 5
verbose = False

# calculating the side length of the final activation maps
input_size = 28  
final_length = input_size // (2 * 2)  

if verbose:
    print(f"final_length = {final_length}")


class LeNet5(nn.Module):

    def __init__(self, num_classes, grayscale=False):
        super(LeNet5, self).__init__()

        self.grayscale = grayscale
        self.num_classes = num_classes

        if self.grayscale:
            in_channels = 1
        else:
            in_channels = 3

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, layer_1_n_filters, kernel_size=kernel_size, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(layer_1_n_filters, layer_2_n_filters, kernel_size=kernel_size, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(final_length * final_length * layer_2_n_filters, fc_1_n_nodes),
            nn.ReLU(),
            nn.Linear(fc_1_n_nodes, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas


model = LeNet5(num_classes=num_classes, grayscale=grayscale)

if cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=lr)

# Learning rate scheduler with warm-up
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: min(1.0, (epoch + 1) / warmup_epochs) if epoch < warmup_epochs else 1.0)

"""# Step 4: Train/Test Loop"""

# Defining the test and training loops

def train(epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        logits, probas = model(data)  # forward
        loss = criterion(logits, target)
        loss.backward()  # backward pass
        optimizer.step()  # optimizer update

        if batch_idx % logging_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                  f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
            writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + batch_idx)

    # Log model parameters to TensorBoard at every epoch
    for name, param in model.named_parameters():
        writer.add_histogram(name, param, epoch)

    # Step the learning rate scheduler
    scheduler.step()


def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for data, target in test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()

            logits, probas = model(data)
            test_loss += criterion(logits, target).item()  # sum up batch loss
            pred = probas.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)")

    writer.add_scalar('Test Loss', test_loss, epoch)
    writer.add_scalar('Test Accuracy', accuracy, epoch)

    return accuracy

# Running test and training over epochs
best_accuracy = 0
for epoch in range(1, epochs + 1):
    train(epoch)
    accuracy = test(epoch)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), "best_model.pth")

writer.close()

# Step 5: Visualization and Statistics

# Visualize the filters of the first convolutional layer
def visualize_filters(model):
    first_conv_layer = model.features[0]
    filters = first_conv_layer.weight.data.cpu().numpy()

    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if i < filters.shape[0]:
            ax.imshow(filters[i, 0, :, :], cmap='gray')
            ax.axis('off')
    plt.suptitle('Filters of the First Convolutional Layer')
    plt.show()

visualize_filters(model)

# Visualize the activations of the first convolutional layer for a batch of test images
def visualize_activations(model, test_loader):
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    if cuda:
        images = images.cuda()

    with torch.no_grad():
        activations = model.features[0](images).cpu().numpy()

    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if i < activations.shape[0]:
            ax.imshow(activations[i, 0, :, :], cmap='gray')
            ax.axis('off')
    plt.suptitle('Activations of the First Convolutional Layer')
    plt.show()

    # Statistics for activations (mean and standard deviation)
    activation_means = activations.mean(axis=(0, 2, 3))
    activation_stds = activations.std(axis=(0, 2, 3))

    print("Activation Means:", activation_means)
    print("Activation Standard Deviations:", activation_stds)

visualize_activations(model, test_loader)

# Feature Maximization Visualization
def feature_maximization(model, layer_index, filter_index, steps=30, lr=0.1):
    model.eval()
    input_image = torch.randn(1, 1, 28, 28, requires_grad=True)
    if cuda:
        input_image = input_image.cuda()

    optimizer = optim.Adam([input_image], lr=lr)

    for step in range(steps):
        optimizer.zero_grad()
        x = input_image
        for idx, layer in enumerate(model.features):
            x = layer(x)
            if idx == layer_index:
                break
        loss = -x[0, filter_index].mean()
        loss.backward()
        optimizer.step()

    # Visualize the resulting image
    input_image = input_image.detach().cpu().numpy()[0, 0]
    plt.imshow(input_image, cmap='gray')
    plt.title(f'Feature Maximization for Filter {filter_index} in Layer {layer_index}')
    plt.axis('off')
    plt.show()

# Example: Visualize what maximally activates filter 0 in the first convolutional layer
feature_maximization(model, layer_index=0, filter_index=0, steps=50, lr=0.1)

# Final Model Accuracy
print(f"Best Test Accuracy: {best_accuracy:.2f}%")

# Commented out IPython magic to ensure Python compatibility.
"""
#https://stackoverflow.com/questions/55970686/tensorboard-not-found-as-magic-function-in-jupyter

#seems to be working in firefox when not working in Google Chrome when running in Colab
#https://stackoverflow.com/questions/64218755/getting-error-403-in-google-colab-with-tensorboard-with-firefox


# %load_ext tensorboard
# %tensorboard --logdir [dir]

"""
