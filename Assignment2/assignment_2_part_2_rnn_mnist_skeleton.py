# -*- coding: utf-8 -*-
"""Assignment_2_Part_2_RNN_MNIST_vp1.ipynb
Overall structure:

1) Set Pytorch metadata
- seed
- tensorflow output
- whether to transfer to gpu (cuda)

2) Import data
- download data
- create data loaders with batch size, transforms, scaling

3) Define Model architecture, loss and optimizer

4) Define Test and Training loop
    - Train:
        a. get next batch
        b. forward pass through model
        c. calculate loss
        d. backward pass from loss (calculates the gradient for each parameter)
        e. optimizer: performs weight updates

5) Perform Training over multiple epochs:
    Each epoch:
    - call train loop
    - call test loop

# Step 1: Pytorch and Training Metadata
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from pathlib import Path
import matplotlib.pyplot as plt

batch_size = 64
test_batch_size = 1000
epochs = 10
lr = 0.001
try_cuda = True
seed = 1000
logging_interval = 10  # how many batches to wait before logging
logging_dir = None

INPUT_SIZE = 28  # MNIST images are 28x28 pixels

# Setting up the logging

datetime_str = datetime.now().strftime('%b%d_%H-%M-%S')

if logging_dir is None:
    runs_dir = Path("./") / Path(f"runs/")
    runs_dir.mkdir(exist_ok=True)

    logging_dir = runs_dir / Path(f"{datetime_str}")
    logging_dir.mkdir(exist_ok=True)
    logging_dir = str(logging_dir.absolute())

writer = SummaryWriter(log_dir=logging_dir)

# Deciding whether to use CPU or GPU
if torch.cuda.is_available() and try_cuda:
    cuda = True
    torch.cuda.manual_seed(seed)
else:
    cuda = False
    torch.manual_seed(seed)

"""# Step 2: Data Setup"""

# Setting up data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Downloading and transforming the MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Creating data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False)

# Plot one example to verify the data
print(train_dataset.data.size())  # (60000, 28, 28)
print(train_dataset.targets.size())  # (60000)
plt.imshow(train_dataset.data[0].numpy(), cmap='gray')
plt.title('%i' % train_dataset.targets[0])
plt.show()

"""# Step 3: Creating the Model"""

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNNModel, self).__init__()
        # self.rnn = nn.RNN(input_size, hidden_size, n_layers, batch_first=True)
        # self.rnn = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)
        self.rnn = nn.GRU(input_size, hidden_size, n_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        r_out, hidden = self.rnn(x, None)  # None represents zero initial hidden state
        out = self.out(r_out[:, -1, :])  # select output at the last time step
        return out

hidden_size = 128  # Number of nodes in the hidden layer
output_size = 10  # Number of classes (0-9 for MNIST)
model = RNNModel(INPUT_SIZE, hidden_size, output_size)

if cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=lr)

"""# Step 4: Train/Test"""

# Defining training and testing functions
criterion = nn.CrossEntropyLoss()

def train(epoch):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()

        data = data.view(-1, 28, 28)  # Reshape data for RNN (batch, time_step, input_size)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()  # Backpropagation
        optimizer.step()  # Optimize the weights

        # Calculate accuracy for the current batch
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % logging_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + batch_idx)

    accuracy = 100. * correct / len(train_loader.dataset)
    print(f'Train Epoch: {epoch} \tAccuracy: {accuracy:.2f}%')
    writer.add_scalar('Training Accuracy', accuracy, epoch)

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()

            data = data.view(-1, 28, 28)  # Reshape data for RNN
            output = model(data)
            test_loss += criterion(output, target).item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    writer.add_scalar('Test Loss', test_loss, epoch)
    writer.add_scalar('Test Accuracy', accuracy, epoch)

"""# Step 5: Training Loop"""

best_accuracy = 0
for epoch in range(1, epochs + 1):
    train(epoch)
    test(epoch)

writer.close()

# Commented out IPython magic to ensure Python compatibility.
"""
# https://stackoverflow.com/questions/55970686/tensorboard-not-found-as-magic-function-in-jupyter

# seems to be working in firefox when not working in Google Chrome when running in Colab
# https://stackoverflow.com/questions/64218755/getting-error-403-in-google-colab-with-tensorboard-with-firefox

# %load_ext tensorboard
# %tensorboard --logdir [dir]

"""
