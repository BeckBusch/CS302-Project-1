# This is the code to implement the AlexNet architecture

# Necessary imports
from torch import nn, optim, cuda
from torch.utils import data
from torchvision import datasets, transforms
import torch.nn.functional as F
import time

# Training settings
batch_size = 64
device = 'cuda' if cuda.is_available() else 'cpu'

# MNIST dataset
training_dataset = datasets.MNIST(root = 'mnist_data/',
                                  train = True, 
                                  transform = transforms.ToTensor(), 
                                  transform = transforms.Resize(99, 99, antialias = True),
                                  download = True)

testing_dataset = datasets.MNIST(root = 'mnist_data/',
                                 train = False,
                                 transform = transforms.ToTensor(),
                                 transform = transforms.Resize(99, 99, antialias = True))

# Create data loaders
training_loader = data.DataLoader(training_dataset, batch_size = batch_size, shuffle = True)

testing_loader = data.DataLoader(testing_dataset, batch_size = batch_size, shuffle = False)

# AlexNet Model
class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()
        self.l1 = nn.Conv2d(3, 96, 11, stride = 4)
        self.l2 = nn.MaxPool2d(3, stride = 2)
        self.l3 = nn.Conv2d(96, 256, 5, padding = 2)
        self.l4 = nn.MaxPool2d(3, stride = 2)
        self.l5 = nn.Conv2d(256, 384, 3, padding = 1)
        self.l6 = nn.Conv2d(384, 384, 3, padding = 1)
        self.l7 = nn.Conv2d(384, 256, 3, padding = 1)
        self.l8 = nn.MaxPool2d(3, stride = 2)

    def forward(self, next):
        next = F.relu(self.l1(next))
        next = self.l2
        next = F.relu(self.l3(next))
        next = self.l4
        next = F.relu(self.l5(next))
        next = F.relu(self.l6(next))
        next = F.relu(self.l7(next))
        next = self.l8
        next = self.flatten(next)
        for i in range (2):
            next = F.dropout(next, p = 0.5)
        next = F.softmax(next, dim = 1)

# Loss function and optimisation
model = AlexNet().to(device)
loss_function = nn.CrossEntropyLoss
optimiser = optim.SGD(model.parameters(), lr = 1e-3, momentum = 0.5)

# Training loop
def train():
    model.train()
    for batch, (source, target) in enumerate(training_loader):
        source, target = source.to(device), target.to(device)

        # Calculate prediction error
        optimiser.zero_grad()
        prediction = model(source)
        loss = loss_function(prediction, target)

        # Backpropagation
        loss.backward()
        optimiser.step()
    if batch % 50 == 0:
        print("working")

def test():
    model.eval()
    testing_loss = 0
    correct = 0
    for (source, target) in testing_loader:
        source, target = source.to(device), target.to(device)
        prediction = model(source)
        testing_loss += loss_function(source, target).item()
        ##
        correct += (prediction.argmax(1) == target).type(float).sum().item()

    testing_loss /= len(testing_loader)
    correct /= len(testing_loader.dataset)

if __name__ == '__main__':
    epoch = 10
    for i in range(epoch):
        epoch_start = time.time()
        train()
        test()
        print(f"Epoch {i}")
    print("Complete")
