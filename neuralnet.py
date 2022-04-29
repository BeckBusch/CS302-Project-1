""" This code implements 3 different CNN architectures, LeNet-5, AlexNet and VGG-11.
    Each architecture has been written manually, with adjustments as required to train
    on the MNIST dataset."""

# Necessary imports
import torch
from torch import nn, optim, cuda
from torch.utils import data
from torchvision import datasets, transforms
import time
import math

# Training settings
batch_size = 64
epoch_count = 5
train_ratio = 0.7
device = 'cuda:0' if cuda.is_available() else 'cpu'
print(f"Training MNIST on {device}")

# LeNet-5 Model
class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.l1 = nn.Conv2d(1, 6, 5, padding = 2)
        self.l2 = nn.Conv2d(6, 16, 5)
        self.mp = nn.MaxPool2d(2, stride = 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = self.mp(x)
        x = torch.relu(self.l2(x))
        x = self.mp(x)
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# AlexNet Model
class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()
        self.l1 = nn.Conv2d(1, 96, 11, stride = 4)
        self.l2 = nn.Conv2d(96, 256, 5, padding = 2)
        self.l3 = nn.Conv2d(256, 384, 3, padding = 1)
        self.l4 = nn.Conv2d(384, 384, 3, padding = 1)
        self.l5 = nn.Conv2d(384, 256, 3, padding = 1)
        self.mp = nn.MaxPool2d(3, stride = 2)
        self.fc1 = nn.Linear(2 * 2 * 256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 62)

    def forward(self, x):
        x = self.mp(torch.relu(self.l1(x)))
        x = self.mp(torch.relu(self.l2(x)))
        x = torch.relu(self.l3(x))
        x = torch.relu(self.l4(x))
        x = self.mp(torch.relu(self.l5(x)))
        x = x.view(-1, 2 * 2 * 256)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# VGG11 Model
class VGG11(nn.Module):

    def __init__(self):
        super(VGG11, self).__init__()
        self.l1 = nn.Conv2d(3, 64, 3, padding = 1)
        self.l2 = nn.Conv2d(64, 128, 3, padding = 1)
        self.l3 = nn.Conv2d(128, 256, 3, padding = 1)
        self.l4 = nn.Conv2d(256, 256, 3, padding = 1)
        self.l5 = nn.Conv2d(256, 512, 3, padding = 1)
        self.l6 = nn.Conv2d(512, 512, 3, padding = 1) # x3
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.mp = nn.MaxPool2d(2, stride = 2)
        self.fc1 = nn.Linear(2 * 2 * 512, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 62)
        self.do = nn.Dropout(p = 0.5)

    def forward(self, x):
        x = self.bn1(torch.relu_(self.l1(x)))
        x = self.mp(x)
        x = self.bn2(torch.relu_(self.l2(x)))
        x = self.mp(x)
        x = self.bn3(torch.relu_(self.l3(x)))
        x = self.bn3(torch.relu_(self.l4(x)))
        x = self.mp(x)
        x = self.bn4(torch.relu_(self.l5(x)))
        x = self.bn4(torch.relu_(self.l6(x)))
        x = self.mp(x)
        x = self.bn4(torch.relu_(self.l6(x)))
        x = self.bn4(torch.relu_(self.l6(x)))
        x = self.mp(x)
        x = x.view(-1, 2 * 2 * 512)
        x = self.do(torch.relu_(self.fc1(x)))
        x = self.do(torch.relu_(self.fc2(x)))
        x = self.fc3(x)
        return x

# User input
model = None
selected_model = input("Please select one of LeNet, AlexNet or VGG11 to use:")
if selected_model == "LeNet":
    model = LeNet().to(device)
    transform = transforms.ToTensor()
elif selected_model == "AlexNet":
    model = AlexNet().to(device)

    # Resize the images
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(99)])
elif selected_model == "VGG11":
    model = VGG11().to(device)

    # Resize the images and change to 3-channel (rgb)
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(64),
         transforms.Lambda(lambda y: y.repeat(3, 1, 1))])
else:
    input("Invalid model, please select from LeNet, AlexNet, VGG11 only:")

# MNIST dataset
full_dataset = datasets.MNIST(root = 'mnist_data/',
                         train = True, 
                         transform = transform,
                         download = True)

train_size = int(train_ratio * len(full_dataset))
test_size = len(full_dataset) - train_size
training_dataset, testing_dataset = data.random_split(full_dataset, [train_size, test_size])

# Create data loaders
training_loader = data.DataLoader(training_dataset, batch_size = batch_size, shuffle = True)
testing_loader = data.DataLoader(testing_dataset, batch_size = batch_size, shuffle = False)

# Loss function and optimisation
loss_function = nn.CrossEntropyLoss()
optimiser = optim.SGD(model.parameters(), lr = 1e-3, momentum = 0.9)

# Training loop
def train(epoch):
    model.train()
    for batch, (source, target) in enumerate(training_loader):
        source, target = source.to(device), target.to(device)

        # Calculate prediction error
        prediction = model(source)
        loss = loss_function(prediction, target)

        # Backpropagation
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        # Print statements with lots of formatting
        if batch % 10 == 0:
            print(f"Train epoch: {epoch + 1} | Batch status: ",
                  f"{batch * len(source)}/{len(training_loader.dataset)} ".rjust(math.ceil(math.log(train_size)) + 1),
                  f"({100 * batch / len(training_loader):.0f}%)".rjust(5),
                  f" | Loss: {loss.item():.6f}") 

# Testing loop
def test():
    model.eval()
    with torch.no_grad():
        testing_loss = 0
        correct = 0
        for source, target in testing_loader:
            source, target = source.to(device), target.to(device)
            prediction = model(source)
            _, predicted = torch.max(prediction, 1)
            testing_loss += loss_function(prediction, target).item()
            correct += (predicted == target).sum().item()

        testing_loss /= len(testing_loader)
    
        # Print statement to track loss and accuracy for this epoch
        print(f"\n[Test Set] Average loss: {testing_loss:.4f}, Accuracy: {correct} / {len(testing_loader.dataset)} "
              f"({100 * correct / len(testing_loader.dataset):.2f}%)")

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    start_time = time.time()
    for epoch in range(epoch_count):
        epoch_start = time.time()
        train(epoch)
        minutes, seconds = divmod(time.time() - epoch_start, 60)
        print(f"Time spent training: {minutes:.0f} minutes {seconds:.0f} seconds")
        epoch_start = time.time()
        test()
        minutes, seconds = divmod(time.time() - epoch_start, 60)
        print(f"Time spent testing: {minutes:.0f} minutes {seconds:.0f} seconds")
        print("")

    minutes, seconds = divmod(time.time() - start_time, 60)
    print(f"Total time spent testing & training: {minutes:.0f} minutes {seconds:.0f} seconds")