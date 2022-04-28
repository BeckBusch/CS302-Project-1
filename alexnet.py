# This is the code to implement the AlexNet architecture

# Necessary imports
import torch
from torch import nn, optim, cuda
from torch.utils import data
from torchvision import datasets, transforms
import time

# Training settings
batch_size = 64
device = 'cuda:0' if cuda.is_available() else 'cpu'
print(f"Training MNIST on {device}")

# Resize the images
# transform = transforms.Compose(
#     [transforms.ToTensor()])#,
     #transforms.Resize(99)])

# MNIST dataset
training_dataset = datasets.MNIST(root = 'mnist_data/',
                                  train = True, 
                                  transform = transforms.ToTensor(),
                                  download = True)

testing_dataset = datasets.MNIST(root = 'mnist_data/',
                                 train = False,
                                 transform = transforms.ToTensor())

# Create data loaders
training_loader = data.DataLoader(training_dataset, batch_size = batch_size, shuffle = True)

testing_loader = data.DataLoader(testing_dataset, batch_size = batch_size, shuffle = False)

# LeNet-5 Model
class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.l1 = nn.Conv2d(1, 6, 5, padding = 2)
        self.l2 = nn.MaxPool2d(2, stride = 2)
        self.l3 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = self.l2(x)
        x = torch.relu(self.l3(x))
        x = self.l2(x)
        x = x.view(-1, 16*5*5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# AlexNet Model
# class AlexNet(nn.Module):

#     def __init__(self):
        # self.l1 = nn.Conv2d(1, 96, 11, stride = 4)
        # self.l2 = nn.MaxPool2d(3, stride = 2)
        # self.l3 = nn.Conv2d(96, 256, 5, padding = 2)
        # self.l4 = nn.MaxPool2d(3, stride = 2)
        # self.l5 = nn.Conv2d(256, 384, 3, padding = 1)
        # self.l6 = nn.Conv2d(384, 384, 3, padding = 1)
        # self.l7 = nn.Conv2d(384, 256, 3, padding = 1)
        # self.l8 = nn.MaxPool2d(3, stride = 2)
        # self.fc1 = nn.Linear(2 * 2 * 256, 256)
        # self.fc2 = nn.Linear(256, 256)
        # self.fc3 = nn.Linear(256, 62)

    # def forward(self, x):

        # next = self.l2(F.relu(self.l1(next)))
        # next = self.l4(F.relu(self.l3(next)))
        # next = F.relu(self.l5(next))
        # next = F.relu(self.l6(next))
        # next = self.l8(F.relu(self.l7(next)))
        # next = next.view(-1, 2 * 2 * 256)
        # next = F.dropout(self.fc1(next), p = 0.5)
        # next = F.dropout(self.fc2(next), p = 0.5)
        # return self.fc3(next)

# Loss function and optimisation
model = LeNet().to(device)
loss_function = nn.CrossEntropyLoss()
optimiser = optim.SGD(model.parameters(), lr = 0.01)

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
        if batch % 10 == 0:
            print(f"Train epoch: {epoch} | Batch status: {batch * len(source)}/{len(training_loader.dataset)} "
                f"({100 * batch / len(training_loader):.0f}%) | Loss: {loss.item():.6f}")

def test():
    model.eval()
    testing_loss = 0
    correct = 0
    for source, target in testing_loader:
        source, target = source.to(device), target.to(device)
        prediction = model(source)
        _, predicted = torch.max(prediction, 1)
        testing_loss += loss_function(prediction, target).item()
        correct += (predicted == target).sum().item()

    testing_loss /= len(testing_loader)
    #change this
    print(f"\nTest Set: Average loss: {testing_loss:.4f}, Accuracy: {correct} / {len(testing_loader.dataset)} "
          f"({100 * correct / len(testing_loader.dataset):.0f}%)")

if __name__ == '__main__':
    #torch.backends.cudnn.benchmark = True
    start_time = time.time()
    for epoch in range(3):
        epoch_start = time.time()
        train(epoch)
        minutes, seconds = divmod(time.time() - epoch_start, 60)
        print(f"Time spent training: {minutes:.0f} minutes {seconds:.0f} seconds")
        epoch_start = time.time()
        test()
        minutes, seconds = divmod(time.time() - epoch_start, 60)
        print(f"Time spent testing: {minutes:.0f} minutes {seconds:.0f} seconds")

    minutes, seconds = divmod(time.time() - start_time, 60)
    print(f"Total time spent testing & training: {minutes:.0f} minutes {seconds:.0f} seconds")
