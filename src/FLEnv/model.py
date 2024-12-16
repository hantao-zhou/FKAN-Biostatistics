import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Note the model and functions here defined do not have any FL-specific components.


class Dummy_Model(nn.Module):
    """A simple CNN suitable for simple vision tasks."""

    def __init__(self, num_classes: int) -> None:
        super(Dummy_Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(44944, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

def train(net, trainloader, optimizer, epochs, device: str):
    """Train the network on the training set.

    This is a fairly simple training loop for PyTorch.
    """
    criterion = torch.nn.BCEWithLogitsLoss()
    net.train()
    net.to(device)
    for _ in range(epochs):
        for images, labels in tqdm(trainloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(torch.squeeze(outputs, dim=1), labels)
            loss.backward()
            optimizer.step()

            



def test(net, testloader, device: str):
    """Validate the network on the entire test set.

    and report loss and accuracy.
    """
    criterion = torch.nn.BCEWithLogitsLoss()
    correct, loss = 0, 0.0
    net.eval()
    net.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(torch.squeeze(outputs, dim=1), labels).item()
            predicted = torch.squeeze(F.sigmoid(outputs), dim=1)
            #_, predicted = torch.max(outputs.data, 1)
            correct += (predicted.round() == labels).sum().item() / images.shape[0]
    accuracy = correct / len(testloader)
    loss = loss / len(testloader)
    return loss, accuracy