import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torchmetrics
import math

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
    

class KANLinear_v1(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, 
                 scale_noise=0.1, scale_base=1.0, scale_spline=1.0, 
                 enable_standalone_scale_spline=True, base_activation=nn.SiLU, 
                 grid_eps=0.02, grid_range=[-1, 1]):
        super(KANLinear_v1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = ((torch.arange(-spline_order, grid_size + spline_order + 1) * h 
                 + grid_range[0]).expand(in_features, -1).contiguous())
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = ((torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5)
                     * self.scale_noise / self.grid_size)
            self.spline_weight.data.copy_(
                self.scale_spline * self.curve2coeff(self.grid.T[self.spline_order : -self.spline_order], noise)
            )
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x):
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x - grid[:, :-(k+1)]) / (grid[:, k:-1] - grid[:, :-(k+1)]) * bases[:, :, :-1]
                     + (grid[:, k+1:] - x) / (grid[:, k+1:] - grid[:, 1:-k]) * bases[:, :, 1:])
        return bases.contiguous()

    def curve2coeff(self, x, y):
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        return solution.permute(2, 0, 1).contiguous()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1), 
            self.spline_weight.view(self.out_features, -1)
        )
        return base_output + spline_output

# ConvNeXtKAN_v1 Model
class ConvNeXtKAN_v1(nn.Module):
    def __init__(self):
        super(ConvNeXtKAN_v1, self).__init__()
        # Base model
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),  # First convolutional layer: 1 Gray channels
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # MaxPooling

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),  # Second convolutional layer
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # MaxPooling

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),  # Third convolutional layer
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # MaxPooling

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),  # Fourth convolutional layer
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # MaxPooling

            nn.AdaptiveAvgPool2d((1, 1))  # Adaptive pooling to reduce the size
        )

        self.flatten = nn.Flatten()

        # KAN layers
        self.kan1 = KANLinear_v1(256, 512)
        self.kan2 = KANLinear_v1(512, 1)  # 2 classes (suitable for binary classification)

    def forward(self, x):
        x = self.conv_layers(x)  # Pass through convolutions
        x = self.flatten(x)  # Flatten the output
        x = self.kan1(x)  # Pass through the first KAN layer
        x = self.kan2(x)  # Pass through the second KAN layer
        return x
    

def train(net, trainloader, optimizer, epochs, device: str):
    """Train the network on the training set.

    This is a fairly simple training loop for PyTorch.
    """
    criterion = torch.nn.BCEWithLogitsLoss()
    net.train()
    net.to(device)
   
    for _ in range(epochs):
        p_loss = 0
        pbar = tqdm(trainloader)
        for i, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(torch.squeeze(outputs, dim=1), labels)
            loss.backward()
            optimizer.step()
            p_loss += loss.item()
            #pbar.set_description(f'Loss: {p_loss / (i + 1)}')
            

            



def test(net, testloader, device: str):
    """Validate the network on the entire test set.

    and report loss and accuracy.
    """
    criterion = torch.nn.BCEWithLogitsLoss()
    recall_score_metric = torchmetrics.Recall(task='binary')
    precision_score_metric = torchmetrics.Precision(task='binary')
    correct, loss = 0, 0.0
    net.eval()
    net.to(device)
    predictions = []
    labels = []
    with torch.no_grad():
        for images, label in testloader:
            images, label = images.to(device), label.to(device)
            logits = net(images)
            logits = F.sigmoid(torch.squeeze(logits))
            loss += criterion(logits, label).item()
            prediction = torch.round(logits)
            predictions.extend(prediction)
            labels.extend(label)
            
    predictions = torch.Tensor(predictions)
    labels = torch.Tensor(labels)
    accuracy = torch.sum(predictions == labels) / len(testloader)
    recall = recall_score_metric(predictions, labels)
    precision = precision_score_metric(predictions, labels)
    f1 = 2 * ((precision *recall)/(precision + recall))
    loss = loss / len(testloader)
    return loss, accuracy, f1, precision, recall