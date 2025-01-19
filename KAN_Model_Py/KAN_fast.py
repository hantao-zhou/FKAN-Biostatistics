import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import torch
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from sklearn.metrics import precision_score, recall_score
from torchsummary import summary
import math
import logging

labels = ['PNEUMONIA', 'NORMAL']
img_size = 224

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_training_data(data_dir):
    data = []
    
    for label in labels:
        path = os.path.join(data_dir, label)
        #logger.info(path)
        class_num = labels.index(label)
        
        for img in os.listdir(path):
            try:
                # Load and resize the image
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size))  # Resize the image
                
                # Add the image and label as a pair
                data.append([resized_arr, class_num])
            except Exception as e:
                logger.info(f"Error loading image {img}: {e}")
    
    # Convert the list to a NumPy array
    data = np.array(data, dtype=object)  # Use dtype=object to allow image-label pairing
    return data

# Load the data
train_data = get_training_data('../normData/train') 
test_data = get_training_data('../normData/test')
val_data = get_training_data( '../normData/val')

#Separate the images and the labels
train_images = np.array([x[0] for x in train_data])  # Extract only the images
train_labels = np.array([x[1] for x in train_data])  # Extract only the labels
    
val_images = np.array([x[0] for x in val_data])  # Extract only the images
val_labels = np.array([x[1] for x in val_data])  # Extract only the labels
    
test_images = np.array([x[0] for x in test_data])  # Extract only the images
test_labels = np.array([x[1] for x in test_data])  # Extract only the labels

# Check the shape and an example of the dataset
logger.info(f"Shape of train images: {train_images.shape}")
logger.info(f"Shape of validation images: {val_images.shape}")

# Display the first image from the training data
plt.figure(figsize=(5, 5))
plt.imshow(train_images[0], cmap='gray')
plt.title('Pneumonia' if train_labels[0] == 0 else 'Normal')
plt.axis('off')
plt.show()

# Display the last image from the validation data
plt.figure(figsize=(5, 5))
plt.imshow(val_images[-1], cmap='gray')
plt.title('Pneumonia' if val_labels[-1] == 0 else 'Normal')
plt.axis('off')
plt.show()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from sklearn.metrics import precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
import math
from typing import *

# Function to flatten images
def flatten_images(images):
    return torch.flatten(images, start_dim=1)


train_images_tensor = torch.tensor(train_images)
val_images_tensor = torch.tensor(val_images)
test_images_tensor = torch.tensor(test_images)

train_images_flat = flatten_images(train_images_tensor)
val_images_flat = flatten_images(val_images_tensor)
test_images_flat = flatten_images(test_images_tensor)

train_labels_tensors = torch.tensor(train_labels, dtype=torch.long)
val_labels_tensors = torch.tensor(val_labels, dtype=torch.long)

train_dataset = TensorDataset(train_images_flat, train_labels_tensors)
val_dataset = TensorDataset(val_images_flat, val_labels_tensors)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)

class RadialBasisFunction(nn.Module):
    def __init__(
        self,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        denominator: float = None,  # larger denominators lead to smoother basis
    ):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)

class FastKANLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        use_base_update: bool = True,
        use_layernorm: bool = True,
        base_activation = F.silu,
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layernorm = None
        if use_layernorm:
            assert input_dim > 1, "Do not use layernorms on 1D inputs. Set `use_layernorm=False`."
            self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale)
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, use_layernorm=True):
        if self.layernorm is not None and use_layernorm:
            spline_basis = self.rbf(self.layernorm(x))
        else:
            spline_basis = self.rbf(x)
        ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret

    def plot_curve(
        self,
        input_index: int,
        output_index: int,
        num_pts: int = 1000,
        num_extrapolate_bins: int = 2
    ):
        '''this function returns the learned curves in a FastKANLayer.
        input_index: the selected index of the input, in [0, input_dim) .
        output_index: the selected index of the output, in [0, output_dim) .
        num_pts: num of points sampled for the curve.
        num_extrapolate_bins (N_e): num of bins extrapolating from the given grids. The curve 
            will be calculate in the range of [grid_min - h * N_e, grid_max + h * N_e].
        '''
        ng = self.rbf.num_grids
        h = self.rbf.denominator
        assert input_index < self.input_dim
        assert output_index < self.output_dim
        w = self.spline_linear.weight[
            output_index, input_index * ng : (input_index + 1) * ng
        ]   # num_grids,
        x = torch.linspace(
            self.rbf.grid_min - num_extrapolate_bins * h,
            self.rbf.grid_max + num_extrapolate_bins * h,
            num_pts
        )   # num_pts, num_grids
        with torch.no_grad():
            y = (w * self.rbf(x.to(w.dtype))).sum(-1)
        return x, y


class FastKAN(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        use_base_update: bool = True,
        base_activation = F.silu,
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            FastKANLayer(
                in_dim, out_dim,
                grid_min=grid_min,
                grid_max=grid_max,
                num_grids=num_grids,
                use_base_update=use_base_update,
                base_activation=base_activation,
                spline_weight_init_scale=spline_weight_init_scale,
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class AttentionWithFastKANTransform(nn.Module):
    
    def __init__(
        self,
        q_dim: int,
        k_dim: int,
        v_dim: int,
        head_dim: int,
        num_heads: int,
        gating: bool = True,
    ):
        super(AttentionWithFastKANTransform, self).__init__()

        self.num_heads = num_heads
        total_dim = head_dim * self.num_heads
        self.gating = gating
        self.linear_q = FastKANLayer(q_dim, total_dim)
        self.linear_k = FastKANLayer(k_dim, total_dim)
        self.linear_v = FastKANLayer(v_dim, total_dim)
        self.linear_o = FastKANLayer(total_dim, q_dim)
        self.linear_g = None
        if self.gating:
            self.linear_g = FastKANLayer(q_dim, total_dim)
        # precompute the 1/sqrt(head_dim)
        self.norm = head_dim**-0.5

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias: torch.Tensor = None,      # additive attention bias
    ) -> torch.Tensor:         

        wq = self.linear_q(q).view(*q.shape[:-1], 1, self.num_heads, -1) * self.norm     # *q1hc
        wk = self.linear_k(k).view(*k.shape[:-2], 1, k.shape[-2], self.num_heads, -1)    # *1khc
        att = (wq * wk).sum(-1).softmax(-2)     # *qkh
        del wq, wk
        if bias is not None:
            att = att + bias[..., None]

        wv = self.linear_v(v).view(*v.shape[:-2],1, v.shape[-2], self.num_heads, -1)     # *1khc
        o = (att[..., None] * wv).sum(-3)        # *qhc
        del att, wv

        o = o.view(*o.shape[:-2], -1)           # *q(hc)

        if self.linear_g is not None:
            # gating, use raw query input
            g = self.linear_g(q)
            o = torch.sigmoid(g) * o

        # merge heads
        o = self.linear_o(o)
        return o


# Setup and model initialization

input_dim = 224 * 224  # For example, 224*224 if that's your image size
layers_hidden = [input_dim, 128, 64, 10]  # Example layers configuration
model = FastKAN(layers_hidden)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Removed torchsummary import and summary call

optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

def calculate_precision_recall(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, preds = torch.max(output, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')

    return precision, recall


import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score

# Assume train_images_flat, val_images_flat, and their corresponding label tensors are provided
# They were created using the flattening and tensor conversion logic



# Labels are already in tensor form from the previous code
train_labels_tensor = train_labels_tensors
val_labels_tensor = val_labels_tensors

# Create the dataset and DataLoader
train_dataset = TensorDataset(train_images_flat, train_labels_tensor)
val_dataset = TensorDataset(val_images_flat, val_labels_tensor)

# Define the batch size
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model and device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Correcting the input dimensions to match the flattened image dimensions
input_dim = train_images_tensor.shape[1]

# Define the loss function and optimizer 
criterion = nn.CrossEntropyLoss()  # For multi-class or binary classification
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)  # AdamW with L2 regularization 

# Now the data is ready for training and validation

# Function to calculate precision and recall
def calculate_precision_recall(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(torch.float32).to(device), target.to(torch.long).to(device)
            output = model(data)
            _, preds = torch.max(output, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    
    return precision, recall

# Training function with Early Stopping
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, patience=10):
    best_val_loss = float("inf")
    epochs_without_improvement = 0  # Count the number of epochs without improvement

    for epoch in range(num_epochs):
        model.train()  # Training mode
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(torch.float32).to(device), labels.to(torch.long).to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute the loss
            loss = criterion(outputs, labels)

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Compute metrics
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels).item()
            total_preds += labels.size(0)

        # Calculate average loss and accuracy
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct_preds / total_preds

        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        logger.info(f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}")

        # Validation phase
        model.eval()  # Evaluation mode
        val_loss = 0.0
        val_correct_preds = 0
        val_total_preds = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(torch.float32).to(device), labels.to(torch.long).to(device)

                # Forward pass
                outputs = model(inputs)

                # Compute the loss
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                # Compute metrics
                _, preds = torch.max(outputs, 1)
                val_correct_preds += torch.sum(preds == labels).item()
                val_total_preds += labels.size(0)

        # Calculate validation loss and accuracy
        val_loss /= len(val_loader.dataset)
        val_accuracy = val_correct_preds / val_total_preds

        logger.info(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Calculate precision and recall on validation
        val_precision, val_recall = calculate_precision_recall(model, val_loader, device)
        logger.info(f"Validation Precision: {val_precision:.4f}, Validation Recall: {val_recall:.4f}")

        # Early Stopping: stop if no improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Save the model with the best validation loss
            torch.save(model.state_dict(), "best_model_fast.pth")
            logger.info("Model saved!")
        else:
            epochs_without_improvement += 1
            logger.info(f"Early Stopping Counter: {epochs_without_improvement}/{patience}")

        # If the number of epochs without improvement exceeds patience, stop
        if epochs_without_improvement >= patience:
            logger.info("Early stopping triggered. Stopping training.")
            break

# Start training
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, patience=5)


import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score

# Model and device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Correcting the input dimensions to match the flattened image dimensions
input_dim = train_images_flat.shape[1]  # Assuming train_images_flat was defined previously

input_dim = 224 * 224  # For example, 224*224 if that's your image size
layers_hidden = [input_dim, 128, 64, 10]  # Example layers configuration
model = FastKAN(layers_hidden)
model = model.to(device)

# Load the saved model state
# Note: Be sure to use the appropriate path and file for your model
state_dict = torch.load("best_model_fast.pth", weights_only=True)
model.load_state_dict(state_dict, strict=False)

def calculate_metrics(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(torch.float32).to(device), target.to(torch.long).to(device)
            output = model(data)
            _, preds = torch.max(output, 1)
            
            correct_preds += torch.sum(preds == target).item()
            total_preds += target.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    accuracy = correct_preds / total_preds

    return precision, recall, f1, accuracy

def validate_model(model, val_loader, device):
    precision, recall, f1, accuracy = calculate_metrics(model, val_loader, device)
    logger.info(f"Validation Accuracy: {accuracy:.4f}")
    logger.info(f"Validation Precision: {precision:.4f}")
    logger.info(f"Validation Recall: {recall:.4f}")
    logger.info(f"Validation F1-Score: {f1:.4f}")

# Validate the model on the validation dataset
validate_model(model, val_loader, device)

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score

#Convert them to tensors if they aren't already
test_label_tensor = torch.tensor(test_labels, dtype=torch.long)
test_dataset = TensorDataset(test_images_flat, test_label_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)



def calculate_metrics(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(torch.float32).to(device), target.to(torch.long).to(device)
            output = model(data)
            _, preds = torch.max(output, 1)
            
            correct_preds += torch.sum(preds == target).item()
            total_preds += target.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    accuracy = correct_preds / total_preds

    return precision, recall, f1, accuracy

def test_model(model, test_loader, device):
    precision, recall, f1, accuracy = calculate_metrics(model, test_loader, device)
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Test Precision: {precision:.4f}")
    logger.info(f"Test Recall: {recall:.4f}")
    logger.info(f"Test F1-Score: {f1:.4f}")

# Define and initialize your model using fast KAN

input_dim = 224 * 224  # For example, 224*224 if that's your image size
layers_hidden = [input_dim, 128, 64, 10]  # Example layers configuration
model = FastKAN(layers_hidden)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load the saved model state
#model.load_state_dict(torch.load("best_model_v4.pth"))
#model.eval()

# Load the saved state_dict
state_dict = torch.load("best_model_fast.pth", weights_only=True)

model.load_state_dict(state_dict, strict=False)
model.load_state_dict(state_dict)


# Print the state_dict keys of your current model
current_state_dict = model.state_dict()
#logger.info("Current Model State Dict Keys:", list(current_state_dict.keys()))

test_model(model, test_loader, device)