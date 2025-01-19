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
from sklearn.metrics import precision_score, recall_score,f1_score
from torchsummary import summary
import math
import logging
import torch as th
from torch.utils.data import DataLoader, TensorDataset
import math
from typing import *

import torch.optim as optim
from tqdm import tqdm



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



train_images_tensor = torch.tensor(train_images)
val_images_tensor = torch.tensor(val_images)
test_images_tensor = torch.tensor(test_images)

train_images_flat = train_images_tensor
val_images_flat = val_images_tensor
test_images_flat = test_images_tensor

train_labels_tensors = torch.tensor(train_labels, dtype=torch.long)
val_labels_tensors = torch.tensor(val_labels, dtype=torch.long)

train_dataset = TensorDataset(train_images_flat, train_labels_tensors)
val_dataset = TensorDataset(val_images_flat, val_labels_tensors)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class KAL_Net(nn.Module):
    def __init__(self, layers_hidden=[50176, 1024, 128], polynomial_order=3, base_activation=nn.SiLU, dropout_prob=0.5):
        super(KAL_Net, self).__init__()
        self.layers_hidden = layers_hidden
        self.polynomial_order = polynomial_order
        self.base_activation = base_activation()
        self.dropout_prob = dropout_prob

        self.base_weights = nn.ParameterList()
        self.poly_weights = nn.ParameterList()
        self.layer_norms = nn.ModuleList()

        # Define layers based on the provided hidden layers structure
        for in_features, out_features in zip(layers_hidden[:-1], layers_hidden[1:]):
            self.base_weights.append(nn.Parameter(torch.randn(out_features, in_features)))
            self.poly_weights.append(nn.Parameter(torch.randn(out_features, in_features * (polynomial_order + 1))))
            self.layer_norms.append(nn.LayerNorm(out_features))

        # Dropout Layers
        self.dropouts = nn.ModuleList([nn.Dropout(p=dropout_prob) for _ in self.layer_norms])

        # Output layer for binary classification
        self.output_layer = nn.Linear(layers_hidden[-1], 2)
        
        for weight in self.base_weights:
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')
        for weight in self.poly_weights:
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')

    @staticmethod
    def compute_legendre_polynomials(x, order):
        P0 = torch.ones_like(x)
        if order == 0:
            return P0.unsqueeze(-1)
        P1 = x
        legendre_polys = [P0, P1]
        
        for n in range(1, order):
            Pn = ((2.0 * n + 1.0) * x * legendre_polys[-1] - n * legendre_polys[-2]) / (n + 1.0)
            legendre_polys.append(Pn)
        
        return torch.stack(legendre_polys, dim=-1)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten the input correctly to match the input size of 22500
        
        for base_weight, poly_weight, layer_norm, dropout in zip(self.base_weights, self.poly_weights, self.layer_norms, self.dropouts):
            base_output = F.linear(self.base_activation(x), base_weight)
            
            x_normalized = 2 * (x - x.min()) / (x.max() - x.min()) - 1
            legendre_basis = self.compute_legendre_polynomials(x_normalized, self.polynomial_order)
            legendre_basis = legendre_basis.view(batch_size, -1)

            poly_output = F.linear(legendre_basis, poly_weight)
            x = self.base_activation(layer_norm(base_output + poly_output))
            x = dropout(x)

        x = self.output_layer(x)
        return torch.softmax(x, dim=1)  # Use softmax for multi-class classification

# Example instantiation
model = KAL_Net()

# Define an optimizer with L2 Regularization (Weight Decay)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Ensure the model is on the appropriate device
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
            torch.save(model.state_dict(), "best_model_KAL.pth")
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


# Model and device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Correcting the input dimensions to match the flattened image dimensions
#input_dim = train_images_flat.shape[1]  # Assuming train_images_flat was defined previously

# Example instantiation
model = KAL_Net()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# Load the saved model state
# Note: Be sure to use the appropriate path and file for your model
state_dict = torch.load("best_model_KAL.pth", weights_only=True)
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

# Example instantiation
model = KAL_Net()
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