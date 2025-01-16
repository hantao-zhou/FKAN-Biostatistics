import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from sklearn.metrics import precision_recall_fscore_support
from kan import KAN
from rich.progress import Progress
import time
import os
from dataset import prepare_dataset

# Define the KANClassifier with Dropout layers
class KANClassifier(nn.Module):
    def __init__(self, kan_model, num_classes):
        super(KANClassifier, self).__init__()
        self.kan_model = kan_model
        last_layer_width = kan_model.width[-1]
        if isinstance(last_layer_width, list):
            last_layer_width = last_layer_width[0]
        self.fc = nn.Linear(last_layer_width, num_classes)
        self.dropout = nn.Dropout(p=0.5)  # Dropout with 50% probability
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        x = self.kan_model(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def train_kan(self, dataloader, steps, lamb, lamb_entropy, progress, task):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-5)
        loss_fn = nn.CrossEntropyLoss()
        for step in range(steps):
            for inputs, labels in dataloader:
                inputs, labels = inputs.cuda(), labels.cuda()
                optimizer.zero_grad()
                outputs = self(inputs.float())
                loss = loss_fn(outputs, labels.long())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                progress.advance(task)

# Function to compute metrics
def compute_metrics(model, loader, progress, task):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs.float())
            loss = loss_fn(outputs, labels.long())
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            progress.advance(task)
    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / total
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    return avg_loss, accuracy, precision, recall, f1

# Prepare dataset
num_partitions = 2  # Number of clients
batch_size = 16
train_loaders, val_loaders, valid_loader, test_loader = prepare_dataset(num_partitions, batch_size)

# Initialize KAN model
kan_width = [4, 10, 5, 1]  # Example sizes; adjust as needed
print("Initializing KAN model...")
base_kan_model = KAN(width=kan_width, grid=5, k=3, seed=0,).cuda()
kan_model = KANClassifier(base_kan_model, num_classes=2).cuda()

# Train and validate the KAN model
num_rounds = 20
train_steps = 10
start_time = time.time()
print("Training KAN model...")

with Progress() as progress:
    round_task = progress.add_task("Training Rounds", total=num_rounds)
    for round_num in range(num_rounds):
        print(f"Round {round_num + 1}/{num_rounds}")

        # Train the model for all clients
        train_task = progress.add_task("Training Clients", total=len(train_loaders) * train_steps)
        for client_loader in train_loaders:
            kan_model.train_kan(client_loader, steps=train_steps, lamb=0.01, lamb_entropy=10.0, progress=progress, task=train_task)

        # Validate the model
        val_task = progress.add_task("Validating Model", total=len(valid_loader))
        val_loss, val_acc, val_prec, val_recall, val_f1 = compute_metrics(kan_model, valid_loader, progress, val_task)
        print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, Precision: {val_prec:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
        progress.advance(round_task)

# Test the model
with Progress() as progress:
    test_task = progress.add_task("Testing Model", total=len(test_loader))
    test_loss, test_acc, test_prec, test_recall, test_f1 = compute_metrics(kan_model, test_loader, progress, test_task)

print(f"Test Results - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")

end_time = time.time()
print(f"Total training time: {end_time - start_time:.2f} seconds")
