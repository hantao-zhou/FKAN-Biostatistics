import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import logging
import torch as th
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from tqdm import tqdm
from collections import OrderedDict

os.chdir('/gpfs/cssb/user/alsaadiy/FKAN-Biostatistics')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check how many GPUs are being used
num_gpus = torch.cuda.device_count()
if num_gpus > 1:
    logger.info(f"Using {num_gpus} GPUs for training.")
else:
    logger.info("Using a single GPU for training.")
logger.info("=" * 100)

labels = ["PNEUMONIA", "NORMAL"]
img_size = 224


def get_training_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append([resized_arr, class_num])
            except Exception as e:
                logger.info(f"Error loading image {img}: {e}")
    data = np.array(data, dtype=object)
    return data


def flatten_images(images):
    return torch.flatten(images, start_dim=1)


# ------------------------------------------------------------------
# 2) DEFINE THE MODEL
# ------------------------------------------------------------------
# Now the data is ready for training and validation
class NaiveFourierKANLayer(torch.nn.Module):
    def __init__(self, inputdim, outdim, initial_gridsize, addbias=True):
        super(NaiveFourierKANLayer, self).__init__()
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim

        # Learnable gridsize parameter
        self.gridsize_param = torch.nn.Parameter(
            torch.tensor(initial_gridsize, dtype=torch.float32)
        )

        # Fourier coefficients as a learnable parameter with Xavier initialization
        self.fouriercoeffs = torch.nn.Parameter(
            torch.empty(2, outdim, inputdim, initial_gridsize)
        )
        torch.nn.init.xavier_uniform_(self.fouriercoeffs)

        if self.addbias:
            self.bias = torch.nn.Parameter(torch.zeros(1, outdim))

    def forward(self, x):
        gridsize = torch.clamp(self.gridsize_param, min=1).round().int()
        xshp = x.shape
        outshape = xshp[:-1] + (self.outdim,)
        x = torch.reshape(x, (-1, self.inputdim))
        k = torch.reshape(
            torch.arange(1, gridsize + 1, device=x.device), (1, 1, 1, gridsize)
        )
        xrshp = torch.reshape(x, (x.shape[0], 1, x.shape[1], 1))
        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)
        y = torch.sum(c * self.fouriercoeffs[0:1, :, :, :gridsize], (-2, -1))
        y += torch.sum(s * self.fouriercoeffs[1:2, :, :, :gridsize], (-2, -1))
        if self.addbias:
            y += self.bias
        y = torch.reshape(y, outshape)
        return y


class FourierKAN(torch.nn.Module):
    def __init__(self, params_list):
        super(FourierKAN, self).__init__()
        self.layer1 = torch.nn.Linear(params_list[0], params_list[1])
        torch.nn.init.xavier_uniform_(self.layer1.weight)
        self.bn1 = torch.nn.BatchNorm1d(1)
        self.relu1 = torch.nn.ReLU()
        self.layer2 = NaiveFourierKANLayer(
            params_list[1], params_list[2], initial_gridsize=16
        )
        self.layer3 = NaiveFourierKANLayer(
            params_list[2], params_list[3], initial_gridsize=8
        )
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        # print(x.shape)
        x = self.layer1(x)
        # print(x.shape)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = torch.squeeze(x, dim=1)
        x = self.softmax(x)
        return x


# ------------------------------------------------------------------
# 3) DEFINE TRAIN FUNCTION
# ------------------------------------------------------------------
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        model_name,
        num_epochs=100,
        patience=100,
):
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    model_dir = os.path.join("models", model_name)
    ensure_dir(model_dir)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Track metrics
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_preds += torch.sum(preds == labels).item()
                total_preds += labels.size(0)

                pbar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct_preds / total_preds
        train_losses.append(epoch_loss)

        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        logger.info(
            f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}"
        )

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct_preds = 0
        val_total_preds = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct_preds += torch.sum(preds == labels).item()
                val_total_preds += labels.size(0)

        val_loss /= len(val_loader.dataset)
        val_accuracy = val_correct_preds / val_total_preds
        val_losses.append(val_loss)

        logger.info(
            f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}"
        )

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            # Save best checkpoint
            model_path = os.path.join(model_dir, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            logger.info(f"Model saved to {model_path}")
        else:
            epochs_without_improvement += 1
            logger.info(
                f"Early Stopping Counter: {epochs_without_improvement}/{patience}"
            )
            logger.info("=" * 100)

        if epochs_without_improvement >= patience:
            logger.info("Early stopping triggered. Stopping training.")
            logger.info("=" * 100)
            break

    # Plot train vs val loss
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    loss_plot_path = os.path.join(model_dir, "loss_function.png")
    plt.savefig(loss_plot_path)
    logger.info(f"Loss plot saved to {loss_plot_path}")
    plt.close()


# ------------------------------------------------------------------
# 4) EVALUATION FUNCTION (FOR VAL & TEST)
# ------------------------------------------------------------------
def evaluate_model(model, dataloader, criterion):
    """
    Evaluate the model on a given DataLoader to obtain Loss, Accuracy,
    Precision, Recall, and F1-Score.
    """
    device = torch.device("cpu")  # Use CPU for evaluation
    model.to(device)  # Move the model to CPU
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, targets in dataloader:
            data = data.to(torch.float32).to(device)
            targets = targets.to(torch.long).to(device)

            outputs = model(data)
            loss = criterion(outputs, targets)

            # Accumulate total loss
            total_loss += loss.item() * data.size(0)

            _, preds = torch.max(outputs, 1)

            # Collect predictions and labels on CPU
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    # Ensure binary labels and predictions
    unique_labels = set(all_labels)
    unique_preds = set(all_preds)
    assert unique_labels <= {0, 1}, f"Invalid labels: {unique_labels}"
    assert unique_preds <= {0, 1}, f"Invalid predictions: {unique_preds}"

    # Compute average loss
    avg_loss = total_loss / len(dataloader.dataset)

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="binary")
    recall = recall_score(all_labels, all_preds, average="binary")
    f1 = f1_score(all_labels, all_preds, average="binary")

    return avg_loss, accuracy, precision, recall, f1


def normalize_images(data):
    images = []
    labels = []

    for img, label in tqdm(data):
        # Normalization: each pixel is divided by 255
        normalized_img = img / 255.0
        images.append(normalized_img)
        labels.append(label)

    # Convert the images and labels into separate arrays
    images = np.array(images)
    labels = np.array(labels)

    return images, labels


def save_metrics_to_file(file_path, phase, loss, acc, prec, rec, f1):
    """
    Save the evaluation metrics to a text file.
    phase can be 'Validation' or 'Test' or anything else.
    """
    with open(file_path, "a") as f:  # 'a' for append, 'w' for overwrite
        f.write(f"{phase} Metrics:\n")
        f.write(f"  Loss:      {loss:.4f}\n")
        f.write(f"  Accuracy:  {acc:.4f}\n")
        f.write(f"  Precision: {prec:.4f}\n")
        f.write(f"  Recall:    {rec:.4f}\n")
        f.write(f"  F1-Score:  {f1:.4f}\n\n")


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1) LOAD YOUR DATA
    # ------------------------------------------------------------------
    train_data = get_training_data("./normData/train")
    test_data = get_training_data("./normData/test")
    val_data = get_training_data("./normData/val")
    # Separate the images and the labels
    train_images_display = np.array([x[0] for x in train_data])  # Extract only the images
    train_labels_display = np.array([x[1] for x in train_data])  # Extract only the labels

    # Initialize counts for each label
    count_0 = 0
    count_1 = 0

    # Loop through the labels and count occurrences
    for label in train_labels_display:
        if label == 0:
            count_0 += 1
        elif label == 1:
            count_1 += 1

    # Print the results
    logger.info(f"Count of label 0: {count_0}")
    logger.info(f"Count of label 1: {count_1}")
    logger.info("=" * 100)

    val_images_display = np.array([x[0] for x in val_data])  # Extract only the images
    val_labels_display = np.array([x[1] for x in val_data])  # Extract only the labels

    count_0 = 0
    count_1 = 0

    # Loop through the labels and count occurrences
    for label in val_labels_display:
        if label == 0:
            count_0 += 1
        elif label == 1:
            count_1 += 1

    # Print the results
    logger.info(f"Count of label 0: {count_0}")
    logger.info(f"Count of label 1: {count_1}")
    logger.info("=" * 100)

    test_images_display = np.array([x[0] for x in test_data])  # Extract only the images
    test_labels_display = np.array([x[1] for x in test_data])  # Extract only the labels

    # Check the shape and an example of the dataset
    logger.info(f"Shape of train images: {train_images_display.shape}")
    logger.info(f"Shape of validation images: {val_images_display.shape}")
    logger.info("=" * 100)

    # =================

    # Normalize the images in the training dataset
    train_images, train_labels = normalize_images(train_data)
    val_images, val_labels = normalize_images(val_data)
    test_images, test_labels = normalize_images(test_data)

    # Check the shape and an example of the normalized and shuffled data
    logger.info(f"Shape of normalized and shuffled test images: {test_images.shape}")
    logger.info(f"Shape of normalized and shuffled train images: {train_images.shape}")
    logger.info(f"Shape of normalized and shuffled validation images: {val_images.shape}")
    logger.info("=" * 100)

    # ===============================

    # Convert the images and labels to PyTorch tensors

    # Apply the transformation to training and validation images
    train_images_tensor = torch.stack([torch.tensor(img, dtype=torch.float) for img in train_images]).unsqueeze(1)
    train_images_tensor = torch.flatten(train_images_tensor, start_dim=2)

    val_images_tensor = torch.stack([torch.tensor(img, dtype=torch.float) for img in val_images]).unsqueeze(1)
    val_images_tensor = torch.flatten(val_images_tensor, start_dim=2)

    test_images_tensor = torch.stack([torch.tensor(img, dtype=torch.float) for img in test_images]).unsqueeze(1)
    test_images_tensor = torch.flatten(test_images_tensor, start_dim=2)

    # Now permute them
    train_images_tensor = train_images_tensor.permute(0, 1, 2)  # (N, 1, 244 x 244)
    val_images_tensor = val_images_tensor.permute(0, 1, 2)
    test_images_tensor = test_images_tensor.permute(0, 1, 2)  # (N, 1, 244 x 244)
    logger.info(f"test data shape {test_images_tensor.shape}")
    logger.info("=" * 100)

    # The tensors are now in the shape (N, 1, 244, 244), where N is the number of images

    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
    val_labels_tensor = torch.tensor(val_labels, dtype=torch.long)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

    # Create the dataset and DataLoader
    train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
    val_dataset = TensorDataset(val_images_tensor, val_labels_tensor)
    test_dataset = TensorDataset(test_images_tensor, test_labels_tensor)

    # Define the batch size
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    logger.info(('DataLoader for train_loader and val_loader and test_loader done!'))
    logger.info("=" * 100)

    # -------------
    model_str = "FourierKAN"
    model = FourierKAN([224 * 224, 128, 64, 2])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Quick check of data shape
    for data_batch, target_batch in train_loader:
        data_batch, target_batch = data_batch.to(device), target_batch.to(device)
        logger.info(f"Shape of input data: {data_batch.shape}")
        output = model(data_batch)
        logger.info(f"Shape of model output: {output.shape}")
        logger.info("=" * 100)
        break

    # Define optimizer
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Define loss and create final DataLoader for training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-7, weight_decay=1e-5)

    model_name = "KAN_Fourier_Resnet"

    # Uncomment the following line to train:
    train_model(model, train_loader, val_loader, criterion, optimizer, model_name, num_epochs=100, patience=100)

    # -------------------------------
    # If you have trained and have "best_model.pth" saved:
    model_dir = os.path.join("models", model_name)
    best_model_path = os.path.join(model_dir, "best_model.pth")

    if os.path.exists(best_model_path):
        logger.info(f"Loading best model from: {best_model_path}")
        logger.info("=" * 100)
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)

        # Handle "module." prefix if trained with DataParallel
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            if k.startswith("module."):
                name = k[len("module."):]
            else:
                name = k
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        model.to(device)

        # 5A) Evaluate on Validation Set
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate_model(
            model, val_loader, criterion
        )
        logger.info("Validation Metrics:")
        logger.info(f"Loss:      {val_loss:.4f}")
        logger.info(f"Accuracy:  {val_acc:.4f}")
        logger.info(f"Precision: {val_prec:.4f}")
        logger.info(f"Recall:    {val_rec:.4f}")
        logger.info(f"F1-Score:  {val_f1:.4f}")
        logger.info("=" * 100)

        # Save Validation metrics to file
        model_dir = os.path.join("models", model_name)
        metrics_file = os.path.join(model_dir, "evaluation_metrics.txt")
        save_metrics_to_file(metrics_file, "Validation", val_loss, val_acc, val_prec, val_rec, val_f1)

        # 5B) Evaluate on Test Set
        test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate_model(
            model, test_loader, criterion
        )
        logger.info("Test Metrics:")
        logger.info(f"Loss:      {test_loss:.4f}")
        logger.info(f"Accuracy:  {test_acc:.4f}")
        logger.info(f"Precision: {test_prec:.4f}")
        logger.info(f"Recall:    {test_rec:.4f}")
        logger.info(f"F1-Score:  {test_f1:.4f}")
        logger.info("=" * 100)

        # Save Test metrics to file
        save_metrics_to_file(metrics_file, "Test", test_loss, test_acc, test_prec, test_rec, test_f1)
    else:
        logger.info("best_model.pth not found. Please train or check your path.")
