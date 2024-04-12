import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torch.utils.data import Dataset
from typing import Tuple
import matplotlib.pyplot as plt

class MultiLabelMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(MultiLabelMLP, self).__init__()

        # Initialize the ModuleList for hidden layers
        self.hidden_layers = nn.ModuleList()

        # The dropout layer
        self.dropout = nn.Dropout(p=0.5)

        # Create the first hidden layer
        self.hidden_layers.append(nn.Linear(input_size, hidden_sizes[0]))

        # Create subsequent hidden layers
        for h1, h2 in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            self.hidden_layers.append(nn.Linear(h1, h2))

        # Create the output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], num_classes)

    def forward(self, x):
        # Apply each hidden layer with ReLU activation and dropout
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            x = self.dropout(x)  # Apply dropout after activation

        # Apply the output layer
        x = self.output_layer(x)
        return x


class FrameDataset(Dataset):
    def __init__(self, npz_path):
        self.data = utils.load_npz_file_with_condition(npz_path, max_size=1024**3)
        self.keys = [k for k in self.data.keys() if "_data" in k]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        data_key = self.keys[idx]
        data = self.data[data_key]
        labels = self.data[f'{data_key.split("_data_")[0]}_labels']
        return torch.tensor(data.reshape(-1), dtype=torch.float32), torch.tensor(
            labels, dtype=torch.float32
        )


def train_model(
    model, train_dataloader, validation_dataloader, criterion, optimizer, epochs=5
) -> Tuple[list, list]:
    train_accuracies = []
    validation_accuracies = []

    for epoch in range(epochs):
        logging.info(f"Epoch {epoch+1}")
        model.train()  # Set model to training mode
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        for data, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted = torch.sigmoid(outputs) > 0.5
            correct_predictions += (predicted == labels).float().sum()
            total_predictions += torch.numel(labels)

        train_accuracy = correct_predictions / total_predictions
        logging.info(f"Loss: {total_loss}")
        logging.info(f"Train Accuracy: {train_accuracy.item()}")
        train_accuracies.append(train_accuracy.item())

        # Validation phase
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            correct_predictions = 0
            total_predictions = 0
            for data, labels in validation_dataloader:
                outputs = model(data)
                predicted = torch.sigmoid(outputs) > 0.5
                correct_predictions += (predicted == labels).float().sum()
                total_predictions += torch.numel(labels)

            validation_accuracy = correct_predictions / total_predictions
            logging.info(f"Validation Accuracy: {validation_accuracy.item()}")
            validation_accuracies.append(validation_accuracy.item())

    return train_accuracies, validation_accuracies


def plot_accuracy(
    train_accuracies: list, validation_accuracies: list, epoch_count: int
):
    epochs = range(1, epoch_count + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracies, label="Training Accuracy")
    plt.plot(epochs, validation_accuracies, label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


def test_model(model, test_dataloader):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        correct_predictions = 0
        total_predictions = 0
        for data, labels in test_dataloader:
            outputs = model(data)
            predicted = torch.sigmoid(outputs) > 0.5
            correct_predictions += (predicted == labels).float().sum()
            total_predictions += torch.numel(labels)
        logging.info(
            f"Test Accuracy: {(correct_predictions / total_predictions).item()}"
        )


def save_model(model, path: str):
    torch.save(model.state_dict(), path)
    logging.info(f"Model saved to {path}")


def load_model(*parameters, path: str):
    loaded_model = MultiLabelMLP(parameters)

    # Then, load the saved state dict
    loaded_model.load_state_dict(torch.load(path))

    return loaded_model