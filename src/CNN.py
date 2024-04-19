# MODEL DEFINITON

from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import Sequential
from torch.nn import BatchNorm2d
from torch.nn import ELU
from torch import flatten
from torch import save
from torch import load
import logging
from utils import *
from torch.nn import BCEWithLogitsLoss
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch

class MultiLabelCNN(Module):
    def __init__(self, num_classes):
        super(MultiLabelCNN, self).__init__()
        
        self.hidden_layers = Sequential (
            Conv2d(in_channels=1, out_channels=64, kernel_size=(3,3)),
            BatchNorm2d(64),
            ELU(),
            MaxPool2d((2,2)),

            Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3)),
            BatchNorm2d(128),
            ELU(),
            MaxPool2d((2, 2)),

            Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3)),
            BatchNorm2d(256),
            ELU(),
            MaxPool2d((3, 3)),

            Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3)),
            BatchNorm2d(256),
            ELU(),
            MaxPool2d((3, 3)),
        )

        self.linear_layers = Sequential(
            Linear(in_features=256, out_features=128),  # Adjusted input features to match flattened conv output
            ELU(),
            Linear(in_features=128, out_features=num_classes)
        )

    def forward(self, x):
        # Pass input through the convolutional layers
        x = self.hidden_layers(x)
        
        # Flatten the output of the convolutional layers to fit linear layer input
        x = flatten(x, 1)  # Flatten all dimensions except the batch
        
        # Pass data through linear layers
        x = self.linear_layers(x)
        
        return x
    
    def save_model(self, path: str):
        save(self.state_dict(), path)
        print(f"Model saved to {path}")


    def load_model(*parameters, path: str):
        loaded_model = MultiLabelCNN(parameters)

        # Then, load the saved state dict
        loaded_model.load_state_dict(load(path))

        return loaded_model
    
class FrameDataset(Dataset):
    def __init__(self, file_path):
        """
        Initialize dataset.
        :param file_path: Path to the HDF5 file.
        """
        self.file_path = file_path
        self.file = h5py.File(self.file_path, 'r')
        self.features = self.file['features']
        self.labels = self.file['labels']
    
    def __len__(self):
        """
        Return the total number of samples.
        """
        return int(self.file['overall_metadata'][2])
    
    def __getitem__(self, idx):
        """
        Fetch the data and labels at the specified index.
        """
        feature = torch.tensor(self.features[idx], dtype=torch.float32).reshape(1, 96, 87)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return feature, label
    
    def close(self):
        """
        Close the HDF5 file.
        """
        self.file.close()

def train_model(
    model, train_dataloader, validation_dataloader, criterion, optimizer, epochs=20
) -> Tuple[list, list]:
    train_accuracies = []
    validation_accuracies = []
    #Start of time remaining
    start_time = time.time()

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
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
        print(f"Loss: {total_loss}")
        print(f"Train Accuracy: {train_accuracy.item()}")
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
            print(f"Validation Accuracy: {validation_accuracy.item()}")
            validation_accuracies.append(validation_accuracy.item())
            
        # Time remaining counter
        time_remaining = (epochs - (epoch+1)) * (
            time.time() - start_time
        )/(epoch+1)
        hours, remainder = divmod(time_remaining, 3600)
        minutes, seconds = divmod(remainder, 60)
        # Formatted time output
        print(
            f"Time remaining: {int(hours):02}:{int(minutes):02}:{int(seconds):02}"
        )

    return train_accuracies, validation_accuracies

def load_npz_file_with_condition(file_path, max_size: int):

    file_size = os.path.getsize(file_path)

    if file_size > max_size:
        print(
            f"File size is {file_size / (1024**2):.2f}MB. Using mmap_mode='r'."
        )
        data = np.load(file_path, mmap_mode="r", allow_pickle=True)
    else:
        print(f"File size is {file_size / (1024**2):.2f}MB. Loading normally.")
        data = np.load(file_path, allow_pickle=True)

    return data