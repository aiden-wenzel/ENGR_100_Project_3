import h5py
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
import utils
from torch.nn import BCEWithLogitsLoss
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import CNN
import os

curr = "../sample_audio_training/train/"

folders = ['oboe', 'trumpet', 'violin']

utils.create_hdf5("../sample_audio_training/train", folders, "data.h5")

h5_path = "./data.h5"
mel_dataset = CNN.FrameDataset(h5_path)

print("split dataset")
# split dataset into training, validation, and testing
# sizes of 80% 10% and 10%
train_size = int(0.8 * len(mel_dataset))
validation_size = int(0.1 * len(mel_dataset))
test_size = len(mel_dataset) - train_size - validation_size

# split data sets into their respective sizes
train_dataset, validation_dataset, test_dataset = random_split(
    mel_dataset, [train_size, validation_size, test_size]
)

print("create dataloader")
train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
validation_dataloader = DataLoader(
    validation_dataset, batch_size=32, shuffle=True
)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

print("make model")
# make model
num_classes = 3
model = CNN.MultiLabelCNN(num_classes)
criterion = BCEWithLogitsLoss() # loss function
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9) # optimizer

print("training")
train_accuracies, validation_accuracies = CNN.train_model(
    model, train_dataloader, validation_dataloader, criterion, optimizer, epochs=20
)

model.save_model("./cnn.pkl")
