from utils import *
from sklearn.preprocessing import LabelEncoder
import tkinter as tk
import os
from tkinter import filedialog
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import Sequential
from torch.nn import BatchNorm2d
from torch.nn import ELU
from torch import flatten
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

class MILR_APP():
    def __init__(self, root, model):
        self.model = model
        self.predicted_labels = []
        self.file_path = ""
        self.root = root
        self.setup_ui()
    
    def setup_ui(self):
        self.root.title("MLIR")
        self.root.minsize(640, 360)
        self.root.maxsize(640, 360)

        # test button
        self.test_button = tk.Button(self.root, text="Predict", command=self.output_predicted_values)
        self.test_button.place(x=20, y=180)

        # file prompt button
        self.file_prompt_button = tk.Button(self.root, text="Input File", command=self.file_prompt)
        self.file_prompt_button.place(x=20, y=20)

        # file input text box
        self.input_output_box = tk.Text(self.root, font=("Helvetica", "12"), width=65, height=4)
        self.input_output_box.place(x=20, y=60)
        self.input_output_box['state'] = 'disabled'

        # output text box
        self.test_output = tk.Text(self.root, font=("Helvetica", "16"), width=49, height=4)
        self.test_output.place(x=20, y=220)
        self.test_output['state'] = 'disabled'

    def predict_labels(self):
        if (self.file_path == 1):
            print("no file was selected")
        
        else:
            self.predicted_labels = prediction(self.model, self.file_path)

    def output_predicted_values(self):
        self.test_output['state'] = 'normal'
        self.test_output.delete('1.0', tk.END)

        self.predict_labels()
        for label in self.predicted_labels:
             self.test_output.insert(tk.END, f"{label} predicted\n")

        self.test_output['state'] = 'disabled'

    def file_prompt(self):
        self.input_output_box['state'] = 'normal'
        self.input_output_box.delete('1.0', tk.END)
        file_path = filedialog.askopenfilename()
        # Check if a file was selected
        if file_path:
            split_path = file_path.split("/", -1)
            file = split_path[-1]
            self.input_output_box.insert(tk.END, f"File selected: {file}")
            self.file_path = file_path
        else:
            self.input_output_box.insert(tk.END, "No file was selected")

        self.input_output_box['state'] = 'disabled'

def main():
    # make model
    model = MultiLabelCNN(3)
    model.load_state_dict(torch.load("../pretrained_models/CNN/cnn.pkl"))

    root = tk.Tk()
    app = MILR_APP(root, model)
    root.mainloop()

if __name__ == "__main__":
    main()
    
