from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib as plt
import numpy as np

import torch

class confuse_matrix:
    def __init__(self, model, testloader, classes):
        self.y_pred = []
        self.y_true = []

        for inputs, labels in testloader:
            ouput = model(inputs) # feed forward

            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output) # save prediction

            labels = labels.data.cpu().numpy()
            y_true.extend(labels) # save truth

        self.confuse_matrix = confusion_matrix(y_true, y_pred)
        self.df_confuse_matrix = pd.DataFrame(
                self.confuse_matrix / np.sum(self.confuse_matrix, axis=1)[:, None],
                index = [i for i in classes],
                columns = [i for i in classes]
        )

        self.fig = plt.figure(figsize = (12,7))
        self.heatmap = sn.heatmap(df_confuse_matrix, annot=True)

    def show():
        plt.show()

    def save():
        plt.savefig('output.png')
