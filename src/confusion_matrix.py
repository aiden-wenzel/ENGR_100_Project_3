from sklearn.metrics import multilabel_confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt

import torch

class MultiLabelConfusionMatrix:
    def __init__(self, model, testloader, classes):
        self.classes = classes

        self.y_pred = []
        self.y_true = []

        for inputs, labels in testloader:
            output = model(inputs) # feed forward
            predicted = torch.sigmoid(output) > 0.5
            self.y_pred.extend(predicted.numpy().astype(int)) # save predictions

            labels = labels.data.cpu().numpy()
            self.y_true.extend(labels.astype(int)) # save truth

        self.confuse_matrices = multilabel_confusion_matrix(self.y_true, self.y_pred)

def plot_confusion_matrices(cm: MultiLabelConfusionMatrix, classes: list):
    for idx, cm in enumerate(cm.confuse_matrices):
        plt.figure(figsize=(2, 2))
        sn.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title(f'confusion matrix {classes[idx]}')
        plt.savefig(f'{classes[idx]}.png')
        plt.close()