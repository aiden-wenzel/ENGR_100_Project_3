from utils import *
from sklearn.preprocessing import LabelEncoder
import os

def main():
    curr = "DIR/sample_audio_training/"
    print(curr)

    folders = ['oboe', 'trumpet', 'violin']
    files = []
    labels = []

    for folder in folders:
        folderPath = os.path.join(curr, folder)
        for filename in os.listdir(folderPath):
            file_path = os.path.join(folderPath, filename)
            if os.path.isfile(file_path):  # Make sure it's a file, not a directory or a symlink
                files.append(file_path)
                labels.append(folder)

    label_encoder = LabelEncoder()
    numeric_labels = label_encoder.fit_transform(labels)

    process_and_save_audio(files=files, labels=numeric_labels, output_path="data.npz", sr=22050, add_noise=False)
    
if __name__ == "__main__":
    main()
    