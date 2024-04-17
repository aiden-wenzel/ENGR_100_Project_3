curr = os.getcwd()
if (os.name == "nt"):
    curr = curr.replace("\src", "\sample_audio_training")
else:
    curr = curr.replace("/src", "/sample_audio_training")

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