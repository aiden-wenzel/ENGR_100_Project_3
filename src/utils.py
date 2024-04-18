import librosa
import numpy as np
from typing import Tuple
import os
import logging
from sklearn.preprocessing import LabelEncoder
import sys
import time
import h5py


def combine_tracks(
    path_1: str, path_2: str, mult_1=0.5, mult_2=0.5, sr=22050
) -> Tuple[np.array, int]:
    y_1, sr_1 = librosa.load(path_1, sr=sr)
    y_2, sr_2 = librosa.load(path_2, sr=sr)
    if y_1.size > y_2.size:
        y_1 = np.pad(
            y_1, (0, sr_1 - (y_1.size % sr_1)), "constant", constant_values=(0)
        )
        y_2 = np.pad(y_2, (0, y_1.size - y_2.size), "constant", constant_values=(0))
    else:
        y_2 = np.pad(
            y_2, (0, sr_2 - (y_2.size % sr_2)), "constant", constant_values=(0)
        )
        y_1 = np.pad(y_1, (0, y_2.size - y_1.size), "constant", constant_values=(0))
    y = np.add(y_1 * mult_1, y_2 * mult_2)

    return (y, sr)


def DB_spectogram(y: np.array, sr=22050) -> np.array:
    return librosa.amplitude_to_db(
        librosa.feature.melspectrogram(
            y=y.reshape((int)(y.size / sr), sr),  # split into 1-second intervals
            hop_length=int(0.0116 * sr),
            n_fft=int(0.0464 * sr),
            n_mels=96,
        )
    )


def add_gaussian_noise(data: np.array, std=0.005):
    """
    Adds Gaussian noise to an array.
    :param data: numpy array of audio data.
    :param mean: Mean of the Gaussian noise.
    :param std: Standard deviation of the Gaussian noise.
    :return: Noisy numpy array.
    """
    noise = np.random.normal(0, std, data.shape)
    return data + noise


def process_and_save_audio_hdf5(
    files: list, labels: list, output_path: str, sr=22050, add_noise=False
):
    if len(files) != len(labels):
        raise Exception("Length of files must equal labels length")

    # Creating the actual list of files and labels to process
    files_to_process = []
    labels_to_process = []
    for i, f in enumerate(files):
        l = labels[i]
        if os.path.isdir(f):
            for dirpath, _, filenames in os.walk(f):
                for filename in filenames:
                    full_path = os.path.join(dirpath, filename)
                    files_to_process.append(full_path)
                    labels_to_process.append(l)
        else:
            files_to_process.append(f)
            labels_to_process.append(l)

    # Open a new HDF5 file
    with h5py.File(output_path, "w") as hdf:
        
        label_size = max(labels) + 1
        progress = 0
        start_time = time.time()
        sample_count = 0
        
        n = 0
        mean = 0
        M2 = 0
        
        for i in range(len(files_to_process)):
            for j in range(i + 1, len(files_to_process)):
                y, y_labels = process_and_save_audio_helper(
                    file_path=files_to_process[i],
                    file_path_2=files_to_process[j],
                    label=labels_to_process[i],
                    label_2=labels_to_process[j],
                    label_size=label_size,
                    add_noise=add_noise,
                )
                if sample_count == 0:
                    hdf.create_dataset('features', data=y, compression="gzip", chunks=True, maxshape=(None,96,87))
                    hdf.create_dataset('labels', data=y_labels, compression="gzip", chunks=True, maxshape=(None,3))
                else:
                    hdf['features'].resize((hdf['features'].shape[0] + y.shape[0]), axis=0)
                    hdf['features'][-y.shape[0]:] = y
                    
                    hdf['labels'].resize((hdf['labels'].shape[0] + y_labels.shape[0]), axis=0)
                    hdf['labels'][-y_labels.shape[0]:] = y_labels
                
                n += y.size
                delta = y - mean
                mean += delta.sum() / n
                delta2 = y - mean
                M2 += (delta * delta2).sum()
                
                sample_count += y.shape[0]
            
            y, y_labels = process_and_save_audio_helper(
                file_path=files_to_process[i],
                label=labels_to_process[i],
                label_size=label_size,
                add_noise=add_noise,
            )
            hdf['features'].resize((hdf['features'].shape[0] + y.shape[0]), axis=0)
            hdf['features'][-y.shape[0]:] = y
            
            hdf['labels'].resize((hdf['labels'].shape[0] + y_labels.shape[0]), axis=0)
            hdf['labels'][-y_labels.shape[0]:] = y_labels
            n += y.size
            delta = y - mean
            mean += delta.sum() / n
            delta2 = y - mean
            M2 += (delta * delta2).sum()
            
            sample_count += y.shape[0]

            #Track progress
            if i / len(files_to_process) > progress:
                time_remaining = (len(files_to_process) - i) * (
                    time.time() - start_time
                )/i
                hours, remainder = divmod(time_remaining, 3600)
                minutes, seconds = divmod(remainder, 60)
                # Formatted time output
                logging.info(
                    f"\r\n\tTraining data creation progress: [{int(progress*100)}%]\n\tTime remaining: {int(hours):02}:{int(minutes):02}:{int(seconds):02}"
                )
                progress += 0.1
                
        overall_mean = mean
        overall_stddev = np.sqrt(M2 / n) if n > 1 else 0.0
        
        for i in range(sample_count):
            hdf['features'][i] = (hdf['features'][i] - overall_mean) / overall_stddev
                
        hdf.create_dataset(
            "overall_metadata", data=np.array([overall_mean, overall_stddev, sample_count])
        )

        logging.info(f"\n\tTraining data creation progress: [100%]")

        # Log progress or any important info
        logging.info(f"\n\tAudio processing and saving completed successfully.\n\tDataset saved to {output_path} with size {os.path.getsize(output_path)/(1024**2):.2f}MB")


def process_and_save_audio_helper(
    file_path: str,
    label: int,
    label_size: int,
    file_path_2: str = None,
    label_2: int = None,
    sr=22050,
    add_noise=False,
):
    if file_path_2 is not None:
        audio, _ = combine_tracks(file_path, file_path_2, sr=sr)
    else:
        audio, _ = librosa.load(file_path, sr=sr)
        audio = np.pad(
            audio, (0, sr - (audio.size % sr)), "constant", constant_values=(0)
        )

    # Optionally add Gaussian noise
    if add_noise:
        audio = add_gaussian_noise(audio)

    y = DB_spectogram(audio)

    labels = np.full((y.shape[0], label_size), False)
    labels[:, label] = True
    if label_2 is not None:
        labels[:, label_2] = True

    return y, labels



def create_hdf5(
    Training_Data_Directory: str,
    Training_Data_Sub_Directories: list,
    output_path: str,
    sr=22050,
    add_noise=False,
):
    
    """
    Creates an HDF5 file containing audio data for training purposes.

    Parameters:
    - Training_Data_Directory (str): The root directory containing all the training data subdirectories.
    - Training_Data_Sub_Directories (list): A list of subdirectory names under the Training_Data_Directory that contain the actual audio files.
    - output_path (str): The path where the HDF5 file will be saved.
    - sr (int, optional): The sampling rate to use for audio files. Default is 22050 Hz.
    - add_noise (bool, optional): A flag indicating whether to add synthetic noise to the audio data. Default is False.

    The function constructs a list of folder paths based on the given subdirectory names, and assigns labels to these folders.
    Each folder's name is used as a label which is then transformed into numeric format using LabelEncoder.
    Logging is used to print out the paths and numeric labels of the training data.
    Finally, it calls a helper function `process_and_save_audio_hdf5` to process the audio files and save them in the specified HDF5 format at the given output path.
    """
    
    Training_Data_Folder_Paths = []
    Training_Data_Folder_Labels = []

    for folder in Training_Data_Sub_Directories:
        folderPath = os.path.join(Training_Data_Directory, folder)
        Training_Data_Folder_Paths.append(folderPath)
        Training_Data_Folder_Labels.append(folder)

    Training_Data_Folder_Numeric_Labels = LabelEncoder().fit_transform(
        Training_Data_Folder_Labels
    )

    logging.info(
        f"\nTraining Data\n\tFolders:\n\t\t{Training_Data_Folder_Paths}\n\tRespective Labels:\n\t\t{Training_Data_Folder_Numeric_Labels}"
    )

    process_and_save_audio_hdf5(
        files=Training_Data_Folder_Paths,
        labels=Training_Data_Folder_Numeric_Labels,
        output_path=output_path,
        sr=sr,
        add_noise=add_noise,
    )