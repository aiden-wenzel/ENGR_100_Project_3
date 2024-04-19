import librosa
import numpy as np
from typing import Tuple
import os
import logging
from sklearn.preprocessing import LabelEncoder
import sys
import time
import h5py
import torch


def combine_tracks(
    path_1: str, path_2: str, mult_1=0.5, mult_2=0.5, sr=22050
) -> Tuple[np.array, int]:
    """
    Combines two audio tracks into one by adding them together after optional scaling.

    Parameters:
    - path_1 (str): Path to the first audio file.
    - path_2 (str): Path to the second audio file.
    - mult_1 (float, optional): Multiplier for the first audio track's amplitude. Default is 1.
    - mult_2 (float, optional): Multiplier for the second audio track's amplitude. Default is 1.
    - sr (int, optional): Sampling rate to use when loading the audio files. Default is 22050 Hz.

    Returns:
    - Tuple[np.array, int]: A tuple containing the combined audio array and the sampling rate.
    """
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
    """
    Generates a decibel-scaled spectrogram for a given audio array.

    Parameters:
    - y (np.array): The audio signal array.
    - sr (int, optional): Sampling rate of the audio signal. Default is 22050 Hz.

    Returns:
    - np.array: A decibel-scaled spectrogram of the input audio.
    """
    return librosa.amplitude_to_db(
        librosa.feature.melspectrogram(
            y=y.reshape((int)(y.size / sr), sr),  # split into 1-second intervals
            hop_length=int(0.0116 * sr),
            n_fft=int(0.0464 * sr),
            n_mels=96,
        )
    )


def add_gaussian_noise(data: np.array, std: float) -> Tuple[np.array, float]:
    """
    Adds Gaussian noise to an array.

    Parameters:
    - data (np.array): Array of audio data.
    - std (float): Standard deviation of the Gaussian noise.

    Returns:
    - np.array: Noisy version of the input array.
    """
    original_signal = np.sum(np.square(data))
    noise = np.random.normal(0, std, data.shape)
    noisy_signal = data + noise
    return noisy_signal, 10 * np.log10(original_signal / np.sum(np.square(noisy_signal)))


def process_and_save_audio_hdf5(
    files: list, labels: list, output_path: str, sr=22050, noise: float = None
) -> None:
    """
    Processes a list of audio files and their labels, and saves the resulting dataset in an HDF5 file format.

    Parameters:
    - files (list): A list of paths to directories or individual audio files to process.
    - labels (list): A list of labels corresponding to the audio files. Each label should correspond to the directory or file in the 'files' list.
    - output_path (str): The path where the HDF5 file will be saved.
    - sr (int, optional): Sampling rate to be used for audio files. Default is 22050 Hz.
    - noise (float, optional): If provided, adds Gaussian noise to the audio data with parameter as standard deviation. Default is None / 0.

    This function iterates through each audio file, generates spectrograms, optionally adds noise, and saves these features along with their labels into an HDF5 file. If the file path is a directory, it processes all audio files within that directory. The function uses a helper function to generate spectrograms and to handle audio combinations. It also calculates and stores overall dataset statistics such as mean and standard deviation for normalization purposes.

    Raises:
    - Exception: If the number of provided files and labels do not match.

    The function logs the progress of data processing, including the estimated time remaining and the final completion status.
    """
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
        avg_SNR = 0.0
        SNR_index = 1

        for i in range(len(files_to_process)):
            for j in range(i + 1, len(files_to_process)):
                y, y_labels, SNR = process_and_save_audio_helper(
                    file_path=files_to_process[i],
                    file_path_2=files_to_process[j],
                    label=labels_to_process[i],
                    label_2=labels_to_process[j],
                    label_size=label_size,
                    noise=noise,
                )
                if sample_count == 0:
                    hdf.create_dataset(
                        "features",
                        data=y,
                        compression="gzip",
                        chunks=True,
                        maxshape=(None, 96, 87),
                    )
                    hdf.create_dataset(
                        "labels",
                        data=y_labels,
                        compression="gzip",
                        chunks=True,
                        maxshape=(None, 3),
                    )
                else:
                    hdf["features"].resize(
                        (hdf["features"].shape[0] + y.shape[0]), axis=0
                    )
                    hdf["features"][-y.shape[0] :] = y

                    hdf["labels"].resize(
                        (hdf["labels"].shape[0] + y_labels.shape[0]), axis=0
                    )
                    hdf["labels"][-y_labels.shape[0] :] = y_labels

                # Calculate metadata of overall mean, overall standard deviation, and overall SNR
                n += y.size
                delta = y - mean
                mean += delta.sum() / n
                delta2 = y - mean
                M2 += (delta * delta2).sum()
                if(noise is not None):
                    avg_SNR = (avg_SNR * (SNR_index - 1) / SNR_index) + SNR / SNR_index
                    SNR_index += 1

                sample_count += y.shape[0]

            y, y_labels, SNR = process_and_save_audio_helper(
                file_path=files_to_process[i],
                label=labels_to_process[i],
                label_size=label_size,
                noise=noise,
            )
            hdf["features"].resize((hdf["features"].shape[0] + y.shape[0]), axis=0)
            hdf["features"][-y.shape[0] :] = y

            hdf["labels"].resize((hdf["labels"].shape[0] + y_labels.shape[0]), axis=0)
            hdf["labels"][-y_labels.shape[0] :] = y_labels
            n += y.size
            delta = y - mean
            mean += delta.sum() / n
            delta2 = y - mean
            M2 += (delta * delta2).sum()
            if(noise is not None):
                avg_SNR = (avg_SNR * (SNR_index - 1) / SNR_index) + SNR / SNR_index
                SNR_index += 1

            sample_count += y.shape[0]

            # Track progress
            if i / len(files_to_process) > progress:
                time_remaining = (
                    (len(files_to_process) - i) * (time.time() - start_time) / i
                )
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
            hdf["features"][i] = (hdf["features"][i] - overall_mean) / overall_stddev

        hdf.create_dataset(
            "overall_metadata",
            data=np.array([overall_mean, overall_stddev, sample_count, avg_SNR]),
        )
        
        logging.info(
            f"\n\tMetadata: {[overall_mean, overall_stddev, sample_count, avg_SNR]}"
        )

        logging.info(f"\n\tTraining data creation progress: [100%]")

        # Log progress or any important info
        logging.info(
            f"\n\tAudio processing and saving completed successfully.\n\tDataset saved to {output_path} with size {os.path.getsize(output_path)/(1024**2):.2f}MB"
        )


def process_and_save_audio_helper(
    file_path: str,
    label: int,
    label_size: int,
    file_path_2: str = None,
    label_2: int = None,
    sr=22050,
    noise: float = None,
) -> Tuple[np.array, np.array, float]:
    if file_path_2 is not None:
        audio, _ = combine_tracks(file_path, file_path_2, sr=sr)
    else:
        audio, _ = librosa.load(file_path, sr=sr)
        audio = np.pad(
            audio, (0, sr - (audio.size % sr)), "constant", constant_values=(0)
        )

    SNR = 0.0

    # Optionally add Gaussian noise
    if noise is not None:
        audio, SNR = add_gaussian_noise(audio, noise)

    y = DB_spectogram(audio)

    labels = np.full((y.shape[0], label_size), False)
    labels[:, label] = True
    if label_2 is not None:
        labels[:, label_2] = True

    return y, labels, SNR


def create_hdf5(
    Training_Data_Directory: str,
    Training_Data_Sub_Directories: list,
    output_path: str,
    sr=22050,
    noise:float = None,
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
        noise=noise,
    )

# data is a n by 96 by 87 numpy array
def prediction(model, wav_path: str) -> list:
    classes = ["oboe", "trumpet", "violin"]
    data = process_audio(wav_path)
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        # oboe trumpet violin
        outputs = model(data)
        
        binary_classifications = torch.sigmoid(outputs) > 0.5
        # max_values = np.amax(predicted_weights, axis=0)
        # binary_classifications = max_values > 0.5
        binary_classifications = binary_classifications.numpy()
        binary_classifications = binary_classifications.squeeze()
        predicted_labels = []
        for i in range(len(binary_classifications)):
            if binary_classifications[i]:
                predicted_labels.append(classes[i])
        
        return predicted_labels
    
def process_audio(wav_path: str, sr = 22050):
    audio_signal_array, sr = librosa.load(wav_path, sr=sr)
    audio_signal_array = np.pad(
            audio_signal_array, (0, sr - (audio_signal_array.size % sr)), "constant", constant_values=(0)
        )
    spectrogram_matrix = DB_spectogram(audio_signal_array, sr=sr)
    average_spectrogram = np.mean(spectrogram_matrix, axis=0)
    average_spectrogram = np.expand_dims(average_spectrogram, axis=0)
    spectrogram_tensor = torch.tensor(average_spectrogram)
    spectrogram_tensor = spectrogram_tensor.unsqueeze(0)
    return spectrogram_tensor