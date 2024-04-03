# %%
# Full work file as py (for easier import)
import librosa
import numpy as np
from typing import Tuple
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import tempfile
import os
import tensorflow as tf
import tensorflow_datasets as tfds
print(tf.__version__)

# %%
def combine_tracks(path_1:str, path_2:str, mult_1=1, mult_2=1, sr=22050) -> Tuple[np.array, int]:
    y_1, sr_1 = librosa.load(path_1, sr=sr)
    y_2, sr_2 = librosa.load(path_2, sr=sr)
    if(y_1.size > y_2.size):
        y_1 = np.pad(y_1, (0,sr_1 - (y_1.size%sr_1)), 'constant', constant_values= (0))
        y_2 = np.pad(y_2, (0,y_1.size-y_2.size), 'constant', constant_values= (0))
    else:
        y_2 = np.pad(y_2, (0,sr_2 - (y_2.size%sr_2)), 'constant', constant_values= (0))
        y_1 = np.pad(y_1, (0,y_2.size-y_1.size), 'constant', constant_values= (0))
    y = np.add(y_1 * mult_1, y_2 * mult_2)
    
    return (y, sr)

# 46.4 ms block size
def get_mel_spectograms(y:np.array, sr:int, block_size:float, hop_size:float) -> np.array:
    return librosa.power_to_db(
        librosa.feature.melspectrogram(
            y = y.reshape((int) (y.size/sr), sr),
            sr=sr,
            n_fft=int(block_size*1e-3*sr), 
            hop_length = int(hop_size*1e-3*sr), 
            n_mels = 96
        ),
        ref=np.max
    )
    

# %%
y, sr = combine_tracks("sample_audio_training/trumpet/0005.wav", "sample_audio_training/violin/0010.wav")
out = get_mel_spectograms(y, sr)
out.shape


# %%
librosa.display.specshow(out[2], sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')

# %% [markdown]
# Model Time :)

# %%
tf.random.set_seed(22)

# %%



