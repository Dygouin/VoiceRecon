import pandas as pd
import numpy as np
import librosa

# Define a frame size
FRAME_SIZE = 2048

# Define a hop size.
HOP_SIZE = 512

# Define the default sampling rate of librosa
SR = 22050

N_MELS = 110

N_MFCCS = 13

def create_mel_spectogram(signal, sr, n_fft, hop_size, n_mels):
    """Function that creates a log mel spectogram"""

    # Get vanilla mel spectogram
    mel_spectogram = librosa.feature.melspectrogram(signal, sr=sr, n_fft= n_fft, hop_length= hop_size, n_mels= n_mels)

    # Transform vanilla mel spectogram into log spectogram.
    log_mel_spectogram = librosa.power_to_db(mel_spectogram)

    return np.array(log_mel_spectogram)

def create_MFCCS(signal, n_mfccs, sr):
    """Define function to create MFCCS"""

    #Use librosa library to generate MFCCS
    mfccs = librosa.feature.mfcc(signal, n_mfcc= n_mfccs, sr=sr)
    return mfccs

def delta_MFCCS(mfccs):
    """Function that gets the first order derivative of MFCCS"""
    return librosa.feature.delta(mfccs)

def delta2_MFCCS(delta_mfccs):
    """Function that returns the second order derivative of MFCCS"""
    return librosa.feature.delta(delta_mfccs, order=2)

def create_comprehensive_MFCCS(signal, n_mfccs, sr):
    """Function that returns MFCCS numpy array with first and second order derivative."""
    mfccs = create_MFCCS(signal, n_mfccs, sr)
    delta_mfccs= delta_MFCCS(mfccs)
    delta2_mfccs = delta2_MFCCS(delta_mfccs)
    comprehensive_mfccs = np.concatenate((mfccs, delta_mfccs, delta2_mfccs))
    return np.array(comprehensive_mfccs)

def create_chromas(signal, sr):
    """Function that returns a chroma given a signal"""
    chroma = librosa.feature.chroma_stft(signal, sr)
    return np.array(chroma)

def create_full_features(mel_spectogram, comprehensive_mfccs, chroma):
    full_features = np.vstack((mel_spectogram, comprehensive_mfccs, chroma))
    full_features = full_features.reshape(1, 161, 176, 1)
    return full_features
