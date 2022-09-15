import re
import os
from tkinter import X
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import scipy
import json
import copy
import sys
import pickle
import string
import logging
from functools import lru_cache
from copy import copy

import librosa
import resampy
import soundfile as sf

import torch


def remove_drift(signal, fs):
    '''
    From David Gaddy's https://github.com/dgaddy/subvocal
    '''
    b, a = scipy.signal.butter(3, 2, 'highpass', fs=fs)
    return scipy.signal.filtfilt(b, a, signal)


def notch(signal, freq, sample_frequency):
    '''
    From David Gaddy's https://github.com/dgaddy/subvocal
    '''
    b, a = scipy.signal.iirnotch(freq, 30, sample_frequency)
    return scipy.signal.filtfilt(b, a, signal)


def notch_harmonics(signal, freq, sample_frequency):
    '''
    From David Gaddy's https://github.com/dgaddy/subvocal
    '''
    for harmonic in range(1,8):
        signal = notch(signal, freq*harmonic, sample_frequency)
    return signal


def subsample(signal, new_freq, old_freq):
    '''
    From David Gaddy's https://github.com/dgaddy/subvocal
    '''
    times = np.arange(len(signal))/old_freq
    sample_times = np.arange(0, times[-1], 1/new_freq)
    result = np.interp(sample_times, times, signal)
    return result


def apply_to_all(function, signal_array, *args, **kwargs):
    '''
    From David Gaddy's https://github.com/dgaddy/subvocal
    '''
    results = []
    for i in range(signal_array.shape[1]):
        results.append(function(signal_array[:,i], *args, **kwargs))
    return np.stack(results, 1)


def preprocess_emg(x):
    '''
    Closely based on David Gaddy's https://github.com/dgaddy/subvocal

    Args:
        x: shape (seq_len, num_feats)
    '''
    x = apply_to_all(notch_harmonics, x, 60, 1000)
    x = apply_to_all(remove_drift, x, 1000)
    x = apply_to_all(subsample, x, 689.06, 1000)
    return x

def resample_16_22(x):
    x = resampy.resample(x, 16000, 22050)
    x = np.clip(x, -1, 1)
    return x
