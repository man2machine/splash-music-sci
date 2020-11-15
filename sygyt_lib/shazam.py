# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 14:29:56 2019

@author: Shahir
"""

import os
import pickle
import numpy as np
from numpy.fft import fft as nfft
from numpy.fft import ifft as nifft

from .wav_utils import wav_read, wav_write

def fft(x):
    return (nfft(x)/len(x)).tolist()

def ifft(x):
    return (nifft(x)*len(x)).tolist()


def stft(x, window_size, step_size, sample_rate, window_shape='rectangle'):
    # return a Short-Time Fourier Transform of x, using the specified window
    # size, step size, and window shape.
    #
    # window_shape is given as a string ('rectangle', 'hann', or 'triangle')
    #
    # return your result as a list of lists, where each internal list
    # represents the DFT coefficients of one window.  I.e., output[n][k] should
    # represent the kth DFT coefficient from the nth window.
    out = []
    N = len(x)
    left = 0

    # use the following to define your window
    if window_shape == 'hann':
        w = [0.5*(1 - np.cos(2*np.pi*n/window_size)) for n in range(window_size)]
    elif window_shape == 'triangle':
        half = window_size/2
        w = [1 - abs((n-half)/half) for n in range(window_size)]
    elif window_shape == 'rectangle':
        w = [1]*window_size
    else:
        raise ValueError("Expected window_shape to be rectangle, "
                         "triangle, or hann")

    # now apply the window to successive regions of the input signal
    # and add each result to the output
    while left <= N-window_size:
        out.append(fft([xn*wn for xn, wn in zip(x[left:left+window_size], w)]))
        left += step_size

    return np.array(out)

def spectrogram(X):
    return np.abs(X)**2

def peaks_2d(data, threshold, min_x_spacing, min_y_spacing):
    data = np.array(data)
    N, M = data.shape
    out = []
    
    while True:
        max_val = np.max(data)
        if max_val <= threshold:
            return sorted(out)
        max_index = tuple(np.argwhere(data == max_val)[0])
        out.append(max_index)
        # works with Python 3 only
        startx = max(0, max_index[0]-min_x_spacing+1)
        endx = min(N, max_index[0]+min_x_spacing)
        starty = max(0, max_index[1]-min_y_spacing+1)
        endy = min(M, max_index[1]+min_y_spacing)
        data[startx:endx, starty:endy] = 0

def stddev(x):
    return np.std(x)

def get_fingerprint(x, sr, hz_cutoff=4000, window_size=2048, step_size=256):
    max_k = int(((hz_cutoff/sr)*window_size)+0.5)
    X = X = stft(x, window_size, step_size, sr, window_shape='hann')
    #librosa.stft(x, n_fft=window_size, hop_length=step_size, window='hann')
    sg = spectrogram(X[:, :max_k])
    thresh = stddev(sg.flatten())*1.5
    return peaks_2d(sg, thresh, 30, 20)

def best_match(d, q):
    min_offset = d[0][0] - q[-1][0]
    max_offset = d[-1][0] - q[0][0]
    offsets = list(range(min_offset, max_offset + 1))
    matches = []
    for o in offsets:
        q_offset = list(map(tuple, np.array(q) + np.array([o, 0])))
        common = set(d).intersection(q_offset)
        matches.append(len(common))
    max_common = max(matches)
    max_offset = offsets[matches.index(max_common)]
    return max_offset, max_common

def offset_to_time(offset, sr, step_size):
    return (offset*step_size)/sr

def generate_song_shazam_data(songs_dir, out_dir):
    fingerprints = {}
    for fname in os.listdir(songs_dir):
        full_fname = os.path.join(songs_dir, fname)
        song_name = fname.split(".")[0]
        print(song_name)
        y, sr = wav_read(full_fname)
        y_short = y[:min(int(sr*11), len(y))]
        out_fname = os.path.join(out_dir, "{}_sample.wav".format(song_name))
        wav_write(y_short, sr, out_fname)
        q = get_fingerprint(y_short, sr)
        f = open(os.path.join(out_dir, "{}_fingerprint.pckl".format(song_name)), 'wb')
        pickle.dump(q, f)
        f.close()
        print("Done")
        fingerprints[song_name] = q
    return fingerprints

def compute_shazam(y, sr, fingerprint_dir):
    q = get_fingerprint(y, sr)
    results = {}
    for fname in os.listdir(fingerprint_dir):
        if fname.split(".")[-1] == "pckl":
            d = pickle.load(open(os.path.join(fingerprint_dir, fname), 'rb'))
            res = best_match(d, q)
            t = offset_to_time(res[0], sr, 256)
            results[fname.split(".")[0]] = (t, res[1])
    
    return results

if __name__ == '__main__':
    songs_dir = os.path.join(__file__, "../", "audio_samples", "shazam_samples")
    out_dir = os.path.join(__file__, "../", "audio_samples", "shazam_fingerprints")
    fingerprints = generate_song_shazam_data(songs_dir, out_dir)
    
    fname = os.path.join(__file__, "../", "audio_samples", "fugue_record.wav")
    y, sr = wav_read(fname)
    results = compute_shazam(y, sr, out_dir)