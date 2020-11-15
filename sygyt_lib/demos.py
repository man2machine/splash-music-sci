# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 22:26:24 2019

@author: Shahir
"""

import os

from ipywidgets import widgets
from IPython.display import IFrame, Audio
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import librosa

from .wav_utils import wav_read
from .process_audio import decode_song
from .shazam import compute_shazam

import warnings
warnings.filterwarnings("ignore")

AUDIO_SAMPLES_DIR = os.path.abspath(os.path.join(__file__, "../", "audio_samples"))
SONGS_DIR = os.path.abspath(os.path.join(__file__, "../", "audio_samples", "shazam_input_samples"))
FINGERPRINT_DIR = os.path.abspath(os.path.join(__file__, "../", "audio_samples", "shazam_fingerprints"))
SHAZAM_SONGS = os.listdir(SONGS_DIR)

SR = 44100
FANTASIA_PATH = os.path.abspath(os.path.join(AUDIO_SAMPLES_DIR, "fantasia.wav"))
FMIN = 65.41 #32.70 #C0
NUM_KEYS = 72
BINS_PER_NOTE = 4

def get_signal_spectrum(x, sr):
    N = len(x)
    k = np.arange(0, N)
    f = k*sr/N
    X = np.abs(np.fft.fft(x))
    return f, X

def plot_signal_harmonics(x, sr, fund_freq, num_harmonics=30):
    f, X = get_signal_spectrum(x, sr)
    hs = f/fund_freq
    cutoff = np.where(f > (fund_freq*num_harmonics))[0][0]
    
    N = len(x)
    length = int(sr*(3/fund_freq))
    bounds = [N//2-length//2, N//2+length//2]
    t = (np.arange(N)/sr)[bounds[0]:bounds[1]]
    x_short = x[bounds[0]:bounds[1]]
    
    fig = plt.figure(figsize=(8, 6))
    ax1, ax2 = fig.subplots(2, 1)
    ax1.plot(hs[:cutoff], X[:cutoff])
    ax1.set_xlabel("Harmonic")
    ax1.set_ylabel("Magnitude")
    ax1.grid(True)
    ax2.plot(t, x_short)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Magnitue")
    ax2.grid(True)
    fig.show()

def compute_spectrogram(samples, sr):
    Sxx = librosa.stft(samples)
    return librosa.amplitude_to_db(Sxx, ref=np.max)

def display_spectrogram(Zxx, sr):
    librosa.display.specshow(Zxx, sr=sr, y_axis='log', x_axis='time')
    plt.title('Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

def compute_warped_cqt(samples, sr):
    n_bins = NUM_KEYS*BINS_PER_NOTE
    bins_per_octave = 12*BINS_PER_NOTE
    Cxx = librosa.cqt(samples, sr=sr, n_bins=n_bins, bins_per_octave=bins_per_octave, fmin=FMIN)
    return librosa.amplitude_to_db(np.abs(Cxx)**2, ref=np.max)

def display_cqt(Sxx, sr):
    librosa.display.specshow(Sxx, sr=sr, x_axis='time', y_axis='off')
    plt.title('CQT spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()
    
def run_signal_fft_demo():
    @widgets.interact_manual(sig_type=['sine', 'square', 'sawtooth'], num_harmonics_filter=(1, 40))
    def demo(sig_type, num_harmonics_filter):
        fund_freq = librosa.note_to_hz("C5")
        sr = SR
        duration = 3 # seconds
        t = np.linspace(0, duration, int(sr*duration))
        if sig_type == 'square':
            x = sp.signal.square(2*np.pi*t*fund_freq)
        elif sig_type == 'sine':
            x = np.sin(2*np.pi*t*fund_freq)
        else:
            x = sp.signal.sawtooth(2*np.pi*t*fund_freq)
        
        X = np.fft.fft(x)
        
        f, _ = get_signal_spectrum(x, sr)
        cutoff = np.where(f > (fund_freq*num_harmonics_filter))
        X[cutoff] = 0
        x_filtered = np.fft.ifft(X)
    
        plot_signal_harmonics(x_filtered, sr, fund_freq)
        return Audio(x_filtered, rate=sr, autoplay=False)
    return demo
        
def run_instruments_harmonic_demo():
    @widgets.interact_manual(instrument=['oboe', 'sax', 'trumpet', 'violin'])
    def demo(instrument):
        fund_freq = librosa.note_to_hz("C4")
        fname = os.path.join(AUDIO_SAMPLES_DIR, "{}_C4.wav".format(instrument))
        y, sr = librosa.load(fname)
        plot_signal_harmonics(y, sr, fund_freq)
        return Audio(fname)
    return demo

def run_live_spectrogram_demo():
    return IFrame('https://borismus.github.io/spectrogram/', width=700, height=700, frameborder="0", allow="microphone")

def run_play_fantasia():
    return Audio(FANTASIA_PATH)

def run_show_fantasia_warped_cqt():
    y, sr = wav_read(FANTASIA_PATH)
    t_cutoff = 5*sr # first 2 seconds
    y_short = y[:t_cutoff]
    Zxx = compute_warped_cqt(y_short, sr)
    display_cqt(Zxx, sr)

def run_note_detection_demo():
    @widgets.interact_manual(length=['short', 'full'])
    def demo(length):
        y, sr = wav_read(FANTASIA_PATH)
        if length == 'short':
            t_cutoff = 2*sr # first 2 seconds
            y = y[:t_cutoff]
        song = decode_song(y, sr, draw=True)
        y_synth = song.resynthesize_song(sr)
        return Audio(y_synth, rate=sr, autoplay=False)
    return demo

def run_play_shazam_songs():
    @widgets.interact_manual(fname=SHAZAM_SONGS)
    def demo(fname):
        path = os.path.join(SONGS_DIR, fname)
        return Audio(path)
    return demo

def run_shazam_demo():
    @widgets.interact_manual(fname=SHAZAM_SONGS)
    def demo(fname):
        y, sr = wav_read(os.path.join(SONGS_DIR, fname))
        results = compute_shazam(y, sr, FINGERPRINT_DIR)
        print(results)
        best_match = max(results, key=lambda x: results[x][1])
        print("Best match", best_match)
    return demo