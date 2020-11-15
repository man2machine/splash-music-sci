# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 22:53:56 2019

@author: Shahir
"""

import os
import sys
import wave
import numpy as np
import tempfile

def wav_read(fname):
    """
    Read a wave file.  This will always convert to mono.

    Arguments:
      * fname: a string containing a file name of a WAV file.

    Returns a tuple with 2 elements:
      * a Python list with floats in the range [-1, 1] representing samples.
        the length of this list will be the number of samples in the given wave
        file.
      * an integer containing the sample rate
    """
    f = wave.open(fname, "r")
    chan, bd, sr, count, _, _ = f.getparams()

    assert bd == 2, "bit depth must be 16"

    data = np.fromstring(f.readframes(count), dtype="<h")
    data = data.reshape((count, chan)).astype(float).mean(1) * (2 ** (-15))
    return data, sr


def wav_write(samples, sr, fname):
    """
    Write a mono wave file.

    Arguments:
      * samples: a Python list of numbers in the range [-1, 1], one for each
                 sample in the output WAV file.  Numbers in the list that are
                 outside this range will be clipped to -1 or 1.
      * sr: an integer representing the sampling rate of the output
            (samples/second).
      * fname: a string containing a file name of the WAV file to be written.
    """
    output_file = wave.open(fname, "w")
    output_file.setparams((1, 2, sr, 0, "NONE", "not compressed"))

    out = np.array(samples)
    out[out > 1.0] = 1.0
    out[out < -1.0] = -1.0
    out = np.array(out * (2 ** 15 - 1), dtype=np.int16)
    output_file.writeframes(out.tobytes())
    output_file.close()