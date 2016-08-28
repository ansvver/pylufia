# -*- coding: utf-8 -*-

import scipy as sp


def bpm_to_beat_pos(bpm, length, fs=44100):
    n_smp_beat = 60.0/float(bpm) * fs
    beat_pos_in_smp = sp.arange(0, length, n_smp_beat)
    return beat_pos_in_smp
