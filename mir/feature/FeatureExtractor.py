# -*- coding: utf-8 -*-

"""
============================================================
@file   FeatureExtractor.py
@date   2012/05/31
@author sasai

@brief
特徴量計算のフロントエンド

============================================================
"""

import scipy as sp
import audioio
import feature.timbre as timbre
import feature.rhythm.beat as beat
import feature.spectral as spectral
import segment
import common


class FeatureExtractor():
    def __init__(self):
        pass

    def spectrogram(self, fname, framesize=1024, hopsize=512, window='hamming'):
        x,fs,bit = audioio.wavread(fname)
        x = audioio.stereo_to_mono(x)
        X,F,T = spectral.stft(x, framesize, hopsize, fs, window)
        X = sp.absolute(X)
        return X,F,T

    def segmental_cqt(self, fname, beat_pos_arr, beatunit=16, framesize=1024, hopsize=512, n_cq=88, n_per_semitone=1, fmin=60.0):
        x,fs,bit = audioio.wavread(fname)
        x = audioio.stereo_to_mono(x)
        X,F,T = spectral.cqt(x, framesize, hopsize, fs, n_cq*n_per_semitone, n_per_semitone, fmin)
        Xseg = segment.normalize_time_axis_by_beat(X.T, beat_pos_arr, beatunit, framesize, hopsize, fs).T
        return Xseg

    def segmental_cqt_bpm(self, fname, bpm, beatunit=16, framesize=1024, hopsize=512, n_cq=88, n_per_semitone=1, fmin=60.0):
        x,fs,bit = audioio.wavread(fname)
        x = audioio.stereo_to_mono(x)
        X,F,T = spectral.cqt(x, framesize, hopsize, fs, n_cq*n_per_semitone, n_per_semitone, fmin)
        Xseg = segment.normalize_time_axis_by_bpm(X.T, bpm, beatunit, framesize, hopsize, fs).T
        return Xseg

    def segmental_mfcc(self, fname, beat_pos_arr, beatunit=16, framesize=1024, hopsize=512, max_freq=22050, n_ceps=22, nopower=False):
        x,fs,bit = audioio.wavread(fname)
        x = audioio.stereo_to_mono(x)
        mfccdata = timbre.mfcc(x, framesize, hopsize, fs, max_freq, n_ceps)
        if nopower:
            mfccdata = mfccdata[:,1:]
        mfccdata = segment.normalize_time_axis_by_beat(mfccdata, beat_pos_arr, beatunit, framesize, hopsize, fs)
        trans_mfcc = segment.transform_time_axis_to_one_bar(mfccdata, beatunit)
        return trans_mfcc

    def segmental_mfcc_bpm(self, fname, bpm, beatunit=16, framesize=1024, hopsize=512, max_freq=22050, n_ceps=22, nopower=False):
        x,fs,bit = audioio.wavread(fname)
        x = audioio.stereo_to_mono(x)
        mfccdata = timbre.mfcc(x, framesize, hopsize, fs, max_freq, n_ceps)
        if nopower:
            mfccdata = mfccdata[:,1:]
        mfccdata = segment.normalize_time_axis_by_bpm(mfccdata, bpm, beatunit, framesize, hopsize, fs)
        trans_mfcc = segment.transform_time_axis_to_one_bar(mfccdata, beatunit)
        return trans_mfcc

    def segmental_mel_spectrogram_bpm(self, fname, bpm, beatunit=16, framesize=1024, hopsize=512, n_ceps=22):
        x,fs,bit = audioio.wavread(fname)
        x = audioio.stereo_to_mono(x)
        melspe = timbre.mel_spectrogram(x, framesize, hopsize, fs, 'barthann', 7000, n_ceps)
        melspe = segment.normalize_time_axis_by_bpm(melspe, bpm, beatunit, framesize, hopsize, fs)
        trans_melspe = segment.transform_time_axis_to_one_bar(melspe, beatunit)
        return trans_melspe

    def segmental_beat_spectrogram(self, fname, bpm, segsize=32, framesize=1024, hopsize=512, n_lags=256):
        x,fs,bit = audioio.wavread(fname)
        x = audioio.stereo_to_mono(x)
        
        # BS = rhythm.beat_spectrogram_kurth_4beat_force20sec_cy(x, framesize, hopsize, fs, 'hamming', bpm, n_lags)
        # BS = signal.smoothByBeat(BS, framesize, hopsize, fs, bpm, segsize)
        
        BS = sp.zeros( (segsize*4, n_lags) )
        BS[:] = rhythm.beat_spectrum_kurth_4beat_force20sec_cy(x, framesize, hopsize, fs, 'barthann', bpm, n_lags)

        # B = rhythm.beat_spectrum_kurth_4beat_force20sec_cy(x, framesize, hopsize, fs, 'hamming', bpm, n_lags)
        # BS = sp.zeros( (segsize*4, 256) )
        # for f in xrange(segsize*4):
        #     BS[f,:] = B
            
        return BS

    def beat_spectrum(self, x, fs, framesize=1024, hopsize=512):
        BS = beat.beat_spectrum_force20sec_cy(x, framesize, hopsize, fs, "barthann")
        # BS = BS.mean(0)
        return BS

    def beat_spectrum_fromFile(self, fname, framesize=1024, hopsize=512):
        x,fs,bit = audioio.wavread(fname)
        x = audioio.stereo_to_mono(x)
        return self.beat_spectrum(x, fs, framesize, hopsize)

    def beat_spectrum_4beat(self, x, fs, bpm, framesize=1024, hopsize=512, n_lags=256):
        # BS = rhythm.beat_spectrogram_4beat_force20sec_cy(x, framesize, hopsize, fs, 'barthann', bpm, n_lags)
        # BS = BS.mean(0)
        BS = beat.beat_spectrum_4beat_force20sec_cy(x, framesize, hopsize, fs, "barthann", bpm, n_lags)
        return BS

    def beat_spectrum_unit(self, x, fs, framesize=1024, hopsize=512, n_dim=128):
        BS = self.beat_spectrum(x, fs, framesize, hopsize)
        p = beat.find_beat_spectrum_period(BS)
        unitBS = BS[:p]
        unitBS = beat.decimate_beat_spectrum(unitBS, n_dim)
        return unitBS

    def beat_spectrum_unit_fromFile(self, fname, framesize=1024, hopsize=512, n_dim=128):
        x,fs,bit = audioio.wavread(fname)
        x = audioio.stereo_to_mono(x)
        return self.beat_spectrum_unit(x, fs, framesize, hopsize, n_dim)

    def beat_spectrum_4beat_fromFile(self, fname, bpm, framesize=1024, hopsize=512, n_lags=256):
        x,fs,bit = audioio.wavread(fname)
        x = audioio.stereo_to_mono(x)
        return self.beat_spectrum_4beat(x, fs, bpm, framesize, hopsize, n_lags)

    def segmental_spectral_centroid(self, fname, bpm, framesize=2048, hopsize=512, segsize=32):
        x,fs,bit = audioio.wavread(fname)
        x = audioio.stereo_to_mono(x)
        SC = spectral.spectral_centroid(x, framesize, hopsize, fs)
        SC = segment.normalize_time_axis_by_bpm(SC, framesize, hopsize, fs, bpm, segsize)

        return SC

    def segmental_spectral_kurtosis(self, fname, bpm, framesize=2048, hopsize=512, segsize=32):
        x,fs,bit = audioio.wavread(fname)
        x = audioio.stereo_to_mono(x)
        SK = spectral.spectral_kurtosis(x, framesize, hopsize, fs)
        SK = segment.normalize_time_axis_by_bpm(SK, framesize, hopsize, fs, bpm, segsize)

        return SK
