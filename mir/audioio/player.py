# -*- coding: utf-8 -*-

"""
====================================================================
Audio playback
====================================================================
"""

import pyaudio
import wave


def playsnd(fname):
    """
    Play wave sound
    
    Parameters:
      fname: string
        input filename(.wav)
    
    Returns:
      None
    """
    chunk = 1024
    wf = wave.open(fname, 'rb')
    py_aud = pyaudio.PyAudio()

    stream = py_aud.open(format = py_aud.get_format_from_width(wf.getsampwidth()),
                         channels = wf.getnchannels(),
                         rate = wf.getframerate(),
                         output = True)

    data = wf.readframes(chunk)

    while data != '':
        stream.write(data)
        data = wf.readframes(chunk)

    stream.close()
    py_aud.terminate()
