# -*- coding: shift_jis -*-

######################### インポートファイル #########################

from ctypes import *
from midiio import MIDI, midiio_dll



######################### クラス定義エリア #########################

__all__ = [
        'MIDIIn', 'GetDeviceNum', 'GetDeviceName', 'Open', 'Reopen',
        'Close', 'Reset', 'GetMIDIMessage', 'GetByte', 'GetBytes']

MIDIIn = MIDI

GetDeviceNum = WINFUNCTYPE(c_long)(
        ('MIDIIn_GetDeviceNum', midiio_dll),
        )

GetDeviceName = WINFUNCTYPE(c_long, c_long, c_char_p, c_long)(
        ('MIDIIn_GetDeviceName', midiio_dll),
        (
            (1, 'lIndex'),
            (1, 'pszDeviceName'),
            (1, 'lLen'),
            )
        )

Open = WINFUNCTYPE(POINTER(MIDIIn), c_char_p)(
        ('MIDIIn_Open', midiio_dll),
        (
            (1, 'pszDeviceName'),
            )
        )

Reopen = WINFUNCTYPE(POINTER(MIDIIn), POINTER(MIDIIn), c_char_p)(
        ('MIDIIn_Reopen', midiio_dll),
        (
            (1, 'pMIDIIn'),
            (1, 'pszDeviceName'),
            )
        )

Close = WINFUNCTYPE(c_long, POINTER(MIDIIn))(
        ('MIDIIn_Close', midiio_dll),
        (
            (1, 'pMIDIIn'),
            )
        )

Reset = WINFUNCTYPE(c_long, POINTER(MIDIIn))(
        ('MIDIIn_Reset', midiio_dll),
        (
            (1, 'pMIDIIn'),
            )
        )

GetMIDIMessage = WINFUNCTYPE(c_long, POINTER(MIDIIn), c_char_p, c_long)(
        ('MIDIIn_GetMIDIMessage', midiio_dll),
        (
            (1, 'pMIDIIn'),
            (1, 'pMessage'),
            (1, 'lLen'),
            )
        )

GetByte = WINFUNCTYPE(c_long, POINTER(MIDIIn), c_ubyte)(
        ('MIDIIn_GetByte', midiio_dll),
        (
            (1, 'pMIDIIn'),
            (1, 'ucByte'),
            )
        )

GetBytes = WINFUNCTYPE(c_long, POINTER(MIDIIn), c_char_p, c_long)(
        ('MIDIIn_GetBytes', midiio_dll),
        (
            (1, 'pMIDIIn'),
            (1, 'pBuf'),
            (1, 'lLen'),
            )
        )
