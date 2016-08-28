# -*- coding: shift_jis -*-

######################### インポートファイル #########################

from ctypes import *
from midiio import MIDI, midiio_dll



######################### クラス定義エリア #########################

__all__ = [
    'MIDIOut', 'GetDeviceNum', 'GetDeviceName', 'Open', 'Reopen',
    'Close', 'Reset', 'PutMIDIMessage', 'PutByte', 'PutBytes']

MIDIOut = MIDI

GetDeviceNum = WINFUNCTYPE(c_long)(
    ('MIDIOut_GetDeviceNum', midiio_dll),
)

GetDeviceName = WINFUNCTYPE(c_long, c_long, c_char_p, c_long)(
    ('MIDIOut_GetDeviceName', midiio_dll),
    (
        (1, 'lIndex'),
        (1, 'pszDeviceName'),
        (1, 'lLen'),
    )
)

Open = WINFUNCTYPE(POINTER(MIDIOut), c_char_p)(
    ('MIDIOut_Open', midiio_dll),
    (
        (1, 'pszDeviceName'),
    )
)

Reopen = WINFUNCTYPE(POINTER(MIDIOut), POINTER(MIDIOut), c_char_p)(
    ('MIDIOut_Reopen', midiio_dll),
    (
        (1, 'pMIDIOut'),
        (1, 'pszDeviceName'),
    )
)

Close = WINFUNCTYPE(c_long, POINTER(MIDIOut))(
    ('MIDIOut_Close', midiio_dll),
    (
        (1, 'pMIDIOut'),
    )
)

Reset = WINFUNCTYPE(c_long, POINTER(MIDIOut))(
    ('MIDIOut_Reset', midiio_dll),
    (
        (1, 'pMIDIOut'),
    )
)

PutMIDIMessage = WINFUNCTYPE(c_long, POINTER(MIDIOut), c_char_p, c_long)(
    ('MIDIOut_PutMIDIMessage', midiio_dll),
    (
        (1, 'pMIDIOut'),
        (1, 'pMessage'),
        (1, 'lLen'),
    )
)

PutByte = WINFUNCTYPE(c_long, POINTER(MIDIOut), c_ubyte)(
    ('MIDIOut_PutByte', midiio_dll),
    (
        (1, 'pMIDIOut'),
        (1, 'ucByte'),
    )
)

PutBytes = WINFUNCTYPE(c_long, POINTER(MIDIOut), c_char_p, c_long)(
    ('MIDIOut_PutBytes', midiio_dll),
    (
        (1, 'pMIDIOut'),
        (1, 'pBuf'),
        (1, 'lLen'),
    )
)
