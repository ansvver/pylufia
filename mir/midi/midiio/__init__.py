# -*- coding: shift_jis -*-

######################### インポートファイル #########################

from ctypes import *



######################### クラス定義エリア #########################

__all__ = ['midiin', 'midiout']


midiio_dll = windll.MIDIIO

class MIDI(Structure):
    _fields_ = [
            ('m_pDeviceHandle', c_void_p),
            ('m_szDeviceName', c_char * 32),
            ('m_lMode', c_long),
            ('m_pBuf', POINTER(c_ubyte)),
            ('m_lBufSize', c_long),
            ('m_lReadPosition', c_long),
            ('m_lWritePosition', c_long),
            ]
