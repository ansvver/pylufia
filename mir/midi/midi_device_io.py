# -*- coding: shift_jis -*-

######################### インポートファイル #########################

import time, sys
sys.path.append('/midiio')
from ctypes import *
from midiio import midiin, midiout


######################### クラス定義エリア #########################

class MidiDeviceIO(object):
    ## Construnctor
    def __init__(self, mode):
        self.m_devlist = []
        self.m_midiobj = []
        self.m_mode = mode
        
        
    ## Methods
    def get_device_list(self):
        if (self.m_mode == 'input'):
            midi = midiin
            
        elif (self.m_mode == 'output'):
            midi = midiout
            
        else:
            return -1
        
        str_buf = create_string_buffer(32)
        self.m_devlist = []
            
        for i in range(midi.GetDeviceNum()):
            ret = midi.GetDeviceName(i, str_buf, 32)
            self.m_devlist.append(str_buf.value)
                
        return self.m_devlist
        
        
    def open(self, id):
        if (self.m_mode == 'input'):
            self.m_midiobj = midiin.Open(self.m_devlist[id])
            
            return self.m_midiobj
        
        elif (self.m_mode == 'output'):
            self.m_midiobj = midiout.Open(self.m_devlist[id])
            
            return self.m_midiobj
            
            
    def send_midi_message(self, msg):
        midiout.PutMIDIMessage(self.m_midiobj, msg, len(msg))
        
        
    def close(self):
        if (self.m_mode == 'input'):
            midiin.Close(self.m_midiobj)
            
        elif (self.m_mode == 'output'):
            midiout.Close(self.m_midiobj)
