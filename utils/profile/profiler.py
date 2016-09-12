# -*- coding: utf-8 -*-

"""
====================================================================
Profiling class library
====================================================================
"""

import time


class Profiler():
    """
    Python run-time profiler class
    """
    def __init__(self):
        self.record = {}
        # 時刻計測用関数
        # Winの場合はtime.clock(), それ以外のPlatformの場合はtime.time()を推奨
        self.time_func = time.clock
        #self.timeFunc = time.time
        
        self.t1 = 0.0
        self.t2 = 0.0
        
        self.beginID = ''
        
    def begin(self, ID):
        """
        Begin profiling
        """
        self.begin_id = ID
        self.record[ID] = 0.0
        self.t1 = self.time_func()
        
    def end(self, ID):
        """
        End profiling
        """
        #if ID != self.beginID:
        #    self.record[self.beginID] = -1.0
        #    print 'Error: Begin and End ID mismatch'
        #    return
    
        self.t2 = self.time_func()
        self.record[ID] = self.t2 - self.t1
        self.show_record(ID)

    def get_record(self, ID):
        return self.record[ID]

    def show_record(self, ID):
        print( "PROCTIME({})={:.6f}sec".format(ID,self.record[ID]) )
