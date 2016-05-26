# -*- coding: utf-8 -*-

"""
====================================================================
Profiling class library
====================================================================
"""

import time

class CProfiler():
    """
    Python run-time profiler class
    """

    def __init__(self):
        self.record = {}
        # Œv‘ª—pŠÖ”
        # Win‚Ìê‡‚Ítime.clock(), ‚»‚êˆÈŠO‚ÌPlatform‚Ìê‡‚Ítime.time()‚ğ„§
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
