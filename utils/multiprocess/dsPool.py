# -*- coding: utf-8 -*-

from multiprocessing import Process, Pipe

"""
multiprocessing.Pool()と同じ機能を実現するPool()
(クラス内関数も利用できる)
"""

def pipefunc(func, conn,arg):
    conn.send(func(arg))
    conn.close()

class dsPool:
    proc_num = 8

    def __init__(self, proc_num):
        self.proc_num = proc_num

    def map(self, func, args):
        '''
        指定した関数funcにargsの引数を一つ一つ与え実行します。
        これらはあらかじめ指定された数のプロセスで並列実行されます。
        '''
        
        ret = []
        k = 0
        while(k < len(args)):
            plist = []
            clist = []
            end = min(k + self.proc_num, len(args))
            for arg in args[k:end]:
                pconn, cconn = Pipe()
                plist.append(Process(target = pipefunc, args=(func, cconn, arg,)))
                clist.append(pconn)
            for p in plist:
                p.start()
            for conn in clist:
                ret.append(conn.recv())
            for p in plist:
                p.join()
            k += self.proc_num
        return ret

    def map2(self, func_args):
        '''
        (func, arg1, arg2, ...)のタプルで実行する関数と引数の組を指定する
        '''
        return self.map(self.argwrapper, func_args)

    def argwrapper(self, args):
        '''
        ラッパー関数
        '''
        return args[0](*args[1:])
