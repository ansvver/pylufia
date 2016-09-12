# -*- coding: utf-8 -*-


break_index = 0


def print_var(x, name=None):
    if name:
        print( "{}={}".format(name,x) )
    else:
        print x

def print_array(x, name=None):
    if name:
        print( "{}:".format(name) )
    print x

def print_break():
    print( "b{}".format(break_index) )
    break_index += 1
