# -*- coding: utf-8 -*-

"""
@file plot2d.py
@brief wrapper functions of matplotlib (2d)
@author ふぇいと (@stfate)

@description

"""

import matplotlib.pyplot as pp
import seaborn
import functools


def clear():
    pp.clf()

def axis(*items):
    pp.axis(*items)

def grid(*items):
    pp.grid(*items)

def hold(*items):
    pp.hold(*items)

def plot(*items, **options):
    xlabel = ''
    ylabel = ''
    title = ''
    plot_options = {}
    for k,v in options.iteritems():
        if k == 'xlabel':
            xlabel = v
        elif k == 'ylabel':
            ylabel = v
        elif k == 'title':
            title = v
        else:
            plot_options[k] = v
    pp.plot(*items, **plot_options)
    pp.xlabel(xlabel)
    pp.ylabel(ylabel)
    pp.title(title)

def imshow(*items, **options):
    xlabel = ''
    ylabel = ''
    title = ''
    imshow_options = {}
    for k,v in options.iteritems():
        if k == 'xlabel':
            xlabel = v
        elif k == 'ylabel':
            ylabel = v
        elif k == 'title':
            title = v
        else:
            imshow_options[k] = v

    pp.imshow(*items, aspect='auto', origin='lower', cmap="jet", **imshow_options)
    pp.xlabel(xlabel)
    pp.ylabel(ylabel)
    pp.title(title)
    pp.colorbar()

def subplot(*items):
    pp.subplot(*items)

def subplots_adjust(**items):
    pp.subplots_adjust(**items)

def show():
    pp.show()
