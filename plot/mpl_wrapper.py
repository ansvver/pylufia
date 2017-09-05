# -*- coding: utf-8 -*-

"""
mpl_wrapper.py

グラフ描画関連の便利関数群
(matplotlibのラッパー)
"""

# import matplotlib.pyplot as pp
from matplotlib.pyplot import *
from seaborn import *
set(context="notebook", style="ticks")
set_style({"xtick.direction": "in", "ytick.direction": "in"})

# def clear():
#     pp.clf()

# def axis(*items):
#     pp.axis(*items)

# def grid(*items):
#     pp.grid(*items)

# def hold(*items):
#     pp.hold(*items)

# def plot(*items, **options):
#     pp.plot(*items, **options)

# def bar(*items, **options):
#     pp.bar(*items, **plot_options)

# def hist(*items, **options):
#     pp.hist(*items, **options)

def imshow(*items, **options):
    if "aspect" not in options:
        options["aspect"] = "auto"
    if "origin" not in options:
        options["origin"] = "lower"
    matplotlib.pyplot.imshow(*items, **options)

# def axhline(*args, **kwargs):
#     pp.axhline(*args, **kwargs)
    
# def axvline(*args, **kwargs):
#     pp.axvline(*args, **kwargs)

# def subplot(*args, **kwargs):
#     pp.subplot(*args, **kwargs)

# def subplots_adjust(**kwargs):
#     pp.subplots_adjust(**kwargs)

# def savefig(*args, **kwargs):
#     pp.savefig(*args, **kwargs)

# def xlabel(*args, **kwargs):
#     pp.xlabel(*args, **kwargs)

# def ylabel(*args, **kwargs):
#     pp.ylabel(*args, **kwargs)

# def title(*args, **kwargs):
#     pp.title(*args, **kwargs)

# def legend(*args, **kwargs):
#     pp.legend(*args, **kwargs)

# def show():
#     pp.show()
