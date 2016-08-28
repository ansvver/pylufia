# -*- coding: utf-8 -*-

"""
plottools.py

グラフ描画関連の便利関数群
(matplotlibのラッパー)
"""

# import matplotlib.pyplot as pp
from matplotlib.pyplot import *
import seaborn
seaborn.set_style("dark")


def imshow(*items, **options):
    matplotlib.pyplot.imshow(*items, aspect="auto", origin="lower", cmap="jet", **options)
    colorbar()
