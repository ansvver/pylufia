# -*- coding: utf-8 -*-

"""
====================================================================
string.py

文字列処理系便利関数
====================================================================
"""

import re

def multi_replace(text, adict):
    """
    一度に複数の文字列を置換する
    """
    rx = re.compile('|'.join(map(re.escape,adict)))
    def one_xlat(match):
        return adict[match.group(0)]
    return rx.sub(one_xlat, text)
