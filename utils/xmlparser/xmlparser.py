# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET


def parse(fname):
    """
    XMLファイルから内容をパースしてフィールドと値を格納した辞書を返す
    """
    tree = ET.parse(fname)
    root = tree.getroot()
    
    parsed = []
    for e in root.getiterator():
        cur_dict = e.attrib.copy()
        cur_dict['elem'] = e.tag
        parsed.append(cur_dict)
    
    # for child in root:
        # parsed[child.tag] = child.text
    
    return parsed

def parse_from_string(strdata):
    """
    文字列からパースするバージョン．
    戻り値はparse()と同様．
    """
    root = ET.fromstring(strdata)
    parsed = dict()
    for child in root:
        parsed[child.tag] = child.text
    
    return parsed
    