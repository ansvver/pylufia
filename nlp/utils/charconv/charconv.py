# -*- coding: utf-8 -*-

"""
@package charconv.py
@brief 文字変換系
@author stfate
"""

import re
import mojimoji


tt_ksuji = str.maketrans('一二三四五六七八九〇壱弐参', '1234567890123')
tt_dsuji = str.maketrans('１２３４５６７８９０', '1234567890')
tt_dsuji_inv = str.maketrans('1234567890', '１２３４５６７８９０')

re_suji = re.compile(r'[十拾百千万億兆\d]+')
re_kunit = re.compile(r'[十拾百千]|\d+')
re_manshin = re.compile(r'[万億兆]|[^万億兆]+')

TRANSUNIT = {
    '十': 10,
    '拾': 10,
    '百': 100,
    '千': 1000
}
TRANSMANS = {
    '万': 10000,
    '億': 100000000,
    '兆': 1000000000000
}

KATAKANA_ROMAN_DICT = {
    "ア": "a", "イ": "i", "ウ": "u", "エ": "e", "オ": "o",
    "カ": "ka", "キ": "ki", "ク": "ku", "ケ": "ke", "コ": "ko",
    "ガ": "ga", "ギ": "gi", "グ": "gu", "ゲ": "ge", "ゴ": "ko",
    "サ": "sa", "シ": "si", "シ": "su", "セ": "se", "ソ": "so",
    "ザ": "za", "ジ": "zi", "ズ": "zu", "ゼ": "ze", "ゾ": "zo",
    "タ": "ta", "チ": "ti", "ツ": "tu", "テ": "te", "ト": "to",
    "ダ": "da", "ヂ": "di", "ヅ": "du", "デ": "de", "ド": "do",
    "ナ": "na", "ニ": "ni", "ヌ": "nu", "ネ": "ne", "ノ": "no",
    "ハ": "ha", "ヒ": "hi", "フ": "fu", "ヘ": "he", "ホ": "ho",
    "バ": "ba", "ビ": "bi", "ブ": "bu", "ベ": "be", "ボ": "bo",
    "パ": "pa", "ピ": "pi", "プ": "pu", "ペ": "pe", "ポ": "po",
    "マ": "ma", "ミ": "mi", "ム": "mu", "メ": "me", "モ": "mo",
    "ヤ": "ya", "ユ": "yu", "ヨ": "yo", 
    "ラ": "ra", "リ": "ri", "ル": "ru", "レ": "re", "ロ": "ro",
    "ワ": "wa", "ヲ": "wo", "ン": "N",
    "ァ": "la", "ィ": "li", "ゥ": "lu", "ェ": "le", "ォ": "lo",
    "ャ": "lya", "ュ": "lyu", "ョ": "lyo", "ヮ": "lwa"
}


def trans_alphabet_zen_to_han(input_str):
    """ 全角英字を半角英字に変換
    @param input_str 入力文字列
    @return 変換後の文字列
    """
    pass

def trans_number_zen_to_han(input_str):
    """ 全角数字を半角数字に変換
    @param 入力文字列
    @return 変換後の文字列
    """
    output = input_str.translate(tt_dsuji)
    return output

def trans_number_han_to_zen(input_str):
    output = input_str.translate(tt_dsuji_inv)
    return output

def trans_number_chinese_to_arabic(input_str, sep=False):
    """ 漢数字をアラビア数字に変換
    @param input_str 入力文字列
    @param sep 桁区切り文字(,)があるか否か
    @return 変換後の文字列
    """
    def _transvalue(sj, re_obj=re_kunit, transdic=TRANSUNIT):
        unit = 1
        result = 0
        for piece in reversed(re_obj.findall(sj)):
            if piece in transdic:
                if unit > 1:
                    result += unit
                unit = transdic[piece]
            else:
                val = int(piece) if piece.isdecimal() else _transvalue(piece)
                result += val * unit
                unit = 1

        if unit > 1:
            result += unit

        return result

    transuji = input_str.translate(tt_ksuji)
    for suji in sorted( set( re_suji.findall(transuji) ), key=lambda s: len(s), reverse=True ):
        if not suji.isdecimal():
            arabic = _transvalue(suji, re_manshin, TRANSMANS)
            arabic = '{:,}'.format(arabic) if sep else str(arabic)
            transuji = transuji.replace(suji, arabic)

    transuji = transuji.replace("．", ".")
    return transuji

def trans_number_chinese_to_arabic_str(input_str, sep=False):
    transuji = trans_number_chinese_to_arabic(input_str, sep)
    return mojimoji.han_to_zen( str(transuji) )

def katakana_to_roman(input_str):
    """ カタカナをローマ字に変換
    @param
    @return
    """
    output_str = []
    for s in input_str:
        output_str.append(KATAKANA_ROMAN_DICT[s])
    return output_str
