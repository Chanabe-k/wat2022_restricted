#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
usage:
  python termex_eng.py english_text.txt

　・引数に入力とする英文テキストファイル(utf8)を指定
　・処理結果ファイル: 　eng_extracted.txt

 　パッケージ中のテストデータを使う例は以下
 　python termex_eng.py ../test_data/eng_sample.txt

"""
import sys
import collections
import argparse

import nltk
import termextract.english_postagger
import termextract.core

def output(data, output_path='eng_extracted.txt'):
    """
    処理結果を"eng_extracted.txt"に出力
    """
    outfile = open(output_path, "w", encoding="utf-8")
    data_collection = collections.Counter(data)
    for cmp_noun, value in data_collection.most_common():
        outfile.write(cmp_noun)
        outfile.write("\t")
        outfile.write(str(value))
        outfile.write("\n")
    outfile.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='description')
    parser.add_argument('-input', type=str, help='Path to input file')
    parser.add_argument('-output', type=str, help='Output name', default='eng_extracted.txt')
    args = parser.parse_args()
    # ARGVS = sys.argv
    input_path = args.input
    output_path = args.output

    infile = open(input_path, "r", encoding="utf-8")
    text = infile.read()
    infile.close()
    tagged_text = nltk.pos_tag(nltk.word_tokenize(text))
    frequency = termextract.english_postagger.cmp_noun_dict(tagged_text)
    #term_list = termextract.english_postagger.cmp_noun_list(tagged_text)
    lr = termextract.core.score_lr(
        frequency,
        ignore_words=termextract.english_postagger.IGNORE_WORDS,
        lr_mode=1, average_rate=1)
    term_imp = termextract.core.term_importance(frequency, lr)
    output(term_imp, output_path)
