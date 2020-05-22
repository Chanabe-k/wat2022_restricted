
# usage
# python ./tokenize_data.py [input]

#main()
from tqdm import tqdm
import codecs
import logging 
import argparse
import MeCab
import pandas as pd
import tensorflow as tf
import os
from transformers import BertJapaneseTokenizer

#count_label_freq()
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Meiryo'

# logging
logging.basicConfig(format='[%(levelname)s] %(asctime)s : %(message)s', level=logging.DEBUG, datefmt='%Y/%m/%d %p %I:%M:%S')

# args
parser = argparse.ArgumentParser("extract important data from original data")
parser.add_argument('input', help="input file name")
parser.add_argument('-max_len', help="max length", default=128, type=int)
parser.add_argument('-train_batch', help="train batch size", default=8, type=int)
parser.add_argument('-val_batch', help="validation batch size", default=32, type=int)
#parser.add_argument('output', help="output file name")
args = parser.parse_args()
# input, output
input_path = args.input
#output_path = args.output
max_length = args.max_len
train_batch = args.train_batch
val_batch = args.val_batch


def count_label_freq():
    """visualize time label frequency"""
    # counter for label freq
    c = Counter()

    with open(input_path) as f_input:
        f_input.readline()

        for line in f_input:
            line = line.strip()
            split_line = line.split('\t')
            time = split_line[3]

            # time_labelをcount
            c[time] += 1

        # count
        print("label_freq:\t", c)
        sorted_c = c.most_common()
        print(sorted_c)

        # matplotlibでcを可視化
        left = [t[0] for t in sorted_c]
        height = [t[1] for t in sorted_c]
        print(left, height)

        plt.xticks(rotation = 270)
        plt.bar(left, height, width = 0.4)
        plt.savefig('./fig/time_label_freq.pdf', bbox_inches="tight")

def main():
    input_path = args.input
    tmp_path = "./tmp/{}.tok".format(os.path.basename(input_path))
    logging.info("tokenize {} ... (write {})".format(input_path, tmp_path))

    # tokenizerを定義、vocabファイルやtokenizerの設定が読み込まれる
    tokenizer = BertJapaneseTokenizer.from_pretrained('bert-base-japanese') 

    # tokenize
    # tmp_fileにtokenizeしたデータを書き起こす
    with open(input_path) as f_input, open(tmp_path, 'w') as f_tmp:
        # read header
        line = f_input.readline()
        line = line.strip()
        # write header 
        f_tmp.write(line.strip() + '\t' + 'tokenized_script' + '\n')

        for line in f_input:
            line = line.strip()
            split_line = line.split()
            script = split_line[2]
            tokenized_script = tokenizer.tokenize(script)
            f_tmp.write(line + '\t' + ' '.join(tokenized_script) + '\n')

#count_label_freq()
main()