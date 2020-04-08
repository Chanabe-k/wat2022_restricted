
# usage
# python ./tokenize_data [input]

#main()
from tqdm import tqdm
import codecs
import logging 
import argparse
import MeCab
import pandas as pd
import tensorflow as tf
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
        header = f_input.readline()

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

## 以下, (https://tksmml.hatenablog.com/entry/2019/12/15/090900)のcopy

def tokenize_map_fn(tokenizer, max_length=100):
    """to be applied to map function for pretrained tokenizer"""
    def _tokenize(text_a, text_b, label):
        # BertJapaneseTokenizerを適用して
        # 「分かち書き」「テキストをidに変換」「token_type_idsを生成」
        inputs = tokenizer.encode_plus(
            text_a.numpy().decode('utf-8'),
            text_b.numpy().decode('utf-8'),
            add_special_tokens=True,
            max_length=max_length,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # attention_maskを作成
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1] * len(input_ids)
        return input_ids, token_type_ids, attention_mask, label
    
    def _map_fn(data):
        """入出力の調整"""
        text_a = data['sentence1']
        text_b = data['sentence2']
        label = data['label']
        out = tf.py_function(_tokenize, inp=[text_a, text_b, label], Tout=(tf.int32, tf.int32, tf.int32, tf.int32))
        return (
            {"input_ids": out[0], "token_type_ids": out[1], "attention_mask": out[2]},
            out[3]
        )
    return _map_fn

def load_dataset(data, tokenizer, max_length=128, train_batch=8, val_batch=32):
    # Prepare dataset for BERT as a tf.data.Dataset instance
    train_dataset = data['train'].map(tokenize_map_fn(tokenizer, max_length=max_length))
    train_dataset = train_dataset.shuffle(100).padded_batch(train_batch, padded_shapes=({'input_ids': max_length, 'token_type_ids': max_length, 'attention_mask': max_length}, []), drop_remainder=True)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # validation dataにも同じ処理
    valid_dataset = data['validation'].map(tokenize_map_fn(tokenizer, max_length=max_length))
    valid_dataset = valid_dataset.padded_batch(val_batch, padded_shapes=({'input_ids': max_length, 'token_type_ids': max_length, 'attention_mask': max_length}, []), drop_remainder=True)
    valid_dataset = valid_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return train_dataset, valid_dataset

def main():
    logging.info("tokenize {} ... ".format(args.input))
    
    with codecs.open(args.input, encoding='utf-8') as f_input:
        df = pd.read_table(args.input)
        df_train = df[:32550]
        df_val = df[32550:42550]
        df_test = df[42550:]
        logging.info("divided into train: {}, dev: {}, test: {}".format(len(df_train), len(df_val), len(df_test)))

        # sentence1がテキスト。分かち書きは不要
        data_no_wakati = {
            "train": tf.data.Dataset.from_tensor_slices({
                'sentence1': df_train['script'].tolist(),          
                'sentence2': ['', ] * len(df_train),
                'label': df_train['time_id'].tolist()
            }),
            "validation": tf.data.Dataset.from_tensor_slices({
                'sentence1': df_train['script'].tolist(),
                'sentence2': ['', ] * len(df_train),
                'label': df_train['time_id'].tolist()
            })
        }
        # tokenizerを定義、vocabファイルやtokenizerの設定が読み込まれる
        tokenizer = BertJapaneseTokenizer.from_pretrained('bert-base-japanese')
        # 実行
        train_dataset, valid_dataset = load_dataset(data_no_wakati, tokenizer, max_length=max_length, train_batch=train_batch, val_batch=val_batch)

#count_label_freq()
main()