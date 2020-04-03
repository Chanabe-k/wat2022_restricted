
# usage
# python ./tokenize_data [input]

#main()
from tqdm import tqdm
import logging 
import argparse
import MeCab

#count_label_freq()
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Meiryo'

# logging
#logging.basicConfig(format='[%(levelname)s] %(asctime)s : %(message)s', level=logging.DEBUG, datefmt='%Y/%m/%d %p %I:%M:%S')

# args
parser = argparse.ArgumentParser("extract important data from original data")
parser.add_argument('input', help="input file name")
#parser.add_argument('output', help="output file name")
args = parser.parse_args()

# input, output
input_path = args.input
#output_path = args.output

# main

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

def main():
    logging.info("tokenize {} ... ".format(args.input))
    # counter for label freq
    c = Counter()
    with open(input_path) as f_input:
        header = f_input.readline()

        for line in f_input:
            line = line.strip()
            split_line = line.split('\t')
            
            # line[2] = script, line[3] = time_label
            script = split_line[2]
            time = split_line[3]

            # scriptをmacabでtokenize
            tokenized_script = script

            # timelabelをid化
            time_labels = ['過去', '過去-最近', '最近（1か月以内）', '現在（状態、性質、考えなど）', '過去-現在（習慣など）', '未来（予定、予測、願望、仮定など）', '現在-未来', '最近-未来', '過去-未来', '最近-現在（習慣など）']
            timelabel2timeid = {time_label : i for i, time_label in enumerate(time_labels)}
            time_id = timelabel2timeid[time]

            #print(timelabel2timeid)
            print(script, tokenized_script)
            print(time, timelabel2timeid[time])

            print(tokenized_script, time)

    # 出力先
    #logging.info("filename: '{}'".format(output_path))

#count_label_freq()
main()