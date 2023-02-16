from itertools import groupby
import pprint
import argparse

parser = argparse.ArgumentParser(description='description')
parser.add_argument('-input', type=str, help='Path to input file', default='../../rt2021_result/final_scores_ja-en.txt')
args = parser.parse_args()
print(args.input)

file_names = []
all_scores = []
for is_bleu, g in groupby(open(args.input), lambda line: line.startswith('BLEU')):
    if is_bleu:
        lines = list(g)
        for l in lines:
            _, scores = l.split(' = ', 1)
            BLEU_score = float(scores.split()[0])
            all_scores.append(BLEU_score)
    else:
        lines = list(g)
        for l in lines:
            file_names.append(l.strip())

score_list = [(file, score) for file, score in zip(file_names, all_scores)]
pprint.pprint(sorted(score_list, key=lambda t:t[1], reverse=True))
