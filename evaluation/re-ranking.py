from itertools import groupby
import pprint
import argparse

parser = argparse.ArgumentParser(description='Re-rank to use the results of system evaluation (`./calc_BLUE.sh [lang] [lang_pair] > [results]`)')
parser.add_argument('--lang_pair', type=str, choices=['en-ja', 'ja-en'], help='Select lang_pair to calculate BLEU')
args = parser.parse_args()

lang_pair = args.lang_pair

print(f'# {lang_pair}')
file_names = []
all_scores = []
for is_bleu, g in groupby(open(f'../../work/all_results_{lang_pair}.txt'), lambda line: line.startswith('BLEU')):
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
