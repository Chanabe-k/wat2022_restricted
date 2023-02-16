# Evaluation

python=3.8.6

1. `filter_out_sentences.py` : filter out system outputs that do not include restricted terms
2. `./calc_BLUE.sh` : calculate BLUE with filtered outputs (need saclebleu[ja])
3. `re-ranking.py` : system re-ranking with final scores (1-2)
