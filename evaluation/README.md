# Evaluation

python=3.8.6

1. `filter_out_sentences.py`で，指定termが含まれていないsystem output文をfilter out
2. `./calc_BLUE.sh`でBLUEを計算 (saclebleuが必要)
3. `re-ranking.py`で，各systemをre-ranking
