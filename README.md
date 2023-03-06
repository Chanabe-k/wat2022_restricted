# Restricted Translation Task 2022 in WAT2022

Restricted Translation Task is held in WAT2021-2022, which need to translate based on given restricted target vocabularies. Please see [Website](https://sites.google.com/view/restricted-translation-task/top?authuser=0) in details.
In 2022, we add a new direction Zh–Ja datasets and restricted vocabulary lists. The restricted vocabulary lists are made by automatic term extraction ([termextract](http://gensen.dl.itc.u-tokyo.ac.jp/pytermextract/#:~:text=termextract%E3%81%AF%E3%83%86%E3%82%AD%E3%82%B9%E3%83%88%E3%83%87%E3%83%BC%E3%82%BF%E3%81%8B%E3%82%89,%E6%80%A7%E3%81%8C%E9%AB%98%E3%81%8F%E3%81%AA%E3%82%8A%E3%81%BE%E3%81%99%EF%BC%89%E3%80%82)) and manual alignments & annotations.

- 2022 (here)
- [2021](https://github.com/Chanabe-k/wat2021_restricted)

## requirements
- python 3.8
- (for evaluation) `pip install sacrebleu[ja]`

## terminologies (restricted target vocabulary lists)
- Please download the above website.

## evaluation
- We calculate two distinct metrics in this task. In detail, please see `src/evaluation`.
1. BLEU score
2. A consistency score: the ratio of #sentences satisfying **exact match** of given constraints over the whole test corpus

For the "exact match" evaluation, we will conduct the following process:

- English: simply lowercase hypotheses and constraints, then judge character level sequence matching (including whitespaces) for each constraint
- Japanese: judge character level sequence matching (including whitespaces) for each constraint without preprocessing
