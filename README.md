# Restricted Translation Task 2022 in WAT2022

Restricted Translation Task is held in WAT2021-2022, which need to translate based on given restricted target vocabularies. Please see [Website](https://sites.google.com/view/restricted-translation-task/top?authuser=0) in details.
In 2022, we add a new direction Zh–Ja datasets and restricted vocabulary lists. The restricted vocabulary lists are made by automatic term extraction and manual alignments & annotations.

## requirements
- python 3.8
- `pip install -r requirements.txt`
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
