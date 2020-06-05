#!/usr/bin/env bash

EXP_DIR="/home/abe-k/dementia_dialogue/dementia_dialogue/"

DATA_DIR=${EXP_DIR}"data/sample/"
OUTPUT_DIR=${EXP_DIR}"output/sample_pred/"

python ${EXP_DIR}src/predict_test.py \
--data_dir=${DATA_DIR} \
--model_type=bert \
--model_name_or_path=bert-base-japanese-whole-word-masking \
--fine_tuned_model_path=${OUTPUT_DIR}pytorch_model.bin \
--seed 0 \
--num_train_epochs 3 \
--per_gpu_train_batch_size 8 \
--per_gpu_eval_batch_size 8 \
--output_dir=${OUTPUT_DIR}
