#!/usr/bin/env bash

EXP_DIR="/home/abe-k/dementia_dialogue/dementia_dialogue/"

#DATA_DIR=$EXP_DIR"data/"
DATA_DIR=$EXP_DIR"data/sample"
#OUTPUT_DIR=$EXP_DIR"output/"
OUTPUT_DIR=$EXP_DIR"output/sample_pred"

python ~/src/transformers/examples/text-classification/run_glue.py \
--data_dir=$DATA_DIR \
--model_type=bert \
--model_name_or_path=bert-base-japanese-whole-word-masking \
--task_name=original \
--do_train \
--do_eval \
--do_pred \
--seed 0 \
--num_train_epochs 3 \
--per_gpu_train_batch_size 8 \
--per_gpu_eval_batch_size 8 \
--output_dir=$OUTPUT_DIR
