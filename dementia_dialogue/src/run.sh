EXP_DIR="/home/abe-k/dementia_dialogue/dementia_dialogue/"


DATA_DIR=$EXP_DIR"data/"
OUTPUT_DIR=$EXP_DIR"output/"

python ~/src/transformers/examples/text-classification/run_glue.py \
--data_dir=$DATA_DIR \
--model_type=bert \
--model_name_or_path=bert-base-japanese-whole-word-masking \
--task_name=original \
--do_train \
--do_eval \
--output_dir=$OUTPUT_DIR
