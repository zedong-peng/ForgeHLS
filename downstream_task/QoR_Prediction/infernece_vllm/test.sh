#! /bin/bash

base_model_path=""
standtest_dataset_path=""
unseentest_dataset_path=""


standtest_output_dir="./standtest_output"
python test.py \
    --base_model_path $base_model_path \
    --test_dataset_path $standtest_dataset_path \
    --test_output_dir $standtest_output_dir

unseentest_output_dir="./unseentest_output"
python test.py \
    --base_model_path $base_model_path \
    --test_dataset_path $unseentest_dataset_path \
    --test_output_dir $unseentest_output_dir
