#!/bin/bash

## You can perform one of the following tasks for your evalutation
#==========================Table Detection=========================
# task='TD'
# model_name_or_path="pretrained_pth/TabPedia/pretrained.pth"
# jsonl_file_path="samples/json/Table_Detection.json"
# config="./tools/configs/Internlm2_7b_chat_TabPedia.py"
# save_path="./results/"
# eval_samples=8
# all_processes=1
# batch_size=1
# repetition_penalty=1.0
#==========================Table Detection=========================

#==========================Table Structure Recognition=============
task='TSR'
model_name_or_path="pretrained_pth/TabPedia/pretrained.pth"
jsonl_file_path="samples/json/Table_Structure_Recognition.json"
config="./tools/configs/Internlm2_7b_chat_TabPedia.py"
save_path="./results/"
eval_samples=8
all_processes=1
batch_size=1
repetition_penalty=1.0
#==========================Table Structure Recognition=============

#==========================Table Query=============================
# task='TQ'
# model_name_or_path="pretrained_pth/TabPedia/pretrained.pth"
# jsonl_file_path="samples/json/Table_Query.json"
# config="./tools/configs/Internlm2_7b_chat_TabPedia.py"
# save_path="./results/"
# eval_samples=8
# all_processes=1
# batch_size=1
# repetition_penalty=1.0
#==========================Table Query=============================

#==========================Table Question Answering=============================
# task='TQA'
# model_name_or_path="pretrained_pth/TabPedia/pretrained.pth"
# jsonl_file_path="samples/json/Table_QA.json"
# config="tools/configs/Internlm2_7b_chat_TabPedia.py"
# save_path="results/"
# eval_samples=8
# all_processes=1
# batch_size=1
# repetition_penalty=1.1
#==========================Table Question Answering=============================



CUDA_VISIBLE_DEVICES=0 python ./eval/eval_TabPedia.py \
 --model_name_or_path $model_name_or_path \
 --jsonl_file_path $jsonl_file_path \
 --save_path $save_path${task}_gpu_0.json \
 --config $config --eval_samples $eval_samples \
 --process_id 0 \
 --all_processes $all_processes \
 --batch_size $batch_size \
 --repetition_penalty $repetition_penalty
