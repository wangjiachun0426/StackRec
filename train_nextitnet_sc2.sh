#!/usr/bin/env bash

data_ratio1=1
data_ratio2=1
data_ratio3=1

dilation_count1=4
dilation_count2=8
dilation_count3=16


eval_iter1=3000
step1=50000

eval_iter2=3000
step2=100000

eval_iter3=3000
step3=200000

early_stop1=10
early_stop2=10
early_stop3=10


learning_rate1=0.001
learning_rate2=0.001
learning_rate3=0.001


method1='from_scratch'
method2='stackA'
method3='stackA'


load_model1=False
load_model2=True
load_model3=True

suffix=a0201
save_dir="Models/ml20_${dilation_count1}_${dilation_count2}_${dilation_count3}_${suffix}"

mkdir -p ${save_dir}

time=$(date "+%Y%m%d%H%M%S")
logfile="${save_dir}/log_${time}.txt"


model_path1=${save_dir}
model_path2="${save_dir}/${dilation_count1}_${learning_rate1}_${data_ratio1}_${step1}.ckpt"
model_path3="${save_dir}/${dilation_count2}_${learning_rate2}_${data_ratio2}_${step2}.ckpt"

gpu=3

eval "CUDA_VISIBLE_DEVICES=${gpu} python -u deep_nextitnet.py --eval_iter ${eval_iter1} --data_ratio ${data_ratio1} --step ${step1} --early_stop ${early_stop1} --learning_rate ${learning_rate1} --dilation_count ${dilation_count1} --method ${method1} --load_model ${load_model1} --model_path ${model_path1} --save_dir ${save_dir} | tee ${logfile}"

eval "CUDA_VISIBLE_DEVICES=${gpu} python -u deep_nextitnet.py --eval_iter ${eval_iter2} --data_ratio ${data_ratio2} --step ${step2} --early_stop ${early_stop2} --learning_rate ${learning_rate2} --dilation_count ${dilation_count2} --method ${method2} --load_model ${load_model2} --model_path ${model_path2} --save_dir ${save_dir} | tee -a ${logfile}"

eval "CUDA_VISIBLE_DEVICES=${gpu} python -u deep_nextitnet.py --eval_iter ${eval_iter3} --data_ratio ${data_ratio3} --step ${step3} --early_stop ${early_stop3} --learning_rate ${learning_rate3} --dilation_count ${dilation_count3} --method ${method3} --load_model ${load_model3} --model_path ${model_path3} --save_dir ${save_dir} | tee -a ${logfile}"
