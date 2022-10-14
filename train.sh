#!/bin/bash
#SBATCH -G 1
#SBATCH -w thunlp-215-5

cmd="python3 train.py"
cmd+=" --batch_size 8"
cmd+=" --grad_acc_steps 4"
cmd+=" --lr 0.001"

$cmd