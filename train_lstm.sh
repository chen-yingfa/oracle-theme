#!/bin/bash
#SBATCH -G 1
#SBATCH -w thunlp-215-5

cmd="python3 train_lstm.py"

$cmd