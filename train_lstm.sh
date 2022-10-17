#!/bin/bash
#SBATCH -G 1
#SBATCH -p rtx2080

cmd="python3 train_lstm.py"

$cmd
