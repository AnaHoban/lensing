#!/bin/sh
#SBATCH --job-name=12epochs-12tiles
#SBATCH --account=def-sfabbro
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=8000M
#SBATCH --output=%x-%j.out
python job2_128.py
