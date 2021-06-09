#!/bin/sh 
#SBATCH --job-name=mastercat
#SBATCH --account=def-sfabbro
#SBATCH --time=25:00:00
#SBATCH --mem=8000M
#SBATCH --output=%x-%j.out
python create_mastercat.py