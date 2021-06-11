#!/bin/bash 
#SBATCH --job-name=mastercat
#SBATCH --account=def-sfabbro
#SBATCH --time=25:00:00
#SBATCH --mem=8000M
source $HOME/lensing/bin/activate
python create_mastercat.py