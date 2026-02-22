#!/bin/bash
#SBATCH --job-name=selftts
#SBATCH --output=./slurm/selfaug_%j.out
#SBATCH --error=./slurm/selfaug_%j.err
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=64G          
#SBATCH --partition=p5000 
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL


source ~/miniconda3/bin/activate
conda activate selftts

# Remove the existing link/directory
rm -rf DUMMY3

# Create the symbolic link
ln -s /hadatasets/lucas.ueda/esd/files/ DUMMY3

python train_ms_emotion_selfaug.py -c configs/selftts_selfaugmentation.json -m selftts_selfaugmentation