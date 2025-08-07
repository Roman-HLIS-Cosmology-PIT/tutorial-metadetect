#!/bin/bash
#SBATCH -p HENON
#SBATCH --array=0-143           # 143 jobs since there are 12x12 blocks
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8     
#SBATCH --time=48:00:00
#SBATCH -o ./output/logs/out_%a.txt
#SBATCH -e ./output/logs/out_%a.txt

cd $SLURM_SUBMIT_DIR
python run_batch_metadetect.py $SLURM_ARRAY_TASK_ID