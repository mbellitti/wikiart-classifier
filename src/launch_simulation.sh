#!/bin/bash -l

#$ -j y
#$ -l mem_per_core=2G
#$ -l h_rt=01:00:00
#$ -o /scratch/output.out
#$ -pe omp 4
#$ -l gpu_c=3.5
#$ -l gpus=0.25

# Available Memory: 64, 128, 192, 256, 512, or 1024

module load python/3.6.2
module load tensorflow/r1.10

# Suppress TensorFlow complaints about irrelevant things
export TF_CPP_MIN_LOG_LEVEL=2

# NROWS=30
# SEED=42
DATADIR="~/wikiart-classifier/data"

telegram_notify "Running: $JOB_ID"

python /usr3/graduate/bellitti/wikiart-classifier/src/transfer_learning.py

mv /scratch/output.out $DATADIR

telegram_notify "Finished: $JOB_ID"
