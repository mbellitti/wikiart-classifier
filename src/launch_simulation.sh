#!/bin/bash -l

#$ -j y
#$ -l mem_per_core=2G
#$ -l h_rt=01:0:00
#$ -o /scratch/output.out
#$ -pe omp 8

# Available Memory: 64, 128, 192, 256, 512, or 1024

module load python/3.6.2
# module load python/3.6_intel-2018.1.023

NROWS=30
SEED=42
DATADIR="~/ml/wikiart-classifier/data"

telegram_notify "Running: $JOB_ID"

python
/usr3/graduate/bellitti/ml/wikiart-classifier/src/transfer_learning.py
$SEED $NROWSë

mv /scratch/output.out $DATADIR

telegram_notify "Finished: $JOB_ID"
