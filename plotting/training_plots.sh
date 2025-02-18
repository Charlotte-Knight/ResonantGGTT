#!/usr/bin/env bash

cd /home/hep/mdk16/PhD/ggtt/ResonantGGTT
source setup.sh

python plotting/training_plots.py $1 $2 $3 $SGE_TASK_ID
