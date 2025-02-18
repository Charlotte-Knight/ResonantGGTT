#!/usr/bin/env bash

cd /home/hep/mdk16/PhD/ggtt/ResonantGGTT
source setup.sh

echo $1
echo $2
python processInputs/tauID.py $1 $2
