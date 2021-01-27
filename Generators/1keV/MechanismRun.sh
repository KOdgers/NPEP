#!/bin/bash
# MechanismRush.sh: Analysis of one point in the parameter space

module load stashcache

stashcp -r # Transfering files around the OSG

module load python/3.7.0
tar -xzf MechEnv.tar.gz
python -m venv MechEnv
source MechEnv/bin/activate

printf "Run python starting at: "; /bin/date
python3 PreCalcMaster.py $1

printf "Job finished at: "; /bin/date

deactivate
