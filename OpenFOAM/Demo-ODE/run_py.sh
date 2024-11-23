#!/bin/bash
#BSUB -J t1
#BSUB -m 'agassi'
#BSUB -o testout
#BSUB -e testerr
#BSUB -n 1
#BSUB -N

python3 ./feedback_ODE.py > log &
