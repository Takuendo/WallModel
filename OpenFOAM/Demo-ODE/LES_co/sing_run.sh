#!/bin/bash
#BSUB -J t1
#BSUB -m 'lendl'
#BSUB -o testout
#BSUB -e testerr
#BSUB -n 1
#BSUB -N

pimpleFoam > log &
