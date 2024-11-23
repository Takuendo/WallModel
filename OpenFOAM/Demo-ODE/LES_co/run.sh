#!/bin/bash
#BSUB -J t1
#BSUB -m 'lendl'
#BSUB -o testout
#BSUB -e testerr
#BSUB -n 8
#BSUB -N

mpirun -np 8 pimpleFoamfeedback -parallel > log &
