#!/bin/sh

# Source tutorial run functions
. $WM_PROJECT_DIR/bin/tools/RunFunctions

foamCleanPolyMesh

rm -rf *0

rm -rf log*
rm -rf processor*
rm -r 0* 1* 2* 3* 4* 5* 6* 7* 8* 9*

cp -r org.0.org 0

rm -r postProcessing/*
