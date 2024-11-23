#!/bin/sh

# Source tutorial run functions
. $WM_PROJECT_DIR/bin/tools/RunFunctions

blockMesh

perturbUChannel

postProcess -func writeCellCentres

cp ./0/C ./constant/polyMesh/
