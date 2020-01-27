#!/bin/bash

# Fail fast on errors.
set -e

# Remove previous instances of run.
if [ -d output ];
then
    rm -rf output
fi
mkdir output

# Copy all data to the output directory to sandbox runs.
cp input/topol.top output
cp -r input/mdp/ output
cd output

# Create a system of water molecules.
gmx solvate -cs tip4p -o conf.gro -box 2.3 2.3 2.3 -p topol.top

# Minimize the bonds of the solvent.
gmx grompp -f mdp/min.mdp -o min -pp min -po min
gmx mdrun -deffnm min

# Minimize the system of water.
gmx grompp -f mdp/min2.mdp -o min2 -pp min2 -po min2 -c min -t min
gmx mdrun -deffnm min2

# Equilibrate temperature.
gmx grompp -f mdp/eql.mdp -o eql -pp eql -po eql -c min2 -t min2
gmx mdrun -deffnm eql

# Equilibrate pressure.
gmx grompp -f mdp/eql2.mdp -o eql2 -pp eql2 -po eql2 -c eql -t eql
gmx mdrun -deffnm eql2

# Production run
gmx grompp -f mdp/prd.mdp -o prd -pp prd -po prd -c eql2 -t eql2
gmx mdrun -deffnm prd
