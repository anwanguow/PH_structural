#!/bin/bash
module purge
module load GCC/11.3.0 OpenMPI/4.1.4

mpirun -np 12 /storage/chem/msrgxt/apps/lammps/src/lmp_mpi < in.LAMMPS
