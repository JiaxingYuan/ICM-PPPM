# ICM-PPPM
Description: 

The ICM-P3M is an efficient approach for simulating charged particle systems confined between two planar dielectric interfaces. 
The system is 2D-periodic in X and Y dimensions where charged particles reside in middle region. 
The simulation domain is [xlo,xhi]x[ylo,yhi]x[-Lz/2,Lz/2], and two dielectric interfaces are located at Z=-L/2 and z=L/2.  

Currently, we have implemented the ICM-P3M into the LAMMPS molecular dynamics package (stable versions of 3Mar2020, 7Aug2019, 12Dec2018, and 31Mar2017; 31Mar2017 is the extensively tested version). To use the code, one simply needs to replace the corresponding files in the unmodified LAMMPS code, and then compile LAMMPS as in the normal way.

# Publication:

Jiaxing Yuan, Hanne Antila, and Erik Luijten. Particle-particle particle-mesh algorithm for electrolytes between charged dielectric interfaces. J. Chem. Phys. 154, 094115 (2021).


# Usage in LAMMPS:

kspace_modify slab #0 mismatch #1 #2 interface_height #3 order_image_charge #4 #5 interface_charge #6 #7

#0 is the slab factor used in Ewald summation

#1=(\epsilon_2-\epsilon_1)/(\epsilon_2+\epsilon_1) 

#2=(\epsilon_2-\epsilon_3)/(\epsilon_2+\epsilon_3).

#3 is the distance L between two dielectric interfaces

#4 and #5 are the order (integer) of image charges for the truncation of short range part and long range part

#6 and #7 are the constant surface charge densities on the top and bottom interfaces (zero by default)

# Reproducibility：

In the folder /examples, there are two example input files with the corresponding output files for benchmark test. Also, one example input file for practical salt (1:1 salt) MD simulation is presented. 
