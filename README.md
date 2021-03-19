# ICM-PPPM
The ICM-P3M is an efficient approach for simulating charged particle systems confined between two planar dielectric interfaces. 
The system is 2D-periodic in X and Y dimensions where charged particles reside in middle region. 
The simulation domain is [xlo,xhi]x[ylo,yhi]x[-Lz/2,Lz/2], and two dielectric interfaces are located at Z=-L/2 and z=L/2.  
Currently, we have implemented the ICM-P3M into the LAMMPS molecular dynamics package (stable version of 31Mar2017). To use the code, one simply needs to replace the corresponding files in the unmodified LAMMPS code, and then compile LAMMPS as in the normal way.
