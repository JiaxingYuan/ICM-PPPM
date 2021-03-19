/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories Steve
   Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Roy Pollock (LLNL), Paul Crozier (SNL)
     per-atom energy/virial added by German Samolyuk (ORNL), Stan Moore (BYU)
     group/group energy/force added by Stan Moore (BYU)
     triclinic added by Stan Moore (SNL)
------------------------------------------------------------------------- */

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "ewald.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "pair.h"
#include "domain.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace MathConst;

#define SMALL 0.00001

/* ---------------------------------------------------------------------- */

Ewald::Ewald(LAMMPS *lmp, int narg, char **arg) : KSpace(lmp, narg, arg),
  kxvecs(NULL), kyvecs(NULL), kzvecs(NULL), ug(NULL), eg(NULL), vg(NULL),
  ek(NULL), sfacrl(NULL), sfacim(NULL), sfacrl_all(NULL), sfacim_all(NULL),
  cs(NULL), sn(NULL), sfacrl_A(NULL), sfacim_A(NULL), sfacrl_A_all(NULL),
  sfacim_A_all(NULL), sfacrl_B(NULL), sfacim_B(NULL), sfacrl_B_all(NULL),
  sfacim_B_all(NULL),image_sfacrl(NULL),image_sfacim(NULL),image_sfacrl_all(NULL),
  image_sfacim_all(NULL),A_rl(NULL),A_im(NULL),B_rl(NULL),B_im(NULL),pre_up(NULL),pre_down(NULL)
{
  group_allocate_flag = 0;
  kmax_created = 0;
  if (narg != 1) error->all(FLERR,"Illegal kspace_style ewald command");

  ewaldflag = 1;
  group_group_enable = 1;
  
  accuracy_relative = fabs(force->numeric(FLERR,arg[0]));

  kmax = 0;
  kxvecs = kyvecs = kzvecs = NULL;
  ug = NULL;
  pre_up =NULL;
  pre_down = NULL;
  eg = vg = NULL;
  A_rl=A_im=B_rl=B_im=NULL;
  sfacrl = sfacim = sfacrl_all = sfacim_all = image_sfacrl = image_sfacim = image_sfacrl_all = image_sfacim_all = NULL;

  nmax = 0;
  ek = NULL;
  cs = sn = NULL;

  kcount = 0;
}

/* ----------------------------------------------------------------------
   free all memory
------------------------------------------------------------------------- */

Ewald::~Ewald()
{
  deallocate();
  if (group_allocate_flag) deallocate_groups();
  memory->destroy(ek);
  memory->destroy3d_offset(cs,-kmax_created);
  memory->destroy3d_offset(sn,-kmax_created);
}

/* ---------------------------------------------------------------------- */

void Ewald::init()
{
  if (comm->me == 0) {
    if (screen) fprintf(screen,"Ewald initialization ...\n");
    if (logfile) fprintf(logfile,"Ewald initialization ...\n");
  }

  // error check

  triclinic_check();
  if (domain->dimension == 2)
    error->all(FLERR,"Cannot use Ewald with 2d simulation");

  if (!atom->q_flag) error->all(FLERR,"Kspace style requires atom attribute q");

  if (slabflag == 0 && domain->nonperiodic > 0)
    error->all(FLERR,"Cannot use nonperiodic boundaries with Ewald");
  if (slabflag) {
    if (domain->xperiodic != 1 || domain->yperiodic != 1 ||
        domain->boundary[2][0] != 1 || domain->boundary[2][1] != 1)
      error->all(FLERR,"Incorrect boundaries with slab Ewald");
    if (domain->triclinic)
      error->all(FLERR,"Cannot (yet) use Ewald with triclinic box "
                 "and slab correction");
  }

  // extract short-range Coulombic cutoff from pair style

  triclinic = domain->triclinic;
  pair_check();

  int itmp;
  double *p_cutoff = (double *) force->pair->extract("cut_coul",itmp);
  if (p_cutoff == NULL)
    error->all(FLERR,"KSpace style is incompatible with Pair style");
  double cutoff = *p_cutoff;

  // compute qsum & qsqsum and warn if not charge-neutral

  scale = 1.0;
  qqrd2e = force->qqrd2e;
  qsum_qsq();
  natoms_original = atom->natoms;

  // set accuracy (force units) from accuracy_relative or accuracy_absolute

  if (accuracy_absolute >= 0.0) accuracy = accuracy_absolute;
  else accuracy = accuracy_relative * two_charge_force;

  // setup K-space resolution

  bigint natoms = atom->natoms;

  // use xprd,yprd,zprd even if triclinic so grid size is the same
  // adjust z dimension for 2d slab Ewald
  // 3d Ewald just uses zprd since slab_volfactor = 1.0

  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;
  double zprd_slab = zprd*slab_volfactor;
  imageflag=force->kspace->imageflag;
  dielectric_interface_L=force->kspace->dielectric_interface_L;
  order_image_short=force->kspace->order_image_short;
  order_image_long=force->kspace->order_image_long;
  dielectric_jump1=force->kspace->dielectric_jump1;//dielectric interface
  dielectric_jump2=force->kspace->dielectric_jump2;//dielectric interface
  surface_chargeflag=force->kspace->surface_chargeflag;
  charge_up=force->kspace->charge_up;
  charge_down=force->kspace->charge_down;
  if((imageflag==1)&&(zprd<(2*order_image_short+1)*dielectric_interface_L)){
	  error->all(FLERR,"System size in z direction does not allow specified real space images");
  }
  if((surface_chargeflag==1)&&(fabs(qsum+(charge_up+charge_down)*xprd*yprd)>SMALL)){
    if (screen) fprintf(screen,"Total valence of ions and surface charge is not neutral\n");
    if (logfile) fprintf(logfile,"Total valence of ions and surface charge is not neutral\n");
  }
  // make initial g_ewald estimate
  // based on desired accuracy and real space cutoff
  // fluid-occupied volume used to estimate real-space error
  // zprd used rather than zprd_slab

  if (!gewaldflag) {
    if (accuracy <= 0.0)
      error->all(FLERR,"KSpace accuracy must be > 0");
    g_ewald = accuracy*sqrt(natoms*cutoff*xprd*yprd*zprd) / (2.0*q2);
    if (g_ewald >= 1.0) g_ewald = (1.35 - 0.15*log(accuracy))/cutoff;
    else g_ewald = sqrt(-log(g_ewald)) / cutoff;
  }

  // setup Ewald coefficients so can print stats

  setup();

  // final RMS accuracy

  double lprx = rms(kxmax_orig,xprd,natoms,q2);
  double lpry = rms(kymax_orig,yprd,natoms,q2);
  double lprz = rms(kzmax_orig,zprd_slab,natoms,q2);
  double lpr = sqrt(lprx*lprx + lpry*lpry + lprz*lprz) / sqrt(3.0);
  double q2_over_sqrt = q2 / sqrt(natoms*cutoff*xprd*yprd*zprd_slab);
  double spr = 2.0 *q2_over_sqrt * exp(-g_ewald*g_ewald*cutoff*cutoff);
  double tpr = estimate_table_accuracy(q2_over_sqrt,spr);
  double estimated_accuracy = sqrt(lpr*lpr + spr*spr + tpr*tpr);

  // stats

  if (comm->me == 0) {
    if (screen) {
      fprintf(screen,"  G vector (1/distance) = %g\n",g_ewald);
      fprintf(screen,"  estimated absolute RMS force accuracy = %g\n",
              estimated_accuracy);
      fprintf(screen,"  estimated relative force accuracy = %g\n",
              estimated_accuracy/two_charge_force);
      fprintf(screen,"  KSpace vectors: actual max1d max3d = %d %d %d\n",
              kcount,kmax,kmax3d);
      fprintf(screen,"                  kxmax kymax kzmax  = %d %d %d\n",
              kxmax,kymax,kzmax);
    }
    if (logfile) {
      fprintf(logfile,"  G vector (1/distance) = %g\n",g_ewald);
      fprintf(logfile,"  estimated absolute RMS force accuracy = %g\n",
              estimated_accuracy);
      fprintf(logfile,"  estimated relative force accuracy = %g\n",
              estimated_accuracy/two_charge_force);
      fprintf(logfile,"  KSpace vectors: actual max1d max3d = %d %d %d\n",
              kcount,kmax,kmax3d);
      fprintf(logfile,"                  kxmax kymax kzmax  = %d %d %d\n",
              kxmax,kymax,kzmax);

    }
  }
  
}

/* ----------------------------------------------------------------------
   adjust Ewald coeffs, called initially and whenever volume has changed
------------------------------------------------------------------------- */

void Ewald::setup()
{
  // volume-dependent factors

  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;

  // adjustment of z dimension for 2d slab Ewald
  // 3d Ewald just uses zprd since slab_volfactor = 1.0

  double zprd_slab = zprd*slab_volfactor;
  volume = xprd * yprd * zprd_slab;

  unitk[0] = 2.0*MY_PI/xprd;
  unitk[1] = 2.0*MY_PI/yprd;
  unitk[2] = 2.0*MY_PI/zprd_slab;

  int kmax_old = kmax;

  if (kewaldflag == 0) {

    // determine kmax
    // function of current box size, accuracy, G_ewald (short-range cutoff)

    bigint natoms = atom->natoms;
    double err;
    kxmax = 1;
    kymax = 1;
    kzmax = 1;

    err = rms(kxmax,xprd,natoms,q2);
    while (err > accuracy) {
      kxmax++;
      err = rms(kxmax,xprd,natoms,q2);
    }

    err = rms(kymax,yprd,natoms,q2);
    while (err > accuracy) {
      kymax++;
      err = rms(kymax,yprd,natoms,q2);
    }

    err = rms(kzmax,zprd_slab,natoms,q2);
    while (err > accuracy) {
      kzmax++;
      err = rms(kzmax,zprd_slab,natoms,q2);
    }

    kmax = MAX(kxmax,kymax);
    kmax = MAX(kmax,kzmax);
    kmax3d = 4*kmax*kmax*kmax + 6*kmax*kmax + 3*kmax;

    double gsqxmx = unitk[0]*unitk[0]*kxmax*kxmax;
    double gsqymx = unitk[1]*unitk[1]*kymax*kymax;
    double gsqzmx = unitk[2]*unitk[2]*kzmax*kzmax;
    gsqmx = MAX(gsqxmx,gsqymx);
    gsqmx = MAX(gsqmx,gsqzmx);

    kxmax_orig = kxmax;
    kymax_orig = kymax;
    kzmax_orig = kzmax;

    // scale lattice vectors for triclinic skew

    if (triclinic) {
      double tmp[3];
      tmp[0] = kxmax/xprd;
      tmp[1] = kymax/yprd;
      tmp[2] = kzmax/zprd;
      lamda2xT(&tmp[0],&tmp[0]);
      kxmax = MAX(1,static_cast<int>(tmp[0]));
      kymax = MAX(1,static_cast<int>(tmp[1]));
      kzmax = MAX(1,static_cast<int>(tmp[2]));

      kmax = MAX(kxmax,kymax);
      kmax = MAX(kmax,kzmax);
      kmax3d = 4*kmax*kmax*kmax + 6*kmax*kmax + 3*kmax;
    }

  } else {

    kxmax = kx_ewald;
    kymax = ky_ewald;
    kzmax = kz_ewald;

    kxmax_orig = kxmax;
    kymax_orig = kymax;
    kzmax_orig = kzmax;

    kmax = MAX(kxmax,kymax);
    kmax = MAX(kmax,kzmax);
    kmax3d = 4*kmax*kmax*kmax + 6*kmax*kmax + 3*kmax;

    double gsqxmx = unitk[0]*unitk[0]*kxmax*kxmax;
    double gsqymx = unitk[1]*unitk[1]*kymax*kymax;
    double gsqzmx = unitk[2]*unitk[2]*kzmax*kzmax;
    gsqmx = MAX(gsqxmx,gsqymx);
    gsqmx = MAX(gsqmx,gsqzmx);
  }

  gsqmx *= 1.00001;

  // if size has grown, reallocate k-dependent and nlocal-dependent arrays

  if (kmax > kmax_old) {
    deallocate();
    allocate();
    group_allocate_flag = 0;

    memory->destroy(ek);
    memory->destroy3d_offset(cs,-kmax_created);
    memory->destroy3d_offset(sn,-kmax_created);
    nmax = atom->nmax;
    memory->create(ek,nmax,3,"ewald:ek");
    memory->create3d_offset(cs,-kmax,kmax,3,nmax,"ewald:cs");
    memory->create3d_offset(sn,-kmax,kmax,3,nmax,"ewald:sn");
    kmax_created = kmax;
  }

  // pre-compute Ewald coefficients

  if (triclinic == 0)
    coeffs();
  else{
	 coeffs_triclinic();
  }
  int temp_count=0;
  int i=0;
  int n=0;
  int count_k=0;
  double tempA_rl=0;
  double tempA_im=0;
  double tempB_rl=0;
  double tempB_im=0;
  int ic,pre_temp;
  int m,k,l;
  double sqk,clpm,slpm;
  imageflag=force->kspace->imageflag;
  dielectric_jump1=force->kspace->dielectric_jump1;//dielectric interface
  dielectric_jump2=force->kspace->dielectric_jump2;//dielectric interface
  dielectric_interface_L=force->kspace->dielectric_interface_L;
  order_image_short=force->kspace->order_image_short;
  order_image_long=force->kspace->order_image_long;
  n=0;
  if (imageflag==1) {
     if (order_image_short>order_image_long)
	 {
		 pre_temp = order_image_short;
	 }
     else {
		 pre_temp = order_image_long;		 
	 }		
     pre_up[0] = dielectric_jump1;
     pre_down[0] = dielectric_jump2;	 
     for(i=0;i<(pre_temp+2);i++){
	   if(i%2==0){
		  pre_up[i+1] = pre_up[i]*dielectric_jump2; 
		  pre_down[i+1] = pre_down[i]*dielectric_jump1;
	  } else{
		  pre_up[i+1] = pre_up[i]*dielectric_jump1; 
		  pre_down[i+1] = pre_down[i]*dielectric_jump2;
	  }
  }
	  
    // (k,0,0), (0,l,0), (0,0,m)
	
    for (ic = 0; ic < 3; ic++) {
      sqk = unitk[ic]*unitk[ic];
      if (sqk <= gsqmx) {
		  if (ic==0||ic==1) {
			  tempA_rl=0;
		    tempA_im=0;
		    tempB_rl=0;
		    tempB_im=0;  
        for(temp_count=0;temp_count<order_image_long;temp_count=temp_count+2){
				  tempA_rl=tempA_rl+pre_up[temp_count]+pre_down[temp_count];
				  tempA_im=0;
			  }
			  for(temp_count=1;temp_count<order_image_long;temp_count=temp_count+2){
				  tempB_rl=tempB_rl+pre_up[temp_count]+pre_down[temp_count];
				  tempB_im=0;
			  }
			  A_rl[n]=tempA_rl;
			  A_im[n]=tempA_im;
			  B_rl[n]=tempB_rl;
			  B_im[n]=tempB_im;
			  n=n+1;
		    } else{
          tempA_rl=0;
		      tempA_im=0;
		      tempB_rl=0;
		      tempB_im=0;
			    for(temp_count=0;temp_count<order_image_long;temp_count=temp_count+2){
			      tempA_rl=tempA_rl+(pre_down[temp_count]+pre_up[temp_count])*cos(unitk[2]*(1+temp_count)*dielectric_interface_L);
				    tempA_im=tempA_im+(-pre_down[temp_count]+pre_up[temp_count])*sin(unitk[2]*(1+temp_count)*dielectric_interface_L);
			    }
			    for(temp_count=1;temp_count<order_image_long;temp_count=temp_count+2){
			      tempB_rl=tempB_rl+(pre_down[temp_count]+pre_up[temp_count])*cos(unitk[2]*(1+temp_count)*dielectric_interface_L);
				    tempB_im=tempB_im+(-pre_down[temp_count]+pre_up[temp_count])*sin(unitk[2]*(1+temp_count)*dielectric_interface_L);
			    }
			    A_rl[n]=tempA_rl;
			    A_im[n]=tempA_im;
			    B_rl[n]=tempB_rl;
			    B_im[n]=tempB_im;
			    n=n+1;
		  }
	  }
	}
		  
    for (m = 2; m <= kmax; m++) {
      for (ic = 0; ic < 3; ic++) {
        sqk = m*unitk[ic] * m*unitk[ic];
        if (sqk <= gsqmx) {
          if (ic==0||ic==1) {
			     tempA_rl=0;
		       tempA_im=0;
		       tempB_rl=0;
		       tempB_im=0;
			     for(temp_count=0;temp_count<order_image_long;temp_count=temp_count+2){
			       tempA_rl=tempA_rl+pre_up[temp_count]+pre_down[temp_count];
				     tempA_im=0;
           }
			     for(temp_count=1;temp_count<order_image_long;temp_count=temp_count+2){
			       tempB_rl=tempB_rl+pre_up[temp_count]+pre_down[temp_count];
				     tempB_im=0;
           }
			     A_rl[n]=tempA_rl;
			     A_im[n]=tempA_im;
			     B_rl[n]=tempB_rl;
			     B_im[n]=tempB_im;
			     n=n+1;
         } else{
          tempA_rl=0;
		      tempA_im=0;
		      tempB_rl=0;
		      tempB_im=0;
          for(temp_count=0;temp_count<order_image_long;temp_count=temp_count+2){
			       tempA_rl=tempA_rl+(pre_down[temp_count]+pre_up[temp_count])*cos(m*unitk[2]*(1+temp_count)*dielectric_interface_L);
             tempA_im=tempA_im+(-pre_down[temp_count]+pre_up[temp_count])*sin(m*unitk[2]*(1+temp_count)*dielectric_interface_L);
           }
           for(temp_count=1;temp_count<order_image_long;temp_count=temp_count+2){
			       tempB_rl=tempB_rl+(pre_down[temp_count]+pre_up[temp_count])*cos(m*unitk[2]*(1+temp_count)*dielectric_interface_L);
				     tempB_im=tempB_im+(-pre_down[temp_count]+pre_up[temp_count])*sin(m*unitk[2]*(1+temp_count)*dielectric_interface_L);
           }
			     A_rl[n]=tempA_rl;
			     A_im[n]=tempA_im;
			     B_rl[n]=tempB_rl;
			     B_im[n]=tempB_im;
			     n=n+1;
		      }
		  }
	  }
	}
    // 1 = (k,l,0), 2 = (k,-l,0)

    for (k = 1; k <= kxmax; k++) {
      for (l = 1; l <= kymax; l++) {
        sqk = (k*unitk[0] * k*unitk[0]) + (l*unitk[1] * l*unitk[1]);
        if (sqk <= gsqmx) {
          tempA_rl=0;
		      tempA_im=0;
		      tempB_rl=0;
		      tempB_im=0;
			    for(temp_count=0;temp_count<order_image_long;temp_count=temp_count+2){
			      tempA_rl=tempA_rl+pre_up[temp_count]+pre_down[temp_count];
            tempA_im=0;
          }
			    for(temp_count=1;temp_count<order_image_long;temp_count=temp_count+2){
			       tempB_rl=tempB_rl+pre_up[temp_count]+pre_down[temp_count];
				     tempB_im=0;
           }
			     A_rl[n]=tempA_rl;
			     A_im[n]=tempA_im;
			     B_rl[n]=tempB_rl;
		     	 B_im[n]=tempB_im;
			     n=n+1;
			     A_rl[n]=tempA_rl;
			     A_im[n]=tempA_im;
			     B_rl[n]=tempB_rl;
			     B_im[n]=tempB_im;
			     n=n+1;
        }
      }
    }

    // 1 = (0,l,m), 2 = (0,l,-m)

    for (l = 1; l <= kymax; l++) {
      for (m = 1; m <= kzmax; m++) {
        sqk = (l*unitk[1] * l*unitk[1]) + (m*unitk[2] * m*unitk[2]);
        if (sqk <= gsqmx) {
          tempA_rl=0;
		      tempA_im=0;
		      tempB_rl=0;
		      tempB_im=0;
			    for(temp_count=0;temp_count<order_image_long;temp_count=temp_count+2){
            tempA_rl=tempA_rl+(pre_down[temp_count]+pre_up[temp_count])*cos(m*unitk[2]*(1+temp_count)*dielectric_interface_L);
            tempA_im=tempA_im+(-pre_down[temp_count]+pre_up[temp_count])*sin(m*unitk[2]*(1+temp_count)*dielectric_interface_L);
          }
          for(temp_count=1;temp_count<order_image_long;temp_count=temp_count+2){
            tempB_rl=tempB_rl+(pre_down[temp_count]+pre_up[temp_count])*cos(m*unitk[2]*(1+temp_count)*dielectric_interface_L);
            tempB_im=tempB_im+(-pre_down[temp_count]+pre_up[temp_count])*sin(m*unitk[2]*(1+temp_count)*dielectric_interface_L);
          }
          A_rl[n]=tempA_rl;
			    A_im[n]=tempA_im;
			    B_rl[n]=tempB_rl;
			    B_im[n]=tempB_im;
			    n=n+1;
			    A_rl[n]=tempA_rl;
			    A_im[n]=-tempA_im;
			    B_rl[n]=tempB_rl;
			    B_im[n]=-tempB_im;
			    n=n+1;
        }
      }
    }

    // 1 = (k,0,m), 2 = (k,0,-m)

    for (k = 1; k <= kxmax; k++) {
      for (m = 1; m <= kzmax; m++) {
        sqk = (k*unitk[0] * k*unitk[0]) + (m*unitk[2] * m*unitk[2]);
        if (sqk <= gsqmx) {
          tempA_rl=0;
		      tempA_im=0;
		      tempB_rl=0;
		      tempB_im=0;
			    for(temp_count=0;temp_count<order_image_long;temp_count=temp_count+2){
			      tempA_rl=tempA_rl+(pre_down[temp_count]+pre_up[temp_count])*cos(m*unitk[2]*(1+temp_count)*dielectric_interface_L);
			      tempA_im=tempA_im+(-pre_down[temp_count]+pre_up[temp_count])*sin(m*unitk[2]*(1+temp_count)*dielectric_interface_L);
          }
			    for(temp_count=1;temp_count<order_image_long;temp_count=temp_count+2){
			      tempB_rl=tempB_rl+(pre_down[temp_count]+pre_up[temp_count])*cos(m*unitk[2]*(1+temp_count)*dielectric_interface_L);
				    tempB_im=tempB_im+(-pre_down[temp_count]+pre_up[temp_count])*sin(m*unitk[2]*(1+temp_count)*dielectric_interface_L);
			    }
          A_rl[n]=tempA_rl;
			    A_im[n]=tempA_im;
			    B_rl[n]=tempB_rl;
			    B_im[n]=tempB_im;
			    n=n+1;
			    A_rl[n]=tempA_rl;
			    A_im[n]=-tempA_im;
			    B_rl[n]=tempB_rl;
			    B_im[n]=-tempB_im;
			    n=n+1;
        }
      }
    }

    // 1 = (k,l,m), 2 = (k,-l,m), 3 = (k,l,-m), 4 = (k,-l,-m)

    for (k = 1; k <= kxmax; k++) {
      for (l = 1; l <= kymax; l++) {
        for (m = 1; m <= kzmax; m++) {
          sqk = (k*unitk[0] * k*unitk[0]) + (l*unitk[1] * l*unitk[1]) +
          (m*unitk[2] * m*unitk[2]);
          if (sqk <= gsqmx) {
            tempA_rl=0;
            tempA_im=0;
		        tempB_rl=0;
		        tempB_im=0;
			      for(temp_count=0;temp_count<order_image_long;temp_count=temp_count+2){
			        tempA_rl=tempA_rl+(pre_down[temp_count]+pre_up[temp_count])*cos(m*unitk[2]*(1+temp_count)*dielectric_interface_L);
			        tempA_im=tempA_im+(-pre_down[temp_count]+pre_up[temp_count])*sin(m*unitk[2]*(1+temp_count)*dielectric_interface_L);
            }
			      for(temp_count=1;temp_count<order_image_long;temp_count=temp_count+2){
			        tempB_rl=tempB_rl+(pre_down[temp_count]+pre_up[temp_count])*cos(m*unitk[2]*(1+temp_count)*dielectric_interface_L);
				      tempB_im=tempB_im+(-pre_down[temp_count]+pre_up[temp_count])*sin(m*unitk[2]*(1+temp_count)*dielectric_interface_L);
			      }
			      A_rl[n]=tempA_rl;
			      A_im[n]=tempA_im;
			      B_rl[n]=tempB_rl;
			      B_im[n]=tempB_im;
			      n=n+1;
			      A_rl[n]=tempA_rl;
			      A_im[n]=tempA_im;
			      B_rl[n]=tempB_rl;
			      B_im[n]=tempB_im;
			      n=n+1;
			      A_rl[n]=tempA_rl;
			      A_im[n]=-tempA_im;
			      B_rl[n]=tempB_rl;
            B_im[n]=-tempB_im;
			      n=n+1;
			      A_rl[n]=tempA_rl;
			      A_im[n]=-tempA_im;
		        B_rl[n]=tempB_rl;
			      B_im[n]=-tempB_im;
			      n=n+1;
          }
        }
      }
    }
  }

}

/* ----------------------------------------------------------------------
   compute RMS accuracy for a dimension
------------------------------------------------------------------------- */

double Ewald::rms(int km, double prd, bigint natoms, double q2)
{
  double value = 2.0*q2*g_ewald/prd *
    sqrt(1.0/(MY_PI*km*natoms)) *
    exp(-MY_PI*MY_PI*km*km/(g_ewald*g_ewald*prd*prd));

  return value;
}

/* ----------------------------------------------------------------------
   compute the Ewald long-range force, energy, virial
------------------------------------------------------------------------- */

void Ewald::compute(int eflag, int vflag)
{
  int i,j,k;
  int temp_1,temp_2,temp_3;
  imageflag=force->kspace->imageflag;
  dielectric_jump1=force->kspace->dielectric_jump1;//dielectric interface
  dielectric_jump2=force->kspace->dielectric_jump2;//dielectric interface
  dielectric_interface_L=force->kspace->dielectric_interface_L;
  order_image_short=force->kspace->order_image_short;
  order_image_long=force->kspace->order_image_long;
  
  // set energy/virial flags
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = evflag_atom = eflag_global = vflag_global =
         eflag_atom = vflag_atom = 0;

  // if atom count has changed, update qsum and qsqsum

  if (atom->natoms != natoms_original) {
    qsum_qsq();
    natoms_original = atom->natoms;
  }

  // return if there are no charges

  if (qsqsum == 0.0) return;

  // extend size of per-atom arrays if necessary

  if (atom->nmax > nmax) {
    memory->destroy(ek);
    memory->destroy3d_offset(cs,-kmax_created);
    memory->destroy3d_offset(sn,-kmax_created);
    nmax = atom->nmax;
    memory->create(ek,nmax,3,"ewald:ek");
    memory->create3d_offset(cs,-kmax,kmax,3,nmax,"ewald:cs");
    memory->create3d_offset(sn,-kmax,kmax,3,nmax,"ewald:sn");
    kmax_created = kmax;
  }

  // partial structure factors on each processor
  // total structure factor by summing over procs

  if (triclinic == 0)
    eik_dot_r();
  else
    eik_dot_r_triclinic();

  MPI_Allreduce(sfacrl,sfacrl_all,kcount,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(sfacim,sfacim_all,kcount,MPI_DOUBLE,MPI_SUM,world);
  
  if (imageflag==1) {
	   MPI_Allreduce(image_sfacrl,image_sfacrl_all,kcount,MPI_DOUBLE,MPI_SUM,world);
     MPI_Allreduce(image_sfacim,image_sfacim_all,kcount,MPI_DOUBLE,MPI_SUM,world);
  }

  // K-space portion of electric field
  // double loop over K-vectors and local atoms
  // perform per-atom calculations if needed

  double **f = atom->f;
  double *q = atom->q;
  int nlocal = atom->nlocal;

  int kx,ky,kz;
  double cypz,sypz,exprl,expim,partial,partial_peratom,partial_image,partial_image_extra,cypz_image,sypz_image,exprl_image,expim_image;

  for (i = 0; i < nlocal; i++) {
    ek[i][0] = 0.0;
    ek[i][1] = 0.0;
    ek[i][2] = 0.0;
  }

  for (k = 0; k < kcount; k++) {
    kx = kxvecs[k];
    ky = kyvecs[k];
    kz = kzvecs[k];

    for (i = 0; i < nlocal; i++) {
      cypz = cs[ky][1][i]*cs[kz][2][i] - sn[ky][1][i]*sn[kz][2][i];
      sypz = sn[ky][1][i]*cs[kz][2][i] + cs[ky][1][i]*sn[kz][2][i];
      exprl = cs[kx][0][i]*cypz - sn[kx][0][i]*sypz;
      expim = sn[kx][0][i]*cypz + cs[kx][0][i]*sypz;
      partial = expim*sfacrl_all[k] - exprl*sfacim_all[k];
      ek[i][0] += partial*eg[k][0];
      ek[i][1] += partial*eg[k][1];
      ek[i][2] += partial*eg[k][2];
	  if (imageflag==1) {
	      ek[i][0] += partial*eg[k][0]*B_rl[k];
        ek[i][1] += partial*eg[k][1]*B_rl[k];
        ek[i][2] += partial*eg[k][2]*B_rl[k];
		    cypz_image=cs[ky][1][i]*cs[kz][2][i] + sn[ky][1][i]*sn[kz][2][i];
        sypz_image=sn[ky][1][i]*cs[kz][2][i] - cs[ky][1][i]*sn[kz][2][i];
        exprl_image=cs[kx][0][i]*cypz_image - sn[kx][0][i]*sypz_image;
        expim_image=sn[kx][0][i]*cypz_image + cs[kx][0][i]*sypz_image;
        partial_image=(expim*image_sfacrl_all[k] - exprl*image_sfacim_all[k])*A_rl[k]-
        (exprl*image_sfacrl_all[k]+expim*image_sfacim_all[k])*A_im[k];
        partial_image_extra=(sfacrl_all[k]*expim_image-sfacim_all[k]*exprl_image)*A_rl[k]+
		    (sfacrl_all[k]*exprl_image+sfacim_all[k]*expim_image)*A_im[k];
        ek[i][0] += (partial_image+partial_image_extra)*0.5*eg[k][0];
        ek[i][1] += (partial_image+partial_image_extra)*0.5*eg[k][1];
        ek[i][2] += (partial_image-partial_image_extra)*0.5*eg[k][2];
      }

      if (evflag_atom) {
        partial_peratom = exprl*sfacrl_all[k] + expim*sfacim_all[k];
		  if (imageflag==1) {
          partial_peratom=partial_peratom+B_rl[k]*(exprl*sfacrl_all[k] + expim*sfacim_all[k])+B_im[k]*(-exprl*sfacim_all[k]+expim*sfacrl_all[k])
          +A_rl[k]*(exprl*image_sfacrl_all[k] + expim*image_sfacim_all[k])
          +A_im[k]*(-exprl*image_sfacim_all[k]+expim*image_sfacrl_all[k]);
        }
        if (eflag_atom) eatom[i] += q[i]*ug[k]*partial_peratom;
        if (vflag_atom)
          for (j = 0; j < 6; j++)
            vatom[i][j] += ug[k]*vg[k][j]*partial_peratom;
      }
    }
  }

  // convert E-field to force

  const double qscale = qqrd2e * scale;

  for (i = 0; i < nlocal; i++) {
    f[i][0] += qscale * q[i]*ek[i][0];
    f[i][1] += qscale * q[i]*ek[i][1];
    if (slabflag != 2) f[i][2] += qscale * q[i]*ek[i][2];
  }

  // sum global energy across Kspace vevs and add in volume-dependent term

  if (eflag_global) {
    for (k = 0; k < kcount; k++) {
      energy += ug[k] * (sfacrl_all[k]*sfacrl_all[k] +
                         sfacim_all[k]*sfacim_all[k]);
      if (imageflag==1) {
        energy = energy + ug[k] * (sfacrl_all[k]*sfacrl_all[k] + sfacim_all[k]*sfacim_all[k])*B_rl[k]
		    +ug[k] *(sfacrl_all[k]*image_sfacrl_all[k]+sfacim_all[k]*image_sfacim_all[k])*A_rl[k]
		    +ug[k] *(sfacim_all[k]*image_sfacrl_all[k]-image_sfacim_all[k]*sfacrl_all[k])*A_im[k];
      }
    }

    energy -= g_ewald*qsqsum/MY_PIS;//+MY_PI2*qsum*qsum / (g_ewald*g_ewald*volume);
    energy *= qscale;
  }

  // global virial

  if (vflag_global) {
    double uk;
    for (k = 0; k < kcount; k++) {
      uk = ug[k] * (sfacrl_all[k]*sfacrl_all[k] + sfacim_all[k]*sfacim_all[k])*B_rl[k]
      +ug[k] *(sfacrl_all[k]*image_sfacrl_all[k]+sfacim_all[k]*image_sfacim_all[k])*A_rl[k]
      +ug[k] *(sfacim_all[k]*image_sfacrl_all[k]-image_sfacim_all[k]*sfacrl_all[k])*A_im[k];
      for (j = 0; j < 6; j++) virial[j] += uk*vg[k][j];
    }
    for (j = 0; j < 6; j++) virial[j] *= qscale;
  }

  // per-atom energy/virial
  // energy includes self-energy correction

  if (evflag_atom) {
    if (eflag_atom) {
      for (i = 0; i < nlocal; i++) {
        eatom[i] -= g_ewald*q[i]*q[i]/MY_PIS; //+ MY_PI2*q[i]*qsum/(g_ewald*g_ewald*volume);
        eatom[i] *= qscale;
      }
    }

    if (vflag_atom)
      for (i = 0; i < nlocal; i++)
        for (j = 0; j < 6; j++) vatom[i][j] *= q[i]*qscale;
  }

  // 2d slab correction

  if (slabflag == 1) slabcorr();
}

/* ---------------------------------------------------------------------- */
void Ewald::eik_dot_r()
{
  dielectric_jump1=force->kspace->dielectric_jump1;//dielectric interface
  dielectric_jump2=force->kspace->dielectric_jump2;//dielectric interface
  dielectric_interface_L=force->kspace->dielectric_interface_L;
  order_image_short=force->kspace->order_image_short;
  order_image_long=force->kspace->order_image_long;
  int i,k,l,m,n,ic;
  double cstr1,sstr1,cstr2,sstr2,cstr3,sstr3,cstr4,sstr4;
  double sqk,clpm,slpm;

  double **x = atom->x;
  double *q = atom->q;
  int nlocal = atom->nlocal;
  n=0;

  if (imageflag==1) {

    // (k,0,0), (0,l,0), (0,0,m)

    for (ic = 0; ic < 3; ic++) {
      sqk = unitk[ic]*unitk[ic];
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;
        for (i = 0; i < nlocal; i++) {
          cs[0][ic][i] = 1.0;
          sn[0][ic][i] = 0.0;
          cs[1][ic][i] = cos(unitk[ic]*x[i][ic]);
          sn[1][ic][i] = sin(unitk[ic]*x[i][ic]);
          cs[-1][ic][i] = cs[1][ic][i];
          sn[-1][ic][i] = -sn[1][ic][i];
          cstr1 += q[i]*cs[1][ic][i];
          sstr1 += q[i]*sn[1][ic][i];
        }
        if (ic==0||ic==1) {
          sfacrl[n] = cstr1;
          image_sfacrl[n]=cstr1;
          image_sfacim[n]=sstr1;
          sfacim[n] = sstr1;
          n=n+1;
        } else {
          sfacrl[n] = cstr1;
          image_sfacrl[n]=cstr1;
          image_sfacim[n]=-sstr1;
          sfacim[n] = sstr1;
          n=n+1;
        }
      }
    }

    for (m = 2; m <= kmax; m++) {
      for (ic = 0; ic < 3; ic++) {
        sqk = m*unitk[ic] * m*unitk[ic];
        if (sqk <= gsqmx) {
          cstr1 = 0.0;
          sstr1 = 0.0;
          for (i = 0; i < nlocal; i++) {
            cs[m][ic][i] = cs[m-1][ic][i]*cs[1][ic][i] -
            sn[m-1][ic][i]*sn[1][ic][i];
            sn[m][ic][i] = sn[m-1][ic][i]*cs[1][ic][i] +
            cs[m-1][ic][i]*sn[1][ic][i];
            cs[-m][ic][i] = cs[m][ic][i];
            sn[-m][ic][i] = -sn[m][ic][i];
            cstr1 += q[i]*cs[m][ic][i];
            sstr1 += q[i]*sn[m][ic][i];
          }
          if (ic==0||ic==1) {
            sfacrl[n] = cstr1;
            image_sfacrl[n]=cstr1;
            sfacim[n] = sstr1;
            image_sfacim[n]=sstr1;
            n=n+1;
          } else {
            sfacrl[n] = cstr1;
            image_sfacrl[n]=cstr1;
            sfacim[n] = sstr1;
            image_sfacim[n]=-sstr1;
            n=n+1;
          }
        }
      }
    }

    // 1 = (k,l,0), 2 = (k,-l,0)

    for (k = 1; k <= kxmax; k++) {
      for (l = 1; l <= kymax; l++) {
        sqk = (k*unitk[0] * k*unitk[0]) + (l*unitk[1] * l*unitk[1]);
        if (sqk <= gsqmx) {
          cstr1 = 0.0;
          sstr1 = 0.0;
          cstr2 = 0.0;
          sstr2 = 0.0;
          for (i = 0; i < nlocal; i++) {
            cstr1 += q[i]*(cs[k][0][i]*cs[l][1][i] - sn[k][0][i]*sn[l][1][i]);
            sstr1 += q[i]*(sn[k][0][i]*cs[l][1][i] + cs[k][0][i]*sn[l][1][i]);
            cstr2 += q[i]*(cs[k][0][i]*cs[l][1][i] + sn[k][0][i]*sn[l][1][i]);
            sstr2 += q[i]*(sn[k][0][i]*cs[l][1][i] - cs[k][0][i]*sn[l][1][i]);
          }
          sfacrl[n] = cstr1;
          image_sfacrl[n]=cstr1;
          sfacim[n] = sstr1;
          image_sfacim[n]=sstr1;
          n=n+1;
          sfacrl[n] = cstr2;
          image_sfacrl[n]=cstr2;
          sfacim[n] = sstr2;
          image_sfacim[n]=sstr2;
          n=n+1;
        }
      }
    }

    // 1 = (0,l,m), 2 = (0,l,-m)

    for (l = 1; l <= kymax; l++) {
      for (m = 1; m <= kzmax; m++) {
        sqk = (l*unitk[1] * l*unitk[1]) + (m*unitk[2] * m*unitk[2]);
        if (sqk <= gsqmx) {
          cstr1 = 0.0;
          sstr1 = 0.0;
          cstr2 = 0.0;
          sstr2 = 0.0;
          for (i = 0; i < nlocal; i++) {
            cstr1 += q[i]*(cs[l][1][i]*cs[m][2][i] - sn[l][1][i]*sn[m][2][i]);
            sstr1 += q[i]*(sn[l][1][i]*cs[m][2][i] + cs[l][1][i]*sn[m][2][i]);
            cstr2 += q[i]*(cs[l][1][i]*cs[m][2][i] + sn[l][1][i]*sn[m][2][i]);
            sstr2 += q[i]*(sn[l][1][i]*cs[m][2][i] - cs[l][1][i]*sn[m][2][i]);
          }
          sfacrl[n] = cstr1;
          image_sfacrl[n]=cstr2;
          sfacim[n] = sstr1;
          image_sfacim[n]=sstr2;
          n=n+1;
          sfacrl[n] = cstr2;
          image_sfacrl[n]=cstr1;
          sfacim[n] = sstr2;
          image_sfacim[n]=sstr1;
          n=n+1;
        }
      }
    }

    // 1 = (k,0,m), 2 = (k,0,-m)

    for (k = 1; k <= kxmax; k++) {
      for (m = 1; m <= kzmax; m++) {
        sqk = (k*unitk[0] * k*unitk[0]) + (m*unitk[2] * m*unitk[2]);
        if (sqk <= gsqmx) {
          cstr1 = 0.0;
          sstr1 = 0.0;
          cstr2 = 0.0;
          sstr2 = 0.0;
          for (i = 0; i < nlocal; i++) {
            cstr1 += q[i]*(cs[k][0][i]*cs[m][2][i] - sn[k][0][i]*sn[m][2][i]);
            sstr1 += q[i]*(sn[k][0][i]*cs[m][2][i] + cs[k][0][i]*sn[m][2][i]);
            cstr2 += q[i]*(cs[k][0][i]*cs[m][2][i] + sn[k][0][i]*sn[m][2][i]);
            sstr2 += q[i]*(sn[k][0][i]*cs[m][2][i] - cs[k][0][i]*sn[m][2][i]);
          }
          sfacrl[n] = cstr1;
          image_sfacrl[n]=cstr2;
          sfacim[n] = sstr1;
          image_sfacim[n]=sstr2;
          n=n+1;
          sfacrl[n] = cstr2;
          image_sfacrl[n]=cstr1;
          sfacim[n] = sstr2;
          image_sfacim[n]=sstr1;
          n=n+1;
        }
      }
    }

    // 1 = (k,l,m), 2 = (k,-l,m), 3 = (k,l,-m), 4 = (k,-l,-m)

    for (k = 1; k <= kxmax; k++) {
      for (l = 1; l <= kymax; l++) {
        for (m = 1; m <= kzmax; m++) {
          sqk = (k*unitk[0] * k*unitk[0]) + (l*unitk[1] * l*unitk[1]) +
          (m*unitk[2] * m*unitk[2]);
          if (sqk <= gsqmx) {
            cstr1 = 0.0;
            sstr1 = 0.0;
            cstr2 = 0.0;
            sstr2 = 0.0;
            cstr3 = 0.0;
            sstr3 = 0.0;
            cstr4 = 0.0;
            sstr4 = 0.0;
            for (i = 0; i < nlocal; i++) {
              clpm = cs[l][1][i]*cs[m][2][i] - sn[l][1][i]*sn[m][2][i];
              slpm = sn[l][1][i]*cs[m][2][i] + cs[l][1][i]*sn[m][2][i];
              cstr1 += q[i]*(cs[k][0][i]*clpm - sn[k][0][i]*slpm);
              sstr1 += q[i]*(sn[k][0][i]*clpm + cs[k][0][i]*slpm);

              clpm = cs[l][1][i]*cs[m][2][i] + sn[l][1][i]*sn[m][2][i];
              slpm = -sn[l][1][i]*cs[m][2][i] + cs[l][1][i]*sn[m][2][i];
              cstr2 += q[i]*(cs[k][0][i]*clpm - sn[k][0][i]*slpm);
              sstr2 += q[i]*(sn[k][0][i]*clpm + cs[k][0][i]*slpm);

              clpm = cs[l][1][i]*cs[m][2][i] + sn[l][1][i]*sn[m][2][i];
              slpm = sn[l][1][i]*cs[m][2][i] - cs[l][1][i]*sn[m][2][i];
              cstr3 += q[i]*(cs[k][0][i]*clpm - sn[k][0][i]*slpm);
              sstr3 += q[i]*(sn[k][0][i]*clpm + cs[k][0][i]*slpm);

              clpm = cs[l][1][i]*cs[m][2][i] - sn[l][1][i]*sn[m][2][i];
              slpm = -sn[l][1][i]*cs[m][2][i] - cs[l][1][i]*sn[m][2][i];
              cstr4 += q[i]*(cs[k][0][i]*clpm - sn[k][0][i]*slpm);
              sstr4 += q[i]*(sn[k][0][i]*clpm + cs[k][0][i]*slpm);
            }
            sfacrl[n] = cstr1;
            image_sfacrl[n]=cstr3;
            sfacim[n] = sstr1;
            image_sfacim[n]=sstr3;
            n=n+1;
            sfacrl[n] = cstr2;
            image_sfacrl[n]=cstr4;
            sfacim[n] = sstr2;
            image_sfacim[n]=sstr4;
            n=n+1;
            sfacrl[n] = cstr3;
            image_sfacrl[n]=cstr1;
            sfacim[n] = sstr3;
            image_sfacim[n]=sstr1;
            n=n+1;
            sfacrl[n] = cstr4;
            image_sfacrl[n]=cstr2;
            sfacim[n] = sstr4;
            image_sfacim[n]=sstr2;
            n=n+1;
          }
        }
      }
    }
  } else{
	    n = 0;

  // (k,0,0), (0,l,0), (0,0,m)

  for (ic = 0; ic < 3; ic++) {
    sqk = unitk[ic]*unitk[ic];
    if (sqk <= gsqmx) {
      cstr1 = 0.0;
      sstr1 = 0.0;
      for (i = 0; i < nlocal; i++) {
        cs[0][ic][i] = 1.0;
        sn[0][ic][i] = 0.0;
        cs[1][ic][i] = cos(unitk[ic]*x[i][ic]);
        sn[1][ic][i] = sin(unitk[ic]*x[i][ic]);
        cs[-1][ic][i] = cs[1][ic][i];
        sn[-1][ic][i] = -sn[1][ic][i];
        cstr1 += q[i]*cs[1][ic][i];
        sstr1 += q[i]*sn[1][ic][i];
      }
      sfacrl[n] = cstr1;
      sfacim[n++] = sstr1;
    }
  }

  for (m = 2; m <= kmax; m++) {
    for (ic = 0; ic < 3; ic++) {
      sqk = m*unitk[ic] * m*unitk[ic];
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;
        for (i = 0; i < nlocal; i++) {
          cs[m][ic][i] = cs[m-1][ic][i]*cs[1][ic][i] -
            sn[m-1][ic][i]*sn[1][ic][i];
          sn[m][ic][i] = sn[m-1][ic][i]*cs[1][ic][i] +
            cs[m-1][ic][i]*sn[1][ic][i];
          cs[-m][ic][i] = cs[m][ic][i];
          sn[-m][ic][i] = -sn[m][ic][i];
          cstr1 += q[i]*cs[m][ic][i];
          sstr1 += q[i]*sn[m][ic][i];
        }
        sfacrl[n] = cstr1;
        sfacim[n++] = sstr1;
      }
    }
  }

  // 1 = (k,l,0), 2 = (k,-l,0)

  for (k = 1; k <= kxmax; k++) {
    for (l = 1; l <= kymax; l++) {
      sqk = (k*unitk[0] * k*unitk[0]) + (l*unitk[1] * l*unitk[1]);
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;
        cstr2 = 0.0;
        sstr2 = 0.0;
        for (i = 0; i < nlocal; i++) {
          cstr1 += q[i]*(cs[k][0][i]*cs[l][1][i] - sn[k][0][i]*sn[l][1][i]);
          sstr1 += q[i]*(sn[k][0][i]*cs[l][1][i] + cs[k][0][i]*sn[l][1][i]);
          cstr2 += q[i]*(cs[k][0][i]*cs[l][1][i] + sn[k][0][i]*sn[l][1][i]);
          sstr2 += q[i]*(sn[k][0][i]*cs[l][1][i] - cs[k][0][i]*sn[l][1][i]);
        }
        sfacrl[n] = cstr1;
        sfacim[n++] = sstr1;
        sfacrl[n] = cstr2;
        sfacim[n++] = sstr2;
      }
    }
  }

  // 1 = (0,l,m), 2 = (0,l,-m)

  for (l = 1; l <= kymax; l++) {
    for (m = 1; m <= kzmax; m++) {
      sqk = (l*unitk[1] * l*unitk[1]) + (m*unitk[2] * m*unitk[2]);
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;
        cstr2 = 0.0;
        sstr2 = 0.0;
        for (i = 0; i < nlocal; i++) {
          cstr1 += q[i]*(cs[l][1][i]*cs[m][2][i] - sn[l][1][i]*sn[m][2][i]);
          sstr1 += q[i]*(sn[l][1][i]*cs[m][2][i] + cs[l][1][i]*sn[m][2][i]);
          cstr2 += q[i]*(cs[l][1][i]*cs[m][2][i] + sn[l][1][i]*sn[m][2][i]);
          sstr2 += q[i]*(sn[l][1][i]*cs[m][2][i] - cs[l][1][i]*sn[m][2][i]);
        }
        sfacrl[n] = cstr1;
        sfacim[n++] = sstr1;
        sfacrl[n] = cstr2;
        sfacim[n++] = sstr2;
      }
    }
  }

  // 1 = (k,0,m), 2 = (k,0,-m)

  for (k = 1; k <= kxmax; k++) {
    for (m = 1; m <= kzmax; m++) {
      sqk = (k*unitk[0] * k*unitk[0]) + (m*unitk[2] * m*unitk[2]);
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;
        cstr2 = 0.0;
        sstr2 = 0.0;
        for (i = 0; i < nlocal; i++) {
          cstr1 += q[i]*(cs[k][0][i]*cs[m][2][i] - sn[k][0][i]*sn[m][2][i]);
          sstr1 += q[i]*(sn[k][0][i]*cs[m][2][i] + cs[k][0][i]*sn[m][2][i]);
          cstr2 += q[i]*(cs[k][0][i]*cs[m][2][i] + sn[k][0][i]*sn[m][2][i]);
          sstr2 += q[i]*(sn[k][0][i]*cs[m][2][i] - cs[k][0][i]*sn[m][2][i]);
        }
        sfacrl[n] = cstr1;
        sfacim[n++] = sstr1;
        sfacrl[n] = cstr2;
        sfacim[n++] = sstr2;
      }
    }
  }

  // 1 = (k,l,m), 2 = (k,-l,m), 3 = (k,l,-m), 4 = (k,-l,-m)

  for (k = 1; k <= kxmax; k++) {
    for (l = 1; l <= kymax; l++) {
      for (m = 1; m <= kzmax; m++) {
        sqk = (k*unitk[0] * k*unitk[0]) + (l*unitk[1] * l*unitk[1]) +
          (m*unitk[2] * m*unitk[2]);
        if (sqk <= gsqmx) {
          cstr1 = 0.0;
          sstr1 = 0.0;
          cstr2 = 0.0;
          sstr2 = 0.0;
          cstr3 = 0.0;
          sstr3 = 0.0;
          cstr4 = 0.0;
          sstr4 = 0.0;
          for (i = 0; i < nlocal; i++) {
            clpm = cs[l][1][i]*cs[m][2][i] - sn[l][1][i]*sn[m][2][i];
            slpm = sn[l][1][i]*cs[m][2][i] + cs[l][1][i]*sn[m][2][i];
            cstr1 += q[i]*(cs[k][0][i]*clpm - sn[k][0][i]*slpm);
            sstr1 += q[i]*(sn[k][0][i]*clpm + cs[k][0][i]*slpm);

            clpm = cs[l][1][i]*cs[m][2][i] + sn[l][1][i]*sn[m][2][i];
            slpm = -sn[l][1][i]*cs[m][2][i] + cs[l][1][i]*sn[m][2][i];
            cstr2 += q[i]*(cs[k][0][i]*clpm - sn[k][0][i]*slpm);
            sstr2 += q[i]*(sn[k][0][i]*clpm + cs[k][0][i]*slpm);

            clpm = cs[l][1][i]*cs[m][2][i] + sn[l][1][i]*sn[m][2][i];
            slpm = sn[l][1][i]*cs[m][2][i] - cs[l][1][i]*sn[m][2][i];
            cstr3 += q[i]*(cs[k][0][i]*clpm - sn[k][0][i]*slpm);
            sstr3 += q[i]*(sn[k][0][i]*clpm + cs[k][0][i]*slpm);

            clpm = cs[l][1][i]*cs[m][2][i] - sn[l][1][i]*sn[m][2][i];
            slpm = -sn[l][1][i]*cs[m][2][i] - cs[l][1][i]*sn[m][2][i];
            cstr4 += q[i]*(cs[k][0][i]*clpm - sn[k][0][i]*slpm);
            sstr4 += q[i]*(sn[k][0][i]*clpm + cs[k][0][i]*slpm);
          }
          sfacrl[n] = cstr1;
          sfacim[n++] = sstr1;
          sfacrl[n] = cstr2;
          sfacim[n++] = sstr2;
          sfacrl[n] = cstr3;
          sfacim[n++] = sstr3;
          sfacrl[n] = cstr4;
          sfacim[n++] = sstr4;
        }
      }
    }
  }
  }
}


/* ---------------------------------------------------------------------- */

void Ewald::eik_dot_r_triclinic()
{
  int i,k,l,m,n,ic;
  double cstr1,sstr1;
  double sqk,clpm,slpm;

  double **x = atom->x;
  double *q = atom->q;
  int nlocal = atom->nlocal;

  double unitk_lamda[3];

  double max_kvecs[3];
  max_kvecs[0] = kxmax;
  max_kvecs[1] = kymax;
  max_kvecs[2] = kzmax;

  // (k,0,0), (0,l,0), (0,0,m)

  for (ic = 0; ic < 3; ic++) {
    unitk_lamda[0] = 0.0;
    unitk_lamda[1] = 0.0;
    unitk_lamda[2] = 0.0;
    unitk_lamda[ic] = 2.0*MY_PI;
    x2lamdaT(&unitk_lamda[0],&unitk_lamda[0]);
    sqk = unitk_lamda[ic]*unitk_lamda[ic];
    if (sqk <= gsqmx) {
      for (i = 0; i < nlocal; i++) {
        cs[0][ic][i] = 1.0;
        sn[0][ic][i] = 0.0;
        cs[1][ic][i] = cos(unitk_lamda[0]*x[i][0] + unitk_lamda[1]*x[i][1] + unitk_lamda[2]*x[i][2]);
        sn[1][ic][i] = sin(unitk_lamda[0]*x[i][0] + unitk_lamda[1]*x[i][1] + unitk_lamda[2]*x[i][2]);
        cs[-1][ic][i] = cs[1][ic][i];
        sn[-1][ic][i] = -sn[1][ic][i];
      }
    }
  }

  for (ic = 0; ic < 3; ic++) {
    for (m = 2; m <= max_kvecs[ic]; m++) {
      unitk_lamda[0] = 0.0;
      unitk_lamda[1] = 0.0;
      unitk_lamda[2] = 0.0;
      unitk_lamda[ic] = 2.0*MY_PI*m;
      x2lamdaT(&unitk_lamda[0],&unitk_lamda[0]);
      sqk = unitk_lamda[ic]*unitk_lamda[ic];
      for (i = 0; i < nlocal; i++) {
        cs[m][ic][i] = cs[m-1][ic][i]*cs[1][ic][i] -
          sn[m-1][ic][i]*sn[1][ic][i];
        sn[m][ic][i] = sn[m-1][ic][i]*cs[1][ic][i] +
          cs[m-1][ic][i]*sn[1][ic][i];
        cs[-m][ic][i] = cs[m][ic][i];
        sn[-m][ic][i] = -sn[m][ic][i];
      }
    }
  }

  for (n = 0; n < kcount; n++) {
    k = kxvecs[n];
    l = kyvecs[n];
    m = kzvecs[n];
    cstr1 = 0.0;
    sstr1 = 0.0;
    for (i = 0; i < nlocal; i++) {
      clpm = cs[l][1][i]*cs[m][2][i] - sn[l][1][i]*sn[m][2][i];
      slpm = sn[l][1][i]*cs[m][2][i] + cs[l][1][i]*sn[m][2][i];
      cstr1 += q[i]*(cs[k][0][i]*clpm - sn[k][0][i]*slpm);
      sstr1 += q[i]*(sn[k][0][i]*clpm + cs[k][0][i]*slpm);
    }
    sfacrl[n] = cstr1;
    sfacim[n] = sstr1;
  }
}

/* ----------------------------------------------------------------------
   pre-compute coefficients for each Ewald K-vector
------------------------------------------------------------------------- */

void Ewald::coeffs()
{
  int k,l,m;
  double sqk,vterm;

  double g_ewald_sq_inv = 1.0 / (g_ewald*g_ewald);
  double preu = 4.0*MY_PI/volume;

  kcount = 0;

  // (k,0,0), (0,l,0), (0,0,m)

  for (m = 1; m <= kmax; m++) {
    sqk = (m*unitk[0]) * (m*unitk[0]);
    if (sqk <= gsqmx) {
      kxvecs[kcount] = m;
      kyvecs[kcount] = 0;
      kzvecs[kcount] = 0;
      ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
      eg[kcount][0] = 2.0*unitk[0]*m*ug[kcount];
      eg[kcount][1] = 0.0;
      eg[kcount][2] = 0.0;
      vterm = -2.0*(1.0/sqk + 0.25*g_ewald_sq_inv);
      vg[kcount][0] = 1.0 + vterm*(unitk[0]*m)*(unitk[0]*m);
      vg[kcount][1] = 1.0;
      vg[kcount][2] = 1.0;
      vg[kcount][3] = 0.0;
      vg[kcount][4] = 0.0;
      vg[kcount][5] = 0.0;
      kcount++;
    }
    sqk = (m*unitk[1]) * (m*unitk[1]);
    if (sqk <= gsqmx) {
      kxvecs[kcount] = 0;
      kyvecs[kcount] = m;
      kzvecs[kcount] = 0;
      ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
      eg[kcount][0] = 0.0;
      eg[kcount][1] = 2.0*unitk[1]*m*ug[kcount];
      eg[kcount][2] = 0.0;
      vterm = -2.0*(1.0/sqk + 0.25*g_ewald_sq_inv);
      vg[kcount][0] = 1.0;
      vg[kcount][1] = 1.0 + vterm*(unitk[1]*m)*(unitk[1]*m);
      vg[kcount][2] = 1.0;
      vg[kcount][3] = 0.0;
      vg[kcount][4] = 0.0;
      vg[kcount][5] = 0.0;
      kcount++;
    }
    sqk = (m*unitk[2]) * (m*unitk[2]);
    if (sqk <= gsqmx) {
      kxvecs[kcount] = 0;
      kyvecs[kcount] = 0;
      kzvecs[kcount] = m;
      ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
      eg[kcount][0] = 0.0;
      eg[kcount][1] = 0.0;
      eg[kcount][2] = 2.0*unitk[2]*m*ug[kcount];
      vterm = -2.0*(1.0/sqk + 0.25*g_ewald_sq_inv);
      vg[kcount][0] = 1.0;
      vg[kcount][1] = 1.0;
      vg[kcount][2] = 1.0 + vterm*(unitk[2]*m)*(unitk[2]*m);
      vg[kcount][3] = 0.0;
      vg[kcount][4] = 0.0;
      vg[kcount][5] = 0.0;
      kcount++;
    }
  }

  // 1 = (k,l,0), 2 = (k,-l,0)

  for (k = 1; k <= kxmax; k++) {
    for (l = 1; l <= kymax; l++) {
      sqk = (unitk[0]*k) * (unitk[0]*k) + (unitk[1]*l) * (unitk[1]*l);
      if (sqk <= gsqmx) {
        kxvecs[kcount] = k;
        kyvecs[kcount] = l;
        kzvecs[kcount] = 0;
        ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
        eg[kcount][0] = 2.0*unitk[0]*k*ug[kcount];
        eg[kcount][1] = 2.0*unitk[1]*l*ug[kcount];
        eg[kcount][2] = 0.0;
        vterm = -2.0*(1.0/sqk + 0.25*g_ewald_sq_inv);
        vg[kcount][0] = 1.0 + vterm*(unitk[0]*k)*(unitk[0]*k);
        vg[kcount][1] = 1.0 + vterm*(unitk[1]*l)*(unitk[1]*l);
        vg[kcount][2] = 1.0;
        vg[kcount][3] = vterm*unitk[0]*k*unitk[1]*l;
        vg[kcount][4] = 0.0;
        vg[kcount][5] = 0.0;
        kcount++;

        kxvecs[kcount] = k;
        kyvecs[kcount] = -l;
        kzvecs[kcount] = 0;
        ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
        eg[kcount][0] = 2.0*unitk[0]*k*ug[kcount];
        eg[kcount][1] = -2.0*unitk[1]*l*ug[kcount];
        eg[kcount][2] = 0.0;
        vg[kcount][0] = 1.0 + vterm*(unitk[0]*k)*(unitk[0]*k);
        vg[kcount][1] = 1.0 + vterm*(unitk[1]*l)*(unitk[1]*l);
        vg[kcount][2] = 1.0;
        vg[kcount][3] = -vterm*unitk[0]*k*unitk[1]*l;
        vg[kcount][4] = 0.0;
        vg[kcount][5] = 0.0;
        kcount++;;
      }
    }
  }

  // 1 = (0,l,m), 2 = (0,l,-m)

  for (l = 1; l <= kymax; l++) {
    for (m = 1; m <= kzmax; m++) {
      sqk = (unitk[1]*l) * (unitk[1]*l) + (unitk[2]*m) * (unitk[2]*m);
      if (sqk <= gsqmx) {
        kxvecs[kcount] = 0;
        kyvecs[kcount] = l;
        kzvecs[kcount] = m;
        ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
        eg[kcount][0] =  0.0;
        eg[kcount][1] =  2.0*unitk[1]*l*ug[kcount];
        eg[kcount][2] =  2.0*unitk[2]*m*ug[kcount];
        vterm = -2.0*(1.0/sqk + 0.25*g_ewald_sq_inv);
        vg[kcount][0] = 1.0;
        vg[kcount][1] = 1.0 + vterm*(unitk[1]*l)*(unitk[1]*l);
        vg[kcount][2] = 1.0 + vterm*(unitk[2]*m)*(unitk[2]*m);
        vg[kcount][3] = 0.0;
        vg[kcount][4] = 0.0;
        vg[kcount][5] = vterm*unitk[1]*l*unitk[2]*m;
        kcount++;

        kxvecs[kcount] = 0;
        kyvecs[kcount] = l;
        kzvecs[kcount] = -m;
        ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
        eg[kcount][0] =  0.0;
        eg[kcount][1] =  2.0*unitk[1]*l*ug[kcount];
        eg[kcount][2] = -2.0*unitk[2]*m*ug[kcount];
        vg[kcount][0] = 1.0;
        vg[kcount][1] = 1.0 + vterm*(unitk[1]*l)*(unitk[1]*l);
        vg[kcount][2] = 1.0 + vterm*(unitk[2]*m)*(unitk[2]*m);
        vg[kcount][3] = 0.0;
        vg[kcount][4] = 0.0;
        vg[kcount][5] = -vterm*unitk[1]*l*unitk[2]*m;
        kcount++;
      }
    }
  }

  // 1 = (k,0,m), 2 = (k,0,-m)

  for (k = 1; k <= kxmax; k++) {
    for (m = 1; m <= kzmax; m++) {
      sqk = (unitk[0]*k) * (unitk[0]*k) + (unitk[2]*m) * (unitk[2]*m);
      if (sqk <= gsqmx) {
        kxvecs[kcount] = k;
        kyvecs[kcount] = 0;
        kzvecs[kcount] = m;
        ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
        eg[kcount][0] =  2.0*unitk[0]*k*ug[kcount];
        eg[kcount][1] =  0.0;
        eg[kcount][2] =  2.0*unitk[2]*m*ug[kcount];
        vterm = -2.0*(1.0/sqk + 0.25*g_ewald_sq_inv);
        vg[kcount][0] = 1.0 + vterm*(unitk[0]*k)*(unitk[0]*k);
        vg[kcount][1] = 1.0;
        vg[kcount][2] = 1.0 + vterm*(unitk[2]*m)*(unitk[2]*m);
        vg[kcount][3] = 0.0;
        vg[kcount][4] = vterm*unitk[0]*k*unitk[2]*m;
        vg[kcount][5] = 0.0;
        kcount++;

        kxvecs[kcount] = k;
        kyvecs[kcount] = 0;
        kzvecs[kcount] = -m;
        ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
        eg[kcount][0] =  2.0*unitk[0]*k*ug[kcount];
        eg[kcount][1] =  0.0;
        eg[kcount][2] = -2.0*unitk[2]*m*ug[kcount];
        vg[kcount][0] = 1.0 + vterm*(unitk[0]*k)*(unitk[0]*k);
        vg[kcount][1] = 1.0;
        vg[kcount][2] = 1.0 + vterm*(unitk[2]*m)*(unitk[2]*m);
        vg[kcount][3] = 0.0;
        vg[kcount][4] = -vterm*unitk[0]*k*unitk[2]*m;
        vg[kcount][5] = 0.0;
        kcount++;
      }
    }
  }

  // 1 = (k,l,m), 2 = (k,-l,m), 3 = (k,l,-m), 4 = (k,-l,-m)

  for (k = 1; k <= kxmax; k++) {
    for (l = 1; l <= kymax; l++) {
      for (m = 1; m <= kzmax; m++) {
        sqk = (unitk[0]*k) * (unitk[0]*k) + (unitk[1]*l) * (unitk[1]*l) +
          (unitk[2]*m) * (unitk[2]*m);
        if (sqk <= gsqmx) {
          kxvecs[kcount] = k;
          kyvecs[kcount] = l;
          kzvecs[kcount] = m;
          ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
          eg[kcount][0] = 2.0*unitk[0]*k*ug[kcount];
          eg[kcount][1] = 2.0*unitk[1]*l*ug[kcount];
          eg[kcount][2] = 2.0*unitk[2]*m*ug[kcount];
          vterm = -2.0*(1.0/sqk + 0.25*g_ewald_sq_inv);
          vg[kcount][0] = 1.0 + vterm*(unitk[0]*k)*(unitk[0]*k);
          vg[kcount][1] = 1.0 + vterm*(unitk[1]*l)*(unitk[1]*l);
          vg[kcount][2] = 1.0 + vterm*(unitk[2]*m)*(unitk[2]*m);
          vg[kcount][3] = vterm*unitk[0]*k*unitk[1]*l;
          vg[kcount][4] = vterm*unitk[0]*k*unitk[2]*m;
          vg[kcount][5] = vterm*unitk[1]*l*unitk[2]*m;
          kcount++;

          kxvecs[kcount] = k;
          kyvecs[kcount] = -l;
          kzvecs[kcount] = m;
          ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
          eg[kcount][0] = 2.0*unitk[0]*k*ug[kcount];
          eg[kcount][1] = -2.0*unitk[1]*l*ug[kcount];
          eg[kcount][2] = 2.0*unitk[2]*m*ug[kcount];
          vg[kcount][0] = 1.0 + vterm*(unitk[0]*k)*(unitk[0]*k);
          vg[kcount][1] = 1.0 + vterm*(unitk[1]*l)*(unitk[1]*l);
          vg[kcount][2] = 1.0 + vterm*(unitk[2]*m)*(unitk[2]*m);
          vg[kcount][3] = -vterm*unitk[0]*k*unitk[1]*l;
          vg[kcount][4] = vterm*unitk[0]*k*unitk[2]*m;
          vg[kcount][5] = -vterm*unitk[1]*l*unitk[2]*m;
          kcount++;

          kxvecs[kcount] = k;
          kyvecs[kcount] = l;
          kzvecs[kcount] = -m;
          ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
          eg[kcount][0] = 2.0*unitk[0]*k*ug[kcount];
          eg[kcount][1] = 2.0*unitk[1]*l*ug[kcount];
          eg[kcount][2] = -2.0*unitk[2]*m*ug[kcount];
          vg[kcount][0] = 1.0 + vterm*(unitk[0]*k)*(unitk[0]*k);
          vg[kcount][1] = 1.0 + vterm*(unitk[1]*l)*(unitk[1]*l);
          vg[kcount][2] = 1.0 + vterm*(unitk[2]*m)*(unitk[2]*m);
          vg[kcount][3] = vterm*unitk[0]*k*unitk[1]*l;
          vg[kcount][4] = -vterm*unitk[0]*k*unitk[2]*m;
          vg[kcount][5] = -vterm*unitk[1]*l*unitk[2]*m;
          kcount++;

          kxvecs[kcount] = k;
          kyvecs[kcount] = -l;
          kzvecs[kcount] = -m;
          ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
          eg[kcount][0] = 2.0*unitk[0]*k*ug[kcount];
          eg[kcount][1] = -2.0*unitk[1]*l*ug[kcount];
          eg[kcount][2] = -2.0*unitk[2]*m*ug[kcount];
          vg[kcount][0] = 1.0 + vterm*(unitk[0]*k)*(unitk[0]*k);
          vg[kcount][1] = 1.0 + vterm*(unitk[1]*l)*(unitk[1]*l);
          vg[kcount][2] = 1.0 + vterm*(unitk[2]*m)*(unitk[2]*m);
          vg[kcount][3] = -vterm*unitk[0]*k*unitk[1]*l;
          vg[kcount][4] = -vterm*unitk[0]*k*unitk[2]*m;
          vg[kcount][5] = vterm*unitk[1]*l*unitk[2]*m;
          kcount++;
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   pre-compute coefficients for each Ewald K-vector for a triclinic
   system
------------------------------------------------------------------------- */

void Ewald::coeffs_triclinic()
{
  int k,l,m;
  double sqk,vterm;

  double g_ewald_sq_inv = 1.0 / (g_ewald*g_ewald);
  double preu = 4.0*MY_PI/volume;

  double unitk_lamda[3];

  kcount = 0;

  // 1 = (k,l,m), 2 = (k,-l,m), 3 = (k,l,-m), 4 = (k,-l,-m)

  for (k = 1; k <= kxmax; k++) {
    for (l = -kymax; l <= kymax; l++) {
      for (m = -kzmax; m <= kzmax; m++) {
        unitk_lamda[0] = 2.0*MY_PI*k;
        unitk_lamda[1] = 2.0*MY_PI*l;
        unitk_lamda[2] = 2.0*MY_PI*m;
        x2lamdaT(&unitk_lamda[0],&unitk_lamda[0]);
        sqk = unitk_lamda[0]*unitk_lamda[0] + unitk_lamda[1]*unitk_lamda[1] +
          unitk_lamda[2]*unitk_lamda[2];
        if (sqk <= gsqmx) {
          kxvecs[kcount] = k;
          kyvecs[kcount] = l;
          kzvecs[kcount] = m;
          ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
          eg[kcount][0] = 2.0*unitk_lamda[0]*ug[kcount];
          eg[kcount][1] = 2.0*unitk_lamda[1]*ug[kcount];
          eg[kcount][2] = 2.0*unitk_lamda[2]*ug[kcount];
          vterm = -2.0*(1.0/sqk + 0.25*g_ewald_sq_inv);
          vg[kcount][0] = 1.0 + vterm*unitk_lamda[0]*unitk_lamda[0];
          vg[kcount][1] = 1.0 + vterm*unitk_lamda[1]*unitk_lamda[1];
          vg[kcount][2] = 1.0 + vterm*unitk_lamda[2]*unitk_lamda[2];
          vg[kcount][3] = vterm*unitk_lamda[0]*unitk_lamda[1];
          vg[kcount][4] = vterm*unitk_lamda[0]*unitk_lamda[2];
          vg[kcount][5] = vterm*unitk_lamda[1]*unitk_lamda[2];
          kcount++;
        }
      }
    }
  }

  // 1 = (0,l,m), 2 = (0,l,-m)

  for (l = 1; l <= kymax; l++) {
    for (m = -kzmax; m <= kzmax; m++) {
      unitk_lamda[0] = 0.0;
      unitk_lamda[1] = 2.0*MY_PI*l;
      unitk_lamda[2] = 2.0*MY_PI*m;
      x2lamdaT(&unitk_lamda[0],&unitk_lamda[0]);
      sqk = unitk_lamda[1]*unitk_lamda[1] + unitk_lamda[2]*unitk_lamda[2];
      if (sqk <= gsqmx) {
        kxvecs[kcount] = 0;
        kyvecs[kcount] = l;
        kzvecs[kcount] = m;
        ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
        eg[kcount][0] =  0.0;
        eg[kcount][1] =  2.0*unitk_lamda[1]*ug[kcount];
        eg[kcount][2] =  2.0*unitk_lamda[2]*ug[kcount];
        vterm = -2.0*(1.0/sqk + 0.25*g_ewald_sq_inv);
        vg[kcount][0] = 1.0;
        vg[kcount][1] = 1.0 + vterm*unitk_lamda[1]*unitk_lamda[1];
        vg[kcount][2] = 1.0 + vterm*unitk_lamda[2]*unitk_lamda[2];
        vg[kcount][3] = 0.0;
        vg[kcount][4] = 0.0;
        vg[kcount][5] = vterm*unitk_lamda[1]*unitk_lamda[2];
        kcount++;
      }
    }
  }

  // (0,0,m)

  for (m = 1; m <= kmax; m++) {
    unitk_lamda[0] = 0.0;
    unitk_lamda[1] = 0.0;
    unitk_lamda[2] = 2.0*MY_PI*m;
    x2lamdaT(&unitk_lamda[0],&unitk_lamda[0]);
    sqk = unitk_lamda[2]*unitk_lamda[2];
    if (sqk <= gsqmx) {
      kxvecs[kcount] = 0;
      kyvecs[kcount] = 0;
      kzvecs[kcount] = m;
      ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
      eg[kcount][0] = 0.0;
      eg[kcount][1] = 0.0;
      eg[kcount][2] = 2.0*unitk_lamda[2]*ug[kcount];
      vterm = -2.0*(1.0/sqk + 0.25*g_ewald_sq_inv);
      vg[kcount][0] = 1.0;
      vg[kcount][1] = 1.0;
      vg[kcount][2] = 1.0 + vterm*unitk_lamda[2]*unitk_lamda[2];
      vg[kcount][3] = 0.0;
      vg[kcount][4] = 0.0;
      vg[kcount][5] = 0.0;
      kcount++;
    }
  }
}

/* ----------------------------------------------------------------------
   allocate memory that depends on # of K-vectors
------------------------------------------------------------------------- */

void Ewald::allocate()
{
  imageflag=force->kspace->imageflag;
  order_image_long = force->kspace->order_image_long;
  kxvecs = new int[kmax3d];
  kyvecs = new int[kmax3d];
  kzvecs = new int[kmax3d];

  ug = new double[kmax3d];
  memory->create(eg,kmax3d,3,"ewald:eg");
  memory->create(vg,kmax3d,6,"ewald:vg");

  sfacrl = new double[kmax3d];
  sfacim = new double[kmax3d];
  sfacrl_all = new double[kmax3d];
  sfacim_all = new double[kmax3d];
  if (imageflag==1) {
    image_sfacrl=new double [kmax3d];
    image_sfacim=new double [kmax3d];
    image_sfacrl_all=new double[kmax3d];
    image_sfacim_all=new double [kmax3d];//for image charges
  double *pre_up=new double[2*order_image_long];//image charge prefactor
  double *pre_down=new double[2*order_image_long];//image charge prefactor
	A_rl=new double [kmax3d];
	A_im=new double [kmax3d];
	B_rl=new double [kmax3d];
	B_im=new double [kmax3d];
  }
}

/* ----------------------------------------------------------------------
   deallocate memory that depends on # of K-vectors
------------------------------------------------------------------------- */

void Ewald::deallocate()
{
  delete [] kxvecs;
  delete [] kyvecs;
  delete [] kzvecs;

  delete [] ug;
  memory->destroy(eg);
  memory->destroy(vg);

  delete [] sfacrl;
  delete [] sfacim;
  delete [] sfacrl_all;
  delete [] sfacim_all;
  delete [] image_sfacim;
  delete [] image_sfacrl;
  delete [] image_sfacim_all;
  delete [] image_sfacrl_all;
  delete [] A_rl;
  delete [] A_im;
  delete [] B_rl;
  delete [] B_im;
  delete [] pre_up;
  delete [] pre_down;
}

/* ----------------------------------------------------------------------
   Slab-geometry correction term to dampen inter-slab interactions between
   periodically repeating slabs.  Yields good approximation to 2D Ewald if
   adequate empty space is left between repeating slabs (J. Chem. Phys.
   111, 3155).  Slabs defined here to be parallel to the xy plane. Also
   extended to non-neutral systems (J. Chem. Phys. 131, 094107).
------------------------------------------------------------------------- */

void Ewald::slabcorr()
{
  // compute local contribution to global dipole moment

  double *q = atom->q;
  double **x = atom->x;
  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;
  double zprd_slab = zprd*slab_volfactor;
  int nlocal = atom->nlocal;
  int w=0;
  double energy_temp1,energy_temp2,energy_temp,surface_field;

  double dipole = 0.0;
  for (int i = 0; i < nlocal; i++) dipole += q[i]*x[i][2];

  // sum local contributions to get global dipole moment

  double dipole_all;
  MPI_Allreduce(&dipole,&dipole_all,1,MPI_DOUBLE,MPI_SUM,world);

  // need to make non-neutral systems and/or
  //  per-atom energy translationally invariant

  double dipole_r2 = 0.0;
  if (eflag_atom || fabs(qsum) > SMALL) {
    for (int i = 0; i < nlocal; i++)
      dipole_r2 += q[i]*x[i][2]*x[i][2];

    // sum local contributions

    double tmp;
    MPI_Allreduce(&dipole_r2,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
    dipole_r2 = tmp;
  }

  // compute corrections
  double series_1, series_2,series_3,series_4,Ec1,Ec2;
  int c=0;
  dielectric_jump1=force->kspace->dielectric_jump1;//dielectric interface
  dielectric_jump2=force->kspace->dielectric_jump2;//dielectric interface
  order_image_long=force->kspace->order_image_long;
  charge_up=force->kspace->charge_up;
  charge_down=force->kspace->charge_down;
  imageflag=force->kspace->imageflag;
  surface_energy_flag=force->kspace->surface_energy_flag;
  order_image_long=force->kspace->order_image_long;
  surface_chargeflag=force->kspace->surface_chargeflag;
  dielectric_interface_L=force->kspace->dielectric_interface_L;
  order_image_short=force->kspace->order_image_short;
  series_1=0;
  series_2=0;
  series_3=0;
  series_4=0;
  if(surface_chargeflag==1){
  energy_temp1=0;
  energy_temp2=0;
  for(int e=1;e<=order_image_long;e++){
  energy_temp1=energy_temp1+pow(dielectric_jump1,ceil(e*0.5))*pow(dielectric_jump2,floor(e*0.5))
  +pow(dielectric_jump2,ceil(e*0.5))*pow(dielectric_jump1,floor(e*0.5));
  energy_temp2=energy_temp2
  +pow(dielectric_jump1,floor(e*0.5))*pow(dielectric_jump2,ceil(e*0.5))*(-pow(-1,e)*0.5*dielectric_interface_L-e*dielectric_interface_L)*charge_down
  +pow(dielectric_jump1,floor(e*0.5))*pow(dielectric_jump2,ceil(e*0.5))*(pow(-1,e)*0.5*dielectric_interface_L-e*dielectric_interface_L)*charge_up
  -pow(dielectric_jump1,ceil(e*0.5))*pow(dielectric_jump2,floor(e*0.5))*(-pow(-1,e)*0.5*dielectric_interface_L+e*dielectric_interface_L)*charge_down
  -pow(dielectric_jump1,ceil(e*0.5))*pow(dielectric_jump2,floor(e*0.5))*(pow(-1,e)*0.5*dielectric_interface_L+e*dielectric_interface_L)*charge_up;
  }
  surface_field=MY_2PI*((charge_down-charge_up)+(charge_down+charge_up)*
    (dielectric_jump2-dielectric_jump1)*(1-pow(dielectric_jump1*dielectric_jump2,ceil(order_image_long*0.5)))/(1-dielectric_jump2*dielectric_jump1));
  if(abs(abs(dielectric_jump1)-1)<SMALL&&abs(abs(dielectric_jump2)-1)<SMALL&&dielectric_jump1*dielectric_jump2>0)
    surface_field=MY_2PI*(charge_down-charge_up);
  }
  if(imageflag==1){
    for(c=0;c<order_image_long;c++){
    if(c%2==0) {
      series_1=series_1-pre_up[c]-pre_down[c];
      if(fabs(qsum) > SMALL){
        series_3=series_3+2.0*(c+1)*(pre_up[c]-pre_down[c]);     
    }
    }
    else {
      series_1=series_1+pre_up[c]+pre_down[c]; 
    }  
  }
  if(fabs(qsum) > SMALL){
  for(c=0;c<order_image_long;c++){
    series_2=series_2+pre_up[c]+pre_down[c]; 
    series_4=series_4+(pre_up[c]+pre_down[c])*(c+1)*(c+1);
  }
}
  }
  const double e_slabcorr = MY_2PI*(dipole_all*dipole_all*(1+series_1) -
    qsum*dipole_r2*(1+series_2) - 0.5*qsum*qsum*series_4*dielectric_interface_L
    *dielectric_interface_L + series_3*dielectric_interface_L*qsum*dipole_all
    )/volume;
  const double qscale = qqrd2e * scale;
  
  if (eflag_global) {
    energy += qscale * e_slabcorr;
    if(surface_energy_flag==1)
      {
        Ec1=(1+energy_temp1)*0.5*(-MY_2PI*0.5/(volume*g_ewald*g_ewald))*qsum*qsum;
        Ec2=(1+energy_temp1)*(charge_down+charge_up)*xprd*yprd*MY_2PI*zprd_slab*zprd_slab*qsum/(6*volume)
        -0.5*MY_2PI*dielectric_interface_L*(charge_down+charge_up)*qsum
        +MY_2PI*qsum*energy_temp2;
        energy=energy+qscale*Ec1+qscale*Ec2+qscale*surface_field*(-1.0)*dipole_all;
      }
  }

  // per-atom energy
  double distance_1,distance_2;
  if (eflag_atom) {
    double efact = qscale * MY_2PI/volume;
    for (int i = 0; i < nlocal; i++){
      eatom[i] += efact * q[i]*(x[i][2]*dipole_all*(1+series_1) - 0.5*(dipole_r2 +
        qsum*x[i][2]*x[i][2])*(1+series_2)-0.5*qsum*series_4*dielectric_interface_L
    *dielectric_interface_L+series_3*dielectric_interface_L*0.5*(dipole_all+qsum*x[i][2]));
    if(surface_energy_flag==1) eatom[i] += q[i]*(qscale *surface_field*(-1.0)*x[i][2]
      +qscale*energy_temp*(-1.0));
    //self image
    if(imageflag==1){
      for(w=0;w<order_image_short;w++){
        distance_1=fabs(x[i][2]+pow(-1,w)*x[i][2]-(w+1)*dielectric_interface_L);
        distance_2=fabs(x[i][2]+pow(-1,w)*x[i][2]+(w+1)*dielectric_interface_L);
        eatom[i] +=0.5*q[i]*q[i]*qscale*pre_up[w]*erfc(g_ewald*distance_1)/distance_1; 
        eatom[i] +=0.5*q[i]*q[i]*qscale*pre_down[w]*erfc(g_ewald*distance_2)/distance_2;
      }
    }
  }
  }

  // add on force corrections

  double ffact = qscale * (-4.0*MY_PI/volume);
  double **f = atom->f;
if(imageflag==1){
  for (int i = 0; i < nlocal; i++) {
    f[i][2] += ffact * q[i]*(dipole_all*(1+series_1) - 
      qsum*x[i][2]*(1+series_2) + series_3*dielectric_interface_L*qsum*0.5);
  if(surface_chargeflag==1) f[i][2] +=q[i]*surface_field*qscale;
  }
} else{
  for (int i = 0; i < nlocal; i++) f[i][2] += ffact * q[i]*(dipole_all - qsum*x[i][2]);
}
 
}
/* ----------------------------------------------------------------------
   memory usage of local arrays
------------------------------------------------------------------------- */

double Ewald::memory_usage()
{
  double bytes = 3 * kmax3d * sizeof(int);
  bytes += (1 + 3 + 6) * kmax3d * sizeof(double);
  bytes += 4 * kmax3d * sizeof(double);
  bytes += nmax*3 * sizeof(double);
  bytes += 2 * (2*kmax+1)*3*nmax * sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
   group-group interactions
 ------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   compute the Ewald total long-range force and energy for groups A and B
 ------------------------------------------------------------------------- */

void Ewald::compute_group_group(int groupbit_A, int groupbit_B, int AA_flag)
{
  if (slabflag && triclinic)
    error->all(FLERR,"Cannot (yet) use K-space slab "
               "correction with compute group/group for triclinic systems");

  int i,k;

  if (!group_allocate_flag) {
    allocate_groups();
    group_allocate_flag = 1;
  }

  e2group = 0.0; //energy
  f2group[0] = 0.0; //force in x-direction
  f2group[1] = 0.0; //force in y-direction
  f2group[2] = 0.0; //force in z-direction

  // partial and total structure factors for groups A and B

  for (k = 0; k < kcount; k++) {

    // group A

    sfacrl_A[k] = 0.0;
    sfacim_A[k] = 0.0;
    sfacrl_A_all[k] = 0.0;
    sfacim_A_all[k] = 0;

    // group B

    sfacrl_B[k] = 0.0;
    sfacim_B[k] = 0.0;
    sfacrl_B_all[k] = 0.0;
    sfacim_B_all[k] = 0.0;
  }

  double *q = atom->q;
  int nlocal = atom->nlocal;
  int *mask = atom->mask;

  int kx,ky,kz;
  double cypz,sypz,exprl,expim;

  // partial structure factors for groups A and B on each processor

  for (k = 0; k < kcount; k++) {
    kx = kxvecs[k];
    ky = kyvecs[k];
    kz = kzvecs[k];

    for (i = 0; i < nlocal; i++) {

      if (!((mask[i] & groupbit_A) && (mask[i] & groupbit_B)))
        if (AA_flag) continue;

      if ((mask[i] & groupbit_A) || (mask[i] & groupbit_B)) {

        cypz = cs[ky][1][i]*cs[kz][2][i] - sn[ky][1][i]*sn[kz][2][i];
        sypz = sn[ky][1][i]*cs[kz][2][i] + cs[ky][1][i]*sn[kz][2][i];
        exprl = cs[kx][0][i]*cypz - sn[kx][0][i]*sypz;
        expim = sn[kx][0][i]*cypz + cs[kx][0][i]*sypz;

        // group A

        if (mask[i] & groupbit_A) {
          sfacrl_A[k] += q[i]*exprl;
          sfacim_A[k] += q[i]*expim;
        }

        // group B

        if (mask[i] & groupbit_B) {
          sfacrl_B[k] += q[i]*exprl;
          sfacim_B[k] += q[i]*expim;
        }
      }
    }
  }

  // total structure factor by summing over procs

  MPI_Allreduce(sfacrl_A,sfacrl_A_all,kcount,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(sfacim_A,sfacim_A_all,kcount,MPI_DOUBLE,MPI_SUM,world);

  MPI_Allreduce(sfacrl_B,sfacrl_B_all,kcount,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(sfacim_B,sfacim_B_all,kcount,MPI_DOUBLE,MPI_SUM,world);

  const double qscale = qqrd2e * scale;
  double partial_group;

  // total group A <--> group B energy
  // self and boundary correction terms are in compute_group_group.cpp

  for (k = 0; k < kcount; k++) {
    partial_group = sfacrl_A_all[k]*sfacrl_B_all[k] +
      sfacim_A_all[k]*sfacim_B_all[k];
    e2group += ug[k]*partial_group;
  }

  e2group *= qscale;

  // total group A <--> group B force

  for (k = 0; k < kcount; k++) {
    partial_group = sfacim_A_all[k]*sfacrl_B_all[k] -
      sfacrl_A_all[k]*sfacim_B_all[k];
    f2group[0] += eg[k][0]*partial_group;
    f2group[1] += eg[k][1]*partial_group;
    if (slabflag != 2) f2group[2] += eg[k][2]*partial_group;
  }

  f2group[0] *= qscale;
  f2group[1] *= qscale;
  f2group[2] *= qscale;

  // 2d slab correction

  if (slabflag == 1)
    slabcorr_groups(groupbit_A, groupbit_B, AA_flag);
}

/* ----------------------------------------------------------------------
   Slab-geometry correction term to dampen inter-slab interactions between
   periodically repeating slabs.  Yields good approximation to 2D Ewald if
   adequate empty space is left between repeating slabs (J. Chem. Phys.
   111, 3155).  Slabs defined here to be parallel to the xy plane. Also
   extended to non-neutral systems (J. Chem. Phys. 131, 094107).
------------------------------------------------------------------------- */

void Ewald::slabcorr_groups(int groupbit_A, int groupbit_B, int AA_flag)
{
  // compute local contribution to global dipole moment

  double *q = atom->q;
  double **x = atom->x;
  double zprd = domain->zprd;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  double qsum_A = 0.0;
  double qsum_B = 0.0;
  double dipole_A = 0.0;
  double dipole_B = 0.0;
  double dipole_r2_A = 0.0;
  double dipole_r2_B = 0.0;

  for (int i = 0; i < nlocal; i++) {
    if (!((mask[i] & groupbit_A) && (mask[i] & groupbit_B)))
      if (AA_flag) continue;

    if (mask[i] & groupbit_A) {
      qsum_A += q[i];
      dipole_A += q[i]*x[i][2];
      dipole_r2_A += q[i]*x[i][2]*x[i][2];
    }

    if (mask[i] & groupbit_B) {
      qsum_B += q[i];
      dipole_B += q[i]*x[i][2];
      dipole_r2_B += q[i]*x[i][2]*x[i][2];
    }
  }

  // sum local contributions to get total charge and global dipole moment
  //  for each group

  double tmp;
  MPI_Allreduce(&qsum_A,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  qsum_A = tmp;

  MPI_Allreduce(&qsum_B,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  qsum_B = tmp;

  MPI_Allreduce(&dipole_A,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  dipole_A = tmp;

  MPI_Allreduce(&dipole_B,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  dipole_B = tmp;

  MPI_Allreduce(&dipole_r2_A,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  dipole_r2_A = tmp;

  MPI_Allreduce(&dipole_r2_B,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  dipole_r2_B = tmp;

  // compute corrections

  const double qscale = qqrd2e * scale;
  const double efact = qscale * MY_2PI/volume;

  e2group += efact * (dipole_A*dipole_B - 0.5*(qsum_A*dipole_r2_B +
    qsum_B*dipole_r2_A) - qsum_A*qsum_B*zprd*zprd/12.0);

  // add on force corrections

  const double ffact = qscale * (-4.0*MY_PI/volume);
  f2group[2] += ffact * (qsum_A*dipole_B - qsum_B*dipole_A);
}

/* ----------------------------------------------------------------------
   allocate group-group memory that depends on # of K-vectors
------------------------------------------------------------------------- */

void Ewald::allocate_groups()
{
  // group A

  sfacrl_A = new double[kmax3d];
  sfacim_A = new double[kmax3d];
  sfacrl_A_all = new double[kmax3d];
  sfacim_A_all = new double[kmax3d];

  // group B

  sfacrl_B = new double[kmax3d];
  sfacim_B = new double[kmax3d];
  sfacrl_B_all = new double[kmax3d];
  sfacim_B_all = new double[kmax3d];
}

/* ----------------------------------------------------------------------
   deallocate group-group memory that depends on # of K-vectors
------------------------------------------------------------------------- */

void Ewald::deallocate_groups()
{
  // group A

  delete [] sfacrl_A;
  delete [] sfacim_A;
  delete [] sfacrl_A_all;
  delete [] sfacim_A_all;

  // group B

  delete [] sfacrl_B;
  delete [] sfacim_B;
  delete [] sfacrl_B_all;
  delete [] sfacim_B_all;
}
