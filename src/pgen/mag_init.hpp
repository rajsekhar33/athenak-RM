#ifndef PGEN_MAG_INIT_HPP_
#define PGEN_MAG_INIT_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mag_init.hpp
//  \brief defines Mag Init class, which implements data and functions for
//  poloidal/toroidal/randomly seeded turbulent magnetic field

#include <iostream>
#include <memory>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "utils/random.hpp"

//----------------------------------------------------------------------------------------
//! \class MagInit

class MagInit {
 public:
  MagInit(MeshBlockPack *pp, ParameterInput *pin);
  ~MagInit();

  DvceArray5D<Real> a_vecp;  // arrays used for turb forcing
  RNG_State rstate;                    // random state

  DualArray2D<Real> aka, akb; //to store amplitude coefficients
  DualArray1D<Real> kx_mode, ky_mode, kz_mode;
  DvceArray3D<Real> xcos, xsin, ycos, ysin, zcos, zsin;

  // parameters of driving
  int mag_flag; // poloidal/toroidal/turbulent initial field
  int nlow, nhigh, spect_form;
  int mode_count;
  Real kpeak;
  Real inv_beta, expo;
  Real r_in; // sink radius
  Real amin;

  // functions
  TaskStatus InitializeAVecModes(int stage);
  TaskStatus InitMagField(int stage);
  void Initialize();

 private:
  bool first_time = true;   // flag to enable initialization on first call
  MeshBlockPack *pmy_pack;  // ptr to MeshBlockPack containing this MagInit
};



MagInit::MagInit(MeshBlockPack *pp, ParameterInput *pin) :
  a_vecp("a_vecp",1,1,1,1,1),
  aka("aka",1,1),akb("zssc",1,1),
  kx_mode("kx_mode",1),ky_mode("ky_mode",1),kz_mode("kz_mode",1),
  xcos("xcos",1,1,1),xsin("xsin",1,1,1),ycos("ycos",1,1,1),
  ysin("ysin",1,1,1),zcos("zcos",1,1,1),zsin("zsin",1,1,1),
  pmy_pack(pp) {
  // allocate memory for a_vecp registers
  int nmb = pmy_pack->nmb_thispack;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;

  Kokkos::realloc(a_vecp, nmb, 3, ncells3, ncells2, ncells1);

  Mesh *pm = pmy_pack->pmesh;
  Real lx = pm->mesh_size.x1max - pm->mesh_size.x1min;
  Real dkx = 2.0*M_PI/lx;

  // range of modes including, corresponding to kmin and kmax
  nlow = pin->GetOrAddInteger("mag_init", "nlow", 1);
  nhigh = pin->GetOrAddInteger("mag_init", "nhigh", 3);
  // Peak of power when spectral form is parabolic, in units of 2*(PI/L)
  kpeak = pin->GetOrAddReal("mag_init", "kpeak", dkx*0.5*(nlow+nhigh));
  // spect form - 1 for parabola, 2 for power-law
  spect_form = pin->GetOrAddInteger("mag_init", "spect_form", 1);
  // power-law exponent for isotropic driving
  expo = pin->GetOrAddReal("mag_init", "expo", 5.0/3.0);
  // inverse of plasma beta
  inv_beta = pin->GetOrAddReal("mag_init", "inv_beta", 0.0);
  amin = pin->GetOrAddReal("mag_init", "mag_amin", 10.0);
  // sink radius in code units to set B=0 inside
  r_in = pin->GetOrAddReal("problem", "r_in", 0.0);

  // magnetic field properties - 0 for toroidal, 1 for poloidal, turbulent otherwise
  mag_flag = pin->GetOrAddInteger("mag_init", "mag_flag", 2);
  if (global_variable::my_rank == 0) {
    std::cout << "Initializing non-uniform Magnetic fields module" << std::endl;
  }
  Real nlow_sqr = nlow*nlow;
  Real nhigh_sqr = nhigh*nhigh;

  mode_count = 0;

  int nkx, nky, nkz;
  Real nsqr;
  if(mag_flag>1) {
    for (nkx = 0; nkx <= nhigh; nkx++) {
      for (nky = 0; nky <= nhigh; nky++) {
        for (nkz = 0; nkz <= nhigh; nkz++) {
          if (nkx == 0 && nky == 0 && nkz == 0) continue;
          nsqr = SQR(nkx) + SQR(nky) + SQR(nkz);
          if (nsqr >= nlow_sqr && nsqr <= nhigh_sqr) {
            mode_count++;
          }
        }
      }
    }

    Kokkos::realloc(aka, 3, mode_count); // Amplitude of real component
    Kokkos::realloc(akb, 3, mode_count); // Amplitude of imaginary component

    Kokkos::realloc(kx_mode, mode_count);
    Kokkos::realloc(ky_mode, mode_count);
    Kokkos::realloc(kz_mode, mode_count);

    Kokkos::realloc(xcos, nmb, mode_count, ncells1);
    Kokkos::realloc(xsin, nmb, mode_count, ncells1);
    Kokkos::realloc(ycos, nmb, mode_count, ncells2);
    Kokkos::realloc(ysin, nmb, mode_count, ncells2);
    Kokkos::realloc(zcos, nmb, mode_count, ncells3);
    Kokkos::realloc(zsin, nmb, mode_count, ncells3);

    Initialize();
  }
}

//----------------------------------------------------------------------------------------
// destructor

MagInit::~MagInit() {
}

//----------------------------------------------------------------------------------------
//! \fn  void Initialize
//  \brief Function to initialize the driver

void MagInit::Initialize() {
  Mesh *pm = pmy_pack->pmesh;
  int nmb = pmy_pack->nmb_thispack;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is;
  int js = indcs.js;
  int ks = indcs.ks;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  int &nx1 = indcs.nx1;
  int &nx2 = indcs.nx2;
  int &nx3 = indcs.nx3;

  rstate.idum = -1;

  auto kx_mode_ = kx_mode;
  auto ky_mode_ = ky_mode;
  auto kz_mode_ = kz_mode;

  auto xcos_ = xcos;
  auto xsin_ = xsin;
  auto ycos_ = ycos;
  auto ysin_ = ysin;
  auto zcos_ = zcos;
  auto zsin_ = zsin;

  Real dkx, dky, dkz, kx, ky, kz;
  Real lx = pm->mesh_size.x1max - pm->mesh_size.x1min;
  Real ly = pm->mesh_size.x2max - pm->mesh_size.x2min;
  Real lz = pm->mesh_size.x3max - pm->mesh_size.x3min;
  dkx = 2.0*M_PI/lx;
  dky = 2.0*M_PI/ly;
  dkz = 2.0*M_PI/lz;

  int nmode = 0;
  int nkx, nky, nkz;
  Real nsqr;
  Real nlow_sqr = nlow*nlow;
  Real nhigh_sqr = nhigh*nhigh;
  for (nkx = 0; nkx <= nhigh; nkx++) {
    for (nky = 0; nky <= nhigh; nky++) {
      for (nkz = 0; nkz <= nhigh; nkz++) {
        if (nkx == 0 && nky == 0 && nkz == 0) continue;
        nsqr = SQR(nkx) + SQR(nky) + SQR(nkz);
        if (nsqr >= nlow_sqr && nsqr <= nhigh_sqr) {
          kx = dkx*nkx;
          ky = dky*nky;
          kz = dkz*nkz;
          kx_mode_.h_view(nmode) = kx;
          ky_mode_.h_view(nmode) = ky;
          kz_mode_.h_view(nmode) = kz;
          nmode++;
        }
      }
    }
  }

  kx_mode_.template modify<HostMemSpace>();
  kx_mode_.template sync<DevExeSpace>();
  ky_mode_.template modify<HostMemSpace>();
  ky_mode_.template sync<DevExeSpace>();
  kz_mode_.template modify<HostMemSpace>();
  kz_mode_.template sync<DevExeSpace>();

  auto &size = pmy_pack->pmb->mb_size;

  par_for("xsin/xcos", DevExeSpace(),0,nmb-1,0,mode_count-1, 0, ncells1-1,
  KOKKOS_LAMBDA(int m, int n, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
    Real k1v = kx_mode_.d_view(n);
    xsin_(m,n,i) = sin(k1v*x1v);
    xcos_(m,n,i) = cos(k1v*x1v);
  });

  par_for("ysin/ycos", DevExeSpace(),0,nmb-1,0,mode_count-1, 0, ncells2-1,
  KOKKOS_LAMBDA(int m, int n, int j) {
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
    Real k2v = ky_mode_.d_view(n);
    ysin_(m,n,j) = sin(k2v*x2v);
    ycos_(m,n,j) = cos(k2v*x2v);
    if (ncells2-1 == 0) {
      ysin_(m,n,j) = 0.0;
      ycos_(m,n,j) = 1.0;
    }
  });

  par_for("zsin/zcos", DevExeSpace(),0,nmb-1,0,mode_count-1, 0, ncells3-1,
  KOKKOS_LAMBDA(int m, int n, int k) {
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
    Real k3v = kz_mode_.d_view(n);
    zsin_(m,n,k) = sin(k3v*x3v);
    zcos_(m,n,k) = cos(k3v*x3v);
    if (ncells3-1 == 0) {
      zsin_(m,n,k) = 0.0;
      zcos_(m,n,k) = 1.0;
    }
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void IncludeMagInitTasks
//  \brief Includes task in the operator split task list that constructs new modes with
//  random amplitudes and phases that can be used to initialize a_vecp and B-fields
//  Called by MeshBlockPack::AddPhysics() function

TaskStatus MagInit::InitializeAVecModes(int) {
  Mesh *pm = pmy_pack->pmesh;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is;
  int js = indcs.js;
  int ks = indcs.ks;
  int &nmb = pmy_pack->nmb_thispack;
  int &nx1 = indcs.nx1;
  int &nx2 = indcs.nx2;
  int &nx3 = indcs.nx3;
  int &ng  = indcs.ng;
  int ncells1 =  nx1 +   2*(ng);
  int ncells2 = (nx2 > 1)? (nx2 + 2*(ng)) : 1;
  int ncells3 = (nx3 > 1)? (nx3 + 2*(ng)) : 1;

  int nlow_sqr = SQR(nlow);
  int nhigh_sqr = SQR(nhigh);
  auto mode_count_ = mode_count;

  auto aka_ = aka;
  auto akb_ = akb;

  auto amin_ = amin;
  auto r_in_ = r_in;

  Real dkx, dky, dkz, kx, ky, kz;
  Real lx = pm->mesh_size.x1max - pm->mesh_size.x1min;
  Real ly = pm->mesh_size.x2max - pm->mesh_size.x2min;
  Real lz = pm->mesh_size.x3max - pm->mesh_size.x3min;
  dkx = 2.0*M_PI/lx;
  dky = 2.0*M_PI/ly;
  dkz = 2.0*M_PI/lz;

  Real &ex = expo;
  Real norm, kiso;
  Real khigh = nhigh*fmax(fmax(dkx,dky),dkz);
  Real klow  = nlow *fmin(fmin(dkx,dky),dkz);
  Real parab_prefact = -4.0 / pow(khigh-klow,2.0);
  Real &k_peak = kpeak;

  auto a_vecp_ = a_vecp;
  auto &size = pmy_pack->pmb->mb_size;
  int no_dir=3;
  // mag_flag == 0 : toroidal field
  if (mag_flag == 0) {
    Real rmin = amin_*r_in;
    par_for("a_toro", DevExeSpace(),0,nmb-1,0,ncells3-1,0,ncells2-1,0,ncells1-1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real x1v = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      Real x2v = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
      Real r_cyl = sqrt(SQR(x1v)+SQR(x2v));
      a_vecp_(m,0,k,j,i) = 0.0;
      a_vecp_(m,1,k,j,i) = 0.0;
      a_vecp_(m,2,k,j,i) = fmax(r_cyl-rmin,0.0);
    });
  } else if (mag_flag == 1) {
    // Poloidal field but slight radial within rmin.
    Real rmin = amin_*r_in;
    par_for("a_polo_rad", DevExeSpace(),0,nmb-1,0,ncells3-1,0,ncells2-1,0,ncells1-1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real x1v = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      Real x2v = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
      Real x3v = CellCenterX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
      Real rad = sqrt(SQR(x1v)+SQR(x2v)+SQR(x3v));
      Real r_cyl = sqrt(SQR(x1v)+SQR(x2v));
      Real a_phi = (rad+rmin)*(r_cyl/rad);
      a_vecp_(m,0,k,j,i) = -a_phi*x2v/r_cyl;
      a_vecp_(m,1,k,j,i) = a_phi*x1v/r_cyl;
      a_vecp_(m,2,k,j,i) = 0.0;
    });
  } else { // initialize turbulent Mag-vector potential to zero
    par_for("a_vec_zero",DevExeSpace(),0,nmb-1,0,2,0,ncells3-1,0,ncells2-1,0,ncells1-1,
    KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
      a_vecp_(m,n,k,j,i) = 0.0;
    });

    int &nmb = pmy_pack->nmb_thispack;

    // if (global_variable::my_rank == 0) std::cout << "force_tmp2_ zeroed." << std::endl;

    int nmode = 0;
    int nkx, nky, nkz, nsqr;

    for (nkx = 0; nkx <= nhigh; nkx++) {
      for (nky = 0; nky <= nhigh; nky++) {
        for (nkz = 0; nkz <= nhigh; nkz++) {
          if (nkx == 0 && nky == 0 && nkz == 0) continue;
          norm = 0.0;
          nsqr = 0.0;
          nsqr = SQR(nkx) + SQR(nky) + SQR(nkz);
          if (nsqr >= nlow_sqr && nsqr <= nhigh_sqr) {
            kx = dkx*nkx;
            ky = dky*nky;
            kz = dkz*nkz;
            Real k[3] = {kx, ky, kz};
            // Generate Fourier amplitudes
            kiso = sqrt(SQR(kx) + SQR(ky) + SQR(kz));
            if (kiso > 1e-16) {
              if (spect_form==2) {
                norm = 1.0/pow(kiso,(ex+2.0)/2.0); // power-law driving
              } else if (spect_form==1) {
                norm = fabs(parab_prefact*pow(kiso-k_peak,2.0)+1.0);// parabola in k-space
                norm = pow(norm,0.5) * pow(k_peak/kiso, (static_cast<int>(no_dir)-1)/2.);
              } else {
              norm = 0.0;
              }
            } else {
              norm = 0.0;
            }

            Real ka = 0.0;
            Real kb = 0.0;

            for (int dir = 0; dir < no_dir; dir ++) {
              aka_.h_view(dir,nmode) = norm*RanGaussianSt(&(rstate));
              akb_.h_view(dir,nmode) = norm*RanGaussianSt(&(rstate));

              // ka = ka + k[dir]*aka_.h_view(dir,nmode);
              // kb = kb + k[dir]*akb_.h_view(dir,nmode);
              ka = ka + k[dir]*akb_.h_view(dir,nmode);
              kb = kb + k[dir]*aka_.h_view(dir,nmode);
            }

            // Now decompose into solenoidal/compressive modes
            if(norm > 0.) {
              for (int dir = 0; dir < no_dir; dir ++) {
                Real diva = k[dir]*ka/SQR(kiso);
                Real divb = k[dir]*kb/SQR(kiso);

                Real curla = aka_.h_view(dir,nmode) - divb;
                Real curlb = akb_.h_view(dir,nmode) - diva;
                aka_.h_view(dir,nmode) = curla;
                akb_.h_view(dir,nmode) = curlb;
              }
            }

            nmode++;
          }
        }
      }
    }
    // if (global_variable::my_rank == 0) std::cout << "Sines and cosines updated on
    // host" << std::endl;

    aka_.template modify<HostMemSpace>();
    aka_.template sync<DevExeSpace>();
    akb_.template modify<HostMemSpace>();
    akb_.template sync<DevExeSpace>();

    // if (global_variable::my_rank == 0) std::cout << "Sines and cosines updated on
    // device" << std::endl;
    auto xcos_ = xcos;
    auto xsin_ = xsin;
    auto ycos_ = ycos;
    auto ysin_ = ysin;
    auto zcos_ = zcos;
    auto zsin_ = zsin;

    for (int n=0; n<mode_count_; n++) {
      par_for("a_vec_compute", DevExeSpace(),0,nmb-1,0,ncells3-1,0,ncells2-1,0,ncells1-1,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        Real forc_real =
            ( xcos_(m,n,i)*ycos_(m,n,j) - xsin_(m,n,i)*ysin_(m,n,j) ) * zcos_(m,n,k) -
            ( xsin_(m,n,i)*ycos_(m,n,j) + xcos_(m,n,i)*ysin_(m,n,j) ) * zsin_(m,n,k);
        Real forc_imag =
            ( ycos_(m,n,j)*zsin_(m,n,k) + ysin_(m,n,j)*zcos_(m,n,k) ) * xcos_(m,n,i) +
            ( ycos_(m,n,j)*zcos_(m,n,k) - ysin_(m,n,j)*zsin_(m,n,k) ) * xsin_(m,n,i);
        for (int dir = 0; dir < no_dir; dir ++){
          a_vecp_(m,dir,k,j,i) += aka_.d_view(dir,n)*forc_real -
                                   akb_.d_view(dir,n)*forc_imag;
        }
      });
    }
  } // end of else condition on mag_flag
  if (global_variable::my_rank == 0) {
    std::cout << "a_vecp_ computed." << std::endl;
    std::cout << "Now setting avec amplitude inside r_in to zero" << std::endl;
  }
  if(r_in > 0.0) {
    par_for("a_vecp_amp",DevExeSpace(),0,nmb-1,0,ncells3-1,0,ncells2-1,0,ncells1-1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real x1v = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      Real x2v = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
      Real x3v = CellCenterX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
      Real rad = sqrt(SQR(x1v)+SQR(x2v)+SQR(x3v));
      for (int dir = 0; dir < no_dir; dir ++){
        if (rad < r_in_) {
          a_vecp_(m,dir,k,j,i) = 0.0;
        } else if (rad < amin_*r_in_) {
          Real fac_smooth = SQR(sin((rad-r_in_)/r_in_/(amin_-1.0)*M_PI/2.0));
          a_vecp_(m,dir,k,j,i) *= fac_smooth;
        }
      }
    });
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn Now we initialize the magnetic field using the vector potential

TaskStatus MagInit::InitMagField(int) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie, nx1 = indcs.nx1;
  int js = indcs.js, je = indcs.je, nx2 = indcs.nx2;
  int ks = indcs.ks, ke = indcs.ke, nx3 = indcs.nx3;
  const int nmkji = (pmy_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;

  EOS_Data &eos_data = pmy_pack->pmhd->peos->eos_data;
  Real gamma = eos_data.gamma;
  Real gm1 = gamma - 1.0;

  bool is_gr = pmy_pack->pcoord->is_general_relativistic;

  if (pmy_pack->pionn == nullptr) {
    // modify conserved variables
    DvceArray5D<Real> u,w;
    if (pmy_pack->phydro != nullptr) {
      u = (pmy_pack->phydro->u0);
      w = (pmy_pack->phydro->w0);
    }
    if (pmy_pack->pmhd != nullptr) {
      u = (pmy_pack->pmhd->u0);
      w = (pmy_pack->pmhd->w0);
    }

    auto a_vecp_ = a_vecp;

    auto &b0_ = pmy_pack->pmhd->b0;
    auto &size = pmy_pack->pmb->mb_size;
    par_for("turb-b", DevExeSpace(), 0,(pmy_pack->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real dx1 = size.d_view(m).dx1;
      Real dx2 = size.d_view(m).dx2;
      Real dx3 = size.d_view(m).dx3;
      b0_.x1f(m,k,j,i)      = ((a_vecp_(m,2,k,j+1,i-1)-a_vecp_(m,2,k,j-1,i-1))
                              +(a_vecp_(m,2,k,j+1,i  )-a_vecp_(m,2,k,j-1,i  )))/dx2
                             -((a_vecp_(m,1,k+1,j,i-1)-a_vecp_(m,1,k-1,j,i-1))
                              +(a_vecp_(m,1,k+1,j,i  )-a_vecp_(m,1,k-1,j,i  )))/dx3;
      b0_.x2f(m,k,j,i)      = ((a_vecp_(m,0,k+1,j-1,i)-a_vecp_(m,0,k-1,j-1,i))
                              +(a_vecp_(m,0,k+1,j  ,i)-a_vecp_(m,0,k-1,j  ,i)))/dx3
                             -((a_vecp_(m,2,k,j-1,i+1)-a_vecp_(m,2,k,j-1,i-1))
                              +(a_vecp_(m,2,k,j  ,i+1)-a_vecp_(m,2,k,j  ,i-1)))/dx1;
      b0_.x3f(m,k,j,i)      = ((a_vecp_(m,1,k-1,j,i+1)-a_vecp_(m,1,k-1,j,i-1))
                              +(a_vecp_(m,1,k  ,j,i+1)-a_vecp_(m,1,k  ,j,i-1)))/dx1
                             -((a_vecp_(m,0,k-1,j+1,i)-a_vecp_(m,0,k-1,j-1,i))
                              +(a_vecp_(m,0,k  ,j+1,i)-a_vecp_(m,0,k  ,j-1,i)))/dx2;
      if (i==ie) {
        b0_.x1f(m,k,j,ie+1) = ((a_vecp_(m,2,k,j+1,i  )-a_vecp_(m,2,k,j-1,i  ))
                              +(a_vecp_(m,2,k,j+1,i+1)-a_vecp_(m,2,k,j-1,i+1)))/dx2
                             -((a_vecp_(m,1,k+1,j,i  )-a_vecp_(m,1,k-1,j,i  ))
                              +(a_vecp_(m,1,k+1,j,i+1)-a_vecp_(m,1,k-1,j,i+1)))/dx3;
      }
      if (j==je) {
        b0_.x2f(m,k,je+1,i) = ((a_vecp_(m,0,k+1,j  ,i)-a_vecp_(m,0,k-1,j  ,i))
                              +(a_vecp_(m,0,k+1,j+1,i)-a_vecp_(m,0,k-1,j+1,i)))/dx3
                             -((a_vecp_(m,2,k,j  ,i+1)-a_vecp_(m,2,k,j  ,i-1))
                              +(a_vecp_(m,2,k,j+1,i+1)-a_vecp_(m,2,k,j+1,i-1)))/dx1;
      }
      if (k==ke) {
        b0_.x3f(m,ke+1,j,i) = ((a_vecp_(m,1,k  ,j,i+1)-a_vecp_(m,1,k  ,j,i-1))
                              +(a_vecp_(m,1,k+1,j,i+1)-a_vecp_(m,1,k+1,j,i-1)))/dx1
                             -((a_vecp_(m,0,k  ,j+1,i)-a_vecp_(m,0,k  ,j-1,i))
                              +(a_vecp_(m,0,k+1,j+1,i)-a_vecp_(m,0,k+1,j-1,i)))/dx2;
      }
    });
    // Now deallocate a_vecp_
    // a_vecp_ = Kokkos::View<Real *****, LayoutWrapper, DevMemSpace>();

    Real m0 = 0.0, m1 = 0.0;
    Kokkos::parallel_reduce("b-sum", Kokkos::RangePolicy<>(DevExeSpace(),0,nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &sum_m0, Real &sum_m1) {
      // compute n,k,j,i indices of thread
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;
      Real dx1 = size.d_view(m).dx1;
      Real dx2 = size.d_view(m).dx2;
      Real dx3 = size.d_view(m).dx3;
      Real dvol = dx1*dx2*dx3;
      Real eint = w(m,IEN,k,j,i);
      sum_m0 += dvol*gm1*eint;
      sum_m1 += dvol*0.125*(SQR(b0_.x1f(m,k,j,i)+b0_.x1f(m,k,j,i+1))
                           +SQR(b0_.x2f(m,k,j,i)+b0_.x2f(m,k,j+1,i))
                           +SQR(b0_.x3f(m,k,j,i)+b0_.x3f(m,k+1,j,i)));
    }, Kokkos::Sum<Real>(m0), Kokkos::Sum<Real>(m1));
#if MPI_PARALLEL_ENABLED
    Real m_sum2[2] = {m0,m1};
    Real gm_sum2[2];
    MPI_Allreduce(m_sum2, gm_sum2, 2, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
    m0 = gm_sum2[0];
    m1 = gm_sum2[1];
#endif
    Real norm = sqrt(inv_beta*m0/m1);
    par_for("b-norm", DevExeSpace(), 0,(pmy_pack->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      b0_.x1f(m,k,j,i) *= norm;
      b0_.x2f(m,k,j,i) *= norm;
      b0_.x3f(m,k,j,i) *= norm;
      if (i==ie) {
        b0_.x1f(m,k,j,ie+1) *= norm;
      }
      if (j==je) {
        b0_.x2f(m,k,je+1,i) *= norm;
      }
      if (k==ke) {
        b0_.x3f(m,ke+1,j,i) *= norm;
      }
    });
    if (!is_gr) {
      par_for("turb-be", DevExeSpace(), 0,(pmy_pack->nmb_thispack-1),ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        u(m,IEN,k,j,i)+=0.125*(SQR(b0_.x1f(m,k,j,i)+b0_.x1f(m,k,j,i+1))
                              +SQR(b0_.x2f(m,k,j,i)+b0_.x2f(m,k,j+1,i))
                              +SQR(b0_.x3f(m,k,j,i)+b0_.x3f(m,k+1,j,i)));
      });
    } else { // GR
      par_for("turb-be", DevExeSpace(), 0,(pmy_pack->nmb_thispack-1),ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        u(m,IEN,k,j,i)-=0.125*(SQR(b0_.x1f(m,k,j,i)+b0_.x1f(m,k,j,i+1))
                              +SQR(b0_.x2f(m,k,j,i)+b0_.x2f(m,k,j+1,i))
                              +SQR(b0_.x3f(m,k,j,i)+b0_.x3f(m,k+1,j,i)));
      });
    }
  }

  return TaskStatus::complete;
}


#endif  // PGEN_MAG_INIT_HPP_
