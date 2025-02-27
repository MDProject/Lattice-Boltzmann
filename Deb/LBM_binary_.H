#ifndef LBM_H_
#define LBM_H_

#include "LBM_d3q19.H"
#include <iostream>
#include <random>
#include <cmath>


AMREX_GPU_MANAGED Real tau_f = 2.; // tau=0.5*(1.+1./sqrt(3.)) minimizes spurious currents according to Swift et al.
AMREX_GPU_MANAGED Real tau_g = tau_f;
AMREX_GPU_MANAGED Real alpha0 = 3.;
AMREX_GPU_MANAGED Real alpha1 = 0.;//alpha1;
const Real epsilon = 1e-10;

AMREX_GPU_MANAGED Real kappa = 0.02;

AMREX_GPU_MANAGED Real T = 0.5;//

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real draw_from_pdf_normal(const double& mean, const double& variance){
    // random seed 
    static std::random_device rd;
    // Mersenne twister PRNG, seeded with rd
    static std::mt19937 gen(rd());
    std::normal_distribution<double> d(mean, std::sqrt(variance));
    return d(gen);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real pressure(Real rho, Real phi) {

  Real p0 = rho*T;

  return p0;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
inline void thermal_noise(MultiFab& fnoisevs, MultiFab& gnoisevs, MultiFab& hydrovsbar){
  auto const & fnoise = fnoisevs.arrays();
  auto const & gnoise = gnoisevs.arrays();
  auto const & hydrobar = hydrovsbar.arrays();
  const Real tau_f_bar = 1./(tau_f+0.5);
  const Real tau_g_bar = tau_f_bar;
  const Real tau_f_bar2 = tau_f_bar*tau_f_bar;
  const Real tau_g_bar2 = tau_g_bar*tau_g_bar;
  ParallelFor(fnoisevs, IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) { // IntVect(0): no ghost cells when loop over the grid sites;
    Real rho = hydrobar[nbx](x, y, z, 0);
    Real phi = hydrobar[nbx](x, y, z, 1);
    /*if(rho<0){
        Print() << "density rho < 0: " << rho << '\n';
    }
    if(phi<0){
        Print() << "density phi < 0: " << phi << '\n';
    }*/
    //Print() << rho << "++++" << phi << '\n';
    Real rhot = rho + phi; 
    if(rhot<epsilon){
      Print() << "RHO TOTAL" << rho << "----" << phi << '\n';
    }
    fnoise[nbx](x,y,z,0) = 0.;
    gnoise[nbx](x,y,z,0) = 0.;
    for (int a=1; a<=AMREX_SPACEDIM; a++) { // ?????
      fnoise[nbx](x,y,z,a) = sqrt(2.*(tau_f_bar - tau_f_bar2)*kBT*std::abs(rho*phi)/rhot)*draw_from_pdf_normal(0., 1.);
      //Print() << "noize term: " << "-(" << x << "," << y << ',' << z << "," << nbx << ")-"<< fnoise[nbx](x,y,z,a) << '\n';
      gnoise[nbx](x,y,z,a) = -fnoise[nbx](x,y,z,a);
    }
    for (int a=AMREX_SPACEDIM+1; a<nvel; a++) {
      fnoise[nbx](x,y,z,a) = sqrt(2.*(tau_f_bar - tau_f_bar2)*kBT/cs2*b[a]*std::abs(rho))*draw_from_pdf_normal(0., 1.);
      gnoise[nbx](x,y,z,a) = sqrt(2.*(tau_g_bar - tau_g_bar2)*kBT/cs2*b[a]*std::abs(phi))*draw_from_pdf_normal(0., 1.);
      //Print() << "noize term: " << gnoise[nbx](x,y,z,a) << '\n';
    }
  });
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RealVect gradient(int x, int y, int z, const Array4<Real>& field, int icomp) {
  RealVect gradient = { 0.0, 0.0, 0.0 };
  for (int i=0; i<nvel; i++) {
    int xp = x + c[i][0];
    int yp = y + c[i][1];
    int zp = z + c[i][2];
    for (int dir=0; dir<3; dir++) {
      gradient[dir] += w[i]/cs2*field(xp,yp,zp,icomp)*c[i][dir]; // icomp is the input field component index we hope to calculate
    }
    /*
    For D3Q19, \nabla_\alpha [field] = 1/cs^2*\sum_i *w_i n(r+e_i\Delta t)e_{i\alpha}
    */
  }
  return gradient;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real laplacian(int x, int y, int z, const Array4<Real>& field, int icomp) {
  Real laplacian = 0.0;
  for (int i=0; i<nvel; i++) {
    int xp = x + c[i][0];
    int yp = y + c[i][1];
    int zp = z + c[i][2];
    laplacian += 2.*w[i]/cs2*(field(xp,yp,zp,icomp)-field(x,y,z,icomp));
  }
  /*
    \nabla^2 [field] = \sum_i w_i*(n(r+e_i\Delta t) - n(r))/(0.5*cs^2*(\Delta t)^2)
  */
  return laplacian;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RealVect grad_laplacian_2nd(int x, int y, int z, const Array4<Real>& field, int icomp) {
  RealVect gradient = { 0.0, 0.0, 0.0 };
  for(int dir=0; dir<AMREX_SPACEDIM; dir++){
    for (int i=0; i<nvel; i++) {
      for(int j=0; j<nvel; j++){
        int xc = x + c[j][0];
        int yc = y + c[j][1];
        int zc = z + c[j][2];

        int xp = xc + c[i][0];
        int yp = yc + c[i][1];
        int zp = zc + c[i][2];
        gradient[dir] += 2.*w[i]*w[j]/cs4*(field(xp,yp,zp,icomp)-field(xc,yc,zc,icomp))*c[j][dir];
      }
    }
  }
  /*  2nd order accuracy scheme of laplacian is used;
    \partial_\alpha \nabla^2 [field] = \sum_i w_i*w_j*(n(r+e_j\Delta t+e_i\Delta t) - n(r+e_j\Delta t))*c[j][\alpha]/(0.5*cs^4*(\Delta t)^3)
  */
  return gradient;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void hydrovars_bar(int x, int y, int z, // modified density & momentum calculated directly by \overline{f,g}, the original quantities are contained in function [hydrovars]
	       const Array4<Real>& f,
	       const Array4<Real>& g,
	       const Array4<Real>& hb) {
  Real rho, phi, jf[3], jg[3];
  rho = phi = jf[0] = jf[1] = jf[2] = jg[0] = jg[1] = jg[2] = 0.0;
  for (int i=0; i<nvel; ++i) {
    Real fi, gi;
    fi = f(x,y,z,i);
    gi = g(x,y,z,i);
    rho  += fi;
    phi  += gi;
    jf[0] += fi*c[i][0];
    jf[1] += fi*c[i][1];
    jf[2] += fi*c[i][2];
    jg[0] += gi*c[i][0];
    jg[1] += gi*c[i][1];
    jg[2] += gi*c[i][2];
  }
  hb(x,y,z,0) = rho;
  hb(x,y,z,1) = phi;
  if(rho>epsilon&&phi>epsilon){
    hb(x,y,z,2) = jf[0]/rho;
    hb(x,y,z,3) = jf[1]/rho;
    hb(x,y,z,4) = jf[2]/rho;
    hb(x,y,z,5) = rho + phi;//pressure(rho,phi);
    hb(x,y,z,6) = jg[0]/phi;
    hb(x,y,z,7) = jg[1]/phi;
    hb(x,y,z,8) = jg[2]/phi;
  }else{
    hb(x,y,z,2) = 0.;
    hb(x,y,z,3) = 0.;
    hb(x,y,z,4) = 0.;
    hb(x,y,z,5) = rho + phi;//pressure(rho,phi);
    hb(x,y,z,6) = 0.;
    hb(x,y,z,7) = 0.;
    hb(x,y,z,8) = 0.;
  }

  // preparation for the acceleration of fluid; we should know that the acceleration field is same with the modified version;
  const Array1D<RealVect,0,2> grad_laplacians = {
    grad_laplacian_2nd(x,y,z,hb,0),
    grad_laplacian_2nd(x,y,z,hb,1),
  };
  const Array1D<RealVect,0,2> grad = {
    gradient(x,y,z,hb,0),
    gradient(x,y,z,hb,1),
  };

  // acceleration of fluid f;
  const RealVect grad_phi = grad(1);
  const RealVect grad_laplacian_phi = grad_laplacians(1);
  hb(x,y,z,9) = -cs2*(alpha0*grad_phi[0] + alpha1*grad_laplacian_phi[0]);
  hb(x,y,z,10) = -cs2*(alpha0*grad_phi[1] + alpha1*grad_laplacian_phi[1]);
  hb(x,y,z,11) = -cs2*(alpha0*grad_phi[2] + alpha1*grad_laplacian_phi[2]);
  // acceleration of fluid g;
  const RealVect grad_rho = grad(0);
  const RealVect grad_laplacian_rho = grad_laplacians(0);
  hb(x,y,z,12) = -cs2*(alpha0*grad_rho[0] + alpha1*grad_laplacian_rho[0]);
  hb(x,y,z,13) = -cs2*(alpha0*grad_rho[1] + alpha1*grad_laplacian_rho[1]);
  hb(x,y,z,14) = -cs2*(alpha0*grad_rho[2] + alpha1*grad_laplacian_rho[2]);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void hydrovars(int x, int y, int z, // real density & momentum;
	       const Array4<Real>& f,
	       const Array4<Real>& g,
	       const Array4<Real>& h,
         const Array4<Real>& hbar, // here modified density & momentum in [h] are passed in 
         const Array4<Real>& nf,
         const Array4<Real>& ng) {

  // hydrovars_bar(x, y, z, f, g, h);
  Array1D<Real,0,AMREX_SPACEDIM> ufbar; // ONLY WORKS for 3D;
  Array1D<Real,0,AMREX_SPACEDIM> ugbar;
  Array1D<Real,0,AMREX_SPACEDIM> afbar;
  Array1D<Real,0,AMREX_SPACEDIM> agbar;
  for(int k=0; k<AMREX_SPACEDIM; k++){
    ufbar(k) = hbar(x,y,z,k+2);
    ugbar(k) = hbar(x,y,z,k+6);
    afbar(k) = hbar(x,y,z,k+9);
    agbar(k) = hbar(x,y,z,k+12);
  }
  const Real rho = hbar(x,y,z,0);
  const Real phi = hbar(x,y,z,1);

  // next, update the modified quantities to real quantities;
  h(x,y,z,0) = rho; // rho
  h(x,y,z,1) = phi;  // phi
  // velocity of fluid f, nf[1~D] for x~z components         !!!!!!!!!!!!!!!!!!!!!!!!  modification I !!!!!!!!!!!!!!
  if(std::abs(rho)>0){
    h(x,y,z,2) = (rho*ufbar(0) + 0.5*rho*afbar(0) - 0.5/(tau_f+0.5)*rho*phi/(rho+phi)*(ufbar(0)-ugbar(0) + 0.5*(afbar(0)-agbar(0))) + 0.5*nf(x,y,z,1)); 
    h(x,y,z,3) = (rho*ufbar(1) + 0.5*rho*afbar(1) - 0.5/(tau_f+0.5)*rho*phi/(rho+phi)*(ufbar(1)-ugbar(1) + 0.5*(afbar(1)-agbar(1))) + 0.5*nf(x,y,z,2));
    h(x,y,z,4) = (rho*ufbar(2) + 0.5*rho*afbar(2) - 0.5/(tau_f+0.5)*rho*phi/(rho+phi)*(ufbar(2)-ugbar(2) + 0.5*(afbar(2)-agbar(2))) + 0.5*nf(x,y,z,3));
  }else{
    h(x,y,z,2) = 0; 
    h(x,y,z,3) = 0;
    h(x,y,z,4) = 0;
  }

  h(x,y,z,5) = hbar(x,y,z,5);  // pressure, temperally keep unchanged;

  // velocity of fluid g, ng[1~D] for x~z components
  if(std::abs(phi)>0){
    h(x,y,z,6) = (phi*ugbar(0) + 0.5*phi*agbar(0) - 0.5/(tau_g+0.5)*rho*phi/(rho+phi)*(ugbar(0)-ufbar(0) + 0.5*(agbar(0)-afbar(0))) + 0.5*ng(x,y,z,1));
    h(x,y,z,7) = (phi*ugbar(1) + 0.5*phi*agbar(1) - 0.5/(tau_g+0.5)*rho*phi/(rho+phi)*(ugbar(1)-ufbar(1) + 0.5*(agbar(1)-afbar(1))) + 0.5*ng(x,y,z,2));
    h(x,y,z,8) = (phi*ugbar(2) + 0.5*phi*agbar(2) - 0.5/(tau_g+0.5)*rho*phi/(rho+phi)*(ugbar(2)-ufbar(2) + 0.5*(agbar(2)-afbar(2))) + 0.5*ng(x,y,z,3));
  }else{
    h(x,y,z,6) = 0;
    h(x,y,z,7) = 0;
    h(x,y,z,8) = 0;
  }

  // acceleration of fluid f; equal to modified quantities;
  h(x,y,z,9) = hbar(x,y,z,9);
  h(x,y,z,10) = hbar(x,y,z,10);
  h(x,y,z,11) = hbar(x,y,z,11);
  // acceleration of fluid g; equal to modified quantities;
  h(x,y,z,12) = hbar(x,y,z,12);
  h(x,y,z,13) = hbar(x,y,z,13);
  h(x,y,z,14) = hbar(x,y,z,14);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Array1D<Real,0,nvel> fequilibrium_moments(const Array1D<Real,0,2>& fields, // rho&phi
			      const RealVect& u) {  // u is real velocity field of fluid f;
  Array1D<Real,0,nvel> mEq;

  const Real rho = fields(0); // density distribution on current site only;
  //const Real phi = fields(1);

  const Real coefA = rho;
  const Real coefB = 1.;
  const Real coefAB = coefA*coefB;
  const Real coefC = 1./cs2;
  const Real coefAC = coefA*coefC;
  RealVect coefE = { u[0], u[1], u[2] };
  //Print() << coefE << '\n';
  Real coefD[3][3];   // \frac{\bm{v}\bm{v}}{2c^4}, tensor of size 3*3;
  coefD[0][0] = u[0]*u[0]/2./cs4; coefD[0][1] = u[0]*u[1]/2./cs4; coefD[0][2] = u[0]*u[2]/2./cs4;
  coefD[1][0] = coefD[0][1]; coefD[1][1] = u[1]*u[1]/2./cs4; coefD[1][2] = u[1]*u[2]/2./cs4;
  coefD[2][0] = coefD[0][2]; coefD[2][1] = coefD[1][2]; coefD[2][2] = u[2]*u[2]/2./cs4;
  const Real trace_coefD = coefD[0][0] + coefD[1][1] + coefD[2][2];

  mEq( 0) = coefAB;

  mEq( 1) = coefAC*cs2*coefE[0];
  mEq( 2) = coefAC*cs2*coefE[1];
  mEq( 3) = coefAC*cs2*coefE[2];

  mEq( 4) = 2.*coefA*cs4*trace_coefD;
  mEq( 5) = 6.*coefA*cs4*coefD[0][0] - 2.*coefA*cs4*trace_coefD;
  mEq( 6) = 2.*coefA*cs4*(coefD[1][1] - coefD[2][2]);
  mEq( 7) = coefA*cs4*(coefD[0][1] + coefD[1][0]);
  mEq( 8) = coefA*cs4*(coefD[1][2] + coefD[2][1]);
  mEq( 9) = coefA*cs4*(coefD[0][2] + coefD[2][0]);

  mEq(10) = 0.;
  mEq(11) = 0.;
  mEq(12) = 0.;
  mEq(13) = 0.;
  mEq(14) = 0.;
  mEq(15) = 0.;

  mEq(16) = coefAB*(3.*(3.*cs2 + 6.*cs4) - 18.*cs2 + 1) + 3.*coefA*trace_coefD*(6.*cs4+cs2) - coefA*trace_coefD*(6.*cs2+12.*cs4+9.*cs4+18.*cs4*cs2)
  + coefA*trace_coefD*cs2*(18.*cs2+1) - coefA*cs2*trace_coefD;
  mEq(17) = 2.*coefA*cs2*(2.*coefD[0][0] - coefD[1][1] - coefD[2][2]) - 18.*coefA*cs4*coefD[0][0] + 6.*coefA*cs4*trace_coefD;
  mEq(18) = 0.;

  /*for(int i=0;i<nvel;i++){
    Print() << mEq(i) << '\t';
  }
  Print() << '\n';*/

  return mEq;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Array1D<Real,0,nvel> gequilibrium_moments(const Array1D<Real,0,2>& fields,
				   const RealVect& u) { // u is real velocity field of fluid g;
  Array1D<Real,0,nvel> mEq;

  //const Real rho = fields(0);
  const Real phi = fields(1);

  const Real coefA = phi;
  const Real coefB = 1.;
  const Real coefAB = coefA*coefB;
  const Real coefC = 1./cs2;
  const Real coefAC = coefA*coefC;
  RealVect coefE = { u[0], u[1], u[2] };
  Real coefD[3][3];   // \frac{\bm{v}\bm{v}}{2c^4}, tensor of size 3*3;
  coefD[0][0] = u[0]*u[0]/2./cs4; coefD[0][1] = u[0]*u[1]/2./cs4; coefD[0][2] = u[0]*u[2]/2./cs4;
  coefD[1][0] = coefD[0][1]; coefD[1][1] = u[1]*u[1]/2./cs4; coefD[1][2] = u[1]*u[2]/2./cs4;
  coefD[2][0] = coefD[0][2]; coefD[2][1] = coefD[1][2]; coefD[2][2] = u[2]*u[2]/2./cs4;
  const Real trace_coefD = coefD[0][0] + coefD[1][1] + coefD[2][2];

  mEq( 0) = coefAB;

  mEq( 1) = coefAC*cs2*coefE[0];
  mEq( 2) = coefAC*cs2*coefE[1];
  mEq( 3) = coefAC*cs2*coefE[2];

  mEq( 4) = 2.*coefA*cs4*trace_coefD;
  mEq( 5) = 6.*coefA*cs4*coefD[0][0] - 2.*coefA*cs4*trace_coefD;
  mEq( 6) = 2.*coefA*cs4*(coefD[1][1] - coefD[2][2]);
  mEq( 7) = coefA*cs4*(coefD[0][1] + coefD[1][0]);
  mEq( 8) = coefA*cs4*(coefD[1][2] + coefD[2][1]);
  mEq( 9) = coefA*cs4*(coefD[0][2] + coefD[2][0]);

  mEq(10) = 0.;
  mEq(11) = 0.;
  mEq(12) = 0.;
  mEq(13) = 0.;
  mEq(14) = 0.;
  mEq(15) = 0.;

  mEq(16) = coefAB*(3.*(3.*cs2 + 6.*cs4) - 18.*cs2 + 1) + 3.*coefA*trace_coefD*(6.*cs4+cs2) - coefA*trace_coefD*(6.*cs2+12.*cs4+9.*cs4+18.*cs4*cs2)
  + coefA*trace_coefD*cs2*(18.*cs2+1) - coefA*cs2*trace_coefD;
  mEq(17) = 2.*coefA*cs2*(2.*coefD[0][0] - coefD[1][1] - coefD[2][2]) - 18.*coefA*cs4*coefD[0][0] + 6.*coefA*cs4*trace_coefD;
  mEq(18) = 0.;

  return mEq;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Array1D<Real,0,nvel> Phif_equilibrium_moments(const Array1D<Real,0,2>& fields,
				   const RealVect& u, const RealVect& a) {  // u,a are real velocity & acceleration field of fluid f;
  Array1D<Real,0,nvel> mEq;

  const Real rho = fields(0);

  const Real coefA = rho;
  const Real coefB = 0.;
  const Real coefAB = coefA*coefB;
  const Real coefC = 1./cs2;
  const Real coefAC = coefA*coefC;
  RealVect coefE = { a[0], a[1], a[2] };
  Real coefD[3][3];   // \frac{\bm{a}\bm{v}}{c^4}, tensor of size 3*3; a,v are real hydrodynamic variables;
  coefD[0][0] = a[0]*u[0]/cs4; coefD[0][1] = a[0]*u[1]/cs4; coefD[0][2] = a[0]*u[2]/cs4;
  coefD[1][0] = a[1]*u[0]/cs4; coefD[1][1] = a[1]*u[1]/cs4; coefD[1][2] = a[1]*u[2]/cs4;
  coefD[2][0] = a[2]*u[0]/cs4; coefD[2][1] = a[2]*u[1]/cs4; coefD[2][2] = a[2]*u[2]/cs4;
  const Real trace_coefD = coefD[0][0] + coefD[1][1] + coefD[2][2];

  // Correction factor 1./(1.+1./(2.*tau_f)) should be multiplied before each moment's expression;
  const Real modifactor = 1./(1.+1./(2.*tau_f));
  mEq( 0) = modifactor*coefAB;

  mEq( 1) = modifactor*coefAC*cs2*coefE[0];
  mEq( 2) = modifactor*coefAC*cs2*coefE[1];
  mEq( 3) = modifactor*coefAC*cs2*coefE[2];

  mEq( 4) = modifactor*2.*coefA*cs4*trace_coefD;
  mEq( 5) = modifactor*6.*coefA*cs4*coefD[0][0] - 2.*coefA*cs4*trace_coefD;
  mEq( 6) = modifactor*2.*coefA*cs4*(coefD[1][1] - coefD[2][2]);
  mEq( 7) = modifactor*coefA*cs4*(coefD[0][1] + coefD[1][0]);
  mEq( 8) = modifactor*coefA*cs4*(coefD[1][2] + coefD[2][1]);
  mEq( 9) = modifactor*coefA*cs4*(coefD[0][2] + coefD[2][0]);

  mEq(10) = 0.;
  mEq(11) = 0.;
  mEq(12) = 0.;
  mEq(13) = 0.;
  mEq(14) = 0.;
  mEq(15) = 0.;

  mEq(16) = coefAB*(3.*(3.*cs2 + 6.*cs4) - 18.*cs2 + 1) + 3.*coefA*trace_coefD*(6.*cs4+cs2) - coefA*trace_coefD*(6.*cs2+12.*cs4+9.*cs4+18.*cs4*cs2)
  + coefA*trace_coefD*cs2*(18.*cs2+1) - coefA*cs2*trace_coefD;
  mEq(16) = modifactor*mEq(16);
  
  mEq(17) = 2.*coefA*cs2*(2.*coefD[0][0] - coefD[1][1] - coefD[2][2]) - 18.*coefA*cs4*coefD[0][0] + 6.*coefA*cs4*trace_coefD;
  mEq(17) = modifactor*mEq(17);

  mEq(18) = 0.;

  return mEq;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Array1D<Real,0,nvel> Phig_equilibrium_moments(const Array1D<Real,0,2>& fields,
				   const RealVect& u, const RealVect& a) {  // u,a are real velocity & acceleration field of fluid g;
  Array1D<Real,0,nvel> mEq;

  const Real phi = fields(1);

  const Real coefA = phi;
  const Real coefB = 0.;
  const Real coefAB = coefA*coefB;
  const Real coefC = 1./cs2;
  const Real coefAC = coefA*coefC;
  RealVect coefE = { a[0], a[1], a[2] };
  Real coefD[3][3];   // \frac{\bm{a}\bm{v}}{c^4}, tensor of size 3*3; a,v are real hydrodynamic variables;
  coefD[0][0] = a[0]*u[0]/cs4; coefD[0][1] = a[0]*u[1]/cs4; coefD[0][2] = a[0]*u[2]/cs4;
  coefD[1][0] = a[1]*u[0]/cs4; coefD[1][1] = a[1]*u[1]/cs4; coefD[1][2] = a[1]*u[2]/cs4;
  coefD[2][0] = a[2]*u[0]/cs4; coefD[2][1] = a[2]*u[1]/cs4; coefD[2][2] = a[2]*u[2]/cs4;
  const Real trace_coefD = coefD[0][0] + coefD[1][1] + coefD[2][2];

  // Correction factor 1./(1.+1./(2.*tau_g)) should be multiplied before each moment's expression;
  const Real modifactor = 1./(1.+1./(2.*tau_g));
  mEq( 0) = modifactor*coefAB;

  mEq( 1) = modifactor*coefAC*cs2*coefE[0];
  mEq( 2) = modifactor*coefAC*cs2*coefE[1];
  mEq( 3) = modifactor*coefAC*cs2*coefE[2];

  mEq( 4) = modifactor*2.*coefA*cs4*trace_coefD;
  mEq( 5) = modifactor*6.*coefA*cs4*coefD[0][0] - 2.*coefA*cs4*trace_coefD;
  mEq( 6) = modifactor*2.*coefA*cs4*(coefD[1][1] - coefD[2][2]);
  mEq( 7) = modifactor*coefA*cs4*(coefD[0][1] + coefD[1][0]);
  mEq( 8) = modifactor*coefA*cs4*(coefD[1][2] + coefD[2][1]);
  mEq( 9) = modifactor*coefA*cs4*(coefD[0][2] + coefD[2][0]);

  mEq(10) = 0.;
  mEq(11) = 0.;
  mEq(12) = 0.;
  mEq(13) = 0.;
  mEq(14) = 0.;
  mEq(15) = 0.;

  mEq(16) = coefAB*(3.*(3.*cs2 + 6.*cs4) - 18.*cs2 + 1) + 3.*coefA*trace_coefD*(6.*cs4+cs2) - coefA*trace_coefD*(6.*cs2+12.*cs4+9.*cs4+18.*cs4*cs2)
  + coefA*trace_coefD*cs2*(18.*cs2+1) - coefA*cs2*trace_coefD;
  mEq(16) = modifactor*mEq(16);
  
  mEq(17) = 2.*coefA*cs2*(2.*coefD[0][0] - coefD[1][1] - coefD[2][2]) - 18.*coefA*cs4*coefD[0][0] + 6.*coefA*cs4*trace_coefD;
  mEq(17) = modifactor*mEq(17);

  mEq(18) = 0.;

  return mEq;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void collide(int x, int y, int z,
	     Array4<Real> f,  Array4<Real> g,
	     Array4<Real> h,  //Array4<Real> hbar,
       Array4<Real> fn, Array4<Real> gn) {

  const Array1D<Real,0,2> fields = { h(x,y,z,0), h(x,y,z,1) }; // 0: rho; 1: phi
  const RealVect uf = { h(x,y,z,2), h(x,y,z,3), h(x,y,z,4) }; // velocity of f
  const RealVect ug = { h(x,y,z,6), h(x,y,z,7), h(x,y,z,8) }; // velocity of g
  const RealVect af = { h(x,y,z,9), h(x,y,z,10), h(x,y,z,11) }; // acceleration of f
  const RealVect ag = { h(x,y,z,12), h(x,y,z,13), h(x,y,z,14) }; // acceleration of g

  Array1D<Real,0,nvel> mf, mg, mfEq, mgEq, mPhif, mPhig;

  mf = moments(x,y,z,f); // here is the modified moments since all program is based on modified distribution functions f,g;
  mg = moments(x,y,z,g);

  // mf_eq and mg_eq, moments of equilibirum distribution function
  mfEq = fequilibrium_moments(fields, uf);
  mgEq = gequilibrium_moments(fields, ug);

  // moments of force term
  mPhif = Phif_equilibrium_moments(fields, uf, af);
  mPhig = Phig_equilibrium_moments(fields, ug, ag);

  // thermal noise in moment space: fn(x,y,z,ncomp) & gn(x,y,z,ncomp);

  const Real tau_f_bar = tau_f*(1.+0.5/tau_f);
  const Real tau_g_bar = tau_g*(1.+0.5/tau_g);
  for (int a=0; a<nvel; ++a) {  // the correction factor for the force term is included in the function [Phif_equilibrium_moments];
    Real Raf = 1./tau_f_bar * (mfEq(a) - mf(a)) + mPhif(a) + fn(x,y,z,a);
    if(std::abs(fn(x,y,z,a)/(1./tau_f_bar * (mfEq(a) - mf(a)) + mPhif(a)))>0.01){
      Print() << fn(x,y,z,a) << " -*- " << (1./tau_f_bar * (mfEq(a) - mf(a)) + mPhif(a)) << '\n';
    }
    Real Rag = 1./tau_g_bar * (mgEq(a) - mg(a)) + mPhig(a) + gn(x,y,z,a);
    mf(a) = mf(a) + Raf;
    mg(a) = mg(a) + Rag;
  }

  populations(x,y,z,f,mf);
  populations(x,y,z,g,mg);

}

// stream from \vec{x} to \vec{x}+\vec{c}_i (push scheme)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void stream_push(int x, int y, int z,
		 const Array4<Real>& fOld, const Array4<Real>& gOld,
		 const Array4<Real>& fNew, const Array4<Real>& gNew) {
  for (int i=0; i<nvel; ++i) { // push populations to neighbors
    int xp = x + c[i][0];
    int yp = y + c[i][1];
    int zp = z + c[i][2];
    fNew(xp,yp,zp,i) = fOld(x,y,z,i);
    gNew(xp,yp,zp,i) = gOld(x,y,z,i);
  }
}

// collide and stream (push scheme)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void collide_stream(int x, int y, int z,
		    const Array4<Real>& fOld,  const Array4<Real>& gOld,
		    const Array4<Real>& fNew,  const Array4<Real>& gNew,
		    const Array4<Real>& h,     //const Array4<Real>& hbar,
			  const Array4<Real>& fn,    const Array4<Real>& gn ) {
  collide(x, y, z, fOld, gOld, h, fn, gn); // collide populations on current site
  stream_push(x, y, z, fOld, gOld, fNew, gNew); // push populations to neighbors
}

// calculate real hydrodynamic moments for whole lattice
inline void LBM_hydrovars(MultiFab& mf, MultiFab& mg, MultiFab& hydrovs, MultiFab& hydrovsbar, MultiFab& fnoisevs, MultiFab& gnoisevs) {
  auto const & f = mf.arrays();
  auto const & g = mg.arrays();
  auto const & h = hydrovs.arrays();
  auto const & hbar = hydrovsbar.arrays();
  auto const & nf = fnoisevs.arrays(); // multiple noise Array4D arrays of size [box_index, geometry_size, nvel], i.e., nf[nbx](i,j,k,ncomp); 
  auto const & ng = gnoisevs.arrays();
  ParallelFor(hydrovs, IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
    hydrovars(x, y, z, f[nbx], g[nbx], h[nbx], hbar[nbx], nf[nbx], ng[nbx]);
  });
}

// calculate modified hydrodynamic moments for whole lattice
inline void LBM_hydrovars_modified(MultiFab& mf, MultiFab& mg, MultiFab& hydrovsbar) {
  auto const & f = mf.arrays();
  auto const & g = mg.arrays();
  auto const & hbar = hydrovsbar.arrays();
  const IntVect halo = IntVect(2);
  ParallelFor(hydrovsbar, halo, [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) { // Ghost cells settings????????
    hydrovars_bar(x, y, z, f[nbx], g[nbx], hbar[nbx]);
  });
}

// LB timestep for whole lattice
inline void LBM_timestep(const Geometry& geom,
			 MultiFab& fold, MultiFab& gold,
			 MultiFab& fnew, MultiFab& gnew,
			 MultiFab& hydrovs, MultiFab& hydrovsbar,
       MultiFab& fnoisevs, MultiFab& gnoisevs) {
  fold.FillBoundary(geom.periodicity());
  gold.FillBoundary(geom.periodicity());  /*  inner boundary ?? different boxes */
  hydrovs.FillBoundary(geom.periodicity());
  auto const & fOld = fold.arrays();
  auto const & gOld = gold.arrays();
  auto const & fNew = fnew.arrays();
  auto const & gNew = gnew.arrays();
  auto const & h = hydrovs.arrays();
  //auto const & hbar = hydrovsbar.arrays();
  auto const & fn = fnoisevs.arrays();
  auto const & gn = gnoisevs.arrays();

  IntVect halo = IntVect(1); // need to push from first halo layer into domain
  //IntVect halo = IntVect(2);
  ParallelFor(fold, halo, [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
    collide_stream(x, y, z,
		   fOld[nbx], gOld[nbx],
		   fNew[nbx], gNew[nbx], h[nbx], fn[nbx], gn[nbx]);
  });
  /*
  Swap (MultiFab &dst, MultiFab &src, int srccomp, int dstcomp, int numcomp, int nghost)
 	Swap from src to dst including nghost ghost cells. 
  */
  MultiFab::Swap(fold, fnew, 0, 0, nvel, 0);
  MultiFab::Swap(gold, gnew, 0, 0, nvel, 0);

  // here all Multifab variables;
  LBM_hydrovars_modified(fold, gold, hydrovsbar);
  thermal_noise(fnoisevs, gnoisevs, hydrovsbar);
  LBM_hydrovars(fold, gold, hydrovs, hydrovsbar, fnoisevs, gnoisevs);
}

// homogeneous mixture of C1 and C2 with random concentration fluctuations
/*
inline void LBM_init_mixture(MultiFab& mf,
			     MultiFab& mg,
			     MultiFab& hydrovs) {
  const Real C1 = 0.5;
  const Real C2 = 0.5;
  for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
    const Box& valid_box = mfi.validbox();
    const Array4<Real>& f = mf.array(mfi);
    const Array4<Real>& g = mg.array(mfi);
    ParallelForRNG(valid_box, [=] AMREX_GPU_DEVICE(int x, int y, int z, RandomEngine const& engine) {
      const Real rho = C1 + C2;
      const Real phi = (C1-C2)+0.01*RandomNormal(0., 1., engine);
      for (int i=0; i<nvel; i++) {
	      f(x,y,z,i) = w[i]*rho;
	      g(x,y,z,i) = w[i]*phi;
      }
    });
  }
  LBM_hydrovars(mf, mg, hydrovs);
}

// flat interface between C1 and C2
inline void LBM_init_flat_interface(const Geometry& geom,
				    MultiFab& mf,
				    MultiFab& mg,
				    MultiFab& hydrovs) {
  const IntVect box = geom.Domain().length();

  auto const & f = mf.arrays();
  auto const & g = mg.arrays();
  GpuArray<Real,3> dx = geom.CellSizeArray();
  ParallelFor(mf, IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
    const Real rx = x-box[0]/2;
    const Real ry = y-box[1]/2;
#if (AMREX_SPACEDIM == 2)
    const Real r2 = rx*rx+ry*ry;
#elif (AMREX_SPACEDIM == 3)
    const Real rz = z-box[0]/2;
    const Real r2 = rx*rx+ry*ry+rz*rz;
#endif

    const Real rho=1.0;
    Real phi;

    if (rx < 0.) {
      phi = 0.5;
    } else {
      phi = -0.5;
    }

    for (int i=0; i<nvel; i++) {
      f[nbx](x,y,z,i) = w[i]*rho;
      g[nbx](x,y,z,i) = w[i]*phi;
    }

  });
  LBM_hydrovars(mf, mg, hydrovs);
}
*/

// binary droplet of C1 and C2
inline void LBM_init_droplet(const Real r,
			     const Geometry& geom,
			     MultiFab& mf,
			     MultiFab& mg,
			     MultiFab& hydrovs,
           MultiFab& hydrovsbar,
           MultiFab& fnoisevs,
           MultiFab& gnoisevs) {
  const IntVect box = geom.Domain().length();
  /*
  const Box& amrex::Geometry::Domain() const: Returns our rectangular domain.
  AMREX_GPU_HOST_DEVICE IntVect amrex::Box::length() const: Return the length of the Box.
  */
  const Real R = r*box[0];

  auto const & f = mf.arrays(); 
  auto const & g = mg.arrays();
  GpuArray<Real,3> dx = geom.CellSizeArray();
  ParallelFor(mf, IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) { 
    const Real rx = x-box[0]/2.;
    const Real ry = y-box[1]/2;
#if (AMREX_SPACEDIM == 2)
    const Real r2 = rx*rx+ry*ry;
#elif (AMREX_SPACEDIM == 3)
    const Real rz = z-box[0]/2;
    const Real r2 = rx*rx+ry*ry+rz*rz;
#endif
    const Real r = sqrt(r2);

    const Real rho = 1.0;
    const Real phi0 = 0.5;
    const Real phi = phi0*tanh((R-r)/sqrt(kappa));

    // transform from free energy model (rho,phi) --> (rhof, rhog); droplet is fluid f
    const Real rhof = phi + 0.5;
    const Real rhog = 0.5 - phi;
    for (int i=0; i<nvel; i++) {
      f[nbx](x,y,z,i) = w[i]*rhof;
      g[nbx](x,y,z,i) = w[i]*rhog;
    }

  });
  LBM_hydrovars_modified(mf, mg, hydrovsbar);
  thermal_noise(fnoisevs, gnoisevs, hydrovsbar);
  LBM_hydrovars(mf, mg, hydrovs, hydrovsbar, fnoisevs, gnoisevs);
}

#endif
