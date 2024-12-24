#ifndef LBM_HYDRO_
#define LBM_HYDRO_
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <sys/stat.h>
#include "LBM_binary.H"
#include "Debug.H"
#include "externlib.H"

extern void paramsVariations(Real* param_vars, const std::vector<std::vector<Real>>& combNomial, const std::vector<Real>& S_array,
    Function3DAMReX& func_rho, Real Wn, Real Rn, Real eta_W, Real eta_R, Real dt);

extern const int NumOfTerms;
extern std::vector<Real> getCoefS(int N);
extern std::vector<std::vector<Real>> getCombNomial(int n_max);

void getCenterOfMass(RealVect& vec_com, Function3DAMReX& func_rho){
    const BoxArray& ba = func_rho.getBoxArray();
    const Geometry& geom = func_rho.getGeometry();
    const DistributionMapping& dm = func_rho.getDistributionMapping();
    int ncomp = func_rho.getnComp();
    int ngrow = func_rho.getnGrow();

    const IntVect box = geom.Domain().length();
    RealBox real_box = geom.ProbDomain();
    Real Lx = real_box.hi(0) - real_box.lo(0);
    Real Ly = real_box.hi(1) - real_box.lo(1);
    Real Lz = real_box.hi(2) - real_box.lo(2);
    IndexType itype = ba.ixType(); // using Box or BoxArray;
    Array<Real, 3> cell_size;
    if(itype.cellCentered()){
        cell_size = {Lx/box[0], Ly/box[1], Lz/box[2]};
    }else if(itype.nodeCentered()){
        cell_size = {Lx/(box[0]-1), Ly/(box[1]-1), Lz/(box[2]-1)};
    }

    MultiFab mfab_X(ba, dm, ncomp, ngrow);
    MultiFab mfab_Y(ba, dm, ncomp, ngrow);
    MultiFab mfab_Z(ba, dm, ncomp, ngrow);
    for (MFIter mfi(mfab_X); mfi.isValid(); ++mfi) {
        const Box& valid_box = mfi.validbox();
        Array4<Real> const& X_array = mfab_X.array(mfi); // This function include ghost layers by default;
        Array4<Real> const& Y_array = mfab_Y.array(mfi);
        Array4<Real> const& Z_array = mfab_Z.array(mfi);
        ParallelFor(valid_box, [=] AMREX_GPU_DEVICE(int i, int j, int k){
            X_array(i,j,k,0) = (i+0.5*(1-itype[0]))*cell_size[0];
            Y_array(i,j,k,0) = (j+0.5*(1-itype[1]))*cell_size[1];
            Z_array(i,j,k,0) = (k+0.5*(1-itype[2]))*cell_size[2];
        });
    }
    Function3DAMReX func_X(mfab_X, geom);
    Function3DAMReX func_Y(mfab_Y, geom);
    Function3DAMReX func_Z(mfab_Z, geom);

    Real*** wt;
    Integration::trapezoid3DWeightTensor(geom.Domain(), &wt);
    Real mass = func_rho.integral3D(wt);
    vec_com[0] = func_rho.integral3D(func_X, wt)/mass;
    vec_com[1] = func_rho.integral3D(func_Y, wt)/mass;
    vec_com[2] = func_rho.integral3D(func_Z, wt)/mass;

    Integration::free_3d_array(wt, box[0], box[1], box[2]);
}

void fittingDropletRadius(Function3DAMReX& func_rho, std::vector<Array<Real, 2>>& param_vec,
    const std::vector<std::vector<Real>>& combNomial, const std::vector<Real>& S_array,
    Real W0=0.02, Real R0=0.3, Real eta_W=0.2, Real eta_R=0.2, Real dt=0.02, int Nstep=500, bool printDetails=true){    // kappa := 2W;
    
    param_vec.resize(Nstep);
    Array<Real, 2> params;  params[0] = W0; params[1] = R0;
    param_vec.push_back(params);
    Real param_arr[2];
    Real Wn = W0;   Real Rn = R0;
    for(int k=1; k<Nstep; k++){
        paramsVariations(param_arr, combNomial, S_array, func_rho, Wn, Rn, eta_W, eta_R, dt);
        Wn = Wn + param_arr[0]; 
        Rn = Rn + param_arr[1];
        Array<Real, 2> params;  params[0] = Wn; params[1] = Rn;
        param_vec.push_back(params);
        if(printDetails){
            printf("(Wn=%f,Rn=%f)\n", Wn, Rn);
        }
    }
}

void fittingDropletRadius(Function3DAMReX& func_rho, std::vector<Array<Real, 2>>& param_vec,
    Real W0=0.02, Real R0=0.3, Real eta_W=0.2, Real eta_R=0.2, Real dt=0.02, int Nstep=1500){

    const std::vector<std::vector<Real>> combNomial = getCombNomial(4); // verified
    const std::vector<Real> S_array = getCoefS(NumOfTerms); // verified
    fittingDropletRadius(func_rho, param_vec, combNomial, S_array, W0, R0, eta_W, eta_R, dt, Nstep);
}




    


#endif
