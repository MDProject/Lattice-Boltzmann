#ifndef LBM_DEB_
#define LBM_DEB_
#include "LBM_binary.H"
#include <cmath>

const Real PI = 3.1415926;


AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool loopNaN(MultiFab& mfab, bool ifPrintNANGrid){
    auto const & mfab_multi_array4D = mfab.arrays();
    Array4<Real> const mfab_array4D_nbx0 = mfab_multi_array4D[0];
    int ncomp = mfab_array4D_nbx0.nComp();
    bool iftrue = false;    // default return result is "is not NaN"
    for(int n=0; n<ncomp; n++){
        // reference [&] use instead of [=] value in Lambda expression;
        ParallelFor(mfab, IntVect(0), [&] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
        //Print() << '(' << x << "," << y << ',' << z << "," << nbx << ")-" << mfab_multi_array4D[nbx](x,y,z,1) << '\t';
            if(std::isnan(mfab_multi_array4D[nbx](x,y,z,n))){
                if(ifPrintNANGrid){
                    Print() << n << "-(" << x << "," << y << ',' << z << "," << nbx << ")-" << mfab_multi_array4D[nbx](x,y,z,n) << '\t';
                }
                iftrue = true;
            }
        });
    }
    return iftrue;
}

// check if [mfab] has NAN errors for all elements; 
// print lattice points where NAN values appear if ifPrintNANGrid = true; 
// if nstep>0 information is not provided, it will only check the NAN existence without printing step info;
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
inline bool MultiFabNANCheck(MultiFab& mfab, bool ifPrintNANGrid, int nstep = -1) {
    if(loopNaN(mfab, ifPrintNANGrid)){
        if(nstep>=0){
            Print() << "NAN accurs at step " << nstep << " stop and check!" << '\n';
            exit(0);
        }else{
            Print() << "The MultiFab has NAN elements" << '\n'; return true;
        }
    }else{
        return false;
    }
}


AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Array1D<Real,0,2> compute_multifab_fluctuation(MultiFab& mfab, const int ncomp){
    // calculate the fluctuation of component [ncomp]
    auto const & mfab_multi_array4D = mfab.arrays();
    Real mean = 0.;   Real var = 0.;
    int N = 0;
    ParallelFor(mfab, IntVect(0), [&] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
        mean = mean + mfab_multi_array4D[nbx](x,y,z,ncomp);
        N++;
    });
    mean = mean/N;
    ParallelFor(mfab, IntVect(0), [&] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
        Real delta = mfab_multi_array4D[nbx](x,y,z,ncomp) - mean;
        var = var + delta*delta;
    });
    var = var/N;
    Array1D<Real,0,2> density_info = {mean, sqrt(var)};
    return density_info;
}


// [densityIdx] contains index of density needed to print; set -1 if empty for that site, i.e., {5, -1, -1};
// Print step info if [nstep] is provided;
inline void PrintDensityFluctuation(MultiFab& hydrovs, Vector<std::string> VariableNames,
    int nstep = -1, const IntVect densityIdx = {0,1,5}){
    for(int n=0; n<3; n++){
        if(densityIdx[n]>=0){
            Array1D<Real,0,2> rho_info = compute_multifab_fluctuation(hydrovs, densityIdx[n]);
            Print() << VariableNames[densityIdx[n]] << ": " << "(mean = " << rho_info(0) << ", ";
            Print() << "standard deviation = " << rho_info(1) << ") ";
            if(nstep>=0){
                Print() << "at step " << nstep << '\n';
            }else{
                Print() << '\n';
            }
        }
    }
}

// [densityIdx] contains index of density needed to print; set -1 if empty for that site, i.e., {5, -1, -1};
// Generally called at last step
inline void PrintMassConservation(MultiFab& hydrovs, Vector<std::string> VariableNames, 
    Real L, Real radius, const IntVect densityIdx = {0,1,5}, int tag = 0){
    for(int n=0; n<3; n++){
        if(n == tag){
            Array1D<Real,0,2> rho_info;
            if(densityIdx[n]>=0){
                rho_info = compute_multifab_fluctuation(hydrovs, densityIdx[n]);
                Print() << VariableNames[densityIdx[n]] << ": " << "(mean = " << rho_info(0) << ", ";
                Print() << "standard deviation = " << rho_info(1) << ") ";
            }
            if(tag == 0){
                Real ratio_rho_f = rho_info(0)*L*L*L/(4./3.*PI*radius*radius*radius);
                Print() << "(rho_f*L^3)/(droplet volume): " << ratio_rho_f << '\n';
            }
        }
    }
}

/*  
    amrex::MultiFabFileFullPrefix():    e.g., return "** /plt[nstep]/Level_[nlevel]/Cell"
    plotfilename (e.g., "** /plt[nstep]") is assembled by function std::string amrex::Concatenate(const std::string& root, int num, int mindigits)
    (returns rootNNNN where NNNN == num) using input "plot_file_root"
    This function calculates the p-norm of the density variations (ensemble average mean substracted) matrix from [step1] ~ [step2].
    const amrex::BoxArray& ba = mf.boxArray(); 
*/		
inline void PrintConvergence(std::string plot_file_root, int step1, int step2, int nsteps, int compIdx=0, int nlevel=0, int mindigits=5,
             const std::string &levelPrefix="Level_", const std::string &mfPrefix="Cell"){
    int step_intval = (step2 - step1)/(nsteps-1);
    Print() << "Convergence test emsemble selection: from step " << step1 << " to step " << step2 << " with step interval " << step_intval << '\n';
    for(int n=step1; n<=step2; n=n+step_intval){
        const std::string& checkpointname = amrex::Concatenate(plot_file_root,n,mindigits);
        //VisMF::Read((*hydrovs_readin_ptr), amrex::MultiFabFileFullPrefix(nlevel, checkpointname, levelPrefix, mfPrefix));
    }

}



#endif