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
// Compute the total fluctuation for all grid points of a single multifab; measure of the data uniformity
// return Array2D object: (comp_index={k:densityIdx[k], k=0,1,2}, info_index={0: mean; 1: standard deviation})
inline Array2D<Real,0,3,0,2> PrintDensityFluctuation(MultiFab& hydrovs, Vector<std::string> VariableNames,
    int nstep = -1, const IntVect densityIdx = {0,1,5}){
    Array2D<Real,0,3,0,2> hydrovs_info;
    for(int n=0; n<3; n++){
        if(densityIdx[n]>=0){
            Array1D<Real,0,2> rho_info = compute_multifab_fluctuation(hydrovs, densityIdx[n]);
            Print() << VariableNames[densityIdx[n]] << ": " << "(mean = " << rho_info(0) << ", ";
            Print() << "standard deviation = " << rho_info(1) << ") ";
            hydrovs_info(n, 0) = rho_info(0);
            hydrovs_info(n, 1) = rho_info(1);
            if(nstep>=0){
                Print() << "at step " << nstep << '\n';
            }else{
                Print() << '\n';
            }
        }
    }
    return hydrovs_info;
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
    This function calculates the p-like norm of the density variations (ensemble average mean substracted) matrix from [step1] ~ [step2] (both included).
    ifSave: indicator of whether save the ensemble average hydrovs component into [mfab_ref]; false (do not save) by default;
*/		
inline void PrintConvergence(std::string plot_file_root, int step1, int step2, int step_int, MultiFab& mfab_ref, int compIdx=0,int p_tag=1, bool ifSave=false, 
            int nlevel=0, int mindigits=5, const std::string &levelPrefix="Level_", const std::string &mfPrefix="Cell"){
    std::string File(amrex::Concatenate(plot_file_root,step1,mindigits) + "/Header");
    //VisMF::IO_Buffer io_buffer(VisMF::GetIOBufferSize());
    Vector<char> fileCharPtr;
    ParallelDescriptor::ReadAndBcastFile(File, fileCharPtr); // char stream of [File]
    /*for (int i = 0; i < fileCharPtr.size(); ++i) {
        std::cout << fileCharPtr[i] << " ";
    }*/
    std::string fileCharPtrString(fileCharPtr.dataPtr()); // get access to the underlying data pointer, ONLY for Vector in AMRex; same with .data() here;
    std::istringstream is(fileCharPtrString, std::istringstream::in);
    std::string line, word;
    // read in title line, meaningless generally;
    std::getline(is, line);
    // read in finest_level
    int numComp;
    is >> numComp;
    //Print() << numComp << '\n';
    
    Print() << "Convergence test emsemble selection: from step " << step1 << " to step " << step2 << " with step interval " << step_int << '\n';
    int Nsteps = (step2 - step1)/step_int + 1;
    MultiFab** hydrovs_readin_array = new MultiFab*[Nsteps];    // arrays of hydrovs MultiFab pointers, used for storing read-in multifabs ensemble

    MultiFab hydrovs_mean(mfab_ref.boxArray(), mfab_ref.DistributionMap(), 1, mfab_ref.nGrow());
    hydrovs_mean.setVal(0., 0, 1); // set values 0. from comp 0, total 1 component;
    auto const & hydrovs_mean_array4D = hydrovs_mean.arrays();
    for(int n=0; n<Nsteps; n++){
        int nstep = step1 + step_int*n;
        // construct multifab object with same params with given [mfab_ref];
        hydrovs_readin_array[n] = new MultiFab(mfab_ref.boxArray(), mfab_ref.DistributionMap(), numComp, mfab_ref.nGrow()); // why here with ncomp = 1, it still can access component >=1?
        const std::string& checkpointname = amrex::Concatenate(plot_file_root,nstep,mindigits);

        VisMF::Read((*(hydrovs_readin_array[n])), amrex::MultiFabFileFullPrefix(nlevel, checkpointname, levelPrefix, mfPrefix));
        //Print() << hydrovs_readin_array[n]->nComp() << '\n';
        auto const & hydrovs_readin_ptr_array4D = hydrovs_readin_array[n]->arrays();
        // calculate the mean multifab for given component index [compIdx]
        ParallelFor((*(hydrovs_readin_array[n])), IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
            hydrovs_mean_array4D[nbx](x,y,z,0) = hydrovs_mean_array4D[nbx](x,y,z,0) + hydrovs_readin_ptr_array4D[nbx](x,y,z,compIdx);
        });
    }
    hydrovs_mean.mult(1./Nsteps, 0, 1, mfab_ref.nGrow());  // multilication for component 0, total 1 components with [mfab_ref.nGrow()] ghost layer;
    if(ifSave){
        mfab_ref.copy(hydrovs_mean, 0, 0, 1);   // derived from BaseFab member function;
    }

    // calculate the p-residues (p=1 or \infty) for ensemble MultiFabs stored in array [hydrovs_readin_array];
    // Here mfab data are considered as reshaped vectors;
    // ||A||_1: 1/N*\sum |aijk| and ||A||_\infty: max_{ijk} |aijk|
    MultiFab hydrovs_deviation(mfab_ref.boxArray(), mfab_ref.DistributionMap(), 1, mfab_ref.nGrow());
    hydrovs_deviation.setVal(0.,0,1);
    auto const & hydrovs_deviation_array4D = hydrovs_deviation.arrays();

    for(int n=0; n<Nsteps; n++){
        ParallelFor((*(hydrovs_readin_array[n])), IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
            auto const & hydrovs_readin_ptr_array4D = hydrovs_readin_array[n]->arrays();
            hydrovs_deviation_array4D[nbx](x,y,z,0) = hydrovs_deviation_array4D[nbx](x,y,z,0)
                    + std::abs(hydrovs_readin_ptr_array4D[nbx](x,y,z,compIdx) - hydrovs_mean_array4D[nbx](x,y,z,0));  // for given component index [compIdx]
        });
    }
    hydrovs_deviation.mult(1./Nsteps);

    const BoxArray& ba = mfab_ref.boxArray();
    Box domain_box = ba.minimalBox();
    IntVect total_size = domain_box.size();
    switch(p_tag) {
        case 1:
            Print() << hydrovs_deviation.sum()/(total_size[0]*total_size[1]*total_size[2]) << '\n';
            break;
        case -1:    // for p=\infty
            Print() << hydrovs_deviation.max(0) << '\n';    // maximum value for component 0;
            break;
    }
    /*for(int n=0; n<Nsteps; n++){
        delete[] hydrovs_readin_array[n]; 
    }*/
    delete[] hydrovs_readin_array;
}

void WriteVectorToFile(const std::vector<double>& data, const std::string& filename) {
    // Open the file for writing
    std::ofstream outFile(filename, std::ios::app | std::ios::out);

    // Check if the file was opened successfully
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }

    // Write the vector elements to the file, separated by '\t'
    for (size_t i = 0; i < data.size(); ++i) {
        outFile << data[i];
        if (i < data.size() - 1) {
            outFile << '\t';  // Add tab separator between elements
        }
    }

    // Close the file
    outFile.close();
}

inline void WriteOutNoise(string plot_file_root, int step,
			const MultiFab& fnoise, const MultiFab& gnoise,
			const Geometry& geom, const int Ndigits,
            const Vector<std::string>& var_names_f = {},
            const Vector<std::string>& var_names_g = {}) {

  const Real time = step;
  int pltlen = plot_file_root.length();
  const int ncomp = fnoise.nComp();
  string plot_file_root_fnoise = plot_file_root.substr(0, pltlen-3); // 15; substract the "plt" length;
  plot_file_root_fnoise = plot_file_root_fnoise + "data_fnoise/fn";
  const std::string& pltfile_fn = amrex::Concatenate(plot_file_root_fnoise,step,Ndigits);
  Vector <std::string> vname;
  for(int n=0; n<ncomp; n++){
    string compname = var_names_f.empty()? format("fa%d", n) : var_names_f[n];
    vname.push_back(compname);
  }
  WriteSingleLevelPlotfile(pltfile_fn, fnoise, vname, geom, time, step);

  string plot_file_root_gnoise = plot_file_root.substr(0, pltlen-3);
  plot_file_root_gnoise = plot_file_root_gnoise + "data_gnoise/gn";
  const std::string& pltfile_gn = amrex::Concatenate(plot_file_root_gnoise,step,Ndigits);
  vname.clear();
  for(int n=0; n<ncomp; n++){
    string compname = var_names_g.empty()? format("ga%d", n) : var_names_g[n];
    vname.push_back(compname);
  }
  WriteSingleLevelPlotfile(pltfile_gn, gnoise, vname, geom, time, step);
}

#endif