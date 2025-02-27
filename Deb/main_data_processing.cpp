#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_Print.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>

using namespace amrex;

#include "../LBM_binary.H"
#include "../Debug.H"
#include "../AMReX_FileIO.H"
#include "../externlib.H"
#include "../AMREX_Analysis.H"
#include "../LBM_hydrovs.H"
#include "../AMReX_DFT.H"

const bool tagHDF5 = false; 
string plot_file_root = "./lbm_data_shshan_alpha0_0.95_xi_1e-5_size64-4-4_continue/plt";

inline void WriteOutput(int step,
			const MultiFab& hydrovs,
			const Vector<std::string>& var_names,
			const Geometry& geom, bool tagHDF5 = false) {
  const Real time = step;
  const std::string& pltfile = amrex::Concatenate(plot_file_root,step,5);

  /*if(!filesystem::exists(pltfile)){
    if (mkdir(pltfile.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == 0 && step < 10 ){    // -1 for unsuccessful;
        Print() << "folder " << pltfile << "has been created" << std::endl;   
    }
  }*/

  if(tagHDF5){
    //WriteSingleLevelPlotfileHDF5(pltfile, hydrovs, var_names, geom, time, step);
  }else{
    WriteSingleLevelPlotfile(pltfile, hydrovs, var_names, geom, time, step);
  }
}

inline Vector<std::string> VariableNames(const int numVars) {
  // set variable names for output
  Vector<std::string> var_names(numVars);
  std::string name;
  int cnt = 0;
  // rho, phi
  var_names[cnt++] = "rho";
  var_names[cnt++] = "phi";
  // velx, vely, velz of fluid f
  for (int d=0; d<AMREX_SPACEDIM; d++) {
    name = "uf";
    name += (120+d);
    var_names[cnt++] = name;
  }
  var_names[cnt++] = "p_bulk";
  // velx, vely, velz of fluid g
  for (int d=0; d<AMREX_SPACEDIM; d++) {
    name = "ug";
    name += (120+d);
    var_names[cnt++] = name;
  }
  // acceleration_{x,y,z} of fluid f
  for (int d=0; d<AMREX_SPACEDIM; d++) {
    name = "af";
    name += (120+d);
    var_names[cnt++] = name;
  }
  // acceleration_{x,y,z} of fluid g
  for (int d=0; d<AMREX_SPACEDIM; d++) {
    name = "ag";
    name += (120+d);
    var_names[cnt++] = name;
  }

  // The following part is cutoff due to [cnt]>=variables number we input
  // pxx, pxy, pxz, pyy, pyz, pzz
  for (int i=0; i<AMREX_SPACEDIM, cnt<numVars; ++i) {
    for (int j=i; j<AMREX_SPACEDIM, cnt<numVars; ++j) {
      name = "p";
      name += (120+i);
      name += (120+j);
      var_names[cnt++] = name;
    }
  }
  // kinetic moments
  for (; cnt<numVars;) {
    name = "m";
    name += std::to_string(cnt);
    var_names[cnt++] = name;
  }

  // Print all variable names
  Print() << "Variable names list: ";
  for(int n=0; n<cnt; n++){
    Print() << var_names[n] << " ";
  }
  Print() << '\n';

  return var_names;
}


void main_driver(const char* argv) {

  // store the current time so we can later compute total run time.
  Real strt_time = ParallelDescriptor::second();
    
  // default grid parameters
  int nx = 16; //40;
  int max_grid_size = 8;

  // default time stepping parameters
  int nsteps = 5000;
  int plot_int = 10;

  // default droplet radius (% of box size)
  Real radius = 0.3;

  // input parameters
  ParmParse pp;
  pp.query("nx", nx);
  pp.query("max_grid_size", max_grid_size);
  pp.query("nsteps", nsteps);
  pp.query("plot_int", plot_int);
  pp.query("T", T);
  pp.query("kappa", kappa);
  pp.query("R", radius);

  // set up Box and Geomtry
  IntVect dom_lo(0, 0, 0);
  IntVect dom_hi(64-1, 4-1, 4-1); // size 64*2*2 does not work, will break down;
  Array<int,3> periodicity({1,1,1});

  Box domain(dom_lo, dom_hi);

  RealBox real_box({0.,0.,0.},{1.,1.,1.});
  
  Geometry geom(domain, real_box, CoordSys::cartesian, periodicity);

  BoxArray ba(domain);

  // split BoxArray into chunks no larger than "max_grid_size" along a direction
  ba.maxSize(max_grid_size);

  DistributionMapping dm(ba);

  // need two halo layers for gradients
  int nghost = 2; //2;

  // number of hydrodynamic fields to output
  int nhydro = 15; //6; 

  // set up MultiFabs
  Print() << "Start\n";
  MultiFab rho_frames(ba, dm, 11, nghost);
  LoadSetOfMultiFabs(plot_file_root, rho_frames, 0, 100, plot_int, 0); // component 0;
  
  
}