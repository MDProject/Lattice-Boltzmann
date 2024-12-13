#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_Print.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>

using namespace amrex;

#include "LBM_binary.H"
#include "Debug.H"
#include "AMReX_FileIO.H"
#include "externlib.H"
//#include "AMReX_FileIO.H"


const bool tagHDF5 = false; 
string plot_file_root = "./lbm_data_shshan_alpha0_4_xi_1e-6/plt";

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
  int nx = 16;
  int max_grid_size = 8;

  // default time stepping parameters
  int nsteps = 300;
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
  IntVect dom_hi(nx-1, nx-1, nx-1);
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
  MultiFab Test;
  MultiFab fold(ba, dm, nvel, nghost);
  MultiFab fnew(ba, dm, nvel, nghost);
  MultiFab gold(ba, dm, nvel, nghost);
  MultiFab gnew(ba, dm, nvel, nghost);
  MultiFab hydrovs(ba, dm, nhydro, nghost);
  MultiFab hydrovsbar(ba, dm, nhydro, nghost);  // modified hydrodynamic variables
  MultiFab fnoisevs(ba, dm, nvel, nghost); // thermal noise storage of fluid f for each step;
  MultiFab gnoisevs(ba, dm, nvel, nghost);

  // set up variable names for output
  const Vector<std::string> var_names = VariableNames(nhydro);

  MultiFab rho_eq(ba, dm, 1, nghost); rho_eq.setVal(0.);  // default values applies for 0 noise cases
  MultiFab phi_eq(ba, dm, 1, nghost); phi_eq.setVal(0.);
  MultiFab rhot_eq(ba, dm, 1, nghost);  rhot_eq.setVal(1.);

  bool noiseSwitch;
  if(mass==0){
    noiseSwitch = false;
  }else{
    noiseSwitch = true;
  }
  
  if(noiseSwitch){
    Print() << "Noise switch on\n";
    LoadSingleMultiFab("./equilibrium_rho", rho_eq);  // the ghost layers values are meaningless
    LoadSingleMultiFab("./equilibrium_phi", phi_eq);
    LoadSingleMultiFab("./equilibrium_rhot", rhot_eq);
  }else{
    Print() << "Noise switch off\n";
  }
  
  //***********
  Function3DAMReX func3D(ba, geom, dm, func3D_test);
  Function3DAMReX func3D_mfab(rho_eq, geom);
  MultiFab Imfab(ba, dm, 1, nghost);  Imfab.setVal(1.);
  Function3DAMReX func3D_weight(Imfab, geom);
  //Print() << func3D.getElement(1,1,1,0) << '\n';

  Print() << func3D_mfab.getElement(1,1,1,0) << '\n';
  func3D_mfab.add(func3D_weight);
  Print() << func3D_mfab.getElement(1,1,1,0) << '\n';
  //func3D_mfab.integral3D(func3D_weight);


  //Print() << func3D_mfab.getBoxArray() << '\n';
  //Print() << func3D_mfab.getnComp() << '\n';
  // if use like "BoxArray& ba_func = ***", [ba_func] will not be able to be modified; Since member function return the constant reference
  //BoxArray ba_func = func3D.getBoxArray(); 
  //Geometry geom_func = func3D.getGeometry();
  //func3D.setBoxArray(ba_func.maxSize(4));
  //Print() << func3D.getBoxArray() << '\n';
  //Print() << geom_func.Domain() << '\n';

  //func3D_mfab.setElement(-1,0,0,0,0);
  //Print() << func3D_mfab.getElement(0,0,0,0) << '\n';

  //***********/

  /**********
  int n = 4;
  Real d = 3;
  Real c = 4;
  int N = 15+1; // maximum order of taylor expansion terms for 1/cosh(dx-c); 15~20 for recommendation
  std::vector<Real> coefSVec = getCoefS(N);
  std::vector<std::vector<Real>> combVec = getCombNomial(n);
  //Print() << integral_func2_series(n, d, c, combVec, coefSVec, 1/d) << '\n';  // different numerical precision compared with python code;
  //Print() << integral_func3_series(3, 1.) << '\n';
  //Print() << integral_func1_series(3, 1., 100) << '\n';
  **********/

  /************
  // INITIALIZE
  LBM_init_droplet(radius, geom, fold, gold, hydrovs, hydrovsbar, fnoisevs, gnoisevs, rho_eq, phi_eq, rhot_eq);
  MultiFabNANCheck(hydrovs, true, 0);
  // Write a plotfile of the initial data if plot_int > 0
  if (plot_int > 0)
    WriteOutput(0, hydrovs, var_names, geom);
  Print() << "LB initialized\n";

  
  // TIMESTEP
  for (int step=1; step <= nsteps; ++step) {
    LBM_timestep(geom, fold, gold, fnew, gnew, hydrovs, hydrovsbar, fnoisevs, gnoisevs, rho_eq, phi_eq, rhot_eq);
    MultiFabNANCheck(hydrovs, false, step);
    if (plot_int > 0 && step%plot_int ==0){
      //PrintDensityFluctuation(hydrovs, var_names, step);
      WriteOutput(step, hydrovs, var_names, geom, tagHDF5);
      Print() << "LB step " << step << "\n";
    }
  }
  
  PrintMassConservation(hydrovs, var_names, 1., radius);

  // Call the timer again and compute the maximum difference between the start time 
  // and stop time over all processors
  Real stop_time = ParallelDescriptor::second() - strt_time;
  ParallelDescriptor::ReduceRealMax(stop_time);

  amrex::Print() << "Run time = " << stop_time << std::endl;

  // Print convergence and extract the equilibrium state solution;
  MultiFab mfab_rho_eq(ba, dm, 1, nghost);
  MultiFab mfab_phi_eq(ba, dm, 1, nghost);
  MultiFab mfab_rhot_eq(ba, dm, 1, nghost);
  PrintConvergence(plot_file_root, 200, 300, plot_int, mfab_rho_eq, 0, 1, (!noiseSwitch)); // copy the ensemble averaged solution to [mfab_rho_eq];
  Vector< std::string > vec_varname;  vec_varname.push_back("rho_eq");
  if(!noiseSwitch){
    WriteSingleLevelPlotfile("./equilibrium_rho", mfab_rho_eq, vec_varname, geom, 0, 0);  // time & step = 0 just for simplicity; meaningless here;
  }
  
  vec_varname.clear();  vec_varname.push_back("phi_eq");
  PrintConvergence(plot_file_root, 200, 300, plot_int, mfab_phi_eq, 1, 1, (!noiseSwitch));  // [phi] at index 1;
  if(!noiseSwitch){
    WriteSingleLevelPlotfile("./equilibrium_phi", mfab_phi_eq, vec_varname, geom, 0, 0);  // time & step = 0 just for simplicity; meaningless here;
  }
  
  vec_varname.clear();  vec_varname.push_back("rhot_eq");
  PrintConvergence(plot_file_root, 200, 300, plot_int, mfab_rhot_eq, 5, 1, (!noiseSwitch)); // total density at index 5;
  if(!noiseSwitch){
    WriteSingleLevelPlotfile("./equilibrium_rhot", mfab_rhot_eq, vec_varname, geom, 0, 0);  // time & step = 0 just for simplicity; meaningless here;
  }
  ***********/
  
}


/*
Visualization Tools Amrvis relies on "openmotif" on mac.
volpack.h files directory needs to be included in makefile
*/

// For TEST ONLY;
/*void main_driver(const char* argv){
  std::vector<int> g1;
  for (int i = 1; i <= 5; i++)
    g1.push_back(i);
  SliceWriteToPlainText(g1, ".", "test.dat");
}*/

/***************
  ******************  MultiFab related operations *******************
  
  // Read in last N output files
  MultiFab** vec_mfp = new MultiFab*[20];
  //MultiFab* vec_mfobj = new MultiFab[20]; see below, operator = overload is deleted;
  for(int n=0; n<10; n++){
    MultiFab* hydrovs_readin_ptr = new MultiFab(ba, dm, nvel, nghost);
    for (MFIter mfi((*hydrovs_readin_ptr)); mfi.isValid(); ++mfi) {
        Print() << (*hydrovs_readin_ptr)[mfi].box() << '\n';  // including ghost layers
        Print() << mfi.tilebox() << '\n';   // excluding ghost layers
        Print() << mfi.validbox() << '\n';  // excluding ghost layers
    }
    //MultiFab hydrovs_readin(ba, dm, nhydro, nghost);
    const std::string& checkpointname = amrex::Concatenate("./lbm_data_shshan_alpha0_4_xi_0/plt",200+10*n,5);
    VisMF::Read((*hydrovs_readin_ptr), amrex::MultiFabFileFullPrefix(0, checkpointname, "Level_", "Cell"));
    vec_mfp[n] = hydrovs_readin_ptr;

    /*  overload resolution selected deleted operator '=' so this CANNOT work
    MultiFab hydrovs_readin_obj(ba, dm, nvel, nghost);
    VisMF::Read(hydrovs_readin_obj, amrex::MultiFabFileFullPrefix(0, checkpointname, "Level_", "Cell"));
    vec_mfobj[n] = hydrovs_readin_obj;***************/

    /****************
    for (MFIter mfi((*hydrovs_readin_ptr)); mfi.isValid(); ++mfi) {
      const Box& valid_box = mfi.validbox();  // It excludes any regions added due to ghost cells or other boundary extensions.

      Print() << mfi.tilebox() << '\n'; // excluding ghost layers
      Print() << mfi.validbox() << '\n';  // excluding ghost layers
      Print() << (*hydrovs_readin_ptr)[mfi].box() << '\n';  // excluding ghost layers?? why
      // Access the FArrayBox corresponding to the current box
      const FArrayBox& fab = (*hydrovs_readin_ptr)[mfi];

      // Create a new FArrayBox to hold the extracted component
      FArrayBox extracted_component(valid_box, 1); // One component only

      // Copy the desired component from the original FArrayBox
      extracted_component.copy(fab, valid_box, 0, valid_box, 0, 1); // srcfab, srcbox, srccomp, destbox, destcomp, numcomp
      /*Note that although the srcbox and the destbox may be disjoint, they must be the same size and shape.
      If the sizes differ, the copy is undefined and a runtime error results.
      // Do something with the extracted component (e.g., print or process)
      // std::cout << "Extracted component " << 0 << " for box " << valid_box << std::endl;
    }
  }***************/

  /****************
  // print multifab stored in MultiFab array [vec_mfp]
  for(int n=0; n<10; n++){
    auto const & mfab_multi_array4D = (*vec_mfp[n]).arrays();
    ParallelFor((*vec_mfp[n]), IntVect(0), [&] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
        if(x==0&&y==0&&z==0){
          Print() << mfab_multi_array4D[nbx](x,y,z,5) << "  ";
        }
    });
  }

  // extract single component from MultiFab A to a new MultiFab B;
  // construct new MultiFab [testfunc_new] of total 1 component with same BoxArray, DistributionMapping and ghost layers;
  MultiFab mfab_rho(vec_mfp[0]->boxArray(), vec_mfp[0]->DistributionMap(), 10, vec_mfp[0]->nGrow());
  Print() << mfab_rho.nGrow() << '\n';
  mfab_rho.copy((*vec_mfp[0]), 5, 0, 1);   // copy the 5th component of [*vec_mfp[0]] to [mfab_rho]'s 0th component, with total num of comps=1;
  auto const & mfab_rho_array = mfab_rho.arrays();
  auto const & mfab_multi_array4D = (*vec_mfp[0]).arrays();
  // mfab_rho.FillBoundary(geom.periodicity());  // copy() does not deal with the periodic boundary condition; need to fill boundary condition again!!
  ParallelFor(mfab_rho, IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
    Print() << '(' << nbx << "," << x << ',' << y << ',' << z << ")-(" << mfab_rho_array[nbx](x,y,z,0) 
    << ',' << mfab_multi_array4D[nbx](x,y,z,5) << ')' << '\t';
  }); 
  *******************/