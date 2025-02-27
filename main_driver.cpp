#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_Print.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <format>
/*
  The definition of the correlation <,> is opposite to the paper, in which the conjugate operator is on left <var^*(k), var(k)>;
  In FHDeX: 1. calculate rho(k) for each frame and sum them with frames increasing;
            2. then remove the zero-freq components by Shift func
            3. compute the mean <rho(k)> for selected frames;
            2. compute the magnitude of <rho(k)> ?????
*/

//  data backup cmd:  cp -rp ./data_droplet* /Volumes/Extreme\ SSD/Lattice-Boltzmann-Data/

#include <StructFact.H>


using namespace amrex;

#include "LBM_binary.H"
#include "Debug.H"
#include "AMReX_FileIO.H"
#include "externlib.H"
#include "AMREX_Analysis.H"
#include "LBM_hydrovs.H"
#include "AMReX_DFT.H"

/*
    change "MAIN PARAMS SETTING" part and "GENERAL SETTINGS" for each job
    define Macro [MAIN_SOLVER] to run LBM evolution
    define Macro [POST_PROCESS] to run post data processing
*/


#define MAIN_SOLVER
//#define POST_PROCESS_DFT
//#define POST_PROCESS_DROPLET


extern Real alpha0; //  %.2f format
extern Real T;      //  %.1e format

//  **************************************    GENERAL SETTINGS     *****************************
const bool tagHDF5 = false; 

const bool is_flate_interface = false;
const bool is_droplet = false;
const bool is_mixture = true;
const int Ndigits = 6;
//  ********************************************************************************************

/*std::string format(string format_str, Real param){
  int length = format_str.length();
  char format_cstr[20];
  char param_cstr[20];
  for(int k=0; k<length-2; k++){
    if(k==0){ 
      format_cstr[k] = '%';
    }else{
      format_cstr[k] = format_str[k+1];
    }
  }
  format_cstr[length-2] = '\0';   // string terminator CHAR !!
  sprintf(param_cstr, format_cstr, param);
  string param_str(param_cstr);
  //printf(format_cstr);
  //std::cout << "param string " << param_str << std::endl;
  return param_str;
}*/


inline void WriteOutput(string plot_file_root, int step,
			const MultiFab& hydrovs,
			const Vector<std::string>& var_names,
			const Geometry& geom, StructFact& structFact, int plot_SF=0,
      bool tagHDF5 = false) {

  const Real time = step;
  const std::string& pltfile = amrex::Concatenate(plot_file_root,step,Ndigits);

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
  const int zero_avg = 1;
  if (plot_SF > 0) {
    string plot_file_root_SF = plot_file_root + "_SF";
    structFact.WritePlotFile(step, static_cast<Real>(step), plot_file_root_SF, zero_avg);
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

  bool noiseSwitch;
  if(T==0){
    noiseSwitch = false;
  }else{
    noiseSwitch = true;
  }
  // store the current time so we can later compute total run time.
  Real strt_time = ParallelDescriptor::second();
  int my_rank = amrex::ParallelDescriptor::MyProc(); 
  int nprocs = amrex::ParallelDescriptor::NProcs();
  // ***********************************  Basic AMReX Params ***********************************************************
  // default grid parameters
  int nx = 32; //40;
  int max_grid_size = nx/2;//4;


                  // ****************************************************************************
          // ***************************************************************************************************
  // ************************************************  MAIN PARAMS SETTING   ******************************************************
  // *******************************************  change for each job *******************************************************************
  int step_continue = 0; // set to be 0 for noise=0 case; set to be the steps number of the checkpoint file when noise != 0;
  bool continueFromNonFluct = true; // true for first time switching on noise; setting the suffix of the chkpoint file to be loaded (true->0 or false->T);
                                    // set "false" only if hope to continue from chkpoint with noise case;
  int nsteps = 400;//900000;//100000;
  int Out_Step = noiseSwitch? step_continue: step_continue;// + nsteps/2;
  int plot_int = 10;
  int plot_SF_window = 20000;
  // default output parameters
  int plot_SF = noiseSwitch? plot_int: 0; // switch on writting Structure Factor for noise!=0 case only;
  const int t_window = 10*plot_int;   //  specifying time window for calculating the equilibrium state solution;
                              //    must be multiples of [plot_int]
                              //  from step [last_step_index-t_window] to [last_step_index] (total frames determined by [plot_int])
  // ****************************************************************************************************************************************
      // **********************************************************************************************************************  
                  // ****************************************************************************


  // default droplet radius (% of box size)
  Real radius = 0.35;
  // set up Box and Geomtry
  IntVect dom_lo(0, 0, 0);
  // **********************************************************************************************


  // ********************  Box Domain Setting, change with the system  ************************
  IntVect dom_hi;
  if(is_mixture){
    dom_hi = IntVect(64-1, 2-1, 2-1);     // for mixture
  }else if(is_droplet){
    dom_hi = IntVect(nx-1, nx-1, nx-1); // for droplet 
  }else if(is_flate_interface){

  }
  
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
  // **********************************************************************************************


  // ************************************  File directory settings  ********************************
  string check_point_root_f, check_point_root_g;
  char plot_file_root_cstr[200];
  if(is_flate_interface){
    check_point_root_f = "./data_interface/f_checkpoint";
    check_point_root_g = "./data_interface/g_checkpoint";
    if(noiseSwitch){
      sprintf(plot_file_root_cstr, "./data_interface/lbm_data_shshan_alpha0_%.2f_xi_%.1e_size%d-%d-%d_continue/plt", alpha0, T,
      dom_hi[0]-dom_lo[0]+1, dom_hi[1]-dom_lo[1]+1, dom_hi[2]-dom_lo[2]+1);
    }else{
      sprintf(plot_file_root_cstr, "./data_interface/lbm_data_shshan_alpha0_%.2f_xi_%.1e_size%d-%d-%d/plt", alpha0, T,
      dom_hi[0]-dom_lo[0]+1, dom_hi[1]-dom_lo[1]+1, dom_hi[2]-dom_lo[2]+1);
    }
  }else if(is_droplet){
    check_point_root_f = "./data_droplet/f_checkpoint";
    check_point_root_g = "./data_droplet/g_checkpoint";
    if(noiseSwitch){
      sprintf(plot_file_root_cstr, "./data_droplet/lbm_data_shshan_alpha0_%.2f_xi_%.1e_size%d-%d-%d_continue/plt", alpha0, T,
      dom_hi[0]-dom_lo[0]+1, dom_hi[1]-dom_lo[1]+1, dom_hi[2]-dom_lo[2]+1);
    }else{
      sprintf(plot_file_root_cstr, "./data_droplet/lbm_data_shshan_alpha0_%.2f_xi_%.1e_size%d-%d-%d/plt", alpha0, T,
      dom_hi[0]-dom_lo[0]+1, dom_hi[1]-dom_lo[1]+1, dom_hi[2]-dom_lo[2]+1);
    }
  }else if(is_mixture){
    check_point_root_f = "./data_mixture/f_checkpoint";
    check_point_root_g = "./data_mixture/g_checkpoint";
    if(noiseSwitch){
      sprintf(plot_file_root_cstr, "./data_mixture/lbm_data_shshan_alpha0_%.2f_xi_%.1e_size%d-%d-%d_continue/plt", alpha0, T,
      dom_hi[0]-dom_lo[0]+1, dom_hi[1]-dom_lo[1]+1, dom_hi[2]-dom_lo[2]+1);
    }else{
      sprintf(plot_file_root_cstr, "./data_mixture/lbm_data_shshan_alpha0_%.2f_xi_%.1e_size%d-%d-%d/plt", alpha0, T,
      dom_hi[0]-dom_lo[0]+1, dom_hi[1]-dom_lo[1]+1, dom_hi[2]-dom_lo[2]+1);
    }
  }
  string plot_file_root(plot_file_root_cstr);
  // **********************************************************************************************************

  //  Set up equilibirum state solutions
  MultiFab rho_eq(ba, dm, 1, nghost); rho_eq.setVal(0.);  // default values applies for 0 noise cases
  MultiFab phi_eq(ba, dm, 1, nghost); phi_eq.setVal(0.);
  MultiFab rhot_eq(ba, dm, 1, nghost);  rhot_eq.setVal(1.);
  string rho_eq_file, phi_eq_file, rhot_eq_file;
  if(is_droplet){
    rho_eq_file = "./data_droplet/equilibrium_rho";
    phi_eq_file = "./data_droplet/equilibrium_phi";
    rhot_eq_file = "./data_droplet/equilibrium_rhot";
  }else if(is_mixture){
    rho_eq_file = "./data_mixture/equilibrium_rho";
    phi_eq_file = "./data_mixture/equilibrium_phi";
    rhot_eq_file = "./data_mixture/equilibrium_rhot";
  }else if(is_flate_interface){

  }
  rho_eq_file = rho_eq_file + "_alpha0_" + format("%.2f", alpha0) 
    + "_size" + format("%d-%d-%d", dom_hi[0]-dom_lo[0]+1, dom_hi[1]-dom_lo[1]+1, dom_hi[2]-dom_lo[2]+1);
  phi_eq_file = phi_eq_file + "_alpha0_" + format("%.2f", alpha0)
    + "_size" + format("%d-%d-%d", dom_hi[0]-dom_lo[0]+1, dom_hi[1]-dom_lo[1]+1, dom_hi[2]-dom_lo[2]+1);
  rhot_eq_file = rhot_eq_file + "_alpha0_" + format("%.2f", alpha0)
    + "_size" + format("%d-%d-%d", dom_hi[0]-dom_lo[0]+1, dom_hi[1]-dom_lo[1]+1, dom_hi[2]-dom_lo[2]+1);
  
  #ifdef MAIN_SOLVER    // **********************************   MICRO for time evolution  ****************************
  // *************************************  Set up Physical MultiFab Variables  *********************************
  MultiFab fold(ba, dm, nvel, nghost);
  MultiFab fnew(ba, dm, nvel, nghost);
  MultiFab gold(ba, dm, nvel, nghost);
  MultiFab gnew(ba, dm, nvel, nghost);
  MultiFab hydrovs(ba, dm, nhydro, nghost);
  MultiFab hydrovsbar(ba, dm, nhydro, nghost); // modified hydrodynamic variables
  MultiFab fnoisevs(ba, dm, nvel, nghost); // thermal noise storage of fluid f for each step;
  MultiFab gnoisevs(ba, dm, nvel, nghost);

  // set up variable names for output
  const Vector<std::string> var_names = VariableNames(nhydro);

  if(noiseSwitch){
    Print() << "Noise switch on\n";
    LoadSingleMultiFab(rho_eq_file, rho_eq);  // the ghost layers values are meaningless
    LoadSingleMultiFab(phi_eq_file, phi_eq);
    LoadSingleMultiFab(rhot_eq_file, rhot_eq);
  }else{
    Print() << "Noise switch off\n";
  }
  if(noiseSwitch){  printf("Numerical equilibrium state solution lower bound:\tmin rho_eq: %f\tmin phi_eq: %f\n", rho_eq.min(0), phi_eq.min(0));  }

  // *********************************** pre-processing numerical solution data for the droplet-equilibrium state  ************************************
  if(is_droplet && noiseSwitch){
    Function3DAMReX func_rho_eq(rho_eq, geom);
    RealVect vec_com = {0., 0., 0.};  
    getCenterOfMass(vec_com, func_rho_eq);
    printf("Center Of Mass: (%f,%f,%f)\n", vec_com[0], vec_com[1], vec_com[2]);
    // Fitting equilibrirum droplet radius; \frac{1}{2}\left(1+\tanh\frac{R-\left|\bm{r}-\bm{r}_{0}\right|}{\sqrt{2W}}\right)
    Array<Real, 3> param_arr = fittingDropletParams(func_rho_eq, 20, 0.005);  // last 20 steps ensemble mean, relative error < 0.005 bound;
    printf("fitting parameters for equilibrium density rho: (W=%f, R=%f)\n", param_arr[0], param_arr[1]);

    Real W_fit = param_arr[0];  Real R_fit = param_arr[1];
    // replace equilibrium state numerical solutions by fitted ones, i.e., [rho_eq], [phi_eq] and [rhot_eq];
    fitting_density_mfab(rho_eq, phi_eq, rhot_eq, geom, W_fit, R_fit, vec_com);
    printf("Fitting density profile:\tmin rho_eq: %f\tmin phi_eq: %f\n", rho_eq.min(0), phi_eq.min(0));
  }
  
  // ***********************************************  INITIALIZE **********************************************************************  
  bool if_continue_from_last_frame = noiseSwitch? true: false;
  string str_step_continue = "";
  if(step_continue>0){  str_step_continue = amrex::Concatenate(str_step_continue,step_continue,Ndigits);  }
  std::string pltfile_f, pltfile_g;

  // continue running from a checkpoint ...
  if(if_continue_from_last_frame){
    MultiFab f_last_frame(ba, dm, nvel, nghost);
    MultiFab g_last_frame(ba, dm, nvel, nghost);
    Print() << "Load in last frame checkpoint ....\n";
    pltfile_f = check_point_root_f + str_step_continue;
    pltfile_g = check_point_root_g + str_step_continue;
    Real chk_temp = continueFromNonFluct? 0.: T;
    pltfile_f = pltfile_f + "_alpha0_" + format("%.2f", alpha0) + "_xi_" + format("%.1e", chk_temp)
      + "_size" + format("%d-%d-%d", dom_hi[0]-dom_lo[0]+1, dom_hi[1]-dom_lo[1]+1, dom_hi[2]-dom_lo[2]+1);
    pltfile_g = pltfile_g + "_alpha0_" + format("%.2f", alpha0) + "_xi_" + format("%.1e", chk_temp)
      + "_size" + format("%d-%d-%d", dom_hi[0]-dom_lo[0]+1, dom_hi[1]-dom_lo[1]+1, dom_hi[2]-dom_lo[2]+1);
    if(is_flate_interface){
      LoadSingleMultiFab(pltfile_f, f_last_frame);
      LoadSingleMultiFab(pltfile_g, g_last_frame);
    }else if(is_droplet){
      LoadSingleMultiFab(pltfile_f, f_last_frame);
      LoadSingleMultiFab(pltfile_g, g_last_frame);
    }else if(is_mixture){
      LoadSingleMultiFab(pltfile_f, f_last_frame);
      LoadSingleMultiFab(pltfile_g, g_last_frame);
    }
    // continue from given initial populations f & g
    LBM_init(geom, fold, gold, hydrovs, hydrovsbar, fnoisevs, gnoisevs, 
    f_last_frame, g_last_frame, rho_eq, phi_eq, rhot_eq);

    PrintDensityFluctuation(hydrovs, var_names, -1); // check the data uniformity
  }else{  // continuing running from initial default states
    if(is_flate_interface){
      
    }else if(is_droplet){
      LBM_init_droplet(radius, geom, fold, gold, hydrovs, hydrovsbar, fnoisevs, gnoisevs, rho_eq, phi_eq, rhot_eq);
    }else if(is_mixture){
      LBM_init_mixture(geom, fold, gold, hydrovs, hydrovsbar, fnoisevs, gnoisevs, rho_eq, phi_eq, rhot_eq);
    }
  }
  Print() << "check modified hydrodynamic quantities validity ...\n";
  MultiFabNANCheck(hydrovsbar, true, 0);
  Print() << "check real hydrodynamic quantities validity ...\n";
  MultiFabNANCheck(hydrovs, true, 0);

  // set up StructFact
  int nStructVars = 2;
  //const Vector<std::string> var_names = hydrovars_names(nStructVars);
  const Vector<int> pairA = { 0, 1, 2, 3, 4};
  const Vector<int> pairB = { 0, 1, 2, 3, 4 };
  const Vector<Real> var_scaling = { 1.0, 1.0, 1.0, 1.0, 1.0};
  StructFact structFact(ba, dm, var_names, var_scaling, pairA, pairB);
  structFact.Reset();

  // Write a plotfile of the initial data if plot_int > 0 and starting from initial default states
  if (plot_int > 0 && step_continue == 0)
    WriteOutput(plot_file_root, 0, hydrovs, var_names, geom,  structFact, 0); 
  Print() << "LB initialized with alpha0 = " << alpha0 << " and T = " << T << '\n';
  if(is_droplet && my_rank == 0){  printf("Current case is droplet system with initial radius = %f\n", radius);   }

  // *****************************************************  TIMESTEP  *********************************************************
  int SF_start = step_continue + nsteps - plot_SF_window;
  std::vector<Real> radius_frames;
  std::vector<Real> rho_mean_frames;  //  rho mean value for each frame
  std::vector<Real> rho_sigma_frames; //  rho standard deviation for each frame
  for (int step=step_continue+1; step <= step_continue+nsteps; ++step) {
    amrex::ParallelDescriptor::Barrier();
    LBM_timestep(geom, fold, gold, fnew, gnew, hydrovs, hydrovsbar, fnoisevs, gnoisevs, rho_eq, phi_eq, rhot_eq);
    amrex::ParallelDescriptor::Barrier();

    if(noiseSwitch && step>=SF_start){
      amrex::ParallelDescriptor::Barrier();
      structFact.FortStructure(hydrovs, 0); // default reset value = 0 used here for accumulating correlations for each frame >= [SF_start] 
      amrex::ParallelDescriptor::Barrier();
      WriteOutNoise(plot_file_root, step, fnoisevs, gnoisevs, geom, Ndigits);
    }
    //MultiFabNANCheck(hydrovs, false, step);
    if (plot_int > 0 && step%plot_int == 0){
      Print() << "LB step " << step << " info:\n";
      // ******************************* Running Process Monitor *******************************************
      if(is_droplet && step>=step_continue/* + nsteps/2*/ && nprocs == 1){ // print out last 50% steps droplet size
        amrex::ParallelDescriptor::Barrier();
        MultiFab rho_current_step(ba, dm, 1, 0);
        rho_current_step.ParallelCopy(hydrovs, 0, 0, 1); // src, srccomp, dstcomp, numcomp
        Real rho_max = rho_current_step.max(0);
        Real rho_min = rho_current_step.min(0); // use original AMReX provided functions, thus need all threads prepared before calling;
        amrex::ParallelDescriptor::Barrier();
        
        if(my_rank == 0)  { printf("rho density maximum = %f, minimum = %f\n", rho_max, rho_min); }
        Print() << " info:\n";
        Function3DAMReX func_rho(rho_current_step, geom);
        RealVect vec_com = {0., 0., 0.};
        amrex::ParallelDescriptor::Barrier();
        getCenterOfMass(vec_com, func_rho);
        amrex::ParallelDescriptor::Barrier();

        if(amrex::ParallelDescriptor::IOProcessor())  { printf("Center Of Mass: (%f,%f,%f)\n", vec_com[0], vec_com[1], vec_com[2]); }

        amrex::ParallelDescriptor::Barrier();
        Array<Real, 3> param_arr;
        param_arr = fittingDropletParams(func_rho, 20, 0.01, 300);
        amrex::ParallelDescriptor::Barrier();

        if(amrex::ParallelDescriptor::IOProcessor()){
          printf("fitting parameters for density rho: (W=%f, R=%f) at step %d with relative undulation %.2e\n",
                      param_arr[0], param_arr[1], step, param_arr[2]);
          radius_frames.push_back(param_arr[1]);
        }
      }else if(is_mixture){
        Array2D<Real,0,3,0,2> density_info = PrintDensityFluctuation(hydrovs, var_names, step);
        rho_mean_frames.push_back(density_info(0, 0));  // (0: density rho; 0: mean)
        rho_sigma_frames.push_back(density_info(0, 1));  // (0: density rho; 1: standard deviation)
      }
      if(step >= Out_Step && step!=step_continue+nsteps){
        WriteOutput(plot_file_root, step, hydrovs, var_names, geom, structFact, 0); // do not output [structFact] during running time;
      }
    }
    if(step == step_continue+nsteps){
      WriteOutput(plot_file_root, step, hydrovs, var_names, geom, structFact, plot_SF);
    }
  }
  if(is_droplet && amrex::ParallelDescriptor::IOProcessor()){ // vector [radius_frames] is written by I/O rank so must output it in same rank;
    string radius_frames_file = plot_file_root.substr(0, 15);
    radius_frames_file = radius_frames_file + "radius_steps_out";
    Print() << "write out radius for each frame to file " << radius_frames_file << '\n';
    WriteVectorToFile(radius_frames, radius_frames_file);
  }
  if(is_mixture && amrex::ParallelDescriptor::IOProcessor()){ // vector [radius_frames] is written by I/O rank so must output it in same rank;
    string rho_mean_file = plot_file_root.substr(0, 15);
    rho_mean_file = rho_mean_file + "rho_mean_steps";
    Print() << "write out density rho mean for each frame to file " << rho_mean_file << '\n';
    WriteVectorToFile(rho_mean_frames, rho_mean_file);
    Print() << "write out density rho standard deviation for each frame to file " << rho_mean_file << '\n';
    string rho_sigma_file = plot_file_root.substr(0, 15);
    rho_sigma_file = rho_sigma_file + "rho_sigma_steps";
    WriteVectorToFile(rho_sigma_frames, rho_sigma_file);
  }

  // *****************************************************  Post-Processing  *********************************************************
  // write out last frame checkpoint
  pltfile_f = amrex::Concatenate(check_point_root_f, step_continue+nsteps, Ndigits);
  pltfile_g = amrex::Concatenate(check_point_root_g, step_continue+nsteps, Ndigits);
  pltfile_f = pltfile_f + "_alpha0_" + format("%.2f", alpha0) + "_xi_" + format("%.1e", T)
    + "_size" + format("%d-%d-%d", dom_hi[0]-dom_lo[0]+1, dom_hi[1]-dom_lo[1]+1, dom_hi[2]-dom_lo[2]+1);
  pltfile_g = pltfile_g + "_alpha0_" + format("%.2f", alpha0) + "_xi_" + format("%.1e", T)
    + "_size" + format("%d-%d-%d", dom_hi[0]-dom_lo[0]+1, dom_hi[1]-dom_lo[1]+1, dom_hi[2]-dom_lo[2]+1);
  if(is_flate_interface){
     
  }else if(is_droplet){
    Vector< std::string > varname_chk;  varname_chk.push_back("rho_chk");
    WriteSingleLevelPlotfile(pltfile_f, fold, varname_chk, geom, 0, 0);
    varname_chk.clear();  varname_chk.push_back("phi_chk");
    WriteSingleLevelPlotfile(pltfile_g, gold, varname_chk, geom, 0, 0);
  }else if(is_mixture){
    Vector< std::string > varname_chk;  varname_chk.push_back("rho_chk");
    WriteSingleLevelPlotfile(pltfile_f, fold, varname_chk, geom, 0, 0); // time & step = 0 just for simplicity; meaningless here;
    varname_chk.clear();  varname_chk.push_back("phi_chk");
    WriteSingleLevelPlotfile(pltfile_g, gold, varname_chk, geom, 0, 0); // time & step = 0 just for simplicity; meaningless here;
  }
  
  const IntVect box = geom.Domain().length();
  if(is_droplet && my_rank == 0){ PrintMassConservation(hydrovs, var_names, box[0], radius*box[0]); }

  // Call the timer again and compute the maximum difference between the start time
  // and stop time over all processors
  Real stop_time = ParallelDescriptor::second() - strt_time;
  ParallelDescriptor::ReduceRealMax(stop_time);

  amrex::Print() << "Run time = " << stop_time << std::endl;

  // Extract the equilibrium state solution;
  MultiFab mfab_rho_eq(ba, dm, 1, nghost);
  MultiFab mfab_phi_eq(ba, dm, 1, nghost);
  MultiFab mfab_rhot_eq(ba, dm, 1, nghost);
  int step1 = step_continue + nsteps - t_window; int step2 = step_continue + nsteps;

  // copy the ensemble averaged solution to [mfab_*_eq];
  if(!noiseSwitch){
    PrintConvergence(plot_file_root, step1, step2, plot_int, mfab_rho_eq, 0, 1/*lp,p=1 norm*/, (!noiseSwitch), 0, Ndigits);
    Vector< std::string > vec_varname;  vec_varname.push_back("rho_eq");
    WriteSingleLevelPlotfile(rho_eq_file, mfab_rho_eq, vec_varname, geom, 0, 0);  // time & step = 0 just for simplicity; meaningless here;
    vec_varname.clear();  vec_varname.push_back("phi_eq");
    PrintConvergence(plot_file_root, step1, step2, plot_int, mfab_phi_eq, 1, 1/*lp,p=1 norm*/, (!noiseSwitch), 0, Ndigits);  // [phi] at index 1;
    WriteSingleLevelPlotfile(phi_eq_file, mfab_phi_eq, vec_varname, geom, 0, 0);  // time & step = 0 just for simplicity; meaningless here;
    vec_varname.clear();  vec_varname.push_back("rhot_eq");
    PrintConvergence(plot_file_root, step1, step2, plot_int, mfab_rhot_eq, 5, 1, (!noiseSwitch), 0, Ndigits); // total density at index 5; nlevel=0;
    WriteSingleLevelPlotfile(rhot_eq_file, mfab_rhot_eq, vec_varname, geom, 0, 0);  // time & step = 0 just for simplicity; meaningless here;
  }

  #endif  // **********************************   END MICRO for time evolution  ****************************
  

  // **********************************   MICRO for post processing DFT ****************************
  #ifdef POST_PROCESS_DFT
  Print() << "Start Fourier Transform ...\n";
  int Nf1 = nsteps - plot_SF_window; int Nf2 = nsteps;
  int Nf_total = (Nf2-Nf1)/plot_int + 1;
  printf("Total %d frames in ensemble average\n", Nf_total);
  MultiFab rho_frames(ba, dm, Nf_total, nghost);
  MultiFab rho_frames_dft_norm2_mean(ba, dm, 1, nghost); rho_frames_dft_norm2_mean.setVal(0.);
  //MultiFab rho_frames_dft_real_mean(ba, dm, 1, nghost); rho_frames_dft_real_mean.setVal(0.);
  //MultiFab rho_frames_dft_imag_mean(ba, dm, 1, nghost); rho_frames_dft_imag_mean.setVal(0.);

  LoadSetOfMultiFabs(plot_file_root, rho_frames, Nf1, Nf2, plot_int, 0, false, Ndigits); // component 0;
  // specify each component's name as index [k], k=0~Nf_total-1;
  Vector< std::string > vec_rho_frames; vec_rho_frames.reserve(Nf_total);
  for(int k=0; k<Nf_total; k++){
    vec_rho_frames.push_back(std::to_string(k));
  }

  // substracting the equilibrium mean value;
  MultiFab rho_frames_mean(ba, dm, 1, nghost); rho_frames_mean.setVal(0.);
  /*void amrex::MultiFab::Saxpy	(	MultiFab& dst, Real a, const MultiFab& src, int srccomp, int dstcomp, int numcomp, int nghost)		
  static function,  dst += a*src  */
  for(int n=0; n<Nf_total; n++){
    amrex::MultiFab::Saxpy(rho_frames_mean, 1., rho_frames, n, 0, 1, nghost);
  }
  rho_frames_mean.mult(1./Nf_total, 0, 1);
  printf("rho mean: = %f\n", rho_frames_mean.sum(0)/domain.numPts());

  // substracting mean values from rho_frames
  for (MFIter mfi(rho_frames); mfi.isValid(); ++mfi) {
    Array4<Real> const& rho_frames_ptr = rho_frames.array(mfi);
    Array4<Real> const& rho_frames_mean_ptr = rho_frames_mean.array(mfi);
    Box bx = mfi.validbox();
    ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
      for(int n=0; n<Nf_total; n++){
        rho_frames_ptr(i,j,k,n) = rho_frames_ptr(i,j,k,n) - rho_frames_mean_ptr(i,j,k,0);
      }
    });
  }
  Real rho_variation_mean = 0.;
  for(int n=0; n<Nf_total; n++){
    rho_variation_mean += rho_frames.sum(n);
  }
  printf("variation of rho has mean: = %.1e\n", rho_variation_mean/Nf_total/domain.numPts());

  Real ratio_kBTcs2 = kBT/cs2;
  printf("kBT/cs2 = %f\n", ratio_kBTcs2);
  MultiFab field_dft_complex;
  for(int n=0; n<Nf_total; n++){
    amrex_fftw_r2c_3d(field_dft_complex, rho_frames, n, true);
    amrex_shift_fft_3d(field_dft_complex);    // ?? shift
    for (MFIter mfi(field_dft_complex); mfi.isValid(); ++mfi) {
      //Array4<Real> const& rho_frames_ptr = rho_frames.array(mfi);
      Array4<Real> const& field_dft_complex_ptr = field_dft_complex.array(mfi);
      Array4<Real> const& rho_frames_dft_norm2_mean_ptr = rho_frames_dft_norm2_mean.array(mfi);
      Box bx = mfi.validbox();
      ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
        rho_frames_dft_norm2_mean_ptr(i,j,k,0) = rho_frames_dft_norm2_mean_ptr(i,j,k,0) 
                      + field_dft_complex_ptr(i,j,k,0)*field_dft_complex_ptr(i,j,k,0) + field_dft_complex_ptr(i,j,k,1)*field_dft_complex_ptr(i,j,k,1);
      });
    }
  }
  rho_frames_dft_norm2_mean.mult(1./Nf_total, 0, 1);
  //amrex_shift_fft_3d(rho_frames_dft_norm2_mean);
  PrintMultiFab(rho_frames_dft_norm2_mean, 0);
  std::string plot_file_SF_root = plot_file_root.substr(0, plot_file_root.length()-3);
  plot_file_SF_root = plot_file_SF_root + "correlation_rho";
  Vector< std::string > cor_varname;  cor_varname.push_back("S_rho");
  WriteSingleLevelPlotfile(plot_file_SF_root, rho_frames_dft_norm2_mean, cor_varname, geom, 0, 0);

  #endif   // **********************************   END MICRO for POST_PROCESS_DFT  ****************************

  // **********************************   MICRO for POST_PROCESS_DROPLET ****************************
  #ifdef POST_PROCESS_DROPLET
  const Vector<std::string> var_names = VariableNames(nhydro);
  printf("Start fitting droplet radius ...\n");
  int step1 = 0;  int step2 = nsteps;
  MultiFab hydrovs_frames(ba, dm, nhydro, 0);
  MultiFab rho_frames(ba, dm, 1, 0);
  std::vector<Real> radius_frames;
  for(int n=step1; n<=step2; n=n+plot_int){
    const std::string& pltfile = amrex::Concatenate(plot_file_root,n,Ndigits);
    LoadSingleMultiFab(pltfile, hydrovs_frames);
    rho_frames.ParallelCopy(hydrovs_frames, 0, 0, 1);
    Real rho_max = rho_frames.max(0);
    Real rho_min = rho_frames.min(0); // use original AMReX provided functions, thus need all threads prepared before calling;
    amrex::ParallelDescriptor::Barrier();
    printf("rho density maximum = %f, minimum = %f\n", rho_max, rho_min);
    Function3DAMReX func_rho_frames(rho_frames, geom);
    RealVect vec_com = {0., 0., 0.};
    getCenterOfMass(vec_com, func_rho_frames);
    printf("Center Of Mass: (%f,%f,%f)\n", vec_com[0], vec_com[1], vec_com[2]);
    Array<Real, 3> param_arr = fittingDropletParams(func_rho_frames, 20, 0.01, 300);
    printf("fitting parameters for density rho: (W=%f, R=%f) at step %d with relative undulation %.2e\n",
                      param_arr[0], param_arr[1], n, param_arr[2]);
    radius_frames.push_back(param_arr[1]);
  }
  string radius_frames_file = plot_file_root.substr(0, 15);
  radius_frames_file = radius_frames_file + "radius_steps_out";
  Print() << "write out radius for each frame to file " << radius_frames_file << '\n';
  WriteVectorToFile(radius_frames, radius_frames_file);
  //fittingDropletCovariance(geom, rho_frames, param_arr[1]);

  const IntVect box = geom.Domain().length();
  PrintMassConservation(hydrovs_frames, var_names, box[0], radius*box[0]);
  #endif  // **********************************   END MICRO for POST_PROCESS_DFT  ****************************

}

/*
Print() << "Start Fourier Transform ...\n";
  int Nf1 = 10000; int Nf2 = 10300;
  int Nf_total = (Nf2-Nf1)/plot_int + 1;
  printf("Total %d frames in ensemble average\n", Nf_total);
  MultiFab rho_frames(ba, dm, Nf_total, nghost);
  LoadSetOfMultiFabs(plot_file_root, rho_frames, Nf1, Nf2, plot_int, 0, false, Ndigits); // component 0;

  LoadSingleMultiFab(rho_eq_file, rho_eq);

  Function3DAMReX func_rho_eq(rho_eq, geom);
  RealVect vec_com = {0., 0., 0.};  
  getCenterOfMass(vec_com, func_rho_eq);
  printf("Center Of Mass for equilibrium state solution rho_f: (%f,%f,%f)\n", vec_com[0], vec_com[1], vec_com[2]);
  // Fitting equilibrirum droplet radius; \frac{1}{2}\left(1+\tanh\frac{R-\left|\bm{r}-\bm{r}_{0}\right|}{\sqrt{2W}}\right)
  Array<Real, 3> param_arr = fittingDropletParams(func_rho_eq, 20, 0.005);  // last 20 steps ensemble mean, relative error < 0.005 bound;
  printf("fitting parameters for equilibrium density rho: (W0=%f, R0=%f)\n", param_arr[0], param_arr[1]);

*/