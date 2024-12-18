#ifndef LBM_EXTERN_
#define LBM_EXTERN_
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include "AMReX_FileIO.H"
#include <AMReX_FabArray.H>
#include "AMReX_FileIO.H"

/*  *****************************************************************************************************
    *************************** Based on AMReX-MultiFab arithmetic operations ***************************
    *****************************************************************************************************
  // The function returns a single scalar (Real) value, which is the sum of the products of corresponding elements in the specified components of x and y;
  Print() << "multifab dot(): " << amrex::MultiFab::Dot(rho_eq, 0, Imfab, 0, 1, 0) << '\n';
  // Returns the sum of component "comp" (0 by default) over the MultiFab; :sum(int comp = 0, bool local = false)
  Print() << "multifab sum() " << rho_eq.sum() << '\n';
  // element-wise scaling amrex::MultiFab::Multiply(MultiFab& dst, const MultiFab& src, int srccomp, int dstcomp, int numcomp, int nghost)
  amrex::MultiFab::Multiply(rho_eq, Imfab, 0, 0, 1, 0);
  Print() << "after multifab Multiply() " << rho_eq.sum() << '\n';
  //Scales the value of each cell in the valid region of each component of the MultiFab by the scalar val
  void amrex::MultiFab::mult(Real val, int nghost=0)
  //Adds the scalar value val to the value of each cell in the specified subregion of the MultiFab which consists of the num_comp components starting at component comp. 
  void amrex::MultiFab::plus(Real val, int 	comp, int num_comp, int nghost=0)
*/



Real func3D_test(Real x, Real y, Real z, int n){
    return 1.;//x; //x*x * y*y * z*z;
}

class Function3DAMReX;  // forward declaration allows you to use pointers (and references) to that class without yet having the whole class definition.

AMREX_GPU_HOST_DEVICE
class Integration{
    public:
        static double*** create_3d_array(int Nx, int Ny, int Nz, Real initVal=1.) {    // by default, all values are initialized to be 1.;
            double*** arr = (double***)malloc(Nx * sizeof(double**));
            for (int i = 0; i < Nx; i++) {
                arr[i] = (double**)malloc(Ny * sizeof(double*));
                for (int j = 0; j < Ny; j++) {
                    arr[i][j] = (double*)malloc(Nz * sizeof(double));
                }
            }
            // Initialize all elements to 1
            for (int i = 0; i < Nx; i++) {
                for (int j = 0; j < Ny; j++) {
                    for (int k = 0; k < Nz; k++) {
                        arr[i][j][k] = 1.0;
                    }
                }
            }
            return arr;
        }

        // Static member function shared with external funcion;
        static Real trapezoid3D(const Function3DAMReX& func3D){
            return 0.;
        }

        static void trapezoid3DWeightTensor(const Box& box, Real**** wPtr){
            IntVect boxSize = box.length();
            int Nx = boxSize[0];    int Ny = boxSize[1];    int Nz = boxSize[2];
            (*wPtr) = create_3d_array(boxSize[0], boxSize[1], boxSize[2]);
            // Corner points, total 8 points
            (*wPtr)[0][0][0] = 1.0/8;
            (*wPtr)[Nx-1][0][0] = 1.0/8;
            (*wPtr)[0][Ny-1][0] = 1.0/8;
            (*wPtr)[Nx-1][Ny-1][0] = 1.0/8;
            (*wPtr)[0][0][Nz-1] = 1.0/8;
            (*wPtr)[Nx-1][0][Nz-1] = 1.0/8;
            (*wPtr)[0][Ny-1][Nz-1] = 1.0/8;
            (*wPtr)[Nx-1][Ny-1][Nz-1] = 1.0/8;

            // Boundary edge points, total 12 edges
            // 4 vertical edges
            for (int k = 1; k < Nz-1; k++) {
                (*wPtr)[0][0][k] = 1.0/4;
                (*wPtr)[Nx-1][0][k] = 1.0/4;
                (*wPtr)[0][Ny-1][k] = 1.0/4;
                (*wPtr)[Nx-1][Ny-1][k] = 1.0/4;
            }
            // 4 x-parallel horizontal edges
            for (int i = 1; i < Nx-1; i++) {
                (*wPtr)[i][0][0] = 1.0/4;
                (*wPtr)[i][0][Nz-1] = 1.0/4;
                (*wPtr)[i][Ny-1][Nz-1] = 1.0/4;
                (*wPtr)[i][Ny-1][0] = 1.0/4;
            }
            // 4 y-parallel horizontal edges
            for (int j = 1; j < Ny-1; j++) {
                (*wPtr)[0][j][0] = 1.0/4;
                (*wPtr)[0][j][Nz-1] = 1.0/4;
                (*wPtr)[Nx-1][j][Nz-1] = 1.0/4;
                (*wPtr)[Nx-1][j][0] = 1.0/4;
            }
            // Interior points of the bounding surfaces, total 6 surfaces
            // Surfaces parallel with xy-plane
            for (int i = 0; i < Nx; i++) {
                for (int j = 0; j < Ny; j++) {
                    if ((*wPtr)[i][j][0] == 1.0) {
                        (*wPtr)[i][j][0] = 1.0/2;
                    }
                    if ((*wPtr)[i][j][Nz-1] == 1.0) {
                        (*wPtr)[i][j][Nz-1] = 1.0/2;
                    }
                }
            }
            // Surfaces parallel with xz-plane
            for (int i = 0; i < Nx; i++) {
                for (int k = 0; k < Nz; k++) {
                    if ((*wPtr)[i][0][k] == 1.0) {
                        (*wPtr)[i][0][k] = 1.0/2;
                    }
                    if ((*wPtr)[i][Ny-1][k] == 1.0) {
                        (*wPtr)[i][Ny-1][k] = 1.0/2;
                    }
                }
            }
            // Surfaces parallel with yz-plane
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    if ((*wPtr)[0][j][k] == 1.0) {
                        (*wPtr)[0][j][k] = 1.0/2;
                    }
                    if ((*wPtr)[Nx-1][j][k] == 1.0) {
                        (*wPtr)[Nx-1][j][k] = 1.0/2;
                    }
                }
            }
        }
};


AMREX_GPU_HOST_DEVICE
class Function3DAMReX{
    private:
        BoxArray ba;
        Geometry geom;               
        DistributionMapping dm;    
        int ncomp;                 // Number of components of the 3D function
        int ngrow;                 // Number of ghost cells

        MultiFab func_mfab;
    public:
        Function3DAMReX(BoxArray ba_, Geometry geom_, DistributionMapping dm_, Real (*func) (Real, Real, Real, int), int ncomp_, int ngrow_);

        Function3DAMReX(BoxArray ba_, Geometry geom_, DistributionMapping dm_, int ngrow_, Real*** dataPtr); // 1 component total, constructed from 3D array with same size of Box;

        Function3DAMReX(const MultiFab& mfab, const Geometry& geom_) {
            ba = mfab.boxArray();
            geom = geom_;
            dm = mfab.DistributionMap();
            ncomp = mfab.nComp();
            ngrow = mfab.nGrow();

            func_mfab.define(ba, dm, ncomp, ngrow); 
            // MUST FIRST define the MultiFab with same DistributionMapping, otherwise dest mfab has empty [dm] property; SAME for boxArray;
            func_mfab.ParallelCopy(mfab);
        }

        // Accessors for member variables, cannot MODIFY them;
        // In this way, BoxArray ba_func = func3D.getBoxArray(); [ba_func] will be a new object which violates the purpose of "const"
        //  BoxArray getBoxArray() const { return ba; }  
        // providing two choices: return const or non-const reference of MultiFab members 
        const MultiFab& getMultiFab() const { return func_mfab; }   // Call it by using "CONST reference" to void using the deleted constructor of MultiFab;
        MultiFab& getMultiFab() { return func_mfab; }
        
        const BoxArray getBoxArray() const { return ba; }
        const Geometry getGeometry() const { return geom; }
        const DistributionMapping getDistributionMapping() const { return dm; }
        const int getnComp() const { return ncomp; }
        const int getnGrow() const { return ngrow; }
        
        // Modify elementary properties;
        void setBoxArray(const BoxArray& ba_) { ba = ba_; }
        void setGeometry(const Geometry& geom_) { geom = geom_; }
        void setDistributionMapping(const DistributionMapping& dm_) { dm = dm_; }
        void setnComp(int ncomp_) { ncomp = ncomp_; }
        void setnGrow(int ngrow_) { ngrow = ngrow_; }

        // ******Ghost layers are not included by default******;
        // Modify element at grid point (x,y,z) of component k
        void setElement(Real val, int x, int y, int z, int n);
        // Get element at grid point (x,y,z) of component k
        Real getElement(int x, int y, int z, int n);

        // *********** Basic ELEMENT-WISE arithmetic operation ***********
        /* cannot add "const" at the ending, in that case the member variable [func_mfab] declared const, you cannot bind its Array4<const amrex::Real> result
        to a variable of type Array4<amrex::Real> or const Array4<amrex::Real> */
        void add(const Function3DAMReX& func3D);
        void add(const Real scalar);    
        void mult(const Function3DAMReX& func3D);
        void mult(const Real scalar);
        /*  Return the sum of component "compIdx" (-1 by default) over the MultiFab; if compIdx<0, then summation over all components
        are returned */
        Real sum(const int compIdx);

        // member functions provided for numerical computations
        /* This function is build on [integral3D] without multiplicand function, which consumes double data memory size when calling constructor; 
            Both functions DO NOT modify the original data  */
        Real integral3D(const Function3DAMReX& func_multiplicand, Real*** weightTensor=NULL);    // calculate the integral of \int (*this)\cdot func_weight dV

        Real integral3D(Real*** weightTensor=NULL);

        void fillPeriodicBC(){  func_mfab.FillBoundary(geom.periodicity()); }
        
        void convertNodal2Cell();
        void convertCell2Nodal(bool ifPeriodic);

        // Remove Overlapping Regions for nodal-centered data; 
        //  This function WILL MODIFY original boxArray passed in for the constructor; 
        //  i.e., when calling "Function3DAMReX(BoxArray ba_, ...)", though ba_ is passed in by value, it will be affected by this function;
        void makeDivision();    
        void invmakeDivision(); // inverse operation of function makeDivision();

};

Function3DAMReX::Function3DAMReX(BoxArray ba_, Geometry geom_, DistributionMapping
    dm_, int ngrow_, Real*** dataPtr){
    
    ba = ba_;
    geom = geom_;
    dm = dm_;
    ngrow = ngrow_;
    ncomp = 1;
    func_mfab.define(ba, dm, ncomp, ngrow); // Allocating multifab object;
    for(amrex::MFIter mfi(func_mfab); mfi.isValid(); ++mfi){
        const amrex::Box& bx = mfi.validbox();
        amrex::Array4<amrex::Real> const& mf_array = func_mfab.array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k){
            mf_array(i,j,k,0) = dataPtr[i][j][k];
        });
    }
}

Function3DAMReX::Function3DAMReX(BoxArray ba_, Geometry geom_, DistributionMapping
    dm_, Real (*func) (Real, Real, Real, int), int ncomp_=1, int ngrow_=0) : ncomp(ncomp_), ngrow(ngrow_){
            
    ba = ba_;
    geom = geom_;
    dm = dm_; 
    ngrow = ngrow_;
    ncomp = ncomp_;
    //  IndexType typ({AMREX_D_DECL(1,1,1)}); // nodal
    // ************     to be consistent with Trapezoid formula     ************
    const IntVect box = geom.Domain().length();
    RealBox real_box = geom.ProbDomain();
    Real Lx = real_box.hi(0) - real_box.lo(0);
    Real Ly = real_box.hi(1) - real_box.lo(1);
    Real Lz = real_box.hi(2) - real_box.lo(2);
    IndexType itype = ba.ixType(); // using Box or BoxArray;
    Print() << itype[0] << '-' << itype[1] << '\n';
    Array<Real, 3> cell_size;
    if(itype.cellCentered()){
        Print() << "cell centered\n";
        cell_size = {Lx/box[0], Ly/box[1], Lz/box[2]};
        //cell_size = {Lx/(box[0]-1), Ly/(box[1]-1), Lz/(box[2]-1)};
    }else if(itype.nodeCentered()){
        cell_size = {Lx/(box[0]-1), Ly/(box[1]-1), Lz/(box[2]-1)};
        Print() << "nodal centered\n";
    }
    
    func_mfab.define(ba, dm, ncomp, ngrow); // Allocating multifab object;
    for(amrex::MFIter mfi(func_mfab); mfi.isValid(); ++mfi){
        const amrex::Box& bx = mfi.validbox();
        amrex::Array4<amrex::Real> const& mf_array = func_mfab.array(mfi);

        ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n){
            //mf_array(i,j,k,n) = func(i*cell_size[0],j*cell_size[1],k*cell_size[2],n);
            mf_array(i,j,k,n) = func((i+0.5*(1-itype[0]))*cell_size[0],(j+0.5*(1-itype[1]))*cell_size[1],(k+0.5*(1-itype[2]))*cell_size[2],n);
            //printf("Nodal(%d,%d,%d) = %f\t",i,j,k,mf_array(i,j,k,n));
        });
    }
}

/*
    copy constructor of 'Function3DAMReX' is implicitly deleted because field 'func_mfab' has a deleted copy constructor
        MultiFab func_mfab;
*/
void Function3DAMReX::add(const Function3DAMReX& func3D) {
    const MultiFab& mfab_add = func3D.getMultiFab();

    for(amrex::MFIter mfi(func_mfab); mfi.isValid(); ++mfi){
        const amrex::Box& bx = mfi.validbox();
        const amrex::Array4<amrex::Real>& mf_array = func_mfab.array(mfi);
        /* function to be added; Since it is declared to be const, so here use type "Array4<const amrex::Real>" instead;
        see comments in class definition */
        const amrex::Array4<const amrex::Real>& mf_array_add = mfab_add.array(mfi);   
        ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n){
            mf_array(i,j,k,n) = mf_array(i,j,k,n) + mf_array_add(i,j,k,n);
        });
    }
}

void Function3DAMReX::add(const Real scalar){
    func_mfab.plus(scalar, 0, ncomp);
}

// overlapping points in nodal-centered data will NOT affect the amrex::Multiply operations;
void Function3DAMReX::mult(const Function3DAMReX& func3D){
    const MultiFab& mfab_mult = func3D.getMultiFab();
    amrex::MultiFab::Multiply(func_mfab, mfab_mult, 0, 0, ncomp, 0); // using library provided by AMReX;
}

void Function3DAMReX::mult(const Real scalar){
    func_mfab.mult(scalar);
}

Real Function3DAMReX::sum(const int compIdx){
    IndexType itype = ba.ixType();
    if(itype.nodeCentered()){ // remove overlapping grid points' effect
    //  Basic idea is:  remove overlaps & restore the original amrex data structure;
        //Print() << ba << '\n';
        this->makeDivision();
        //Print() << ba << '\n';
        //Print() << func_mfab.boxArray() << '\n';
    }else{
        if(!itype.cellCentered()){
            throw runtime_error("sum() only supports either nodal-centered or cell-centered data!\n");
        }
    }

    Real sumVal = 0.;
    if(compIdx>=0){
        for (MFIter mfi(func_mfab); mfi.isValid(); ++mfi) {
            const Box& valid_box = mfi.validbox();
            Array4<Real> const& nodal_arr = func_mfab.array(mfi);
            ParallelFor(valid_box, [&] AMREX_GPU_DEVICE(int i, int j, int k){     
                printf("Nodal(%d,%d,%d)=%f\t",i,j,k,nodal_arr(i,j,k,compIdx));
                sumVal = sumVal + nodal_arr(i,j,k,compIdx);
            });
        }
    }else{
        for (MFIter mfi(func_mfab); mfi.isValid(); ++mfi) {
            const Box& valid_box = mfi.validbox();
            Array4<Real> const& nodal_arr = func_mfab.array(mfi);
            ParallelFor(valid_box, ncomp, [&] AMREX_GPU_DEVICE(int i, int j, int k, int n){     
                sumVal = sumVal + nodal_arr(i,j,k,n);
            });
        }
    }
    
    if(itype.nodeCentered()){ // recover overlapping grid points' effect if possible;
        this->invmakeDivision();
    }
    Print() << "ba in sum(): " << func_mfab.boxArray() << '\n';
    Print() << "sum in sum(): " << sumVal << '\n';
    return sumVal;
}

void Function3DAMReX::setElement(Real val, int x, int y, int z, int n){
    for (MFIter mfi(func_mfab); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        FArrayBox& fab = func_mfab[mfi];
        if (bx.contains(IntVect(x, y, z))) {
            fab(IntVect(x, y, z), n) = val;
            break;
        }
    }
}

Real Function3DAMReX::getElement(int x, int y, int z, int n){
    Real val = -1e-30;
    for (MFIter mfi(func_mfab); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        FArrayBox& fab = func_mfab[mfi];
        auto fab_array = fab.array();
        if (bx.contains(IntVect(x, y, z))) {
            val = fab_array(x,y,z,n);
            break;
        }
    }
    return val;
}

Real Function3DAMReX::integral3D(const Function3DAMReX& func_multiplicand, Real*** weightTensor){
    // First compare box size if they are same;
    // Build copy for function itself to avoid directly modifying the data during integral calculations;
    Function3DAMReX func_copy(func_mfab, geom); 
    
    Real*** wt = weightTensor;
    if(wt == NULL){
        Integration::trapezoid3DWeightTensor(geom.Domain(), &wt);
    }
    func_copy.mult(func_multiplicand);
    Real intVal = func_copy.integral3D(wt);
    return intVal;
}

Real Function3DAMReX::integral3D(Real*** weightTensor){
    Function3DAMReX func_copy(func_mfab, geom);
    IndexType itype = ba.ixType();
    Real*** wt = weightTensor;
    if(itype.nodeCentered()){
        if(wt == NULL){
            Print() << "weight matrix initialized" << '\n';
            Integration::trapezoid3DWeightTensor(geom.Domain(), &wt);
        }
        Function3DAMReX func_weight(ba, geom, dm, ngrow, wt); // cannot use reference here;
        func_copy.mult(func_weight);    //MultiFab& func_mfab_copy = func_copy.getMultiFab();
    }
    //WriteMultiFabToPlainText('z', weight_mfab, "./Deb");
    
    const IntVect box = geom.Domain().length();
    RealBox real_box = geom.ProbDomain();
    Real Lx = real_box.hi(0) - real_box.lo(0);
    Real Ly = real_box.hi(1) - real_box.lo(1);
    Real Lz = real_box.hi(2) - real_box.lo(2);
    //Print() << itype << '-' << box << '\n';
    Array<Real, 3> cell_size;
    if(itype.cellCentered()){
        //cell_size = {Lx/(box[0]-1), Ly/(box[1]-1), Lz/(box[2]-1)};
        cell_size = {Lx/box[0], Ly/box[1], Lz/box[2]};
    }else if(itype.nodeCentered()){
        cell_size = {Lx/(box[0]-1), Ly/(box[1]-1), Lz/(box[2]-1)};
    }
    Real cell_volume = cell_size[0]*cell_size[1]*cell_size[2];
    Print() << "Cell Size: " << cell_size << '\n';
    Real val = func_copy.sum(0);
    // Or implemented by definition
    /*Real val = 0.;
    for(amrex::MFIter mfi(func_mfab); mfi.isValid(); ++mfi){
        const amrex::Box& bx = mfi.validbox();
        amrex::Array4<amrex::Real> const& mf_array = func_mfab.array(mfi);

        ParallelFor(bx, [&] AMREX_GPU_DEVICE(int i, int j, int k){
            val += mf_array(i,j,k,0)*wt[i][j][k];
        });
    }*/
    return val*cell_volume;
}

void Function3DAMReX::convertNodal2Cell(){
    // MultiFab copy for original data; By default it is nodal-centered;
    MultiFab func_mfab_copy;
    func_mfab_copy.define(ba, dm, ncomp, ngrow);
    func_mfab_copy.ParallelCopy(func_mfab);

    // convert(func_mfab.boxArray(), IntVect::TheNodeVector()); // does not change dm;
    // !!!!!!!!!!!!!!!  UPDATING boxArray & Geometry    !!!!!!!!!!!!!!!
    ba = convert(ba, IntVect::TheCellVector());
    // Get the physical domain (RealBox)
    const RealBox& physical_domain = geom.ProbDomain();
    const Array<int, AMREX_SPACEDIM> is_periodic = {geom.isPeriodic(0), geom.isPeriodic(1), geom.isPeriodic(2)};
    geom.define(ba.minimalBox(), physical_domain, CoordSys::cartesian, is_periodic);

    func_mfab.define(ba, dm, ncomp, ngrow);
    Print() << "node to cell boxArray: " << func_mfab.boxArray() << '\n';
    for (MFIter mfi(func_mfab); mfi.isValid(); ++mfi) {
        const Box& valid_box = mfi.validbox();
    
        Array4<Real> const& node_arr = func_mfab_copy.array(mfi); // This function include ghost layers by default;
        Array4<Real> const& cell_arr = func_mfab.array(mfi);

        ParallelFor(valid_box, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n){
            cell_arr(i, j, k, n) = 0.125 * (node_arr(i, j, k, n) + node_arr(i + 1, j, k, n) + node_arr(i, j + 1, k, n) +
                node_arr(i, j, k + 1, n) + node_arr(i + 1, j + 1, k, n) + node_arr(i + 1, j, k + 1, n) + node_arr(i, j + 1, k + 1, n) + node_arr(i + 1, j + 1, k + 1, n));
        });
    }
}

void Function3DAMReX::convertCell2Nodal(bool ifPeriodic=true){
    // MultiFab copy for original data; By default it is cell-centered;
    MultiFab func_mfab_copy;
    func_mfab_copy.define(ba, dm, ncomp, ngrow+1);  // to ensure the cell-centered data will include ghost cells;

    func_mfab_copy.ParallelCopy(func_mfab);
    func_mfab_copy.FillBoundary(geom.periodicity());

    // !!!!!!!!!! UPDATING boxArray & Geometry  !!!!!!!!!!!!!!!
    ba = convert(ba, IntVect::TheNodeVector());
    // Get the physical domain (RealBox)
    const RealBox& physical_domain = geom.ProbDomain();
    const Array<int, AMREX_SPACEDIM> is_periodic = {geom.isPeriodic(0), geom.isPeriodic(1), geom.isPeriodic(2)};
    geom.define(ba.minimalBox(), physical_domain, CoordSys::cartesian, is_periodic);

    // convert(func_mfab.boxArray(), IntVect::TheNodeVector()); // does not change dm;
    func_mfab.define(ba, dm, ncomp, ngrow);
    //Print() << func_mfab.boxArray() << "ghost layer = " << ngrow << '\n';

    for (MFIter mfi(func_mfab); mfi.isValid(); ++mfi) {
        //const Box& node_box = amrex::convert(mfi.validbox(), IntVect::TheNodeVector()); // here the input params are constant reference &;
        const Box& valid_box = mfi.validbox();
        const amrex::Box& ghost_box = amrex::grow(valid_box, 1); 
    
        Array4<Real> const& cell_arr = func_mfab_copy.array(mfi); // This function include ghost layers by default;
        Array4<Real> const& node_arr = func_mfab.array(mfi);

        ParallelFor(valid_box, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n){
            node_arr(i, j, k, n) = 0.125 * (cell_arr(i, j, k, n) + cell_arr(i - 1, j, k, n) + cell_arr(i, j - 1, k, n) +
                cell_arr(i, j, k - 1, n) + cell_arr(i - 1, j - 1, k, n) + cell_arr(i - 1, j, k - 1, n) + cell_arr(i, j - 1, k - 1, n) + cell_arr(i - 1, j - 1, k - 1, n));
        });
    }
}

void Function3DAMReX::makeDivision(){
    IndexType itype = ba.ixType();
    Box domain_nodal = geom.Domain();
    if(itype.nodeCentered()){ // remove overlapping grid points' effect
    //  Basic idea is:  remove overlaps by shrinking the bigEnd points of each box within boxArray;
        for (int i = 0; i < ba.size(); ++i) {
            Box box = ba[i];   // operator [] return temporary object, so CANNOT directly modify the boxes within boxArray;
            // Box& boxref = ba[i]; THIS CANNOT be compiled; Binding non-constant reference to temporary object box;
            // Remove the overlap by shrinking the box on its upper boundaries
            for (int dim = 0; dim < AMREX_SPACEDIM; ++dim) {
                if (box.bigEnd(dim) == domain_nodal.bigEnd(dim)){
                    continue; // Keep domain boundary
                }else{
                    box.growHi(dim, -1);  // Shrink the upper boundary to avoid overlap
                    ba.set(i, box);
                }
            }
        }
    }else{
        throw runtime_error("makeDivision() only supports nodal-centered data!\n");
    }
}

void Function3DAMReX::invmakeDivision(){
    IndexType itype = ba.ixType();
    Box domain_nodal = geom.Domain();
    if(itype.nodeCentered()){ // recover overlapping grid points
        for (int i = 0; i < ba.size(); ++i) {
            Box box = ba[i];   // operator [] return temporary object, so CANNOT directly modify the boxes within boxArray;
            // Box& boxref = ba[i]; THIS CANNOT be compiled; Binding non-constant reference to temporary object box;
            // Remove the overlap by shrinking the box on its upper boundaries
            for (int dim = 0; dim < AMREX_SPACEDIM; ++dim) {
                if (box.bigEnd(dim) == domain_nodal.bigEnd(dim)){
                    continue; // Keep domain boundary
                }else{
                    box.growHi(dim, 1);  // Shrink the upper boundary to avoid overlap
                    ba.set(i, box);
                }
            }
        }
    }else{
        throw runtime_error("makeDivision() only supports nodal-centered data!\n");
    }
}



// These library function are directly converted from python, see "CodingNote_analysis_supp.ipynb" for details; 

double inv_acosh(double x) {
    if (fabs(x) < 710.4) {
        return 1 / cosh(x);
    } else {
        return 0;
    }
}

//  calculate C_n^k, binomial coefficients;
double comb(int n, int k) {
    if (k > n) return 0;
    if (k == 0 || k == n) return 1;
    double result = 1;
    for (int i = 1; i <= k; i++) {
        result *= (double)(n - i + 1) / i;
    }
    return result;
}

double factorial(int n) {
    if (n == 0) return 1;
    double result = 1;
    for (int i = 1; i <= n; i++) {
        result *= i;
    }
    return result;
}

std::vector<Real> getCoefS(int N) {
    std::vector<Real> coefSVec(N, 0.);
    double* A_array = (double*)malloc(N * sizeof(double));
    A_array[0] = 1.0;

    for (int k = 1; k < N; k++) {
        double A2k = 0.0;
        for (int kp = 0; kp < k; kp++) {
            A2k += A_array[kp] * comb(2 * k, 2 * kp);
        }
        A2k = -A2k;
        A_array[k] = A2k;
    }

    double* A_array_prime = (double*)malloc(N * sizeof(double));

    for (int k = 0; k < N; k++) {
        A_array_prime[k] = A_array[k] / factorial(2 * k);
        double Sk = 0.0;
        for (int k1 = 0; k1 <= k; k1++) {
            for (int k2 = 0; k2 <= k - k1; k2++) {
                for (int k3 = 0; k3 <= k - k1 - k2; k3++) {
                    int k4 = k - k1 - k2 - k3;
                    Sk += A_array_prime[k1] * A_array_prime[k2] * A_array_prime[k3] * A_array_prime[k4];
                }
            }
        }
        coefSVec[k] = Sk;
    }

    free(A_array);
    free(A_array_prime);
    
    return coefSVec;
}

// get given fixed size 2D matrix for binomial coefficients; same with function comb(), just for efficiency;
std::vector<std::vector<Real>> getCombNomial(int n_max=4) {
    std::vector<std::vector<Real>> comb_nomial_array;

    for (int i = 0; i <= n_max; i++) {
        std::vector<Real> comb_row(n_max, 0.);
        for(int j=0; j<=i; j++){
            comb_row[j] = comb(i, j);
        }
        comb_nomial_array.push_back(comb_row);
    }
    return comb_nomial_array;
}

//  Total terms of series summation is equal to the size of vector [S_array]; More terms, more accurate;
//  Input [n] can only be 2, 3 & 4 three cases; 
Real integral_func2_series(int n, double d, double c, std::vector<std::vector<Real>> comb_nomial_array, std::vector<Real> S_array, double delta=0.4) {

    long double int_val = 0.0;
    const long double factor = 16.0/6.0;
    long double ratio_cd = c/d;
    int N = S_array.size();

    for (int k = 0; k < N; k++) {
        long double exponent = (2*k+4)*d;
        long double inv_exponent = 1.0/exponent;
        long double inv_exponent2 = pow(inv_exponent, 2.);
        long double inv_exponent3 = pow(inv_exponent, 3.);
        long double inv_exponent4 = pow(inv_exponent, 4.);
        long double inv_exponent5 = pow(inv_exponent, 5.);

        long double val_term1, val_term2;

        // different cases for n = 2, 3 & 4
        if (n == 4) {
            val_term1 = (inv_exponent*pow(ratio_cd-delta, 4.) - 4*inv_exponent2*pow(ratio_cd-delta, 3.) + 12*inv_exponent3*pow(ratio_cd-delta, 2.) - 24*inv_exponent4*(ratio_cd-delta)
                         + 24*inv_exponent5)*exp(-exponent*delta) - 24*inv_exponent5*exp(-(2*k+4)*c);
            val_term2 = (inv_exponent*pow(ratio_cd+delta, 4.) + 4*inv_exponent2*pow(ratio_cd+delta, 3.) + 12*inv_exponent3*pow(ratio_cd+delta, 2.) + 24*inv_exponent4*(ratio_cd+delta)
                         + 24*inv_exponent5)*exp(-exponent*delta);
        } else if (n == 3) {
            val_term1 = (inv_exponent*pow(ratio_cd-delta, 3.) - 3*inv_exponent2*pow(ratio_cd-delta, 2.) + 6*inv_exponent3*(ratio_cd-delta) - 6*inv_exponent4
                         )*exp(-exponent*delta) + 6*inv_exponent4*exp(-(2*k+4)*c);
            val_term2 = (inv_exponent*pow(ratio_cd+delta, 3.) + 3*inv_exponent2*pow(ratio_cd+delta, 2.) + 6*inv_exponent3*(ratio_cd+delta) + 6*inv_exponent4)*exp(-exponent*delta);
        } else if (n == 2) {
            val_term1 = (inv_exponent*pow(ratio_cd-delta, 2.) - 2*inv_exponent2*(ratio_cd-delta) + 2*inv_exponent3)*exp(-exponent*delta)
                - 2*inv_exponent3*exp(-(2*k+4)*c);
            val_term2 = (inv_exponent*pow(ratio_cd+delta, 2.) + 2*inv_exponent2*(ratio_cd+delta) + 2*inv_exponent3)*exp(-exponent*delta);
        }
        //Print() << val_term1 + val_term2 << '\n'; // same with python code
        if(k%2==0){
            int_val += factor*(k+1)*(k+2)*(k+3)*(long double)(val_term1 + val_term2);
        }else{
            int_val = int_val - factor*(k+1)*(k+2)*(k+3)*(long double)(val_term1 + val_term2);
        }
        //Print() << int_val << '\n'; //  DIFFER with python code ??
        long double val_n = 0.0;
        for (int l = 0; l <= n; l++) {
            long double val_tmp = pow(delta, 2*k+l+1) - pow(-delta, 2*k+l+1);
            val_n += comb_nomial_array[n][l]*pow(c/d, n-l)*val_tmp/(2*k+l+1);
        }
        int_val += S_array[k]*pow(d, 2*k)*val_n;
    }

    return int_val;
}


double integral_func3_series(int n, double c, int N=50) {
    double int_val = 0;
    for (int k = 1; k <= N; k++) {
        double k2 = k * k;
        double k3 = k2 * k;
        if (n == 3) {
            int_val += 6 * pow(-1, k+1) * (c/k2 + 1.0/4/k3 * exp(-2*k*c));
        } else if (n == 2) {
            int_val += pow(-1, k) * exp(-2*k*c)/k2 + pow(-1, k+1) * 2.0/k2;
        }
    }
    int_val += 2 * pow(c, n);
    return int_val;
}

// here n is restricted to only two cases: n = 2 & n=3
double integral_func1_series(int n, double a, int N=100) {
    double int_val = 0.0;
    if (n == 3) {
        double val_sum_1 = 0.0;
        double val_sum_2 = 0.0;
        for (int k = 1; k <= N; k++) {
            double k2 = k * k;
            val_sum_1 += pow(-1, k+1)/k2 * exp(-2*k*a);
            val_sum_2 += pow(-1, k+1)/k2;
        }
        int_val = 1.5 * val_sum_1 - 3 * val_sum_2 - 3 * a * a;
    } else {
        int_val = -a - log(2) - log(cosh(a));
    }
    return int_val;
}

#endif


/**************         Examples using integral3D      ************    
  Function3DAMReX func3D(ba, geom, dm, func3D_test);
  Function3DAMReX func3D_mfab(rho_eq, geom);
  MultiFab Imfab(ba, dm, 1, nghost);  Imfab.setVal(1.);
  MultiFab& func_mfab = func3D.getMultiFab();
  const MultiFab& func_mfab = func3D.getMultiFab();
  Function3DAMReX func3D_weight(Imfab, geom);
  Print() << func3D.getElement(1,1,1,0) << '\n';

  Print() << func3D_mfab.getElement(1,1,1,0) << '\n';
  func3D_mfab.add(func3D_weight);
  Print() << func3D_mfab.getElement(1,1,1,0) << '\n';

  // The function returns a single scalar (Real) value, which is the sum of the products of corresponding elements in the specified components of x and y;
  Print() << "multifab dot(): " << amrex::MultiFab::Dot(rho_eq, 0, Imfab, 0, 1, 0) << '\n';
  Print() << "multifab sum() " << rho_eq.sum() << '\n';
  // element-wise scaling amrex::MultiFab::Multiply(MultiFab& dst, const MultiFab& src, int srccomp, int dstcomp, int numcomp, int nghost)
  amrex::MultiFab::Multiply(rho_eq, Imfab, 0, 0, 1, 0);
  Print() << "after multifab Multiply() " << rho_eq.sum() << '\n';
  func3D_mfab.mult(2.);
  Print() << "rho_eq scaled by values using mult(): " << func3D_mfab.getMultiFab().sum() << '\n';
  func3D_mfab.add(1.);
  Print() << "rho_eq added by values using add(): " << func3D_mfab.getMultiFab().sum() << "(above result +" << nx << "^3=" << pow(nx, 3.) << ")\n";

  Print() << "Integral = " << func3D.integral3D() << " & analytical result = " << 1./27. << '\n';
  // func3D = Function3DAMReX(ba, geom, dm, func3D_test);
  Print() << "Integral = " << func3D.integral3D(func3D_weight) << " & analytical result = " << 1./27. << '\n';
*******************************************************/

/************** understand periodic boundary condition in AMReX ************** 
  // default grid parameters; examples of 4*4*4 cells total, divided into FArrayBox of size 2*2*2; 
  int nx = 4;
  int max_grid_size = 2;
  IntVect dom_lo(0, 0, 0);
  IntVect dom_hi(nx-1, nx-1, nx-1);
  Array<int,3> periodicity({1,1,1});
  Box domain(dom_lo, dom_hi);
  RealBox real_box({0.,0.,0.},{1.,1.,1.});
  
  Geometry geom(domain, real_box, CoordSys::cartesian, periodicity);
  BoxArray ba(domain);
  ba.maxSize(max_grid_size);
  DistributionMapping dm(ba);

  //    test function to show periodicity;  Thus RealBox has interval = (1./3, 1./3, 1./3);
  Real func3D_test(Real x, Real y, Real z, int n){
    return (x-0.5)*(x-0.5)+(y-0.5)*(y-0.5)+(z-0.5)*(z-0.5);   //x*x * y*y * z*z;
  }   

  
  Function3DAMReX func3D(ba, geom, dm, func3D_test, 1, 1);
  func3D.fillPeriodicBC();
  Function3DAMReX func3D_mfab(rho_eq, geom);
  MultiFab Imfab(ba, dm, 1, nghost);  Imfab.setVal(1.);
  MultiFab& func_mfab = func3D.getMultiFab();
  Function3DAMReX func3D_weight(Imfab, geom);

  const IntVect box = geom.Domain().length();
  MultiFab& test_mfab_cell = func3D.getMultiFab();
  MultiFab test_mfab_nodal(convert(test_mfab_cell.boxArray(), IntVect::TheNodeVector()), dm, test_mfab_cell.nComp(), 0);
  for (MFIter mfi(test_mfab_cell); mfi.isValid(); ++mfi) {
    const Box& node_box = amrex::convert(mfi.validbox(), IntVect::TheNodeVector()); // here the input params are constant reference &;
    const Box& valid_box = mfi.validbox();
    const amrex::Box& ghost_box = amrex::grow(valid_box, 1);
    
    Array4<Real> const& cell_arr = test_mfab_cell.array(mfi);
    Array4<Real> const& node_arr = test_mfab_nodal.array(mfi);

    Print() << cell_arr << " ghost: " << test_mfab_cell.nGrow() << '\n';
    Print() << node_arr << '\n';
    Print() << ghost_box << '\n';
    //  Periodicity Check
    ParallelFor(ghost_box, 1, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n){
      int i_shift = i;  int j_shift = j;  int k_shift = k;
      //Print() << "Outside of geometry ghost cells check:" << '\n';
      if(i_shift < 0){
        i_shift = i_shift + box[0];
      }
      if(j_shift < 0){
        j_shift = j_shift + box[1];
      }
      if(k_shift < 0){
        k_shift = k_shift + box[2];
      }
      if(i_shift >= box[0]){
        i_shift = i_shift - box[0];
      }
      if(j_shift >= box[1]){
        j_shift = j_shift - box[1];
      }
      if(k_shift >= box[2]){
        k_shift = k_shift - box[2];
      }
      printf("Cell(%d,%d,%d) - Cell(%d,%d,%d) = %f\n", i_shift, j_shift, k_shift, i, j, k, 
      (pow(i_shift*1./3.-0.5, 2.)+pow(j_shift*1./3.-0.5, 2.)+pow(k_shift*1./3.-0.5, 2.)) - cell_arr(i, j, k, n));
      if(!valid_box.contains(IntVect(i, j, k)) && geom.Domain().contains(IntVect(i, j, k))){
        printf("Inner ghost Cell(%d,%d,%d) - analytical_val = %f\n", i, j, k, cell_arr(i, j, k, n) - (pow(i*1./3.-0.5, 2.)+pow(j*1./3.-0.5, 2.)+pow(k*1./3.-0.5, 2.)));
      }
    });


    /*    test function to show linear interpolation;  Thus RealBox has interval = (1./3, 1./3, 1./3);
    Real func3D_test(Real x, Real y, Real z, int n){
        return x+y+z;   //x*x * y*y * z*z;
    }   */
    /*ParallelFor(node_box, 1, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n){
      node_arr(i, j, k, n) = 0.125 * (cell_arr(i, j, k, n) + cell_arr(i - 1, j, k, n) + cell_arr(i, j - 1, k, n) +
                cell_arr(i, j, k - 1, n) + cell_arr(i - 1, j - 1, k, n) + cell_arr(i - 1, j, k - 1, n) + cell_arr(i, j - 1, k - 1, n) + cell_arr(i - 1, j - 1, k - 1, n));
      printf("Nodal(%d,%d,%d) = %f\tCell(%d,%d,%d) = %f\n", i, j, k, node_arr(i, j, k, n), i, j, k, cell_arr(i, j, k, n));
    });
  }
*/

/***********    example for convert Cell to Nodal ***********
  Function3DAMReX func3D(ba, geom, dm, func3D_test, 1, 1);
  Function3DAMReX func3D2(ba, geom, dm, func3D_test, 1, 1);
  func3D2.fillPeriodicBC(); MultiFab& func_mfab_cell = func3D2.getMultiFab(); func_mfab_cell.FillBoundary(geom.periodicity());

  func3D.fillPeriodicBC();
  func3D.convertCell2Nodal();

  Function3DAMReX func3D_mfab(rho_eq, geom);
  MultiFab Imfab(ba, dm, 1, nghost);  Imfab.setVal(1.);
  
  Function3DAMReX func3D_weight(Imfab, geom);

  const IntVect box = geom.Domain().length();
  MultiFab& test_mfab_cell2nodal = func3D.getMultiFab();
  //MultiFab test_mfab_nodal(convert(test_mfab_cell.boxArray(), IntVect::TheNodeVector()), dm, test_mfab_cell.nComp(), 0);

  IndexType itype = func3D.getMultiFab().boxArray().ixType();
  Print() << itype << '\n';

  for (MFIter mfi(test_mfab_cell2nodal); mfi.isValid(); ++mfi) {
    //const Box& node_box = amrex::convert(mfi.validbox(), IntVect::TheNodeVector()); // here the input params are constant reference &;
    const Box& node_box = mfi.validbox();

    Array4<Real> const& node_arr = test_mfab_cell2nodal.array(mfi); 
    Array4<Real> const& cell_arr = func_mfab_cell.array(mfi);

    ParallelFor(node_box, 1, [&] AMREX_GPU_DEVICE(int i, int j, int k, int n){
      printf("Nodal(%d,%d,%d) = %f\tCell value(%d,%d,%d)(%d,%d,%d)(%d,%d,%d)(%d,%d,%d)(%d,%d,%d)(%d,%d,%d)(%d,%d,%d)(%d,%d,%d)= %f, %f, %f, %f, %f, %f, %f, %f\n",
      i,j,k, node_arr(i,j,k,0), i,j,k, i - 1, j, k,i, j - 1, k,i, j, k - 1,i - 1, j - 1, k,i - 1, j, k - 1,i, j - 1, k - 1,i - 1, j - 1, k - 1,
      cell_arr(i, j, k, n) , cell_arr(i - 1, j, k, n) , cell_arr(i, j - 1, k, n) ,
                cell_arr(i, j, k - 1, n) , cell_arr(i - 1, j - 1, k, n) , cell_arr(i - 1, j, k - 1, n) , cell_arr(i, j - 1, k - 1, n) , cell_arr(i - 1, j - 1, k - 1, n));
    });
  }

  ********/

 /*************     seems change the boxArray will not affect the sum() values ???
Print() << '\n' << "SUM: " << func3D2_mfab.sum(0) << '\n';  
  func3D2.makeDivision();
  Print() << "modified nodal center boxarray: " << ba2 << '\n';
  Print() << "modified nodal center boxarray: " << func3D2.getBoxArray() << '\n';
  Print() << "modified nodal center boxarray: " << func3D2.getMultiFab().boxArray() << '\n';
  Print() << '\n' << "SUM: " << func3D2_mfab.sum(0) << '\n';  
  func3D2.makeDivision();
  Print() << "modified nodal center boxarray: " << ba2 << '\n';
  Print() << "modified nodal center boxarray: " << func3D2.getBoxArray() << '\n';
  Print() << "modified nodal center boxarray: " << func3D2.getMultiFab().boxArray() << '\n';
  Print() << '\n' << "SUM: " << func3D2_mfab.sum(0) << '\n'; 
  * **************/