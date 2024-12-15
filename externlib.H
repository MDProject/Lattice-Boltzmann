#ifndef LBM_EXTERN_
#define LBM_EXTERN_
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
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
    return x*x * y*y * z*z;
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
        Function3DAMReX(const BoxArray& ba_, const Geometry& geom_, const DistributionMapping&
    dm_, Real (*func) (Real, Real, Real, int), int ncomp_, int ngrow_);

        Function3DAMReX(const BoxArray& ba_, const Geometry& geom_, const DistributionMapping&
    dm_, int ngrow_, Real*** dataPtr); // 1 component total, constructed from 3D array with same size of Box;

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

};

Function3DAMReX::Function3DAMReX(const BoxArray& ba_, const Geometry& geom_, const DistributionMapping&
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
            /*if(mf_array(i,j,k,0) != 1){
                Print() << i << "--" << j << "--" << k << '\t';
            }*/
        });
    }
}

Function3DAMReX::Function3DAMReX(const BoxArray& ba_, const Geometry& geom_, const DistributionMapping&
    dm_, Real (*func) (Real, Real, Real, int), int ncomp_=1, int ngrow_=0) : ncomp(ncomp_), ngrow(ngrow_){
            
    ba = ba_;
    geom = geom_;
    dm = dm_; 
    ngrow = ngrow_;
    ncomp = ncomp_;
    // ************     to be consistent with Trapezoid formula     ************
    const IntVect box = geom.Domain().length();
    RealBox real_box = geom.ProbDomain();
    Real Lx = real_box.hi(0) - real_box.lo(0);
    Real Ly = real_box.hi(1) - real_box.lo(1);
    Real Lz = real_box.hi(2) - real_box.lo(2);
    const Array<Real, 3> cell_size = {Lx/(box[0]-1), Ly/(box[1]-1), Lz/(box[2]-1)};
    
    func_mfab.define(ba, dm, ncomp, ngrow); // Allocating multifab object;
    for(amrex::MFIter mfi(func_mfab); mfi.isValid(); ++mfi){
        const amrex::Box& bx = mfi.validbox();
        amrex::Array4<amrex::Real> const& mf_array = func_mfab.array(mfi);

        ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n){
            mf_array(i,j,k,n) = func(i*cell_size[0],j*cell_size[1],k*cell_size[2],n);
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

void Function3DAMReX::mult(const Function3DAMReX& func3D){
    const MultiFab& mfab_mult = func3D.getMultiFab();
    amrex::MultiFab::Multiply(func_mfab, mfab_mult, 0, 0, ncomp, 0); // using library provided by AMReX;
}

void Function3DAMReX::mult(const Real scalar){
    func_mfab.mult(scalar);
}

Real Function3DAMReX::sum(const int compIdx){
    if(compIdx>=0){
        return func_mfab.sum(compIdx);
    }else{
        Real sumVal = 0.;
        for(int n=0; n<ncomp; n++){
            sumVal = sumVal + func_mfab.sum(n);
        }
        return sumVal;
    }
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
    Real*** wt = weightTensor;
    if(wt == NULL){
        Print() << "weight matrix initialized" << '\n';
        Integration::trapezoid3DWeightTensor(geom.Domain(), &wt);
    }
    Function3DAMReX func_weight(ba, geom, dm, ngrow, wt); // cannot use reference here;
    //WriteMultiFabToPlainText('z', weight_mfab, "./Deb");
    
    // ****** The index type in AMReX is Cell-Centered by default, so here we need to modify the cell-size to be consistent with Trapezoid formula ******
    const IntVect box = geom.Domain().length();
    RealBox real_box = geom.ProbDomain();
    Real Lx = real_box.hi(0) - real_box.lo(0);
    Real Ly = real_box.hi(1) - real_box.lo(1);
    Real Lz = real_box.hi(2) - real_box.lo(2);
    const Array<Real, 3> cell_size = {Lx/(box[0]-1), Ly/(box[1]-1), Lz/(box[2]-1)};
    Real cell_volume = cell_size[0]*cell_size[1]*cell_size[2];

    func_copy.mult(func_weight);
    Real val = func_copy.sum(0);
    /*// Or implemented by definition
    Real val = 0.;
    for(amrex::MFIter mfi(func_mfab); mfi.isValid(); ++mfi){
        const amrex::Box& bx = mfi.validbox();
        amrex::Array4<amrex::Real> const& mf_array = func_mfab.array(mfi);

        ParallelFor(bx, [&] AMREX_GPU_DEVICE(int i, int j, int k){
            val += mf_array(i,j,k,0)*wt[i][j][k];
        });
    }*/
    return val*cell_volume;
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


/**************         Examples       ************    
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
*******************************************************/