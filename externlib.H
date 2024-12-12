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

Real func3D_test(int x, int y, int z, int n){
    return x*x + y*y + z*z + n;
}

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
    dm_, Real (*func) (int, int, int, int), int ncomp_, int ngrow_);

        Function3DAMReX(const MultiFab& mfab, const Geometry& geom_) {
            func_mfab.FabArray<FArrayBox>::ParallelCopy(mfab);
            ba = mfab.boxArray();
            geom = geom_;
            dm = mfab.DistributionMap();
            ncomp = mfab.nComp();
            ngrow = mfab.nGrow();
        }

        // Accessors for member variables, cannot MODIFY them;
        // In this way, BoxArray ba_func = func3D.getBoxArray(); [ba_func] will be a new object which violates the purpose of "const"
        //  BoxArray getBoxArray() const { return ba; }  
        const BoxArray& getBoxArray() const { return ba; }
        const Geometry& getGeometry() const { return geom; }
        const DistributionMapping& getDistributionMapping() const { return dm; }
        const int getnComp() const { return ncomp; }
        const int getnGrow() const { return ngrow; }
        
        // Modify elementary properties;
        void setBoxArray(const BoxArray& ba_) { ba = ba_; }
        void setDistributionMapping(const DistributionMapping& dm_) { dm = dm_; }
        void setnComp(int ncomp_) { ncomp = ncomp_; }
        void setnGrow(int ngrow_) { ngrow = ngrow_; }

        // Modify element at grid point (x,y,z) of component k
        void setElement(int x, int y, int z, int n);
        // Get element at grid point (x,y,z) of component k
        Real getElement(int x, int y, int z, int n);

        // member functions provided for numerical computations
        void integral3D(const Function3DAMReX& func_weight){    // calculate the integral of \int (*this)\cdot func_weight dV
            
        }

};

Function3DAMReX::Function3DAMReX(const BoxArray& ba_, const Geometry& geom_, const DistributionMapping&
    dm_, Real (*func) (int, int, int, int), int ncomp_=1, int ngrow_=0) : ncomp(ncomp_), ngrow(ngrow_){
            
    ba = ba_;
    geom = geom_;
    dm = dm_; 
    ngrow = ngrow_;
    ncomp = ncomp_;

    func_mfab.define(ba, dm, ncomp, ngrow);
    for(amrex::MFIter mfi(func_mfab); mfi.isValid(); ++mfi){
        const amrex::Box& bx = mfi.validbox();
        const amrex::Array4<amrex::Real>& mf_array = func_mfab.array(mfi);

        ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n){
            mf_array(i,j,k,n) = func(i,j,k,n);
        });
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