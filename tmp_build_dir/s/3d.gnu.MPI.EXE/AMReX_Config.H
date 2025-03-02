#ifndef AMREX_HAVE_NO_CONFIG_H
#define AMREX_HAVE_NO_CONFIG_H
#ifndef AMREX_EXPORT_DYNAMIC
#define AMREX_EXPORT_DYNAMIC 1
#endif
#ifndef BL_USE_MPI
#define BL_USE_MPI 1
#endif
#ifndef AMREX_USE_MPI
#define AMREX_USE_MPI 1
#endif
#ifndef MAX_SPECIES
#define MAX_SPECIES 2
#endif
#ifndef AMREX_GPU_MAX_THREADS
#define AMREX_GPU_MAX_THREADS 0
#endif
#ifndef BL_SPACEDIM
#define BL_SPACEDIM 3
#endif
#ifndef AMREX_SPACEDIM
#define AMREX_SPACEDIM 3
#endif
#ifndef BL_FORT_USE_UNDERSCORE
#define BL_FORT_USE_UNDERSCORE 1
#endif
#ifndef AMREX_FORT_USE_UNDERSCORE
#define AMREX_FORT_USE_UNDERSCORE 1
#endif
#ifndef BL_Darwin
#define BL_Darwin 1
#endif
#ifndef AMREX_Darwin
#define AMREX_Darwin 1
#endif
#ifndef NDEBUG
#define NDEBUG 1
#endif
#ifndef OMPI_SKIP_MPICXX
#define OMPI_SKIP_MPICXX 1
#endif
#ifndef AMREX_USE_HDF5
#define AMREX_USE_HDF5 1
#endif
#ifdef __cplusplus
#endif
#if defined(AMREX_USE_OMP) && !defined(_OPENMP) && !defined(AMREX_USE_HIP)
#error libamrex was built with OpenMP
#endif
#endif
