AMREX_HOME ?= ../../amrex
FHDEX_HOME ?= ../../FHDeX


EIGEN_DIR = /Users/dengxiaohui/Documents/Eigen/
CXXFLAGS += -I$(EIGEN_DIR) -std=c++20

DEFINES += -DMAX_SPECIES=2

DEBUG	= FALSE

DIM	= 3

COMP    = g++

USE_MPI   = TRUE
USE_OMP   = FALSE
USE_CUDA  = FALSE
USE_HIP   = FALSE

USE_HDF5=TRUE 
#HDF5_HOME=/usr/local/hdf5

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ./Make.package

include $(FHDEX_HOME)/src_common/Make.package
VPATH_LOCATIONS   += $(FHDEX_HOME)/src_common/
INCLUDE_LOCATIONS += $(FHDEX_HOME)/src_common/

INCLUDE_LOCATIONS += $(EIGEN_DIR)

include $(FHDEX_HOME)/src_analysis/Make.package
VPATH_LOCATIONS   += $(FHDEX_HOME)/src_analysis/
INCLUDE_LOCATIONS += $(FHDEX_HOME)/src_analysis/

include $(AMREX_HOME)/Src/Base/Make.package
include $(AMREX_HOME)/Src/FFT/Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.rules

ifeq ($(findstring cgpu, $(HOST)), cgpu)
  CXXFLAGS += $(FFTW)
endif

ifeq ($(USE_CUDA),TRUE)
  LIBRARIES += -lcufft 
else
  LIBRARIES += $(shell pkg-config --libs fftw3 fftw3f) #-lfftw3f #-lfftw3_mpi 
  LIBRARIES += $(shell pkg-config --libs hdf5)
endif
