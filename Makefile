#
# Makefile for the demo of the 3D l1-SPIRiT recon
#
# To build the Cuda-based GPU reconstruction, build the target: l1spirit
#
# Mark Murphy 2010


# If the required Cuda, ACML, and FFTW libraries are not installed in
# standard places, you must change these variables to point to the appropriate
# locations. Note also that if you are not on a 64-bit machine, you will
# need to adjust the appropriate linker flags.

fftw := /usr
cuda := /usr/local/cuda
mkl  := /opt/intel/mkl
arma := /usr

mkl_cppflags := -I$(mkl)/include
mkl_ldflags := -L$(mkl)/lib/intel64 -lmkl_intel_ilp64 -lmkl_core -lmkl_gnu_thread

arma_cppflags := -I$(arma)/include
# link to standard BLAS and LAPACK 
arma_ldflags := -L$(arma)/lib -larmadillo   

fftw_cppflags := -I$(fftw)/include
fftw_ldflags  := -L$(fftw)/lib/x86_64-linux-gnu/ -lfftw3f -lfftw3f_threads

cuda_cppflags := -I$(cuda)/include
cuda_ldflags  :=  -L$(cuda)/lib64 -lcudart -lcuda -lcublas -lcufft


cppflags :=  $(fftw_cppflags) $(mkl_cppflags) $(arma_cppflags) 
ldflags  :=  $(fftw_ldflags)  -g -lbz2 -lrt  -Wl,--no-as-needed $(mkl_ldflags) -lpthread -lm -ldl

cxxflags := -fopenmp -Wall -g  -fPIC  -DARMA_64BIT_WORD -DARMA_DONT_USE_WRAPPER  -DMKL_ILP64 -m64

nvcc := $(cuda)/bin/nvcc --compiler-options '-fPIC'
cxx := g++
debug := 0
ifeq ($(debug),1)
  cxxflags := -O0 $(cxxflags)
  nvccflags := -O0 $(nvccflags)
else
  cxxflags := -O3 $(cxxflags)
  nvccflags := -O3 $(nvccflags)
endif

#all: l1spirit_omp l1spirit_gpu

#l1spirit_omp: driver.o l1spirit3d_nocuda.o calib3d.o pocs-omp.o bz2mat.o shmmat.o
#	$(cxx)  $^ -o $@ $(ldflags)

#l1spirit_gpu: driver.o l1spirit3d_cuda.o calib3d.o pocs-cuda.o pocs-omp.o bz2mat.o shmmat.o
#	$(cxx)  $^ -o $@ $(cuda_ldflags) $(ldflags)

all: libl1spirit_omp.so libl1spirit_gpu.so

libl1spirit_omp.so: l1spirit3d_nocuda.o calib3d.o pocs-omp.o bz2mat.o shmmat.o
	$(cxx) -shared $^ -o $@ $(ldflags)

libl1spirit_gpu.so: l1spirit3d_cuda.o calib3d.o pocs-cuda.o pocs-omp.o bz2mat.o shmmat.o
	$(cxx) -shared $^ -o $@ $(cuda_ldflags) $(ldflags)

pocs-cuda.o: pocs-cuda/pocs_l1spirit2d.cu_o \
             pocs-cuda/soft_threshold.cu_o \
             pocs-cuda/spirit_filter.cu_o \
             pocs-cuda/fwt_stack.cu_o \
             pocs-cuda/iwt_stack.cu_o \
             pocs-cuda/fft_stack.cu_o \
             pocs-cuda/ifft_stack.cu_o \
             pocs-cuda/util.cu_o
	ld -r $^ -o $@

pocs-cuda/%.cu_o: pocs-cuda/%.cu
	$(nvcc) -c $^ -o $@ $(cppflags) $(nvccflags) -I. -Ipocs-cuda 

pocs-omp.o: pocs-omp/pocs_l1spirit2d.o \
            pocs-omp/spirit_filter.o \
            pocs-omp/wavelets.o
	ld -r $^ -o $@

pocs-omp/%.o: pocs-omp/%.cc
	$(cxx) -c $^ -o $@ $(cppflags) $(cxxflags) -I. -Ipocs-omp

%.o: %.cc
	$(cxx) -c $^ $(cppflags) $(cxxflags)

l1spirit3d_cuda.o: l1spirit3d.cc
	$(cxx) -o $@ -c $^ $(cuda_cppflags) $(cppflags) $(cxxflags)

l1spirit3d_nocuda.o: l1spirit3d.cc
	$(cxx) -DNO_CUDA -o $@ -c $^ $(cppflags) $(cxxflags)

clean: 
	rm -f *.o pocs-omp/*.o pocs-cuda/*.cu_o *.so

.PHONY: clean
