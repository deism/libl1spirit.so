#ifndef cuda_macros_h
#define cuda_macros_h

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cublas.h>
#include <cufft.h>

#ifdef _MEXIFY_
# include  "mex.h"
# define output(...) mexPrintf(__VA_ARGS__)
#else
# define output(...) fprintf(stderr, __VA_ARGS__)
#endif

class cuda_error_exception {};

extern void initialize_cublas();

#define cuda(call) do{\
	cudaError err = cuda ## call ; \
	if (err != cudaSuccess){\
		output("Cuda error at %s:%d -- %s\n", __FILE__, __LINE__,\
			cudaGetErrorString(err));\
		throw cuda_error_exception();\
	}\
} while (0)

static inline char const *cublas_err_string (cublasStatus err){
	#define docase(X) \
		case CUBLAS_STATUS_##X: return #X;
	switch(err){
		docase(SUCCESS)
		docase(NOT_INITIALIZED)
		docase(ALLOC_FAILED)
		docase(INVALID_VALUE)
		docase(ARCH_MISMATCH)
		docase(MAPPING_ERROR)
		docase(EXECUTION_FAILED)
		docase(INTERNAL_ERROR)
		default: return "Unknown Error Code";
	}
	#undef docase
}

#define cublas(call) do{\
	cublas ## call ; \
	cublasStatus err = cublasGetError(); \
	if (err != CUBLAS_STATUS_SUCCESS){\
		output("Cublas error at %s:%d -- %s\n", __FILE__, __LINE__,\
			cublas_err_string (err));\
		throw cuda_error_exception();\
	}\
} while(0)

static inline char const * cufft_errstring(cufftResult err){
	switch (err){
	  case CUFFT_SUCCESS: return "Success";
	  case CUFFT_INVALID_PLAN: return "Invalid plan";
	  case CUFFT_ALLOC_FAILED: return "Alloc failed";
	  case CUFFT_INVALID_TYPE: return "Invalid type";
	  case CUFFT_INVALID_VALUE: return "Invalid value";
	  case CUFFT_INTERNAL_ERROR: return "Internal error";
	  case CUFFT_EXEC_FAILED: return "Exec failed";
	  case CUFFT_SETUP_FAILED: return "Setup failed";
	  //case CUFFT_SHUTDOWN_FAILED: return "Shutdown failed";
	  case CUFFT_INVALID_SIZE: return "Invalid size";
	  default: return "Unknown error code";
	}
}

#define cufft(call) \
do{\
	cufftResult res = cufft ## call ;\
	if (res != CUFFT_SUCCESS){ \
		output("%s at line %d: %s\n", \
			__FILE__, __LINE__, cufft_errstring(res));\
		throw cuda_error_exception();\
	}\
} while(0)


#define config(M,N,BM,BN,SM) \
	<<< dim3(((M)+(BM)-1)/(BM),((N)+(BN)-1)/(BN)), dim3(BM,BN), SM >>>

#define cuda_sync() do{\
	cuda (ThreadSynchronize());\
	cuda (GetLastError());\
} while(0)

#endif
