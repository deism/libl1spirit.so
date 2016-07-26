

#include "cuda_macros.h"
#include "util.h"

namespace pocs_cuda {

void
ifft_stack (float *S, int M, int N, int C)
{
	cufftHandle plan;
	int c;

	cufft (Plan2d (&plan, N, M, CUFFT_C2C));

	for (c=0; c<C; c++){
		cufft (ExecC2C
			(plan,
			 (cufftComplex*)(S + c*2*M*N),
			 (cufftComplex*)(S + c*2*M*N),
			 CUFFT_INVERSE));
	}

	scale (S, C*M*N, 1.0f/sqrtf(M*N), 0);

	cufft (Destroy (plan));
}

};
