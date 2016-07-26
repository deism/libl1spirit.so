
#include "cuda_macros.h"
#include "util.h"

namespace pocs_cuda {

__global__ void
_spirit_filter
	(float *rr0, float *ri0, float *kr0, float *ki0, float *xr0, float *xi0,
	 int M, int N, int C)
{
	int const i = threadIdx.x + blockIdx.x*blockDim.x;
	int const j = threadIdx.y + blockIdx.y*blockDim.y;
	float *xrc, *xic, *krc, *kic, *rrc, *ric;
	int cc, c;

	for (cc = 0; cc<C; cc++)
	{
		float xr=0, xi=0, kr=0, ki=0, rr=0, ri=0;

		xrc = xr0;
		xic = xi0;

		krc = kr0 + cc*M*N*C;
		kic = ki0 + cc*M*N*C;

		rrc = rr0 + cc*M*N;
		ric = ri0 + cc*M*N;

		for (c = 0; c<C; c++)
		{
			if (i < M && j < N){
				xr = xrc[i+M*j];
				xi = xic[i+M*j];
				kr = krc[i+M*j];
				ki = kic[i+M*j];

				rr += xr*kr - xi*ki;
				ri += xr*ki + xi*kr;
			}

			krc += M*N;
			kic += M*N;
			xrc += M*N;
			xic += M*N;
		}

		if (i < M && j < N){
			rrc[i+M*j] = rr;
			ric[i+M*j] = ri;
		}
	}
}

void
spirit_filter (float *rr, float *ri,
	float *kr, float *ki, float *xr, float *xi, int M, int N, int C)
{
	_spirit_filter config(M,N, 16,16, 0) (rr, ri, kr, ki, xr, xi, M, N, C);
	cuda_sync();
}

};
