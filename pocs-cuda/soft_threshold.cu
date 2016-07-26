
#include "cuda_macros.h"
#include "util.h"

namespace pocs_cuda {

float const eps = 1.1921e-7f;

__global__ void
_soft_threshold (float *xr, float *xi, float t, int M, int N, int C)
{
	int const i = threadIdx.x + blockDim.x*blockIdx.x;
	int const j = threadIdx.y + blockDim.y*blockIdx.y;
	float xrij=0, xiij=0, s2=0, s, r;
	int c;

	for (c = 0; c < C; c++){
		if (i<M && j<N){
			xrij = xr[i + M*(j + N*c)];
			xiij = xi[i + M*(j + N*c)];
		}
		s2 += xrij*xrij + xiij*xiij;
	}

	s = sqrtf(s2);

	r = s-t;
	r = 0.5f*(r + fabs(r));
	r = r / (s+eps);

	for (c = 0; c < C; c++){
		if (i<M && j<N){
			xr[i + M*(j + N*c)] = r * xr[i + M*(j + N*c)];
			xi[i + M*(j + N*c)] = r * xi[i + M*(j + N*c)];
		}
	}
}

void
soft_threshold (float *xr, float *xi, float t, int M, int N, int C)
{
	_soft_threshold config(M,N, 16,16, 0) (xr,xi,t,M,N,C);
	cuda_sync();
}

};

#ifdef STANDALONE_TEST

int
main()
{
	size_t M=256, N=256, C=8;
	float *x, *xr, *xi, *y, *yi, *yr, t = 0.1;

	xr = (float*) malloc(M*N*C*sizeof(float));
	xi = (float*) malloc(M*N*C*sizeof(float));

	yr = (float*) malloc(M*N*C*sizeof(float));
	yi = (float*) malloc(M*N*C*sizeof(float));

	cuda (Malloc (&x, 2*M*N*C*sizeof(float))); 
	cuda (Malloc (&y, 2*M*N*C*sizeof(float)));

	upload_interleave (x, xr, xi, M*N*C);

	soft_threshold (y, x, t, M, N, C);

	deinterleave_download (yr, yi, y, M*N*C);

	cuda (Free (x));
	cuda (Free (y));

	free (xr); free (xi);
	free (yr); free (yi);

	return 0;
}


#elif defined MEX_ENTRY

#include "mex.h"
#define trace mexPrintf("At %s:%d\n", __FILE__, __LINE__);
extern "C" void
mexFunction (int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{
	size_t n_dims, M, N, C;
	size_t const *dims;
	float *_xr, *_xi, *xr, *xi, *_yr, *_yi, t;

	if (nrhs != 2){
		mexPrintf("st = soft_threshold (x, threshold)\n");
		return;
	}
	if (!mxIsSingle (prhs[0])){
		mexPrintf("x must be single precision\n");
		return;
	}
	if (!mxIsComplex (prhs[0])){
		mexPrintf("x must be complex\n");
		return;
	}

 	if (!mxIsSingle(prhs[1]))
		t = *((double*) mxGetData (prhs[1]));
	else
		t = *((float*) mxGetData (prhs[1]));

	n_dims = (size_t) mxGetNumberOfDimensions (prhs[0]);
	dims = (size_t const*) mxGetDimensions (prhs[0]);

	if (n_dims != 3){
		mexPrintf("x must be MxNxC\n");
		return;
	}

	M = dims[0]; N = dims[1]; C = dims[2];

	_xr = (float*) mxGetData (prhs[0]);
	_xi = (float*) mxGetImagData (prhs[0]);

	try {

		cuda (Malloc (&xr, M*N*C*sizeof(float))); 
		cuda (Memcpy (xr, _xr, M*N*C*sizeof(float), cudaMemcpyHostToDevice));

		cuda (Malloc (&xi, M*N*C*sizeof(float)));
		cuda (Memcpy (xi, _xi, M*N*C*sizeof(float), cudaMemcpyHostToDevice));

		soft_threshold (xr, xi, t, M, N, C);

		plhs[0] = mxCreateNumericArray (n_dims, dims,
		                                mxSINGLE_CLASS, mxCOMPLEX);

		_yr = (float*) mxGetData (plhs[0]);
		_yi = (float*) mxGetImagData (plhs[0]);

		cuda (Memcpy (_yr, xr, M*N*C*sizeof(float), cudaMemcpyDeviceToHost));
		cuda (Memcpy (_yi, xi, M*N*C*sizeof(float), cudaMemcpyDeviceToHost));

		cuda (Free (xr));
		cuda (Free (xi));
	}
	catch (cuda_error_exception e){
		mexPrintf("Caught exception! Aborting\n");
	}
}
#endif
