#include "cuda_macros.h"
#include "util.h"


__global__ void
_shift_stack (float *Sr, float *Si, int M, int N, int C){
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	int j = threadIdx.y + blockDim.y*blockIdx.y;
	int c;
	float s = (i+j)&1 ? -1.0f : 1.0f; 

	Sr += i+M*j;
	Si += i+M*j;

	if (i < M && j < N){
		for (c = 0; c < C; c++){
			*Sr = s*(*Sr);
			*Si = s*(*Si); 

			Sr += M*N;
			Si += M*N;
		}
	}
}

void
shift_stack (float *Sr, float *Si, int M, int N, int C){
	_shift_stack config(M,N, 16,16, 0)  (Sr, Si, M, N, C);

	cuda_sync();
}

__global__ void
_scale (float *x, int N, float ar, float ai){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i<N){
		float xr = x[2*i], xi = x[2*i+1];

		x[2*i+0] = xr*ar - xi*ai; 
		x[2*i+1] = xr*ai + xi*ar;
	}
}

void
scale (float *x, int N, float sr, float si)
{
	_scale config(N,1,128,1,0) (x,N,sr,si);

	cuda_sync();
}

__global__ void
_scale_2 (float *x, int M, int N, float ar, float ai){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;

	if (i<M && j<N){
		float xr = x[2*(i+M*j)], xi = x[2*(i+M*j)+1];

		x[2*(i+M*j)+0] = xr*ar - xi*ai; 
		x[2*(i+M*j)+1] = xr*ai + xi*ar;
	}
}

void
scale_2 (float *x, int M, int N, float ar, float ai){
	_scale_2 config(M,N,16,16,0) (x,M,N,ar,ai);

	cuda_sync ();
}



__global__ void
_interleave (float *x, float *xr, float *xi, int N)
{
	int i = threadIdx.x + blockDim.x*blockIdx.x;

	if (i < N){
		x[0+2*i] = xr[i];
		x[1+2*i] = xi[i];
	}
}

void
interleave (float *x, float *xr, float *xi, int N)
{
	_interleave config(N,1, 64,1, 0) (x, xr, xi, N);
	cuda_sync();
}

void
upload_interleave (float *x, float *_xr, float *_xi, int N)
{
	float *xr, *xi;

	cuda (Malloc(&xr, N*sizeof(float)));
	cuda (Malloc(&xi, N*sizeof(float)));

	cuda (Memcpy (xr, _xr, N*sizeof(float), cudaMemcpyHostToDevice));
	cuda (Memcpy (xi, _xi, N*sizeof(float), cudaMemcpyHostToDevice));

	_interleave config(N,1, 128,1, 0) (x, xr, xi, N);
	cuda_sync();

	cuda (Free (xr));
	cuda (Free (xi));
}


__global__ void
interleave_2 (float *x, float *xr, float *xi, int M, int N)
{
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	int j = threadIdx.y + blockDim.y*blockIdx.y;

	if (i < M && j < N){
		x[0+2*(i+M*(j))] = xr[i+M*(j)];
		x[1+2*(i+M*(j))] = xi[i+M*(j)];
	}
}

void
upload_interleave_2 (float *x, float *_xr, float *_xi, int M, int N)
{
	float *xr, *xi;

	cuda (Malloc(&xr, M*N*sizeof(float)));
	cuda (Malloc(&xi, M*N*sizeof(float)));

	cuda (Memcpy (xr, _xr, M*N*sizeof(float), cudaMemcpyHostToDevice));
	cuda (Memcpy (xi, _xi, M*N*sizeof(float), cudaMemcpyHostToDevice));

	interleave_2 config(M,N, 16,16, 0) (x, xr, xi, M, N);

	cuda_sync();

	cuda (Free (xr));
	cuda (Free (xi));
}

__global__ void
_deinterleave (float *xr, float *xi, float *x, int N)
{
	int i = threadIdx.x + blockDim.x*blockIdx.x;

	if (i < N){
		xr[i] = x[0+2*i];
		xi[i] = x[1+2*i];
	}
}

void deinterleave (float *xr, float *xi, float *x, int N)
{
	_deinterleave config(N,1, 64,1,0) (xr, xi, x, N);

	cuda_sync();
}

void
deinterleave_download (float *_xr, float *_xi, float *x, int N)
{
	float *xr, *xi;

	cuda (Malloc (&xr, N*sizeof(float)));
	cuda (Malloc (&xi, N*sizeof(float)));

	_deinterleave config(N,1, 64,1,0) (xr, xi, x, N);

	cuda_sync();

	cuda (Memcpy (_xr, xr, N*sizeof(float), cudaMemcpyDeviceToHost));
	cuda (Memcpy (_xi, xi, N*sizeof(float), cudaMemcpyDeviceToHost));

	cuda (Free (xr));
	cuda (Free (xi));
}

__global__ void
deinterleave_2 (float *xr, float *xi, float *x, int M, int N)
{
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	int j = threadIdx.y + blockDim.y*blockIdx.y;

	if (i < M && j < N){
		xr[i+M*(j)] = x[0+2*(i+M*(j))];
		xi[i+M*(j)] = x[1+2*(i+M*(j))];
	}
}

void
deinterleave_download_2 (float *_xr, float *_xi, float *x, int M, int N)
{
	float *xr, *xi;

	cuda (Malloc (&xr, M*N*sizeof(float)));
	cuda (Malloc (&xi, M*N*sizeof(float)));

	deinterleave_2 config(M,N,16,16,0) (xr, xi, x, M, N);

	cuda_sync();

	cuda (Memcpy (_xr, xr, M*N*sizeof(float), cudaMemcpyDeviceToHost));
	cuda (Memcpy (_xi, xi, M*N*sizeof(float), cudaMemcpyDeviceToHost));

	cuda (Free (xr));
	cuda (Free (xi));
}

void
pad_indices (int* i0, int* j0, int M2, int N2, int M, int N){
	*i0 = (int) floor(M2/2.0) + (int) ceil(-M/2.0);
	*j0 = (int) floor(N2/2.0) + (int) ceil(-N/2.0);
}

__global__ void
zpad2d1 (float *xpr, float *xpi, float *xr, float *xi,
         int i0, int j0, int m2, int n2, int m, int n, int c)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	
	for (int k = threadIdx.z; k < c; k += blockDim.z){
		if (i<m && j<n){
			xpr[i0+i + m2*(j0+j + n2*k)] = xr[i + m*(j + n*k)];
			xpi[i0+i + m2*(j0+j + n2*k)] = xi[i + m*(j + n*k)];
		}
	}
}

void
zpad2d (float *xpr, float *xpi, int m2, int n2,
        float *xr, float *xi, int m, int n, int c)
{
	int i0, j0;

	cuda (Memset (xpr, 0, m2*n2*c*sizeof(float)));
	cuda (Memset (xpi, 0, m2*n2*c*sizeof(float)));

	pad_indices (&i0, &j0, m2,  n2, m, n);

	zpad2d1 <<< dim3 ((m+15)/16, (n+15)/16, 1), dim3 (16,16,2) >>>
	          ( xpr, xpi, xr, xi, i0, j0, m2, n2, m, n, c );
	cuda_sync();
}

__global__ void
crop2d1 (float *xr, float *xi, float *xpr, float *xpi,
         int i0, int j0, int m2, int n2, int m, int n, int c)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	
	for (int k = threadIdx.z; k < c; k += blockDim.z){
		if (i<m && j<n){
			xr[i + m*(j + n*k)] = xpr[i0+i + m2*(j0+j + n2*k)];
			xi[i + m*(j + n*k)] = xpi[i0+i + m2*(j0+j + n2*k)];
		}
	}
}

void
crop2d (float *xr, float *xi, int m, int n,
        float *xpr, float *xpi, int m2, int n2, int c)
{
	int i0, j0;

	pad_indices (&i0, &j0, m2, n2, m, n);
	crop2d1 <<< dim3((m+15)/16, (n+15)/16, 1), dim3(16,16,2) >>>
	          (xr, xi, xpr, xpi, i0, j0, m2, n2, m, n, c );
	cuda_sync();
}

