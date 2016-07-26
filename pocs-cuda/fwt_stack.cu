#include "cuda_macros.h"

#include "Daubechies4.h"

namespace pocs_cuda {

static inline int _log2 (int N){
	int k = 1, J=0;
	while (k<N) k = 2*k, J=J+1;
	return J;
}

static inline __device__ int mod (int a, int n){
	int amn = a&(n-1);
	return amn<0 ? amn+n : amn;
}

struct fwt_args {
	float *Xr, *Xi;
	int n, N, m, M;
};
__constant__ fwt_args args;


// fwt_cols <<< (2, C*(m/K)), (T/K, K), K*m*sz(float) >>>
static __global__ void
fwt_cols (void)
{
	extern __shared__ float cols [];

	int const K = blockDim.y;
	int const cj0 = blockIdx.y * K;

	int ti = threadIdx.x;
	int tj = threadIdx.y;
	float *X;

	int m = args.m, M = args.M;
	int n = args.n, N = args.N;

	int c = cj0 / n;
	int j = tj + cj0 - c*n;

	if (blockIdx.x == 0){
		X = args.Xr + c*M*N;
	}
	else{
		X = args.Xi + c*M*N;
	}

	for (int i = ti; i < m; i += blockDim.x){
		cols[i + tj*m] = X[i + j*M];
	}

	__syncthreads();

// Low-Pass Downsample
	for (int i = ti; i < m/2; i += blockDim.x){
		float y = 0;
#pragma unroll
		for (int k = 0; k < P; k++){
			y += cols[((i*2+k) & (m-1)) + tj*m] * lpf[k];
		}
		X[i + j*M] = y;
	}

// High-Pass Downsample
	for (int i = ti; i < m/2; i += blockDim.x){
		float y = 0;
#pragma unroll
		for (int k = 0; k < P; k++){
			y += cols[mod(i*2+1-k, m) + tj*m] * hpf[k];
		}

		X[m/2+i + j*M] = y;
	}
}

// fwt_rows <<< (2*(n/K), C), (K, T/K), K*n*sz(float) >>>
static __global__ void
fwt_rows (void)
{
	int i;
	float *X;
	extern __shared__ float rows [];
	int const K = blockDim.x;

	int ti = threadIdx.x;
	int tj = threadIdx.y;
	int c = blockIdx.y;

	int n = args.n, N = args.N;
	int m = args.m, M = args.M;

	if (blockIdx.x < m/K){
		X = args.Xr + c*M*N;
		i = ti + K*blockIdx.x;
	}
	else{
		X = args.Xi + c*M*N;
		i = ti + K*blockIdx.x - m;

	}

	for (int j = tj; j < n; j += blockDim.y){
		rows[ti + j*K] = X[i + M*j];
	}

	__syncthreads();

// Low-Pass Downsample
	for (int j = tj; j < n/2; j += blockDim.y){
		float y = 0;
#pragma unroll
		for (int k = 0; k < P; k++)
			y += rows[ti + K*((2*j+k)&(n-1))] * lpf[k];

		X[i + j*M] = y;
	}

	// High-Pass Downsample
	for (int j = tj; j < n/2; j += blockDim.y){
		float y = 0;
#pragma unroll
		for (int k = 0; k < P; k++)
			y += rows[ti + K*mod(2*j+1-k, n)] * hpf[k];

		X[i + (n/2)*M + j*M] = y;
	}
}

void
fwt_stack (float *sr, float *si, int M, int N, int C, int L)
{
/* Nvidia's compiler steals 16 bytes of shared registers, so we can't
 * actually use all 16384 B. Adjust K accordingly */
	int K0 = 16, K;
	int const SHMEM_SIZE = 16384;
	int const T = 512;

	fwt_args cargs;
	cargs.Xr = sr;
	cargs.Xi = si;
	cargs.N = N;
	cargs.M = M;

	int min_size = (1 << L);
	int JM = _log2(M), JN = _log2(N), J = JM > JN ? JM : JN;

	for (int m = M, n = N, j = J; j > L; j--)
	{
		cargs.n = n;
		cargs.m = m;

		cuda (MemcpyToSymbol(args, &cargs, sizeof(fwt_args)));


		if (m > min_size)
		{
			for (K = K0; K*m*sizeof(float) >= SHMEM_SIZE; K >>= 1);
			if (K == 0) throw cuda_error_exception();

			fwt_cols<<< dim3(2, C*(n/K)), dim3(T/K, K), K*m*sizeof(float) >>>();
			cuda_sync();
		}

		if (n > min_size)
		{
			for (K = K0; K*n*sizeof(float) >= SHMEM_SIZE; K >>= 1) ;
			if (K == 0) throw cuda_error_exception();

			fwt_rows<<< dim3(2*(m/K), C), dim3(K, T/K), K*n*sizeof(float) >>>();
			cuda_sync();
		}

		if (m > min_size) m = m/2;
		if (n > min_size) n = n/2;
	}
}

};
