
#include "cuda_macros.h"
#include "Daubechies4.h"

namespace pocs_cuda {

static inline __device__ int mod (int a, int n){
	int amn = a&(n-1);
	return amn<0 ? n+amn : amn;
}

static inline int _log2(int N){
	int k = 1, J = 0;
	while (k<N) k=2*k, J=J+1;
	return J;
}

static inline int _pow2 (int K){
	int k = 1, p = 1;
	for (k = 0; k < K; k++) p = p*2;
	return p;
}

// Really squeezing every byte out of the shared memory. Can't have function
// arguments taking up precious shared regs, now can we?

struct iwt_args {
	float *Xr, *Xi;
	int n, N, m, M;
};

__constant__ iwt_args args;

// iwt_cols<<< dim3(2, C*(n/K)), dim3(T/K, K), K*m*sizeof(float) >>>();
static __global__ void
iwt_cols (void)
{
	extern __shared__ float cols[];
	int ti = threadIdx.x;
	int tj = threadIdx.y;
	float *X;
	int n = args.n, N = args.N;
	int m = args.m, M = args.M;
	int const K = blockDim.y;
	int cj0 = blockIdx.y * K;

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

// Upsample Lowpass col[0:m/2 ... ]
	for (int i = ti; i < m/2; i += blockDim.x)
	{
		float y = 0;
#pragma unroll
		for (int k = 0; k < P/2; k++){
			y += cols[mod(i-k,m/2) + tj*m] * lpf[2*k];
		}
		X[2*i + j*M] = y;

		y = 0;
#pragma unroll
		for  (int k = 0; k < P/2; k++){
			y += cols[mod(i-k,m/2) + tj*m] * lpf[2*k+1];
		}
		X[2*i+1 + j*M] = y; 
	}

	__syncthreads();

// Upsample Highpass col[m/2:m ... ]
	for (int i = ti; i < m/2; i += blockDim.x)
	{
		float y = 0;
#pragma unroll
		for (int k = 0; k < P/2; k++){
			y += cols[m/2 + ((i+k)&(m/2-1)) + tj*m] * hpf[2*k+1];
		}
		X[2*i + j*M] += y;

		y = 0;
#pragma unroll
		for (int k = 0; k < P/2; k++){
			y += cols[m/2 + ((i+k)&(m/2-1)) + tj*m] * hpf[2*k];
		}
		X[2*i+1 + j*M] += y;
	}
}

//	iwt_rows<<< dim3(2*(m/K), C), dim3(K, T/K), K*n*sizeof(float) >>>();
static __global__ void
iwt_rows (void)
{
	int i;
	float *X;
	extern __shared__ float rows[];

	int ti = threadIdx.x;
	int tj = threadIdx.y;
	int const K = blockDim.x;

	int c = blockIdx.y;
	int n = args.n, N = args.N;
	int m = args.m, M = args.M;

	if (blockIdx.x < m/K){
		X = args.Xr + c*M*N;
		i = ti + blockIdx.x * K;
	}
	else{
		X = args.Xi + c*M*N;
		i = ti + blockIdx.x * K - m;
	}

	for (int j = tj; j < n; j += blockDim.y){
		rows[ti + j*K] = X[i + M*j];
	}

	__syncthreads();

// Upsample Lowpass rows[0:n/2,*]
	for (int j = tj; j < n/2; j += blockDim.y){
		float y = 0;
#pragma unroll
		for (int k = 0; k < P/2; k++)
			y += rows[ti + K*mod(j-k,n/2)] * lpf[2*k];
		X[i + 2*j*M] = y;

		y = 0;
#pragma unroll
		for (int k = 0; k < P/2; k++)
			y += rows[ti + K*mod(j-k,n/2)] * lpf[2*k+1];
		X[i + (2*j+1)*M] = y;
	}

	__syncthreads();

// Upsample Highpass rows[n/2:n,*]
	for (int j = tj; j < n/2; j += blockDim.y){
		float y = 0;
		for (int k = 0; k < P/2; k++)
			y += rows[ti + K*(n/2 + ((j+k)&(n/2-1)))] * hpf[2*k+1];
		X[i + 2*j*M] += y;

		y = 0;
		for (int k = 0; k < P/2; k++)
			y += rows[ti + K*(n/2 + ((j+k)&(n/2-1)))] * hpf[2*k]; 
		X[i + (2*j+1)*M] += y;
	}
}

void
iwt_stack (float *sr, float *si, int M, int N, int C, int L)
{
	int m, n;

	/* Nvidia's compiler steals 16 bytes of shared registers, so we can't
	 * actually use all 16384 B. Adjust K accordingly */

	int K0 = 16, K;
	int const T = 512;
	int const SHMEM_SIZE = 16384;

	iwt_args cargs;

	cargs.Xr = sr;
	cargs.Xi = si;
	cargs.N = N;
	cargs.M = M;

	for (m = _pow2(L+1), n = _pow2(L+1); m <= M && n <= N; )
	{
		if ((M/m) > (N/n))
		{
			cargs.n = n/2; cargs.m = m;
			cuda (MemcpyToSymbol (args, &cargs, sizeof(iwt_args)));

			for (K = K0; K*m*sizeof(float) >= SHMEM_SIZE; K = K/2) ;

			iwt_cols<<<dim3(2, C*(n/2/K)), dim3(T/K, K), K*m*sizeof(float)>>>();
			cuda_sync();

			m = 2*m;

		} else
		if ((N/n) > (M/m))
		{
			cargs.n = n; cargs.m = m/2;
			cuda (MemcpyToSymbol (args, &cargs, sizeof(iwt_args)));

			for (K = K0; K*n*sizeof(float) >= SHMEM_SIZE; K = K/2) ;

			iwt_rows<<<dim3(2*(m/2/K), C), dim3(K, T/K), K*n*sizeof(float)>>>();
			cuda_sync();

			n = 2*n;
		}
		else{
			cargs.n = n; cargs.m = m;
			cuda (MemcpyToSymbol (args, &cargs, sizeof(iwt_args)));

			for (K = K0; K*n*sizeof(float) >= SHMEM_SIZE; K = K/2) ;

			iwt_rows<<< dim3(2*(m/K), C), dim3(K, T/K), K*n*sizeof(float) >>>();
			cuda_sync();

			for (K = K0; K*m*sizeof(float) >= SHMEM_SIZE; K = K/2) ;

			iwt_cols<<< dim3(2, C*(n/K)), dim3(T/K, K), K*m*sizeof(float) >>>();
			cuda_sync();

			m = 2*m;
			n = 2*n;
		}
	}
}

};
