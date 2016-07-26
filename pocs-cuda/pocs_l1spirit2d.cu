
#include "cuda_macros.h"
#include "util.h"
#include <stdarg.h>
#include <sys/time.h>
#include "bz2mat.h"
#include <iostream>

#include "l1spirit3d.h"
using l1spirit3d::parameters::pocs_2fix;

static const int L = 4;

/******************************************************************************
******************************************************************************/

namespace pocs_cuda {

void spirit_filter (float*,float*,float*,float*,float*,float*,int,int,int);
void soft_threshold (float*, float*, float, int, int, int);
void fft_stack (float*, int, int, int);
void ifft_stack (float*, int, int, int);
void fwt_stack (float *sr, float *si, int M, int N, int C, int L);
void iwt_stack (float *sr, float *si, int M, int N, int C, int L);

static void ifft2c (float *Xr, float *Xi, float *Xc, int M, int N, int C);
static void fft2c (float *Xr, float *Xi, float *Xc, int M, int N, int C) ;

static __global__ void masked_add (
			float *Xr, float *Xi, float *Tr, float *Ti,
            float *Dr, float *Di, float *mask, int M, int N, int C);

static void prepare_kernels (float *kr, float *ki, float *kcr, float *kci,
                             int K, int C, int M2, int N2);
static void create_mask (float *mask, int M2, int N2,
                         float *Dr, float *Di, int M, int N);
static void debug_array (char const* str, int i,
                         float *k[2], int A, int B, int C=-1, int D=-1);

/******************************************************************************
******************************************************************************/

void
pocs_l1spirit2d (float *X[2], float *kc[2], int K,
                 float *D[2], int M, int N, int C,
                 int n_iter, float lambda)
{
	int iter, M2, N2;
	float *Xp[2], *k[2], *Xc, *T[2], *mask;

        //----------------------------------------
        // Modified by JiaSen: Continuation of the soft-thresholding parameter
        //   from high penalty to a very low penalty.
        // This leads to much faster convergence of the iterations.
        // Radiologists prefer that images are not denoised.
        // l1-norm penalty is set such that minimal final denoizing is performed.
        float *lambda_continuation = new float[n_iter];
        for (int ilambda = 0; ilambda < n_iter; ilambda++){
            lambda_continuation[ilambda] = lambda*2*(1 - ilambda/n_iter);
        }
        float lambda1 = 0;
        //----------------------------------------


	for (N2 = 1; N2 < N; N2 *= 2); 	for (M2 = 1; M2 < M; M2 *= 2);
	//for (N2 = 1; N2 < N || N2 < M; N2 *= 2);

	cuda (Malloc (Xp, 2*M2*N2*C*sizeof(float))); Xp[1] = Xp[0] + M2*N2*C;
	cuda (Malloc (k, 2*M*N*C*C*sizeof(float))); k[1] = k[0] + M*N*C*C;
	cuda (Malloc (&Xc, 4*M*N*C*sizeof(float)));
	T[0] = Xc + 2*M*N*C; T[1] = T[0] + M*N*C;
	cuda (Malloc (&mask, M*N*sizeof(float)));

	create_mask (mask, M, N, D[0], D[1], M, N);

	prepare_kernels (k[0], k[1], kc[0], kc[1], K, C, M, N);      // get image domain spirit kernel
	
	for (iter = 1; iter <= n_iter; iter++)
	{
		ifft2c (X[0], X[1], Xc, M, N, C);                             // go to image domain

		spirit_filter (T[0], T[1], k[0], k[1], X[0], X[1], M, N, C);  // apply (G-I)*x + x

		if (pocs_2fix){
		// Mar10 -- Adding second fixing step as a band-aid
			fft2c (T[0], T[1], Xc, M, N, C);
			masked_add config(M,N,16,16,0) (T[0], T[1], T[0], T[1],
			                                D[0], D[1], mask, M, N, C);
			ifft2c (T[0], T[1], Xc, M, N, C);
		}

		if (lambda != 0){
			zpad2d (Xp[0], Xp[1], M2, N2, T[0], T[1], M, N, C);   // zpad to the closest diadic
			fwt_stack (Xp[0], Xp[1], M2, N2, C, L);               // apply wavelet
			//soft_threshold (Xp[0], Xp[1], lambda, M2, N2, C);
                        lambda1 = lambda_continuation[iter-1];
                        soft_threshold (Xp[0], Xp[1], lambda1, M2, N2, C);    // threshold (joint sparsity)
			iwt_stack (Xp[0], Xp[1], M2, N2, C, L);               // get back the image
			crop2d (T[0], T[1], M, N, Xp[0], Xp[1], M2, N2, C);   // return to the original size
		}

		fft2c (T[0], T[1], Xc, M, N, C);                              // go back to k-space

		masked_add config(M,N,16,16,0)
			(X[0], X[1], T[0], T[1], D[0], D[1], mask, M, N, C);
		cuda_sync();
	}

	cuda (Free (Xp[0]));
	cuda (Free (k[0]));
	cuda (Free (Xc));
	cuda (Free (mask));
}

static __global__ void
masked_add (float *Xr, float *Xi, float *Tr, float *Ti,
            float *Dr, float *Di, float *mask, int M, int N, int C)
{
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	int j = threadIdx.y + blockDim.y*blockIdx.y;
	int m = (i<M && j<N) ? mask[i + M*j] : 0;

	for (int c = 0; c < C; c++){
		if (i<M && j<N){
			int ijc = i + M*(j + N*c);

			Xr[ijc] = m ? Dr[ijc] : Tr[ijc];
			Xi[ijc] = m ? Di[ijc] : Ti[ijc];
		}
	}
}

// reverse the convolution kernel and scale 
// matlab code: kernel(end:-1:1,end:-1:1,:,n)*sqrt(imSize(1)*imSize(2))
static __global__ void
scale_and_reverse (float *krcrc, float *krcic, float *kcrc, float *kcic,
                   int M2, int N2, int K, int C)
{
	int i = threadIdx.x;
	int j = threadIdx.y;
	float s = sqrtf(M2*N2);

	for (int cc = threadIdx.z; cc < C; cc++){
		if (i<K && j<K){
			krcrc[i + K*(j + K*cc)] = s * kcrc[K-1-i + K*(K-1-j + K*cc)];
			krcic[i + K*(j + K*cc)] = s * kcic[K-1-i + K*(K-1-j + K*cc)];
		}
	}
}


// precompute the image domain kernel by zero-padding and inverse fft
// matlab code: ifft2c(zpad( kernel_after_scale_and_reverse ))
static void
prepare_kernels (float *kr, float *ki, float *kcr, float *kci,
                 int K, int C, int M2, int N2)
{
	float *krcrc, *krcic;

	cuda (Malloc (&krcrc, 2*sizeof(float)*K*K*C)); krcic = krcrc + K*K*C;

	float *Xc;
	cuda (Malloc (&Xc, 2*M2*N2*C*sizeof(float)));

	for (int c = 0; c < C; c++)
	{
		float *krc, *kic, *kcrc, *kcic;

		krc = kr + M2*N2*C*c;
		kic = ki + M2*N2*C*c;

		kcrc = kcr + K*K*C*c;
		kcic = kci + K*K*C*c;

		scale_and_reverse <<<1, dim3 (K,K,2)>>>
		                  (krcrc,krcic, kcrc,kcic, M2,N2,K,C);
		cuda_sync();

		zpad2d (krc, kic, M2, N2, krcrc, krcic, K, K, C);

		cuda_sync();

		ifft2c (krc, kic, Xc, M2, N2, C);
		cuda_sync();
	}

	cuda (Free (krcrc));
	cuda (Free (Xc));
}

static void
create_mask (float *_mask, int M2, int N2,
             float *_Dr, float *_Di, int M, int N)
{
	float *dr = (float*) malloc (M*N*sizeof(float));
	float *di = (float*) malloc (M*N*sizeof(float));

	cuda (Memcpy (dr, _Dr, M*N*sizeof(float), cudaMemcpyDeviceToHost));
	cuda (Memcpy (di, _Di, M*N*sizeof(float), cudaMemcpyDeviceToHost));

	float *mask = (float*) malloc (M*N*sizeof(float));

	double maxr = 0;
	for (int j = 0; j < N; j++){
		for (int i = 0; i < M; i++)
		{
			double ii = i==(M-1) ? 1.0 : -1.0 + i*(2.0)/(M-1);
			double jj = j==(M-1) ? 1.0 : -1.0 + j*(2.0)/(N-1);
			double r = sqrt(ii*ii + jj*jj);

			mask[i + M*j] = (dr[i+M*j] != 0 || di[i+M*j] != 0);

			if (mask[i+M*j] != 0 && r > maxr)
				maxr = r;
		}
	}

	for (int j = 0; j < N; j++){
		for (int i = 0; i < M; i++)
		{
			double ii = i==(M-1) ? 1.0 : -1.0 + i*(2.0)/(M-1);
			double jj = j==(M-1) ? 1.0 : -1.0 + j*(2.0)/(N-1);
			double r = sqrt(ii*ii + jj*jj);

			if (r > maxr)
				mask[i + M*j] = 1.0;
		}
	}

	float *pmask = (float*) malloc (M2 * N2 * sizeof(float));
	int i0 = (int)floor(M2/2.0) + (int)ceil(-M/2.0);
	int j0 = (int)floor(N2/2.0) + (int)ceil(-N/2.0);

	for (int j = 0; j < N2; j++){
		for (int i = 0; i < M2; i++){
			if (i < i0 || i > i0+M || j < j0 || j > j0+N){
				pmask[i + M2*j] = 1.0;
			}else{
				pmask[i + M2*j] = mask[i-i0 + M*(j-j0)];
			}
		}
	}

	cuda (Memcpy (_mask, pmask, M2*N2*sizeof(float), cudaMemcpyHostToDevice));

	free (dr); free (di);
	free (mask);
	free (pmask);
}

static __global__ void
_shift (float *Yr, float *Yi, int Ys, float *Xr, float *Xi, int Xs,
	int M, int N, int C, int *idxM, int *idxN)
{
	int c = blockIdx.x;

	Yr += M*N*c*Ys; Yi += M*N*c*Ys;
	Xr += M*N*c*Xs; Xi += M*N*c*Xs;

	for (int i = threadIdx.x; i < M; i += blockDim.x){
		for (int j = threadIdx.y; j < N; j += blockDim.y)
		{
			int iY = (i + M*j)*Ys;
			int iX = (idxM[i] + M*idxN[j])*Xs;

			Yr[iY] = Xr[iX];
			Yi[iY] = Xi[iX];
		}
	}
}

static void
ifftshift_indices (int *idx, int M){
	int i = 0;

	int p = (int) floor ((1.0*M) / 2.0);

	for (; p < M; i++, p++)
		idx[i] = p;
	for (int j = 0; i < M; i++, j++)
		idx[i] = j;
}

static void
fftshift_indices (int *idx, int M){
	int i = 0;

	int p = (int) ceil ((1.0*M) / 2.0);

	for (; p < M; i++, p++)
		idx[i] = p;
	for (int j = 0; i < M; i++, j++)
		idx[i] = j;
}

static void
shift_convert (float *Xc, float *Xr, float *Xi, int M, int N, int C,
                  bool is_ifftshift, bool is_interleave)
{
	int *_idxM, *idxM, *_idxN, *idxN;

	cuda_sync();
	idxM = new int[M];
	cuda (Malloc (&_idxM, M*sizeof(int)));
	cuda_sync();

	idxN = new int[N];
	cuda (Malloc (&_idxN, N*sizeof(int)));
	cuda_sync();

	if (is_ifftshift){
		ifftshift_indices (idxM, M);
		ifftshift_indices (idxN, N);
	cuda_sync();
	}else{
		fftshift_indices (idxM, M);
		fftshift_indices (idxN, N);
	cuda_sync();
	}

	cuda (Memcpy (_idxM, idxM, M*sizeof(int), cudaMemcpyHostToDevice));
	cuda (Memcpy (_idxN, idxN, N*sizeof(int), cudaMemcpyHostToDevice));

	if (is_interleave){
		_shift <<< C, dim3(16,16), 0 >>>
			(Xc, Xc+1, 2, Xr, Xi, 1, M, N, C, _idxM, _idxN);
		cuda_sync();
	}
	else{
		_shift <<< C, dim3(16,16), 0 >>>
			(Xr, Xi, 1, Xc, Xc+1, 2, M, N, C, _idxM, _idxN);
		cuda_sync();
	}

	delete [] idxM;
	cuda (Free (_idxM));

	delete [] idxN;
	cuda (Free (_idxN));
}

static void
ifftshift_interleave (float *Xc, float *Xr, float *Xi, int M, int N, int C){
	shift_convert (Xc, Xr, Xi, M, N, C, true, true);
}

static void
fftshift_deinterleave (float *Xc, float *Xr, float *Xi, int M, int N, int C){
	shift_convert (Xc, Xr, Xi, M, N, C, false, false);
}

static void
ifft2c (float *Xr, float *Xi, float *Xc, int M, int N, int C)
{
	ifftshift_interleave (Xc, Xr, Xi, M, N, C);
	cuda_sync();

	ifft_stack (Xc, M, N, C);
	cuda_sync();

	fftshift_deinterleave (Xc, Xr, Xi, M, N, C);
	cuda_sync();
}

static void
fft2c (float *Xr, float *Xi, float *Xc, int M, int N, int C) 
{
	ifftshift_interleave (Xc, Xr, Xi, M, N, C);
	cuda_sync();

	fft_stack (Xc, M, N, C);
	cuda_sync();

	fftshift_deinterleave (Xc, Xr, Xi, M, N, C);
	cuda_sync();
}

static void debug_array (char const* str, int i,
                         float *k[2], int A, int B, int C, int D)
{
	char name[1024];
	sprintf(name, "%s%d", str, i);
	bool isc = (k[1] != 0);

	int ndims = 4;
	if (D < 0) ndims--, D=1;
	if (C < 0) ndims--, C=1;

	std::cerr << "debug_array: " << name << std::endl
		<< "\t A = " << A << std::endl
		<< "\t B = " << B << std::endl
		<< "\t C = " << C << std::endl
		<< "\t D = " << D << std::endl;

	float *t[2] = { new float[A*B*C*D], isc ? new float[A*B*C*D] : 0 };
	int dims[4] = {A, B, C, D};

	cuda (Memcpy (t[0], k[0], A*B*C*D*sizeof(float),
	              cudaMemcpyDeviceToHost));

	if (isc)
		cuda (Memcpy (t[1], k[1], A*B*C*D*sizeof(float),
		              cudaMemcpyDeviceToHost));

	write_bz2mat ((void**)t, false, ndims, dims, name);

	delete[] t[0];
	if (isc) delete[] t[1];
}


}; //namespace pocs_cuda
