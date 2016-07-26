
#include <stdarg.h>
#include <sys/time.h>
#include <iostream>

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "bz2mat.h"

#include <fftw3.h>

static int __dump__ = 0;

static const int L = 4;

/******************************************************************************
******************************************************************************/

namespace pocs_omp {

void spirit_filter (float*,float*,float*,float*,float*,float*,int,int,int);
void soft_threshold (float*, float*, float, int, int, int);
void fft_stack (float*, int, int, int, fftwf_plan*);
void ifft_stack (float*, int, int, int, fftwf_plan*);
void fwt_stack (float *sr, float *si, int M, int N, int C, int L);
void iwt_stack (float *sr, float *si, int M, int N, int C, int L);

static void ifft2c (float *Xr, float *Xi, float *Xc,
                    int M, int N, int C, fftwf_plan*);

static void fft2c (float *Xr, float *Xi, float *Xc,
                   int M, int N, int C, fftwf_plan*) ;

static void prepare_kernels (float *kr, float *ki, float *kcr, float *kci,
                             int K, int C, int M2, int N2, fftwf_plan*);

static void masked_add (float *Xr, float *Xi, float *Tr, float *Ti,
                        float *Dr, float *Di, float *mask, int M, int N, int C);


static void create_mask (float *mask, int M2, int N2,
                         float *Dr, float *Di, int M, int N);

static void zpad2d (float *pr, float *pi, int m2, int n2,
                    float *r,  float *i,  int m,  int n,  int c);

static void crop2d (float *r,  float *i,  int m,  int n,
                    float *pr, float *pi, int m2, int n2, int c);

static void debug_array (char const* str, int i,
                         float *k[2], int A, int B, int C=-1, int D=-1);

/******************************************************************************
******************************************************************************/

void
pocs_l1spirit2d (float *X[2], float *kc[2], int K,
                 float *D[2], int M, int N, int C,
                 int n_iter, float lambda, fftwf_plan *stack_plans)
{
	int iter, M2, N2;
	float *Xp[2], *k[2], *T[2], *mask;
	float *Xc = new float[2*M*N*C];

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

	for (N2 = 1; N2 < N; N2 *= 2);
	for (M2 = 1; M2 < M; M2 *= 2);

	Xp[0] = new float[M2*N2*C];
	Xp[1] = new float[M2*N2*C];

	k[0] = new float[M*N*C*C];
	k[1] = new float[M*N*C*C];

	T[0] = new float[M*N*C];
	T[1] = new float[M*N*C];

	mask = new float[M*N];

	create_mask (mask, M, N, D[0], D[1], M, N);

	if(__dump__){
		float *p[2] = {mask, 0};
		debug_array ("mask", 0, p, M, N);

		debug_array ("kconv", 0, kc, K, K, C, C);
	}

	prepare_kernels (k[0], k[1], kc[0], kc[1], K, C, M, N, stack_plans);

	if(__dump__){
		debug_array ("kernels", 0, k, M, N, C, C);
	}
	
	for (iter = 1; iter <= n_iter; iter++)
	{
		ifft2c (X[0], X[1], Xc, M, N, C, stack_plans);

		if(__dump__){
			debug_array ("ifft2c", iter, X, M, N, C);
		}

		spirit_filter (T[0], T[1], k[0], k[1], X[0], X[1], M, N, C);

		if(__dump__){
			debug_array ("filter", iter, T, M, N, C);
		}

		if (lambda != 0)
		{
			zpad2d (Xp[0], Xp[1], M2, N2, T[0], T[1], M, N, C);

			if(__dump__){	
				debug_array ("padded", iter, Xp, M2, N2, C);
			}

			fwt_stack (Xp[0], Xp[1], M2, N2, C, L);

			if(__dump__){
				debug_array ("fwave", iter, Xp, M2, N2, C);
			}

                        //soft_threshold (Xp[0], Xp[1], lambda, M2, N2, C);
                        lambda1 = lambda_continuation[iter-1];
                        soft_threshold (Xp[0], Xp[1], lambda1, M2, N2, C);

			if(__dump__){
				debug_array ("thresh", iter, Xp, M2, N2, C);
			}

			iwt_stack (Xp[0], Xp[1], M2, N2, C, L);

			if(__dump__){
				debug_array ("iwave", iter, Xp, M2, N2, C);
			}

			crop2d (T[0], T[1], M, N, Xp[0], Xp[1], M2, N2, C);

			if(__dump__){
				debug_array ("cropped", iter, T, M, N, C);
			}
		}

		fft2c (T[0], T[1], Xc, M, N, C, stack_plans);

		if(__dump__){
			debug_array ("fft2c", iter, T, M, N, C);
		}

		masked_add (X[0], X[1], T[0], T[1], D[0], D[1], mask, M, N, C);

		if(__dump__){
			debug_array ("fixed", iter, X, M, N, C);
		}

		if(__dump__ && iter >= __dump__){
			exit(-1);
		}
	}

	delete[] Xp[0];
	delete[] Xp[1];

	delete[] k[0];
	delete[] k[1];

	delete[] T[0];
	delete[] T[1];

	delete[] mask;
}

static void
masked_add (float *Xr, float *Xi, float *Tr, float *Ti,
            float *Dr, float *Di, float *mask, int M, int N, int C)
{
	for (int c = 0; c < C; c++){
		for (int i = 0; i < M; i++){
			for (int j = 0; j < N; j++){
				int m = mask[i + M*j];

				int ijc = i + M*(j + N*c);

				Xr[ijc] = m ? Dr[ijc] : Tr[ijc];
				Xi[ijc] = m ? Di[ijc] : Ti[ijc];
			}
		}
	}
}

static void
scale_and_reverse (float *krcrc, float *krcic, float *kcrc, float *kcic,
                   int M2, int N2, int K, int C)
{
	float s = sqrtf(M2*N2);

	for (int cc = 0; cc < C; cc++){
		for (int i = 0; i < K; i++){
			for (int j = 0; j < K; j++){
				krcrc[i + K*(j + K*cc)] = s * kcrc[K-1-i + K*(K-1-j + K*cc)];
				krcic[i + K*(j + K*cc)] = s * kcic[K-1-i + K*(K-1-j + K*cc)];
			}
		}
	}
}

static void
prepare_kernels (float *kr, float *ki, float *kcr, float *kci,
                 int K, int C, int M2, int N2, fftwf_plan* stack_plans)
{
	float *krcrc, *krcic;

	krcrc = new float[K*K*C];
	krcic = new float[K*K*C];

	float *Xc = new float[2*M2*N2*C];

	for (int c = 0; c < C; c++)
	{
		float *krc, *kic, *kcrc, *kcic;

		krc = kr + M2*N2*C*c;
		kic = ki + M2*N2*C*c;

		kcrc = kcr + K*K*C*c;
		kcic = kci + K*K*C*c;

		scale_and_reverse (krcrc,krcic, kcrc,kcic, M2,N2,K,C);

		zpad2d (krc, kic, M2, N2, krcrc, krcic, K, K, C);

		ifft2c (krc, kic, Xc, M2, N2, C, stack_plans);
	}

	delete[] krcrc;
	delete[] krcic;
	delete[] Xc;
}

static void
create_mask (float *mask, int M2, int N2,
             float *dr, float *di, int M, int N)
{
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

	free (pmask);
}

static void
_shift (float *Yr0, float *Yi0, int Ys, float *Xr0, float *Xi0, int Xs,
        int M, int N, int C, int *idxM, int *idxN)
{
	for (int c = 0; c < C; c++){

		float *Yr = Yr0 + M*N*c*Ys, *Yi = Yi0 + M*N*c*Ys;
		float *Xr = Xr0 + M*N*c*Xs, *Xi = Xi0 + M*N*c*Xs;

		for (int i = 0; i < M; i++){
			for (int j = 0; j < N; j++)
			{
				int iY = (i + M*j)*Ys;
				int iX = (idxM[i] + M*idxN[j])*Xs;
	
				Yr[iY] = Xr[iX];
				Yi[iY] = Xi[iX];
			}
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
	int *idxM, *idxN;

	idxM = new int[M];
	idxN = new int[N];

	if (is_ifftshift){
		ifftshift_indices (idxM, M);
		ifftshift_indices (idxN, N);
	}else{
		fftshift_indices (idxM, M);
		fftshift_indices (idxN, N);
	}

	if (is_interleave){
		_shift (Xc, Xc+1, 2, Xr, Xi, 1, M, N, C, idxM, idxN);
	}
	else{
		_shift (Xr, Xi, 1, Xc, Xc+1, 2, M, N, C, idxM, idxN);
	}

	delete[] idxM;
	delete[] idxN;
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
ifft2c (float *Xr, float *Xi, float *Xc, int M, int N, int C, fftwf_plan *plans)
{
	ifftshift_interleave (Xc, Xr, Xi, M, N, C);
	ifft_stack (Xc, M, N, C, plans);
	fftshift_deinterleave (Xc, Xr, Xi, M, N, C);
}

static void
fft2c (float *Xr, float *Xi, float *Xc, int M, int N, int C, fftwf_plan *plans) 
{
	ifftshift_interleave (Xc, Xr, Xi, M, N, C);
	fft_stack (Xc, M, N, C, plans);
	fftshift_deinterleave (Xc, Xr, Xi, M, N, C);
}

void
fft_stack (float* _Xc, int M, int N, int C, fftwf_plan *plans)
{
	fftwf_complex *Xc = (fftwf_complex*)_Xc;
	fftwf_execute_dft (plans[FFTW_FORWARD], Xc, Xc);

	for (int i = 0; i < 2*M*N*C; i++)
		_Xc[i] *= (float)(1.0/sqrt(M*N));
}

void
ifft_stack (float* _Xc, int M, int N, int C, fftwf_plan *plans)
{
	fftwf_complex *Xc = (fftwf_complex*)_Xc;
	fftwf_execute_dft (plans[FFTW_BACKWARD], Xc, Xc);

	for (int i = 0; i < 2*M*N*C; i++)
		_Xc[i] *= (float)(1.0/sqrt(M*N));
}

static void
pad_indices (int& i0, int& j0, int m2, int n2, int m, int n){
	i0 = (int) floor(m2/2.0) + (int) ceil (-m/2.0);
	j0 = (int) floor(n2/2.0) + (int) ceil (-n/2.0);
}

static void
zpad2d (float *pr, float *pi, int m2, int n2,
        float *xr, float *xi, int m,  int n,  int c)
{
	int i0, j0;
	pad_indices (i0, j0, m2, n2, m, n);

	memset (pr, 0, sizeof(*pr)*m2*n2*c);
	memset (pi, 0, sizeof(*pr)*m2*n2*c);

	for (int k = 0; k < c; k++){
		for (int j = 0; j < n; j++){
			for (int i = 0; i < m; i++){
				pr[i0+i + m2*(j0+j + n2*k)] = xr[i + m*(j + n*k)];
				pi[i0+i + m2*(j0+j + n2*k)] = xi[i + m*(j + n*k)];
			}
		}
	}
}

static void
crop2d (float *xr, float *xi,  int m,  int n,
        float *pr, float *pi, int m2, int n2, int c)
{
	int i0, j0;
	pad_indices (i0, j0, m2, n2, m, n);

	for (int k = 0; k < c; k++){
		for (int j = 0; j < n; j++){
			for (int i = 0; i < m; i++){
				xr[i + m*(j + n*k)] = pr[i0+i + m2*(j0+j + n2*k)];
				xi[i + m*(j + n*k)] = pi[i0+i + m2*(j0+j + n2*k)];
			}
		}
	}
}

void
soft_threshold (float *xr, float *xi, float lambda, int m, int n, int nc)
{
	for (int j = 0; j < n; j++){
		for (int i = 0; i < m; i++)
		{
			float absy, absx, s2 = 0, r;
			int ij = i+j*m;
			int nm = n*m;

			for (int c = 0; c < nc; c++){
				s2 += xr[ij + c*nm]*xr[ij + c*nm]
				    + xi[ij + c*nm]*xi[ij + c*nm];
			}

			absy = sqrt (s2);

			if ((absx = absy - lambda) < 0)
				absx = 0;

			r = absx/(absy+1e-6);

			for (int c = 0; c < nc; c++){
				xr[ij + c*nm] *= r;
				xi[ij + c*nm] *= r;
			}
		}
	}
}

static void debug_array (char const* str, int i,
                         float *t[2], int A, int B, int C, int D)
{
	char name[1024];
	sprintf(name, "%s%d", str, i);

	int ndims = 4;
	if (D < 0) ndims--, D=1;
	if (C < 0) ndims--, C=1;

	std::cerr << "debug_array: " << name << std::endl
		<< "\t A = " << A << std::endl
		<< "\t B = " << B << std::endl
		<< "\t C = " << C << std::endl
		<< "\t D = " << D << std::endl;

	int dims[4] = {A, B, C, D};

	write_bz2mat ((void**)t, false, ndims, dims, name);
}

}; // namespace pocs_omp
