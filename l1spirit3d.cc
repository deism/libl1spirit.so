/******************************************************************************
 *****************************************************************************/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>

#include <fftw3.h>

#ifndef NO_CUDA

#include "cuda_macros.h"
#include <cuda_runtime.h>

#endif

#include <pthread.h>
#include <omp.h>

#include "l1spirit3d.h"
#include "bz2mat.h"

#include <armadillo>


#include <sys/time.h>
extern double timestamp(){
	struct timeval tv;
	gettimeofday(&tv, 0);
	return tv.tv_sec + 1e-6*tv.tv_usec;
}


/******************************************************************************
 *****************************************************************************/

extern void calib3d (float *k3d[2], int ksize[3],
                          float *kcalib[2], int csize[3], int coils,
                          float lambda);

#ifndef NO_CUDA

namespace pocs_cuda {
extern void pocs_l1spirit2d (float *X[2], float* kc[2], int K,
                             float *D[2], int M, int N, int C,
                             int n_iter, float lambda);
};

#endif

namespace pocs_omp {
extern void pocs_l1spirit2d (float *X[2], float* kc[2], int K,
                             float *D[2], int M, int N, int C,
                             int n_iter, float lambda,
                             fftwf_plan*);
};

namespace l1spirit3d
{
	namespace parameters {
		bool
                        timing = true, // Print runtime information
                        verbose = true, // Print additional information
			calib_lsqr = false, // Use LSQR to solve calibration equations
			calib_lsqr_ne = false, // Use LSQR on the Normal equations
			pocs_2fix = false, // This one doesn't do anything
			dump_raw = false, // For debugging, dump the raw data to a .bz2
			dump_kernel3d = false,  // Dump 3D SPIRiT kernels to .bz2
                        dump_kernel2d = false, // Dump 2D SPIRiT kernels to .bz2
			dump_ifreq = false, // More debugging information ...
			dump_pocs = false,  // ...
			dump_recon = false, // ...
			dump_calib = false, // ...
			dump_calibAtA = false, // ...
			dump_calibA = false; // ...

		int iters_calib = 50,
			max_gpus = 16;

		char
			*kernel3dfile = 0,
			*kernel2dfile = 0,
			*calibfile = 0;
	};
	using namespace parameters;

void copy_zipped_data (bool& zipped, float *data[2], int& P, int& S, int& C, int& F,
	              float const *DATA[2],
                  long phases, long slices, long coils, long freqs, 
                  long ps, long ss, long cs, long fs,
	              long p0, long s0, long c0, long f0);
void replace_zipped_data (
                  float *RECON[2],int phases, int slices, int coils, int freqs,
                  long ps, long ss, long cs, long fs,
	              long p0, long s0, long c0, long f0,
	              bool zipped, float *data[2], long P, long S, long C, long F);
void scale_data (float& scale_factor, float *data[2], int P, int S, int C, int F);

void unscale_data (float *data[2], long P, long S, long C, long F, float scale_factor);

void ifft_freqencodes (float *data[2], int P, int S, int C, int F);

void fft_freqencodes (float *data[2], int P, int S, int C, int F);

void calibrate3d (float *kernel3d[2], int ksize3d[3], int coils, int calib_fe,
         float *data[2], int P, int S, int C, int F, float lambda, int iters);

#ifndef NO_CUDA

void gpu_2d_recons (float *data[2], int P, int S, int C, int F,
                    float *k2d[2], int *ksize3d, int n_iter_l1, float lambda_l1);

#endif

void omp_2d_recons (float *data[2], int P, int S, int C, int F,
                    float *k2d[2], int *ksize3d, int n_iter_l1, float lambda_l1);


/******************************************************************************
 *****************************************************************************/

void
l1spirit3d (
	int phases, int slices, int coils, int freqs, // data/recon sizes
	int ps, int ss, int cs, int fs, // data/recon strides
	int p0, int s0, int c0, int f0,
	float const *DATA[2], float *RECON[2], // data/recon
	int ksize3d[3], int calib_fe, float lambda_calib, // calibration parameters
	int n_iter_l1, float lambda_l1)
{
	float scale_factor;
	float *data[2], *k2d[2];
	int P, S, C, F;
	bool zipped;

	ksize3d[1] = ksize3d[0]; // HACK! FIX THIS


/******* Preliminary data structure transformation: permute the data array 
       * so that our routines have unit-stride access. **/ 

	double ts = timestamp();

	copy_zipped_data (zipped, data, P, S, C, F,
	                  DATA, phases, slices, coils, freqs,
	                  ps, ss, cs, fs, p0, s0, c0, f0);

	ts = timestamp()-ts;

	if (timing){
		std::cout << "  Timing info:" << std::endl;
		std::cout << "    Copying/Permuting: " << ts << " s" << std::endl;
	}

	if (dump_raw){
		static int counter=0; char name[1024]; 
		sprintf(name, "raw%d", counter++);
		int dims[4] = {P, S, C, F};
		write_bz2mat ((void**)data, false, 4, dims, name);
	}

/******* Rescale the data. This ensures that the pre-selected calibration
       * tychonov regularization constant and the l1-minimization thresholding
       * parameter have the correct values */

	ts = timestamp();
	scale_data (scale_factor, data, P, S, C, F);
	ts = timestamp()-ts;

	if (timing){
		std::cout << "            Rescaling: " << ts << " s" << std::endl;
	}

/******* Perform calibration. This produces a set of 2-dimensional k-space
       * convolution kernels for each frequency encode. */

	ts = timestamp();
	calibrate3d (k2d, ksize3d,  coils, calib_fe,
	             data, P, S, C, F, lambda_calib, iters_calib);

	ts = timestamp()-ts;
	if (timing)
		std::cout << "    3D Calibration: " << ts << " s" << std::endl;

	if (dump_kernel2d){
		static int counter=0; char name[1024];
		sprintf(name, "kernel2d%d", counter++);
		int dims[5] = {ksize3d[0], ksize3d[1], coils, coils, F};
		write_bz2mat ((void**)k2d, false, 5, dims, name);
	}

/******* Inverse FFT the data in the frequency-encode dimension. This decouples
       * all of the individual 2D reconstructions, enabling us to run them
       * in parallel */

	ts = timestamp();
	ifft_freqencodes (data, P, S, C, F);
	ts = timestamp()-ts;
	if (timing)
		std::cout << "    Fully Sampled IFFT: " << ts << " s" << std::endl;

	if (dump_ifreq){
		static int counter = 0; char name[1024];	
		sprintf(name, "ifreq%d", counter++);
		int dims[4] = {P, S, C, F};
		write_bz2mat ((void**)data, false, 4, dims, name);
	}

/****** Perform the 2-Dimensional SPIRiT reconstruction for each frequency
      * encode plane. */

#ifndef NO_CUDA
	if (max_gpus > 0){
		gpu_2d_recons (data, P, S, C, F, k2d, ksize3d, n_iter_l1, lambda_l1);
	}
	else
#endif
	{
		omp_2d_recons (data, P, S, C, F, k2d, ksize3d, n_iter_l1, lambda_l1);
	}

	if (dump_pocs){
		static int counter = 0; char name[1024];
		sprintf(name, "pocs%d", counter++);
		int dims[4] = {P, S, C, F};
		write_bz2mat((void**)data, false, 4, dims, name);
	}

/******* Re-Fourier transform the data in the frequency encode dimension. */

	ts = timestamp();
	fft_freqencodes (data, P, S, C, F);
	ts = timestamp()-ts;
	if (timing)
		std::cout << "    Fully Samplied FFT: " << ts << " s" << std::endl;

/******* Undo the previous scaling step to restore the data's original
       * magnitude*/

	ts = timestamp();
	unscale_data (data, P, S, C, F, scale_factor);
	ts = timestamp()-ts;
	if (timing)
		std::cout << "             Rescaling: " << ts << " s" << std::endl;

	if (dump_recon){
		static int counter = 0; char name[1024];
		sprintf(name, "recon%d", counter++);
		int dims[4] = {P, S, C, F};
		write_bz2mat((void**)data, false, 4, dims, name);
	}

/****** Copy the data back into the original array in the same order it 
      * was given to us */

	ts = timestamp();
	replace_zipped_data (RECON, phases, slices, coils, freqs,
	                     ps, ss, cs, fs, p0, s0, c0, f0,
	                     zipped, data, P, S, C, F);
	ts = timestamp()-ts;
	if (timing)
		std::cout << "    Permuting/Copying: " << ts << " s" << std::endl;
}

/******************************************************************************
 * Data Structure transformations -- i.e. array permutations
 *****************************************************************************/

void
copy_zipped_data (bool& zipped, float *data[2], int&_P, int&_S, int&_C, int&_F,
	              float const *DATA[2],
                  long phases, long slices, long coils, long freqs, 
                  long sp, long ss, long sc, long sf,
	              long p0, long s0, long c0, long f0)
{
	long zip_nz = 0;
#pragma omp parallel for reduction(+:zip_nz)
	for (long c = 0; c < coils; c++)
	{
		long zip_nz_c = 0;
		for (long f = 0; f < freqs; f++){
			for (long s = slices/2; s < slices; s++){
				for (long p = 0; p < phases; p++)
				{
					long idx = (p+p0)*sp + (s+s0)*ss + (c+c0)*sc + (f+f0)*sf;
					zip_nz_c += (DATA[0][idx] != 0 && DATA[1][idx] != 0);
				}
			}
		}
		zip_nz = zip_nz + zip_nz_c;
	}

	long P = phases; long S = slices; long C = coils; long F = freqs;
	_P=P; _S=S; _C=C; _F=F;

	if ((zipped = (zip_nz == 0))){
		S = slices/2;
	}

	data[0] = new float[2*F*C*S*P]; data[1] = data[0]+F*C*S*P;

#pragma omp parallel for
	for (long f = 0; f < F; f++)
	{
		for (long c = 0; c < C; c++){
			for (long s = 0; s < S; s++){
				for (long p = 0; p < P; p++)
				{
					long iD = (p+p0)*sp + (s+s0)*ss + (c+c0)*sc + (f+f0)*sf;
					/*long iD = 0;
					iD += (p+p0)*sp; iD += (s+s0)*ss;
					iD += (c+c0)*sc; iD += (f+f0)*sf;*/

					long id = p+P*(s+S*(c+C*f));
					/*long id = 0;
					id = p; id += P*s;
					id += P*S*c; id += P*S*C*f;*/

					data[0][id] = DATA[0][iD];
					data[1][id] = DATA[1][iD];
				}
			}
		}
	}
	
}

void
replace_zipped_data (
       float *RECON[2], int phases, int slices, int coils, int freqs,
       long sp, long ss, long sc, long sf,
       long p0, long s0, long c0, long f0,
       bool zipped, float *data[2], long P, long S, long C, long F)
{
#pragma omp parallel for
	for (long c = 0; c < C; c++){
		for (long f = 0; f < F; f++){
			for (long s = 0; s < S; s++){
				for (long p = 0; p < P; p++)
				{
					long idR = (p0+p)*sp + (s0+s)*ss + (c0+c)*sc + (f0+f)*sf; 
					long idd = p+P*(s+S*(c+C*f)); 

					RECON[0][idR] = data[0][idd];
					RECON[1][idR] = data[1][idd];
				}
			}
		}
	}

	delete[] data[0];
}

/******************************************************************************
 * Calibration and Data rescaling routines
 *****************************************************************************/

static int
pad_index (int M2, int M){
	return (int) floor(M2/2.0) + (int) ceil(-M/2.0);
}

static void
pad_indices (int& i0, int& j0, int M2, int N2, int M, int N){
	i0 = (int) floor(M2/2.0) + (int) ceil(-M/2.0);
	j0 = (int) floor(N2/2.0) + (int) ceil(-N/2.0);
}

void
calib_size (int& si, int& sj, float *mask, int M, int N)
{
	int fi = 0, fj = 0;

	si=2; sj=2;

	while (!fi || !fj){
		if (!fi){
			int i0, j0;
			pad_indices (i0, j0, M, N, si+1, sj);

			for (int j = j0; j < j0+sj; j++){
				for  (int i = i0; i < i0+si+1; i++){
					if (mask[i + M*j] == 0){
						fi = 1;
						goto fi_out;
					}
				}
			}
		fi_out:
			if (!fi){
				si++;	
			}
		}

		if (!fj){
			int i0, j0;
			pad_indices (i0, j0, M, N, si, sj+1);

			for (int j = j0; j < j0+sj+1; j++){
				for (int i = i0; i < i0+si; i++){
					if (mask[i + M*j] == 0){
						fj = 1;
						goto fj_out;
					}
				}
			}
		fj_out:
			if (!fj){
				sj++;
			}
		}
	}
}

void
density_compensation
	(float *dcomp, float *sampling, int csize[2], int phases, int slices)
{
	int i0, j0;
	pad_indices (i0, j0, phases, slices, csize[0], csize[1]);

	int ie = i0+csize[0], je = j0+csize[1];

	float Rn = 0, Rd = 0;
	for (int j = 0; j < slices; j++){
		for (int i = 0; i < phases; i++){
			double ii = (i == phases-1) ? 1.0 : (-1.0 + 2.0*i/(phases-1.0));
			double jj = (j == slices-1) ? 1.0 : (-1.0 + 2.0*j/(slices-1.0));
			double r = sqrt(ii*ii+jj*jj);

			if (r <= 1.0){
				if (!(i >= i0 && i < ie &&j >= j0 && j < je)){
					Rd += 1.0;
					Rn += sampling[i + phases*j];
				}
			}
		}
	}

	for (int j = 0; j < slices; j++){
		for (int i = 0; i < phases; i++){
			if (i >= i0 && i < ie &&j >= j0 && j < je){
				dcomp[i + phases*j] = 1.0;
			}
			else{
				dcomp[i + phases*j] = 1.0f / (Rn/Rd);
			}
		}
	}
}

static void
sampling_pattern (float *mask, float *d[2], int P, int S, int C, int F)
{
	for (int s = 0; s < S; s++){
		for (int p = 0; p < P; p++){
			mask[p + P*s] = 0.0f;
		}
	}

	float *r = d[0], *i = d[1];

	for (int f = F/2-10; f < F/2+10; f++)
	{
		for (int c = 0; c < C; c++){
			for (int s = 0; s < S; s++){
				for (int p = 0; p < P; p++)
				{
					int idx = p + P*(s + S*(c + C*f));
					mask[p+P*s] += (i[idx] != 0) && (r[idx] != 0);
				}
			}
		}

	}

	for (int s = 0; s < S; s++){
		for (int p = 0; p < P; p++){
			mask[p + P*s] = (mask[p+P*s] > 0.0f);
		}
	}
}


void
scale_data (float& scale_factor, float *data[2], int P, int S, int C, int F)
{
	float *sampling = new float[P*S];

	sampling_pattern (sampling, data, P, S, C, F);

	int csize[2];
	calib_size (csize[0], csize[1], sampling, P, S);

	float *dcomp = new float[P*S];
	density_compensation (dcomp, sampling, csize, P, S);

	float *ch1[2];
	ch1[0] = new float[P*S*F*2]; ch1[1] = ch1[0]+P*S*F;

	for (int f = 0; f < F; f++){
		for (int c = 0; c < C; c++){
			for (int s = 0; s < S; s += 2){
				for (int p = 0; p < P; p++){
					int i = p+P*(s+S*(c+C*f));
					data[0][i] = -data[0][i];
					data[1][i] = -data[1][i];
				}
			}
		}
	}

	for (int f = 0; f < F; f++){
		for (int s = 0; s < S; s++){
			for (int p = 0; p < P; p++){
				ch1[0][p+P*(s+S*f)] = data[0][p+P*(s+S*C*f)]*dcomp[p+P*s];
				ch1[1][p+P*(s+S*f)] = data[1][p+P*(s+S*C*f)]*dcomp[p+P*s];
			}
		}
	}

	fftwf_iodim dims[3] = {
			{F, P*S, P*S},
			{S, P,   P},
			{P, 1,   1}
		};

	fftwf_plan p = fftwf_plan_guru_split_dft (
			3, dims, 0, 0,  // howmany_rank = 0 -> do 1 transform
			ch1[1], ch1[0], // For inverse FFT with the guru_split iterface,
			ch1[1], ch1[0], // switch the real and imaginary pointers
			FFTW_ESTIMATE);
	fftwf_execute(p);
	fftwf_destroy_plan(p);

	float maxabs2 = 0;
	for (int i = 0; i < P*S*F; i++){
		float abs2 = ch1[0][i]*ch1[0][i] + ch1[1][i]*ch1[1][i];
		if (abs2 > maxabs2)
			maxabs2 = abs2;
	}

	scale_factor = (1.0f / sqrtf(P*S*F)) * sqrtf(maxabs2);

	for (int i = 0; i < P*S*C*F; i++){
		data[0][i] /= scale_factor;
		data[1][i] /= scale_factor;
	}

	delete[] sampling;
	delete[] dcomp;
	delete[] ch1[0];
}

void
unscale_data (float *data[2], long P, long S, long C, long F, float scale_factor)
{
	for (int f = 0; f < F; f++){
		for (int c = 0; c < C; c++){
			for (int s = 0; s < S; s += 2){
				for (int p = 0; p < P; p++){
					int i = p+P*(s+S*(c+C*f));
					data[0][i] = -data[0][i];
					data[1][i] = -data[1][i];
				}
			}
		}
	}

	for (long f = 0; f < F; f++){
		for (long c = 0; c < C; c++){
			for (long s = 0; s < S; s++){
				for (long p = 0; p < P; p++){
					data[0][p+P*(s+S*(c+C*f))] *= scale_factor;
					data[1][p+P*(s+S*(c+C*f))] *= scale_factor;
				}
			}
		}
	}
}

/******************************************************************************
 * Calibration
 *****************************************************************************/

void
calibrate3d (float *k2d[2], int ksize[3], int coils, int calib_fe,
        float *data[2], int P, int S, int C, int F, float lambda, int iters)
{
	int f0 = -1, ks1 = ksize[0], ks2 = ksize[1], ks3 = ksize[2];

	if (kernel2dfile){
		bool isdbl = false;
		int ndims, *dims;
		read_bz2mat ((void**)k2d, isdbl, ndims, dims, kernel2dfile);

		if (ndims != 5 || dims[0] != ksize[0] || dims[1] != ksize[1]
		 || dims[2] != C || dims[3] != C || dims[4] != F || isdbl)
		{
			std::cerr << "ERROR: 2D Kernels read from " << kernel2dfile 
				<< " invalid: isdouble=" << isdbl << " ndims=" << ndims
				<< " dims={ ";
			for (int i = 0; i < ndims; i++){
				std::cerr << dims[i];
				if ( i < ndims-1)
					std::cerr << " x ";
			}
			std::cerr << " }" << std::endl;
		}
		else{
			std::cerr << "2D Kernels read from " << kernel2dfile << " OK"
				<< std::endl;
			return;
		}
	}

	float *k3d[2] = {0,0};

	if (kernel3dfile){
		bool isdbl = false;
		int ndims, *dims;
		read_bz2mat ((void**)k3d, isdbl, ndims, dims, kernel3dfile);

		if (ndims != 5 || dims[0] != ksize[0] || dims[1] != ksize[1] 
		 || dims[2] != ksize[2] || dims[3] != C || dims[4] != C)
		{
			std::cerr << "ERROR: 3D Kernels read from " << kernel3dfile 
				<< " invalid: isdouble=" << isdbl << " ndims=" << ndims
				<< " dims={ ";
			for (int i = 0; i < ndims; i++){
				std::cerr << dims[i];
				if ( i < ndims-1)
					std::cerr << " x ";
			}
			std::cerr << " }" << std::endl;
		}
		else{
			std::cerr << "3D kernels read from " << kernel3dfile << " OK"
				<< std::endl;
			goto ifft_k3d_freqencodes;
		}
	}

	int csize[3];
	float *calib[2];
	float *sampling;

	sampling = new float[P*S];

	sampling_pattern (sampling, data, P, S, C, F);
	calib_size (csize[0], csize[1], sampling, P, S);
	csize[2] = calib_fe;

	delete[] sampling;

	if (calibfile){
		bool isdbl = false;
		int ndims, *dims;
		read_bz2mat ((void**)calib, isdbl, ndims, dims, calibfile);

		if (isdbl || ndims != 4 || dims[0] != csize[0] || dims[1] != csize[1] 
		 || dims[2] != csize[2] || dims[3] != C)
		{
			std::cerr << "ERROR: Calib data read from " << calibfile 
				<< " invalid: isdouble=" << isdbl << " ndims=" << ndims
				<< " dims={ ";
			for (int i = 0; i < ndims; i++){
				std::cerr << dims[i];
				if ( i < ndims-1)
					std::cerr << " x ";
			}
			std::cerr << " }" << std::endl;
		}
		else{
			std::cerr << "Calibration data read from " << calibfile << " OK"
				<< std::endl;

			goto compute_calib;
		}
	}

	// Center the calibration data over the frequency encodes
	// with the most energy (~~ SNR)

	float max_sos2; int max_f;
	max_sos2 = 0; max_f = -1;

	for (int f = 0; f < F; f++){
		for (int s = 0; s < S; s++){
			for (int p = 0; p < P; p++){
				float sos2 = 0;
				for (int c = 0; c < C; c++)
				{
					float dr = data[0][p+P*(s+S*(c+C*f))],
					      di = data[1][p+P*(s+S*(c+C*f))];
					sos2 += dr*dr + di*di;
				}
				if (sos2 > max_sos2){
					max_sos2 = sos2;
					max_f = f;
				}
			}
		}
	}

	int p0, s0;
	pad_indices (p0,s0, P, S, csize[0], csize[1]);

	calib[0] = new float [2*csize[0]*csize[1]* csize[2] * C];
	calib[1] = calib[0] + csize[0]*csize[1]*csize[2]*C;

// Watch out -- the calibration array has a different index ordering
// than the data array: coils are last

	for (int f = 0, ff = max_f-calib_fe/2; f < calib_fe; f++, ff++){
		for (int c = 0; c < coils; c++){
			for (int s = 0; s < csize[1]; s++){
				for (int p = 0; p < csize[0]; p++)
				{
					int idc = p + csize[0]*(s + csize[1]*(f + calib_fe*c));
					int idd = p0+p+P*(s0+s+S*(c+C*ff));

					calib[0][idc] = data[0][idd];
					calib[1][idc] = data[1][idd];
				}
			}
		}
	}

	if (dump_calib){
		static int counter = 0;
		int dims[4] = {csize[0], csize[1], calib_fe, coils};
		char name[1024];
		sprintf(name, "calib%d", counter++);
		write_bz2mat ((void**)calib, false, 4, dims, name);
	}

compute_calib:

	k3d[0] = new float [2*ksize[0]*ksize[1]*ksize[2]*coils*coils];
	k3d[1] = k3d[0] + ksize[0]*ksize[1]*ksize[2]*coils*coils;

	calib3d (k3d, ksize, calib,  csize, coils, lambda);

	if (dump_kernel3d){
		static int counter = 0;
		int dims[5] = {ksize[0], ksize[1], ksize[2], coils, coils};
		char name[1024]; sprintf(name, "kernel3d%d", counter++);
		write_bz2mat ((void**)k3d, false, 5, dims, name);
	}

	delete[] calib[0];

ifft_k3d_freqencodes:

	k2d[0] = new float[2*ksize[0]*ksize[1]*coils*coils*F];
	k2d[1] = k2d[0]+ksize[0]*ksize[1]*coils*coils*F;

	for (int i = 0; i < 2*ksize[0]*ksize[1]*coils*coils*F; i++)
		k2d[0][i] = 0;

	f0 = pad_index (F, ks3);

	for (int k3 = ks3-1, f = f0; f < f0+ks3; k3--, f++){
		for (int c2 = 0; c2 < C; c2++){
			for (int c1 = 0; c1 < C; c1++){
				for (int k2 = 0; k2 < ks2; k2++){
					for (int k1 = 0; k1 < ks1; k1++)
					{
						int i3d, i2d;

						i3d = k1 + ks1*(k2 + ks2*(k3 + ks3*(c1 + C*c2)));
						i2d = k1 + ks1*(k2 + ks2*(c1 + C*(c2 + C*f)));

						k2d[0][i2d] = sqrt(F) * k3d[0][i3d];
						k2d[1][i2d] = sqrt(F) * k3d[1][i3d];
					}
				}
			}
		}
	}

	delete[] k3d[0];

	float *y[2] = {new float[ks1*ks2*C*C*F], new float[ks1*ks2*C*C*F]};
	int *idx5 = new int[F];

	{int i = 0;
	 int p = (int) floor((1.0*F)/2.0);
	 for (; p < F; i++, p++)
		idx5[i] = p;
	 for (int j = 0; i < F; i++, j++)
		idx5[i] = j;
	}

	for (int i5 = 0; i5 < F; i5++){
		for (int i4 = 0; i4 < C; i4++){
			for (int i3 = 0; i3 < C; i3++){
				for (int i2 = 0; i2 < ks2; i2++){
					for (int i1 = 0; i1 < ks1; i1++)
					{
						int iy = i1 + ks1*(i2 + ks2*(i3 + C*(i4 + C*i5)));
						int ix = i1 + ks1*(i2 + ks2*(i3 + C*(i4 + C*idx5[i5])));
						y[0][iy] = k2d[0][ix];
						y[1][iy] = k2d[1][ix];
					}
				}
			}
		}
	}

	fftwf_iodim xform[1] = {{F, ks1*ks2*C*C, ks1*ks2*C*C}};
	fftwf_iodim vectr[1] = {{ks1*ks2*C*C, 1, 1,}};

	fftwf_plan p = fftwf_plan_guru_split_dft(
			1, xform, 1, vectr,
			y[1], y[0],
			y[1], y[0],
			FFTW_ESTIMATE);

	fftwf_execute(p);
	fftwf_destroy_plan(p);

	{int i = 0;
	 int p = (int) ceil((1.0*F)/2.0);
	 for (; p < F; i++, p++)
		idx5[i] = p;
	 for (int j = 0; i < F; i++, j++)
		idx5[i] = j;
	}

	for (int i5 = 0; i5 < F; i5++){
		for (int i4 = 0; i4 < C; i4++){
			for (int i3 = 0; i3 < C; i3++){
				for (int i2 = 0; i2 < ks2; i2++){
					for (int i1 = 0; i1 < ks1; i1++)
					{
						int iy = i1 + ks1*(i2 + ks2*(i3 + C*(i4 + C*idx5[i5])));
						int ix = i1 + ks1*(i2 + ks2*(i3 + C*(i4 + C*i5)));

						k2d[0][ix] = y[0][iy] / sqrt(F);
						k2d[1][ix] = y[1][iy] / sqrt(F);
					}
				}
			}
		}
	}

	delete[] y[0]; delete[] y[1];
	delete[] idx5;
}

/******************************************************************************
 * Forward and Backward fourier transforms along the frequency encode dimension
 *****************************************************************************/

void
dft_freqencodes (float *d0, float *d1, int P, int S, int C, int F)
{
	int *idx5 = new int[F];
	float *y0 = new float[P*S*C*F], *y1 = new float[P*S*C*F];

	{int i = 0;
	 int p = (int) floor((1.0*F)/2.0);
	 for (; p < F; i++, p++)
		idx5[i] = p;
	 for (int j = 0; i < F; i++, j++)
		idx5[i] = j;
	}

	for (int f = 0; f < F; f++){
		for (int i = 0; i < P*S*C; i++){
			y0[i + P*S*C*f] = d0[i + P*S*C*idx5[f]];
			y1[i + P*S*C*f] = d1[i + P*S*C*idx5[f]];
		}
	}

	fftwf_iodim xform[1] = {{F, P*S*C, P*S*C}};
	fftwf_iodim vectr[1] = {{P*S*C, 1, 1}};

	fftwf_plan p = fftwf_plan_guru_split_dft (
			1, xform, 1, vectr,
			y0, y1, y0, y1,
			FFTW_ESTIMATE);
	fftwf_execute(p);
	fftwf_destroy_plan(p);

	{int i = 0;
	 int p = (int) ceil((1.0*F)/2.0);
	 for (; p < F; i++, p++)
		idx5[i] = p;
	 for (int j = 0; i < F; i++, j++)
		idx5[i] = j;
	}

	for (int f = 0; f < F; f++){
		for (int i = 0; i < P*S*C; i++){
			d0[i + P*S*C*f] = y0[i + P*S*C*idx5[f]] / sqrt(F);
			d1[i + P*S*C*f] = y1[i + P*S*C*idx5[f]] / sqrt(F);
		}
	}

	delete[] idx5; delete[] y0; delete[] y1;
}

void
fft_freqencodes (float *data[2], int P, int S, int C, int F){
	dft_freqencodes (data[0], data[1], P, S, C, F);
}

void
ifft_freqencodes (float *data[2], int P, int S, int C, int F){
	dft_freqencodes (data[1], data[0], P, S, C, F);
}

/******************************************************************************
 * 2D SPIRiT reconstruction routines. There are two implementations, the first
 * of which parallelizes the 2-D reconstructions across multiple GPUs in a
 * multi-GPU system (and in turn parallelizes each 2D recon within the GPU).
 * The second implementation is slower, but does not require that a Cuda-capable
 * GPU be present.
 *****************************************************************************/

#ifndef NO_CUDA

#if defined(SINGLE_GPU_RECON)

void
gpu_2d_recons (float *data[2], int P, int S, int C, int F,
                   float *k2d[2], int *ksize3d, int n_iter_l1, float lambda_l1)
{
	float *X[2], *kc[2], *D[2];
	double ts = timestamp();

	int FE_size = P*S*C;

	cuda (Malloc (X, 2*FE_size*sizeof(float))); X[1] = X[0] + FE_size;
	cuda (Malloc (D, 2*FE_size*sizeof(float))); D[1] = D[0] + FE_size;

	int FE_ksize = ksize3d[0]*ksize3d[1]*C*C;
	cuda (Malloc (kc, 2*FE_ksize*sizeof(float))); kc[1] = kc[0] + FE_ksize;

	for (int f = 0; f < F; f++)
	{
		double ts = timestamp();
		if (timing)
			std::cout << "    FE " << f << " ... " << std::flush;

		cuda (Memcpy (D[0], &data[0][f*FE_size], FE_size*sizeof(float),
		              cudaMemcpyHostToDevice));
		cuda (Memcpy (D[1], &data[1][f*FE_size], FE_size*sizeof(float),
		              cudaMemcpyHostToDevice));

		cuda (Memcpy (X[0], D[0], FE_size*sizeof(float),
		              cudaMemcpyDeviceToDevice));

		cuda (Memcpy (X[1], D[1], FE_size*sizeof(float),
		              cudaMemcpyDeviceToDevice));

		cuda (Memcpy (kc[0], &k2d[0][f*FE_ksize], FE_ksize*sizeof(float),
		              cudaMemcpyHostToDevice));
		cuda (Memcpy (kc[1], &k2d[1][f*FE_ksize], FE_ksize*sizeof(float),
		              cudaMemcpyHostToDevice));


		pocs_l1spirit2d (X, kc, ksize3d[0], D, P,S,C, n_iter_l1, lambda_l1);

		cuda (Memcpy (&data[0][f*FE_size], X[0], FE_size*sizeof(float),
		              cudaMemcpyDeviceToHost));
		cuda (Memcpy (&data[1][f*FE_size], X[1], FE_size*sizeof(float),
		              cudaMemcpyDeviceToHost));

		if (timing)
			std::cout << (timestamp()-ts) << " s                            \r";
	}
	if (timing)
		std::cout << std::endl;

	cuda (Free (X[0]));
	cuda (Free (D[0]));
	cuda (Free (kc[0]));

	ts = timestamp()-ts;
	if (timing)
		std::cout << "    l1-SPIRiT POCS: " << ts << " s" << std::endl;
}

#else // Multi-GPU recon

struct recon2d_info {
	float *data[2];
	int P, S, C, F, F0, FN;
	float *k2d[2];
	int *ksize3d;
	int n_iter_l1;
	float lambda_l1;
	int device;
	double compute_time;
};

void *
recons2d_thread (void * _info)
{
	struct recon2d_info *info = (recon2d_info*)_info;

	float *data[2] = {info->data[0], info->data[1]};
	int P = info->P, S = info->S, C = info->C, F0 = info->F0, FN = info->FN;
	float *k2d[2] = {info->k2d[0], info->k2d[1]};
	int *ksize3d = info->ksize3d, n_iter_l1 = info->n_iter_l1;
	float lambda_l1 = info->lambda_l1;

	cuda (SetDevice (info->device));

	float *X[2], *kc[2], *D[2];

	int FE_size = P*S*C;

	cuda (Malloc (X, 2*FE_size*sizeof(float))); X[1] = X[0] + FE_size;
	cuda (Malloc (D, 2*FE_size*sizeof(float))); D[1] = D[0] + FE_size;

	int FE_ksize = ksize3d[0]*ksize3d[1]*C*C;
	cuda (Malloc (kc, 2*FE_ksize*sizeof(float))); kc[1] = kc[0] + FE_ksize;

	info->compute_time = 0;

	for (int f = F0; f < FN; f++)
	{
		double ts = timestamp();
//		if (timing)
//			std::cout << "    FE " << f << " ... " << std::flush;

		cuda (Memcpy (D[0], &data[0][f*FE_size], FE_size*sizeof(float),
		              cudaMemcpyHostToDevice));
		cuda (Memcpy (D[1], &data[1][f*FE_size], FE_size*sizeof(float),
		              cudaMemcpyHostToDevice));

		cuda (Memcpy (X[0], D[0], FE_size*sizeof(float),
		              cudaMemcpyDeviceToDevice));

		cuda (Memcpy (X[1], D[1], FE_size*sizeof(float),
		              cudaMemcpyDeviceToDevice));

		cuda (Memcpy (kc[0], &k2d[0][f*FE_ksize], FE_ksize*sizeof(float),
		              cudaMemcpyHostToDevice));
		cuda (Memcpy (kc[1], &k2d[1][f*FE_ksize], FE_ksize*sizeof(float),
		              cudaMemcpyHostToDevice));

		pocs_cuda::pocs_l1spirit2d (X, kc, ksize3d[0], D, P,S,C, n_iter_l1, lambda_l1);

		cuda (Memcpy (&data[0][f*FE_size], X[0], FE_size*sizeof(float),
		              cudaMemcpyDeviceToHost));
		cuda (Memcpy (&data[1][f*FE_size], X[1], FE_size*sizeof(float),
		              cudaMemcpyDeviceToHost));

		info->compute_time += (timestamp()-ts);
//		if (timing)
//			std::cout << (timestamp()-ts) << " s                            \r";
	}
//	if (timing)
//		std::cout << std::endl;

	cuda (Free (X[0]));
	cuda (Free (D[0]));
	cuda (Free (kc[0]));

	return 0;
}

void
gpu_2d_recons (float *data[2], int P, int S, int C, int F,
               float *k2d[2], int *ksize3d, int n_iter_l1, float lambda_l1)
{

	double ts = timestamp();

	int n_gpus;

	cuda (GetDeviceCount (&n_gpus));

	if (max_gpus > 0 && n_gpus > max_gpus)
		n_gpus = max_gpus;

	struct recon2d_info info[n_gpus];
	pthread_t threads[n_gpus];

	// int Fp = (F + n_gpus-1) / n_gpus;
        int Fp = int(F*1.2/2);               // Modified by JiaSen for two GPUs with different computing power/cuda cores

	for (int i = 0; i < n_gpus; i++)
	{
		info[i].data[0] = data[0];
		info[i].data[1] = data[1];

		info[i].P = P;
		info[i].S = S;
		info[i].C = C;
		info[i].F = F;

		info[i].F0 = Fp*i;
		info[i].FN = (i == n_gpus-1) ? F : Fp*(i+1);
		info[i].device = i;

		info[i].k2d[0] = k2d[0];
		info[i].k2d[1] = k2d[1];
		info[i].ksize3d = ksize3d;

		info[i].n_iter_l1 = n_iter_l1;
		info[i].lambda_l1 = lambda_l1;

		if (i > 0){
			if (pthread_create (&threads[i], NULL, recons2d_thread, &info[i])){
				perror("pthread_create");
			}
		}
	}

	recons2d_thread (&info[0]);

	for (int i = 1; i < n_gpus; i++){
		void *ret;
		pthread_join (threads[i], &ret);
	}

	ts = timestamp()-ts;
	if (timing){
		std::cout << "    l1-SPIRiT POCS: " << ts << " s" << std::endl;
		for (int i = 0; i < n_gpus; i++){
			std::cout << "        GPU " << i << " : "
				<< info[i].compute_time << " s" << std::endl;
		}
	}	
}
#endif // Multi-GPU recon
#endif // NO_CUDA

void omp_2d_recons (float *data[2], int P, int S, int C, int F,
                    float *k2d[2], int *ksize3d, int n_iter_l1, float lambda_l1)
{
	double ts = timestamp();

	int FE_size = P*S*C;
	int FE_ksize = ksize3d[0]*ksize3d[1]*C*C;

	fftwf_plan _stack_plans[3]={0,0,0}, *stack_plans = _stack_plans+1;
	double *exec_times;

#pragma omp parallel
{
	float *X[2], *kc[2], *D[2];
	X[0] = new float[2*FE_size]; X[1] = X[0]+FE_size;
	D[0] = new float[FE_size]; D[1] = new float[FE_size];
	kc[0] = new float[FE_ksize]; kc[1] = new float[FE_ksize];

/* FFTW is not thread safe. The only thread-safe functions are the actual
 * fftw_execute() functions.  In particular we cannot let threads plan their
 * own dfts, since the concurrent calls to the FFTW planner will crash. So,
 * plan the DFTs here, and pass the plans along to the individual 2d recons */

	#pragma omp single
	{
		fftwf_iodim dims[2] = {
			{S, P, P},
			{P, 1, 1}};
		fftwf_iodim vect[1] = {
			{C, P*S, P*S}};

		fftwf_plan_with_nthreads (1); // Let's see how 1 thread works :-D

		fftwf_complex *Xc = (fftwf_complex*)X[0];

	/* This is a little cheesy -- I'm relying on the fact that
	 * BACKWARD/FORWARD are +/- 1. */

		stack_plans[FFTW_FORWARD] = fftwf_plan_guru_dft (2, dims, 1, vect,
		                                                 Xc, Xc,
		                                                 FFTW_FORWARD,
		                                                 FFTW_ESTIMATE);

		stack_plans[FFTW_BACKWARD] = fftwf_plan_guru_dft (2, dims, 1, vect,
		                                                 Xc, Xc,
		                                                 FFTW_BACKWARD,
		                                                 FFTW_ESTIMATE);

		exec_times = new double[omp_get_num_threads()];
		for (int i = 0; i < omp_get_num_threads(); i++)
			exec_times[i] = 0;
	}

	#pragma omp barrier

	#pragma omp for schedule(dynamic,1)
	for (int f = 0; f < F; f++)
	{
		double ts = timestamp();

	/* FIXME */ //f = F/2;

	/* memcpying is technically unnecessary, but a small overhead that
	 * assists in debugging */

		memcpy (D[0], &data[0][f*FE_size], FE_size*sizeof(float));
		memcpy (D[1], &data[1][f*FE_size], FE_size*sizeof(float));

		memcpy (X[0], D[0], FE_size*sizeof(float));
		memcpy (X[1], D[1], FE_size*sizeof(float));

		memcpy (kc[0], &k2d[0][f*FE_ksize], FE_ksize*sizeof(float));
		memcpy (kc[1], &k2d[1][f*FE_ksize], FE_ksize*sizeof(float));

		pocs_omp::pocs_l1spirit2d (X, kc, ksize3d[0], D, P,S,C,
		                           n_iter_l1, lambda_l1, stack_plans);

		memcpy (&data[0][f*FE_size], X[0], FE_size*sizeof(float));
		memcpy (&data[1][f*FE_size], X[1], FE_size*sizeof(float));

		exec_times[omp_get_thread_num()] +=	(timestamp()-ts);
	}

	delete[] X[0]; 
	delete[] D[0]; delete[] D[1];
	delete[] kc[0]; delete[] kc[1];

	#pragma omp barrier
	#pragma omp single
	{
		fftwf_destroy_plan (stack_plans[FFTW_FORWARD]);
		fftwf_destroy_plan (stack_plans[FFTW_BACKWARD]);

		if (timing)
			for (int i = 0; i < omp_get_num_threads(); i++){
				std::cout << "        OMP Thread " << i << ": " << 
					exec_times[i] << " s" << std::endl;
			}

		delete[] exec_times;
	}
} // omp parallel

	ts = timestamp()-ts;
	if (timing)
		std::cout << "    l1-SPIRiT POCS: " << ts << " s" << std::endl;
}


}; // namespace l1spirit3d

