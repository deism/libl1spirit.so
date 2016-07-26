
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#include "l1spirit3d.h"
#include "bz2mat.h"

#include "omp.h"

#include <armadillo>


typedef struct
{
  float real, imag;
} complex;

using namespace l1spirit3d::parameters;

/*****************************************************************************/
void convolution_matrix (complex *&A, int &m, int &n,
                    float *calib[2], int ksize[3], int csize[3], int coils);


extern double timestamp();



void
calib3d (float *k3d[2], int ksize[3],
         float *calib[2], int csize[3], int coils,
         float lambda)
{

	int ks1 = ksize[0], ks2 = ksize[1], ks3 = ksize[2];
	complex *A;
	int m, n;

	// Construct convolution matrix based on ACS data
	convolution_matrix (A, m, n, calib, ksize, csize, coils);
	
        //-------------CPU version of Calibration with Armadillo and MKL-------------------
	
	 // Matrix storage of convlution matrix
	 // Modified by JiaSen: store A in Armadillo matrix, then using Armadillo for AtA = At * A, and LSQR(AtA)
	
	arma::cx_fmat Aarma(m,n);
	std::complex<float> rc_arma;
	int ir = 0;
	#pragma omp parallel for shared( A, Aarma, m, n) private(ir)
	 for (ir = 0; ir < m; ir++){
	   for (int ic = 0; ic < n; ic++){
	     std::complex<float> rc_arma;
	     size_t idx = ir + ic*m;
	     rc_arma.real(A[idx].real);
	     rc_arma.imag(A[idx].imag);
	     Aarma(ir,ic) = rc_arma;
	   }
	 }
	 // Computing At * A 
	 arma::cx_fmat AtAarma = Aarma.t() * Aarma;
	 
        //------------------------------------------------------------------ */

        // Calibration via solving x = inv (At * A + lambda * I ) * y
	int c = 0;
#pragma omp parallel for shared(coils, ks1, ks2, ks3, m, n, AtAarma, lambda, k3d) private(c)
	for (c = 0; c < coils; c++)
	{       
		int idxy = (ks1/2) + ks1*((ks2/2) + ks2*(ks3/2 + ks3*c)); 

		arma::cx_fmat Marma (n-1,n-1);
		arma::cx_fvec yarma (n-1);
		
		for (int i = 0, ic = 0; i < n; i++){
		  if (i != idxy){
		    yarma(ic) = AtAarma(i,idxy);	    
		    for (int j = 0, jc = 0; j < n; j++){
		      if (j != idxy){

			Marma(ic,jc) = AtAarma(i,j);
		    
			jc++;
		      }
		    }
		    ic++;
		  }
		}
		
		float frob = 0;
		/*
		std::complex<float> l2norm = arma::accu(conj(Marma) % Marma);
		frob = sqrt(l2norm.real());
		*/
		std::complex<float> forbnorm = arma::norm(Marma,"fro");
		frob = std::abs(forbnorm);
		
		Marma.diag() += frob*lambda / (n-1);         // Tikhonov regularization
		
	       
		arma::cx_fmat xarma;
		xarma = arma::solve(Marma,yarma);
		
	// Copy the kernels out of the vector, inserting a zero in the center of
	// the appropriate kernel

		float *k3dc[2] = {&k3d[0][c*n], &k3d[1][c*n]};

		for (int i = 0, ic = 0; i < n; i++){
			if (i == idxy){
				k3dc[0][i] = k3dc[1][i] = 0;
			}
			else{

				std::complex<float> rc = xarma(ic);
				k3dc[0][i] = rc.real();
				k3dc[1][i] = rc.imag();

				ic++;
			}
		}

	}
  
	delete [] A;
}


void
convolution_matrix (complex *&A, int &m, int &n,
                    float *calib[2], int ksize[3], int csize[3], int coils)
{
	int ks1 = ksize[0], ks2 = ksize[1], ks3 = ksize[2];
	int cs1 = csize[0], cs2 = csize[1], cs3 = csize[2];

	int m1 = cs1-ks1+1, m2 = cs2-ks2+1, m3 = cs3-ks3+1;
	//int m1 = cs1-ks1, m2 = cs2-ks2, m3 = cs3-ks3;

	//m = coils * m1 * m2 * m3;
	m = m1 * m2 * m3;
	n = coils * ks1 * ks2 * ks3;

        A = new complex[m*n];
	/*for (int cc = 0; cc < coils; cc++)*/{
		for (int c3 = 0; c3 < m3; c3++){
			for (int c2 = 0; c2 < m2; c2++){
				for (int c1 = 0; c1 < m1; c1++)
				{
					//int row = c1 + m1*(c2 + m2*(c3 + m3*cc));
					int row = c1 + m1*(c2 + m2*c3);

					for (int c = 0; c < coils; c++){
						for (int i3 = c3; i3 < c3+ks3; i3++){
							for (int i2 = c2; i2 < c2+ks2; i2++){
								for (int i1 = c1; i1 < c1+ks1; i1++)
								{
									int col = (i1-c1) + ks1 *
									         ((i2-c2) + ks2 *
									         ((i3-c3) + ks3 * c));

									int idx = i1 + cs1*
									         (i2 + cs2*
									         (i3 + cs3 * c));

									complex rc = {calib[0][idx], calib[1][idx]};

									A[row + m*col] = rc;
								}
							}
						}
					}
				}
			}
		}
	}
}

