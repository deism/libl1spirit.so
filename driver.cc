/****************************************************************************
 * This driver file provides a command-line interface to the 3D l1-SPIRiT
 * reconstruction. It uses the routines in shmmat.cc for data input and output.
 *
 * $ l1spirit shmkey param1,param2=val2,param3,param4,...
 *
 * shmkey is the integer key for the shared memory region in which the input
 * data lies, and in which the output data will be written. The comma-separated
 * parameters (and optional assignment values) set the various parameters
 * available int he program.
 *
 * For example:
 *     $ ./l1spirt_recon data recon verbose,timing,lambda_l1=0.005
 * Will read input from data.bz2, write output to recon.bz2, print additional
 * verbose information, print out timing information as the program executes, 
 * and use a soft-thresholding lambda of 0.005 during the l1-minimzation.
 *
 *
 * Mark Murphy 2010 
 *****************************************************************************/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#include "shmmat.h"
#include "bz2mat.h"

#include <fstream>
#include <vector>
#include <string.h>
#include <boost/shared_ptr.hpp>


#include "l1spirit3d.h"
using namespace l1spirit3d::parameters;

extern double timestamp();

int
main (int argc, char **argv)
{
// SPIRiT Calibration Parameters
	float lambda_calib = 0.02;
	int calib_freqs = 20;
	int ksize[3] = {7,7,5};
	iters_calib = 100;

// L1-Minimization Paramters
        int n_iter_l1 = 50;
        float lambda_l1 = 0.0015;

	timing = true; verbose = true;
	calib_lsqr = false; calib_lsqr_ne = false;
	pocs_2fix = false;

// Debugging miscellany
	dump_raw = false, dump_kernel2d = false, dump_kernel3d=false,
	dump_ifreq = false,
	dump_pocs = false, dump_recon = false, dump_calib = false,
	dump_calibAtA = false, dump_calibA = false;
	kernel3dfile = 0, kernel2dfile = 0, calibfile = 0;
	bool forbid_calib_lsqr = false;

/******** Parse command line parameters ********/
	if (argc > 2){
		char *s = strdup (argv[2]), **ss = &s, *tok;

		while ((tok = strsep (ss, ",")))
		{
			char *t = strdup (tok), **tt = &t;

			char *name = strsep (tt, "=");
			char *val = strsep (tt, "=");

			if (!strcmp (name, "kernel3dfile"))
				kernel3dfile = strdup(val);

			if (!strcmp (name, "kernel2dfile"))
				kernel2dfile = strdup(val);

			if (!strcmp (name, "calibfile"))
				calibfile = strdup(val);

			if (!strcmp (name, "dump")){
				char *s = strdup (val), **ss = &s, *tok;
				while ((tok = strsep (ss, "+"))){
					if (!strcmp(tok, "raw"))
						dump_raw = true;

					if (!strcmp(tok, "kernel2d"))
						dump_kernel2d = true;
					if (!strcmp(tok, "kernel3d"))
						dump_kernel3d = true;

					if (!strcmp(tok, "ifreq"))
						dump_ifreq = true;

					if (!strcmp(tok, "pocs"))
						dump_pocs = true;

					if (!strcmp(tok, "recon"))
						dump_recon = true;

					if (!strcmp(tok, "calib"))
						dump_calib = true;

					if (!strcmp(tok, "calibAtA"))
						dump_calibAtA = true;

					if (!strcmp(tok, "calibA"))
						dump_calibA = true;

				}
				free (s);
			}

			#include "options.c"

			free (t);
		}
		free (s);
	}

	void *DATA[2];
	bool isdouble=0;
	int ndims, *d;

//        std::cout << "Reading data from shm ... ";

//        double read_timer = timestamp();

//        char const *shmkey = (!strcmp(argv[1], "default")) ? L1SPIRIT : argv[1];

//        std::cout << "shmkey: " << shmkey <<std::endl;

//        if (0 != read_shmmat (DATA, isdouble, ndims, d, shmkey)){
//                std::cerr << "ERROR reading shm key " << shmkey << std::endl;
//                return -1;
//        }

//        std::cout << (timestamp()-read_timer) << " s Elapsed" << std::endl;

//        if (!DATA[1]){
//                std::cerr << "ERROR -- input data is real-valued. Must be complex!"
//                        << std::endl;
//                return -1;
//        }
	
	


//         std::cout << "Reading data from bzip2 file ... ";
// 
//         double read_timer = timestamp();
// 
//         char const *bz2filename = "brain052mm";
// 
//         std::cout << "bz2filename: " << bz2filename <<std::endl;
// 
//         if (0 != read_bz2mat (DATA, isdouble, ndims, d, bz2filename)){
//                 std::cerr << "ERROR reading bz2 file " << bz2filename << std::endl;
//                 return -1;
//         }
// 
//         std::cout << (timestamp()-read_timer) << " s Elapsed" << std::endl;
// 
//         if (!DATA[1]){
//                 std::cerr << "ERROR -- input data is real-valued. Must be complex!"
//                         << std::endl;
//                 return -1;
//         }

        std::cout << "Reading data from cplx file ... ";

        double read_timer = timestamp();

        char const *cplxfilename = "Ismrmrd_Dump_data.cplx";

        std::cout << "cplxfilename: " << cplxfilename <<std::endl;

        int tmp;
        std::fstream f_input(cplxfilename,std::ios::in | std::ios::binary);
	
        if( !f_input.is_open() ){
          std::cout<<"ERROR: Cannot open file " << cplxfilename << std::endl;
          return -1;
        }
	
        int ndims_temp;
        f_input.read(reinterpret_cast<char*>(&ndims_temp),sizeof(int));
        int* d_temp = new int[ndims_temp];
	
        std::cout<<"dims: ";
        for (int i = 0; i < ndims_temp; i++)
        {
          f_input.read(reinterpret_cast<char*>(&tmp),sizeof(int));
          d_temp[i] = tmp;
          std::cout<<d_temp[i]<<" ";
        }
        std::cout<<std::endl;
        
        uint64_t N_temp = 1;
        for (int i = 0; i < ndims_temp; i++){
                N_temp *= d_temp[i];
        }
        uint64_t N_4 = d_temp[0]*d_temp[1]*d_temp[2]*d_temp[3];
	
        uint64_t N = 1;
        if(N_4 == N_temp)
        {
          ndims = 4;
          N = N_4;
          d = new int[ndims];
          for (int i = 0; i<ndims; i++){
            d[i] = d_temp[i];
          }
        }
        else{
          std::cout<<"Only 4D data is supported: [RO,PE1,PE2,CHA]!"<<std::endl;
          return -1;
        }

        if (isdouble)
        {
	  
          DATA[0] = (void*) new double[N];
          f_input.read(reinterpret_cast<char*>(DATA[0]),sizeof(double)*N);
          DATA[1] = (void*) new double[N];
          f_input.read(reinterpret_cast<char*>(DATA[1]),sizeof(double)*N);
	  
        }else{
	  
          DATA[0] = (void*) new float[N];
          f_input.read(reinterpret_cast<char*>(DATA[0]),sizeof(float)*N);
          DATA[1] = (void*) new float[N];
          f_input.read(reinterpret_cast<char*>(DATA[1]),sizeof(float)*N);

        }
        
        std::cout << (timestamp()-read_timer) << " s Elapsed" << std::endl;

        if (!DATA[1]){
                std::cerr << "ERROR -- input data is real-valued. Must be complex!"
                        << std::endl;
                return -1;
        }
  
        
        
        

	uint64_t freq_stride, n_freqs, freq_offset = 0;
	uint64_t phase_stride, n_phases, phase_offset = 0;
	uint64_t echo_stride, n_echos;
	uint64_t slice_stride, n_slices, slice_offset = 0;
	uint64_t coil_stride, n_coils, coil_offset = 0;

	switch (ndims){
	/* If the array is 5-dimensional, then we have some number (possibly 1)
     * of echos. The dimension ordering is:
	 *     (Frequency Encodes, Phase Encodes, Echos, Slice Encodes, Coils) */
	  case 5:
		 freq_stride = 1;                     n_freqs = d[0];
		phase_stride = d[0];                 n_phases = d[1];
		 echo_stride = d[0]*d[1];             n_echos = d[2];
		slice_stride = d[0]*d[1]*d[2];       n_slices = d[3];
		 coil_stride = d[0]*d[1]*d[2]*d[3];   n_coils = d[4];

		break;

	/* 4-D Arrays have the dimension ordering:
 	 *  (frequency encodes, phase encodes, slice encodes, coils) */
	  case 4:
		 echo_stride = 0;               n_echos = 1;
		 freq_stride = 1;               n_freqs = d[0];
		phase_stride = d[0];           n_phases = d[1];
		slice_stride = d[0]*d[1];      n_slices = d[2];
		 coil_stride = d[0]*d[1]*d[2];  n_coils = d[3];

		break;

	  default:
		std::cerr << "ERROR -- input dimension must be 4D or 5D" << std::endl;
		return -1;
	}

	float *data[2];

	if (isdouble){
		double *d[2] = {(double*)DATA[0], (double*)DATA[1]};

		uint64_t N = n_echos * n_freqs * n_phases * n_slices * n_coils;

		data[0] = new float[N];
		data[1] = new float[N];

		for (uint64_t i = 0; i < N; i++){
			data[0][i] = d[0][i];
			data[1][i] = d[1][i];
		}

		delete[] d[0];
		delete[] d[1];
	}
	else {
		data[0] = (float*)DATA[0];
		data[1] = (float*)DATA[1];
	}

	if (verbose){
		std::cout
			<< "Parameters:" << std::endl
#ifndef NO_CUDA
			<< "     max #gpus = " << max_gpus 
				<< " (for OpenMP recon, use gpus <= 0)" <<  std::endl
#endif
			<< "  lambda_calib = " << lambda_calib << std::endl

/*
	There are a number of options for the calibration code that are 
	not to be officially supported for this release of the code. Feel free
	to play with them! - MjM

			<< "    calib_lsqr = " << (calib_lsqr ? "yes" : "no") << std::endl
			<< " calib_lsqr_ne = " << (calib_lsqr_ne ? "yes" : "no") << std::endl
			<< "     pocs_2fix = " << (pocs_2fix ? "yes" : "no") << std::endl
			<< "   iters_calib = " << iters_calib << std::endl
*/
			<< "   calib_freqs = " << calib_freqs << std::endl
			<< "         ksize = [" << ksize[0] << ", " << 
			                           ksize[1] << ", " <<
			                           ksize[2] << "]" << std::endl
			<< "     n_iter_l1 = " << n_iter_l1 << std::endl
			<< "     lambda_l1 = " << lambda_l1 << std::endl
			<< "Data:" << std::endl
			<< "  frequency encodes : " << n_freqs << std::endl
			<< "      phase encodes : " << n_phases << std::endl
			<< "      slice encodes : " << n_slices << std::endl
			<< "           channels : " << n_coils << std::endl
			<< "              echos : " << n_echos << std::endl;
	}

	double ts = timestamp();
	for (uint64_t echo = 0; echo < n_echos; echo++)
	{
		double te = timestamp();

		float const *DE[2] = {
			data[0] + echo*echo_stride,
			data[1] + echo*echo_stride
		};

		float *RE[2] = {
			data[0] + echo*echo_stride,
			data[1] + echo*echo_stride
		};

		l1spirit3d::l1spirit3d (
			n_phases, n_slices, n_coils, n_freqs,
			phase_stride, slice_stride, coil_stride, freq_stride,
			phase_offset, slice_offset, coil_offset, freq_offset,
			DE, RE, ksize, calib_freqs, lambda_calib,
			n_iter_l1, lambda_l1);

		te = timestamp()-te;
		if (n_echos > 1 && (verbose || timing))
			std::cout << "  Echo " << echo << ": " << te << " s" << std::endl;
	}

	ts = timestamp()-ts;
	std::cout << "l1-SPIRiT runtime: " << ts << " seconds." << std::endl;

	double write_timer = timestamp();

//        std::cout << "Writing data out to shm ... ";
//        if (0 != write_shmmat ((void**)data, false, ndims, d, shmkey)){
//          std::cerr << "ERROR: failed writing shm key " << shmkey << std::endl;
//          return -1;
//        }
	
//        std::cout << "Writing data out to bz2 ... ";
//        char const *bz2reconfilename = "brain064mm_recon";
//        if (0 != write_bz2mat ((void**)data, false, ndims, d, bz2reconfilename)){
//          std::cerr << "ERROR: failed writing bz2 file " << bz2reconfilename << std::endl;
//          return -1;
//        }
                
        std::cout << "Writing data out to cplx file ... ";
        char const *cplxreconfilename = "Ismrmrd_Dump_Recon.cplx";
        int* header = new int[ndims+1];
	
        header[0] = static_cast<int>(ndims);
        for (int i = 0; i < header[0]; i++)
        {
          header[i+1] = static_cast<int>(d[i]);
        }
	
        std::fstream f(cplxreconfilename,std::ios::out | std::ios::binary);
	
        if( !f.is_open() ){
          std::cerr<< "ERROR: Cannot write file " << cplxreconfilename << std::endl;
          delete [] header;
          return -1;
        }
	
        N = 1;
        for (int i = 0; i < ndims; i++){
          N *= d[i];
          std::cout<<d[i]<<"  ";
        }
        std::cout<<std::endl;
        f.write(reinterpret_cast<char*>(header),sizeof(int)*(ndims+1));
        if (isdouble)
        {
          std::cout<<"Data type is double"<<std::endl;
          f.write(reinterpret_cast<char*>(data[0]),sizeof(double)*N);
          if (data[1])
          {
            std::cout<<data[1]<<std::endl;
            f.write(reinterpret_cast<char*>(data[1]),sizeof(double)*N);
          }
        }
        else
        {
          std::cout<<"Data type is float"<<std::endl;
          f.write(reinterpret_cast<char*>(data[0]),sizeof(float)*N);
          if (data[1])
          {
            f.write(reinterpret_cast<char*>(data[1]),sizeof(float)*N);
          }
        }
	
        f.close();
        delete [] header;
        delete [] data[0];
        delete [] data[1];



	std::cout << (timestamp()-write_timer) << " s elapsed" << std::endl;

	return 0;
}
