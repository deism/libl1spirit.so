
/****************************************************************************
 * readbz2() -- matlab binding for bzip2'ed matrix file reading. Usage in
 * Matlab:
 * >> readbz2 varname;
 * This will attempt to open the file ``varname.bz2'' and write it into the
 * Matlab variable ``varname''.
 ***************************************************************************/

#include <mex.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <bzlib.h>
#include <stdint.h>

#include "bz2mat.h"

extern "C" void
mexFunction (int nl, mxArray *pl[], int nr, mxArray const *pr[])
{
	int strndims = mxGetNumberOfDimensions (pr[0]);
	mwSize *strdims = (mwSize*) mxGetDimensions (pr[0]);

	int strlen = 1;
	for (int i = 0; i < strndims; i++)
		strlen *= strdims[i];
	char *str = new char[strlen+1];
	mxGetString (pr[0], str, strlen+1);

	void *data[2];
	bool isdouble;
	int ndims, *dims;

	if (0 != read_bz2mat (data, isdouble, ndims, dims, str))
	{
		char errmsg[1024];
		sprintf(errmsg, "Unable to read file %s.bz2\n", str);

		mexErrMsgTxt(errmsg); // Implicity returns to the Matlab prompt
	}

	bool iscomplex = (data[1] != 0); 

	mwSize *mwdims = new mwSize[ndims];
	for (int i = 0; i < ndims; i++)
		mwdims[i] = dims[i];

	mxArray *var = mxCreateNumericArray (
	                    ndims, mwdims,
	                    isdouble  ? mxDOUBLE_CLASS : mxSINGLE_CLASS,
	                    iscomplex ? mxCOMPLEX : mxREAL);

	uint64_t N = 1;
	for (int i = 0; i < ndims; i++)
		N *= dims[i];

	uint64_t elt_size = isdouble ? sizeof(double) : sizeof(float);

	memcpy (mxGetData(var), data[0], N*elt_size);

	if (iscomplex)
		memcpy (mxGetImagData(var), data[1], N*elt_size);

	mexPutVariable ("caller", str, var);

	delete[] str;
	delete[] mwdims;
	delete[] dims;

	if (isdouble){
		delete[] (double*) data[0];
		if (iscomplex)
			delete[] (double*) data[1];
	}
	else{
		delete[] (float*) data[0];
		if (iscomplex)
			delete[] (float*) data[1];
	}
}
