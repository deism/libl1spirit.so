

#include <mex.h>
#include <stdint.h>
#include <string.h>

#include "shmmat.h"

static char *newstrdup (char const *p){
	char *s = new char[strlen(p)+1];
	strcpy (s, p);
	return s;
}

extern "C" void
mexFunction (int nl, mxArray *pl[], int nr, mxArray const *pr[])
{
	char *key = newstrdup(L1SPIRIT);

	if (nr > 0){
		int strndims = mxGetNumberOfDimensions (pr[0]);
		mwSize *strdims = (mwSize*) mxGetDimensions (pr[0]);

		int strlen = 1;
		for (int i = 0; i < strndims; i++)
			strlen *= strdims[i];
		key = new char[strlen+1];
		mxGetString (pr[0], key, strlen+1);
	}

	void *data[2];
	bool isdouble;
	int ndims, *dims;

	if (get_shmmat (data, isdouble, ndims, dims, key)){
		mexErrMsgTxt ("get_shmmat() failure!");
	}

	uint64_t N = 1;
	mwSize mwdims[ndims];
	for (int i = 0; i < ndims; i++){
		N = N * dims[i];
		mwdims[i] = dims[i];
	}

	pl[0] = mxCreateNumericArray (ndims, mwdims,
	                 isdouble ? mxDOUBLE_CLASS : mxSINGLE_CLASS,
	                 data[1] == 0 ? mxREAL : mxCOMPLEX);

	memcpy (mxGetData (pl[0]), data[0],
	        N * (isdouble ? sizeof(double) : sizeof(float)));

	if (data[1] != 0){
		memcpy (mxGetImagData (pl[0]), data[1],
		        N * (isdouble ? sizeof(double) : sizeof(float)));
	}

	put_shmmat (data);

	delete[] key;
}
