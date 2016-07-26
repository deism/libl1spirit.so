
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
	void *data[2] = {mxGetData (pr[0]), mxGetImagData (pr[0])};
	bool isdouble = (mxGetClassID(pr[0]) == mxDOUBLE_CLASS);
	int ndims = mxGetNumberOfDimensions (pr[0]);
	int dims[ndims];

	for (int i = 0; i < ndims; i++)
		dims[i] = mxGetDimensions (pr[0])[i];

	char *key = newstrdup(L1SPIRIT);

	if (nr > 1){
		int strndims = mxGetNumberOfDimensions (pr[0]);
		mwSize *strdims = (mwSize*) mxGetDimensions (pr[0]);

		int strlen = 1;
		for (int i = 0; i < strndims; i++)
			strlen *= strdims[i];
		key = new char[strlen+1];
		mxGetString (pr[0], key, strlen+1);

	}

	if (write_shmmat (data, isdouble, ndims, dims, key)){
		mexErrMsgTxt ("write_shmmat() failure!");
	}

	delete[] key;
}
