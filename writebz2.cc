
/****************************************************************************
 * writebz2() -- matlab binding for bzip2'ed matrix file writing. Usage in
 * Matlab:
 * >> writebz2 varname;
 * This will attempt to open the file ``varname.bz2'' and write into it the
 * contents  of the Matlab variable ``varname''.
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
	const mwSize *strdims = mxGetDimensions (pr[0]);

	int strlen = 1;
	for (int i = 0; i < strndims; i++)
		strlen *= strdims[i];

	char *str = new char[strlen+1];
	mxGetString (pr[0], str, strlen+1);

	mxArray const *var = mexGetVariablePtr("caller", str);

	if (!var){
		char errmsg[1024];
		sprintf(errmsg, "Variable \"%s\" does not exist!", str);
		delete[] str;
		mexErrMsgTxt (errmsg); // Implicitly returns to Matlab prompt
	}

	mwSize mwndims = mxGetNumberOfDimensions(var);
	mwSize const *mwdims = mxGetDimensions(var);

	int ndims = (int) mwndims;
	int *dims = new int[ndims];
	for (int i = 0; i < ndims; i++)
		dims[i] = mwdims[i];

	bool isdouble;

	switch (mxGetClassID (var))
	{
	  case mxDOUBLE_CLASS: isdouble = true; break;
	  case mxSINGLE_CLASS: isdouble = false; break;

	  default:{
		char errmsg [1024];
		sprintf(errmsg, "Variable %s has unsupported mxArray class: \"%s\"",
		            str, mxGetClassName(var));
		delete[] str;
		delete[] dims;
		mexErrMsgTxt (errmsg); // Returns to Matlab prompt
	  }
	}

	void *data[2] = {mxGetData(var), mxGetImagData(var)};

	if (0 != write_bz2mat (data, isdouble, ndims, dims, str)){
		char errmsg [1024];
		sprintf (errmsg, "Error writing variable %s to bzip2'ed file\n", str);
		delete [] str;
		delete [] dims;	
		mexErrMsgTxt (errmsg);
	}

	delete [] str;
	delete [] dims;	
}
