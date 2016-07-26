
#include <mex.h>
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

	if (del_shmmat (key)){
		mexErrMsgTxt ("del_shmmat failure!");
	}

	delete[] key;
}
