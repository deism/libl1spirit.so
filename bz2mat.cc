
#include "bz2mat.h"

#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <bzlib.h>
#include <stdint.h>
#include <stdlib.h>

static void __bz2_error (int line, char const *msg, int err){
	char const *errname;
	switch (err){
	#define errcase(name) case BZ_##name: errname = #name; break;
		default : errname = "UNKNOWN ERROR"; break;
		//errcase(OK)
		case BZ_OK: return;
		case BZ_RUN_OK: return;
		case BZ_FLUSH_OK: return;
		case BZ_FINISH_OK: return;
		case BZ_STREAM_END: return;

		errcase(SEQUENCE_ERROR)
		errcase(PARAM_ERROR)
		errcase(MEM_ERROR)
		errcase(DATA_ERROR)
		errcase(DATA_ERROR_MAGIC)
		errcase(IO_ERROR)
		errcase(UNEXPECTED_EOF)
		errcase(OUTBUFF_FULL)
		errcase(CONFIG_ERROR)
	}

	fprintf(stderr, "[%d] %s : %s\n", line, msg, errname);
}

extern int
read_bz2mat (void *data[2], bool& isdouble, int& ndims, int *& dims,
             char const *str)
{
	uint64_t *_dims = 0;
	uint64_t _ndims = 0;
	uint64_t elt_size = 0;
	uint64_t iscomplex = 0;
	uint64_t N = 0;

	FILE *file;
	char filename[1024];
	sprintf(filename, "%s.bz2", str);

	BZFILE *bz = 0;
	int bzerror = BZ_OK;
	int nrd;

	if (!(file = fopen(filename, "rb"))){
		fprintf(stderr, "%s: %s\n", filename, strerror (errno));
		return -1;
	}

	bz = BZ2_bzReadOpen (&bzerror, file, 0, 0, NULL, 0);
	__bz2_error (__LINE__, "BZ2_bzReadOpen", bzerror);
	
// The number of dimensions
	nrd = BZ2_bzRead (&bzerror, bz, &_ndims, sizeof(_ndims));
	__bz2_error (__LINE__, "BZ2_bzRead", bzerror);

// The size in each dimension
	_dims = new uint64_t[_ndims];
	nrd = BZ2_bzRead (&bzerror, bz, _dims, sizeof(uint64_t)*_ndims);
	__bz2_error (__LINE__, "BZ2_bzRead", bzerror);


	nrd = BZ2_bzRead (&bzerror, bz, &elt_size, sizeof(uint64_t));
	__bz2_error (__LINE__, "BZ2_bzRead", bzerror);

	nrd = BZ2_bzRead (&bzerror, bz, &iscomplex, sizeof(uint64_t));
	__bz2_error (__LINE__, "BZ2_bzRead", bzerror);

	isdouble = (elt_size == 8);

	ndims = _ndims;
	dims = new int[ndims];

	N = 1;
	for (int i = 0; i < ndims; i++){
		dims[i] = _dims[i];
		N *= _dims[i];
	}

	data[0] = elt_size == 8 ? (void*) new double [N] : (void*) new float [N];
	data[1] = 0;

	for (uint64_t i = 0; i < N; i += 1024ul*1024ul){
		uint64_t n = 1024ul*1024ul;
		if (n > (N-i)) n = N-i;
		nrd = BZ2_bzRead (&bzerror, bz, (void*)((char*)data[0]+i*elt_size), n*elt_size);
		__bz2_error (__LINE__, "BZ2_bzRead", bzerror);
	}
	if (iscomplex){
		data[1] = elt_size == 8 ? (void*) new double [N] : (void*) new float [N];

		for (uint64_t i = 0; i < N; i += 1024ul*1024ul){
			uint64_t n = 1024ul*1024ul;
			if (n > (N-i)) n = N-i;
			nrd = BZ2_bzRead (&bzerror, bz, (void*)((char*)data[1]+i*elt_size), n*elt_size);
			__bz2_error (__LINE__, "BZ2_bzRead", bzerror);
		}
	}

	BZ2_bzReadClose (&bzerror, bz);
	__bz2_error (__LINE__, "BZ2_bzReadClose", bzerror);

	fclose (file);
	delete [] _dims;

	return 0;
}

extern int
write_bz2mat (void *data[2], bool isdouble, int ndims, int* dims, char const *str)
{
	BZFILE *bz = 0;
	int bzerror = BZ_OK;
	uint64_t *_dims = 0;
	uint64_t _ndims = 0;
	uint64_t elt_size = 0;
	uint64_t iscomplex = 0;
	uint64_t N = 0;

	N = 1;
	for (int i = 0; i < ndims; i++)
		N *= dims[i];

/* First -- the number of dimensions and the sizes of the array
   as 64-bit unsigned integers */
	_ndims = ndims;
	_dims = new uint64_t[ndims];
	for (int i = 0; i < ndims; i++)
		_dims[i] = dims[i];

/* Next - the number of bytes per element as a 64-bit unsigned integer */
	if (isdouble){
		elt_size = sizeof(double);
	}else{
		elt_size = sizeof(float);
	}

/* Indicate the presence or absence of complex values */
	if (data[1] != 0)
		iscomplex = 1;

	FILE *file;
	char filename[1024];
	sprintf(filename, "%s.bz2", str);

	printf("%s : ( ", str);
	for (int i = 0; i < ndims; i++){
		printf("%d ", dims[i]);
	}
	printf(") => %s\n", filename);

	if (!(file = fopen(filename, "wb"))){
		fprintf(stderr, "%s: %s", filename, strerror (errno));
		return -1;
	}

	bzerror = BZ_OK;
	bz = BZ2_bzWriteOpen(&bzerror, file, 9, 0, 0);
	__bz2_error (__LINE__, "BZ2_bzWriteOpen", bzerror);

	BZ2_bzWrite (&bzerror, bz, &_ndims, sizeof(_ndims));
	__bz2_error (__LINE__, "BZ2_bzWrite", bzerror);

	BZ2_bzWrite (&bzerror, bz, _dims, sizeof(uint64_t)*ndims);
	__bz2_error (__LINE__, "BZ2_bzWrite", bzerror);

	BZ2_bzWrite (&bzerror, bz, &elt_size, sizeof(elt_size));
	__bz2_error (__LINE__, "BZ2_bzWrite", bzerror);

	BZ2_bzWrite (&bzerror, bz, &iscomplex, sizeof(iscomplex));
	__bz2_error (__LINE__, "BZ2_bzWrite", bzerror);

// the bzlib size argument is an int ... I frequently have data sets larger
// than (2^32 - 1) bytes, so we need to do this in multiple calls

	for (uint64_t i = 0; i < N; i += 1024ul*1024ul){
		uint64_t n = 1024ul*1024ul;
		if (n > (N-i)) n = N-i;
		BZ2_bzWrite (&bzerror, bz, (void*)((char*)data[0]+i*elt_size), n*elt_size);
		__bz2_error (__LINE__, "BZ2_bzWrite", bzerror);
	}

	if (data[1] != 0){
		for (uint64_t i = 0; i < N; i += 1024ul*1024ul){
			uint64_t n = 1024ul*1024ul;
			if (n > (N-i)) n = N-i;
			BZ2_bzWrite (&bzerror, bz, (void*)((char*)data[1]+i*elt_size), n*elt_size);
			__bz2_error (__LINE__, "BZ2_bzWrite", bzerror);
		}
	}

	BZ2_bzWriteClose (&bzerror, bz, 0, 0, 0);
	__bz2_error (__LINE__, "BZ2_bzWriteClose", bzerror);

	fclose (file);

	delete[] _dims;

	return 0;
}
