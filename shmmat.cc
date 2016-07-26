
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "shmmat.h"

static const int MAX_SHMMAT_DIMS = 16;

struct shmmat_t {
	int ndims, dims[MAX_SHMMAT_DIMS], isdouble, iscomplex, region_size;
	union {
		float f[0];
		double d[0];
	};
};

extern int
read_shmmat (void *data[2], bool& isdouble, int &ndims, int *& dims,
             char const *key)
{
	int shmid;

	if (0 > (shmid = shm_open (key, O_RDWR, 0666))){
		perror("shmget");
		return -1;
	}

	long size = lseek (shmid, 0, SEEK_END);
	if (size == -1){
		perror("lseek");
		return -1;
	}
	lseek (shmid, 0, SEEK_SET);

	shmmat_t *shmmat = (shmmat_t*) mmap (0, size, PROT_READ, MAP_SHARED, shmid, 0); 

	if (MAP_FAILED == shmmat){
		perror ("mmap");
		return -1;
	}

	close (shmid);

	isdouble = (shmmat->isdouble != 0);
	ndims = shmmat->ndims;
	dims = new int[shmmat->ndims];

	uint64_t N = 1;
	for (int i = 0; i < ndims; i++){
		N *= (dims[i] = shmmat->dims[i]);
	}

	if (isdouble)
	{
		data[0] = (void*) new double[N];
		memcpy (data[0], shmmat->d, sizeof(double)*N);

		if (shmmat->iscomplex){
			data[1] = (void*) new double[N];
			memcpy (data[1], shmmat->d + N, sizeof(double)*N);
		}else
			data[1] = 0;
	}else{
		data[0] = (void*) new float[N];
		memcpy (data[0], shmmat->f, sizeof(float)*N);

		if (shmmat->iscomplex){
			data[1] = (void*) new float[N];
			memcpy (data[1], shmmat->f + N, N*sizeof(float));
		}else
			data[1] = 0;
	}

	munmap (shmmat, size);
	return 0;
}

extern int
write_shmmat (void *data[2], bool isdouble, int ndims, int *dims,
              char const *key)
{
	uint64_t N = 1;
	for (int i = 0; i < ndims; i++){
		N *= dims[i];
	}
	bool iscomplex = (data[1] != 0);

	size_t size = sizeof(shmmat_t) + N
			* (iscomplex ? 2 : 1)
			* (isdouble ? sizeof(double) : sizeof(float));

	int shmid;

	if (0 > (shmid = shm_open (key, O_RDWR | O_CREAT, 0666))){
		perror("shmget");
		return -1;
	}

	if (ftruncate (shmid, size)){
		perror("ftruncate");
		return -1;
	}

	shmmat_t *shmmat = (shmmat_t*) mmap (0, size, PROT_READ|PROT_WRITE,
	                                     MAP_SHARED, shmid, 0); 
	shmmat->region_size = size;

	if (MAP_FAILED == shmmat){
		perror ("mmap");
		return -1;
	}
	
	close (shmid);

	shmmat->isdouble = (isdouble ? 1 : 0);
	shmmat->iscomplex = (iscomplex ? 1 : 0);
	shmmat->ndims = ndims;

	for (int i = 0; i < ndims; i++){
		shmmat->dims[i] = dims[i];
	}

	if (isdouble)
	{
		memcpy (shmmat->d, data[0], N*sizeof(double));

		if (iscomplex){
			memcpy (shmmat->d + N, data[1], sizeof(double)*N);
		}
	}else{
		memcpy (shmmat->f, data[0], sizeof(float)*N);

		if (iscomplex){
			memcpy (shmmat->f + N, data[1], N*sizeof(float));
		}
	}

	munmap (shmmat, size);

	return 0;
}

extern int
del_shmmat (char const *key)
{
	if (shm_unlink (key)){
		perror("shm_unlink");
		return -1;
	}
	return 0;
}

extern int
get_shmmat (void *data[2], bool& isdouble, int &ndims, int *& dims,
            char const * key)
{
	int shmid;

	if (0 > (shmid = shm_open (key, O_RDWR, 0666))){
		perror("shmget");
		return -1;
	}
	
	long size = lseek (shmid, 0, SEEK_END);
	if (size == -1){
		perror("lseek");
		return -1;
	}
	lseek (shmid, 0, SEEK_SET);

	shmmat_t *shmmat = (shmmat_t*) mmap (0, size, PROT_READ|PROT_WRITE,
	                                     MAP_SHARED, shmid, 0); 

	if (MAP_FAILED == shmmat){
		perror ("mmap");
		return -1;
	}

	close (shmid);

	isdouble = (shmmat->isdouble != 0);
	ndims = shmmat->ndims;
	dims = shmmat->dims;

	uint64_t N = 1;
	for (int i = 0; i < ndims; i++)
		N *= dims[i];

	if (isdouble)
	{
		data[0] = (void*)shmmat->d;

		if (shmmat->iscomplex){
			data[1] = (void*)(shmmat->d + N);
		}else
			data[1] = 0;
	}else{
		data[0] = (void*)shmmat->f;
		if (shmmat->iscomplex){
			data[1] =  (void*)(shmmat->f + N);
		}else
			data[1] = 0;
	}

	return 0;
}

extern int put_shmmat (void *data[2]){
	shmmat_t *shmmat = (shmmat_t*)((uint8_t*)data[0] - sizeof(shmmat_t));
	munmap ((void*)shmmat, shmmat->region_size);
	return 0;
}
