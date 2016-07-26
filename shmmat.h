#ifndef shmmat_h
#define shmmat_h

static const char * L1SPIRIT = "L1SPIRIT";

/* The following two routines are intended to be semantically equivalent to
 * read_bz2mat and write_bz2mat -- read will allocate (using new[]) a copy
 * of the SHM matrix, and write will memcpy into the memory region represented
 * by key */

extern int read_shmmat (void *data[2], bool& isdouble, int &ndims, int *&dims,
                        char const *key = L1SPIRIT);

/* write_shmmat() may create a new shared memory region. Note that this
 * operation will fail if the requested region already exists but is smaller
 * than the requested size. Use del_shmmat() to clean up old, unused regions */

extern int write_shmmat (void *data[2], bool isdouble, int ndims, int *dims,
                         char const *key = L1SPIRIT);

/* get_smhmmat() provides a reference to the actual shared memory region. It
 * will also fail if the region already exists but is smaller than requested.
 * It will not allocate any new memory -- modifications to the data or dimension
 * arrays returned by get_shmmat() will be visible to all other processes using
 * the shm region */

extern int get_shmmat (void *data[2], bool& isdouble, int &ndims, int *&dims,
                       char const *key = L1SPIRIT);

extern int put_shmmat (void *data[2]);

/* del_shmmat() releases the shared memory region associated with a given
 * shared matrix. Note that this will only take effect after all processes
 * have removed their references (i.e. pointers returned via get_shmmat())
 * to the matrix */

extern int del_shmmat (char const *key = L1SPIRIT);



#endif
