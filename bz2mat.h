#ifndef bz2mat_h
#define bz2mat_h

extern int read_bz2mat (void *data[2], bool& isdouble, int& ndims, int *& dims,
             char const *str);
extern int write_bz2mat (void *data[2], bool isdouble, int ndims, int* dims,
	         char const *str);


#endif
