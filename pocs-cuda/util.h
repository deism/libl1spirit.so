#ifndef util_h
#define util_h

void upload_interleave (float *x, float *xr, float *xi, int N);
void deinterleave_download (float *xr, float *xi, float *x, int N);

void interleave (float *x, float *xr, float *xi, int N);
void deinterleave (float *xr, float *xi, float *x, int N);

void upload_interleave_2 (float *x, float *xr, float *xi, int M, int N);
void deinterleave_download_2 (float *xr, float *xi, float *x, int M, int N);

void scale (float *x, int N, float sr, float si);
void scale_2 (float *x, int M, int N, float sr, float si);

void shift_stack (float *Sr, float *Si, int M, int N, int C);

void zpad2d (float *pr, float *pi, int m2, int n2,
             float *r, float *i, int m, int n, int c);

void crop2d (float *r, float *i, int m, int n,
             float *pr, float *pi, int m2, int n2, int c);

#endif
