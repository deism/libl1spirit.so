#include "Daubechies4.h"

namespace pocs_omp {


/************************** Utilities ************************* */

static inline int _pow2 (int K){
	int k = 0, p = 1;
	for (k = 0; k < K; k++) p = p*2;
	return p;
}

static inline int _log2 (int N){
	int k = 1, J=0;
	while (k<N) k = 2*k, J=J+1;
	return J;
}

static inline int mod (int a, int n){
	int amn = a&(n-1);
	return amn<0 ? amn+n : amn;
}

/************************* Forward Wavelet ***********************/

static inline void
downsample_lowpass (float *y, int ys, float *x, int xs, int n)
{
	for (int i = 0; i < n/2; i++){
		float yi = 0;
		for (int j = 0; j < P; j++){
			yi += x[xs*((2*i+j)&(n-1))] * lpf[j];
		}
		y[ys*i] = yi;
	}
}

static inline void
downsample_highpass (float *y, int ys, float *x, int xs, int n)
{
	for (int i = 0; i < n/2; i++){
		float yi = 0;
		for (int j = 0; j < P; j++){
			yi += x[xs*mod(2*i+1-j,n)] * hpf[j];
		}
		y[ys*i] = yi;
	}
}

static inline void
fwt_cols (float *Xr, float *Xi, int M, int N, int m, int n, int C){

	float *tmp = new float[M];
	float *X[2] = {Xr, Xi};

	for (int r = 0; r < 2; r++){
		for (int c = 0; c < C; c++){
			for (int j = 0; j < n; j++)
			{
				float *col = &X[r][c*M*N + j*M];

				for (int i = 0; i < m; i++)
					tmp[i] = col[i];

				downsample_lowpass  (col,       1, tmp, 1, m);
				downsample_highpass (col+(m/2), 1, tmp, 1, m);
			}
		}
	}

	delete[] tmp;
}

static inline void
fwt_rows (float *Xr, float *Xi, int M, int N, int m, int n, int C){

	float  *tmp = new float[N];
	float *X[2] = {Xr, Xi};

	for (int r = 0; r < 2; r++){
		for (int c = 0; c < C; c++){
			for (int i = 0; i < m; i++)
			{
				float *row = &X[r][c*M*N + i];

				for (int j = 0; j < n; j++)
					tmp[j] = row[j*M];

				downsample_lowpass  (row,         M, tmp, 1, n);
				downsample_highpass (row+M*(n/2), M, tmp, 1, n);
			}
		}
	}

	delete[] tmp;
}

void
fwt_stack (float *Xr, float *Xi, int M, int N, int C, int L)
{
	int min_size = (1 << L);
	int JM = _log2(M), JN = _log2(N), J = JM > JN ? JM : JN;

	for (int m = M, n = N, j = J; j > L; j--)
	{
		if (m > min_size)
			fwt_cols (Xr, Xi, M, N, m, n, C);

		if (n > min_size)
			fwt_rows (Xr, Xi, M, N, m, n, C);

		if (m > min_size) m = m/2;
		if (n > min_size) n = n/2;
	}
}

/************************* Backward Wavelet ***********************/

static inline void
upsample_lowpass (float *y, int ys, float *x, int xs, int n)
{
	for (int i = 0; i < n/2; i++){
		float yi = 0;
		for (int k = 0; k < P/2; k++){
			yi += x[mod(i-k,n/2)*xs] * lpf[2*k];
		}
		y[ys*2*i] += yi;

		yi = 0;
		for (int k = 0; k < P/2; k++){
			yi += x[mod(i-k,n/2)*xs] * lpf[2*k+1];
		}
		y[ys*(2*i+1)] += yi;
	}
}

static inline void
upsample_highpass (float *y, int ys, float *x, int xs, int n)
{
	for (int i = 0; i < n/2; i++){
		float yi = 0;

		for (int k = 0; k < P/2; k++){
			yi += x[xs*((i+k)&(n/2-1))] * hpf[2*k+1];
		}
		y[ys*2*i] += yi;

		yi = 0;
		for (int k = 0; k < P/2; k++){
			yi += x[xs*((i+k)&(n/2-1))] * hpf[2*k];
		}
		y[ys*(2*i+1)] += yi;
	}
}



static inline void
iwt_cols (float *Xr, float *Xi, int M, int N, int m, int n, int C){

	float *tmp = new float[M];
	float *X[2] = {Xr, Xi};

	for (int r = 0; r < 2; r++){
		for (int c = 0; c < C; c++){
			for (int j = 0; j < n; j++)
			{
				float *col = &X[r][c*M*N + j*M];

				for (int i = 0; i < m; i++){
					tmp[i] = col[i];
					col[i] = 0;
				}

				upsample_lowpass  (col, 1, tmp,       1, m);
				upsample_highpass (col, 1, tmp+(m/2), 1, m);
			}
		}
	}

	delete[] tmp;
}


static inline void
iwt_rows (float *Xr, float *Xi, int M, int N, int m, int n, int C)
{
	float *tmp = new float[N];
	float *X[2] = {Xr, Xi};

	for (int r = 0; r < 2; r++){
		for (int c = 0; c < C; c++){
			for (int i = 0; i < m; i++)
			{
				float *row = &X[r][c*M*N + i];

				for (int j = 0; j < n; j++){
					tmp[j] = row[j*M];
					row[j*M] = 0;
				}

				upsample_lowpass  (row, M, tmp,       1, n); 
				upsample_highpass (row, M, tmp+(n/2), 1, n); 
			}
		}
	}

	delete[] tmp;
}

void
iwt_stack (float *Xr, float *Xi, int M, int N, int C, int L)
{
	for (int m = _pow2(L+1), n = _pow2(L+1); m <= M && n <= N; )
	{
		if ((M/m) > (N/n)){
			iwt_cols (Xr, Xi, M, N, m, n/2, C);
			m = 2*m;
		}else
		if ((N/n) > (M/m)){
			iwt_rows (Xr, Xi, M, N, m/2, n, C);
			n = 2*n;
		}
		else{
			iwt_rows (Xr, Xi, M, N, m, n, C);	
			iwt_cols (Xr, Xi, M, N, m, n, C);	
			m = 2*m;
			n = 2*n;
		}
	}
}

}; // namespace pocs_omp
