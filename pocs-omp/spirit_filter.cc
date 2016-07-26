
namespace pocs_omp {

void
spirit_filter (float *rr0, float *ri0,
               float *kr0, float *ki0,
               float *xr0, float *xi0,
               int M, int N, int C)
{
	for (int cc = 0; cc < C; cc++)
	{
		float *rrcc = rr0 + cc*M*N;
		float *ricc = ri0 + cc*M*N;

		float *krcc = kr0 + cc*M*N*C;
		float *kicc = ki0 + cc*M*N*C;

		for (int ij = 0; ij < M*N; ij++)
			rrcc[ij] = ricc[ij] = 0;

		for (int c = 0; c < C; c++)
		{
			float *krc = krcc + c*M*N;
			float *kic = kicc + c*M*N;
	
			float *xrc = xr0 + c*M*N;
			float *xic = xi0 + c*M*N;

			for (int ij = 0; ij < M*N; ij++){
				rrcc[ij] += xrc[ij]*krc[ij] - xic[ij]*kic[ij];
				ricc[ij] += xrc[ij]*kic[ij] + xic[ij]*krc[ij];
			}
		}
	}
}

};
