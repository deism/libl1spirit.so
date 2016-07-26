
namespace l1spirit3d {

extern void l1spirit3d (
	int n_phases,     int n_slices,     int n_coils,      int n_freqs,
	int phase_stride, int slice_stride, int coils_stride, int stride_freqs,
	int phase_offset, int slice_offset, int coils_offset, int freqs_offset,
	float const *DATA[2], float *RECON[2],
	int ksize3d[3], int calib_freqs, float lambda_calib,
	int n_iter_l1, float lambda_l1);

namespace parameters {
	extern bool timing, verbose, calib_lsqr, calib_lsqr_ne, pocs_2fix,
		dump_raw, dump_kernel3d, dump_kernel2d,
		dump_ifreq, dump_pocs, dump_recon,
		dump_calib, dump_calibA, dump_calibAtA;

	extern int iters_calib, max_gpus;

	extern char *kernel3dfile, *kernel2dfile, *calibfile;
};

};
