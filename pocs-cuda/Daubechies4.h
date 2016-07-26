
static const int P = 4;
static __device__ const float lpf[4] = {
		.482962913145f,
		.836516303738f,
		.224143868042f,
		-.129409522551f
	};

static __device__ const float hpf[4] = {
		.482962913145f,
		-.836516303738f,
		.224143868042f,
		.129409522551f
	};

