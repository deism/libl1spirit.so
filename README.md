# libl1spirit.so

This code was originally released via Mark Murphy based on his paper: Fast L1-SPIRiT Compressed Sensing Parallel Imaging MRI: Scalable Parallel Implementation and Clinically Feasible Runtime, IEEE Transactions on Medical Imaging, volume 31, issue 6, pages 1250-1262.

The source code was modified by Jia Sen:
1. some bug fixes
2. recomplile the src to a dynamic link library (.so) via new makefile to be called in Gadgetron
3. use Armadillo and Intel MKL library for Spirit kernel calibration to replace the old ACML library

Known issues to be fixed:
1. DB4 wavelet transform will lead to a dark line artifact with a strong L1 threshold.
2. The calib3d.cc is not robust.
