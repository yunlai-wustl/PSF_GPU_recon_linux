#ifndef CUDA_COMMON_HEADER_CUH
#define CUDA_COMMON_HEADER_CUH

#include "../Solution_Items/global.h"




class Managed {
public:
	void *operator new(size_t len){
		void *ptr;
		cudaMallocManaged(&ptr, len);
		cudaDeviceSynchronize();
		return ptr;
	}

		void operator delete(void *ptr) {
		cudaDeviceSynchronize();
		cudaFree(ptr);
	}
};


class LST_LORs_uvm :public Managed{
public:
	int num_lines;
	
	float *src_x;
	float *src_y;
	float *src_z;
	float *dest_x;
	float *dest_y;
	float *dest_z;
	float *TOF_dist;
	int *coeff;
	
	/*
	LST_LORs(const LST_LORs &s) {
		num_lines = s.num_lines;
		cudaMallocManaged(&src_x, num_lines);
		memcpy(src_x, s.src_x, num_lines);
		cudaMallocManaged(&src_y, num_lines);
		memcpy(src_y, s.src_y, num_lines);
		cudaMallocManaged(&src_z, num_lines);
		memcpy(src_z, s.src_z, num_lines);
		
		cudaMallocManaged(&dest_x, num_lines);
		memcpy(dest_x, s.dest_x, num_lines);
		cudaMallocManaged(&dest_y, num_lines);
		memcpy(dest_y, s.dest_y, num_lines);
		cudaMallocManaged(&dest_z, num_lines);
		memcpy(dest_z, s.dest_z, num_lines);

		cudaMallocManaged(&TOF_dist, num_lines);
		memcpy(TOF_dist, s.TOF_dist, num_lines);
	}*/
};


typedef struct{
	int num_lines;
	float *src_x;
	float *src_y;
	float *src_z;
	float *dest_x;
	float *dest_y;
	float *dest_z;
	float *TOF_dist;
	float *sigma_left;
	float *sigma_right;
}LST_LORs;


typedef struct{
	int num_lines;

	float *src_x;
	float *src_y;
	float *src_z;
	float *dest_x;
	float *dest_y;
	float *dest_z;
	float *TOF_dist;
	float *sigma_left;
	float *sigma_right;
	float *sc_coeff;
}LST_LORs_scatter;


class PARAMETERS_IN_DEVICE_t{	
public:
	int NUM_X;
	int NUM_Y;
	int NUM_Z;
	int NUM_XY;
	int NUM_XZ;
	int NUM_YZ;
	int NUM_XYZ;

	float X_SAMP;
	float Y_SAMP;
	float Z_SAMP;

	float X_OFFSET;
	float Y_OFFSET;
	float Z_OFFSET;

	int X_INDEX_CENT;
	int Y_INDEX_CENT;
	int Z_INDEX_CENT;

	float TOF_inv;
	float TOF_alpha;

	float FWHM_sigma_inv;
	float FWHM_alpha;

	float FWHM_sigma_inv_ss;
	float FWHM_alpha_ss;

	float FWHM_sigma_inv_is;
	float FWHM_alpha_is;
	
	float FWHM_sigma_inv_ii;
	float FWHM_alpha_ii;

	float FWHM;
	float FWHM_ss;
	float FWHM_is;
	float FWHM_ii;


	float H_CENTER_X;
	float H_CENTER_Y;
	float V_CENTER;

	float spherical_voxel_ratio;

	int RANGE1;
	int RANGE2;
	int RANGE_atten_1;
	int RANGE_atten_2;

};


#endif /* CUDA_COMMON_HEADER_CUH */