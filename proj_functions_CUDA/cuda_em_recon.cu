
#define _USE_MATH_DEFINES
#include <math.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include "../common/inc/helper_functions.h"
#include "../common/inc/helper_cuda.h"

#include "cuda_em_recon.cuh"


//***********************************
//**********Edit this part to adapt to different system geometry***************//

#define BlockSize_forward_x 512
#define GridSize_forward_x  600
#define BlockSize_backward_x 512
#define GridSize_backward_x 600

#define BlockSize_forward_y 512
#define GridSize_forward_y 600
#define BlockSize_backward_y 512
#define GridSize_backward_y 600

#define NUM_SUB_SLICE_x 40
#define SIZE_SUB_SLICE_x 6000
#define SUB_SLICE_HEIGHT_x 10

#define NUM_SUB_SLICE_y 40
#define SIZE_SUB_SLICE_y 6000
#define SUB_SLICE_HEIGHT_y 10

#define GLOBAL_SCALE 4.98


//******************************************************************************//

/*
__device__ float atomicAdd_throughCAS(float* address, float val) {
	int* address_as_ull = (int*)address;
	int old = *address_as_ull;
	int assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __float_as_int(val + __int_as_float(assumed))); // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
	} while (assumed != old);
	
	return __int_as_float(old);

}
*/
	


void
__global__ setImageToZero_kernel(float* device_image_ptr, PARAMETERS_IN_DEVICE_t* parameters_device){
	float *ptr = device_image_ptr;
	int num_xyz = parameters_device->NUM_XYZ;

	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i<num_xyz; i = i + blockDim.x*gridDim.x){
		ptr[i] = 0.0f;
	}
}

void 
__global__ setForwardProjectionValueToZero_kernel(float* device_fp_ptr, int fp_size){
	float *ptr = device_fp_ptr;
	int size = fp_size;
	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i<size; i = i + blockDim.x*gridDim.x){
		ptr[i] = 0.0f;
	}
}

void
__global__ vector_sum_of_log(float* device_fp_ptr1, int fp_size, float* val){
	float *ptr1 = device_fp_ptr1;
	int size = fp_size;
	float temp;
	float per_thread_sum = 0.0f;
	__shared__ float per_block_sum;
	if (threadIdx.x == 0){
		per_block_sum = 0.0f;
	}
	__syncthreads();

	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i<size; i = i + blockDim.x*gridDim.x){
		
		temp = ptr1[i] ;
		if (isfinite(temp)){
			per_thread_sum += temp;
		}
	}
	 atomicAdd(&per_block_sum, per_thread_sum);
	 
	__syncthreads();

	if (threadIdx.x == 0){
		printf("per_block_sum is %f\n", per_block_sum);
		atomicAdd(val, per_block_sum);
	}

}



void
__global__ update_image_kernel(float* device_image_to_be_updated, float* device_update_factor, float* device_sensitivity_image, float* device_mask_image, PARAMETERS_IN_DEVICE_t* parameters_device){
	float *ptr1 = device_image_to_be_updated;
	float *ptr2 = device_update_factor;
	float *norm_image_ptr = device_sensitivity_image;
	float *mask_ptr = device_mask_image;
	int num_xyz = parameters_device->NUM_XYZ;

	if (blockIdx.x == 0 && threadIdx.x == 0){
		printf("num_xyz = %d\n", num_xyz);
	}

	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i<num_xyz; i = i + blockDim.x*gridDim.x){
		ptr1[i] = ptr1[i] * ptr2[i] / norm_image_ptr[i];
		
		if (mask_ptr[i] == 0){
			ptr1[i] = 0.0f;
		};
	}
}

void
__global__ multiply_image_kernel(float* device_image_to_be_multiplied_by, float* device_update_factor, PARAMETERS_IN_DEVICE_t* parameters_device){
	float *ptr1 = device_image_to_be_multiplied_by;
	float *ptr2 = device_update_factor;
	int num_xyz = parameters_device->NUM_XYZ;

	if (blockIdx.x == 0 && threadIdx.x == 0){
		printf("num_xyz = %d\n", num_xyz);
	}

	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i<num_xyz; i = i + blockDim.x*gridDim.x){
		ptr1[i] = ptr1[i] * ptr2[i];
	}
}


void
__global__ _fproj_lst_cuda_x_kernel(float* image, float* fp_x, LST_LORs *_events_x_dominant_uvm, PARAMETERS_IN_DEVICE_t* parameters_device){
	int event_index, current_slice, sub_slice;
	float src_x, src_y, src_z, dest_x, dest_y, dest_z;
	int coeff;
	float dir_x, dir_y, dir_z;
	//float TOF_dist;
	//float dist_to_TOF_center;
	float unit_dir_vector_y, unit_dir_vector_z;
	float projection_vector_y, projection_vector_z;
	float dot_product;
	float point_line_distance_square;
	//float weight;
	float dist, dist_squared;
	float t;
	float solid_angle_ratio;
	float forward_sum;
	float intersection_x, intersection_y, intersection_z;
	int  intersection_index_y, intersection_index_z;
	int  inslice_y, inslice_z;
	int subslice_z0, subslice_z1;
	__shared__ float H_CENTER;
	__shared__ float V_CENTER;
	__shared__ float SIGMA_INV;
	__shared__ float ALPHA;
	__shared__ float current_slice_image[SIZE_SUB_SLICE_x];
	__shared__ float voxel_size_x, voxel_size_y, voxel_size_z;
	__shared__ int center_x, center_y, center_z;
	__shared__ int range1, range2;
	__shared__ int num_x, num_y, num_z;
	__shared__ float *ptr_src_x, *ptr_src_y, *ptr_src_z, *ptr_dest_x, *ptr_dest_y, *ptr_dest_z;
	__shared__ float *ptr_FWHM;
	//__shared__ float *ptr_TOF_dist;
	__shared__ int num_lines;
	__shared__ float *image_ptr;
	__shared__ float *fp_value;
	if (threadIdx.x == 0){
		voxel_size_x = parameters_device->X_SAMP;
		voxel_size_y = parameters_device->Y_SAMP;
		voxel_size_z = parameters_device->Z_SAMP;
		center_x = parameters_device->X_INDEX_CENT;
		center_y = parameters_device->Y_INDEX_CENT;
		center_z = parameters_device->Z_INDEX_CENT;
		num_x = parameters_device->NUM_X;
		num_y = parameters_device->NUM_Y;
		num_z = parameters_device->NUM_Z;
		H_CENTER = parameters_device->H_CENTER_Y;
		V_CENTER = parameters_device->V_CENTER;
		SIGMA_INV = parameters_device->FWHM_sigma_inv;
		ALPHA = parameters_device->FWHM_alpha;
		range1 = parameters_device->RANGE1;
		range2 = parameters_device->RANGE2;
		
		ptr_src_x = _events_x_dominant_uvm->src_x;
		ptr_src_y = _events_x_dominant_uvm->src_y;
		ptr_src_z = _events_x_dominant_uvm->src_z;
		ptr_dest_x = _events_x_dominant_uvm->dest_x;
		ptr_dest_y = _events_x_dominant_uvm->dest_y;
		ptr_dest_z = _events_x_dominant_uvm->dest_z;
		ptr_FWHM = _events_x_dominant_uvm->coeff;
		
		num_lines = _events_x_dominant_uvm->num_lines;
		/*
		ptr_src_x = device_event_x.src_x;
		ptr_src_y = device_event_x.src_y;
		ptr_src_z = device_event_x.src_z;
		ptr_dest_x = device_event_x.dest_x;
		ptr_dest_y = device_event_x.dest_y;
		ptr_dest_z = device_event_x.dest_z;
		num_lines = device_event_x.num_lines;
		*/
		
		//ptr_TOF_dist = events_x.TOF_dist;
		image_ptr = image;
		fp_value = fp_x;
	}
	__syncthreads();
	current_slice = blockIdx.x;

	intersection_x = (current_slice - center_x - 0.5f)*voxel_size_x;
	for (sub_slice = 0; sub_slice<NUM_SUB_SLICE_x; sub_slice++){
		subslice_z0 = (num_z / NUM_SUB_SLICE_x)*sub_slice;
		subslice_z1 = subslice_z0 + SUB_SLICE_HEIGHT_x;//(num_z/NUM_SUB_SLICE)*(sub_slice+1);
		//load current image to shared memory
		for (int i = threadIdx.x; i<SIZE_SUB_SLICE_x; i = i + blockDim.x){
			current_slice_image[i] = image_ptr[(i + sub_slice*SIZE_SUB_SLICE_x)*num_x + current_slice];
		}
		__syncthreads();
		//loop through all the lines and compute the forward projection to this current slice
		for (event_index = threadIdx.x; event_index< num_lines; event_index = event_index + blockDim.x){
			//get event line parameters
			src_x = ptr_src_x[event_index];
			src_y = ptr_src_y[event_index];
			src_z = ptr_src_z[event_index];
			dest_x = ptr_dest_x[event_index];
			dest_y = ptr_dest_y[event_index];
			dest_z = ptr_dest_z[event_index];
			coeff = ptr_FWHM[event_index];
			
			//TOF_dist = ptr_TOF_dist[event_index];


			//only do the computation if current event line intersects current slice
			if (src_x<intersection_x - voxel_size_x && dest_x>intersection_x + voxel_size_x){
				dir_x = dest_x - src_x;
				dir_y = dest_y - src_y;
				dir_z = dest_z - src_z;
				//dist_squared = __fmaf_rn (dir_z,dir_z, __fmaf_rn(dir_y,dir_y,__fmul_rn(dir_x,dir_x) ) );
				dist_squared = dir_x*dir_x + dir_y*dir_y + dir_z*dir_z;
				dist = __fsqrt_rn(dist_squared);

				unit_dir_vector_y = dir_y / dist;
				unit_dir_vector_z = dir_z / dist;

				//t keeps the ratio of the two segment of LOR segmented by the intersection plane
				t = (intersection_x - src_x) / dir_x;
				intersection_y = t*dir_y + src_y;
				intersection_z = t*dir_z + src_z;

				intersection_index_y = (int)ceilf(intersection_y / voxel_size_y) + center_y;
				intersection_index_z = (int)ceilf(intersection_z / voxel_size_z) + center_z;

				//compute the solid angle ratio w.r.t. the location of the intersection
				solid_angle_ratio = 1 + 2 * fabsf(t - 0.5f);
				solid_angle_ratio = solid_angle_ratio*solid_angle_ratio*dist_squared;

				//compute the distance to the TOF center
				//dist_to_TOF_center = (0.5f-t)*dist - TOF_dist;
				//t = TOF_ALPHA * __expf( - dist_to_TOF_center*dist_to_TOF_center*TOF_INV);

				forward_sum = 0.0f;

				for (inslice_z = intersection_index_z - range2; inslice_z <= intersection_index_z + range2; inslice_z++){
					for (inslice_y = intersection_index_y - range1; inslice_y <= intersection_index_y + range1; inslice_y++){
						if (inslice_y >= 0 && inslice_y<num_y && inslice_z >= subslice_z0 && inslice_z<subslice_z1){
							//projection_vector_y = (inslice_y - center_y - 0.5)*voxel_size_y - intersection_y;
							projection_vector_y = inslice_y*voxel_size_y - H_CENTER - intersection_y;
							//projection_vector_z = (inslice_z - center_z - 0.5)*voxel_size_z - intersection_z;
							projection_vector_z = inslice_z*voxel_size_z - V_CENTER - intersection_z;
							dot_product = projection_vector_y*unit_dir_vector_y + projection_vector_z*unit_dir_vector_z;
							point_line_distance_square = projection_vector_y*projection_vector_y + projection_vector_z*projection_vector_z - dot_product*dot_product;

							//forward_sum+= current_slice_image[(inslice_z-subslice_z0)*num_y+inslice_y] * __expf( - point_line_distance_square*SIGMA_INV ) * ALPHA * t;
							forward_sum += current_slice_image[(inslice_z - subslice_z0)*num_y + inslice_y] * __expf(-point_line_distance_square*SIGMA_INV) * ALPHA;

							//atomicAdd(&fp_value[event_index],weight);
						}
					}
				}
				atomicAdd(&fp_value[event_index], forward_sum / solid_angle_ratio);
			}//intersects current slice
		}
	}
}

void
__global__ _fproj_lst_cuda_y_kernel(float* image, float* fp_y, LST_LORs *_events_y_dominant_uvm, PARAMETERS_IN_DEVICE_t* parameters_device){
	int event_index, current_slice, sub_slice;
	float src_x, src_y, src_z, dest_x, dest_y, dest_z;
	int coeff;
	float dir_x, dir_y, dir_z;
	//float TOF_dist;
	//float dist_to_TOF_center;
	float unit_dir_vector_x, unit_dir_vector_z;
	float projection_vector_x, projection_vector_z;
	float dot_product;
	float point_line_distance_square;
	//float weight;
	float dist, dist_squared;
	float t;
	float solid_angle_ratio;
	float forward_sum;
	float intersection_x, intersection_y, intersection_z;
	int intersection_index_x, intersection_index_z;
	int inslice_x, inslice_z;
	int subslice_z0, subslice_z1;
	__shared__ float H_CENTER;
	__shared__ float V_CENTER;
	__shared__ float SIGMA_INV;
	__shared__ float ALPHA;
	__shared__ float current_slice_image[SIZE_SUB_SLICE_y];
	__shared__ float voxel_size_x, voxel_size_y, voxel_size_z;
	__shared__ int center_x, center_y, center_z;
	__shared__ int range1, range2;
	__shared__ int num_x, num_y, num_z;
	__shared__ float *ptr_src_x, *ptr_src_y, *ptr_src_z, *ptr_dest_x, *ptr_dest_y, *ptr_dest_z;
	__shared__ float *ptr_FWHM;
	//__shared__ float *ptr_TOF_dist;
	__shared__ int num_lines;
	__shared__ float *image_ptr;
	__shared__ float *fp_value;

	if (threadIdx.x == 0){
		voxel_size_x = parameters_device->X_SAMP;
		voxel_size_y = parameters_device->Y_SAMP;
		voxel_size_z = parameters_device->Z_SAMP;
		center_x = parameters_device->X_INDEX_CENT;
		center_y = parameters_device->Y_INDEX_CENT;
		center_z = parameters_device->Z_INDEX_CENT;
		num_x = parameters_device->NUM_X;
		num_y = parameters_device->NUM_Y;
		num_z = parameters_device->NUM_Z;
		H_CENTER = parameters_device->H_CENTER_X;
		V_CENTER = parameters_device->V_CENTER;
		SIGMA_INV = parameters_device->FWHM_sigma_inv;
		ALPHA = parameters_device->FWHM_alpha;
		range1 = parameters_device->RANGE1;
		range2 = parameters_device->RANGE2;
		
		ptr_src_x = _events_y_dominant_uvm->src_x;
		ptr_src_y = _events_y_dominant_uvm->src_y;
		ptr_src_z = _events_y_dominant_uvm->src_z;
		ptr_dest_x = _events_y_dominant_uvm->dest_x;
		ptr_dest_y = _events_y_dominant_uvm->dest_y;
		ptr_dest_z = _events_y_dominant_uvm->dest_z;
		ptr_FWHM = _events_y_dominant_uvm->coeff;
	
		num_lines = _events_y_dominant_uvm->num_lines;
		/*
		ptr_src_x = device_event_y.src_x;
		ptr_src_y = device_event_y.src_y;
		ptr_src_z = device_event_y.src_z;
		ptr_dest_x = device_event_y.dest_x;
		ptr_dest_y = device_event_y.dest_y;
		ptr_dest_z = device_event_y.dest_z;
		num_lines = device_event_y.num_lines;
		*/

		//ptr_TOF_dist = events_y.TOF_dist;
		image_ptr = image;
		fp_value = fp_y;
	}
	__syncthreads();
	current_slice = blockIdx.x;

	intersection_y = (current_slice - center_y - 0.5)*voxel_size_y;
	for (sub_slice = 0; sub_slice<NUM_SUB_SLICE_y; sub_slice++){
		subslice_z0 = (num_z / NUM_SUB_SLICE_y)*sub_slice;
		subslice_z1 = subslice_z0 + SUB_SLICE_HEIGHT_y;//(num_z/NUM_SUB_SLICE)*(sub_slice+1);
		//load current image slice to shared memory
		for (int i = threadIdx.x; i<SIZE_SUB_SLICE_y; i = i + blockDim.x){
			current_slice_image[i] = image_ptr[num_x*num_y*(subslice_z0 + i / num_x) + num_x*current_slice + i%num_x];
		}
		__syncthreads();
		//loop through all the lines and compute the backprojection to this current slice
		for (event_index = threadIdx.x; event_index< num_lines; event_index = event_index + blockDim.x){
			//get event line parameters
			src_x = ptr_src_x[event_index];
			src_y = ptr_src_y[event_index];
			src_z = ptr_src_z[event_index];
			dest_x = ptr_dest_x[event_index];
			dest_y = ptr_dest_y[event_index];
			dest_z = ptr_dest_z[event_index];
			coeff = ptr_FWHM[event_index];
		
			//TOF_dist = ptr_TOF_dist[event_index];

			//only do the computation if current event line intersects current slice
			if (src_y<intersection_y - voxel_size_y && dest_y>intersection_y + voxel_size_y){
				dir_x = dest_x - src_x;
				dir_y = dest_y - src_y;
				dir_z = dest_z - src_z;
				//dist_squared = __fmaf_rn (dir_z,dir_z, __fmaf_rn(dir_y,dir_y,__fmul_rn(dir_x,dir_x) ) );
				dist_squared = dir_x*dir_x + dir_y*dir_y + dir_z*dir_z;
				dist = __fsqrt_rn(dist_squared);

				unit_dir_vector_x = dir_x / dist;
				unit_dir_vector_z = dir_z / dist;

				t = (intersection_y - src_y) / dir_y;
				intersection_x = t*dir_x + src_x;
				intersection_z = t*dir_z + src_z;

				intersection_index_x = (int)ceilf(intersection_x / voxel_size_x) + center_x;
				intersection_index_z = (int)ceilf(intersection_z / voxel_size_z) + center_z;

				//compute the solid angle ratio w.r.t. the location of the intersection
				solid_angle_ratio = 1 + 2 * fabsf(t - 0.5f);
				solid_angle_ratio = solid_angle_ratio*solid_angle_ratio*dist_squared;

				//compute the distance to the TOF center
				//dist_to_TOF_center = (0.5f-t)*dist - TOF_dist;
				//t = TOF_ALPHA * __expf( - dist_to_TOF_center*dist_to_TOF_center*TOF_INV);

				forward_sum = 0.0f;

				for (inslice_z = intersection_index_z - range2; inslice_z <= intersection_index_z + range2; inslice_z++){
					for (inslice_x = intersection_index_x - range1; inslice_x <= intersection_index_x + range1; inslice_x++){
						if (inslice_x >= 0 && inslice_x<num_x && inslice_z >= subslice_z0 && inslice_z<subslice_z1){
							//projection_vector_x = (inslice_x - center_x - 0.5)*voxel_size_x - intersection_x;
							projection_vector_x = inslice_x*voxel_size_x - H_CENTER - intersection_x;
							//projection_vector_z = (inslice_z - center_z - 0.5)*voxel_size_z - intersection_z;
							projection_vector_z = inslice_z*voxel_size_z - V_CENTER - intersection_z;
							dot_product = projection_vector_x*unit_dir_vector_x + projection_vector_z*unit_dir_vector_z;
							point_line_distance_square = projection_vector_x*projection_vector_x + projection_vector_z*projection_vector_z - dot_product*dot_product;

							//forward_sum += current_slice_image[(inslice_z-subslice_z0)*num_x+inslice_x] * __expf( - point_line_distance_square*SIGMA_INV ) * ALPHA * t;
							forward_sum += current_slice_image[(inslice_z - subslice_z0)*num_x + inslice_x] * __expf(-point_line_distance_square*SIGMA_INV) * ALPHA ;

							//atomicAdd(&fp_value[event_index],weight);

						}
					}
				}
				atomicAdd(&fp_value[event_index], forward_sum / solid_angle_ratio);
			}//intersects current slice
		}
	}
}




void
__global__ _b_ratio_proj_lst_cuda_x_kernel(float* image, float* fp_x, LST_LORs *_events_x_dominant_uvm, PARAMETERS_IN_DEVICE_t* parameters_device){
	int event_index, current_slice, sub_slice;
	float src_x, src_y, src_z, dest_x, dest_y, dest_z;
	int coeff;
	float dir_x, dir_y, dir_z;
	//float TOF_dist;
	//float dist_to_TOF_center;
	float unit_dir_vector_y, unit_dir_vector_z;
	float projection_vector_y, projection_vector_z;
	float dot_product;
	float point_line_distance_square;
	float weight;
	float dist, dist_squared;
	float t;
	float solid_angle_ratio;
	float backward_ratio;
	float intersection_x, intersection_y, intersection_z;
	int  intersection_index_y, intersection_index_z;
	int  inslice_y, inslice_z;
	int subslice_z0, subslice_z1;
	__shared__ float H_CENTER;
	__shared__ float V_CENTER;
	__shared__ float SIGMA_INV;
	__shared__ float ALPHA;
	__shared__ float current_slice_image[SIZE_SUB_SLICE_x];
	__shared__ float voxel_size_x, voxel_size_y, voxel_size_z;
	__shared__ int center_x, center_y, center_z;
	__shared__ int range1, range2;
	__shared__ int num_x, num_y, num_z;
	__shared__ float *ptr_src_x, *ptr_src_y, *ptr_src_z, *ptr_dest_x, *ptr_dest_y, *ptr_dest_z;
	__shared__ float *ptr_FWHM;

	//__shared__ float *ptr_TOF_dist;
	__shared__ int num_lines;
	__shared__ float *image_ptr;
	__shared__ float *fp_value;
	

	if (threadIdx.x == 0){
		voxel_size_x = parameters_device->X_SAMP;
		voxel_size_y = parameters_device->Y_SAMP;
		voxel_size_z = parameters_device->Z_SAMP;
		center_x = parameters_device->X_INDEX_CENT;
		center_y = parameters_device->Y_INDEX_CENT;
		center_z = parameters_device->Z_INDEX_CENT;
		num_x = parameters_device->NUM_X;
		num_y = parameters_device->NUM_Y;
		num_z = parameters_device->NUM_Z;
		H_CENTER = parameters_device->H_CENTER_Y;
		V_CENTER = parameters_device->V_CENTER;
		SIGMA_INV = parameters_device->FWHM_sigma_inv;
		ALPHA = parameters_device->FWHM_alpha;
		range1 = parameters_device->RANGE1;
		range2 = parameters_device->RANGE2;
		
		ptr_src_x = _events_x_dominant_uvm->src_x;
		ptr_src_y = _events_x_dominant_uvm->src_y;
		ptr_src_z = _events_x_dominant_uvm->src_z;
		ptr_dest_x = _events_x_dominant_uvm->dest_x;
		ptr_dest_y = _events_x_dominant_uvm->dest_y;
		ptr_dest_z = _events_x_dominant_uvm->dest_z;
		ptr_FWHM = _events_x_dominant_uvm->coeff;
		
		num_lines = _events_x_dominant_uvm->num_lines;
		/*
		ptr_src_x = device_event_x.src_x;
		ptr_src_y = device_event_x.src_y;
		ptr_src_z = device_event_x.src_z;
		ptr_dest_x = device_event_x.dest_x;
		ptr_dest_y = device_event_x.dest_y;
		ptr_dest_z = device_event_x.dest_z;
		num_lines = device_event_x.num_lines;
		*/

		//ptr_TOF_dist = events_x.TOF_dist;
		image_ptr = image;
		fp_value = fp_x;
	}

	//initialize shared memory to zero
	for (int i = threadIdx.x; i<SIZE_SUB_SLICE_x; i = i + blockDim.x){
		current_slice_image[i] = 0.0f;
	}
	__syncthreads();

	current_slice = blockIdx.x;
	intersection_x = __fmul_rn((current_slice - center_x - 0.5f), voxel_size_x);
	for (sub_slice = 0; sub_slice<NUM_SUB_SLICE_x; sub_slice++){
		subslice_z0 = (num_z / NUM_SUB_SLICE_x)*sub_slice;
		subslice_z1 = subslice_z0 + SUB_SLICE_HEIGHT_x;//(num_z/NUM_SUB_SLICE)*(sub_slice+1);
		//loop through all the lines and compute the backprojection to this current slice
		for (event_index = threadIdx.x; event_index< num_lines; event_index = event_index + blockDim.x){
			//get event line parameters
			src_x = ptr_src_x[event_index];
			src_y = ptr_src_y[event_index];
			src_z = ptr_src_z[event_index];
			dest_x = ptr_dest_x[event_index];
			dest_y = ptr_dest_y[event_index];
			dest_z = ptr_dest_z[event_index];
			coeff = ptr_FWHM[event_index];
			

			//TOF_dist = ptr_TOF_dist[event_index];

			//only do the computation if current event line intersects current slice
			if (src_x<intersection_x - voxel_size_x && dest_x>intersection_x + voxel_size_x){
				dir_x = dest_x - src_x;
				dir_y = dest_y - src_y;
				dir_z = dest_z - src_z;
				//dist_squared = __fmaf_rn (dir_z,dir_z, __fmaf_rn(dir_y,dir_y,__fmul_rn(dir_x,dir_x) ) );
				dist_squared = dir_x*dir_x + dir_y*dir_y + dir_z*dir_z;
				dist = __fsqrt_rn(dist_squared);

				unit_dir_vector_y = dir_y / dist;
				unit_dir_vector_z = dir_z / dist;

				t = (intersection_x - src_x) / dir_x;
				intersection_y = t*dir_y + src_y;
				intersection_z = t*dir_z + src_z;

				if (fp_value[event_index] < SMALLEST_ALLOWED)
					backward_ratio = 0.0f;
				else
					backward_ratio = __frcp_rn(fp_value[event_index]);

				intersection_index_y = (int)ceilf(intersection_y / voxel_size_y) + center_y;
				intersection_index_z = (int)ceilf(intersection_z / voxel_size_z) + center_z;

				//compute the solid angle ratio w.r.t. the location of the intersection
				solid_angle_ratio = 1 + 2 * fabsf(t - 0.5f);
				solid_angle_ratio = solid_angle_ratio*solid_angle_ratio*dist*dist;

				//compute the distance to the TOF center
				//dist_to_TOF_center = (0.5f-t)*dist - TOF_dist;
				//t = TOF_ALPHA * __expf( - dist_to_TOF_center*dist_to_TOF_center*TOF_INV);

				for (inslice_z = intersection_index_z - range2; inslice_z <= intersection_index_z + range2; inslice_z++){
					for (inslice_y = intersection_index_y - range1; inslice_y <= intersection_index_y + range1; inslice_y++){
						if (inslice_y >= 0 && inslice_y<num_y && inslice_z >= subslice_z0 && inslice_z<subslice_z1){
							//projection_vector_y = (inslice_y - center_y - 0.5)*voxel_size_y - intersection_y;
							projection_vector_y = inslice_y*voxel_size_y - H_CENTER - intersection_y;
							//projection_vector_z = (inslice_z - center_z - 0.5)*voxel_size_z - intersection_z;
							projection_vector_z = inslice_z*voxel_size_z - V_CENTER - intersection_z;
							dot_product = projection_vector_y*unit_dir_vector_y + projection_vector_z*unit_dir_vector_z;
							point_line_distance_square = projection_vector_y*projection_vector_y + projection_vector_z*projection_vector_z - dot_product*dot_product;
							//weight = t * backward_ratio * __expf( - point_line_distance_square*SIGMA_INV );
							weight = backward_ratio * __expf(-point_line_distance_square*SIGMA_INV) * ALPHA ;
							atomicAdd(&current_slice_image[(inslice_z - subslice_z0)*num_y + inslice_y], weight / solid_angle_ratio);
						}
					}
				}

			}//intersects current slice
		}
		__syncthreads();
		//write the current sub slice from shared memory back to global memory using atomic operation
		//Since this write is not coalesced,  think about optimization techniques
		
		for (int j = threadIdx.x; j<SIZE_SUB_SLICE_x; j = j + blockDim.x){
			//atomicAdd(&image_ptr[ (j+sub_slice*SIZE_SUB_SLICE)*image_width+current_slice], current_slice_image[j]);
			atomicAdd(&image_ptr[(j + sub_slice*SIZE_SUB_SLICE_x)*num_x + current_slice], current_slice_image[j]);
			current_slice_image[j] = 0.0f;
		}
		__syncthreads();
	}
}

void
__global__ _b_ratio_proj_lst_cuda_y_kernel(float* image, float* fp_y, LST_LORs *_events_y_dominant_uvm, PARAMETERS_IN_DEVICE_t* parameters_device){
	int event_index, current_slice, sub_slice;
	float src_x, src_y, src_z, dest_x, dest_y, dest_z;
		int coeff;
	float dir_x, dir_y, dir_z;
	//float TOF_dist;
	//float dist_to_TOF_center;
	float unit_dir_vector_x, unit_dir_vector_z;
	float projection_vector_x, projection_vector_z;
	float dot_product;
	float point_line_distance_square;
	float weight;
	float dist, dist_squared;
	float t;
	float solid_angle_ratio;
	float backward_ratio;
	float intersection_x, intersection_y, intersection_z;
	int intersection_index_x, intersection_index_z;
	int inslice_x, inslice_z;
	int subslice_z0, subslice_z1;
	__shared__ float H_CENTER;
	__shared__ float V_CENTER;
	__shared__ float SIGMA_INV;
	__shared__ float ALPHA;
	__shared__ float current_slice_image[SIZE_SUB_SLICE_y];
	__shared__ float voxel_size_x, voxel_size_y, voxel_size_z;
	__shared__ int center_x, center_y, center_z;
	__shared__ int range1, range2;
	__shared__ int num_x, num_y, num_z;
	__shared__ float *ptr_src_x, *ptr_src_y, *ptr_src_z, *ptr_dest_x, *ptr_dest_y, *ptr_dest_z; 
	__shared__ float *ptr_FWHM; 
	//__shared__ float *ptr_TOF_dist;
	__shared__ int num_lines;
	__shared__ float *image_ptr;
	__shared__ float *fp_value;

	if (threadIdx.x == 0){
		voxel_size_x = parameters_device->X_SAMP;
		voxel_size_y = parameters_device->Y_SAMP;
		voxel_size_z = parameters_device->Z_SAMP;
		center_x = parameters_device->X_INDEX_CENT;
		center_y = parameters_device->Y_INDEX_CENT;
		center_z = parameters_device->Z_INDEX_CENT;
		num_x = parameters_device->NUM_X;
		num_y = parameters_device->NUM_Y;
		num_z = parameters_device->NUM_Z;
		H_CENTER = parameters_device->H_CENTER_X;
		V_CENTER = parameters_device->V_CENTER;
		SIGMA_INV = parameters_device->FWHM_sigma_inv;
		ALPHA = parameters_device->FWHM_alpha;
		range1 = parameters_device->RANGE1;
		range2 = parameters_device->RANGE2;

		ptr_src_x = _events_y_dominant_uvm->src_x;
		ptr_src_y = _events_y_dominant_uvm->src_y;
		ptr_src_z = _events_y_dominant_uvm->src_z;
		ptr_dest_x = _events_y_dominant_uvm->dest_x;
		ptr_dest_y = _events_y_dominant_uvm->dest_y;
		ptr_dest_z = _events_y_dominant_uvm->dest_z;
		ptr_FWHM= _events_y_dominant_uvm->coeff;
	
		num_lines = _events_y_dominant_uvm->num_lines;
		/*
		ptr_src_x = device_event_y.src_x;
		ptr_src_y = device_event_y.src_y;
		ptr_src_z = device_event_y.src_z;
		ptr_dest_x = device_event_y.dest_x;
		ptr_dest_y = device_event_y.dest_y;
		ptr_dest_z = device_event_y.dest_z;
		num_lines = device_event_y.num_lines;
		*/

		//ptr_TOF_dist = events_y.TOF_dist;
		image_ptr = image;
		fp_value = fp_y;
	}

	for (int i = threadIdx.x; i<SIZE_SUB_SLICE_y; i = i + blockDim.x){
		current_slice_image[i] = 0.0f;
	}
	__syncthreads();

	current_slice = blockIdx.x;
	intersection_y = __fmul_rn((current_slice - center_y - 0.5f), voxel_size_y);
	for (sub_slice = 0; sub_slice<NUM_SUB_SLICE_y; sub_slice++){
		subslice_z0 = (num_z / NUM_SUB_SLICE_y)*sub_slice;
		subslice_z1 = subslice_z0 + SUB_SLICE_HEIGHT_y;//(num_z/NUM_SUB_SLICE)*(sub_slice+1);
		//loop through all the lines and compute the backprojection to this current slice
		for (event_index = threadIdx.x; event_index< num_lines; event_index = event_index + blockDim.x){
			//get event line parameters
			src_x = ptr_src_x[event_index];
			src_y = ptr_src_y[event_index];
			src_z = ptr_src_z[event_index];
			dest_x = ptr_dest_x[event_index];
			dest_y = ptr_dest_y[event_index];
			dest_z = ptr_dest_z[event_index];
			coeff = ptr_FWHM[event_index];
		
			//TOF_dist = ptr_TOF_dist[event_index];

			//only do the computation if current event line intersects current slice
			if (src_y<intersection_y - voxel_size_y && dest_y>intersection_y + voxel_size_y){
				dir_x = dest_x - src_x;
				dir_y = dest_y - src_y;
				dir_z = dest_z - src_z;
				//dist_squared = __fmaf_rn (dir_z,dir_z, __fmaf_rn(dir_y,dir_y,__fmul_rn(dir_x,dir_x) ) );
				dist_squared = dir_x*dir_x + dir_y*dir_y + dir_z*dir_z;
				dist = __fsqrt_rn(dist_squared);

				unit_dir_vector_x = dir_x / dist;
				unit_dir_vector_z = dir_z / dist;

				t = (intersection_y - src_y) / dir_y;
				intersection_x = t*dir_x + src_x;
				intersection_z = t*dir_z + src_z;

				if (fp_value[event_index] < SMALLEST_ALLOWED)
					backward_ratio = 0.0f;
				else
					backward_ratio = __frcp_rn(fp_value[event_index]);

				intersection_index_x = (int)ceilf(intersection_x / voxel_size_x) + center_x;
				intersection_index_z = (int)ceilf(intersection_z / voxel_size_z) + center_z;

				//compute the solid angle ratio w.r.t. the location of the intersection
				solid_angle_ratio = 1+ 2*fabsf(t - 0.5f);
				solid_angle_ratio = solid_angle_ratio*solid_angle_ratio*dist*dist;

				//compute the distance to the TOF center
				//dist_to_TOF_center = (0.5f-t)*dist - TOF_dist;
				//t = TOF_ALPHA * __expf( - dist_to_TOF_center*dist_to_TOF_center*TOF_INV);

				for (inslice_z = intersection_index_z - range2; inslice_z <= intersection_index_z + range2; inslice_z++){
					for (inslice_x = intersection_index_x - range1; inslice_x <= intersection_index_x + range1; inslice_x++){
						if (inslice_x >= 0 && inslice_x<num_x && inslice_z >= subslice_z0 && inslice_z<subslice_z1){
							//projection_vector_x = (inslice_x - center_x - 0.5)*voxel_size_x - intersection_x;
							projection_vector_x = inslice_x*voxel_size_x - H_CENTER - intersection_x;
							//projection_vector_z = (inslice_z - center_z - 0.5)*voxel_size_z - intersection_z;
							projection_vector_z = inslice_z*voxel_size_z - V_CENTER - intersection_z;
							dot_product = projection_vector_x*unit_dir_vector_x + projection_vector_z*unit_dir_vector_z;
							point_line_distance_square = projection_vector_x*projection_vector_x + projection_vector_z*projection_vector_z - dot_product*dot_product;
							//weight = t * backward_ratio * __expf( - point_line_distance_square*SIGMA_INV );
							weight = backward_ratio * __expf(-point_line_distance_square*SIGMA_INV) * ALPHA ;
							atomicAdd(&current_slice_image[(inslice_z - subslice_z0)*num_x + inslice_x], weight / solid_angle_ratio);
						}
					}
				}
			}//intersects current slice
		}
		__syncthreads();
		//write the current sub slice from shared memory back to global memory using atomic operation
		//Since this write is not coalesced,  think about optimization techniques
		for (int j = threadIdx.x; j<SIZE_SUB_SLICE_y; j = j + blockDim.x){
			atomicAdd(&image_ptr[num_x*num_y*(subslice_z0 + j / num_x) + num_x*current_slice + j%num_x], current_slice_image[j]);
			//image_ptr[ image_width*image_height*(subslice_z0+j/image_width) + image_width*current_slice + j%image_width ] = current_slice_image[j];
			current_slice_image[j] = 0.0f;
		}
		__syncthreads();
	}
}

void
__global__ _TOF_fproj_lst_cuda_x_kernel(float* image, float* fp_x, LST_LORs *_events_x_dominant_uvm, PARAMETERS_IN_DEVICE_t* parameters_device){
	int event_index, current_slice, sub_slice;
	float src_x, src_y, src_z, dest_x, dest_y, dest_z;
	int coeff, c;
	float current_FWHM,current_Alpha,current_FWHM_sigma_inv; //for PSF
	float dir_x, dir_y, dir_z;
	float TOF_dist;
	float dist_to_TOF_center;
	float unit_dir_vector_y, unit_dir_vector_z;
	float projection_vector_y, projection_vector_z;
	float dot_product;
	float point_line_distance_square;
	//float weight;
	float dist, dist_squared;
	float t;
	float solid_angle_ratio;
	float forward_sum, increment;
	float intersection_x, intersection_y, intersection_z;
	int  intersection_index_y, intersection_index_z;
	int  inslice_y, inslice_z;
	int subslice_z0, subslice_z1;
	__shared__ float H_CENTER;
	__shared__ float V_CENTER;
	__shared__ float SIGMA_INV;
	__shared__ float ALPHA;
	__shared__ float current_slice_image[SIZE_SUB_SLICE_x];
	__shared__ float voxel_size_x, voxel_size_y, voxel_size_z;
	__shared__ int center_x, center_y, center_z;
	__shared__ int range1, range2;
	__shared__ int num_x, num_y, num_z;
	__shared__ float *ptr_src_x, *ptr_src_y, *ptr_src_z, *ptr_dest_x, *ptr_dest_y, *ptr_dest_z, *ptr_TOF_dist;
	__shared__ float *ptr_FWHM;
	__shared__ int num_lines;
	__shared__ float *image_ptr;
	__shared__ float *fp_value;
	__shared__ float TOF_INV, TOF_ALPHA;
	if (threadIdx.x == 0){
		voxel_size_x = parameters_device->X_SAMP;
		voxel_size_y = parameters_device->Y_SAMP;
		voxel_size_z = parameters_device->Z_SAMP;
		center_x = parameters_device->X_INDEX_CENT;
		center_y = parameters_device->Y_INDEX_CENT;
		center_z = parameters_device->Z_INDEX_CENT;
		num_x = parameters_device->NUM_X;
		num_y = parameters_device->NUM_Y;
		num_z = parameters_device->NUM_Z;
		H_CENTER = parameters_device->H_CENTER_Y;
		V_CENTER = parameters_device->V_CENTER;
		SIGMA_INV = parameters_device->FWHM_sigma_inv;
		ALPHA = parameters_device->FWHM_alpha;
		range1 = parameters_device->RANGE1;
		range2 = parameters_device->RANGE2;
		TOF_INV = parameters_device->TOF_inv;
		TOF_ALPHA = parameters_device->TOF_alpha;

		ptr_src_x = _events_x_dominant_uvm->src_x;
		ptr_src_y = _events_x_dominant_uvm->src_y;
		ptr_src_z = _events_x_dominant_uvm->src_z;
		ptr_dest_x = _events_x_dominant_uvm->dest_x;
		ptr_dest_y = _events_x_dominant_uvm->dest_y;
		ptr_dest_z = _events_x_dominant_uvm->dest_z;
		ptr_FWHM = _events_x_dominant_uvm->coeff;

		ptr_TOF_dist = _events_x_dominant_uvm->TOF_dist;
		num_lines = _events_x_dominant_uvm->num_lines;
		image_ptr = image;
		fp_value = fp_x;
	}
	c = 1;
	__syncthreads();

	current_slice = blockIdx.x;

	intersection_x = (current_slice - center_x - 0.5f)*voxel_size_x;
	for (sub_slice = 0; sub_slice<NUM_SUB_SLICE_x; sub_slice++){
		subslice_z0 = (num_z / NUM_SUB_SLICE_x)*sub_slice;
		subslice_z1 = subslice_z0 + SUB_SLICE_HEIGHT_x;//(num_z/NUM_SUB_SLICE)*(sub_slice+1);
		//load current image to shared memory
		for (int i = threadIdx.x; i<SIZE_SUB_SLICE_x; i = i + blockDim.x){
			current_slice_image[i] = image_ptr[(i + sub_slice*SIZE_SUB_SLICE_x)*num_x + current_slice];
		}
		__syncthreads();
		//loop through all the lines and compute the forward projection to this current slice
		for (event_index = threadIdx.x; event_index< num_lines; event_index = event_index + blockDim.x){
			//get event line parameters
			src_x = ptr_src_x[event_index];
			src_y = ptr_src_y[event_index];
			src_z = ptr_src_z[event_index];
			dest_x = ptr_dest_x[event_index];
			dest_y = ptr_dest_y[event_index];
			dest_z = ptr_dest_z[event_index];
			coeff = ptr_FWHM[event_index];

			TOF_dist = ptr_TOF_dist[event_index];

			current_FWHM = ptr_FWHM[event_index];
			current_FWHM_sigma_inv = 4 * log(2.0) / (current_FWHM*current_FWHM);
		    current_Alpha =0.93943727 / current_FWHM;
		    
			//only do the computation if current event line intersects current slice
			if (src_x<intersection_x - voxel_size_x && dest_x>intersection_x + voxel_size_x){
				dir_x = dest_x - src_x;
				dir_y = dest_y - src_y;
				dir_z = dest_z - src_z;
				//dist_squared = __fmaf_rn (dir_z,dir_z, __fmaf_rn(dir_y,dir_y,__fmul_rn(dir_x,dir_x) ) );
				dist_squared = dir_x*dir_x + dir_y*dir_y + dir_z*dir_z;
				dist = __fsqrt_rn(dist_squared);

				unit_dir_vector_y = dir_y / dist;
				unit_dir_vector_z = dir_z / dist;

				//t keeps the ratio of the two segment of LOR segmented by the intersection plane
				t = (intersection_x - src_x) / dir_x;
				intersection_y = t*dir_y + src_y;
				intersection_z = t*dir_z + src_z;

				intersection_index_y = (int)ceilf(intersection_y / voxel_size_y) + center_y;
				intersection_index_z = (int)ceilf(intersection_z / voxel_size_z) + center_z;

				//compute the solid angle ratio w.r.t. the location of the intersection
				solid_angle_ratio = 1 + 2 * fabsf(t - 0.5f);
				//solid_angle_ratio = solid_angle_ratio*solid_angle_ratio*dist*dist;
				solid_angle_ratio = 1 * dist*dist;

				//compute the distance to the TOF center
				dist_to_TOF_center = (0.5f - t)*dist - TOF_dist;
				t =  TOF_ALPHA * __expf(-dist_to_TOF_center*dist_to_TOF_center*TOF_INV);

				forward_sum = 0.0f;

				for (inslice_z = intersection_index_z - range2; inslice_z <= intersection_index_z + range2; inslice_z++){
					for (inslice_y = intersection_index_y - range1; inslice_y <= intersection_index_y + range1; inslice_y++){
						if (inslice_y >= 0 && inslice_y < num_y && inslice_z >= subslice_z0 && inslice_z < subslice_z1){
							//projection_vector_y = (inslice_y - center_y - 0.5)*voxel_size_y - intersection_y;
							projection_vector_y = inslice_y*voxel_size_y - H_CENTER - intersection_y;
							//projection_vector_z = (inslice_z - center_z - 0.5)*voxel_size_z - intersection_z;
							projection_vector_z = inslice_z*voxel_size_z - V_CENTER - intersection_z;
							dot_product = projection_vector_y*unit_dir_vector_y + projection_vector_z*unit_dir_vector_z;
							point_line_distance_square = projection_vector_y*projection_vector_y + projection_vector_z*projection_vector_z - dot_product*dot_product;

							//increment = current_slice_image[(inslice_z - subslice_z0)*num_y + inslice_y] * 0.569;
							//increment = current_slice_image[(inslice_z - subslice_z0)*num_y + inslice_y] * __expf(-point_line_distance_square*SIGMA_INV) * ALPHA * t * GLOBAL_SCALE;
							increment = current_slice_image[(inslice_z - subslice_z0)*num_y + inslice_y] * __expf(-point_line_distance_square*current_FWHM_sigma_inv) * current_Alpha * t * GLOBAL_SCALE;;

							forward_sum += increment;
							//forward_sum += current_slice_image[(inslice_z - subslice_z0)*num_y + inslice_y] * 0.569 ;
							//atomicAdd(&fp_value[event_index],weight);
						}
					}
				}

				atomicAdd(&fp_value[event_index], forward_sum/solid_angle_ratio );
				//atomicAdd(&fp_value[event_index], 1);


			}//intersects current slice

		}
	}
}

void
__global__ _TOF_fproj_lst_cuda_x_kernel_atten(float* image, float* fp_x, float* atten_per_event, LST_LORs *_events_x_dominant_uvm, PARAMETERS_IN_DEVICE_t* parameters_device){
	int event_index, current_slice, sub_slice;
	float src_x, src_y, src_z, dest_x, dest_y, dest_z;
	float current_FWHM,current_Alpha,current_FWHM_sigma_inv; //for PSF
	int coeff, c;
	float dir_x, dir_y, dir_z;
	float TOF_dist;
	float dist_to_TOF_center;
	float unit_dir_vector_y, unit_dir_vector_z;
	float projection_vector_y, projection_vector_z;
	float dot_product;
	float point_line_distance_square;
	//float weight;
	float dist, dist_squared;
	float t;
	float solid_angle_ratio;
	float forward_sum, integral_mu;
	float intersection_x, intersection_y, intersection_z;
	int  intersection_index_y, intersection_index_z;
	int  inslice_y, inslice_z;
	int subslice_z0, subslice_z1;
	__shared__ float H_CENTER;
	__shared__ float V_CENTER;
	__shared__ float SIGMA_INV;
	__shared__ float ALPHA;
	__shared__ float current_slice_image[SIZE_SUB_SLICE_x];
	__shared__ float voxel_size_x, voxel_size_y, voxel_size_z;
	__shared__ int center_x, center_y, center_z;
	__shared__ int range1, range2;
	__shared__ int num_x, num_y, num_z;
	__shared__ float *ptr_src_x, *ptr_src_y, *ptr_src_z, *ptr_dest_x, *ptr_dest_y, *ptr_dest_z, *ptr_TOF_dist;
	__shared__ float *ptr_FWHM;
	__shared__ int num_lines;
	__shared__ float *image_ptr;
	__shared__ float *fp_value;
	__shared__ float *atten_factor;
	__shared__ float TOF_INV, TOF_ALPHA;
	if (threadIdx.x == 0){
		voxel_size_x = parameters_device->X_SAMP;
		voxel_size_y = parameters_device->Y_SAMP;
		voxel_size_z = parameters_device->Z_SAMP;
		center_x = parameters_device->X_INDEX_CENT;
		center_y = parameters_device->Y_INDEX_CENT;
		center_z = parameters_device->Z_INDEX_CENT;
		num_x = parameters_device->NUM_X;
		num_y = parameters_device->NUM_Y;
		num_z = parameters_device->NUM_Z;
		H_CENTER = parameters_device->H_CENTER_Y;
		V_CENTER = parameters_device->V_CENTER;
		SIGMA_INV = parameters_device->FWHM_sigma_inv;
		ALPHA = parameters_device->FWHM_alpha;
		range1 = parameters_device->RANGE1;
		range2 = parameters_device->RANGE2;
		TOF_INV = parameters_device->TOF_inv;
		TOF_ALPHA = parameters_device->TOF_alpha;

		ptr_src_x = _events_x_dominant_uvm->src_x;
		ptr_src_y = _events_x_dominant_uvm->src_y;
		ptr_src_z = _events_x_dominant_uvm->src_z;
		ptr_dest_x = _events_x_dominant_uvm->dest_x;
		ptr_dest_y = _events_x_dominant_uvm->dest_y;
		ptr_dest_z = _events_x_dominant_uvm->dest_z;
		ptr_FWHM = _events_x_dominant_uvm->coeff;

		ptr_TOF_dist = _events_x_dominant_uvm->TOF_dist;
		num_lines = _events_x_dominant_uvm->num_lines;
		image_ptr = image;
		fp_value = fp_x;
		atten_factor = atten_per_event;
	}
	c = 1;
	__syncthreads();

	current_slice = blockIdx.x;

	intersection_x = (current_slice - center_x - 0.5f)*voxel_size_x;
	for (sub_slice = 0; sub_slice<NUM_SUB_SLICE_x; sub_slice++){
		subslice_z0 = (num_z / NUM_SUB_SLICE_x)*sub_slice;
		subslice_z1 = subslice_z0 + SUB_SLICE_HEIGHT_x;//(num_z/NUM_SUB_SLICE)*(sub_slice+1);
		//load current image to shared memory
		for (int i = threadIdx.x; i<SIZE_SUB_SLICE_x; i = i + blockDim.x){
			current_slice_image[i] = image_ptr[(i + sub_slice*SIZE_SUB_SLICE_x)*num_x + current_slice];
		}
		__syncthreads();
		//loop through all the lines and compute the forward projection to this current slice
		for (event_index = threadIdx.x; event_index< num_lines; event_index = event_index + blockDim.x){
			//get event line parameters
			src_x = ptr_src_x[event_index];
			src_y = ptr_src_y[event_index];
			src_z = ptr_src_z[event_index];
			dest_x = ptr_dest_x[event_index];
			dest_y = ptr_dest_y[event_index];
			dest_z = ptr_dest_z[event_index];
			current_FWHM = ptr_FWHM[event_index];
			current_FWHM_sigma_inv = 4 * log(2.0) / (current_FWHM*current_FWHM);
		    current_Alpha =0.93943727 / current_FWHM; //2sqrt(log(2)/pi)/FWHM
			TOF_dist = ptr_TOF_dist[event_index];


			//only do the computation if current event line intersects current slice
			if (src_x<intersection_x - voxel_size_x && dest_x>intersection_x + voxel_size_x){
				dir_x = dest_x - src_x;
				dir_y = dest_y - src_y;
				dir_z = dest_z - src_z;
				//dist_squared = __fmaf_rn (dir_z,dir_z, __fmaf_rn(dir_y,dir_y,__fmul_rn(dir_x,dir_x) ) );
				dist_squared = dir_x*dir_x + dir_y*dir_y + dir_z*dir_z;
				dist = __fsqrt_rn(dist_squared);

				unit_dir_vector_y = dir_y / dist;
				unit_dir_vector_z = dir_z / dist;

				//t keeps the ratio of the two segment of LOR segmented by the intersection plane
				t = (intersection_x - src_x) / dir_x;
				intersection_y = t*dir_y + src_y;
				intersection_z = t*dir_z + src_z;

				intersection_index_y = (int)ceilf(intersection_y / voxel_size_y) + center_y;
				intersection_index_z = (int)ceilf(intersection_z / voxel_size_z) + center_z;

				//compute the solid angle ratio w.r.t. the location of the intersection
				solid_angle_ratio = 1 + 2 * fabsf(t - 0.5f);
				//solid_angle_ratio = solid_angle_ratio*solid_angle_ratio;
				solid_angle_ratio = 1 * dist*dist;

				//compute the distance to the TOF center
				dist_to_TOF_center = (0.5f - t)*dist - TOF_dist;
				t = TOF_ALPHA * __expf(-dist_to_TOF_center*dist_to_TOF_center*TOF_INV);

				integral_mu = atten_factor[event_index];

				forward_sum = 0.0f;

				for (inslice_z = intersection_index_z - range2; inslice_z <= intersection_index_z + range2; inslice_z++){
					for (inslice_y = intersection_index_y - range1; inslice_y <= intersection_index_y + range1; inslice_y++){
						if (inslice_y >= 0 && inslice_y < num_y && inslice_z >= subslice_z0 && inslice_z < subslice_z1){
							//projection_vector_y = (inslice_y - center_y - 0.5)*voxel_size_y - intersection_y;
							projection_vector_y = inslice_y*voxel_size_y - H_CENTER - intersection_y;
							//projection_vector_z = (inslice_z - center_z - 0.5)*voxel_size_z - intersection_z;
							projection_vector_z = inslice_z*voxel_size_z - V_CENTER - intersection_z;
							dot_product = projection_vector_y*unit_dir_vector_y + projection_vector_z*unit_dir_vector_z;
							point_line_distance_square = projection_vector_y*projection_vector_y + projection_vector_z*projection_vector_z - dot_product*dot_product;

							//forward_sum += current_slice_image[(inslice_z - subslice_z0)*num_y + inslice_y] * __expf(-point_line_distance_square*SIGMA_INV) * ALPHA * t * GLOBAL_SCALE;
							//PSF
							forward_sum += current_slice_image[(inslice_z - subslice_z0)*num_y + inslice_y] * __expf(-point_line_distance_square*current_FWHM_sigma_inv) * current_Alpha * t * GLOBAL_SCALE;
							//atomicAdd(&fp_value[event_index],weight);
						}
					}
				}

				atomicAdd(&fp_value[event_index], (forward_sum * __expf(-integral_mu))/solid_angle_ratio);
				//atomicAdd(&coeff_value[event_index], coeff_s);


			}//intersects current slice

		}
	}
}


void
__global__ _TOF_fproj_lst_cuda_x_kernel_scatter(float* image, float* fp_x, LST_LORs_scatter *_events_x_dominant_uvm, PARAMETERS_IN_DEVICE_t* parameters_device){
	int event_index, current_slice, sub_slice;
	float src_x, src_y, src_z, dest_x, dest_y, dest_z;
	int coeff, c;
	float sc_coeff=0.0;
	float dir_x, dir_y, dir_z;
	float TOF_dist;
	float dist_to_TOF_center;
	float unit_dir_vector_y, unit_dir_vector_z;
	float projection_vector_y, projection_vector_z;
	float dot_product;
	float point_line_distance_square;
	//float weight;
	float dist, dist_squared;
	float t;
	float solid_angle_ratio;
	float forward_sum;
	float intersection_x, intersection_y, intersection_z;
	int  intersection_index_y, intersection_index_z;
	int  inslice_y, inslice_z;
	int subslice_z0, subslice_z1;
	__shared__ float H_CENTER;
	__shared__ float V_CENTER;
	__shared__ float SIGMA_INV;
	__shared__ float ALPHA;
	__shared__ float current_slice_image[SIZE_SUB_SLICE_x];
	__shared__ float voxel_size_x, voxel_size_y, voxel_size_z;
	__shared__ int center_x, center_y, center_z;
	__shared__ int range1, range2;
	__shared__ int num_x, num_y, num_z;
	__shared__ float *ptr_src_x, *ptr_src_y, *ptr_src_z, *ptr_dest_x, *ptr_dest_y, *ptr_dest_z, *ptr_TOF_dist;
	__shared__ float *ptr_sc_coeff;
	__shared__ float *ptr_FWHM;
	__shared__ int num_lines;
	__shared__ float *image_ptr;
	__shared__ float *fp_value;
	__shared__ float TOF_INV, TOF_ALPHA;
	if (threadIdx.x == 0){
		voxel_size_x = parameters_device->X_SAMP;
		voxel_size_y = parameters_device->Y_SAMP;
		voxel_size_z = parameters_device->Z_SAMP;
		center_x = parameters_device->X_INDEX_CENT;
		center_y = parameters_device->Y_INDEX_CENT;
		center_z = parameters_device->Z_INDEX_CENT;
		num_x = parameters_device->NUM_X;
		num_y = parameters_device->NUM_Y;
		num_z = parameters_device->NUM_Z;
		H_CENTER = parameters_device->H_CENTER_Y;
		V_CENTER = parameters_device->V_CENTER;
		SIGMA_INV = parameters_device->FWHM_sigma_inv;
		ALPHA = parameters_device->FWHM_alpha;
		range1 = parameters_device->RANGE1;
		range2 = parameters_device->RANGE2;
		TOF_INV = parameters_device->TOF_inv;
		TOF_ALPHA = parameters_device->TOF_alpha;

		ptr_src_x = _events_x_dominant_uvm->src_x;
		ptr_src_y = _events_x_dominant_uvm->src_y;
		ptr_src_z = _events_x_dominant_uvm->src_z;
		ptr_dest_x = _events_x_dominant_uvm->dest_x;
		ptr_dest_y = _events_x_dominant_uvm->dest_y;
		ptr_dest_z = _events_x_dominant_uvm->dest_z;
		ptr_FWHM = _events_x_dominant_uvm->coeff;
		ptr_sc_coeff = _events_x_dominant_uvm->sc_coeff;

		ptr_TOF_dist = _events_x_dominant_uvm->TOF_dist;
		num_lines = _events_x_dominant_uvm->num_lines;
		image_ptr = image;
		fp_value = fp_x;
	}
	c = 1;
	__syncthreads();

	current_slice = blockIdx.x;

	intersection_x = (current_slice - center_x - 0.5f)*voxel_size_x;
	for (sub_slice = 0; sub_slice<NUM_SUB_SLICE_x; sub_slice++){
		subslice_z0 = (num_z / NUM_SUB_SLICE_x)*sub_slice;
		subslice_z1 = subslice_z0 + SUB_SLICE_HEIGHT_x;//(num_z/NUM_SUB_SLICE)*(sub_slice+1);
		//load current image to shared memory
		for (int i = threadIdx.x; i<SIZE_SUB_SLICE_x; i = i + blockDim.x){
			current_slice_image[i] = image_ptr[(i + sub_slice*SIZE_SUB_SLICE_x)*num_x + current_slice];
		}
		__syncthreads();
		//loop through all the lines and compute the forward projection to this current slice
		for (event_index = threadIdx.x; event_index< num_lines; event_index = event_index + blockDim.x){
			//get event line parameters
			src_x = ptr_src_x[event_index];
			src_y = ptr_src_y[event_index];
			src_z = ptr_src_z[event_index];
			dest_x = ptr_dest_x[event_index];
			dest_y = ptr_dest_y[event_index];
			dest_z = ptr_dest_z[event_index];
			coeff = ptr_FWHM[event_index];
			sc_coeff = ptr_sc_coeff[event_index];

			TOF_dist = ptr_TOF_dist[event_index];


			//only do the computation if current event line intersects current slice
			if (src_x<intersection_x - voxel_size_x && dest_x>intersection_x + voxel_size_x){
				dir_x = dest_x - src_x;
				dir_y = dest_y - src_y;
				dir_z = dest_z - src_z;
				//dist_squared = __fmaf_rn (dir_z,dir_z, __fmaf_rn(dir_y,dir_y,__fmul_rn(dir_x,dir_x) ) );
				dist_squared = dir_x*dir_x + dir_y*dir_y + dir_z*dir_z;
				dist = __fsqrt_rn(dist_squared);

				unit_dir_vector_y = dir_y / dist;
				unit_dir_vector_z = dir_z / dist;

				//t keeps the ratio of the two segment of LOR segmented by the intersection plane
				t = (intersection_x - src_x) / dir_x;
				intersection_y = t*dir_y + src_y;
				intersection_z = t*dir_z + src_z;

				intersection_index_y = (int)ceilf(intersection_y / voxel_size_y) + center_y;
				intersection_index_z = (int)ceilf(intersection_z / voxel_size_z) + center_z;

				//compute the solid angle ratio w.r.t. the location of the intersection
				solid_angle_ratio = 1 + 2 * fabsf(t - 0.5f);
				//solid_angle_ratio = solid_angle_ratio*solid_angle_ratio;
				solid_angle_ratio = 1 * dist*dist;
				//compute the distance to the TOF center
				dist_to_TOF_center = (0.5f - t)*dist - TOF_dist;
				t = TOF_ALPHA * __expf(-dist_to_TOF_center*dist_to_TOF_center*TOF_INV);

				forward_sum = 0.0f;

				for (inslice_z = intersection_index_z - range2; inslice_z <= intersection_index_z + range2; inslice_z++){
					for (inslice_y = intersection_index_y - range1; inslice_y <= intersection_index_y + range1; inslice_y++){
						if (inslice_y >= 0 && inslice_y < num_y && inslice_z >= subslice_z0 && inslice_z < subslice_z1){
							//projection_vector_y = (inslice_y - center_y - 0.5)*voxel_size_y - intersection_y;
							projection_vector_y = inslice_y*voxel_size_y - H_CENTER - intersection_y;
							//projection_vector_z = (inslice_z - center_z - 0.5)*voxel_size_z - intersection_z;
							projection_vector_z = inslice_z*voxel_size_z - V_CENTER - intersection_z;
							dot_product = projection_vector_y*unit_dir_vector_y + projection_vector_z*unit_dir_vector_z;
							point_line_distance_square = projection_vector_y*projection_vector_y + projection_vector_z*projection_vector_z - dot_product*dot_product;

							forward_sum += current_slice_image[(inslice_z - subslice_z0)*num_y + inslice_y] * __expf(-point_line_distance_square*SIGMA_INV) * ALPHA * t *  GLOBAL_SCALE;

							//atomicAdd(&fp_value[event_index],weight);
						}
					}
				}

				atomicAdd(&fp_value[event_index], (forward_sum /solid_angle_ratio));
				//atomicAdd(&fp_value[event_index], sc_coeff);


			}//intersects current slice

			//atomicAdd(&fp_value[event_index], sc_coeff);

		}
	}
}


void
__global__ _TOF_fproj_lst_cuda_x_kernel_scatter_atten(float* image, float* fp_x, float* atten_per_event, LST_LORs_scatter *_events_x_dominant_uvm, PARAMETERS_IN_DEVICE_t* parameters_device){
	int event_index, current_slice, sub_slice;
	float src_x, src_y, src_z, dest_x, dest_y, dest_z;
	int coeff, c;
	float sc_coeff = 0;
	float dir_x, dir_y, dir_z;
	float TOF_dist;
	float dist_to_TOF_center;
	float unit_dir_vector_y, unit_dir_vector_z;
	float projection_vector_y, projection_vector_z;
	float dot_product;
	float point_line_distance_square;
	//float weight;
	float dist, dist_squared;
	float t;
	float solid_angle_ratio;
	float forward_sum,integral_mu;
	float intersection_x, intersection_y, intersection_z;
	int  intersection_index_y, intersection_index_z;
	int  inslice_y, inslice_z;
	int subslice_z0, subslice_z1;
	__shared__ float H_CENTER;
	__shared__ float V_CENTER;
	__shared__ float SIGMA_INV;
	__shared__ float ALPHA;
	__shared__ float current_slice_image[SIZE_SUB_SLICE_x];
	__shared__ float voxel_size_x, voxel_size_y, voxel_size_z;
	__shared__ int center_x, center_y, center_z;
	__shared__ int range1, range2;
	__shared__ int num_x, num_y, num_z;
	__shared__ float *ptr_src_x, *ptr_src_y, *ptr_src_z, *ptr_dest_x, *ptr_dest_y, *ptr_dest_z, *ptr_TOF_dist;
	__shared__ float *ptr_sc_coeff;
	__shared__ float *ptr_FWHM;
	__shared__ int num_lines;
	__shared__ float *image_ptr;
	__shared__ float *fp_value;
	__shared__ float *atten_factor;
	__shared__ float TOF_INV, TOF_ALPHA;
	if (threadIdx.x == 0){
		voxel_size_x = parameters_device->X_SAMP;
		voxel_size_y = parameters_device->Y_SAMP;
		voxel_size_z = parameters_device->Z_SAMP;
		center_x = parameters_device->X_INDEX_CENT;
		center_y = parameters_device->Y_INDEX_CENT;
		center_z = parameters_device->Z_INDEX_CENT;
		num_x = parameters_device->NUM_X;
		num_y = parameters_device->NUM_Y;
		num_z = parameters_device->NUM_Z;
		H_CENTER = parameters_device->H_CENTER_Y;
		V_CENTER = parameters_device->V_CENTER;
		SIGMA_INV = parameters_device->FWHM_sigma_inv;
		ALPHA = parameters_device->FWHM_alpha;
		range1 = parameters_device->RANGE1;
		range2 = parameters_device->RANGE2;
		TOF_INV = parameters_device->TOF_inv;
		TOF_ALPHA = parameters_device->TOF_alpha;

		ptr_src_x = _events_x_dominant_uvm->src_x;
		ptr_src_y = _events_x_dominant_uvm->src_y;
		ptr_src_z = _events_x_dominant_uvm->src_z;
		ptr_dest_x = _events_x_dominant_uvm->dest_x;
		ptr_dest_y = _events_x_dominant_uvm->dest_y;
		ptr_dest_z = _events_x_dominant_uvm->dest_z;
		ptr_FWHM = _events_x_dominant_uvm->coeff;
		ptr_sc_coeff = _events_x_dominant_uvm->sc_coeff;

		ptr_TOF_dist = _events_x_dominant_uvm->TOF_dist;
		num_lines = _events_x_dominant_uvm->num_lines;
		image_ptr = image;
		fp_value = fp_x;
		atten_factor = atten_per_event;
	}
	c = 1;
	__syncthreads();

	current_slice = blockIdx.x;

	intersection_x = (current_slice - center_x - 0.5f)*voxel_size_x;
	for (sub_slice = 0; sub_slice<NUM_SUB_SLICE_x; sub_slice++){
		subslice_z0 = (num_z / NUM_SUB_SLICE_x)*sub_slice;
		subslice_z1 = subslice_z0 + SUB_SLICE_HEIGHT_x;//(num_z/NUM_SUB_SLICE)*(sub_slice+1);
		//load current image to shared memory
		for (int i = threadIdx.x; i<SIZE_SUB_SLICE_x; i = i + blockDim.x){
			current_slice_image[i] = image_ptr[(i + sub_slice*SIZE_SUB_SLICE_x)*num_x + current_slice];
		}
		__syncthreads();
		//loop through all the lines and compute the forward projection to this current slice
		for (event_index = threadIdx.x; event_index< num_lines; event_index = event_index + blockDim.x){
			//get event line parameters
			src_x = ptr_src_x[event_index];
			src_y = ptr_src_y[event_index];
			src_z = ptr_src_z[event_index];
			dest_x = ptr_dest_x[event_index];
			dest_y = ptr_dest_y[event_index];
			dest_z = ptr_dest_z[event_index];
			coeff = ptr_FWHM[event_index];
			sc_coeff = ptr_sc_coeff[event_index];

			TOF_dist = ptr_TOF_dist[event_index];


			//only do the computation if current event line intersects current slice
			if (src_x<intersection_x - voxel_size_x && dest_x>intersection_x + voxel_size_x){
				dir_x = dest_x - src_x;
				dir_y = dest_y - src_y;
				dir_z = dest_z - src_z;
				//dist_squared = __fmaf_rn (dir_z,dir_z, __fmaf_rn(dir_y,dir_y,__fmul_rn(dir_x,dir_x) ) );
				dist_squared = dir_x*dir_x + dir_y*dir_y + dir_z*dir_z;
				dist = __fsqrt_rn(dist_squared);

				unit_dir_vector_y = dir_y / dist;
				unit_dir_vector_z = dir_z / dist;

				//t keeps the ratio of the two segment of LOR segmented by the intersection plane
				t = (intersection_x - src_x) / dir_x;
				intersection_y = t*dir_y + src_y;
				intersection_z = t*dir_z + src_z;

				intersection_index_y = (int)ceilf(intersection_y / voxel_size_y) + center_y;
				intersection_index_z = (int)ceilf(intersection_z / voxel_size_z) + center_z;

				//compute the solid angle ratio w.r.t. the location of the intersection
				solid_angle_ratio = 1 + 2 * fabsf(t - 0.5f);
				//solid_angle_ratio = solid_angle_ratio*solid_angle_ratio;
				solid_angle_ratio = 1 * dist*dist;
				//compute the distance to the TOF center
				dist_to_TOF_center = (0.5f - t)*dist - TOF_dist;
				t = TOF_ALPHA * __expf(-dist_to_TOF_center*dist_to_TOF_center*TOF_INV);

				forward_sum = 0.0f;

				integral_mu = atten_factor[event_index];

				for (inslice_z = intersection_index_z - range2; inslice_z <= intersection_index_z + range2; inslice_z++){
					for (inslice_y = intersection_index_y - range1; inslice_y <= intersection_index_y + range1; inslice_y++){
						if (inslice_y >= 0 && inslice_y < num_y && inslice_z >= subslice_z0 && inslice_z < subslice_z1){
							//projection_vector_y = (inslice_y - center_y - 0.5)*voxel_size_y - intersection_y;
							projection_vector_y = inslice_y*voxel_size_y - H_CENTER - intersection_y;
							//projection_vector_z = (inslice_z - center_z - 0.5)*voxel_size_z - intersection_z;
							projection_vector_z = inslice_z*voxel_size_z - V_CENTER - intersection_z;
							dot_product = projection_vector_y*unit_dir_vector_y + projection_vector_z*unit_dir_vector_z;
							point_line_distance_square = projection_vector_y*projection_vector_y + projection_vector_z*projection_vector_z - dot_product*dot_product;

							forward_sum += current_slice_image[(inslice_z - subslice_z0)*num_y + inslice_y] * __expf(-point_line_distance_square*SIGMA_INV - integral_mu) * ALPHA * t * GLOBAL_SCALE;

							//atomicAdd(&fp_value[event_index],weight);
						}
					}
				}

				atomicAdd(&fp_value[event_index], (forward_sum / solid_angle_ratio));
				//atomicAdd(&fp_value[event_index], forward_sum / solid_angle_ratio);


			}//intersects current slice

		}
	}
}



void
__global__ _TOF_fproj_lst_cuda_y_kernel(float* image, float* fp_y, LST_LORs *_events_y_dominant_uvm, PARAMETERS_IN_DEVICE_t* parameters_device){
	int event_index, current_slice, sub_slice;
	float src_x, src_y, src_z, dest_x, dest_y, dest_z;
	float current_FWHM,current_Alpha,current_FWHM_sigma_inv; //for PSF
	int coeff, c;
	float dir_x, dir_y, dir_z;
	float TOF_dist;
	float dist_to_TOF_center;
	float unit_dir_vector_x, unit_dir_vector_z;
	float projection_vector_x, projection_vector_z;
	float dot_product;
	float point_line_distance_square;
	//float weight;
	float dist, dist_squared;
	float t;
	float solid_angle_ratio;
	float forward_sum, increment;
	float intersection_x, intersection_y, intersection_z;
	int intersection_index_x, intersection_index_z;
	int inslice_x, inslice_z;
	int subslice_z0, subslice_z1;
	__shared__ float H_CENTER;
	__shared__ float V_CENTER;
	__shared__ float SIGMA_INV;
	__shared__ float ALPHA;
	__shared__ float current_slice_image[SIZE_SUB_SLICE_y];
	__shared__ float voxel_size_x, voxel_size_y, voxel_size_z;
	__shared__ int center_x, center_y, center_z;
	__shared__ int range1, range2;
	__shared__ int num_x, num_y, num_z;
	__shared__ float *ptr_src_x, *ptr_src_y, *ptr_src_z, *ptr_dest_x, *ptr_dest_y, *ptr_dest_z, *ptr_TOF_dist;
	__shared__ float *ptr_FWHM;
	__shared__ int num_lines;
	__shared__ float *image_ptr;
	__shared__ float *fp_value;
	__shared__ float TOF_INV, TOF_ALPHA;
	if (threadIdx.x == 0){
		voxel_size_x = parameters_device->X_SAMP;
		voxel_size_y = parameters_device->Y_SAMP;
		voxel_size_z = parameters_device->Z_SAMP;
		center_x = parameters_device->X_INDEX_CENT;
		center_y = parameters_device->Y_INDEX_CENT;
		center_z = parameters_device->Z_INDEX_CENT;
		num_x = parameters_device->NUM_X;
		num_y = parameters_device->NUM_Y;
		num_z = parameters_device->NUM_Z;
		H_CENTER = parameters_device->H_CENTER_X;
		V_CENTER = parameters_device->V_CENTER;
		SIGMA_INV = parameters_device->FWHM_sigma_inv;
		ALPHA = parameters_device->FWHM_alpha;
		range1 = parameters_device->RANGE1;
		range2 = parameters_device->RANGE2;
		TOF_INV = parameters_device->TOF_inv;
		TOF_ALPHA = parameters_device->TOF_alpha;

		ptr_src_x = _events_y_dominant_uvm->src_x;
		ptr_src_y = _events_y_dominant_uvm->src_y;
		ptr_src_z = _events_y_dominant_uvm->src_z;
		ptr_dest_x = _events_y_dominant_uvm->dest_x;
		ptr_dest_y = _events_y_dominant_uvm->dest_y;
		ptr_dest_z = _events_y_dominant_uvm->dest_z;
		ptr_FWHM = _events_y_dominant_uvm->coeff;
		
		ptr_TOF_dist = _events_y_dominant_uvm->TOF_dist;
		num_lines = _events_y_dominant_uvm->num_lines;
		image_ptr = image;
		fp_value = fp_y;
	}
	__syncthreads();
	current_slice = blockIdx.x;

	intersection_y = (current_slice - center_y - 0.5)*voxel_size_y;
	for (sub_slice = 0; sub_slice<NUM_SUB_SLICE_y; sub_slice++){
		subslice_z0 = (num_z / NUM_SUB_SLICE_y)*sub_slice;
		subslice_z1 = subslice_z0 + SUB_SLICE_HEIGHT_y;//(num_z/NUM_SUB_SLICE)*(sub_slice+1);
		//load current image slice to shared memory
		for (int i = threadIdx.x; i<SIZE_SUB_SLICE_y; i = i + blockDim.x){
			current_slice_image[i] = image_ptr[num_x*num_y*(subslice_z0 + i / num_x) + num_x*current_slice + i%num_x];
		}

		__syncthreads();

		c = 1;
		//loop through all the lines and compute the backprojection to this current slice
		for (event_index = threadIdx.x; event_index< num_lines; event_index = event_index + blockDim.x){
			//get event line parameters
			src_x = ptr_src_x[event_index];
			src_y = ptr_src_y[event_index];
			src_z = ptr_src_z[event_index];
			dest_x = ptr_dest_x[event_index];
			dest_y = ptr_dest_y[event_index];
			dest_z = ptr_dest_z[event_index];
			current_FWHM = ptr_FWHM[event_index];
			current_FWHM_sigma_inv = 4 * log(2.0) / (current_FWHM*current_FWHM);
		    current_Alpha =0.93943727 / current_FWHM;		
			TOF_dist = ptr_TOF_dist[event_index];

			

				//only do the computation if current event line intersects current slice
				if (src_y<intersection_y - voxel_size_y && dest_y>intersection_y + voxel_size_y){
					dir_x = dest_x - src_x;
					dir_y = dest_y - src_y;
					dir_z = dest_z - src_z;
					//dist_squared = __fmaf_rn (dir_z,dir_z, __fmaf_rn(dir_y,dir_y,__fmul_rn(dir_x,dir_x) ) );
					dist_squared = dir_x*dir_x + dir_y*dir_y + dir_z*dir_z;
					dist = __fsqrt_rn(dist_squared);

					unit_dir_vector_x = dir_x / dist;
					unit_dir_vector_z = dir_z / dist;

					t = (intersection_y - src_y) / dir_y;
					intersection_x = t*dir_x + src_x;
					intersection_z = t*dir_z + src_z;

					intersection_index_x = (int)ceilf(intersection_x / voxel_size_x) + center_x;
					intersection_index_z = (int)ceilf(intersection_z / voxel_size_z) + center_z;

					//compute the solid angle ratio w.r.t. the location of the intersection
					solid_angle_ratio = 1 + 2 * fabsf(t - 0.5f);
					//solid_angle_ratio = solid_angle_ratio*solid_angle_ratio*dist*dist;
					solid_angle_ratio = 1 * dist*dist;


					//compute the distance to the TOF center
					dist_to_TOF_center = (0.5f - t)*dist - TOF_dist;
					t = TOF_ALPHA * __expf(-dist_to_TOF_center*dist_to_TOF_center*TOF_INV);

					forward_sum = 0.0f;
					
					for (inslice_z = intersection_index_z - range2; inslice_z <= intersection_index_z + range2; inslice_z++){
						for (inslice_x = intersection_index_x - range1; inslice_x <= intersection_index_x + range1; inslice_x++){
							if (inslice_x >= 0 && inslice_x < num_x && inslice_z >= subslice_z0 && inslice_z < subslice_z1){
								//projection_vector_x = (inslice_x - center_x - 0.5)*voxel_size_x - intersection_x;
								projection_vector_x = inslice_x*voxel_size_x - H_CENTER - intersection_x;
								//projection_vector_z = (inslice_z - center_z - 0.5)*voxel_size_z - intersection_z;
								projection_vector_z = inslice_z*voxel_size_z - V_CENTER - intersection_z;
								dot_product = projection_vector_x*unit_dir_vector_x + projection_vector_z*unit_dir_vector_z;
								point_line_distance_square = projection_vector_x*projection_vector_x + projection_vector_z*projection_vector_z - dot_product*dot_product;
								//increment = current_slice_image[(inslice_z - subslice_z0)*num_x + inslice_x] * __expf(-point_line_distance_square*SIGMA_INV) * ALPHA  * t * GLOBAL_SCALE;
								increment = current_slice_image[(inslice_z - subslice_z0)*num_x + inslice_x] * __expf(-point_line_distance_square*current_FWHM_sigma_inv) * current_Alpha * t * GLOBAL_SCALE;
								forward_sum += increment;
								//forward_sum += current_slice_image[(inslice_z - subslice_z0)*num_x + inslice_x] ;
								//atomicAdd(&fp_value[event_index],weight);

							}
						}
					}
					
					atomicAdd(&fp_value[event_index], forward_sum/solid_angle_ratio );
					//atomicAdd(&fp_value[event_index], forward_sum);
				}//intersects current slice
			
		}
	}
}


void
__global__ _TOF_fproj_lst_cuda_y_kernel_atten(float* image, float* fp_y, float* atten_per_event, LST_LORs *_events_y_dominant_uvm, PARAMETERS_IN_DEVICE_t* parameters_device){
	int event_index, current_slice, sub_slice;
	float src_x, src_y, src_z, dest_x, dest_y, dest_z;
	float current_FWHM,current_Alpha,current_FWHM_sigma_inv; //for PSF
	int coeff, c;
	float dir_x, dir_y, dir_z;
	float TOF_dist;
	float dist_to_TOF_center;
	float unit_dir_vector_x, unit_dir_vector_z;
	float projection_vector_x, projection_vector_z;
	float dot_product;
	float point_line_distance_square;
	//float weight;
	float dist, dist_squared;
	float t;
	float solid_angle_ratio;
	float forward_sum, integral_mu;
	float intersection_x, intersection_y, intersection_z;
	int intersection_index_x, intersection_index_z;
	int inslice_x, inslice_z;
	int subslice_z0, subslice_z1;
	__shared__ float H_CENTER;
	__shared__ float V_CENTER;
	__shared__ float SIGMA_INV;
	__shared__ float ALPHA;
	__shared__ float current_slice_image[SIZE_SUB_SLICE_y];
	__shared__ float voxel_size_x, voxel_size_y, voxel_size_z;
	__shared__ int center_x, center_y, center_z;
	__shared__ int range1, range2;
	__shared__ int num_x, num_y, num_z;
	__shared__ float *ptr_src_x, *ptr_src_y, *ptr_src_z, *ptr_dest_x, *ptr_dest_y, *ptr_dest_z, *ptr_TOF_dist;
	__shared__ float *ptr_FWHM;
	__shared__ int num_lines;
	__shared__ float *image_ptr;
	__shared__ float *fp_value;
	__shared__ float *atten_factor;
	__shared__ float TOF_INV, TOF_ALPHA;
	if (threadIdx.x == 0){
		voxel_size_x = parameters_device->X_SAMP;
		voxel_size_y = parameters_device->Y_SAMP;
		voxel_size_z = parameters_device->Z_SAMP;
		center_x = parameters_device->X_INDEX_CENT;
		center_y = parameters_device->Y_INDEX_CENT;
		center_z = parameters_device->Z_INDEX_CENT;
		num_x = parameters_device->NUM_X;
		num_y = parameters_device->NUM_Y;
		num_z = parameters_device->NUM_Z;
		H_CENTER = parameters_device->H_CENTER_X;
		V_CENTER = parameters_device->V_CENTER;
		SIGMA_INV = parameters_device->FWHM_sigma_inv;
		ALPHA = parameters_device->FWHM_alpha;
		range1 = parameters_device->RANGE1;
		range2 = parameters_device->RANGE2;
		TOF_INV = parameters_device->TOF_inv;
		TOF_ALPHA = parameters_device->TOF_alpha;

		ptr_src_x = _events_y_dominant_uvm->src_x;
		ptr_src_y = _events_y_dominant_uvm->src_y;
		ptr_src_z = _events_y_dominant_uvm->src_z;
		ptr_dest_x = _events_y_dominant_uvm->dest_x;
		ptr_dest_y = _events_y_dominant_uvm->dest_y;
		ptr_dest_z = _events_y_dominant_uvm->dest_z;
		ptr_FWHM = _events_y_dominant_uvm->coeff;

		ptr_TOF_dist = _events_y_dominant_uvm->TOF_dist;
		num_lines = _events_y_dominant_uvm->num_lines;
		image_ptr = image;
		fp_value = fp_y;
		atten_factor = atten_per_event;
	}
	__syncthreads();
	current_slice = blockIdx.x;

	intersection_y = (current_slice - center_y - 0.5)*voxel_size_y;
	for (sub_slice = 0; sub_slice<NUM_SUB_SLICE_y; sub_slice++){
		subslice_z0 = (num_z / NUM_SUB_SLICE_y)*sub_slice;
		subslice_z1 = subslice_z0 + SUB_SLICE_HEIGHT_y;//(num_z/NUM_SUB_SLICE)*(sub_slice+1);
		//load current image slice to shared memory
		for (int i = threadIdx.x; i<SIZE_SUB_SLICE_y; i = i + blockDim.x){
			current_slice_image[i] = image_ptr[num_x*num_y*(subslice_z0 + i / num_x) + num_x*current_slice + i%num_x];
		}

		__syncthreads();

		c = 1;
		//loop through all the lines and compute the backprojection to this current slice
		for (event_index = threadIdx.x; event_index< num_lines; event_index = event_index + blockDim.x){
			//get event line parameters
			src_x = ptr_src_x[event_index];
			src_y = ptr_src_y[event_index];
			src_z = ptr_src_z[event_index];
			dest_x = ptr_dest_x[event_index];
			dest_y = ptr_dest_y[event_index];
			dest_z = ptr_dest_z[event_index];
			current_FWHM = ptr_FWHM[event_index];
			current_FWHM_sigma_inv = 4 * log(2.0) / (current_FWHM*current_FWHM);
		    current_Alpha =0.93943727 / current_FWHM;
			
			TOF_dist = ptr_TOF_dist[event_index];



			//only do the computation if current event line intersects current slice
			if (src_y<intersection_y - voxel_size_y && dest_y>intersection_y + voxel_size_y){
				dir_x = dest_x - src_x;
				dir_y = dest_y - src_y;
				dir_z = dest_z - src_z;
				//dist_squared = __fmaf_rn (dir_z,dir_z, __fmaf_rn(dir_y,dir_y,__fmul_rn(dir_x,dir_x) ) );
				dist_squared = dir_x*dir_x + dir_y*dir_y + dir_z*dir_z;
				dist = __fsqrt_rn(dist_squared);

				unit_dir_vector_x = dir_x / dist;
				unit_dir_vector_z = dir_z / dist;

				t = (intersection_y - src_y) / dir_y;
				intersection_x = t*dir_x + src_x;
				intersection_z = t*dir_z + src_z;

				intersection_index_x = (int)ceilf(intersection_x / voxel_size_x) + center_x;
				intersection_index_z = (int)ceilf(intersection_z / voxel_size_z) + center_z;

				//compute the solid angle ratio w.r.t. the location of the intersection
				solid_angle_ratio = 1 + 2 * fabsf(t - 0.5f);
				//solid_angle_ratio = solid_angle_ratio*solid_angle_ratio;
				solid_angle_ratio = 1 * dist*dist;


				//compute the distance to the TOF center
				dist_to_TOF_center = (0.5f - t)*dist - TOF_dist;
				t = TOF_ALPHA * __expf(-dist_to_TOF_center*dist_to_TOF_center*TOF_INV);

				integral_mu = atten_factor[event_index];

				forward_sum = 0.0f;

				for (inslice_z = intersection_index_z - range2; inslice_z <= intersection_index_z + range2; inslice_z++){
					for (inslice_x = intersection_index_x - range1; inslice_x <= intersection_index_x + range1; inslice_x++){
						if (inslice_x >= 0 && inslice_x < num_x && inslice_z >= subslice_z0 && inslice_z < subslice_z1){
							//projection_vector_x = (inslice_x - center_x - 0.5)*voxel_size_x - intersection_x;
							projection_vector_x = inslice_x*voxel_size_x - H_CENTER - intersection_x;
							//projection_vector_z = (inslice_z - center_z - 0.5)*voxel_size_z - intersection_z;
							projection_vector_z = inslice_z*voxel_size_z - V_CENTER - intersection_z;
							dot_product = projection_vector_x*unit_dir_vector_x + projection_vector_z*unit_dir_vector_z;
							point_line_distance_square = projection_vector_x*projection_vector_x + projection_vector_z*projection_vector_z - dot_product*dot_product;

							forward_sum += current_slice_image[(inslice_z - subslice_z0)*num_x + inslice_x] * __expf(-point_line_distance_square*current_FWHM_sigma_inv) * current_Alpha  * t * GLOBAL_SCALE;

							//atomicAdd(&fp_value[event_index],weight);

						}
					}
				}

				atomicAdd(&fp_value[event_index], (forward_sum * __expf(-integral_mu) )/ solid_angle_ratio);
			}//intersects current slice

		}
	}
}


void
__global__ _TOF_fproj_lst_cuda_y_kernel_scatter(float* image, float* fp_y, LST_LORs_scatter *_events_y_dominant_uvm, PARAMETERS_IN_DEVICE_t* parameters_device){
	int event_index, current_slice, sub_slice;
	float src_x, src_y, src_z, dest_x, dest_y, dest_z;
	int coeff, c;
	float sc_coeff = 0.0;
	float dir_x, dir_y, dir_z;
	float TOF_dist;
	float dist_to_TOF_center;
	float unit_dir_vector_x, unit_dir_vector_z;
	float projection_vector_x, projection_vector_z;
	float dot_product;
	float point_line_distance_square;
	//float weight;
	float dist, dist_squared;
	float t;
	float solid_angle_ratio;
	float forward_sum;
	float intersection_x, intersection_y, intersection_z;
	int intersection_index_x, intersection_index_z;
	int inslice_x, inslice_z;
	int subslice_z0, subslice_z1;
	__shared__ float H_CENTER;
	__shared__ float V_CENTER;
	__shared__ float SIGMA_INV;
	__shared__ float ALPHA;
	__shared__ float current_slice_image[SIZE_SUB_SLICE_y];
	__shared__ float voxel_size_x, voxel_size_y, voxel_size_z;
	__shared__ int center_x, center_y, center_z;
	__shared__ int range1, range2;
	__shared__ int num_x, num_y, num_z;
	__shared__ float *ptr_src_x, *ptr_src_y, *ptr_src_z, *ptr_dest_x, *ptr_dest_y, *ptr_dest_z, *ptr_TOF_dist;
	__shared__ float *ptr_FWHM;
	__shared__ float *ptr_sc_coeff;
	__shared__ int num_lines;
	__shared__ float *image_ptr;
	__shared__ float *fp_value;
	__shared__ float TOF_INV, TOF_ALPHA;
	if (threadIdx.x == 0){
		voxel_size_x = parameters_device->X_SAMP;
		voxel_size_y = parameters_device->Y_SAMP;
		voxel_size_z = parameters_device->Z_SAMP;
		center_x = parameters_device->X_INDEX_CENT;
		center_y = parameters_device->Y_INDEX_CENT;
		center_z = parameters_device->Z_INDEX_CENT;
		num_x = parameters_device->NUM_X;
		num_y = parameters_device->NUM_Y;
		num_z = parameters_device->NUM_Z;
		H_CENTER = parameters_device->H_CENTER_X;
		V_CENTER = parameters_device->V_CENTER;
		SIGMA_INV = parameters_device->FWHM_sigma_inv;
		ALPHA = parameters_device->FWHM_alpha;
		range1 = parameters_device->RANGE1;
		range2 = parameters_device->RANGE2;
		TOF_INV = parameters_device->TOF_inv;
		TOF_ALPHA = parameters_device->TOF_alpha;

		ptr_src_x = _events_y_dominant_uvm->src_x;
		ptr_src_y = _events_y_dominant_uvm->src_y;
		ptr_src_z = _events_y_dominant_uvm->src_z;
		ptr_dest_x = _events_y_dominant_uvm->dest_x;
		ptr_dest_y = _events_y_dominant_uvm->dest_y;
		ptr_dest_z = _events_y_dominant_uvm->dest_z;
		ptr_FWHM = _events_y_dominant_uvm->coeff;
		ptr_sc_coeff = _events_y_dominant_uvm->sc_coeff;

		ptr_TOF_dist = _events_y_dominant_uvm->TOF_dist;
		num_lines = _events_y_dominant_uvm->num_lines;
		image_ptr = image;
		fp_value = fp_y;
	}
	__syncthreads();
	current_slice = blockIdx.x;

	intersection_y = (current_slice - center_y - 0.5)*voxel_size_y;
	for (sub_slice = 0; sub_slice<NUM_SUB_SLICE_y; sub_slice++){
		subslice_z0 = (num_z / NUM_SUB_SLICE_y)*sub_slice;
		subslice_z1 = subslice_z0 + SUB_SLICE_HEIGHT_y;//(num_z/NUM_SUB_SLICE)*(sub_slice+1);
		//load current image slice to shared memory
		for (int i = threadIdx.x; i<SIZE_SUB_SLICE_y; i = i + blockDim.x){
			current_slice_image[i] = image_ptr[num_x*num_y*(subslice_z0 + i / num_x) + num_x*current_slice + i%num_x];
		}

		__syncthreads();

		c = 1;
		//loop through all the lines and compute the backprojection to this current slice
		for (event_index = threadIdx.x; event_index< num_lines; event_index = event_index + blockDim.x){
			//get event line parameters
			src_x = ptr_src_x[event_index];
			src_y = ptr_src_y[event_index];
			src_z = ptr_src_z[event_index];
			dest_x = ptr_dest_x[event_index];
			dest_y = ptr_dest_y[event_index];
			dest_z = ptr_dest_z[event_index];
			coeff = ptr_FWHM[event_index];
			sc_coeff = ptr_sc_coeff[event_index];

			TOF_dist = ptr_TOF_dist[event_index];



			//only do the computation if current event line intersects current slice
			if (src_y<intersection_y - voxel_size_y && dest_y>intersection_y + voxel_size_y){
				dir_x = dest_x - src_x;
				dir_y = dest_y - src_y;
				dir_z = dest_z - src_z;
				//dist_squared = __fmaf_rn (dir_z,dir_z, __fmaf_rn(dir_y,dir_y,__fmul_rn(dir_x,dir_x) ) );
				dist_squared = dir_x*dir_x + dir_y*dir_y + dir_z*dir_z;
				dist = __fsqrt_rn(dist_squared);

				unit_dir_vector_x = dir_x / dist;
				unit_dir_vector_z = dir_z / dist;

				t = (intersection_y - src_y) / dir_y;
				intersection_x = t*dir_x + src_x;
				intersection_z = t*dir_z + src_z;

				intersection_index_x = (int)ceilf(intersection_x / voxel_size_x) + center_x;
				intersection_index_z = (int)ceilf(intersection_z / voxel_size_z) + center_z;

				//compute the solid angle ratio w.r.t. the location of the intersection
				solid_angle_ratio = 1 + 2 * fabsf(t - 0.5f);
				//solid_angle_ratio = solid_angle_ratio*solid_angle_ratio;
				solid_angle_ratio = 1 * dist*dist;


				//compute the distance to the TOF center
				dist_to_TOF_center = (0.5f - t)*dist - TOF_dist;
				t = TOF_ALPHA * __expf(-dist_to_TOF_center*dist_to_TOF_center*TOF_INV);

				forward_sum = 0.0f;

				for (inslice_z = intersection_index_z - range2; inslice_z <= intersection_index_z + range2; inslice_z++){
					for (inslice_x = intersection_index_x - range1; inslice_x <= intersection_index_x + range1; inslice_x++){
						if (inslice_x >= 0 && inslice_x < num_x && inslice_z >= subslice_z0 && inslice_z < subslice_z1){
							//projection_vector_x = (inslice_x - center_x - 0.5)*voxel_size_x - intersection_x;
							projection_vector_x = inslice_x*voxel_size_x - H_CENTER - intersection_x;
							//projection_vector_z = (inslice_z - center_z - 0.5)*voxel_size_z - intersection_z;
							projection_vector_z = inslice_z*voxel_size_z - V_CENTER - intersection_z;
							dot_product = projection_vector_x*unit_dir_vector_x + projection_vector_z*unit_dir_vector_z;
							point_line_distance_square = projection_vector_x*projection_vector_x + projection_vector_z*projection_vector_z - dot_product*dot_product;

							forward_sum += current_slice_image[(inslice_z - subslice_z0)*num_x + inslice_x] * __expf(-point_line_distance_square*SIGMA_INV) * ALPHA  * t * GLOBAL_SCALE;

							//atomicAdd(&fp_value[event_index],weight);

						}
					}
				}

				atomicAdd(&fp_value[event_index], (forward_sum / solid_angle_ratio));
				//atomicAdd(&fp_value[event_index], sc_coeff);
			}//intersects current slice
			//atomicAdd(&fp_value[event_index], sc_coeff);
		}
	}
}

void
__global__ _TOF_fproj_lst_cuda_y_kernel_scatter_atten(float* image, float* fp_y, float* atten_per_event, LST_LORs_scatter *_events_y_dominant_uvm, PARAMETERS_IN_DEVICE_t* parameters_device){
	int event_index, current_slice, sub_slice;
	float src_x, src_y, src_z, dest_x, dest_y, dest_z;
	int coeff, c, count;
	float sc_coeff;
	float dir_x, dir_y, dir_z;
	float TOF_dist;
	float dist_to_TOF_center;
	float unit_dir_vector_x, unit_dir_vector_z;
	float projection_vector_x, projection_vector_z;
	float dot_product;
	float point_line_distance_square;
	//float weight;
	float dist, dist_squared;
	float t;
	float solid_angle_ratio;
	float forward_sum, integral_mu;
	float intersection_x, intersection_y, intersection_z;
	int intersection_index_x, intersection_index_z;
	int inslice_x, inslice_z;
	int subslice_z0, subslice_z1;
	__shared__ float H_CENTER;
	__shared__ float V_CENTER;
	__shared__ float SIGMA_INV;
	__shared__ float ALPHA;
	__shared__ float current_slice_image[SIZE_SUB_SLICE_y];
	__shared__ float voxel_size_x, voxel_size_y, voxel_size_z;
	__shared__ int center_x, center_y, center_z;
	__shared__ int range1, range2;
	__shared__ int num_x, num_y, num_z;
	__shared__ float *ptr_src_x, *ptr_src_y, *ptr_src_z, *ptr_dest_x, *ptr_dest_y, *ptr_dest_z, *ptr_TOF_dist;
	__shared__ float *ptr_FWHM;
	__shared__ float *ptr_sc_coeff;
	__shared__ int num_lines;
	__shared__ float *image_ptr;
	__shared__ float *fp_value;
	__shared__ float *atten_factor;
	__shared__ float TOF_INV, TOF_ALPHA;
	if (threadIdx.x == 0){
		voxel_size_x = parameters_device->X_SAMP;
		voxel_size_y = parameters_device->Y_SAMP;
		voxel_size_z = parameters_device->Z_SAMP;
		center_x = parameters_device->X_INDEX_CENT;
		center_y = parameters_device->Y_INDEX_CENT;
		center_z = parameters_device->Z_INDEX_CENT;
		num_x = parameters_device->NUM_X;
		num_y = parameters_device->NUM_Y;
		num_z = parameters_device->NUM_Z;
		H_CENTER = parameters_device->H_CENTER_X;
		V_CENTER = parameters_device->V_CENTER;
		SIGMA_INV = parameters_device->FWHM_sigma_inv;
		ALPHA = parameters_device->FWHM_alpha;
		range1 = parameters_device->RANGE1;
		range2 = parameters_device->RANGE2;
		TOF_INV = parameters_device->TOF_inv;
		TOF_ALPHA = parameters_device->TOF_alpha;

		ptr_src_x = _events_y_dominant_uvm->src_x;
		ptr_src_y = _events_y_dominant_uvm->src_y;
		ptr_src_z = _events_y_dominant_uvm->src_z;
		ptr_dest_x = _events_y_dominant_uvm->dest_x;
		ptr_dest_y = _events_y_dominant_uvm->dest_y;
		ptr_dest_z = _events_y_dominant_uvm->dest_z;
		ptr_FWHM = _events_y_dominant_uvm->coeff;
		ptr_sc_coeff = _events_y_dominant_uvm->sc_coeff;

		ptr_TOF_dist = _events_y_dominant_uvm->TOF_dist;
		num_lines = _events_y_dominant_uvm->num_lines;
		image_ptr = image;
		atten_factor = atten_per_event;
		fp_value = fp_y;
	}
	__syncthreads();
	current_slice = blockIdx.x;

	intersection_y = (current_slice - center_y - 0.5)*voxel_size_y;
	count = 1;
	for (sub_slice = 0; sub_slice<NUM_SUB_SLICE_y; sub_slice++){
		subslice_z0 = (num_z / NUM_SUB_SLICE_y)*sub_slice;
		subslice_z1 = subslice_z0 + SUB_SLICE_HEIGHT_y;//(num_z/NUM_SUB_SLICE)*(sub_slice+1);
		//load current image slice to shared memory
		for (int i = threadIdx.x; i<SIZE_SUB_SLICE_y; i = i + blockDim.x){
			current_slice_image[i] = image_ptr[num_x*num_y*(subslice_z0 + i / num_x) + num_x*current_slice + i%num_x];
		}

		__syncthreads();

		c = 1;
		//loop through all the lines and compute the backprojection to this current slice
		for (event_index = threadIdx.x; event_index< num_lines; event_index = event_index + blockDim.x){
			//get event line parameters
			src_x = ptr_src_x[event_index];
			src_y = ptr_src_y[event_index];
			src_z = ptr_src_z[event_index];
			dest_x = ptr_dest_x[event_index];
			dest_y = ptr_dest_y[event_index];
			dest_z = ptr_dest_z[event_index];
			coeff = ptr_FWHM[event_index];
			sc_coeff = ptr_sc_coeff[event_index];

			TOF_dist = ptr_TOF_dist[event_index];



			//only do the computation if current event line intersects current slice
			if (src_y<intersection_y - voxel_size_y && dest_y>intersection_y + voxel_size_y){
				dir_x = dest_x - src_x;
				dir_y = dest_y - src_y;
				dir_z = dest_z - src_z;
				//dist_squared = __fmaf_rn (dir_z,dir_z, __fmaf_rn(dir_y,dir_y,__fmul_rn(dir_x,dir_x) ) );
				dist_squared = dir_x*dir_x + dir_y*dir_y + dir_z*dir_z;
				dist = __fsqrt_rn(dist_squared);

				unit_dir_vector_x = dir_x / dist;
				unit_dir_vector_z = dir_z / dist;

				t = (intersection_y - src_y) / dir_y;
				intersection_x = t*dir_x + src_x;
				intersection_z = t*dir_z + src_z;

				intersection_index_x = (int)ceilf(intersection_x / voxel_size_x) + center_x;
				intersection_index_z = (int)ceilf(intersection_z / voxel_size_z) + center_z;

				//compute the solid angle ratio w.r.t. the location of the intersection
				solid_angle_ratio = 1 + 2 * fabsf(t - 0.5f);
				//solid_angle_ratio = solid_angle_ratio*solid_angle_ratio;
				solid_angle_ratio = 1 * dist*dist;

				//compute the distance to the TOF center
				dist_to_TOF_center = (0.5f - t)*dist - TOF_dist;
				t = TOF_ALPHA * __expf(-dist_to_TOF_center*dist_to_TOF_center*TOF_INV);

				forward_sum = 0.0f;

				integral_mu = atten_factor[event_index];



				for (inslice_z = intersection_index_z - range2; inslice_z <= intersection_index_z + range2; inslice_z++){
					for (inslice_x = intersection_index_x - range1; inslice_x <= intersection_index_x + range1; inslice_x++){
						if (inslice_x >= 0 && inslice_x < num_x && inslice_z >= subslice_z0 && inslice_z < subslice_z1){
							//projection_vector_x = (inslice_x - center_x - 0.5)*voxel_size_x - intersection_x;
							projection_vector_x = inslice_x*voxel_size_x - H_CENTER - intersection_x;
							//projection_vector_z = (inslice_z - center_z - 0.5)*voxel_size_z - intersection_z;
							projection_vector_z = inslice_z*voxel_size_z - V_CENTER - intersection_z;
							dot_product = projection_vector_x*unit_dir_vector_x + projection_vector_z*unit_dir_vector_z;
							point_line_distance_square = projection_vector_x*projection_vector_x + projection_vector_z*projection_vector_z - dot_product*dot_product;

							forward_sum += current_slice_image[(inslice_z - subslice_z0)*num_x + inslice_x] * __expf(-point_line_distance_square*SIGMA_INV-integral_mu) * ALPHA  * t * GLOBAL_SCALE;

							//atomicAdd(&fp_value[event_index],weight);

						}
					}
				}

				atomicAdd(&fp_value[event_index], (forward_sum / solid_angle_ratio));
				//atomicAdd(&fp_value[event_index], forward_sum / solid_angle_ratio);
			}//intersects current slice

		}

	}
}


void
__global__ _TOF_b_ratio_proj_lst_cuda_x_kernel(float* image, float* fp_x, LST_LORs *_events_x_dominant_uvm, PARAMETERS_IN_DEVICE_t* parameters_device){
	int event_index, current_slice, sub_slice;
	float src_x, src_y, src_z, dest_x, dest_y, dest_z;
	float current_FWHM,current_Alpha,current_FWHM_sigma_inv; //for PSF
    int coeff, c;
	float dir_x, dir_y, dir_z;
	float TOF_dist;
	float dist_to_TOF_center;
	float unit_dir_vector_y, unit_dir_vector_z;
	float projection_vector_y, projection_vector_z;
	float dot_product;
	float point_line_distance_square;
	float weight;
	float dist, dist_squared;
	float t;
	float solid_angle_ratio;
	float backward_ratio;
	float intersection_x, intersection_y, intersection_z;
	int  intersection_index_y, intersection_index_z;
	int  inslice_y, inslice_z;
	int subslice_z0, subslice_z1;
	__shared__ float H_CENTER;
	__shared__ float V_CENTER;
	__shared__ float SIGMA_INV;
	__shared__ float ALPHA;
	__shared__ float current_slice_image[SIZE_SUB_SLICE_x];
	__shared__ float voxel_size_x, voxel_size_y, voxel_size_z;
	__shared__ int center_x, center_y, center_z;
	__shared__ int range1, range2;
	__shared__ int num_x, num_y, num_z;
	__shared__ float *ptr_src_x, *ptr_src_y, *ptr_src_z, *ptr_dest_x, *ptr_dest_y, *ptr_dest_z, *ptr_TOF_dist;
	__shared__ float *ptr_FWHM;
	__shared__ int num_lines;
	__shared__ float *image_ptr;
	__shared__ float *fp_value;
	__shared__ float TOF_INV, TOF_ALPHA;
	if (threadIdx.x == 0){
		voxel_size_x = parameters_device->X_SAMP;
		voxel_size_y = parameters_device->Y_SAMP;
		voxel_size_z = parameters_device->Z_SAMP;
		center_x = parameters_device->X_INDEX_CENT;
		center_y = parameters_device->Y_INDEX_CENT;
		center_z = parameters_device->Z_INDEX_CENT;
		num_x = parameters_device->NUM_X;
		num_y = parameters_device->NUM_Y;
		num_z = parameters_device->NUM_Z;
		H_CENTER = parameters_device->H_CENTER_Y;
		V_CENTER = parameters_device->V_CENTER;
		SIGMA_INV = parameters_device->FWHM_sigma_inv;
		ALPHA = parameters_device->FWHM_alpha;
		range1 = parameters_device->RANGE1;
		range2 = parameters_device->RANGE2;
		TOF_INV = parameters_device->TOF_inv;
		TOF_ALPHA = parameters_device->TOF_alpha;

		ptr_src_x = _events_x_dominant_uvm->src_x;
		ptr_src_y = _events_x_dominant_uvm->src_y;
		ptr_src_z = _events_x_dominant_uvm->src_z;
		ptr_dest_x = _events_x_dominant_uvm->dest_x;
		ptr_dest_y = _events_x_dominant_uvm->dest_y;
		ptr_dest_z = _events_x_dominant_uvm->dest_z;
		ptr_TOF_dist = _events_x_dominant_uvm->TOF_dist;
		ptr_FWHM = _events_x_dominant_uvm->coeff;
	
        num_lines = _events_x_dominant_uvm->num_lines;
		image_ptr = image;
		fp_value = fp_x;
	}

	//initialize shared memory to zero
	for (int i = threadIdx.x; i<SIZE_SUB_SLICE_x; i = i + blockDim.x){
		current_slice_image[i] = 0.0f;
	}
	__syncthreads();
	c = 1;
	current_slice = blockIdx.x;
	intersection_x = __fmul_rn((current_slice - center_x - 0.5f), voxel_size_x);
	for (sub_slice = 0; sub_slice<NUM_SUB_SLICE_x; sub_slice++){
		subslice_z0 = (num_z / NUM_SUB_SLICE_x)*sub_slice;
		subslice_z1 = subslice_z0 + SUB_SLICE_HEIGHT_x;//(num_z/NUM_SUB_SLICE)*(sub_slice+1);
		//loop through all the lines and compute the backprojection to this current slice
		for (event_index = threadIdx.x; event_index< num_lines; event_index = event_index + blockDim.x){
			//get event line parameters
			src_x = ptr_src_x[event_index];
			src_y = ptr_src_y[event_index];
			src_z = ptr_src_z[event_index];
			dest_x = ptr_dest_x[event_index];
			dest_y = ptr_dest_y[event_index];
			dest_z = ptr_dest_z[event_index];
			current_FWHM = ptr_FWHM[event_index];
			current_FWHM_sigma_inv = 4 * log(2.0) / (current_FWHM*current_FWHM);
		    current_Alpha =0.93943727 / current_FWHM;
		    			
			TOF_dist = ptr_TOF_dist[event_index];

			//only do the computation if current event line intersects current slice
			if (src_x<intersection_x - voxel_size_x && dest_x>intersection_x + voxel_size_x){
				dir_x = dest_x - src_x;
				dir_y = dest_y - src_y;
				dir_z = dest_z - src_z;
				//dist_squared = __fmaf_rn (dir_z,dir_z, __fmaf_rn(dir_y,dir_y,__fmul_rn(dir_x,dir_x) ) );
				dist_squared = dir_x*dir_x + dir_y*dir_y + dir_z*dir_z;
				dist = __fsqrt_rn(dist_squared);

				unit_dir_vector_y = dir_y / dist;
				unit_dir_vector_z = dir_z / dist;

				t = (intersection_x - src_x) / dir_x;
				intersection_y = t*dir_y + src_y;
				intersection_z = t*dir_z + src_z;

				if (fp_value[event_index] == 0)
					backward_ratio = 0;
				else
					backward_ratio = __frcp_rn(fp_value[event_index]);

			

				intersection_index_y = (int)ceilf(intersection_y / voxel_size_y) + center_y;
				intersection_index_z = (int)ceilf(intersection_z / voxel_size_z) + center_z;

				//compute the solid angle ratio w.r.t. the location of the intersection
				solid_angle_ratio = 1 + 2 * fabsf(t - 0.5f);
				//solid_angle_ratio = solid_angle_ratio*solid_angle_ratio*dist*dist;
				solid_angle_ratio = 1 * dist*dist;


				//compute the distance to the TOF center
				dist_to_TOF_center = (0.5f - t)*dist - TOF_dist;
				t = TOF_ALPHA *__expf(-dist_to_TOF_center*dist_to_TOF_center*TOF_INV);
				
				for (inslice_z = intersection_index_z - range2; inslice_z <= intersection_index_z + range2; inslice_z++){
					for (inslice_y = intersection_index_y - range1; inslice_y <= intersection_index_y + range1; inslice_y++){
						if (inslice_y >= 0 && inslice_y<num_y && inslice_z >= subslice_z0 && inslice_z<subslice_z1){
							//projection_vector_y = (inslice_y - center_y - 0.5)*voxel_size_y - intersection_y;
							projection_vector_y = inslice_y*voxel_size_y - H_CENTER - intersection_y;
							//projection_vector_z = (inslice_z - center_z - 0.5)*voxel_size_z - intersection_z;
							projection_vector_z = inslice_z*voxel_size_z - V_CENTER - intersection_z;
							dot_product = projection_vector_y*unit_dir_vector_y + projection_vector_z*unit_dir_vector_z;
							point_line_distance_square = projection_vector_y*projection_vector_y + projection_vector_z*projection_vector_z - dot_product*dot_product;
							weight = t * backward_ratio * __expf(-point_line_distance_square*current_FWHM_sigma_inv) * current_Alpha * GLOBAL_SCALE;
							//weight = backward_ratio * __expf(-point_line_distance_square*SIGMA_INV) * ALPHA * t;
							atomicAdd(&current_slice_image[(inslice_z - subslice_z0)*num_y + inslice_y] ,( weight/solid_angle_ratio ));
						}
					}
				}
			}//intersects current slice
		}
		__syncthreads();
		//write the current sub slice from shared memory back to global memory using atomic operation
		//Since this write is not coalesced,  think about optimization techniques
		for (int j = threadIdx.x; j<SIZE_SUB_SLICE_x; j = j + blockDim.x){
			//atomicAdd(&image_ptr[ (j+sub_slice*SIZE_SUB_SLICE)*image_width+current_slice], current_slice_image[j]);
			atomicAdd(&image_ptr[(j + sub_slice*SIZE_SUB_SLICE_x)*num_x + current_slice], current_slice_image[j]);
			current_slice_image[j] = 0.0f;
		}
		__syncthreads();
	}
}

void
__global__ _TOF_b_ratio_proj_lst_cuda_y_kernel(float* image, float* fp_y, LST_LORs *_events_y_dominant_uvm, PARAMETERS_IN_DEVICE_t* parameters_device){
	int event_index, current_slice, sub_slice;
	float src_x, src_y, src_z, dest_x, dest_y, dest_z;
	int coeff, c;
	float current_FWHM,current_Alpha,current_FWHM_sigma_inv; //for PSF
	float dir_x, dir_y, dir_z;
	float TOF_dist;
	float dist_to_TOF_center;
	float unit_dir_vector_x, unit_dir_vector_z;
	float projection_vector_x, projection_vector_z;
	float dot_product;
	float point_line_distance_square;
	float weight;
	float dist, dist_squared;
	float t;
	float solid_angle_ratio;
	float backward_ratio;
	float intersection_x, intersection_y, intersection_z;
	int intersection_index_x, intersection_index_z;
	int inslice_x, inslice_z;
	int subslice_z0, subslice_z1;
	__shared__ float H_CENTER;
	__shared__ float V_CENTER;
	__shared__ float SIGMA_INV;
	__shared__ float ALPHA;
	__shared__ float current_slice_image[SIZE_SUB_SLICE_y];
	__shared__ float voxel_size_x, voxel_size_y, voxel_size_z;
	__shared__ int center_x, center_y, center_z;
	__shared__ int range1, range2;
	__shared__ int num_x, num_y, num_z;
	__shared__ float *ptr_src_x, *ptr_src_y, *ptr_src_z, *ptr_dest_x, *ptr_dest_y, *ptr_dest_z, *ptr_TOF_dist;
	__shared__ float *ptr_FWHM;
	__shared__ int num_lines;
	__shared__ float *image_ptr;
	__shared__ float *fp_value;
	__shared__ float TOF_INV, TOF_ALPHA;
	if (threadIdx.x == 0){
		voxel_size_x = parameters_device->X_SAMP;
		voxel_size_y = parameters_device->Y_SAMP;
		voxel_size_z = parameters_device->Z_SAMP;
		center_x = parameters_device->X_INDEX_CENT;
		center_y = parameters_device->Y_INDEX_CENT;
		center_z = parameters_device->Z_INDEX_CENT;
		num_x = parameters_device->NUM_X;
		num_y = parameters_device->NUM_Y;
		num_z = parameters_device->NUM_Z;
		H_CENTER = parameters_device->H_CENTER_X;
		V_CENTER = parameters_device->V_CENTER;
		SIGMA_INV = parameters_device->FWHM_sigma_inv;
		ALPHA = parameters_device->FWHM_alpha;
		range1 = parameters_device->RANGE1;
		range2 = parameters_device->RANGE2;
		TOF_INV = parameters_device->TOF_inv;
		TOF_ALPHA = parameters_device->TOF_alpha;

		ptr_src_x = _events_y_dominant_uvm->src_x;
		ptr_src_y = _events_y_dominant_uvm->src_y;
		ptr_src_z = _events_y_dominant_uvm->src_z;
		ptr_dest_x = _events_y_dominant_uvm->dest_x;
		ptr_dest_y = _events_y_dominant_uvm->dest_y;
		ptr_dest_z = _events_y_dominant_uvm->dest_z;
		ptr_FWHM = _events_y_dominant_uvm->coeff;
		
		ptr_TOF_dist = _events_y_dominant_uvm->TOF_dist;
		num_lines = _events_y_dominant_uvm->num_lines;
		image_ptr = image;
		fp_value = fp_y;
	}

	for (int i = threadIdx.x; i<SIZE_SUB_SLICE_y; i = i + blockDim.x){
		current_slice_image[i] = 0.0f;
	}
	__syncthreads();
	c = 1;
	current_slice = blockIdx.x;
	intersection_y = __fmul_rn((current_slice - center_y - 0.5f), voxel_size_y);
	for (sub_slice = 0; sub_slice<NUM_SUB_SLICE_y; sub_slice++){
		subslice_z0 = (num_z / NUM_SUB_SLICE_y)*sub_slice;
		subslice_z1 = subslice_z0 + SUB_SLICE_HEIGHT_y;//(num_z/NUM_SUB_SLICE)*(sub_slice+1);
		//loop through all the lines and compute the backprojection to this current slice
		for (event_index = threadIdx.x; event_index< num_lines; event_index = event_index + blockDim.x){
			//get event line parameters
			src_x = ptr_src_x[event_index];
			src_y = ptr_src_y[event_index];
			src_z = ptr_src_z[event_index];
			dest_x = ptr_dest_x[event_index];
			dest_y = ptr_dest_y[event_index];
			dest_z = ptr_dest_z[event_index];
			current_FWHM = ptr_FWHM[event_index];
			current_FWHM_sigma_inv = 4 * log(2.0) / (current_FWHM*current_FWHM);
		    current_Alpha =0.93943727 / current_FWHM;
		    
			TOF_dist = ptr_TOF_dist[event_index];

			//only do the computation if current event line intersects current slice
			if (src_y<intersection_y - voxel_size_y && dest_y>intersection_y + voxel_size_y){
				dir_x = dest_x - src_x;
				dir_y = dest_y - src_y;
				dir_z = dest_z - src_z;
				//dist_squared = __fmaf_rn (dir_z,dir_z, __fmaf_rn(dir_y,dir_y,__fmul_rn(dir_x,dir_x) ) );
				dist_squared = dir_x*dir_x + dir_y*dir_y + dir_z*dir_z;
				dist = __fsqrt_rn(dist_squared);

				unit_dir_vector_x = dir_x / dist;
				unit_dir_vector_z = dir_z / dist;

				t = (intersection_y - src_y) / dir_y;
				intersection_x = t*dir_x + src_x;
				intersection_z = t*dir_z + src_z;

				if (fp_value[event_index] == 0)
					backward_ratio = 0;
				else
					backward_ratio = __frcp_rn(fp_value[event_index]);

				intersection_index_x = (int)ceilf(intersection_x / voxel_size_x) + center_x;
				intersection_index_z = (int)ceilf(intersection_z / voxel_size_z) + center_z;

				//compute the solid angle ratio w.r.t. the location of the intersection
				solid_angle_ratio = 1 + 2 * fabsf(t - 0.5f);
				//solid_angle_ratio = solid_angle_ratio*solid_angle_ratio*dist*dist;
				solid_angle_ratio = 1 * dist*dist;


				//compute the distance to the TOF center
				dist_to_TOF_center = (0.5f - t)*dist - TOF_dist;
				t = TOF_ALPHA * __expf(-dist_to_TOF_center*dist_to_TOF_center*TOF_INV);
				
				for (inslice_z = intersection_index_z - range2; inslice_z <= intersection_index_z + range2; inslice_z++){
					for (inslice_x = intersection_index_x - range1; inslice_x <= intersection_index_x + range1; inslice_x++){
						if (inslice_x >= 0 && inslice_x<num_x && inslice_z >= subslice_z0 && inslice_z<subslice_z1){
							//projection_vector_x = (inslice_x - center_x - 0.5)*voxel_size_x - intersection_x;
							projection_vector_x = inslice_x*voxel_size_x - H_CENTER - intersection_x;
							//projection_vector_z = (inslice_z - center_z - 0.5)*voxel_size_z - intersection_z;
							projection_vector_z = inslice_z*voxel_size_z - V_CENTER - intersection_z;
							dot_product = projection_vector_x*unit_dir_vector_x + projection_vector_z*unit_dir_vector_z;
							point_line_distance_square = projection_vector_x*projection_vector_x + projection_vector_z*projection_vector_z - dot_product*dot_product;
							weight = t * backward_ratio * __expf(-point_line_distance_square*current_FWHM_sigma_inv) * current_Alpha * GLOBAL_SCALE;
							//weight = backward_ratio * __expf(-point_line_distance_square*SIGMA_INV) * ALPHA * t;
							atomicAdd(&current_slice_image[(inslice_z - subslice_z0)*num_x + inslice_x], (weight/solid_angle_ratio));
						}
					}
				}
			}//intersects current slice
		}
		__syncthreads();
		//write the current sub slice from shared memory back to global memory using atomic operation
		//Since this write is not coalesced,  think about optimization techniques
		for (int j = threadIdx.x; j<SIZE_SUB_SLICE_y; j = j + blockDim.x){
			atomicAdd(&image_ptr[num_x*num_y*(subslice_z0 + j / num_x) + num_x*current_slice + j%num_x], current_slice_image[j]);
			//image_ptr[ image_width*image_height*(subslice_z0+j/image_width) + image_width*current_slice + j%image_width ] = current_slice_image[j];
			current_slice_image[j] = 0.0f;
		}
		__syncthreads();
	}
}

void
__global__ _TOF_b_ratio_proj_lst_cuda_x_kernel_atten(float* image, float* fp_x, float* atten_per_event, LST_LORs *_events_x_dominant_uvm, PARAMETERS_IN_DEVICE_t* parameters_device){
	int event_index, current_slice, sub_slice;
	float src_x, src_y, src_z, dest_x, dest_y, dest_z;
	float current_FWHM,current_Alpha,current_FWHM_sigma_inv; //for PSF
	int coeff, c;
	float dir_x, dir_y, dir_z;
	float TOF_dist;
	float dist_to_TOF_center;
	float unit_dir_vector_y, unit_dir_vector_z;
	float projection_vector_y, projection_vector_z;
	float dot_product;
	float point_line_distance_square;
	float weight;
	float dist, dist_squared;
	float t;
	float solid_angle_ratio;
	float backward_ratio, integral_mu;
	float intersection_x, intersection_y, intersection_z;
	int  intersection_index_y, intersection_index_z;
	int  inslice_y, inslice_z;
	int subslice_z0, subslice_z1;
	__shared__ float H_CENTER;
	__shared__ float V_CENTER;
	__shared__ float SIGMA_INV;
	__shared__ float ALPHA;
	__shared__ float current_slice_image[SIZE_SUB_SLICE_x];
	__shared__ float voxel_size_x, voxel_size_y, voxel_size_z;
	__shared__ int center_x, center_y, center_z;
	__shared__ int range1, range2;
	__shared__ int num_x, num_y, num_z;
	__shared__ float *ptr_src_x, *ptr_src_y, *ptr_src_z, *ptr_dest_x, *ptr_dest_y, *ptr_dest_z, *ptr_TOF_dist;
	__shared__ float *ptr_FWHM;
	__shared__ int num_lines;
	__shared__ float *image_ptr;
	__shared__ float *fp_value;
	__shared__ float *atten_factor;
	__shared__ float TOF_INV, TOF_ALPHA;
	if (threadIdx.x == 0){
		voxel_size_x = parameters_device->X_SAMP;
		voxel_size_y = parameters_device->Y_SAMP;
		voxel_size_z = parameters_device->Z_SAMP;
		center_x = parameters_device->X_INDEX_CENT;
		center_y = parameters_device->Y_INDEX_CENT;
		center_z = parameters_device->Z_INDEX_CENT;
		num_x = parameters_device->NUM_X;
		num_y = parameters_device->NUM_Y;
		num_z = parameters_device->NUM_Z;
		H_CENTER = parameters_device->H_CENTER_Y;
		V_CENTER = parameters_device->V_CENTER;
		SIGMA_INV = parameters_device->FWHM_sigma_inv;
		ALPHA = parameters_device->FWHM_alpha;
		range1 = parameters_device->RANGE1;
		range2 = parameters_device->RANGE2;
		TOF_INV = parameters_device->TOF_inv;
		TOF_ALPHA = parameters_device->TOF_alpha;

		ptr_src_x = _events_x_dominant_uvm->src_x;
		ptr_src_y = _events_x_dominant_uvm->src_y;
		ptr_src_z = _events_x_dominant_uvm->src_z;
		ptr_dest_x = _events_x_dominant_uvm->dest_x;
		ptr_dest_y = _events_x_dominant_uvm->dest_y;
		ptr_dest_z = _events_x_dominant_uvm->dest_z;
		ptr_TOF_dist = _events_x_dominant_uvm->TOF_dist;
		ptr_FWHM = _events_x_dominant_uvm->coeff;

		num_lines = _events_x_dominant_uvm->num_lines;
		image_ptr = image;
		fp_value = fp_x;
		atten_factor = atten_per_event;
	}

	//initialize shared memory to zero
	for (int i = threadIdx.x; i<SIZE_SUB_SLICE_x; i = i + blockDim.x){
		current_slice_image[i] = 0.0f;
	}
	__syncthreads();
	c = 1;
	current_slice = blockIdx.x;
	intersection_x = __fmul_rn((current_slice - center_x - 0.5f), voxel_size_x);
	for (sub_slice = 0; sub_slice<NUM_SUB_SLICE_x; sub_slice++){
		subslice_z0 = (num_z / NUM_SUB_SLICE_x)*sub_slice;
		subslice_z1 = subslice_z0 + SUB_SLICE_HEIGHT_x;//(num_z/NUM_SUB_SLICE)*(sub_slice+1);
		//loop through all the lines and compute the backprojection to this current slice
		for (event_index = threadIdx.x; event_index< num_lines; event_index = event_index + blockDim.x){
			//get event line parameters
			src_x = ptr_src_x[event_index];
			src_y = ptr_src_y[event_index];
			src_z = ptr_src_z[event_index];
			dest_x = ptr_dest_x[event_index];
			dest_y = ptr_dest_y[event_index];
			dest_z = ptr_dest_z[event_index];
			current_FWHM = ptr_FWHM[event_index];
			current_FWHM_sigma_inv = 4 * log(2.0) / (current_FWHM*current_FWHM);
		    current_Alpha =0.93943727 / current_FWHM;
			TOF_dist = ptr_TOF_dist[event_index];

			//only do the computation if current event line intersects current slice
			if (src_x<intersection_x - voxel_size_x && dest_x>intersection_x + voxel_size_x){
				dir_x = dest_x - src_x;
				dir_y = dest_y - src_y;
				dir_z = dest_z - src_z;
				//dist_squared = __fmaf_rn (dir_z,dir_z, __fmaf_rn(dir_y,dir_y,__fmul_rn(dir_x,dir_x) ) );
				dist_squared = dir_x*dir_x + dir_y*dir_y + dir_z*dir_z;
				dist = __fsqrt_rn(dist_squared);

				unit_dir_vector_y = dir_y / dist;
				unit_dir_vector_z = dir_z / dist;

				t = (intersection_x - src_x) / dir_x;
				intersection_y = t*dir_y + src_y;
				intersection_z = t*dir_z + src_z;

				if (fp_value[event_index] == 0)
					backward_ratio = 0;
				else
					backward_ratio = __frcp_rn(fp_value[event_index]);

				integral_mu = atten_factor[event_index];



				intersection_index_y = (int)ceilf(intersection_y / voxel_size_y) + center_y;
				intersection_index_z = (int)ceilf(intersection_z / voxel_size_z) + center_z;

				//compute the solid angle ratio w.r.t. the location of the intersection
				solid_angle_ratio = 1 + 2 * fabsf(t - 0.5f);
				//solid_angle_ratio = solid_angle_ratio*solid_angle_ratio;
				solid_angle_ratio = 1 * dist*dist;


				//compute the distance to the TOF center
				dist_to_TOF_center = (0.5f - t)*dist - TOF_dist;
				t = TOF_ALPHA * __expf(-dist_to_TOF_center*dist_to_TOF_center*TOF_INV);

				for (inslice_z = intersection_index_z - range2; inslice_z <= intersection_index_z + range2; inslice_z++){
					for (inslice_y = intersection_index_y - range1; inslice_y <= intersection_index_y + range1; inslice_y++){
						if (inslice_y >= 0 && inslice_y<num_y && inslice_z >= subslice_z0 && inslice_z<subslice_z1){
							//projection_vector_y = (inslice_y - center_y - 0.5)*voxel_size_y - intersection_y;
							projection_vector_y = inslice_y*voxel_size_y - H_CENTER - intersection_y;
							//projection_vector_z = (inslice_z - center_z - 0.5)*voxel_size_z - intersection_z;
							projection_vector_z = inslice_z*voxel_size_z - V_CENTER - intersection_z;
							dot_product = projection_vector_y*unit_dir_vector_y + projection_vector_z*unit_dir_vector_z;
							point_line_distance_square = projection_vector_y*projection_vector_y + projection_vector_z*projection_vector_z - dot_product*dot_product;
							weight = t * backward_ratio * __expf(-point_line_distance_square*current_FWHM_sigma_inv - integral_mu) * current_Alpha * GLOBAL_SCALE;
							atomicAdd(&current_slice_image[(inslice_z - subslice_z0)*num_y + inslice_y],(weight / solid_angle_ratio));
						}
					}
				}
			}//intersects current slice
		}
		__syncthreads();
		//write the current sub slice from shared memory back to global memory using atomic operation
		//Since this write is not coalesced,  think about optimization techniques
		for (int j = threadIdx.x; j<SIZE_SUB_SLICE_x; j = j + blockDim.x){
			//atomicAdd(&image_ptr[ (j+sub_slice*SIZE_SUB_SLICE)*image_width+current_slice], current_slice_image[j]);
			atomicAdd(&image_ptr[(j + sub_slice*SIZE_SUB_SLICE_x)*num_x + current_slice], current_slice_image[j]);
			current_slice_image[j] = 0.0f;
		}
		__syncthreads();
	}
}

void
__global__ _TOF_b_ratio_proj_lst_cuda_y_kernel_atten(float* image, float* fp_y, float* atten_per_event, LST_LORs *_events_y_dominant_uvm, PARAMETERS_IN_DEVICE_t* parameters_device){
	int event_index, current_slice, sub_slice;
	float src_x, src_y, src_z, dest_x, dest_y, dest_z;
	int coeff, c;
	float dir_x, dir_y, dir_z;
	float TOF_dist;
	float dist_to_TOF_center;
	float unit_dir_vector_x, unit_dir_vector_z;
	float projection_vector_x, projection_vector_z;
	float dot_product;
	float point_line_distance_square;
	float weight;
	float dist, dist_squared;
	float t;
	float solid_angle_ratio;
	float backward_ratio,integral_mu;
	float intersection_x, intersection_y, intersection_z;
	int intersection_index_x, intersection_index_z;
	int inslice_x, inslice_z;
	int subslice_z0, subslice_z1;
	__shared__ float H_CENTER;
	__shared__ float V_CENTER;
	__shared__ float SIGMA_INV;
	__shared__ float ALPHA;
	__shared__ float current_slice_image[SIZE_SUB_SLICE_y];
	__shared__ float voxel_size_x, voxel_size_y, voxel_size_z;
	__shared__ int center_x, center_y, center_z;
	__shared__ int range1, range2;
	__shared__ int num_x, num_y, num_z;
	__shared__ float *ptr_src_x, *ptr_src_y, *ptr_src_z, *ptr_dest_x, *ptr_dest_y, *ptr_dest_z, *ptr_TOF_dist;
	__shared__ float *ptr_FWHM;
	__shared__ int num_lines;
	__shared__ float *image_ptr;
	__shared__ float *fp_value, *atten_factor;
	__shared__ float TOF_INV, TOF_ALPHA;
	if (threadIdx.x == 0){
		voxel_size_x = parameters_device->X_SAMP;
		voxel_size_y = parameters_device->Y_SAMP;
		voxel_size_z = parameters_device->Z_SAMP;
		center_x = parameters_device->X_INDEX_CENT;
		center_y = parameters_device->Y_INDEX_CENT;
		center_z = parameters_device->Z_INDEX_CENT;
		num_x = parameters_device->NUM_X;
		num_y = parameters_device->NUM_Y;
		num_z = parameters_device->NUM_Z;
		H_CENTER = parameters_device->H_CENTER_X;
		V_CENTER = parameters_device->V_CENTER;
		SIGMA_INV = parameters_device->FWHM_sigma_inv;
		ALPHA = parameters_device->FWHM_alpha;
		range1 = parameters_device->RANGE1;
		range2 = parameters_device->RANGE2;
		TOF_INV = parameters_device->TOF_inv;
		TOF_ALPHA = parameters_device->TOF_alpha;

		ptr_src_x = _events_y_dominant_uvm->src_x;
		ptr_src_y = _events_y_dominant_uvm->src_y;
		ptr_src_z = _events_y_dominant_uvm->src_z;
		ptr_dest_x = _events_y_dominant_uvm->dest_x;
		ptr_dest_y = _events_y_dominant_uvm->dest_y;
		ptr_dest_z = _events_y_dominant_uvm->dest_z;
		ptr_FWHM = _events_y_dominant_uvm->coeff;

		ptr_TOF_dist = _events_y_dominant_uvm->TOF_dist;
		num_lines = _events_y_dominant_uvm->num_lines;
		image_ptr = image;
		fp_value = fp_y;
		atten_factor = atten_per_event;
	}

	for (int i = threadIdx.x; i<SIZE_SUB_SLICE_y; i = i + blockDim.x){
		current_slice_image[i] = 0.0f;
	}
	__syncthreads();
	c = 1;
	current_slice = blockIdx.x;
	intersection_y = __fmul_rn((current_slice - center_y - 0.5f), voxel_size_y);
	for (sub_slice = 0; sub_slice<NUM_SUB_SLICE_y; sub_slice++){
		subslice_z0 = (num_z / NUM_SUB_SLICE_y)*sub_slice;
		subslice_z1 = subslice_z0 + SUB_SLICE_HEIGHT_y;//(num_z/NUM_SUB_SLICE)*(sub_slice+1);
		//loop through all the lines and compute the backprojection to this current slice
		for (event_index = threadIdx.x; event_index< num_lines; event_index = event_index + blockDim.x){
			//get event line parameters
			src_x = ptr_src_x[event_index];
			src_y = ptr_src_y[event_index];
			src_z = ptr_src_z[event_index];
			dest_x = ptr_dest_x[event_index];
			dest_y = ptr_dest_y[event_index];
			dest_z = ptr_dest_z[event_index];
			coeff = ptr_FWHM[event_index];

			TOF_dist = ptr_TOF_dist[event_index];

			//only do the computation if current event line intersects current slice
			if (src_y<intersection_y - voxel_size_y && dest_y>intersection_y + voxel_size_y){
				dir_x = dest_x - src_x;
				dir_y = dest_y - src_y;
				dir_z = dest_z - src_z;
				//dist_squared = __fmaf_rn (dir_z,dir_z, __fmaf_rn(dir_y,dir_y,__fmul_rn(dir_x,dir_x) ) );
				dist_squared = dir_x*dir_x + dir_y*dir_y + dir_z*dir_z;
				dist = __fsqrt_rn(dist_squared);

				unit_dir_vector_x = dir_x / dist;
				unit_dir_vector_z = dir_z / dist;

				t = (intersection_y - src_y) / dir_y;
				intersection_x = t*dir_x + src_x;
				intersection_z = t*dir_z + src_z;

				if (fp_value[event_index] == 0)
					backward_ratio = 0;
				else
					backward_ratio = __frcp_rn(fp_value[event_index]);
				integral_mu = atten_factor[event_index];

				intersection_index_x = (int)ceilf(intersection_x / voxel_size_x) + center_x;
				intersection_index_z = (int)ceilf(intersection_z / voxel_size_z) + center_z;

				//compute the solid angle ratio w.r.t. the location of the intersection
				solid_angle_ratio = 1 + 2 * fabsf(t - 0.5f);
				//solid_angle_ratio = solid_angle_ratio*solid_angle_ratio;
				solid_angle_ratio = 1 * dist*dist;


				//compute the distance to the TOF center
				dist_to_TOF_center = (0.5f - t)*dist - TOF_dist;
				t = TOF_ALPHA * __expf(-dist_to_TOF_center*dist_to_TOF_center*TOF_INV);

				for (inslice_z = intersection_index_z - range2; inslice_z <= intersection_index_z + range2; inslice_z++){
					for (inslice_x = intersection_index_x - range1; inslice_x <= intersection_index_x + range1; inslice_x++){
						if (inslice_x >= 0 && inslice_x<num_x && inslice_z >= subslice_z0 && inslice_z<subslice_z1){
							//projection_vector_x = (inslice_x - center_x - 0.5)*voxel_size_x - intersection_x;
							projection_vector_x = inslice_x*voxel_size_x - H_CENTER - intersection_x;
							//projection_vector_z = (inslice_z - center_z - 0.5)*voxel_size_z - intersection_z;
							projection_vector_z = inslice_z*voxel_size_z - V_CENTER - intersection_z;
							dot_product = projection_vector_x*unit_dir_vector_x + projection_vector_z*unit_dir_vector_z;
							point_line_distance_square = projection_vector_x*projection_vector_x + projection_vector_z*projection_vector_z - dot_product*dot_product;
							weight = t * backward_ratio * __expf(-point_line_distance_square*SIGMA_INV - integral_mu) * ALPHA * GLOBAL_SCALE;
							atomicAdd(&current_slice_image[(inslice_z - subslice_z0)*num_x + inslice_x], (weight/ solid_angle_ratio));
						}
					}
				}
			}//intersects current slice
		}
		__syncthreads();
		//write the current sub slice from shared memory back to global memory using atomic operation
		//Since this write is not coalesced,  think about optimization techniques
		for (int j = threadIdx.x; j<SIZE_SUB_SLICE_y; j = j + blockDim.x){
			atomicAdd(&image_ptr[num_x*num_y*(subslice_z0 + j / num_x) + num_x*current_slice + j%num_x], current_slice_image[j]);
			//image_ptr[ image_width*image_height*(subslice_z0+j/image_width) + image_width*current_slice + j%image_width ] = current_slice_image[j];
			current_slice_image[j] = 0.0f;
		}
		__syncthreads();
	}
}


void
__global__ _TOF_b_ratio_proj_lst_cuda_x_kernel_scatter(float* image, float* fp_x, LST_LORs_scatter *_events_x_dominant_uvm, PARAMETERS_IN_DEVICE_t* parameters_device){
	int event_index, current_slice, sub_slice;
	float src_x, src_y, src_z, dest_x, dest_y, dest_z;
	int coeff, c;
	float sc_coeff;
	float dir_x, dir_y, dir_z;
	float TOF_dist;
	float dist_to_TOF_center;
	float unit_dir_vector_y, unit_dir_vector_z;
	float projection_vector_y, projection_vector_z;
	float dot_product;
	float point_line_distance_square;
	float weight;
	float dist, dist_squared;
	float t;
	float solid_angle_ratio;
	float backward_ratio;
	float denominator;
	float intersection_x, intersection_y, intersection_z;
	int  intersection_index_y, intersection_index_z;
	int  inslice_y, inslice_z;
	int subslice_z0, subslice_z1;
	__shared__ float H_CENTER;
	__shared__ float V_CENTER;
	__shared__ float SIGMA_INV;
	__shared__ float ALPHA;
	__shared__ float current_slice_image[SIZE_SUB_SLICE_x];
	__shared__ float voxel_size_x, voxel_size_y, voxel_size_z;
	__shared__ int center_x, center_y, center_z;
	__shared__ int range1, range2;
	__shared__ int num_x, num_y, num_z;
	__shared__ float *ptr_src_x, *ptr_src_y, *ptr_src_z, *ptr_dest_x, *ptr_dest_y, *ptr_dest_z, *ptr_TOF_dist;
	__shared__ float *ptr_FWHM;
	__shared__ int num_lines;
	__shared__ float *image_ptr;
	__shared__ float *fp_value;
	__shared__ float *ptr_scatter_coeff;
	__shared__ float TOF_INV, TOF_ALPHA;
	if (threadIdx.x == 0){
		voxel_size_x = parameters_device->X_SAMP;
		voxel_size_y = parameters_device->Y_SAMP;
		voxel_size_z = parameters_device->Z_SAMP;
		center_x = parameters_device->X_INDEX_CENT;
		center_y = parameters_device->Y_INDEX_CENT;
		center_z = parameters_device->Z_INDEX_CENT;
		num_x = parameters_device->NUM_X;
		num_y = parameters_device->NUM_Y;
		num_z = parameters_device->NUM_Z;
		H_CENTER = parameters_device->H_CENTER_Y;
		V_CENTER = parameters_device->V_CENTER;
		SIGMA_INV = parameters_device->FWHM_sigma_inv;
		ALPHA = parameters_device->FWHM_alpha;
		range1 = parameters_device->RANGE1;
		range2 = parameters_device->RANGE2;
		TOF_INV = parameters_device->TOF_inv;
		TOF_ALPHA = parameters_device->TOF_alpha;

		ptr_src_x = _events_x_dominant_uvm->src_x;
		ptr_src_y = _events_x_dominant_uvm->src_y;
		ptr_src_z = _events_x_dominant_uvm->src_z;
		ptr_dest_x = _events_x_dominant_uvm->dest_x;
		ptr_dest_y = _events_x_dominant_uvm->dest_y;
		ptr_dest_z = _events_x_dominant_uvm->dest_z;
		ptr_TOF_dist = _events_x_dominant_uvm->TOF_dist;
		ptr_FWHM = _events_x_dominant_uvm->coeff;
		ptr_scatter_coeff = _events_x_dominant_uvm->sc_coeff;

		num_lines = _events_x_dominant_uvm->num_lines;
		image_ptr = image;
		fp_value = fp_x;
	}

	//initialize shared memory to zero
	for (int i = threadIdx.x; i<SIZE_SUB_SLICE_x; i = i + blockDim.x){
		current_slice_image[i] = 0.0f;
	}
	__syncthreads();
	c = 1;
	current_slice = blockIdx.x;
	intersection_x = __fmul_rn((current_slice - center_x - 0.5f), voxel_size_x);
	for (sub_slice = 0; sub_slice<NUM_SUB_SLICE_x; sub_slice++){
		subslice_z0 = (num_z / NUM_SUB_SLICE_x)*sub_slice;
		subslice_z1 = subslice_z0 + SUB_SLICE_HEIGHT_x;//(num_z/NUM_SUB_SLICE)*(sub_slice+1);
		//loop through all the lines and compute the backprojection to this current slice
		for (event_index = threadIdx.x; event_index< num_lines; event_index = event_index + blockDim.x){
			//get event line parameters
			src_x = ptr_src_x[event_index];
			src_y = ptr_src_y[event_index];
			src_z = ptr_src_z[event_index];
			dest_x = ptr_dest_x[event_index];
			dest_y = ptr_dest_y[event_index];
			dest_z = ptr_dest_z[event_index];
			coeff = ptr_FWHM[event_index];
			sc_coeff = ptr_scatter_coeff[event_index];

			TOF_dist = ptr_TOF_dist[event_index];

			//only do the computation if current event line intersects current slice
			if (src_x<intersection_x - voxel_size_x && dest_x>intersection_x + voxel_size_x){
				dir_x = dest_x - src_x;
				dir_y = dest_y - src_y;
				dir_z = dest_z - src_z;
				//dist_squared = __fmaf_rn (dir_z,dir_z, __fmaf_rn(dir_y,dir_y,__fmul_rn(dir_x,dir_x) ) );
				dist_squared = dir_x*dir_x + dir_y*dir_y + dir_z*dir_z;
				dist = __fsqrt_rn(dist_squared);

				unit_dir_vector_y = dir_y / dist;
				unit_dir_vector_z = dir_z / dist;

				t = (intersection_x - src_x) / dir_x;
				intersection_y = t*dir_y + src_y;
				intersection_z = t*dir_z + src_z;

				denominator = fp_value[event_index] + ptr_scatter_coeff[event_index];

				if (denominator == 0)
					backward_ratio = 0;
				else
					backward_ratio = __frcp_rn(denominator);
				


				intersection_index_y = (int)ceilf(intersection_y / voxel_size_y) + center_y;
				intersection_index_z = (int)ceilf(intersection_z / voxel_size_z) + center_z;

				//compute the solid angle ratio w.r.t. the location of the intersection
				solid_angle_ratio = 1 + 2 * fabsf(t - 0.5f);      
				//solid_angle_ratio = solid_angle_ratio*solid_angle_ratio;
				solid_angle_ratio = 1 * dist*dist;


				//compute the distance to the TOF center
				dist_to_TOF_center = (0.5f - t)*dist - TOF_dist;
				t = TOF_ALPHA * __expf(-dist_to_TOF_center*dist_to_TOF_center*TOF_INV);

				for (inslice_z = intersection_index_z - range2; inslice_z <= intersection_index_z + range2; inslice_z++){
					for (inslice_y = intersection_index_y - range1; inslice_y <= intersection_index_y + range1; inslice_y++){
						if (inslice_y >= 0 && inslice_y<num_y && inslice_z >= subslice_z0 && inslice_z<subslice_z1){
							//projection_vector_y = (inslice_y - center_y - 0.5)*voxel_size_y - intersection_y;
							projection_vector_y = inslice_y*voxel_size_y - H_CENTER - intersection_y;
							//projection_vector_z = (inslice_z - center_z - 0.5)*voxel_size_z - intersection_z;
							projection_vector_z = inslice_z*voxel_size_z - V_CENTER - intersection_z;
							dot_product = projection_vector_y*unit_dir_vector_y + projection_vector_z*unit_dir_vector_z;
							point_line_distance_square = projection_vector_y*projection_vector_y + projection_vector_z*projection_vector_z - dot_product*dot_product;
							weight = t * backward_ratio * __expf(-point_line_distance_square*SIGMA_INV) * ALPHA * GLOBAL_SCALE;
							atomicAdd(&current_slice_image[(inslice_z - subslice_z0)*num_y + inslice_y], (weight) / solid_angle_ratio);
						}
					}
				}
			}//intersects current slice
		}
		__syncthreads();
		//write the current sub slice from shared memory back to global memory using atomic operation
		//Since this write is not coalesced,  think about optimization techniques
		for (int j = threadIdx.x; j<SIZE_SUB_SLICE_x; j = j + blockDim.x){
			//atomicAdd(&image_ptr[ (j+sub_slice*SIZE_SUB_SLICE)*image_width+current_slice], current_slice_image[j]);
			atomicAdd(&image_ptr[(j + sub_slice*SIZE_SUB_SLICE_x)*num_x + current_slice], current_slice_image[j]);
			current_slice_image[j] = 0.0f;
		}
		__syncthreads();
	}
}

void
__global__ _TOF_b_ratio_proj_lst_cuda_y_kernel_scatter(float* image, float* fp_y, LST_LORs_scatter *_events_y_dominant_uvm, PARAMETERS_IN_DEVICE_t* parameters_device){
	int event_index, current_slice, sub_slice;
	float src_x, src_y, src_z, dest_x, dest_y, dest_z;
	int coeff, c;
	float sc_coeff;
	float dir_x, dir_y, dir_z;
	float TOF_dist;
	float dist_to_TOF_center;
	float unit_dir_vector_x, unit_dir_vector_z;
	float projection_vector_x, projection_vector_z;
	float dot_product;
	float point_line_distance_square;
	float weight;
	float dist, dist_squared;
	float t;
	float solid_angle_ratio;
	float backward_ratio;
	float denominator;
	float intersection_x, intersection_y, intersection_z;
	int intersection_index_x, intersection_index_z;
	int inslice_x, inslice_z;
	int subslice_z0, subslice_z1;
	__shared__ float H_CENTER;
	__shared__ float V_CENTER;
	__shared__ float SIGMA_INV;
	__shared__ float ALPHA;
	__shared__ float current_slice_image[SIZE_SUB_SLICE_y];
	__shared__ float voxel_size_x, voxel_size_y, voxel_size_z;
	__shared__ int center_x, center_y, center_z;
	__shared__ int range1, range2;
	__shared__ int num_x, num_y, num_z;
	__shared__ float *ptr_src_x, *ptr_src_y, *ptr_src_z, *ptr_dest_x, *ptr_dest_y, *ptr_dest_z, *ptr_TOF_dist;
	__shared__ float *ptr_FWHM;
	__shared__ int num_lines;
	__shared__ float *image_ptr;
	__shared__ float *fp_value;
	__shared__ float *ptr_scatter_coeff;
	__shared__ float TOF_INV, TOF_ALPHA;
	if (threadIdx.x == 0){
		voxel_size_x = parameters_device->X_SAMP;
		voxel_size_y = parameters_device->Y_SAMP;
		voxel_size_z = parameters_device->Z_SAMP;
		center_x = parameters_device->X_INDEX_CENT;
		center_y = parameters_device->Y_INDEX_CENT;
		center_z = parameters_device->Z_INDEX_CENT;
		num_x = parameters_device->NUM_X;
		num_y = parameters_device->NUM_Y;
		num_z = parameters_device->NUM_Z;
		H_CENTER = parameters_device->H_CENTER_X;
		V_CENTER = parameters_device->V_CENTER;
		SIGMA_INV = parameters_device->FWHM_sigma_inv;
		ALPHA = parameters_device->FWHM_alpha;
		range1 = parameters_device->RANGE1;
		range2 = parameters_device->RANGE2;
		TOF_INV = parameters_device->TOF_inv;
		TOF_ALPHA = parameters_device->TOF_alpha;

		ptr_src_x = _events_y_dominant_uvm->src_x;
		ptr_src_y = _events_y_dominant_uvm->src_y;
		ptr_src_z = _events_y_dominant_uvm->src_z;
		ptr_dest_x = _events_y_dominant_uvm->dest_x;
		ptr_dest_y = _events_y_dominant_uvm->dest_y;
		ptr_dest_z = _events_y_dominant_uvm->dest_z;
		ptr_FWHM = _events_y_dominant_uvm->coeff;
		ptr_scatter_coeff = _events_y_dominant_uvm->sc_coeff;

		ptr_TOF_dist = _events_y_dominant_uvm->TOF_dist;
		num_lines = _events_y_dominant_uvm->num_lines;
		image_ptr = image;
		fp_value = fp_y;
	}

	for (int i = threadIdx.x; i<SIZE_SUB_SLICE_y; i = i + blockDim.x){
		current_slice_image[i] = 0.0f;
	}
	__syncthreads();
	c = 1;
	current_slice = blockIdx.x;
	intersection_y = __fmul_rn((current_slice - center_y - 0.5f), voxel_size_y);
	for (sub_slice = 0; sub_slice<NUM_SUB_SLICE_y; sub_slice++){
		subslice_z0 = (num_z / NUM_SUB_SLICE_y)*sub_slice;
		subslice_z1 = subslice_z0 + SUB_SLICE_HEIGHT_y;//(num_z/NUM_SUB_SLICE)*(sub_slice+1);
		//loop through all the lines and compute the backprojection to this current slice
		for (event_index = threadIdx.x; event_index< num_lines; event_index = event_index + blockDim.x){
			//get event line parameters
			src_x = ptr_src_x[event_index];
			src_y = ptr_src_y[event_index];
			src_z = ptr_src_z[event_index];
			dest_x = ptr_dest_x[event_index];
			dest_y = ptr_dest_y[event_index];
			dest_z = ptr_dest_z[event_index];
			coeff = ptr_FWHM[event_index];
			sc_coeff = ptr_scatter_coeff[event_index];
			TOF_dist = ptr_TOF_dist[event_index];

			//only do the computation if current event line intersects current slice
			if (src_y<intersection_y - voxel_size_y && dest_y>intersection_y + voxel_size_y){
				dir_x = dest_x - src_x;
				dir_y = dest_y - src_y;
				dir_z = dest_z - src_z;
				//dist_squared = __fmaf_rn (dir_z,dir_z, __fmaf_rn(dir_y,dir_y,__fmul_rn(dir_x,dir_x) ) );
				dist_squared = dir_x*dir_x + dir_y*dir_y + dir_z*dir_z;
				dist = __fsqrt_rn(dist_squared);

				unit_dir_vector_x = dir_x / dist;
				unit_dir_vector_z = dir_z / dist;

				t = (intersection_y - src_y) / dir_y;
				intersection_x = t*dir_x + src_x;
				intersection_z = t*dir_z + src_z;

				
				denominator = fp_value[event_index];
				if (denominator == 0)
					backward_ratio = 0;
				else
					backward_ratio = __frcp_rn(denominator) + ptr_scatter_coeff[event_index];
				

				intersection_index_x = (int)ceilf(intersection_x / voxel_size_x) + center_x;
				intersection_index_z = (int)ceilf(intersection_z / voxel_size_z) + center_z;

				//compute the solid angle ratio w.r.t. the location of the intersection
				solid_angle_ratio = 1 + 2 * fabsf(t - 0.5f);
				//solid_angle_ratio = solid_angle_ratio*solid_angle_ratio;
				solid_angle_ratio = 1 * dist*dist;


				//compute the distance to the TOF center
				dist_to_TOF_center = (0.5f - t)*dist - TOF_dist;
				t = TOF_ALPHA * __expf(-dist_to_TOF_center*dist_to_TOF_center*TOF_INV);

				for (inslice_z = intersection_index_z - range2; inslice_z <= intersection_index_z + range2; inslice_z++){
					for (inslice_x = intersection_index_x - range1; inslice_x <= intersection_index_x + range1; inslice_x++){
						if (inslice_x >= 0 && inslice_x<num_x && inslice_z >= subslice_z0 && inslice_z<subslice_z1){
							//projection_vector_x = (inslice_x - center_x - 0.5)*voxel_size_x - intersection_x;
							projection_vector_x = inslice_x*voxel_size_x - H_CENTER - intersection_x;
							//projection_vector_z = (inslice_z - center_z - 0.5)*voxel_size_z - intersection_z;
							projection_vector_z = inslice_z*voxel_size_z - V_CENTER - intersection_z;
							dot_product = projection_vector_x*unit_dir_vector_x + projection_vector_z*unit_dir_vector_z;
							point_line_distance_square = projection_vector_x*projection_vector_x + projection_vector_z*projection_vector_z - dot_product*dot_product;
							weight = t * backward_ratio * __expf(-point_line_distance_square*SIGMA_INV) * ALPHA * GLOBAL_SCALE;
							atomicAdd(&current_slice_image[(inslice_z - subslice_z0)*num_x + inslice_x], (weight) / solid_angle_ratio);
						}
					}
				}
			}//intersects current slice
		}
		__syncthreads();
		//write the current sub slice from shared memory back to global memory using atomic operation
		//Since this write is not coalesced,  think about optimization techniques
		for (int j = threadIdx.x; j<SIZE_SUB_SLICE_y; j = j + blockDim.x){
			atomicAdd(&image_ptr[num_x*num_y*(subslice_z0 + j / num_x) + num_x*current_slice + j%num_x], current_slice_image[j]);
			//image_ptr[ image_width*image_height*(subslice_z0+j/image_width) + image_width*current_slice + j%image_width ] = current_slice_image[j];
			current_slice_image[j] = 0.0f;
		}
		__syncthreads();
	}
}

void
__global__ _TOF_b_ratio_proj_lst_cuda_x_kernel_scatter_atten(float* image, float* fp_x, float* atten_per_event,LST_LORs_scatter *_events_x_dominant_uvm, PARAMETERS_IN_DEVICE_t* parameters_device){
	int event_index, current_slice, sub_slice;
	float src_x, src_y, src_z, dest_x, dest_y, dest_z;
	int coeff, c;
	float sc_coeff;
	float dir_x, dir_y, dir_z;
	float TOF_dist;
	float dist_to_TOF_center;
	float unit_dir_vector_y, unit_dir_vector_z;
	float projection_vector_y, projection_vector_z;
	float dot_product;
	float point_line_distance_square;
	float weight;
	float dist, dist_squared;
	float t, integral_mu;
	float solid_angle_ratio;
	float backward_ratio;
	float intersection_x, intersection_y, intersection_z;
	int  intersection_index_y, intersection_index_z;
	int  inslice_y, inslice_z;
	int subslice_z0, subslice_z1;
	__shared__ float H_CENTER;
	__shared__ float V_CENTER;
	__shared__ float SIGMA_INV;
	__shared__ float ALPHA;
	__shared__ float current_slice_image[SIZE_SUB_SLICE_x];
	__shared__ float voxel_size_x, voxel_size_y, voxel_size_z;
	__shared__ int center_x, center_y, center_z;
	__shared__ int range1, range2;
	__shared__ int num_x, num_y, num_z;
	__shared__ float *ptr_src_x, *ptr_src_y, *ptr_src_z, *ptr_dest_x, *ptr_dest_y, *ptr_dest_z, *ptr_TOF_dist;
	__shared__ float *ptr_FWHM;
	__shared__ int num_lines;
	__shared__ float *image_ptr;
	__shared__ float *fp_value;
	__shared__ float *atten_factor;
	__shared__ float *ptr_sc_coeff;
	__shared__ float TOF_INV, TOF_ALPHA;
	if (threadIdx.x == 0){
		voxel_size_x = parameters_device->X_SAMP;
		voxel_size_y = parameters_device->Y_SAMP;
		voxel_size_z = parameters_device->Z_SAMP;
		center_x = parameters_device->X_INDEX_CENT;
		center_y = parameters_device->Y_INDEX_CENT;
		center_z = parameters_device->Z_INDEX_CENT;
		num_x = parameters_device->NUM_X;
		num_y = parameters_device->NUM_Y;
		num_z = parameters_device->NUM_Z;
		H_CENTER = parameters_device->H_CENTER_Y;
		V_CENTER = parameters_device->V_CENTER;
		SIGMA_INV = parameters_device->FWHM_sigma_inv;
		ALPHA = parameters_device->FWHM_alpha;
		range1 = parameters_device->RANGE1;
		range2 = parameters_device->RANGE2;
		TOF_INV = parameters_device->TOF_inv;
		TOF_ALPHA = parameters_device->TOF_alpha;

		ptr_src_x = _events_x_dominant_uvm->src_x;
		ptr_src_y = _events_x_dominant_uvm->src_y;
		ptr_src_z = _events_x_dominant_uvm->src_z;
		ptr_dest_x = _events_x_dominant_uvm->dest_x;
		ptr_dest_y = _events_x_dominant_uvm->dest_y;
		ptr_dest_z = _events_x_dominant_uvm->dest_z;
		ptr_TOF_dist = _events_x_dominant_uvm->TOF_dist;
		ptr_FWHM = _events_x_dominant_uvm->coeff;
		ptr_sc_coeff = _events_x_dominant_uvm->sc_coeff;

		num_lines = _events_x_dominant_uvm->num_lines;
		image_ptr = image;
		fp_value = fp_x;
		atten_factor = atten_per_event;
	}

	//initialize shared memory to zero
	for (int i = threadIdx.x; i<SIZE_SUB_SLICE_x; i = i + blockDim.x){
		current_slice_image[i] = 0.0f;
	}
	__syncthreads();
	c = 1;
	current_slice = blockIdx.x;
	intersection_x = __fmul_rn((current_slice - center_x - 0.5f), voxel_size_x);
	for (sub_slice = 0; sub_slice<NUM_SUB_SLICE_x; sub_slice++){
		subslice_z0 = (num_z / NUM_SUB_SLICE_x)*sub_slice;
		subslice_z1 = subslice_z0 + SUB_SLICE_HEIGHT_x;//(num_z/NUM_SUB_SLICE)*(sub_slice+1);
		//loop through all the lines and compute the backprojection to this current slice
		for (event_index = threadIdx.x; event_index< num_lines; event_index = event_index + blockDim.x){
			//get event line parameters
			src_x = ptr_src_x[event_index];
			src_y = ptr_src_y[event_index];
			src_z = ptr_src_z[event_index];
			dest_x = ptr_dest_x[event_index];
			dest_y = ptr_dest_y[event_index];
			dest_z = ptr_dest_z[event_index];
			coeff = ptr_FWHM[event_index];
			sc_coeff = ptr_sc_coeff[event_index];

			TOF_dist = ptr_TOF_dist[event_index];

			//only do the computation if current event line intersects current slice
			if (src_x<intersection_x - voxel_size_x && dest_x>intersection_x + voxel_size_x){
				dir_x = dest_x - src_x;
				dir_y = dest_y - src_y;
				dir_z = dest_z - src_z;
				//dist_squared = __fmaf_rn (dir_z,dir_z, __fmaf_rn(dir_y,dir_y,__fmul_rn(dir_x,dir_x) ) );
				dist_squared = dir_x*dir_x + dir_y*dir_y + dir_z*dir_z;
				dist = __fsqrt_rn(dist_squared);

				unit_dir_vector_y = dir_y / dist;
				unit_dir_vector_z = dir_z / dist;

				t = (intersection_x - src_x) / dir_x;
				intersection_y = t*dir_y + src_y;
				intersection_z = t*dir_z + src_z;

				integral_mu = atten_factor[event_index];

				if (fp_value[event_index] == 0)
					backward_ratio = 0;
				else
					//backward_ratio = __frcp_rn(fp_value[event_index] + (sc_coeff/4));
					backward_ratio = __frcp_rn(fp_value[event_index] + ptr_sc_coeff[event_index]);

				


				intersection_index_y = (int)ceilf(intersection_y / voxel_size_y) + center_y;
				intersection_index_z = (int)ceilf(intersection_z / voxel_size_z) + center_z;

				//compute the solid angle ratio w.r.t. the location of the intersection
				solid_angle_ratio = 1 + 2 * fabsf(t - 0.5f);
				//solid_angle_ratio = solid_angle_ratio*solid_angle_ratio;
				solid_angle_ratio = 1*dist*dist ;


				//compute the distance to the TOF center
				dist_to_TOF_center = (0.5f - t)*dist - TOF_dist;
				t = TOF_ALPHA * __expf(-dist_to_TOF_center*dist_to_TOF_center*TOF_INV);

				for (inslice_z = intersection_index_z - range2; inslice_z <= intersection_index_z + range2; inslice_z++){
					for (inslice_y = intersection_index_y - range1; inslice_y <= intersection_index_y + range1; inslice_y++){
						if (inslice_y >= 0 && inslice_y<num_y && inslice_z >= subslice_z0 && inslice_z<subslice_z1){
							//projection_vector_y = (inslice_y - center_y - 0.5)*voxel_size_y - intersection_y;
							projection_vector_y = inslice_y*voxel_size_y - H_CENTER - intersection_y;
							//projection_vector_z = (inslice_z - center_z - 0.5)*voxel_size_z - intersection_z;
							projection_vector_z = inslice_z*voxel_size_z - V_CENTER - intersection_z;
							dot_product = projection_vector_y*unit_dir_vector_y + projection_vector_z*unit_dir_vector_z;
							point_line_distance_square = projection_vector_y*projection_vector_y + projection_vector_z*projection_vector_z - dot_product*dot_product;
							weight = t * backward_ratio * __expf(-point_line_distance_square*SIGMA_INV - integral_mu) * ALPHA * GLOBAL_SCALE;
							atomicAdd(&current_slice_image[(inslice_z - subslice_z0)*num_y + inslice_y], (weight) / solid_angle_ratio);
						}
					}
				}
			}//intersects current slice
		}
		__syncthreads();
		//write the current sub slice from shared memory back to global memory using atomic operation
		//Since this write is not coalesced,  think about optimization techniques
		for (int j = threadIdx.x; j<SIZE_SUB_SLICE_x; j = j + blockDim.x){
			//atomicAdd(&image_ptr[ (j+sub_slice*SIZE_SUB_SLICE)*image_width+current_slice], current_slice_image[j]);
			atomicAdd(&image_ptr[(j + sub_slice*SIZE_SUB_SLICE_x)*num_x + current_slice], current_slice_image[j]);
			current_slice_image[j] = 0.0f;
		}
		__syncthreads();
	}
}

void
__global__ _TOF_b_ratio_proj_lst_cuda_y_kernel_scatter_atten(float* image, float* fp_y, float* atten_per_event, LST_LORs_scatter *_events_y_dominant_uvm, PARAMETERS_IN_DEVICE_t* parameters_device){
	int event_index, current_slice, sub_slice;
	float src_x, src_y, src_z, dest_x, dest_y, dest_z;
	int coeff, c;
	float sc_coeff;
	float dir_x, dir_y, dir_z;
	float TOF_dist;
	float dist_to_TOF_center;
	float unit_dir_vector_x, unit_dir_vector_z;
	float projection_vector_x, projection_vector_z;
	float dot_product;
	float point_line_distance_square;
	float weight;
	float dist, dist_squared;
	float t, integral_mu;
	float solid_angle_ratio;
	float backward_ratio;
	float intersection_x, intersection_y, intersection_z;
	int intersection_index_x, intersection_index_z;
	int inslice_x, inslice_z;
	int subslice_z0, subslice_z1;
	__shared__ float H_CENTER;
	__shared__ float V_CENTER;
	__shared__ float SIGMA_INV;
	__shared__ float ALPHA;
	__shared__ float current_slice_image[SIZE_SUB_SLICE_y];
	__shared__ float voxel_size_x, voxel_size_y, voxel_size_z;
	__shared__ int center_x, center_y, center_z;
	__shared__ int range1, range2;
	__shared__ int num_x, num_y, num_z;
	__shared__ float *ptr_src_x, *ptr_src_y, *ptr_src_z, *ptr_dest_x, *ptr_dest_y, *ptr_dest_z, *ptr_TOF_dist;
	__shared__ float *ptr_FWHM;
	__shared__ int num_lines;
	__shared__ float *image_ptr;
	__shared__ float *fp_value;
	__shared__ float *atten_factor;
	__shared__ float *ptr_sc_coeff;
	__shared__ float TOF_INV, TOF_ALPHA;
	if (threadIdx.x == 0){
		voxel_size_x = parameters_device->X_SAMP;
		voxel_size_y = parameters_device->Y_SAMP;
		voxel_size_z = parameters_device->Z_SAMP;
		center_x = parameters_device->X_INDEX_CENT;
		center_y = parameters_device->Y_INDEX_CENT;
		center_z = parameters_device->Z_INDEX_CENT;
		num_x = parameters_device->NUM_X;
		num_y = parameters_device->NUM_Y;
		num_z = parameters_device->NUM_Z;
		H_CENTER = parameters_device->H_CENTER_X;
		V_CENTER = parameters_device->V_CENTER;
		SIGMA_INV = parameters_device->FWHM_sigma_inv;
		ALPHA = parameters_device->FWHM_alpha;
		range1 = parameters_device->RANGE1;
		range2 = parameters_device->RANGE2;
		TOF_INV = parameters_device->TOF_inv;
		TOF_ALPHA = parameters_device->TOF_alpha;

		ptr_src_x = _events_y_dominant_uvm->src_x;
		ptr_src_y = _events_y_dominant_uvm->src_y;
		ptr_src_z = _events_y_dominant_uvm->src_z;
		ptr_dest_x = _events_y_dominant_uvm->dest_x;
		ptr_dest_y = _events_y_dominant_uvm->dest_y;
		ptr_dest_z = _events_y_dominant_uvm->dest_z;
		ptr_FWHM = _events_y_dominant_uvm->coeff;
		ptr_sc_coeff = _events_y_dominant_uvm->sc_coeff;

		ptr_TOF_dist = _events_y_dominant_uvm->TOF_dist;
		num_lines = _events_y_dominant_uvm->num_lines;
		image_ptr = image;
		fp_value = fp_y;
		atten_factor = atten_per_event;
	}

	for (int i = threadIdx.x; i<SIZE_SUB_SLICE_y; i = i + blockDim.x){
		current_slice_image[i] = 0.0f;
	}
	__syncthreads();
	c = 1;
	current_slice = blockIdx.x;
	intersection_y = __fmul_rn((current_slice - center_y - 0.5f), voxel_size_y);
	for (sub_slice = 0; sub_slice<NUM_SUB_SLICE_y; sub_slice++){
		subslice_z0 = (num_z / NUM_SUB_SLICE_y)*sub_slice;
		subslice_z1 = subslice_z0 + SUB_SLICE_HEIGHT_y;//(num_z/NUM_SUB_SLICE)*(sub_slice+1);
		//loop through all the lines and compute the backprojection to this current slice
		for (event_index = threadIdx.x; event_index< num_lines; event_index = event_index + blockDim.x){
			//get event line parameters
			src_x = ptr_src_x[event_index];
			src_y = ptr_src_y[event_index];
			src_z = ptr_src_z[event_index];
			dest_x = ptr_dest_x[event_index];
			dest_y = ptr_dest_y[event_index];
			dest_z = ptr_dest_z[event_index];
			coeff = ptr_FWHM[event_index];
			sc_coeff = ptr_sc_coeff[event_index];

			TOF_dist = ptr_TOF_dist[event_index];

			//only do the computation if current event line intersects current slice
			if (src_y<intersection_y - voxel_size_y && dest_y>intersection_y + voxel_size_y){
				dir_x = dest_x - src_x;
				dir_y = dest_y - src_y;
				dir_z = dest_z - src_z;
				//dist_squared = __fmaf_rn (dir_z,dir_z, __fmaf_rn(dir_y,dir_y,__fmul_rn(dir_x,dir_x) ) );
				dist_squared = dir_x*dir_x + dir_y*dir_y + dir_z*dir_z;
				dist = __fsqrt_rn(dist_squared);

				unit_dir_vector_x = dir_x / dist;
				unit_dir_vector_z = dir_z / dist;

				t = (intersection_y - src_y) / dir_y;
				intersection_x = t*dir_x + src_x;
				intersection_z = t*dir_z + src_z;

				integral_mu = atten_factor[event_index];

				if (fp_value[event_index] == 0)
					backward_ratio = 0;
				else
					//backward_ratio = __frcp_rn(fp_value[event_index] + (sc_coeff/4));
					backward_ratio = __frcp_rn(fp_value[event_index] + ptr_sc_coeff[event_index]);
				

				intersection_index_x = (int)ceilf(intersection_x / voxel_size_x) + center_x;
				intersection_index_z = (int)ceilf(intersection_z / voxel_size_z) + center_z;

				//compute the solid angle ratio w.r.t. the location of the intersection
				solid_angle_ratio = 1 + 2 * fabsf(t - 0.5f);
				//solid_angle_ratio = solid_angle_ratio*solid_angle_ratio;
				solid_angle_ratio = 1*dist*dist;


				//compute the distance to the TOF center
				dist_to_TOF_center = (0.5f - t)*dist - TOF_dist;
				t = TOF_ALPHA * __expf(-dist_to_TOF_center*dist_to_TOF_center*TOF_INV);

				for (inslice_z = intersection_index_z - range2; inslice_z <= intersection_index_z + range2; inslice_z++){
					for (inslice_x = intersection_index_x - range1; inslice_x <= intersection_index_x + range1; inslice_x++){
						if (inslice_x >= 0 && inslice_x<num_x && inslice_z >= subslice_z0 && inslice_z<subslice_z1){
							//projection_vector_x = (inslice_x - center_x - 0.5)*voxel_size_x - intersection_x;
							projection_vector_x = inslice_x*voxel_size_x - H_CENTER - intersection_x;
							//projection_vector_z = (inslice_z - center_z - 0.5)*voxel_size_z - intersection_z;
							projection_vector_z = inslice_z*voxel_size_z - V_CENTER - intersection_z;
							dot_product = projection_vector_x*unit_dir_vector_x + projection_vector_z*unit_dir_vector_z;
							point_line_distance_square = projection_vector_x*projection_vector_x + projection_vector_z*projection_vector_z - dot_product*dot_product;
							weight = t * backward_ratio * __expf(-point_line_distance_square*SIGMA_INV - integral_mu) * ALPHA * GLOBAL_SCALE;
							atomicAdd(&current_slice_image[(inslice_z - subslice_z0)*num_x + inslice_x], (weight) / solid_angle_ratio);
						}
					}
				}
			}//intersects current slice
		}
		__syncthreads();
		//write the current sub slice from shared memory back to global memory using atomic operation
		//Since this write is not coalesced,  think about optimization techniques
		for (int j = threadIdx.x; j<SIZE_SUB_SLICE_y; j = j + blockDim.x){
			atomicAdd(&image_ptr[num_x*num_y*(subslice_z0 + j / num_x) + num_x*current_slice + j%num_x], current_slice_image[j]);
			//image_ptr[ image_width*image_height*(subslice_z0+j/image_width) + image_width*current_slice + j%image_width ] = current_slice_image[j];
			current_slice_image[j] = 0.0f;
		}
		__syncthreads();
	}
}



void
__global__ _bproj_one_lst_cuda_x_kernel(float* image, float* fp_x, LST_LORs *_events_x_dominant_uvm, PARAMETERS_IN_DEVICE_t* parameters_device){
	int event_index, current_slice, sub_slice;
	float src_x, src_y, src_z, dest_x, dest_y, dest_z;
	float dir_x, dir_y, dir_z;
	//float TOF_dist;
	//float dist_to_TOF_center;
	float unit_dir_vector_y, unit_dir_vector_z;
	float projection_vector_y, projection_vector_z;
	float dot_product;
	float point_line_distance_square;
	float weight;
	float dist, dist_squared;
	float t;
	float solid_angle_ratio;
	float backward_ratio;
	float intersection_x, intersection_y, intersection_z;
	int  intersection_index_y, intersection_index_z;
	int  inslice_y, inslice_z;
	int subslice_z0, subslice_z1;
	__shared__ float H_CENTER;
	__shared__ float V_CENTER;
	__shared__ float SIGMA_INV;
	__shared__ float ALPHA;
	__shared__ float current_slice_image[SIZE_SUB_SLICE_x];
	__shared__ float voxel_size_x, voxel_size_y, voxel_size_z;
	__shared__ int center_x, center_y, center_z;
	__shared__ int range1, range2;
	__shared__ int num_x, num_y, num_z;
	__shared__ float *ptr_src_x, *ptr_src_y, *ptr_src_z, *ptr_dest_x, *ptr_dest_y, *ptr_dest_z;
	//__shared__ float *ptr_TOF_dist;
	__shared__ int num_lines;
	__shared__ float *image_ptr;
	//__shared__ float *fp_value;

	if (threadIdx.x == 0){
		voxel_size_x = parameters_device->X_SAMP;
		voxel_size_y = parameters_device->Y_SAMP;
		voxel_size_z = parameters_device->Z_SAMP;
		center_x = parameters_device->X_INDEX_CENT;
		center_y = parameters_device->Y_INDEX_CENT;
		center_z = parameters_device->Z_INDEX_CENT;
		num_x = parameters_device->NUM_X;
		num_y = parameters_device->NUM_Y;
		num_z = parameters_device->NUM_Z;
		H_CENTER = parameters_device->H_CENTER_Y;
		V_CENTER = parameters_device->V_CENTER;
		SIGMA_INV = parameters_device->FWHM_sigma_inv;
		ALPHA = parameters_device->FWHM_alpha;
		range1 = parameters_device->RANGE1;
		range2 = parameters_device->RANGE2;

		ptr_src_x = _events_x_dominant_uvm->src_x;
		ptr_src_y = _events_x_dominant_uvm->src_y;
		ptr_src_z = _events_x_dominant_uvm->src_z;
		ptr_dest_x = _events_x_dominant_uvm->dest_x;
		ptr_dest_y = _events_x_dominant_uvm->dest_y;
		ptr_dest_z = _events_x_dominant_uvm->dest_z;
		num_lines = _events_x_dominant_uvm->num_lines;
		/*
		ptr_src_x = device_event_x.src_x;
		ptr_src_y = device_event_x.src_y;
		ptr_src_z = device_event_x.src_z;
		ptr_dest_x = device_event_x.dest_x;
		ptr_dest_y = device_event_x.dest_y;
		ptr_dest_z = device_event_x.dest_z;
		num_lines = device_event_x.num_lines;
		*/

		//ptr_TOF_dist = events_x.TOF_dist;
		image_ptr = image;
		//fp_value = fp_x;
	}

	//initialize shared memory to zero
	for (int i = threadIdx.x; i<SIZE_SUB_SLICE_x; i = i + blockDim.x){
		current_slice_image[i] = 0.0f;
	}
	__syncthreads();

	current_slice = blockIdx.x;
	intersection_x = __fmul_rn((current_slice - center_x - 0.5f), voxel_size_x);
	for (sub_slice = 0; sub_slice<NUM_SUB_SLICE_x; sub_slice++){
		subslice_z0 = (num_z / NUM_SUB_SLICE_x)*sub_slice;
		subslice_z1 = subslice_z0 + SUB_SLICE_HEIGHT_x;//(num_z/NUM_SUB_SLICE)*(sub_slice+1);
		//loop through all the lines and compute the backprojection to this current slice
		for (event_index = threadIdx.x; event_index< num_lines; event_index = event_index + blockDim.x){
			//get event line parameters
			src_x = ptr_src_x[event_index];
			src_y = ptr_src_y[event_index];
			src_z = ptr_src_z[event_index];
			dest_x = ptr_dest_x[event_index];
			dest_y = ptr_dest_y[event_index];
			dest_z = ptr_dest_z[event_index];
			//TOF_dist = ptr_TOF_dist[event_index];

			//only do the computation if current event line intersects current slice
			if (src_x<intersection_x - voxel_size_x && dest_x>intersection_x + voxel_size_x){
				dir_x = dest_x - src_x;
				dir_y = dest_y - src_y;
				dir_z = dest_z - src_z;
				//dist_squared = __fmaf_rn (dir_z,dir_z, __fmaf_rn(dir_y,dir_y,__fmul_rn(dir_x,dir_x) ) );
				dist_squared = dir_x*dir_x + dir_y*dir_y + dir_z*dir_z;
				dist = __fsqrt_rn(dist_squared);

				unit_dir_vector_y = dir_y / dist;
				unit_dir_vector_z = dir_z / dist;

				t = (intersection_x - src_x) / dir_x;
				intersection_y = t*dir_y + src_y;
				intersection_z = t*dir_z + src_z;

				backward_ratio = 1.0;

				intersection_index_y = (int)ceilf(intersection_y / voxel_size_y) + center_y;
				intersection_index_z = (int)ceilf(intersection_z / voxel_size_z) + center_z;

				//compute the solid angle ratio w.r.t. the location of the intersection
				solid_angle_ratio = 1 + 2 * fabsf(t - 0.5f);
				solid_angle_ratio = solid_angle_ratio*solid_angle_ratio;

				//compute the distance to the TOF center
				//dist_to_TOF_center = (0.5f-t)*dist - TOF_dist;
				//t = TOF_ALPHA * __expf( - dist_to_TOF_center*dist_to_TOF_center*TOF_INV);

				for (inslice_z = intersection_index_z - range2; inslice_z <= intersection_index_z + range2; inslice_z++){
					for (inslice_y = intersection_index_y - range1; inslice_y <= intersection_index_y + range1; inslice_y++){
						if (inslice_y >= 0 && inslice_y<num_y && inslice_z >= subslice_z0 && inslice_z<subslice_z1){
							//projection_vector_y = (inslice_y - center_y - 0.5)*voxel_size_y - intersection_y;
							projection_vector_y = inslice_y*voxel_size_y - H_CENTER - intersection_y;
							//projection_vector_z = (inslice_z - center_z - 0.5)*voxel_size_z - intersection_z;
							projection_vector_z = inslice_z*voxel_size_z - V_CENTER - intersection_z;
							dot_product = projection_vector_y*unit_dir_vector_y + projection_vector_z*unit_dir_vector_z;
							point_line_distance_square = projection_vector_y*projection_vector_y + projection_vector_z*projection_vector_z - dot_product*dot_product;
							//weight = t * backward_ratio * __expf( - point_line_distance_square*SIGMA_INV );
							weight = backward_ratio * __expf(-point_line_distance_square*SIGMA_INV) * ALPHA ;
							atomicAdd(&current_slice_image[(inslice_z - subslice_z0)*num_y + inslice_y], weight / solid_angle_ratio);
						}
					}
				}
			}//intersects current slice
		}
		__syncthreads();
		//write the current sub slice from shared memory back to global memory using atomic operation
		//Since this write is not coalesced,  think about optimization techniques
		for (int j = threadIdx.x; j<SIZE_SUB_SLICE_x; j = j + blockDim.x){
			//atomicAdd(&image_ptr[ (j+sub_slice*SIZE_SUB_SLICE)*image_width+current_slice], current_slice_image[j]);
			atomicAdd(&image_ptr[(j + sub_slice*SIZE_SUB_SLICE_x)*num_x + current_slice], current_slice_image[j]);
			current_slice_image[j] = 0.0f;
		}
		__syncthreads();
	}
}

void
__global__ _bproj_one_lst_cuda_y_kernel(float* image, float* fp_y, LST_LORs *_events_y_dominant_uvm, PARAMETERS_IN_DEVICE_t* parameters_device){
	int event_index, current_slice, sub_slice;
	float src_x, src_y, src_z, dest_x, dest_y, dest_z;
	float dir_x, dir_y, dir_z;
	//float TOF_dist;
	//float dist_to_TOF_center;
	float unit_dir_vector_x, unit_dir_vector_z;
	float projection_vector_x, projection_vector_z;
	float dot_product;
	float point_line_distance_square;
	float weight;
	float dist, dist_squared;
	float t;
	float solid_angle_ratio;
	float backward_ratio;
	float intersection_x, intersection_y, intersection_z;
	int intersection_index_x, intersection_index_z;
	int inslice_x, inslice_z;
	int subslice_z0, subslice_z1;
	__shared__ float H_CENTER;
	__shared__ float V_CENTER;
	__shared__ float SIGMA_INV;
	__shared__ float ALPHA;
	__shared__ float current_slice_image[SIZE_SUB_SLICE_y];
	__shared__ float voxel_size_x, voxel_size_y, voxel_size_z;
	__shared__ int center_x, center_y, center_z;
	__shared__ int range1, range2;
	__shared__ int num_x, num_y, num_z;
	__shared__ float *ptr_src_x, *ptr_src_y, *ptr_src_z, *ptr_dest_x, *ptr_dest_y, *ptr_dest_z;
	//__shared__ float *ptr_TOF_dist;
	__shared__ int num_lines;
	__shared__ float *image_ptr;
	//__shared__ float *fp_value;

	if (threadIdx.x == 0){
		voxel_size_x = parameters_device->X_SAMP;
		voxel_size_y = parameters_device->Y_SAMP;
		voxel_size_z = parameters_device->Z_SAMP;
		center_x = parameters_device->X_INDEX_CENT;
		center_y = parameters_device->Y_INDEX_CENT;
		center_z = parameters_device->Z_INDEX_CENT;
		num_x = parameters_device->NUM_X;
		num_y = parameters_device->NUM_Y;
		num_z = parameters_device->NUM_Z;
		H_CENTER = parameters_device->H_CENTER_X;
		V_CENTER = parameters_device->V_CENTER;
		SIGMA_INV = parameters_device->FWHM_sigma_inv;
		ALPHA = parameters_device->FWHM_alpha;
		range1 = parameters_device->RANGE1;
		range2 = parameters_device->RANGE2;

		ptr_src_x = _events_y_dominant_uvm->src_x;
		ptr_src_y = _events_y_dominant_uvm->src_y;
		ptr_src_z = _events_y_dominant_uvm->src_z;
		ptr_dest_x = _events_y_dominant_uvm->dest_x;
		ptr_dest_y = _events_y_dominant_uvm->dest_y;
		ptr_dest_z = _events_y_dominant_uvm->dest_z;
		num_lines = _events_y_dominant_uvm->num_lines;
		/*
		ptr_src_x = device_event_y.src_x;
		ptr_src_y = device_event_y.src_y;
		ptr_src_z = device_event_y.src_z;
		ptr_dest_x = device_event_y.dest_x;
		ptr_dest_y = device_event_y.dest_y;
		ptr_dest_z = device_event_y.dest_z;
		num_lines = device_event_y.num_lines;
		*/

		//ptr_TOF_dist = events_y.TOF_dist;
		image_ptr = image;
		//fp_value = fp_y;
	}

	for (int i = threadIdx.x; i<SIZE_SUB_SLICE_y; i = i + blockDim.x){
		current_slice_image[i] = 0.0f;
	}
	__syncthreads();

	current_slice = blockIdx.x;
	intersection_y = __fmul_rn((current_slice - center_y - 0.5f), voxel_size_y);
	for (sub_slice = 0; sub_slice<NUM_SUB_SLICE_y; sub_slice++){
		subslice_z0 = (num_z / NUM_SUB_SLICE_y)*sub_slice;
		subslice_z1 = subslice_z0 + SUB_SLICE_HEIGHT_y;//(num_z/NUM_SUB_SLICE)*(sub_slice+1);
		//loop through all the lines and compute the backprojection to this current slice
		for (event_index = threadIdx.x; event_index< num_lines; event_index = event_index + blockDim.x){
			//get event line parameters
			src_x = ptr_src_x[event_index];
			src_y = ptr_src_y[event_index];
			src_z = ptr_src_z[event_index];
			dest_x = ptr_dest_x[event_index];
			dest_y = ptr_dest_y[event_index];
			dest_z = ptr_dest_z[event_index];
			//TOF_dist = ptr_TOF_dist[event_index];

			//only do the computation if current event line intersects current slice
			if (src_y<intersection_y - voxel_size_y && dest_y>intersection_y + voxel_size_y){
				dir_x = dest_x - src_x;
				dir_y = dest_y - src_y;
				dir_z = dest_z - src_z;
				//dist_squared = __fmaf_rn (dir_z,dir_z, __fmaf_rn(dir_y,dir_y,__fmul_rn(dir_x,dir_x) ) );
				dist_squared = dir_x*dir_x + dir_y*dir_y + dir_z*dir_z;
				dist = __fsqrt_rn(dist_squared);

				unit_dir_vector_x = dir_x / dist;
				unit_dir_vector_z = dir_z / dist;

				t = (intersection_y - src_y) / dir_y;
				intersection_x = t*dir_x + src_x;
				intersection_z = t*dir_z + src_z;

				backward_ratio = 1.0;

				intersection_index_x = (int)ceilf(intersection_x / voxel_size_x) + center_x;
				intersection_index_z = (int)ceilf(intersection_z / voxel_size_z) + center_z;

				//compute the solid angle ratio w.r.t. the location of the intersection
				solid_angle_ratio = 1 + 2 * fabsf(t - 0.5f);
				solid_angle_ratio = solid_angle_ratio*solid_angle_ratio;

				//compute the distance to the TOF center
				//dist_to_TOF_center = (0.5f-t)*dist - TOF_dist;
				//t = TOF_ALPHA * __expf( - dist_to_TOF_center*dist_to_TOF_center*TOF_INV);

				for (inslice_z = intersection_index_z - range2; inslice_z <= intersection_index_z + range2; inslice_z++){
					for (inslice_x = intersection_index_x - range1; inslice_x <= intersection_index_x + range1; inslice_x++){
						if (inslice_x >= 0 && inslice_x<num_x && inslice_z >= subslice_z0 && inslice_z<subslice_z1){
							//projection_vector_x = (inslice_x - center_x - 0.5)*voxel_size_x - intersection_x;
							projection_vector_x = inslice_x*voxel_size_x - H_CENTER - intersection_x;
							//projection_vector_z = (inslice_z - center_z - 0.5)*voxel_size_z - intersection_z;
							projection_vector_z = inslice_z*voxel_size_z - V_CENTER - intersection_z;
							dot_product = projection_vector_x*unit_dir_vector_x + projection_vector_z*unit_dir_vector_z;
							point_line_distance_square = projection_vector_x*projection_vector_x + projection_vector_z*projection_vector_z - dot_product*dot_product;
							//weight = t * backward_ratio * __expf( - point_line_distance_square*SIGMA_INV );
							weight = backward_ratio * __expf(-point_line_distance_square*SIGMA_INV) * ALPHA ;
							atomicAdd(&current_slice_image[(inslice_z - subslice_z0)*num_x + inslice_x], weight / solid_angle_ratio);
						}
					}
				}
			}//intersects current slice
		}
		__syncthreads();
		//write the current sub slice from shared memory back to global memory using atomic operation
		//Since this write is not coalesced,  think about optimization techniques
		for (int j = threadIdx.x; j<SIZE_SUB_SLICE_y; j = j + blockDim.x){
			atomicAdd(&image_ptr[num_x*num_y*(subslice_z0 + j / num_x) + num_x*current_slice + j%num_x], current_slice_image[j]);
			//image_ptr[ image_width*image_height*(subslice_z0+j/image_width) + image_width*current_slice + j%image_width ] = current_slice_image[j];
			current_slice_image[j] = 0.0f;
		}
		__syncthreads();
	}
}



// for computing attenuation corrected sensitivity images

void
__global__ _fproj_atten_lst_cuda_x_kernel(float* image, float* fp_x, LST_LORs *_events_x_dominant_uvm, PARAMETERS_IN_DEVICE_t* parameters_device){
	int event_index, current_slice, sub_slice;
	float src_x, src_y, src_z, dest_x, dest_y, dest_z;
	int coeff;
	float dir_x, dir_y, dir_z;
	//float TOF_dist;
	//float dist_to_TOF_center;
	float unit_dir_vector_y, unit_dir_vector_z;
	float projection_vector_y, projection_vector_z;
	float dot_product;
	float point_line_distance_square;
	float intersection_length_approx;
	//float weight;
	float dist, dist_squared;
	float t;
	//float solid_angle_ratio;
	float forward_sum;
	float intersection_x, intersection_y, intersection_z;
	int  intersection_index_y, intersection_index_z;
	int  inslice_y, inslice_z;
	int subslice_z0, subslice_z1;
	float spherical_voxel_filling_ratio; //scaling factor on the diameter of the sphere representing the cubic voxel, with 1.0 equal to the diameter of the inner sphere
	float voxel_dim; //voxel dimension considering voxel_size_x/y/z are the same

	__shared__ float H_CENTER;
	__shared__ float V_CENTER;
	//__shared__ float SIGMA_INV;
	//__shared__ float ALPHA;
	__shared__ float current_slice_image[SIZE_SUB_SLICE_x];
	__shared__ float voxel_size_x, voxel_size_y, voxel_size_z;
	__shared__ int center_x, center_y, center_z;
	__shared__ int range1, range2;
	__shared__ int num_x, num_y, num_z;
	__shared__ float *ptr_src_x, *ptr_src_y, *ptr_src_z, *ptr_dest_x, *ptr_dest_y, *ptr_dest_z;
	__shared__ float *ptr_FWHM;
	//__shared__ float *ptr_TOF_dist;
	__shared__ int num_lines;
	__shared__ float *image_ptr;
	__shared__ float *fp_value;
	if (threadIdx.x == 0){
		voxel_size_x = parameters_device->X_SAMP;
		voxel_size_y = parameters_device->Y_SAMP;
		voxel_size_z = parameters_device->Z_SAMP;
		center_x = parameters_device->X_INDEX_CENT;
		center_y = parameters_device->Y_INDEX_CENT;
		center_z = parameters_device->Z_INDEX_CENT;
		num_x = parameters_device->NUM_X;
		num_y = parameters_device->NUM_Y;
		num_z = parameters_device->NUM_Z;
		H_CENTER = parameters_device->H_CENTER_Y;
		V_CENTER = parameters_device->V_CENTER;
		//SIGMA_INV = parameters_device->FWHM_sigma_inv;
		//ALPHA = parameters_device->FWHM_alpha;
		range1 = parameters_device->RANGE1;
		range2 = parameters_device->RANGE2;

		ptr_src_x = _events_x_dominant_uvm->src_x;
		ptr_src_y = _events_x_dominant_uvm->src_y;
		ptr_src_z = _events_x_dominant_uvm->src_z;
		ptr_dest_x = _events_x_dominant_uvm->dest_x;
		ptr_dest_y = _events_x_dominant_uvm->dest_y;
		ptr_dest_z = _events_x_dominant_uvm->dest_z;
		ptr_FWHM = _events_x_dominant_uvm->coeff;
	
		num_lines = _events_x_dominant_uvm->num_lines;
		/*
		ptr_src_x = device_event_x.src_x;
		ptr_src_y = device_event_x.src_y;
		ptr_src_z = device_event_x.src_z;
		ptr_dest_x = device_event_x.dest_x;
		ptr_dest_y = device_event_x.dest_y;
		ptr_dest_z = device_event_x.dest_z;
		num_lines = device_event_x.num_lines;
		*/

		//ptr_TOF_dist = events_x.TOF_dist;
		image_ptr = image;
		fp_value = fp_x;
	}
	__syncthreads();
	current_slice = blockIdx.x;
	spherical_voxel_filling_ratio = parameters_device->spherical_voxel_ratio;
	voxel_dim = voxel_size_x;
	intersection_x = (current_slice - center_x - 0.5f)*voxel_size_x;
	for (sub_slice = 0; sub_slice<NUM_SUB_SLICE_x; sub_slice++){
		subslice_z0 = (num_z / NUM_SUB_SLICE_x)*sub_slice;
		subslice_z1 = subslice_z0 + SUB_SLICE_HEIGHT_x;//(num_z/NUM_SUB_SLICE)*(sub_slice+1);
		//load current image to shared memory
		for (int i = threadIdx.x; i<SIZE_SUB_SLICE_x; i = i + blockDim.x){
			current_slice_image[i] = image_ptr[(i + sub_slice*SIZE_SUB_SLICE_x)*num_x + current_slice];
		}
		__syncthreads();
		//loop through all the lines and compute the forward projection to this current slice
		for (event_index = threadIdx.x; event_index< num_lines; event_index = event_index + blockDim.x){
			//get event line parameters
			src_x = ptr_src_x[event_index];
			src_y = ptr_src_y[event_index];
			src_z = ptr_src_z[event_index];
			dest_x = ptr_dest_x[event_index];
			dest_y = ptr_dest_y[event_index];
			dest_z = ptr_dest_z[event_index];
			coeff = ptr_FWHM[event_index];
		
			//TOF_dist = ptr_TOF_dist[event_index];


			//only do the computation if current event line intersects current slice
			if (src_x<intersection_x - voxel_size_x && dest_x>intersection_x + voxel_size_x){
				dir_x = dest_x - src_x;
				dir_y = dest_y - src_y;
				dir_z = dest_z - src_z;
				//dist_squared = __fmaf_rn (dir_z,dir_z, __fmaf_rn(dir_y,dir_y,__fmul_rn(dir_x,dir_x) ) );
				dist_squared = dir_x*dir_x + dir_y*dir_y + dir_z*dir_z;
				dist = __fsqrt_rn(dist_squared);

				unit_dir_vector_y = dir_y / dist;
				unit_dir_vector_z = dir_z / dist;

				//t keeps the ratio of the two segment of LOR segmented by the intersection plane
				t = (intersection_x - src_x) / dir_x;
				intersection_y = t*dir_y + src_y;
				intersection_z = t*dir_z + src_z;

				intersection_index_y = (int)ceilf(intersection_y / voxel_size_y) + center_y;
				intersection_index_z = (int)ceilf(intersection_z / voxel_size_z) + center_z;

				//compute the distance to the TOF center
				//dist_to_TOF_center = (0.5f-t)*dist - TOF_dist;
				//t = TOF_ALPHA * __expf( - dist_to_TOF_center*dist_to_TOF_center*TOF_INV);

				forward_sum = 0.0f;

				for (inslice_z = intersection_index_z - range2; inslice_z <= intersection_index_z + range2; inslice_z++){
					for (inslice_y = intersection_index_y - range1; inslice_y <= intersection_index_y + range1; inslice_y++){
						if (inslice_y >= 0 && inslice_y<num_y && inslice_z >= subslice_z0 && inslice_z<subslice_z1){
							//projection_vector_y = (inslice_y - center_y - 0.5)*voxel_size_y - intersection_y;
							projection_vector_y = inslice_y*voxel_size_y - H_CENTER - intersection_y;
							//projection_vector_z = (inslice_z - center_z - 0.5)*voxel_size_z - intersection_z;
							projection_vector_z = inslice_z*voxel_size_z - V_CENTER - intersection_z;
							dot_product = projection_vector_y*unit_dir_vector_y + projection_vector_z*unit_dir_vector_z;
							point_line_distance_square = projection_vector_y*projection_vector_y + projection_vector_z*projection_vector_z - dot_product*dot_product;

							intersection_length_approx = 1.54 * voxel_dim*voxel_dim - 4 * point_line_distance_square;
							if (intersection_length_approx <= 0)
								intersection_length_approx = 0;
							intersection_length_approx = __fsqrt_rn(intersection_length_approx)*spherical_voxel_filling_ratio;

							//forward_sum+= current_slice_image[(inslice_z-subslice_z0)*num_y+inslice_y] * __expf( - point_line_distance_square*SIGMA_INV ) * ALPHA * t;
							//forward_sum += current_slice_image[(inslice_z - subslice_z0)*num_y + inslice_y] * __expf(-point_line_distance_square*SIGMA_INV) * ALPHA;
							forward_sum += current_slice_image[(inslice_z - subslice_z0)*num_y + inslice_y] * intersection_length_approx  ;


							//atomicAdd(&fp_value[event_index],weight);
						}
					}
				}
				atomicAdd(&fp_value[event_index], forward_sum );
			}//intersects current slice
		}
	}
}

void
__global__ _fproj_atten_lst_cuda_y_kernel(float* image, float* fp_y, LST_LORs *_events_y_dominant_uvm, PARAMETERS_IN_DEVICE_t* parameters_device){
	int event_index, current_slice, sub_slice;
	float src_x, src_y, src_z, dest_x, dest_y, dest_z;
	int coeff;
	float dir_x, dir_y, dir_z;
	//float TOF_dist;
	//float dist_to_TOF_center;
	float unit_dir_vector_x, unit_dir_vector_z;
	float projection_vector_x, projection_vector_z;
	float dot_product;
	float point_line_distance_square;
	float intersection_length_approx;
	//float weight;
	float dist, dist_squared;
	float t;
	//float solid_angle_ratio;
	float forward_sum;
	float intersection_x, intersection_y, intersection_z;
	int intersection_index_x, intersection_index_z;
	int inslice_x, inslice_z;
	int subslice_z0, subslice_z1;
	float spherical_voxel_filling_ratio; //scaling factor on the diameter of the sphere representing the cubic voxel, with 1.0 equal to the diameter of the inner sphere
	float voxel_dim; //voxel dimension considering voxel_size_x/y/z are the same

	__shared__ float H_CENTER;
	__shared__ float V_CENTER;
	//__shared__ float SIGMA_INV;
	//__shared__ float ALPHA;
	__shared__ float current_slice_image[SIZE_SUB_SLICE_y];
	__shared__ float voxel_size_x, voxel_size_y, voxel_size_z;
	__shared__ int center_x, center_y, center_z;
	__shared__ int range1, range2;
	__shared__ int num_x, num_y, num_z;
	__shared__ float *ptr_src_x, *ptr_src_y, *ptr_src_z, *ptr_dest_x, *ptr_dest_y, *ptr_dest_z;
	__shared__ float *ptr_FWHM;
	//__shared__ float *ptr_TOF_dist;
	__shared__ int num_lines;
	__shared__ float *image_ptr;
	__shared__ float *fp_value;

	if (threadIdx.x == 0){
		voxel_size_x = parameters_device->X_SAMP;
		voxel_size_y = parameters_device->Y_SAMP;
		voxel_size_z = parameters_device->Z_SAMP;
		center_x = parameters_device->X_INDEX_CENT;
		center_y = parameters_device->Y_INDEX_CENT;
		center_z = parameters_device->Z_INDEX_CENT;
		num_x = parameters_device->NUM_X;
		num_y = parameters_device->NUM_Y;
		num_z = parameters_device->NUM_Z;
		H_CENTER = parameters_device->H_CENTER_X;
		V_CENTER = parameters_device->V_CENTER;
		//SIGMA_INV = parameters_device->FWHM_sigma_inv;
		//ALPHA = parameters_device->FWHM_alpha;
		range1 = parameters_device->RANGE1;
		range2 = parameters_device->RANGE2;

		ptr_src_x = _events_y_dominant_uvm->src_x;
		ptr_src_y = _events_y_dominant_uvm->src_y;
		ptr_src_z = _events_y_dominant_uvm->src_z;
		ptr_dest_x = _events_y_dominant_uvm->dest_x;
		ptr_dest_y = _events_y_dominant_uvm->dest_y;
		ptr_dest_z = _events_y_dominant_uvm->dest_z;
		ptr_FWHM = _events_y_dominant_uvm->coeff;
		
		num_lines = _events_y_dominant_uvm->num_lines;
		/*
		ptr_src_x = device_event_y.src_x;
		ptr_src_y = device_event_y.src_y;
		ptr_src_z = device_event_y.src_z;
		ptr_dest_x = device_event_y.dest_x;
		ptr_dest_y = device_event_y.dest_y;
		ptr_dest_z = device_event_y.dest_z;
		num_lines = device_event_y.num_lines;
		*/

		//ptr_TOF_dist = events_y.TOF_dist;
		image_ptr = image;
		fp_value = fp_y;
	}
	__syncthreads();
	current_slice = blockIdx.x;
	spherical_voxel_filling_ratio = parameters_device->spherical_voxel_ratio;
	voxel_dim = voxel_size_x;
	intersection_y = (current_slice - center_y - 0.5)*voxel_size_y;
	for (sub_slice = 0; sub_slice<NUM_SUB_SLICE_y; sub_slice++){
		subslice_z0 = (num_z / NUM_SUB_SLICE_y)*sub_slice;
		subslice_z1 = subslice_z0 + SUB_SLICE_HEIGHT_y;//(num_z/NUM_SUB_SLICE)*(sub_slice+1);
		//load current image slice to shared memory
		for (int i = threadIdx.x; i<SIZE_SUB_SLICE_y; i = i + blockDim.x){
			current_slice_image[i] = image_ptr[num_x*num_y*(subslice_z0 + i / num_x) + num_x*current_slice + i%num_x];
		}
		__syncthreads();
		//loop through all the lines and compute the backprojection to this current slice
		for (event_index = threadIdx.x; event_index< num_lines; event_index = event_index + blockDim.x){
			//get event line parameters
			src_x = ptr_src_x[event_index];
			src_y = ptr_src_y[event_index];
			src_z = ptr_src_z[event_index];
			dest_x = ptr_dest_x[event_index];
			dest_y = ptr_dest_y[event_index];
			dest_z = ptr_dest_z[event_index];
			coeff = ptr_FWHM[event_index];
			
			//TOF_dist = ptr_TOF_dist[event_index];

			//only do the computation if current event line intersects current slice
			if (src_y<intersection_y - voxel_size_y && dest_y>intersection_y + voxel_size_y){
				dir_x = dest_x - src_x;
				dir_y = dest_y - src_y;
				dir_z = dest_z - src_z;
				//dist_squared = __fmaf_rn (dir_z,dir_z, __fmaf_rn(dir_y,dir_y,__fmul_rn(dir_x,dir_x) ) );
				dist_squared = dir_x*dir_x + dir_y*dir_y + dir_z*dir_z;
				dist = __fsqrt_rn(dist_squared);

				unit_dir_vector_x = dir_x / dist;
				unit_dir_vector_z = dir_z / dist;

				t = (intersection_y - src_y) / dir_y;
				intersection_x = t*dir_x + src_x;
				intersection_z = t*dir_z + src_z;

				intersection_index_x = (int)ceilf(intersection_x / voxel_size_x) + center_x;
				intersection_index_z = (int)ceilf(intersection_z / voxel_size_z) + center_z;

				//compute the distance to the TOF center
				//dist_to_TOF_center = (0.5f-t)*dist - TOF_dist;
				//t = TOF_ALPHA * __expf( - dist_to_TOF_center*dist_to_TOF_center*TOF_INV);

				forward_sum = 0.0f;

				for (inslice_z = intersection_index_z - range2; inslice_z <= intersection_index_z + range2; inslice_z++){
					for (inslice_x = intersection_index_x - range1; inslice_x <= intersection_index_x + range1; inslice_x++){
						if (inslice_x >= 0 && inslice_x<num_x && inslice_z >= subslice_z0 && inslice_z<subslice_z1){
							//projection_vector_x = (inslice_x - center_x - 0.5)*voxel_size_x - intersection_x;
							projection_vector_x = inslice_x*voxel_size_x - H_CENTER - intersection_x;
							//projection_vector_z = (inslice_z - center_z - 0.5)*voxel_size_z - intersection_z;
							projection_vector_z = inslice_z*voxel_size_z - V_CENTER - intersection_z;
							dot_product = projection_vector_x*unit_dir_vector_x + projection_vector_z*unit_dir_vector_z;
							point_line_distance_square = projection_vector_x*projection_vector_x + projection_vector_z*projection_vector_z - dot_product*dot_product;

							intersection_length_approx = 1.54 * voxel_dim*voxel_dim - 4 * point_line_distance_square;
							if (intersection_length_approx <= 0)
								intersection_length_approx = 0;
							intersection_length_approx = __fsqrt_rn(intersection_length_approx)*spherical_voxel_filling_ratio;

							//forward_sum += current_slice_image[(inslice_z-subslice_z0)*num_x+inslice_x] * __expf( - point_line_distance_square*SIGMA_INV ) * ALPHA * t;
							//forward_sum += current_slice_image[(inslice_z - subslice_z0)*num_x + inslice_x] * __expf(-point_line_distance_square*SIGMA_INV) * ALPHA;
							forward_sum += current_slice_image[(inslice_z - subslice_z0)*num_x + inslice_x] * intersection_length_approx;

							//atomicAdd(&fp_value[event_index],weight);

						}
					}
				}
				atomicAdd(&fp_value[event_index], forward_sum);
			}//intersects current slice
		}
	}
}

void
__global__ _fproj_atten_lst_cuda_x_kernel_scatter(float* image, float* fp_x, LST_LORs_scatter *_events_x_dominant_uvm, PARAMETERS_IN_DEVICE_t* parameters_device){
	int event_index, current_slice, sub_slice;
	float src_x, src_y, src_z, dest_x, dest_y, dest_z;
	int coeff;
	float dir_x, dir_y, dir_z;
	//float TOF_dist;
	//float dist_to_TOF_center;
	float unit_dir_vector_y, unit_dir_vector_z;
	float projection_vector_y, projection_vector_z;
	float dot_product;
	float point_line_distance_square;
	float intersection_length_approx;
	//float weight;
	float dist, dist_squared;
	float t;
	//float solid_angle_ratio;
	float forward_sum;
	float intersection_x, intersection_y, intersection_z;
	int  intersection_index_y, intersection_index_z;
	int  inslice_y, inslice_z;
	int subslice_z0, subslice_z1;
	float spherical_voxel_filling_ratio; //scaling factor on the diameter of the sphere representing the cubic voxel, with 1.0 equal to the diameter of the inner sphere
	float voxel_dim; //voxel dimension considering voxel_size_x/y/z are the same

	__shared__ float H_CENTER;
	__shared__ float V_CENTER;
	//__shared__ float SIGMA_INV;
	//__shared__ float ALPHA;
	__shared__ float current_slice_image[SIZE_SUB_SLICE_x];
	__shared__ float voxel_size_x, voxel_size_y, voxel_size_z;
	__shared__ int center_x, center_y, center_z;
	__shared__ int range1, range2;
	__shared__ int num_x, num_y, num_z;
	__shared__ float *ptr_src_x, *ptr_src_y, *ptr_src_z, *ptr_dest_x, *ptr_dest_y, *ptr_dest_z;
	__shared__ float *ptr_FWHM;
	//__shared__ float *ptr_TOF_dist;
	__shared__ int num_lines;
	__shared__ float *image_ptr;
	__shared__ float *fp_value;
	if (threadIdx.x == 0){
		voxel_size_x = parameters_device->X_SAMP;
		voxel_size_y = parameters_device->Y_SAMP;
		voxel_size_z = parameters_device->Z_SAMP;
		center_x = parameters_device->X_INDEX_CENT;
		center_y = parameters_device->Y_INDEX_CENT;
		center_z = parameters_device->Z_INDEX_CENT;
		num_x = parameters_device->NUM_X;
		num_y = parameters_device->NUM_Y;
		num_z = parameters_device->NUM_Z;
		H_CENTER = parameters_device->H_CENTER_Y;
		V_CENTER = parameters_device->V_CENTER;
		//SIGMA_INV = parameters_device->FWHM_sigma_inv;
		//ALPHA = parameters_device->FWHM_alpha;
		range1 = parameters_device->RANGE1;
		range2 = parameters_device->RANGE2;

		ptr_src_x = _events_x_dominant_uvm->src_x;
		ptr_src_y = _events_x_dominant_uvm->src_y;
		ptr_src_z = _events_x_dominant_uvm->src_z;
		ptr_dest_x = _events_x_dominant_uvm->dest_x;
		ptr_dest_y = _events_x_dominant_uvm->dest_y;
		ptr_dest_z = _events_x_dominant_uvm->dest_z;
		ptr_FWHM = _events_x_dominant_uvm->coeff;

		num_lines = _events_x_dominant_uvm->num_lines;
		/*
		ptr_src_x = device_event_x.src_x;
		ptr_src_y = device_event_x.src_y;
		ptr_src_z = device_event_x.src_z;
		ptr_dest_x = device_event_x.dest_x;
		ptr_dest_y = device_event_x.dest_y;
		ptr_dest_z = device_event_x.dest_z;
		num_lines = device_event_x.num_lines;
		*/

		//ptr_TOF_dist = events_x.TOF_dist;
		image_ptr = image;
		fp_value = fp_x;
	}
	__syncthreads();
	current_slice = blockIdx.x;
	spherical_voxel_filling_ratio = parameters_device->spherical_voxel_ratio;
	voxel_dim = voxel_size_x;
	intersection_x = (current_slice - center_x - 0.5f)*voxel_size_x;
	for (sub_slice = 0; sub_slice<NUM_SUB_SLICE_x; sub_slice++){
		subslice_z0 = (num_z / NUM_SUB_SLICE_x)*sub_slice;
		subslice_z1 = subslice_z0 + SUB_SLICE_HEIGHT_x;//(num_z/NUM_SUB_SLICE)*(sub_slice+1);
		//load current image to shared memory
		for (int i = threadIdx.x; i<SIZE_SUB_SLICE_x; i = i + blockDim.x){
			current_slice_image[i] = image_ptr[(i + sub_slice*SIZE_SUB_SLICE_x)*num_x + current_slice];
		}
		__syncthreads();
		//loop through all the lines and compute the forward projection to this current slice
		for (event_index = threadIdx.x; event_index< num_lines; event_index = event_index + blockDim.x){
			//get event line parameters
			src_x = ptr_src_x[event_index];
			src_y = ptr_src_y[event_index];
			src_z = ptr_src_z[event_index];
			dest_x = ptr_dest_x[event_index];
			dest_y = ptr_dest_y[event_index];
			dest_z = ptr_dest_z[event_index];
			coeff = ptr_FWHM[event_index];

			//TOF_dist = ptr_TOF_dist[event_index];


			//only do the computation if current event line intersects current slice
			if (src_x<intersection_x - voxel_size_x && dest_x>intersection_x + voxel_size_x){
				dir_x = dest_x - src_x;
				dir_y = dest_y - src_y;
				dir_z = dest_z - src_z;
				//dist_squared = __fmaf_rn (dir_z,dir_z, __fmaf_rn(dir_y,dir_y,__fmul_rn(dir_x,dir_x) ) );
				dist_squared = dir_x*dir_x + dir_y*dir_y + dir_z*dir_z;
				dist = __fsqrt_rn(dist_squared);

				unit_dir_vector_y = dir_y / dist;
				unit_dir_vector_z = dir_z / dist;

				//t keeps the ratio of the two segment of LOR segmented by the intersection plane
				t = (intersection_x - src_x) / dir_x;
				intersection_y = t*dir_y + src_y;
				intersection_z = t*dir_z + src_z;

				intersection_index_y = (int)ceilf(intersection_y / voxel_size_y) + center_y;
				intersection_index_z = (int)ceilf(intersection_z / voxel_size_z) + center_z;

				//compute the distance to the TOF center
				//dist_to_TOF_center = (0.5f-t)*dist - TOF_dist;
				//t = TOF_ALPHA * __expf( - dist_to_TOF_center*dist_to_TOF_center*TOF_INV);

				forward_sum = 0.0f;

				for (inslice_z = intersection_index_z - range2; inslice_z <= intersection_index_z + range2; inslice_z++){
					for (inslice_y = intersection_index_y - range1; inslice_y <= intersection_index_y + range1; inslice_y++){
						if (inslice_y >= 0 && inslice_y<num_y && inslice_z >= subslice_z0 && inslice_z<subslice_z1){
							//projection_vector_y = (inslice_y - center_y - 0.5)*voxel_size_y - intersection_y;
							projection_vector_y = inslice_y*voxel_size_y - H_CENTER - intersection_y;
							//projection_vector_z = (inslice_z - center_z - 0.5)*voxel_size_z - intersection_z;
							projection_vector_z = inslice_z*voxel_size_z - V_CENTER - intersection_z;
							dot_product = projection_vector_y*unit_dir_vector_y + projection_vector_z*unit_dir_vector_z;
							point_line_distance_square = projection_vector_y*projection_vector_y + projection_vector_z*projection_vector_z - dot_product*dot_product;

							intersection_length_approx = 1.54 * voxel_dim*voxel_dim - 4 * point_line_distance_square;
							if (intersection_length_approx <= 0)
								intersection_length_approx = 0;
							intersection_length_approx = __fsqrt_rn(intersection_length_approx)*spherical_voxel_filling_ratio;

							//forward_sum+= current_slice_image[(inslice_z-subslice_z0)*num_y+inslice_y] * __expf( - point_line_distance_square*SIGMA_INV ) * ALPHA * t;
							//forward_sum += current_slice_image[(inslice_z - subslice_z0)*num_y + inslice_y] * __expf(-point_line_distance_square*SIGMA_INV) * ALPHA;
							forward_sum += current_slice_image[(inslice_z - subslice_z0)*num_y + inslice_y] * intersection_length_approx;


							//atomicAdd(&fp_value[event_index],weight);
						}
					}
				}
				atomicAdd(&fp_value[event_index], forward_sum);
			}//intersects current slice
		}
	}
}

void
__global__ _fproj_atten_lst_cuda_y_kernel_scatter(float* image, float* fp_y, LST_LORs_scatter *_events_y_dominant_uvm, PARAMETERS_IN_DEVICE_t* parameters_device){
	int event_index, current_slice, sub_slice;
	float src_x, src_y, src_z, dest_x, dest_y, dest_z;
	int coeff;
	float dir_x, dir_y, dir_z;
	//float TOF_dist;
	//float dist_to_TOF_center;
	float unit_dir_vector_x, unit_dir_vector_z;
	float projection_vector_x, projection_vector_z;
	float dot_product;
	float point_line_distance_square;
	float intersection_length_approx;
	//float weight;
	float dist, dist_squared;
	float t;
	//float solid_angle_ratio;
	float forward_sum;
	float intersection_x, intersection_y, intersection_z;
	int intersection_index_x, intersection_index_z;
	int inslice_x, inslice_z;
	int subslice_z0, subslice_z1;
	float spherical_voxel_filling_ratio; //scaling factor on the diameter of the sphere representing the cubic voxel, with 1.0 equal to the diameter of the inner sphere
	float voxel_dim; //voxel dimension considering voxel_size_x/y/z are the same

	__shared__ float H_CENTER;
	__shared__ float V_CENTER;
	//__shared__ float SIGMA_INV;
	//__shared__ float ALPHA;
	__shared__ float current_slice_image[SIZE_SUB_SLICE_y];
	__shared__ float voxel_size_x, voxel_size_y, voxel_size_z;
	__shared__ int center_x, center_y, center_z;
	__shared__ int range1, range2;
	__shared__ int num_x, num_y, num_z;
	__shared__ float *ptr_src_x, *ptr_src_y, *ptr_src_z, *ptr_dest_x, *ptr_dest_y, *ptr_dest_z;
	__shared__ float *ptr_FWHM;
	//__shared__ float *ptr_TOF_dist;
	__shared__ int num_lines;
	__shared__ float *image_ptr;
	__shared__ float *fp_value;

	if (threadIdx.x == 0){
		voxel_size_x = parameters_device->X_SAMP;
		voxel_size_y = parameters_device->Y_SAMP;
		voxel_size_z = parameters_device->Z_SAMP;
		center_x = parameters_device->X_INDEX_CENT;
		center_y = parameters_device->Y_INDEX_CENT;
		center_z = parameters_device->Z_INDEX_CENT;
		num_x = parameters_device->NUM_X;
		num_y = parameters_device->NUM_Y;
		num_z = parameters_device->NUM_Z;
		H_CENTER = parameters_device->H_CENTER_X;
		V_CENTER = parameters_device->V_CENTER;
		//SIGMA_INV = parameters_device->FWHM_sigma_inv;
		//ALPHA = parameters_device->FWHM_alpha;
		range1 = parameters_device->RANGE1;
		range2 = parameters_device->RANGE2;

		ptr_src_x = _events_y_dominant_uvm->src_x;
		ptr_src_y = _events_y_dominant_uvm->src_y;
		ptr_src_z = _events_y_dominant_uvm->src_z;
		ptr_dest_x = _events_y_dominant_uvm->dest_x;
		ptr_dest_y = _events_y_dominant_uvm->dest_y;
		ptr_dest_z = _events_y_dominant_uvm->dest_z;
		ptr_FWHM = _events_y_dominant_uvm->coeff;

		num_lines = _events_y_dominant_uvm->num_lines;
		/*
		ptr_src_x = device_event_y.src_x;
		ptr_src_y = device_event_y.src_y;
		ptr_src_z = device_event_y.src_z;
		ptr_dest_x = device_event_y.dest_x;
		ptr_dest_y = device_event_y.dest_y;
		ptr_dest_z = device_event_y.dest_z;
		num_lines = device_event_y.num_lines;
		*/

		//ptr_TOF_dist = events_y.TOF_dist;
		image_ptr = image;
		fp_value = fp_y;
	}
	__syncthreads();
	current_slice = blockIdx.x;
	spherical_voxel_filling_ratio = parameters_device->spherical_voxel_ratio;
	voxel_dim = voxel_size_x;
	intersection_y = (current_slice - center_y - 0.5)*voxel_size_y;
	for (sub_slice = 0; sub_slice<NUM_SUB_SLICE_y; sub_slice++){
		subslice_z0 = (num_z / NUM_SUB_SLICE_y)*sub_slice;
		subslice_z1 = subslice_z0 + SUB_SLICE_HEIGHT_y;//(num_z/NUM_SUB_SLICE)*(sub_slice+1);
		//load current image slice to shared memory
		for (int i = threadIdx.x; i<SIZE_SUB_SLICE_y; i = i + blockDim.x){
			current_slice_image[i] = image_ptr[num_x*num_y*(subslice_z0 + i / num_x) + num_x*current_slice + i%num_x];
		}
		__syncthreads();
		//loop through all the lines and compute the backprojection to this current slice
		for (event_index = threadIdx.x; event_index< num_lines; event_index = event_index + blockDim.x){
			//get event line parameters
			src_x = ptr_src_x[event_index];
			src_y = ptr_src_y[event_index];
			src_z = ptr_src_z[event_index];
			dest_x = ptr_dest_x[event_index];
			dest_y = ptr_dest_y[event_index];
			dest_z = ptr_dest_z[event_index];
			coeff = ptr_FWHM[event_index];

			//TOF_dist = ptr_TOF_dist[event_index];

			//only do the computation if current event line intersects current slice
			if (src_y<intersection_y - voxel_size_y && dest_y>intersection_y + voxel_size_y){
				dir_x = dest_x - src_x;
				dir_y = dest_y - src_y;
				dir_z = dest_z - src_z;
				//dist_squared = __fmaf_rn (dir_z,dir_z, __fmaf_rn(dir_y,dir_y,__fmul_rn(dir_x,dir_x) ) );
				dist_squared = dir_x*dir_x + dir_y*dir_y + dir_z*dir_z;
				dist = __fsqrt_rn(dist_squared);

				unit_dir_vector_x = dir_x / dist;
				unit_dir_vector_z = dir_z / dist;

				t = (intersection_y - src_y) / dir_y;
				intersection_x = t*dir_x + src_x;
				intersection_z = t*dir_z + src_z;

				intersection_index_x = (int)ceilf(intersection_x / voxel_size_x) + center_x;
				intersection_index_z = (int)ceilf(intersection_z / voxel_size_z) + center_z;

				//compute the distance to the TOF center
				//dist_to_TOF_center = (0.5f-t)*dist - TOF_dist;
				//t = TOF_ALPHA * __expf( - dist_to_TOF_center*dist_to_TOF_center*TOF_INV);

				forward_sum = 0.0f;

				for (inslice_z = intersection_index_z - range2; inslice_z <= intersection_index_z + range2; inslice_z++){
					for (inslice_x = intersection_index_x - range1; inslice_x <= intersection_index_x + range1; inslice_x++){
						if (inslice_x >= 0 && inslice_x<num_x && inslice_z >= subslice_z0 && inslice_z<subslice_z1){
							//projection_vector_x = (inslice_x - center_x - 0.5)*voxel_size_x - intersection_x;
							projection_vector_x = inslice_x*voxel_size_x - H_CENTER - intersection_x;
							//projection_vector_z = (inslice_z - center_z - 0.5)*voxel_size_z - intersection_z;
							projection_vector_z = inslice_z*voxel_size_z - V_CENTER - intersection_z;
							dot_product = projection_vector_x*unit_dir_vector_x + projection_vector_z*unit_dir_vector_z;
							point_line_distance_square = projection_vector_x*projection_vector_x + projection_vector_z*projection_vector_z - dot_product*dot_product;

							intersection_length_approx = 1.54 * voxel_dim*voxel_dim - 4 * point_line_distance_square;
							if (intersection_length_approx <= 0)
								intersection_length_approx = 0;
							intersection_length_approx = __fsqrt_rn(intersection_length_approx)*spherical_voxel_filling_ratio;

							//forward_sum += current_slice_image[(inslice_z-subslice_z0)*num_x+inslice_x] * __expf( - point_line_distance_square*SIGMA_INV ) * ALPHA * t;
							//forward_sum += current_slice_image[(inslice_z - subslice_z0)*num_x + inslice_x] * __expf(-point_line_distance_square*SIGMA_INV) * ALPHA;
							forward_sum += current_slice_image[(inslice_z - subslice_z0)*num_x + inslice_x] * intersection_length_approx;

							//atomicAdd(&fp_value[event_index],weight);

						}
					}
				}
				atomicAdd(&fp_value[event_index], forward_sum);
			}//intersects current slice
		}
	}
}


void
__global__ _bproj_atten_lst_cuda_x_kernel(float* image, float* fp_x, LST_LORs *_events_x_dominant_uvm, PARAMETERS_IN_DEVICE_t* parameters_device){
	int event_index, current_slice, sub_slice;
	float src_x, src_y, src_z, dest_x, dest_y, dest_z;
	float area, dist_from_center, geom;
	float coeff;
	float dir_x, dir_y, dir_z;
	//float TOF_dist;
	//float dist_to_TOF_center;
	float unit_dir_vector_y, unit_dir_vector_z;
	float projection_vector_y, projection_vector_z;
	float dot_product;
	float point_line_distance_square;
	float weight;
	float dist, dist_squared;
	float t;
	float solid_angle_ratio;
	float backward_ratio;
	float intersection_x, intersection_y, intersection_z;
	int  intersection_index_y, intersection_index_z;
	int  inslice_y, inslice_z;
	int subslice_z0, subslice_z1;

	__shared__ float H_CENTER;
	__shared__ float V_CENTER;
	__shared__ float SIGMA_INV;
	__shared__ float ALPHA;
	__shared__ float current_slice_image[SIZE_SUB_SLICE_x];
	__shared__ float voxel_size_x, voxel_size_y, voxel_size_z;
	__shared__ int center_x, center_y, center_z;
	__shared__ int range1, range2;
	__shared__ int num_x, num_y, num_z;
	__shared__ float *ptr_src_x, *ptr_src_y, *ptr_src_z, *ptr_dest_x, *ptr_dest_y, *ptr_dest_z;
	__shared__ float *ptr_FWHM;
	//__shared__ float *ptr_TOF_dist;
	__shared__ int num_lines;
	__shared__ float *image_ptr;
	__shared__ float *fp_value;

	if (threadIdx.x == 0){
		voxel_size_x = parameters_device->X_SAMP;
		voxel_size_y = parameters_device->Y_SAMP;
		voxel_size_z = parameters_device->Z_SAMP;
		center_x = parameters_device->X_INDEX_CENT;
		center_y = parameters_device->Y_INDEX_CENT;
		center_z = parameters_device->Z_INDEX_CENT;
		num_x = parameters_device->NUM_X;
		num_y = parameters_device->NUM_Y;
		num_z = parameters_device->NUM_Z;
		H_CENTER = parameters_device->H_CENTER_Y;
		V_CENTER = parameters_device->V_CENTER;
		SIGMA_INV = parameters_device->FWHM_sigma_inv;
		ALPHA = parameters_device->FWHM_alpha;
		range1 = parameters_device->RANGE1;
		range2 = parameters_device->RANGE2;

		ptr_src_x = _events_x_dominant_uvm->src_x;
		ptr_src_y = _events_x_dominant_uvm->src_y;
		ptr_src_z = _events_x_dominant_uvm->src_z;
		ptr_dest_x = _events_x_dominant_uvm->dest_x;
		ptr_dest_y = _events_x_dominant_uvm->dest_y;
		ptr_dest_z = _events_x_dominant_uvm->dest_z;
		ptr_FWHM = _events_x_dominant_uvm->coeff;
		
		num_lines = _events_x_dominant_uvm->num_lines;
		/*
		ptr_src_x = device_event_x.src_x;
		ptr_src_y = device_event_x.src_y;
		ptr_src_z = device_event_x.src_z;
		ptr_dest_x = device_event_x.dest_x;
		ptr_dest_y = device_event_x.dest_y;
		ptr_dest_z = device_event_x.dest_z;
		num_lines = device_event_x.num_lines;
		*/

		//ptr_TOF_dist = events_x.TOF_dist;
		image_ptr = image;
		fp_value = fp_x;
	}

	//initialize shared memory to zero
	for (int i = threadIdx.x; i<SIZE_SUB_SLICE_x; i = i + blockDim.x){
		current_slice_image[i] = 0.0f;
	}
	__syncthreads();

	current_slice = blockIdx.x;

	intersection_x = __fmul_rn((current_slice - center_x - 0.5f), voxel_size_x);

	for (sub_slice = 0; sub_slice<NUM_SUB_SLICE_x; sub_slice++){
		subslice_z0 = (num_z / NUM_SUB_SLICE_x)*sub_slice;
		subslice_z1 = subslice_z0 + SUB_SLICE_HEIGHT_x;//(num_z/NUM_SUB_SLICE)*(sub_slice+1);
		//loop through all the lines and compute the backprojection to this current slice
		for (event_index = threadIdx.x; event_index< num_lines; event_index = event_index + blockDim.x){
			//get event line parameters
			src_x = ptr_src_x[event_index];
			src_y = ptr_src_y[event_index];
			src_z = ptr_src_z[event_index];
			dest_x = ptr_dest_x[event_index];
			dest_y = ptr_dest_y[event_index];
			dest_z = ptr_dest_z[event_index];
			coeff = ptr_FWHM[event_index];

			
			//TOF_dist = ptr_TOF_dist[event_index];

			//only do the computation if current event line intersects current slice
			if (src_x<intersection_x - voxel_size_x && dest_x>intersection_x + voxel_size_x){

				
				dir_x = dest_x - src_x;
				dir_y = dest_y - src_y;
				dir_z = dest_z - src_z;
				//dist_squared = __fmaf_rn (dir_z,dir_z, __fmaf_rn(dir_y,dir_y,__fmul_rn(dir_x,dir_x) ) );
				dist_squared = dir_x*dir_x + dir_y*dir_y + dir_z*dir_z;
				dist = __fsqrt_rn(dist_squared);

				area = 0.5 * fabsf(src_x * dest_y - dest_x * src_y);
				dist_from_center = 2 * area / dist;
				geom = (0.00000468*dist_from_center*dist_from_center - 0.0000112*dist_from_center + 0.9321);
				unit_dir_vector_y = dir_y / dist;
				unit_dir_vector_z = dir_z / dist;

				t = (intersection_x - src_x) / dir_x;
				intersection_y = t*dir_y + src_y;
				intersection_z = t*dir_z + src_z;

				backward_ratio = fp_value[event_index];

				intersection_index_y = (int)ceilf(intersection_y / voxel_size_y) + center_y;
				intersection_index_z = (int)ceilf(intersection_z / voxel_size_z) + center_z;

				//compute the solid angle ratio w.r.t. the location of the intersection
				//solid_angle_ratio = 1 + 2 * fabsf(t - 0.5f);
				//solid_angle_ratio = solid_angle_ratio*solid_angle_ratio*dist*dist;
				//solid_angle_ratio = 1 + 12 * (t - 0.5f) * (t - 0.5f);
				solid_angle_ratio = 1 + fabsf(24 * (t - 0.5f) * (t - 0.5f) * (t - 0.5f));
				solid_angle_ratio = 1*dist*dist;

				//compute the distance to the TOF center
				//dist_to_TOF_center = (0.5f-t)*dist - TOF_dist;
				//t = TOF_ALPHA * __expf( - dist_to_TOF_center*dist_to_TOF_center*TOF_INV);

				for (inslice_z = intersection_index_z - range2; inslice_z <= intersection_index_z + range2; inslice_z++){
					for (inslice_y = intersection_index_y - range1; inslice_y <= intersection_index_y + range1; inslice_y++){
						if (inslice_y >= 0 && inslice_y<num_y && inslice_z >= subslice_z0 && inslice_z<subslice_z1){
							//projection_vector_y = (inslice_y - center_y - 0.5)*voxel_size_y - intersection_y;
							projection_vector_y = inslice_y*voxel_size_y - H_CENTER - intersection_y;
							//projection_vector_z = (inslice_z - center_z - 0.5)*voxel_size_z - intersection_z;
							projection_vector_z = inslice_z*voxel_size_z - V_CENTER - intersection_z;
							dot_product = projection_vector_y*unit_dir_vector_y + projection_vector_z*unit_dir_vector_z;
							point_line_distance_square = projection_vector_y*projection_vector_y + projection_vector_z*projection_vector_z - dot_product*dot_product;
							//weight = t * backward_ratio * __expf( - point_line_distance_square*SIGMA_INV );
							weight = __expf(-point_line_distance_square*SIGMA_INV - backward_ratio) * ALPHA * GLOBAL_SCALE ;
							atomicAdd(&current_slice_image[(inslice_z - subslice_z0)*num_y + inslice_y], weight / (solid_angle_ratio) );
						}
					}
				}
			}//intersects current slice
		}
		__syncthreads();
		//write the current sub slice from shared memory back to global memory using atomic operation
		//Since this write is not coalesced,  think about optimization techniques
		for (int j = threadIdx.x; j<SIZE_SUB_SLICE_x; j = j + blockDim.x){
			//atomicAdd(&image_ptr[ (j+sub_slice*SIZE_SUB_SLICE)*image_width+current_slice], current_slice_image[j]);
			atomicAdd(&image_ptr[(j + sub_slice*SIZE_SUB_SLICE_x)*num_x + current_slice], current_slice_image[j]);
			current_slice_image[j] = 0.0f;
		}
		__syncthreads();
	}
}

void
__global__ _bproj_atten_lst_cuda_y_kernel(float* image, float* fp_y, LST_LORs *_events_y_dominant_uvm, PARAMETERS_IN_DEVICE_t* parameters_device){
	int event_index, current_slice, sub_slice;
	float src_x, src_y, src_z, dest_x, dest_y, dest_z;
	float area, dist_from_center, geom;
	float coeff;
	float dir_x, dir_y, dir_z;
	//float TOF_dist;
	//float dist_to_TOF_center;
	float unit_dir_vector_x, unit_dir_vector_z;
	float projection_vector_x, projection_vector_z;
	float dot_product;
	float point_line_distance_square;
	float weight;
	float dist, dist_squared;
	float t;
	float solid_angle_ratio;
	float backward_ratio;
	float intersection_x, intersection_y, intersection_z;
	int intersection_index_x, intersection_index_z;
	int inslice_x, inslice_z;
	int subslice_z0, subslice_z1;

	__shared__ float H_CENTER;
	__shared__ float V_CENTER;
	__shared__ float SIGMA_INV;
	__shared__ float ALPHA;
	__shared__ float current_slice_image[SIZE_SUB_SLICE_y];
	__shared__ float voxel_size_x, voxel_size_y, voxel_size_z;
	__shared__ int center_x, center_y, center_z;
	__shared__ int range1, range2;
	__shared__ int num_x, num_y, num_z;
	__shared__ float *ptr_src_x, *ptr_src_y, *ptr_src_z, *ptr_dest_x, *ptr_dest_y, *ptr_dest_z;
	__shared__ float *ptr_FWHM;
	//__shared__ float *ptr_TOF_dist;
	__shared__ int num_lines;
	__shared__ float *image_ptr;
	__shared__ float *fp_value;

	if (threadIdx.x == 0){
		voxel_size_x = parameters_device->X_SAMP;
		voxel_size_y = parameters_device->Y_SAMP;
		voxel_size_z = parameters_device->Z_SAMP;
		center_x = parameters_device->X_INDEX_CENT;
		center_y = parameters_device->Y_INDEX_CENT;
		center_z = parameters_device->Z_INDEX_CENT;
		num_x = parameters_device->NUM_X;
		num_y = parameters_device->NUM_Y;
		num_z = parameters_device->NUM_Z;
		H_CENTER = parameters_device->H_CENTER_X;
		V_CENTER = parameters_device->V_CENTER;
		SIGMA_INV = parameters_device->FWHM_sigma_inv;
		ALPHA = parameters_device->FWHM_alpha;
		range1 = parameters_device->RANGE1;
		range2 = parameters_device->RANGE2;

		ptr_src_x = _events_y_dominant_uvm->src_x;
		ptr_src_y = _events_y_dominant_uvm->src_y;
		ptr_src_z = _events_y_dominant_uvm->src_z;
		ptr_dest_x = _events_y_dominant_uvm->dest_x;
		ptr_dest_y = _events_y_dominant_uvm->dest_y;
		ptr_dest_z = _events_y_dominant_uvm->dest_z;
		ptr_FWHM = _events_y_dominant_uvm->coeff;
		
		num_lines = _events_y_dominant_uvm->num_lines;
		/*
		ptr_src_x = device_event_y.src_x;
		ptr_src_y = device_event_y.src_y;
		ptr_src_z = device_event_y.src_z;
		ptr_dest_x = device_event_y.dest_x;
		ptr_dest_y = device_event_y.dest_y;
		ptr_dest_z = device_event_y.dest_z;
		num_lines = device_event_y.num_lines;
		*/

		//ptr_TOF_dist = events_y.TOF_dist;
		image_ptr = image;
		fp_value = fp_y;
	}

	for (int i = threadIdx.x; i<SIZE_SUB_SLICE_y; i = i + blockDim.x){
		current_slice_image[i] = 0.0f;
	}
	__syncthreads();

	current_slice = blockIdx.x;

	intersection_y = __fmul_rn((current_slice - center_y - 0.5f), voxel_size_y);
	for (sub_slice = 0; sub_slice<NUM_SUB_SLICE_y; sub_slice++){
		subslice_z0 = (num_z / NUM_SUB_SLICE_y)*sub_slice;
		subslice_z1 = subslice_z0 + SUB_SLICE_HEIGHT_y;//(num_z/NUM_SUB_SLICE)*(sub_slice+1);
		//loop through all the lines and compute the backprojection to this current slice
		for (event_index = threadIdx.x; event_index< num_lines; event_index = event_index + blockDim.x){
			//get event line parameters
			src_x = ptr_src_x[event_index];
			src_y = ptr_src_y[event_index];
			src_z = ptr_src_z[event_index];
			dest_x = ptr_dest_x[event_index];
			dest_y = ptr_dest_y[event_index];
			dest_z = ptr_dest_z[event_index];
			coeff = ptr_FWHM[event_index];
		
			//TOF_dist = ptr_TOF_dist[event_index];

			//only do the computation if current event line intersects current slice
			if (src_y<intersection_y - voxel_size_y && dest_y>intersection_y + voxel_size_y){
				dir_x = dest_x - src_x;
				dir_y = dest_y - src_y;
				dir_z = dest_z - src_z;
				//dist_squared = __fmaf_rn (dir_z,dir_z, __fmaf_rn(dir_y,dir_y,__fmul_rn(dir_x,dir_x) ) );
				dist_squared = dir_x*dir_x + dir_y*dir_y + dir_z*dir_z;
				dist = __fsqrt_rn(dist_squared);

				area = 0.5 * fabsf(src_x * dest_y - dest_x * src_y);
				dist_from_center = 2 * area / dist;
				geom = (0.00000468*dist_from_center*dist_from_center - 0.0000112*dist_from_center + 0.9321);

				unit_dir_vector_x = dir_x / dist;
				unit_dir_vector_z = dir_z / dist;

				t = (intersection_y - src_y) / dir_y;
				intersection_x = t*dir_x + src_x;
				intersection_z = t*dir_z + src_z;

				backward_ratio = fp_value[event_index];

				intersection_index_x = (int)ceilf(intersection_x / voxel_size_x) + center_x;
				intersection_index_z = (int)ceilf(intersection_z / voxel_size_z) + center_z;

				//compute the solid angle ratio w.r.t. the location of the intersection
				//solid_angle_ratio = 1 + 2 * fabsf(t - 0.5f);
				//solid_angle_ratio = solid_angle_ratio*solid_angle_ratio*dist*dist;

				//solid_angle_ratio = 1 + 12 * (t - 0.5f) * (t - 0.5f);
				solid_angle_ratio = 1 + fabsf(24 * (t - 0.5f) * (t - 0.5f) * (t - 0.5f));
				solid_angle_ratio = 1*dist*dist;

				//compute the distance to the TOF center
				//dist_to_TOF_center = (0.5f-t)*dist - TOF_dist;
				//t = TOF_ALPHA * __expf( - dist_to_TOF_center*dist_to_TOF_center*TOF_INV);

				for (inslice_z = intersection_index_z - range2; inslice_z <= intersection_index_z + range2; inslice_z++){
					for (inslice_x = intersection_index_x - range1; inslice_x <= intersection_index_x + range1; inslice_x++){
						if (inslice_x >= 0 && inslice_x<num_x && inslice_z >= subslice_z0 && inslice_z<subslice_z1){
							//projection_vector_x = (inslice_x - center_x - 0.5)*voxel_size_x - intersection_x;
							projection_vector_x = inslice_x*voxel_size_x - H_CENTER - intersection_x;
							//projection_vector_z = (inslice_z - center_z - 0.5)*voxel_size_z - intersection_z;
							projection_vector_z = inslice_z*voxel_size_z - V_CENTER - intersection_z;
							dot_product = projection_vector_x*unit_dir_vector_x + projection_vector_z*unit_dir_vector_z;
							point_line_distance_square = projection_vector_x*projection_vector_x + projection_vector_z*projection_vector_z - dot_product*dot_product;
							//weight = t * backward_ratio * __expf( - point_line_distance_square*SIGMA_INV );
							weight = __expf(-point_line_distance_square*SIGMA_INV - backward_ratio) * ALPHA  * GLOBAL_SCALE ;
							atomicAdd(&current_slice_image[(inslice_z - subslice_z0)*num_x + inslice_x], weight / (solid_angle_ratio));
						}
					}
				}
			}//intersects current slice
		}
		__syncthreads();
		//write the current sub slice from shared memory back to global memory using atomic operation
		//Since this write is not coalesced,  think about optimization techniques
		for (int j = threadIdx.x; j<SIZE_SUB_SLICE_y; j = j + blockDim.x){
			atomicAdd(&image_ptr[num_x*num_y*(subslice_z0 + j / num_x) + num_x*current_slice + j%num_x], current_slice_image[j]);
			//image_ptr[ image_width*image_height*(subslice_z0+j/image_width) + image_width*current_slice + j%image_width ] = current_slice_image[j];
			current_slice_image[j] = 0.0f;
		}
		__syncthreads();
	}
}

void
__global__ _bproj_atten_lst_cuda_x_kernel_norm(float* image, float* fp_x, LST_LORs *_events_x_dominant_uvm, PARAMETERS_IN_DEVICE_t* parameters_device, float* fp2_x){
	int event_index, current_slice, sub_slice;
	float src_x, src_y, src_z, dest_x, dest_y, dest_z;
	int coeff;
	float dir_x, dir_y, dir_z;
	//float TOF_dist;
	//float dist_to_TOF_center;
	float unit_dir_vector_y, unit_dir_vector_z;
	float projection_vector_y, projection_vector_z;
	float dot_product;
	float point_line_distance_square;
	float weight;
	float dist, dist_squared;
	float t;
	float solid_angle_ratio;
	float backward_ratio;
	float intersection_x, intersection_y, intersection_z;
	int  intersection_index_y, intersection_index_z;
	int  inslice_y, inslice_z;
	int subslice_z0, subslice_z1;

	__shared__ float H_CENTER;
	__shared__ float V_CENTER;
	__shared__ float SIGMA_INV;
	__shared__ float ALPHA;
	__shared__ float current_slice_image[SIZE_SUB_SLICE_x];
	__shared__ float voxel_size_x, voxel_size_y, voxel_size_z;
	__shared__ int center_x, center_y, center_z;
	__shared__ int range1, range2;
	__shared__ int num_x, num_y, num_z;
	__shared__ float *ptr_src_x, *ptr_src_y, *ptr_src_z, *ptr_dest_x, *ptr_dest_y, *ptr_dest_z;
	__shared__ float *ptr_FWHM; 
	//__shared__ float *ptr_TOF_dist;
	__shared__ int num_lines;
	__shared__ float *image_ptr;
	__shared__ float *fp_value;
	__shared__ float *fp2_value;

	if (threadIdx.x == 0){
		voxel_size_x = parameters_device->X_SAMP;
		voxel_size_y = parameters_device->Y_SAMP;
		voxel_size_z = parameters_device->Z_SAMP;
		center_x = parameters_device->X_INDEX_CENT;
		center_y = parameters_device->Y_INDEX_CENT;
		center_z = parameters_device->Z_INDEX_CENT;
		num_x = parameters_device->NUM_X;
		num_y = parameters_device->NUM_Y;
		num_z = parameters_device->NUM_Z;
		H_CENTER = parameters_device->H_CENTER_Y;
		V_CENTER = parameters_device->V_CENTER;
		SIGMA_INV = parameters_device->FWHM_sigma_inv;
		ALPHA = parameters_device->FWHM_alpha;
		range1 = parameters_device->RANGE1;
		range2 = parameters_device->RANGE2;

		ptr_src_x = _events_x_dominant_uvm->src_x;
		ptr_src_y = _events_x_dominant_uvm->src_y;
		ptr_src_z = _events_x_dominant_uvm->src_z;
		ptr_dest_x = _events_x_dominant_uvm->dest_x;
		ptr_dest_y = _events_x_dominant_uvm->dest_y;
		ptr_dest_z = _events_x_dominant_uvm->dest_z;
		ptr_FWHM = _events_x_dominant_uvm->coeff;

		num_lines = _events_x_dominant_uvm->num_lines;
		/*
		ptr_src_x = device_event_x.src_x;
		ptr_src_y = device_event_x.src_y;
		ptr_src_z = device_event_x.src_z;
		ptr_dest_x = device_event_x.dest_x;
		ptr_dest_y = device_event_x.dest_y;
		ptr_dest_z = device_event_x.dest_z;
		num_lines = device_event_x.num_lines;
		*/

		//ptr_TOF_dist = events_x.TOF_dist;
		image_ptr = image;
		fp_value = fp_x;
		fp2_value = fp2_x;
	}

	//initialize shared memory to zero
	for (int i = threadIdx.x; i<SIZE_SUB_SLICE_x; i = i + blockDim.x){
		current_slice_image[i] = 0.0f;
	}
	__syncthreads();

	current_slice = blockIdx.x;

	intersection_x = __fmul_rn((current_slice - center_x - 0.5f), voxel_size_x);

	for (sub_slice = 0; sub_slice<NUM_SUB_SLICE_x; sub_slice++){
		subslice_z0 = (num_z / NUM_SUB_SLICE_x)*sub_slice;
		subslice_z1 = subslice_z0 + SUB_SLICE_HEIGHT_x;//(num_z/NUM_SUB_SLICE)*(sub_slice+1);
		//loop through all the lines and compute the backprojection to this current slice
		for (event_index = threadIdx.x; event_index< num_lines; event_index = event_index + blockDim.x){
			//get event line parameters
			src_x = ptr_src_x[event_index];
			src_y = ptr_src_y[event_index];
			src_z = ptr_src_z[event_index];
			dest_x = ptr_dest_x[event_index];
			dest_y = ptr_dest_y[event_index];
			dest_z = ptr_dest_z[event_index];
			coeff = ptr_FWHM[event_index];

			//TOF_dist = ptr_TOF_dist[event_index];

			//only do the computation if current event line intersects current slice
			if (src_x<intersection_x - voxel_size_x && dest_x>intersection_x + voxel_size_x){
				dir_x = dest_x - src_x;
				dir_y = dest_y - src_y;
				dir_z = dest_z - src_z;
				//dist_squared = __fmaf_rn (dir_z,dir_z, __fmaf_rn(dir_y,dir_y,__fmul_rn(dir_x,dir_x) ) );
				dist_squared = dir_x*dir_x + dir_y*dir_y + dir_z*dir_z;
				dist = __fsqrt_rn(dist_squared);

				unit_dir_vector_y = dir_y / dist;
				unit_dir_vector_z = dir_z / dist;

				t = (intersection_x - src_x) / dir_x;
				intersection_y = t*dir_y + src_y;
				intersection_z = t*dir_z + src_z;

				float normfact = 0;
				backward_ratio = fp_value[event_index];
				if (fp2_value[event_index] < 0.01){
					normfact = 0;
				}
				else{
					normfact = coeff/(fp2_value[event_index]);
					//normfact = coeff ;
				}

				intersection_index_y = (int)ceilf(intersection_y / voxel_size_y) + center_y;
				intersection_index_z = (int)ceilf(intersection_z / voxel_size_z) + center_z;

				//compute the solid angle ratio w.r.t. the location of the intersection
				solid_angle_ratio = 1 + 2 * fabsf(t - 0.5f);
				solid_angle_ratio = solid_angle_ratio*solid_angle_ratio*dist*dist;

				//compute the distance to the TOF center
				//dist_to_TOF_center = (0.5f-t)*dist - TOF_dist;
				//t = TOF_ALPHA * __expf( - dist_to_TOF_center*dist_to_TOF_center*TOF_INV);

				for (inslice_z = intersection_index_z - range2; inslice_z <= intersection_index_z + range2; inslice_z++){
					for (inslice_y = intersection_index_y - range1; inslice_y <= intersection_index_y + range1; inslice_y++){
						if (inslice_y >= 0 && inslice_y<num_y && inslice_z >= subslice_z0 && inslice_z<subslice_z1){
							//projection_vector_y = (inslice_y - center_y - 0.5)*voxel_size_y - intersection_y;
							projection_vector_y = inslice_y*voxel_size_y - H_CENTER - intersection_y;
							//projection_vector_z = (inslice_z - center_z - 0.5)*voxel_size_z - intersection_z;
							projection_vector_z = inslice_z*voxel_size_z - V_CENTER - intersection_z;
							dot_product = projection_vector_y*unit_dir_vector_y + projection_vector_z*unit_dir_vector_z;
							point_line_distance_square = projection_vector_y*projection_vector_y + projection_vector_z*projection_vector_z - dot_product*dot_product;
							//weight = t * backward_ratio * __expf( - point_line_distance_square*SIGMA_INV );
							weight = __expf(-point_line_distance_square*SIGMA_INV - backward_ratio) * ALPHA * normfact;
							atomicAdd(&current_slice_image[(inslice_z - subslice_z0)*num_y + inslice_y], weight / solid_angle_ratio);
						}
					}
				}
			}//intersects current slice
		}
		__syncthreads();
		//write the current sub slice from shared memory back to global memory using atomic operation
		//Since this write is not coalesced,  think about optimization techniques
		for (int j = threadIdx.x; j<SIZE_SUB_SLICE_x; j = j + blockDim.x){
			//atomicAdd(&image_ptr[ (j+sub_slice*SIZE_SUB_SLICE)*image_width+current_slice], current_slice_image[j]);
			atomicAdd(&image_ptr[(j + sub_slice*SIZE_SUB_SLICE_x)*num_x + current_slice], current_slice_image[j]);
			current_slice_image[j] = 0.0f;
		}
		__syncthreads();
	}
}

void
__global__ _bproj_atten_lst_cuda_y_kernel_norm(float* image, float* fp_y, LST_LORs *_events_y_dominant_uvm, PARAMETERS_IN_DEVICE_t* parameters_device, float* fp2_y){
	int event_index, current_slice, sub_slice;
	float src_x, src_y, src_z, dest_x, dest_y, dest_z;
		int coeff;
	float dir_x, dir_y, dir_z;
	//float TOF_dist;
	//float dist_to_TOF_center;
	float unit_dir_vector_x, unit_dir_vector_z;
	float projection_vector_x, projection_vector_z;
	float dot_product;
	float point_line_distance_square;
	float weight;
	float dist, dist_squared;
	float t;
	float solid_angle_ratio;
	float backward_ratio;
	float intersection_x, intersection_y, intersection_z;
	int intersection_index_x, intersection_index_z;
	int inslice_x, inslice_z;
	int subslice_z0, subslice_z1;

	__shared__ float H_CENTER;
	__shared__ float V_CENTER;
	__shared__ float SIGMA_INV;
	__shared__ float ALPHA;
	__shared__ float current_slice_image[SIZE_SUB_SLICE_y];
	__shared__ float voxel_size_x, voxel_size_y, voxel_size_z;
	__shared__ int center_x, center_y, center_z;
	__shared__ int range1, range2;
	__shared__ int num_x, num_y, num_z;
	__shared__ float *ptr_src_x, *ptr_src_y, *ptr_src_z, *ptr_dest_x, *ptr_dest_y, *ptr_dest_z;
	__shared__ float *ptr_FWHM;
	//__shared__ float *ptr_TOF_dist;
	__shared__ int num_lines;
	__shared__ float *image_ptr;
	__shared__ float *fp_value;
	__shared__ float *fp2_value;

	if (threadIdx.x == 0){
		voxel_size_x = parameters_device->X_SAMP;
		voxel_size_y = parameters_device->Y_SAMP;
		voxel_size_z = parameters_device->Z_SAMP;
		center_x = parameters_device->X_INDEX_CENT;
		center_y = parameters_device->Y_INDEX_CENT;
		center_z = parameters_device->Z_INDEX_CENT;
		num_x = parameters_device->NUM_X;
		num_y = parameters_device->NUM_Y;
		num_z = parameters_device->NUM_Z;
		H_CENTER = parameters_device->H_CENTER_X;
		V_CENTER = parameters_device->V_CENTER;
		SIGMA_INV = parameters_device->FWHM_sigma_inv;
		ALPHA = parameters_device->FWHM_alpha;
		range1 = parameters_device->RANGE1;
		range2 = parameters_device->RANGE2;

		ptr_src_x = _events_y_dominant_uvm->src_x;
		ptr_src_y = _events_y_dominant_uvm->src_y;
		ptr_src_z = _events_y_dominant_uvm->src_z;
		ptr_dest_x = _events_y_dominant_uvm->dest_x;
		ptr_dest_y = _events_y_dominant_uvm->dest_y;
		ptr_dest_z = _events_y_dominant_uvm->dest_z;
		ptr_FWHM = _events_y_dominant_uvm->coeff;

		num_lines = _events_y_dominant_uvm->num_lines;
		/*
		ptr_src_x = device_event_y.src_x;
		ptr_src_y = device_event_y.src_y;
		ptr_src_z = device_event_y.src_z;
		ptr_dest_x = device_event_y.dest_x;
		ptr_dest_y = device_event_y.dest_y;
		ptr_dest_z = device_event_y.dest_z;
		num_lines = device_event_y.num_lines;
		*/

		//ptr_TOF_dist = events_y.TOF_dist;
		image_ptr = image;
		fp_value = fp_y;
		fp2_value = fp2_y;
	}

	for (int i = threadIdx.x; i<SIZE_SUB_SLICE_y; i = i + blockDim.x){
		current_slice_image[i] = 0.0f;
	}
	__syncthreads();

	current_slice = blockIdx.x;

	intersection_y = __fmul_rn((current_slice - center_y - 0.5f), voxel_size_y);
	for (sub_slice = 0; sub_slice<NUM_SUB_SLICE_y; sub_slice++){
		subslice_z0 = (num_z / NUM_SUB_SLICE_y)*sub_slice;
		subslice_z1 = subslice_z0 + SUB_SLICE_HEIGHT_y;//(num_z/NUM_SUB_SLICE)*(sub_slice+1);
		//loop through all the lines and compute the backprojection to this current slice
		for (event_index = threadIdx.x; event_index< num_lines; event_index = event_index + blockDim.x){
			//get event line parameters
			src_x = ptr_src_x[event_index];
			src_y = ptr_src_y[event_index];
			src_z = ptr_src_z[event_index];
			dest_x = ptr_dest_x[event_index];
			dest_y = ptr_dest_y[event_index];
			dest_z = ptr_dest_z[event_index];
			coeff = ptr_FWHM[event_index];

			//TOF_dist = ptr_TOF_dist[event_index];

			//only do the computation if current event line intersects current slice
			if (src_y<intersection_y - voxel_size_y && dest_y>intersection_y + voxel_size_y){
				dir_x = dest_x - src_x;
				dir_y = dest_y - src_y;
				dir_z = dest_z - src_z;
				//dist_squared = __fmaf_rn (dir_z,dir_z, __fmaf_rn(dir_y,dir_y,__fmul_rn(dir_x,dir_x) ) );
				dist_squared = dir_x*dir_x + dir_y*dir_y + dir_z*dir_z;
				dist = __fsqrt_rn(dist_squared);

				unit_dir_vector_x = dir_x / dist;
				unit_dir_vector_z = dir_z / dist;

				t = (intersection_y - src_y) / dir_y;
				intersection_x = t*dir_x + src_x;
				intersection_z = t*dir_z + src_z;

				float normfact = 0;
				backward_ratio = fp_value[event_index];
				if (fp2_value[event_index] < 0.01){
					normfact = 0;
				}
				else{
					normfact = coeff/(fp2_value[event_index]);
					//normfact = coeff ;
				}


				intersection_index_x = (int)ceilf(intersection_x / voxel_size_x) + center_x;
				intersection_index_z = (int)ceilf(intersection_z / voxel_size_z) + center_z;

				//compute the solid angle ratio w.r.t. the location of the intersection
				solid_angle_ratio = 1 + 2 * fabsf(t - 0.5f);
				solid_angle_ratio = solid_angle_ratio*solid_angle_ratio*dist*dist;

				//compute the distance to the TOF center
				//dist_to_TOF_center = (0.5f-t)*dist - TOF_dist;
				//t = TOF_ALPHA * __expf( - dist_to_TOF_center*dist_to_TOF_center*TOF_INV);

				for (inslice_z = intersection_index_z - range2; inslice_z <= intersection_index_z + range2; inslice_z++){
					for (inslice_x = intersection_index_x - range1; inslice_x <= intersection_index_x + range1; inslice_x++){
						if (inslice_x >= 0 && inslice_x<num_x && inslice_z >= subslice_z0 && inslice_z<subslice_z1){
							//projection_vector_x = (inslice_x - center_x - 0.5)*voxel_size_x - intersection_x;
							projection_vector_x = inslice_x*voxel_size_x - H_CENTER - intersection_x;
							//projection_vector_z = (inslice_z - center_z - 0.5)*voxel_size_z - intersection_z;
							projection_vector_z = inslice_z*voxel_size_z - V_CENTER - intersection_z;
							dot_product = projection_vector_x*unit_dir_vector_x + projection_vector_z*unit_dir_vector_z;
							point_line_distance_square = projection_vector_x*projection_vector_x + projection_vector_z*projection_vector_z - dot_product*dot_product;
							//weight = t * backward_ratio * __expf( - point_line_distance_square*SIGMA_INV );
							weight = __expf(-point_line_distance_square*SIGMA_INV - backward_ratio) * ALPHA * normfact;
							atomicAdd(&current_slice_image[(inslice_z - subslice_z0)*num_x + inslice_x], weight / solid_angle_ratio);
						}
					}
				}
			}//intersects current slice
		}
		__syncthreads();
		//write the current sub slice from shared memory back to global memory using atomic operation
		//Since this write is not coalesced,  think about optimization techniques
		for (int j = threadIdx.x; j<SIZE_SUB_SLICE_y; j = j + blockDim.x){
			atomicAdd(&image_ptr[num_x*num_y*(subslice_z0 + j / num_x) + num_x*current_slice + j%num_x], current_slice_image[j]);
			//image_ptr[ image_width*image_height*(subslice_z0+j/image_width) + image_width*current_slice + j%image_width ] = current_slice_image[j];
			current_slice_image[j] = 0.0f;
		}
		__syncthreads();
	}
}



/*

extern __constant__ PARAMETERS_IN_DEVICE_t parameters_device;
extern __constant__ LST_LORs device_event_x;
extern __constant__ LST_LORs device_event_y;

extern void __global__ setImageToZero_kernel( float* image);
extern void __global__ setForwardProjectionValueToZero_kernel(float* device_fp_ptr, int fp_size);
extern void __global__ update_image_kernel(float* device_image_to_be_updated, float* device_update_factor, float* device_sensitivity_image);

extern void __global__ _fproj_lst_cuda_x_kernel(float* image, float* fp_x, LST_LORs *_events_x_dominant_uvm);
extern void __global__ _fproj_lst_cuda_y_kernel(float* image, float* fp_y, LST_LORs *_events_y_dominant_uvm);
extern void __global__ _b_ratio_proj_lst_cuda_x_kernel(float* image, float* fp_x);
extern void __global__ _b_ratio_proj_lst_cuda_y_kernel(float* image, float* fp_y);

extern void __global__ _TOF_fproj_lst_cuda_x_kernel(float* image, float* fp_x, LST_LORs *_events_x_dominant_uvm);
extern void __global__ _TOF_fproj_lst_cuda_y_kernel(float* image, float* fp_y, LST_LORs *_events_y_dominant_uvm);
extern void __global__ _TOF_b_ratio_proj_lst_cuda_x_kernel(float* image, float* fp_x, LST_LORs *_events_x_dominant_uvm);
extern void __global__ _TOF_b_ratio_proj_lst_cuda_y_kernel(float* image, float* fp_y, LST_LORs *_events_y_dominant_uvm);

*/




cuda_em_recon::cuda_em_recon(parameters_t p, string output_prefix, string output_path){
	
	_Initialize_host_parameters(p);
		
	_num_gpu_start = p.num_gpu_start;
	_num_gpu_end = p.num_gpu_end;

	if (p.TOF_on){
		_TOF_mode = TOF;
	}
	else{
		_TOF_mode = NONTOF;
	}

	if (p.export_negative_log_likelihood == 1){
		export_likelihood_flag = 1;
		printf("Exporting negative log likelihood values to file.\n");
	}
	else{
		export_likelihood_flag = 0;
		printf("Not Exporting negative log likelihood values to file.\n");
	}

	if (p.attenuation_correction_fp == 1){
		atten_flag_fp = 1;
		printf("Forward Projection system matrix has attenuation coefficients.\n");
	}
	else{
		atten_flag_fp = 0;
		printf("Forward Projection system matrix doesn't have attenuation coefficients.\n");
	}
	
	if (p.attenuation_correction_bp == 1){
		atten_flag_bp = 1;
		printf("Backward Projection system matrix has attenuation coefficients.\n");
	}
	else{
		atten_flag_bp = 0;
		printf("Backward Projection system matrix doesn't have attenuation coefficients.\n");
	}


	_output_path = output_path;
	_output_filename_prefix = output_prefix;
}

cuda_em_recon::~cuda_em_recon(){

}

void
cuda_em_recon::_Initialize_host_parameters(parameters_t p){
	
	float TOF_sigma;
	float LOR_FWHM;

	parameters_host.spherical_voxel_ratio = p.spherical_voxel_ratio;

	parameters_host.NUM_X = p.NUM_X;
	parameters_host.NUM_Y = p.NUM_Y;
	parameters_host.NUM_Z = p.NUM_Z;
	parameters_host.NUM_XY = p.NUM_X * p.NUM_Y;
	parameters_host.NUM_XZ = p.NUM_X * p.NUM_Z;
	parameters_host.NUM_YZ = p.NUM_Y * p.NUM_Z;
	parameters_host.NUM_XYZ = p.NUM_X * p.NUM_Y * p.NUM_Z;
	parameters_host.X_OFFSET = p.X_OFFSET;
	parameters_host.Y_OFFSET = p.Y_OFFSET;
	parameters_host.Z_OFFSET = p.Z_OFFSET;
	parameters_host.X_SAMP = p.X_SAMP;
	parameters_host.Y_SAMP = p.Y_SAMP;
	parameters_host.Z_SAMP = p.Z_SAMP;

	parameters_host.X_INDEX_CENT = (p.NUM_X - 1) / 2 - p.X_OFFSET / p.X_SAMP;
	parameters_host.Y_INDEX_CENT = (p.NUM_Y - 1) / 2 - p.Y_OFFSET / p.Y_SAMP;
	parameters_host.Z_INDEX_CENT = (p.NUM_Z - 1) / 2 - p.Z_OFFSET / p.Z_SAMP;

	parameters_host.H_CENTER_X = (parameters_host.X_INDEX_CENT + 0.5)*p.X_SAMP;
	parameters_host.H_CENTER_Y = (parameters_host.Y_INDEX_CENT + 0.5)*p.Y_SAMP;
	parameters_host.V_CENTER = (parameters_host.Z_INDEX_CENT + 0.5)*p.Z_SAMP;


	//printf("X_INDEX_CENT = %d\n", parameters_host.X_INDEX_CENT);
	//printf("Y_INDEX_CENT = %d\n", parameters_host.Y_INDEX_CENT);
	//printf("Z_INDEX_CENT = %d\n", parameters_host.Z_INDEX_CENT);

	//printf("H_CENTER_X = %f\n", parameters_host.H_CENTER_X);
	//printf("H_CENTER_Y = %f\n", parameters_host.H_CENTER_Y);
	//printf("V_CENTER = %f\n", parameters_host.V_CENTER);

	TOF_sigma = p.TOF_res*0.15f / 2.354820045f;
	parameters_host.TOF_inv = 1 / (2 * TOF_sigma*TOF_sigma);;
	parameters_host.TOF_alpha = 0.39894228f / TOF_sigma;

	printf("\n...TOF_sigma = %f, TOF_alpha = %f\n", parameters_host.TOF_inv, parameters_host.TOF_alpha);
	
	parameters_host.FWHM = p.FWHM;
	parameters_host.FWHM_ss = p.FWHM_SS;
	parameters_host.FWHM_is = p.FWHM_IS;
	parameters_host.FWHM_ii = p.FWHM_II;

	LOR_FWHM = p.FWHM;
	parameters_host.FWHM_sigma_inv = 4 * log(2.0) / (LOR_FWHM*LOR_FWHM);
	parameters_host.FWHM_alpha = 2 * sqrt(log(2) / M_PI) / LOR_FWHM;
	//printf("...FWHM_sigma_inv2 = %f, FWHM_alpha = %f\n", parameters_host.FWHM_sigma_inv, parameters_host.FWHM_alpha);

	LOR_FWHM = p.FWHM_SS;
	parameters_host.FWHM_sigma_inv_ss = 4*log(2.0)/(LOR_FWHM*LOR_FWHM);
	parameters_host.FWHM_alpha_ss = 2*sqrt(log(2)/M_PI)/LOR_FWHM;
	//printf("...FWHM_sigma_inv2_ss = %f, FWHM_alpha_ss = %f\n", parameters_host.FWHM_sigma_inv_ss, parameters_host.FWHM_alpha_ss);

	LOR_FWHM = p.FWHM_IS;
	parameters_host.FWHM_sigma_inv_is = 4 * log(2.0) / (LOR_FWHM*LOR_FWHM);
	parameters_host.FWHM_alpha_is = 2 * sqrt(log(2) / M_PI) / LOR_FWHM;
	//printf("...FWHM_sigma_inv2_is = %f, FWHM_alpha_is = %f\n", parameters_host.FWHM_sigma_inv_is, parameters_host.FWHM_alpha_is);

	LOR_FWHM = p.FWHM_II;
	parameters_host.FWHM_sigma_inv_ii = 4 * log(2.0) / (LOR_FWHM*LOR_FWHM);
	parameters_host.FWHM_alpha_ii = 2 * sqrt(log(2) / M_PI) / LOR_FWHM;
	//printf("...FWHM_sigma_inv2_ii = %f, FWHM_alpha_ii = %f\n", parameters_host.FWHM_sigma_inv_ii, parameters_host.FWHM_alpha_ii);
	/*
	//#define SIGMA_INV 0.13286795139986f			//FWHM=4
	//#define ALPHA 0.2348593196749128f			//FWHM=4

	//#define SIGMA_INV 0.6931471805599f			//FWHM=2
	//#define ALPHA 0.4697186393498f			//FWHM=2
	//#define SIGMA_INV 4.332169713f			//FWHM=0.8
	//#define ALPHA 1.17429657592f			//FWHM=0.8
	//#define SIGMA_INV 2.7725886162f			//FWHM=1.0
	//#define ALPHA 0.939437260735f			//FWHM=1.0
	//#define SIGMA_INV 0.69314718056f			//FWHM=2.0
	//#define ALPHA 0.4697186393f			//FWHM=2.0
	//#define SIGMA_INV 69.3147180559f			//FWHM=0.2
	//#define ALPHA 4.6971863934f			//FWHM=0.2
	//#define SIGMA_INV 17.328678852f			//FWHM=0.4
	//#define ALPHA 2.348593152f			//FWHM=0.4
	*/

	parameters_host.RANGE1 = (int)(LOR_FWHM / parameters_host.X_SAMP);
	parameters_host.RANGE2 = (int)(LOR_FWHM / parameters_host.Z_SAMP);

	parameters_host.RANGE_atten_1 = (int)(LOR_FWHM / parameters_host.X_SAMP);;
	parameters_host.RANGE_atten_2 = (int)(LOR_FWHM / parameters_host.Z_SAMP);


}

void
cuda_em_recon::_Mem_allocation_for_LST_events_memcopy(PET_geometry& detector, PET_movement& movement, PET_data& data, int start_index, int end_index, PET_coincidence_type pet_coinc_type, LST_LORs*& device_events_x_global_mem, LST_LORs*& device_events_y_global_mem, int& event_count_x_dominant, int& event_count_y_dominant, int device_ID){

	float dir[3];
	float dist, TOF_dist_temp;
	int count;
	float *src_x_x, *src_y_x, *src_z_x, *dest_x_x, *dest_y_x, *dest_z_x;
	float *src_x_y, *src_y_y, *src_z_y, *dest_x_y, *dest_y_y, *dest_z_y;
	float *src_x_z, *src_y_z, *src_z_z, *dest_x_z, *dest_y_z, *dest_z_z;
	float *coeff_x, *coeff_y, *coeff_z;
	float snorm_x, dnorm_x, centm_x, snorm_z, dnorm_z, centm_z, angle_s_x, angle_d_x, cos_s_x, cos_d_x;
	float FWHM_s, FWHM_d; //FWHM of Detector response function of destination and sourse crystal
	float coeffx, coeffy;
	float snorm_y, dnorm_y, centm_y, angle_s_y, angle_d_y, angle_s_z, angle_d_z, cos_s_y, cos_d_y, cos_s_z, cos_d_z, coeff_s_z, coeff_d_z;

	int position;
	box temp_s, temp_d;


	int lst_event_index;
	int total_event_count = end_index - start_index + 1;

	//allocation host memory to store the list mode events
	src_x_x = (float*)malloc(total_event_count*sizeof(float));
	src_y_x = (float*)malloc(total_event_count*sizeof(float));
	src_z_x = (float*)malloc(total_event_count*sizeof(float));
	dest_x_x = (float*)malloc(total_event_count*sizeof(float));
	dest_y_x = (float*)malloc(total_event_count*sizeof(float));
	dest_z_x = (float*)malloc(total_event_count*sizeof(float));
	coeff_x = (float*)malloc(total_event_count*sizeof(float));

	float *TOF_dist_x = (float*)malloc(total_event_count*sizeof(float));

	src_x_y = (float*)malloc(total_event_count*sizeof(float));
	src_y_y = (float*)malloc(total_event_count*sizeof(float));
	src_z_y = (float*)malloc(total_event_count*sizeof(float));
	dest_x_y = (float*)malloc(total_event_count*sizeof(float));
	dest_y_y = (float*)malloc(total_event_count*sizeof(float));
	dest_z_y = (float*)malloc(total_event_count*sizeof(float));
	coeff_y = (float*)malloc(total_event_count*sizeof(float));

	float *TOF_dist_y = (float*)malloc(total_event_count*sizeof(float));


	event_count_x_dominant = 0;
	event_count_y_dominant = 0;

	_ss[device_ID] = 0;
	_is[device_ID] = 0;
	_ii[device_ID] = 0;
	int m = 0;

	for (lst_event_index = start_index; lst_event_index <= end_index; lst_event_index++){


		PET_LST_event& current_lst_event = data.PET_LST_event_list[lst_event_index];


		if (!current_lst_event.is_event_type(pet_coinc_type)){
			continue;
		}

		if (current_lst_event.is_event_type(SS)){
			_ss[device_ID]++;
		}
		else if (current_lst_event.is_event_type(II)){
			_ii[device_ID]++;
		}
		else{
			_is[device_ID]++;
		}

		//box &s = detector.detector_crystal_list.at(current_lst_event.src_id).geometry;
		//box &d = detector.detector_crystal_list.at(current_lst_event.dest_id).geometry;

		//int timestep = movement.get_time_step();
		//int diststep = movement.get_dist_step();
		//float current_time = current_lst_event.t0;
		float shift_dist = current_lst_event.bed_position;
		//float shift_dist = 0.0;
		//cout << "\n current_lst_event.t0: " << current_lst_event.t0 << " Shift_dist : " << shift_dist;

		temp_s.center[0] = detector.detector_crystal_list.at(current_lst_event.src_id).geometry.center[0];
		temp_s.center[1] = detector.detector_crystal_list.at(current_lst_event.src_id).geometry.center[1];
		temp_s.center[2] = detector.detector_crystal_list.at(current_lst_event.src_id).geometry.center[2] - shift_dist;

		temp_d.center[0] = detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.center[0];
		temp_d.center[1] = detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.center[1];
		temp_d.center[2] = detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.center[2] - shift_dist;


		temp_s.dimension[0] = detector.detector_crystal_list.at(current_lst_event.src_id).geometry.dimension[0];
		temp_s.dimension[1] = detector.detector_crystal_list.at(current_lst_event.src_id).geometry.dimension[1];
		temp_s.dimension[2] = detector.detector_crystal_list.at(current_lst_event.src_id).geometry.dimension[2];

		temp_d.dimension[0] = detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.dimension[0];
		temp_d.dimension[1] = detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.dimension[1];
		temp_d.dimension[2] = detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.dimension[2];

		if (current_lst_event.src_id < NUM_SCANNER_CRYSTALS){

			temp_s.normal_1[0] =  (detector.detector_crystal_list.at(current_lst_event.src_id).geometry.normal_1[0]);
			temp_s.normal_1[1] =  (detector.detector_crystal_list.at(current_lst_event.src_id).geometry.normal_1[1]);
			temp_s.normal_1[2] =  (detector.detector_crystal_list.at(current_lst_event.src_id).geometry.normal_1[2]);
		}
		else if ((current_lst_event.src_id >= NUM_SCANNER_CRYSTALS && current_lst_event.src_id < NUM_SCANNER_CRYSTALS+NUM_INSERT_CRYSTALS)){
			temp_s.normal_1[0] = (detector.detector_crystal_list.at(current_lst_event.src_id).geometry.normal_1[0]);
			temp_s.normal_1[1] = (detector.detector_crystal_list.at(current_lst_event.src_id).geometry.normal_1[1]);
			temp_s.normal_1[2] = (detector.detector_crystal_list.at(current_lst_event.src_id).geometry.normal_1[2]);
		}
		
		if (current_lst_event.dest_id < NUM_SCANNER_CRYSTALS){

			temp_d.normal_1[0] = (detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.normal_1[0]);
			temp_d.normal_1[1] = (detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.normal_1[1]);
			temp_d.normal_1[2] = (detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.normal_1[2]);
		}

		else if ((current_lst_event.dest_id >= NUM_SCANNER_CRYSTALS && current_lst_event.dest_id < NUM_SCANNER_CRYSTALS + NUM_INSERT_CRYSTALS)){
			temp_d.normal_1[0] = (detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.normal_1[0]);
			temp_d.normal_1[1] = (detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.normal_1[1]);
			temp_d.normal_1[2] = (detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.normal_1[2]);
		}
		



		



	



		//For IS type events, we consider applying insert movement before copying crystal coordinates to GPU
		/*if (pet_coinc_type == IS){

			if (current_lst_event.src_id >= 60800){
				//apply movement to temp_s

				TVec3<float> center(temp_s.center[0], temp_s.center[1], temp_s.center[2]);
				position = movement.get_position_index_by_time(current_lst_event.t0);
				//printf("Time %f corresponds to position %d\n", current_lst_event.t0, position);
				TVec3<float> new_center = movement.get_transform_matrix(position).rotation*center;
				new_center += movement.get_transform_matrix(position).translate;

				temp_s.center[0] = new_center[0];
				temp_s.center[1] = new_center[1];
				temp_s.center[2] = new_center[2];

			}
			else{
				//apply movement to temp_d

				TVec3<float> center(temp_d.center[0], temp_d.center[1], temp_d.center[2]);
				position = movement.get_position_index_by_time(current_lst_event.t0);
				//printf("Time %f corresponds to position %d\n", current_lst_event.t0, position);
				TVec3<float> new_center = movement.get_transform_matrix(position).rotation*center;
				new_center += movement.get_transform_matrix(position).translate;

				temp_d.center[0] = new_center[0];
				temp_d.center[1] = new_center[1];
				temp_d.center[2] = new_center[2];

			}

		}*/

		TOF_dist_temp = current_lst_event.TOF_dist;
		count = current_lst_event.normfact;


		box &s = temp_s;
		box &d = temp_d;


		for (int dim = 0; dim < 3; dim++) {
			dir[dim] = d.center[dim] - s.center[dim];
		}
		/***********************************************************************************************************
		* Rewrtie this part into a dedicated function LOR_filter() to implement more complicated filtering strategy
		*/
		dist = sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
		if (dist < 50)
			continue;
		/**********************************************************************************************************/


		if (fabs(dir[0]) >= fabs(dir[1])){


			//horizontal , x predominant = 0
			//always make sure src has smaller x coordinate than dest

			snorm_x = 0, dnorm_x = 0, centm_x = 0, angle_s_x = 0, angle_d_x = 0, cos_s_x = 0, cos_d_x = 0;
			



			if (s.center[0] <= d.center[0]){
				src_x_x[event_count_x_dominant] = s.center[0];
				src_y_x[event_count_x_dominant] = s.center[1];
				src_z_x[event_count_x_dominant] = s.center[2];
				dest_x_x[event_count_x_dominant] = d.center[0];
				dest_y_x[event_count_x_dominant] = d.center[1];
				dest_z_x[event_count_x_dominant] = d.center[2];
				TOF_dist_x[event_count_x_dominant] = TOF_dist_temp;
				
				snorm_x = sqrt(s.normal_1[0] * s.normal_1[0] + s.normal_1[1] * s.normal_1[1] + s.normal_1[2] * s.normal_1[2]);
				dnorm_x = sqrt(d.normal_1[0] * d.normal_1[0] + d.normal_1[1] * d.normal_1[1] + d.normal_1[2] * d.normal_1[2]);
				centm_x = sqrt((d.center[0] - s.center[0]) * (d.center[0] - s.center[0]) + (d.center[1] - s.center[1]) * (d.center[1] - s.center[1]) + (d.center[2] - s.center[2]) * (d.center[2] - s.center[2]));
				cos_s_x = ((s.normal_1[0] * (d.center[0] - s.center[0])) + (s.normal_1[1] * (d.center[1] - s.center[1])) + (s.normal_1[2] * (d.center[2] - s.center[2]))) / (snorm_x * centm_x);
				cos_d_x = ((d.normal_1[0] * (s.center[0] - d.center[0])) + (d.normal_1[1] * (s.center[1] - d.center[1])) + (d.normal_1[2] * (s.center[2] - d.center[2]))) / (dnorm_x * centm_x);
				angle_s_x = acos((cos_s_x)) * 180 / M_PI;
				angle_d_x = acos((cos_d_x)) * 180 / M_PI;




				if ((angle_s_x >= 0 && angle_s_x <= 75) && (angle_d_x >= 0 && angle_d_x <= 75)){
					//if ((angle_s_x <= 1 ) || (angle_d_x <= 1)){
					
					
					if(current_lst_event.dest_id<NUM_SCANNER_CRYSTALS)
						FWHM_d=PSF_SCANNER_P0 + PSF_SCANNER_P1*angle_d_x+PSF_SCANNER_P2*angle_d_x*angle_d_x;
					else
						FWHM_d=PSF_OUTSERT_P0 + PSF_OUTSERT_P1*angle_d_x+PSF_OUTSERT_P2*angle_d_x*angle_d_x;


					if(current_lst_event.src_id<NUM_SCANNER_CRYSTALS)
						FWHM_s=PSF_SCANNER_P0 + PSF_SCANNER_P1*angle_s_x+PSF_SCANNER_P2*angle_s_x*angle_s_x;
					else
						FWHM_s=PSF_OUTSERT_P0 + PSF_OUTSERT_P1*angle_s_x+PSF_OUTSERT_P2*angle_s_x*angle_s_x;

					coeffx = (float)sqrt(FWHM_d*FWHM_d+FWHM_s*FWHM_s);
					
					//cout << angle_s_x << " " << angle_d_x <<" " << cos_s_x<< " " << cos_d_x<< " " <<centm_x<<"\n";
				}
				else{
					continue;
					coeffx = 0;
					
					//cout << angle_s_x << " " << angle_d_x << " " << cos_s_x << " " << cos_d_x << " " << centm_x << "\n";
					//cout << "#";

				}






			}
			else{
				src_x_x[event_count_x_dominant] = d.center[0];
				src_y_x[event_count_x_dominant] = d.center[1];
				src_z_x[event_count_x_dominant] = d.center[2];
				dest_x_x[event_count_x_dominant] = s.center[0];
				dest_y_x[event_count_x_dominant] = s.center[1];
				dest_z_x[event_count_x_dominant] = s.center[2];
				
				//NOTE!!!! When exchange the order of src and dest, be sure to flip the sign of TOF

				TOF_dist_x[event_count_x_dominant] = -TOF_dist_temp;

				snorm_x = sqrt(d.normal_1[0] * d.normal_1[0] + d.normal_1[1] * d.normal_1[1] + d.normal_1[2] * d.normal_1[2]);
				dnorm_x = sqrt(s.normal_1[0] * s.normal_1[0] + s.normal_1[1] * s.normal_1[1] + s.normal_1[2] * s.normal_1[2]);
				centm_x = sqrt((d.center[0] - s.center[0]) * (d.center[0] - s.center[0]) + (d.center[1] - s.center[1]) * (d.center[1] - s.center[1]) + (d.center[2] - s.center[2]) * (d.center[2] - s.center[2]));
				cos_s_x = ((d.normal_1[0] * (s.center[0] - d.center[0])) + (d.normal_1[1] * (s.center[1] - d.center[1])) + (d.normal_1[2] * (s.center[2] - d.center[2]))) / (snorm_x * centm_x);
				cos_d_x = ((s.normal_1[0] * (d.center[0] - s.center[0])) + (s.normal_1[1] * (d.center[1] - s.center[1])) + (s.normal_1[2] * (d.center[2] - s.center[2]))) / (dnorm_x * centm_x);
				angle_s_x = acos((cos_s_x)) * 180 / M_PI;
				angle_d_x = acos((cos_d_x)) * 180 / M_PI;

				//angle_s_x = 0;
				//angle_d_x = 0;

				if ((angle_s_x >= 0 && angle_s_x <= 75) && (angle_d_x >= 0 && angle_d_x <= 75)){
		          //if ((angle_s_x <= 1) || (angle_d_x <= 1)){


					if(current_lst_event.dest_id<NUM_SCANNER_CRYSTALS)
						FWHM_d=PSF_SCANNER_P0 + PSF_SCANNER_P1*angle_d_x+PSF_SCANNER_P2*angle_d_x*angle_d_x;
					else
						FWHM_d=PSF_OUTSERT_P0 + PSF_OUTSERT_P1*angle_d_x+PSF_OUTSERT_P2*angle_d_x*angle_d_x;

					if(current_lst_event.src_id<NUM_SCANNER_CRYSTALS)
						FWHM_s=PSF_SCANNER_P0 + PSF_SCANNER_P1*angle_s_x+PSF_SCANNER_P2*angle_s_x*angle_s_x;
					else
						FWHM_s=PSF_OUTSERT_P0 + PSF_OUTSERT_P1*angle_s_x+PSF_OUTSERT_P2*angle_s_x*angle_s_x;

					coeffx = (float)sqrt(FWHM_d*FWHM_d+FWHM_s*FWHM_s);
					
					//cout << angle_s_x << " " << angle_d_x << " " << cos_s_x << " " << cos_d_x << " " << centm_x << "\n";


				}
				else{
					continue;
					coeffx = 0;
					
					//cout << angle_s_x << " " << angle_d_x << " " << cos_s_x << " " << cos_d_x << " " << centm_x << "\n";
					//cout << "#";

				}
			}

			coeff_x[event_count_x_dominant] = coeffx ;

			event_count_x_dominant++;




		}
		else if (fabs(dir[1]) > fabs(dir[0])){
			//vertical, predominant = 1
			//always make sure src has smaller y coordinate than dest


			snorm_y = 0, dnorm_y = 0, centm_y = 0, angle_s_y = 0, angle_d_y = 0, cos_s_y = 0, cos_d_y = 0;
			

			if (s.center[1] <= d.center[1]){
				src_x_y[event_count_y_dominant] = s.center[0];
				src_y_y[event_count_y_dominant] = s.center[1];
				src_z_y[event_count_y_dominant] = s.center[2];
				dest_x_y[event_count_y_dominant] = d.center[0];
				dest_y_y[event_count_y_dominant] = d.center[1];
				dest_z_y[event_count_y_dominant] = d.center[2];
				TOF_dist_y[event_count_y_dominant] = TOF_dist_temp;
				
				snorm_y = sqrt(s.normal_1[0] * s.normal_1[0] + s.normal_1[1] * s.normal_1[1] + s.normal_1[2] * s.normal_1[2]);
				dnorm_y = sqrt(d.normal_1[0] * d.normal_1[0] + d.normal_1[1] * d.normal_1[1] + d.normal_1[2] * d.normal_1[2]);
				centm_y = sqrt((d.center[0] - s.center[0]) * (d.center[0] - s.center[0]) + (d.center[1] - s.center[1]) * (d.center[1] - s.center[1]) + (d.center[2] - s.center[2]) * (d.center[2] - s.center[2]));
				cos_s_y = ((s.normal_1[0] * (d.center[0] - s.center[0])) + (s.normal_1[1] * (d.center[1] - s.center[1])) + (s.normal_1[2] * (d.center[2] - s.center[2]))) / (snorm_y * centm_y);
				cos_d_y = ((d.normal_1[0] * (s.center[0] - d.center[0])) + (d.normal_1[1] * (s.center[1] - d.center[1])) + (d.normal_1[2] * (s.center[2] - d.center[2]))) / (dnorm_y * centm_y);
				angle_s_y = acos((cos_s_y)) * 180 / M_PI;
				angle_d_y = acos((cos_d_y)) * 180 / M_PI;

				//angle_s_y = 0;
				//angle_d_y = 0;

				if ((angle_s_y >= 0 && angle_s_y <= 75) && (angle_d_y >= 0 && angle_d_y <= 75)){
					//if ((angle_s_y <= 1 ) || (angle_d_y <= 1 )){
					if(current_lst_event.dest_id<NUM_SCANNER_CRYSTALS)
						FWHM_d=PSF_SCANNER_P0 + PSF_SCANNER_P1*angle_d_y+PSF_SCANNER_P2*angle_d_y*angle_d_y;
					else
						FWHM_d=PSF_OUTSERT_P0 + PSF_OUTSERT_P1*angle_d_y+PSF_OUTSERT_P2*angle_d_y*angle_d_y;

					if(current_lst_event.src_id<NUM_SCANNER_CRYSTALS)
						FWHM_s=PSF_SCANNER_P0 + PSF_SCANNER_P1*angle_s_y+PSF_SCANNER_P2*angle_s_y*angle_s_y;
					else
						FWHM_s=PSF_OUTSERT_P0 + PSF_OUTSERT_P1*angle_s_y+PSF_OUTSERT_P2*angle_s_y*angle_s_y;

					coeffy = (float)sqrt(FWHM_d*FWHM_d+FWHM_s*FWHM_s);
					
					//cout << angle_s_y << " " << angle_d_y << " " << cos_s_y << " " << cos_d_y << " " << centm_y << "\n";
				}
				else{
					continue;
					coeffy = 0;
				
					//cout << angle_s_y << " " << angle_d_y << " " << cos_s_y << " " << cos_d_y << " " << centm_y << "\n";
					//cout << "#";
				}




			}

			else {
				src_x_y[event_count_y_dominant] = d.center[0];
				src_y_y[event_count_y_dominant] = d.center[1];
				src_z_y[event_count_y_dominant] = d.center[2];
				dest_x_y[event_count_y_dominant] = s.center[0];
				dest_y_y[event_count_y_dominant] = s.center[1];
				dest_z_y[event_count_y_dominant] = s.center[2];
				//NOTE!!!! When exchange the order of src and dest, be sure to flip the sign of TOF
				TOF_dist_y[event_count_y_dominant] = -TOF_dist_temp;
				
				snorm_y = sqrt(d.normal_1[0] * d.normal_1[0] + d.normal_1[1] * d.normal_1[1] + d.normal_1[2] * d.normal_1[2]);
				dnorm_y = sqrt(s.normal_1[0] * s.normal_1[0] + s.normal_1[1] * s.normal_1[1] + s.normal_1[2] * s.normal_1[2]);
				centm_y = sqrt((d.center[0] - s.center[0]) * (d.center[0] - s.center[0]) + (d.center[1] - s.center[1]) * (d.center[1] - s.center[1]) + (d.center[2] - s.center[2]) * (d.center[2] - s.center[2]));
				cos_s_y = ((d.normal_1[0] * (s.center[0] - d.center[0])) + (d.normal_1[1] * (s.center[1] - d.center[1])) + (d.normal_1[2] * (s.center[2] - d.center[2]))) / (snorm_y * centm_y);
				cos_d_y = ((s.normal_1[0] * (d.center[0] - s.center[0])) + (s.normal_1[1] * (d.center[1] - s.center[1])) + (s.normal_1[2] * (d.center[2] - s.center[2]))) / (dnorm_y * centm_y);
				angle_s_y = acos((cos_s_y)) * 180 / M_PI;
				angle_d_y = acos((cos_d_y)) * 180 / M_PI;

				//angle_s_y = 0;
				//angle_d_y = 0;

				if ((angle_s_y >= 0 && angle_s_y <= 75) && (angle_d_y >= 0 && angle_d_y <= 75)){
				
					//if((angle_s_y <= 1 ) || (angle_d_y <= 1 )){

					if(current_lst_event.dest_id<NUM_SCANNER_CRYSTALS)
						FWHM_d=PSF_SCANNER_P0 + PSF_SCANNER_P1*angle_d_y+PSF_SCANNER_P2*angle_d_y*angle_d_y;
					else
						FWHM_d=PSF_OUTSERT_P0 + PSF_OUTSERT_P1*angle_d_y+PSF_OUTSERT_P2*angle_d_y*angle_d_y;

					if(current_lst_event.src_id<NUM_SCANNER_CRYSTALS)
						FWHM_s=PSF_SCANNER_P0 + PSF_SCANNER_P1*angle_s_y+PSF_SCANNER_P2*angle_s_y*angle_s_y;
					else
						FWHM_s=PSF_OUTSERT_P0 + PSF_OUTSERT_P1*angle_s_y+PSF_OUTSERT_P2*angle_s_y*angle_s_y;

					coeffy = (float)sqrt(FWHM_d*FWHM_d+FWHM_s*FWHM_s);
					
					//cout << angle_s_y << " " << angle_d_y << " " << cos_s_y << " " << cos_d_y << " " << centm_y << "\n";
				}
				else{
					coeffy = 0;
					continue;

					//cout << angle_s_y << " " << angle_d_y << " " << cos_s_y << " " << cos_d_y << " " << centm_y << "\n";
					//cout << "#";
				}


			}
			coeff_y[event_count_y_dominant] = coeffy ;


			event_count_y_dominant++;

		}


	}

			
	


	

	LST_LORs events_x_host;
	LST_LORs events_y_host;

	cout << "Total number of x-predominant event : " << event_count_x_dominant << std::endl;
	cout << "Total number of y-predominant event : " << event_count_y_dominant << std::endl;

	

	events_x_host.num_lines = event_count_x_dominant;
	
	//checkCudaErrors(cudaSetDevice(device_ID));
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMalloc((void**)&events_x_host.src_x, events_x_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_x_host.src_y, events_x_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_x_host.src_z, events_x_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_x_host.dest_x, events_x_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_x_host.dest_y, events_x_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_x_host.dest_z, events_x_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_x_host.coeff, events_x_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_x_host.TOF_dist, events_x_host.num_lines*sizeof(float)));
	

	checkCudaErrors(cudaMemcpy(events_x_host.src_x, src_x_x, events_x_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_x_host.src_y, src_y_x, events_x_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_x_host.src_z, src_z_x, events_x_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_x_host.dest_x, dest_x_x, events_x_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_x_host.dest_y, dest_y_x, events_x_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_x_host.dest_z, dest_z_x, events_x_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_x_host.coeff, coeff_x, events_x_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_x_host.TOF_dist, TOF_dist_x, events_x_host.num_lines*sizeof(float), cudaMemcpyDefault));
	
	
	events_y_host.num_lines = event_count_y_dominant;

	checkCudaErrors(cudaMalloc((void**)&events_y_host.src_x, events_y_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_y_host.src_y, events_y_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_y_host.src_z, events_y_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_y_host.dest_x, events_y_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_y_host.dest_y, events_y_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_y_host.dest_z, events_y_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_y_host.coeff, events_y_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_y_host.TOF_dist, events_y_host.num_lines*sizeof(float)));

	//checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(events_y_host.src_x, src_x_y, events_y_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_y_host.src_y, src_y_y, events_y_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_y_host.src_z, src_z_y, events_y_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_y_host.dest_x, dest_x_y, events_y_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_y_host.dest_y, dest_y_y, events_y_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_y_host.dest_z, dest_z_y, events_y_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_y_host.coeff, coeff_y, events_y_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_y_host.TOF_dist, TOF_dist_y, events_y_host.num_lines*sizeof(float), cudaMemcpyDefault));

	
	checkCudaErrors(cudaMalloc((void**)&device_events_x_global_mem, sizeof(LST_LORs)));
	checkCudaErrors(cudaMalloc((void**)&device_events_y_global_mem, sizeof(LST_LORs)));
	
	checkCudaErrors(cudaMemcpy(device_events_x_global_mem, &events_x_host, sizeof(LST_LORs), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(device_events_y_global_mem, &events_y_host, sizeof(LST_LORs), cudaMemcpyDefault));

	//cudaMemcpyToSymbol(device_event_x, &events_x_host, sizeof(LST_LORs));
	//cudaMemcpyToSymbol(device_event_y, &events_y_host, sizeof(LST_LORs));

	int device_active;
	checkCudaErrors(cudaGetDevice(&device_active));
	size_t freebyte, total;
	cudaMemGetInfo(&freebyte, &total);
	
	/*cout << "On Device " << device_active << ":" << std::endl;
	cout << "Data list length = " << total_event_count << std::endl;
	cout << "Total number of x-predominant event copyed " << events_x_host.num_lines << std::endl;
	cout << "Total number of y-predominant event copyed " << events_y_host.num_lines << std::endl;
	cout << "Insert-Insert: " << _ii[device_ID] << std::endl;
	cout << "Insert-Scanner: " << _is[device_ID] << std::endl;
	cout << "Scanner-Scanner: " << _ss[device_ID] << std::endl;
	cout << "Total: " << total_event_count << endl << std::endl;
	cout << "Cuda memory Free " << freebyte / 1024 / 1024 << "MB out " << "of Total " << total / 1024 / 1024 << "MB" << endl;*/

	//free host memory
	free(src_x_x);	
	free(src_y_x);	
	free(src_z_x);	
	free(dest_x_x);	
	free(dest_y_x);
	free(dest_z_x);	
	free(TOF_dist_x);	
	free(coeff_x);	
	

	free(src_x_y); 
	free(src_y_y); 
	free(src_z_y); 
	free(dest_x_y); 
	free(dest_y_y); 
	free(dest_z_y); 
	free(TOF_dist_y); 
	free(coeff_y); 
	

}



void
cuda_em_recon::_Mem_allocation_for_min_LST_events_memcopy(PET_geometry& detector, PET_movement& movement, PET_data& data, int start_index, int end_index, PET_coincidence_type pet_coinc_type, LST_LORs*& device_events_x_global_mem, LST_LORs*& device_events_y_global_mem, int& event_count_x_dominant, int& event_count_y_dominant, int device_ID){

	float dir[3];
	float dist, TOF_dist_temp;
	int count;
	float *src_x_x, *src_y_x, *src_z_x, *dest_x_x, *dest_y_x, *dest_z_x;
	float *src_x_y, *src_y_y, *src_z_y, *dest_x_y, *dest_y_y, *dest_z_y;
	float *src_x_z, *src_y_z, *src_z_z, *dest_x_z, *dest_y_z, *dest_z_z;
	float *coeff_x, *coeff_y, *coeff_z;
	float snorm_x, dnorm_x, centm_x, snorm_z, dnorm_z, centm_z, angle_s_x, angle_d_x, cos_s_x, cos_d_x;
	float coeffx, coeffy;
	float snorm_y, dnorm_y, centm_y, angle_s_y, angle_d_y, angle_s_z, angle_d_z, cos_s_y, cos_d_y, cos_s_z, cos_d_z, coeff_s_z, coeff_d_z;

	int position;
	box temp_s, temp_d;


	int lst_event_index;
	int total_event_count = end_index - start_index + 1;

	//allocation host memory to store the list mode events
	src_x_x = (float*)malloc(total_event_count*sizeof(float));
	src_y_x = (float*)malloc(total_event_count*sizeof(float));
	src_z_x = (float*)malloc(total_event_count*sizeof(float));
	dest_x_x = (float*)malloc(total_event_count*sizeof(float));
	dest_y_x = (float*)malloc(total_event_count*sizeof(float));
	dest_z_x = (float*)malloc(total_event_count*sizeof(float));
	coeff_x = (float*)malloc(total_event_count*sizeof(float));

	float *TOF_dist_x = (float*)malloc(total_event_count*sizeof(float));

	src_x_y = (float*)malloc(total_event_count*sizeof(float));
	src_y_y = (float*)malloc(total_event_count*sizeof(float));
	src_z_y = (float*)malloc(total_event_count*sizeof(float));
	dest_x_y = (float*)malloc(total_event_count*sizeof(float));
	dest_y_y = (float*)malloc(total_event_count*sizeof(float));
	dest_z_y = (float*)malloc(total_event_count*sizeof(float));
	coeff_y = (float*)malloc(total_event_count*sizeof(float));

	float *TOF_dist_y = (float*)malloc(total_event_count*sizeof(float));


	event_count_x_dominant = 0;
	event_count_y_dominant = 0;

	_ss[device_ID] = 0;
	_is[device_ID] = 0;
	_ii[device_ID] = 0;
	int m = 0;

	for (lst_event_index = start_index; lst_event_index <= end_index; lst_event_index++){


		PET_LST_event& current_lst_event = data.PET_LST_event_list[lst_event_index];


		if (!current_lst_event.is_event_type(pet_coinc_type)){
			continue;
		}

		if (current_lst_event.is_event_type(SS)){
			_ss[device_ID]++;
		}
		else if (current_lst_event.is_event_type(II)){
			_ii[device_ID]++;
		}
		else{
			_is[device_ID]++;
		}

		//box &s = detector.detector_crystal_list.at(current_lst_event.src_id).geometry;
		//box &d = detector.detector_crystal_list.at(current_lst_event.dest_id).geometry;

		//int timestep = movement.get_time_step();
		//int diststep = movement.get_dist_step();
		//float current_time = current_lst_event.t0;
		//float shift_dist = current_lst_event.bed_position;

		//cout << "\n current_lst_event.t0: " << current_lst_event.t0 << " Shift_dist : " << shift_dist;

		temp_s.center[0] = detector.detector_crystal_list.at(current_lst_event.src_id).geometry.center[0];
		temp_s.center[1] = detector.detector_crystal_list.at(current_lst_event.src_id).geometry.center[1];
		temp_s.center[2] = detector.detector_crystal_list.at(current_lst_event.src_id).geometry.center[2];

		temp_d.center[0] = detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.center[0];
		temp_d.center[1] = detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.center[1];
		temp_d.center[2] = detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.center[2];


		temp_s.dimension[0] = detector.detector_crystal_list.at(current_lst_event.src_id).geometry.dimension[0];
		temp_s.dimension[1] = detector.detector_crystal_list.at(current_lst_event.src_id).geometry.dimension[1];
		temp_s.dimension[2] = detector.detector_crystal_list.at(current_lst_event.src_id).geometry.dimension[2];

		temp_d.dimension[0] = detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.dimension[0];
		temp_d.dimension[1] = detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.dimension[1];
		temp_d.dimension[2] = detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.dimension[2];

		if (current_lst_event.src_id < NUM_SCANNER_CRYSTALS){

			temp_s.normal_1[0] =  (detector.detector_crystal_list.at(current_lst_event.src_id).geometry.normal_1[0]);
			temp_s.normal_1[1] =  (detector.detector_crystal_list.at(current_lst_event.src_id).geometry.normal_1[1]);
			temp_s.normal_1[2] =  (detector.detector_crystal_list.at(current_lst_event.src_id).geometry.normal_1[2]);
		}
		else if ((current_lst_event.src_id >= NUM_SCANNER_CRYSTALS && current_lst_event.src_id < NUM_SCANNER_CRYSTALS+NUM_INSERT_CRYSTALS)){
			temp_s.normal_1[0] = (detector.detector_crystal_list.at(current_lst_event.src_id).geometry.normal_1[0]);
			temp_s.normal_1[1] = (detector.detector_crystal_list.at(current_lst_event.src_id).geometry.normal_1[1]);
			temp_s.normal_1[2] = (detector.detector_crystal_list.at(current_lst_event.src_id).geometry.normal_1[2]);
		}
		
		if (current_lst_event.dest_id < NUM_SCANNER_CRYSTALS){

			temp_d.normal_1[0] = (detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.normal_1[0]);
			temp_d.normal_1[1] = (detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.normal_1[1]);
			temp_d.normal_1[2] = (detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.normal_1[2]);
		}

		else if ((current_lst_event.dest_id >= NUM_SCANNER_CRYSTALS && current_lst_event.dest_id < NUM_SCANNER_CRYSTALS + NUM_INSERT_CRYSTALS)){
			temp_d.normal_1[0] = (detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.normal_1[0]);
			temp_d.normal_1[1] = (detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.normal_1[1]);
			temp_d.normal_1[2] = (detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.normal_1[2]);
		}
		
		//For IS type events, we consider applying insert movement before copying crystal coordinates to GPU
		/*if (pet_coinc_type == IS){

			if (current_lst_event.src_id >= 60800){
				//apply movement to temp_s

				TVec3<float> center(temp_s.center[0], temp_s.center[1], temp_s.center[2]);
				position = movement.get_position_index_by_time(current_lst_event.t0);
				//printf("Time %f corresponds to position %d\n", current_lst_event.t0, position);
				TVec3<float> new_center = movement.get_transform_matrix(position).rotation*center;
				new_center += movement.get_transform_matrix(position).translate;

				temp_s.center[0] = new_center[0];
				temp_s.center[1] = new_center[1];
				temp_s.center[2] = new_center[2];

			}
			else{
				//apply movement to temp_d

				TVec3<float> center(temp_d.center[0], temp_d.center[1], temp_d.center[2]);
				position = movement.get_position_index_by_time(current_lst_event.t0);
				//printf("Time %f corresponds to position %d\n", current_lst_event.t0, position);
				TVec3<float> new_center = movement.get_transform_matrix(position).rotation*center;
				new_center += movement.get_transform_matrix(position).translate;

				temp_d.center[0] = new_center[0];
				temp_d.center[1] = new_center[1];
				temp_d.center[2] = new_center[2];

			}

		}*/

		TOF_dist_temp = current_lst_event.TOF_dist;
		count = current_lst_event.normfact;


		box &s = temp_s;
		box &d = temp_d;


		for (int dim = 0; dim < 3; dim++) {
			dir[dim] = d.center[dim] - s.center[dim];
		}
		/***********************************************************************************************************
		* Rewrtie this part into a dedicated function LOR_filter() to implement more complicated filtering strategy
		*/
		dist = sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
		if (dist < 50)
			continue;
		/**********************************************************************************************************/


		if (fabs(dir[0]) >= fabs(dir[1])){


			//horizontal , x predominant = 0
			//always make sure src has smaller x coordinate than dest

			snorm_x = 0, dnorm_x = 0, centm_x = 0, angle_s_x = 0, angle_d_x = 0, cos_s_x = 0, cos_d_x = 0;
			



			if (s.center[0] <= d.center[0]){
				src_x_x[event_count_x_dominant] = s.center[0];
				src_y_x[event_count_x_dominant] = s.center[1];
				src_z_x[event_count_x_dominant] = s.center[2];
				dest_x_x[event_count_x_dominant] = d.center[0];
				dest_y_x[event_count_x_dominant] = d.center[1];
				dest_z_x[event_count_x_dominant] = d.center[2];
				TOF_dist_x[event_count_x_dominant] = TOF_dist_temp;
				
				snorm_x = sqrt(s.normal_1[0] * s.normal_1[0] + s.normal_1[1] * s.normal_1[1] + s.normal_1[2] * s.normal_1[2]);
				dnorm_x = sqrt(d.normal_1[0] * d.normal_1[0] + d.normal_1[1] * d.normal_1[1] + d.normal_1[2] * d.normal_1[2]);
				centm_x = sqrt((d.center[0] - s.center[0]) * (d.center[0] - s.center[0]) + (d.center[1] - s.center[1]) * (d.center[1] - s.center[1]) + (d.center[2] - s.center[2]) * (d.center[2] - s.center[2]));
				cos_s_x = ((s.normal_1[0] * (d.center[0] - s.center[0])) + (s.normal_1[1] * (d.center[1] - s.center[1])) + (s.normal_1[2] * (d.center[2] - s.center[2]))) / (snorm_x * centm_x);
				cos_d_x = ((d.normal_1[0] * (s.center[0] - d.center[0])) + (d.normal_1[1] * (s.center[1] - d.center[1])) + (d.normal_1[2] * (s.center[2] - d.center[2]))) / (dnorm_x * centm_x);
				angle_s_x = acos((cos_s_x)) * 180 / M_PI;
				angle_d_x = acos((cos_d_x)) * 180 / M_PI;




				if ((angle_s_x >= 0 && angle_s_x <= 75) && (angle_d_x >= 0 && angle_d_x <= 75)){
					//if ((angle_s_x <= 1 ) || (angle_d_x <= 1)){

					coeffx = 1;
					
					//cout << angle_s_x << " " << angle_d_x <<" " << cos_s_x<< " " << cos_d_x<< " " <<centm_x<<"\n";


				}
				else{
					coeffx = 0;
					
					//cout << angle_s_x << " " << angle_d_x << " " << cos_s_x << " " << cos_d_x << " " << centm_x << "\n";
					//cout << "#";

				}






			}
			else{
				src_x_x[event_count_x_dominant] = d.center[0];
				src_y_x[event_count_x_dominant] = d.center[1];
				src_z_x[event_count_x_dominant] = d.center[2];
				dest_x_x[event_count_x_dominant] = s.center[0];
				dest_y_x[event_count_x_dominant] = s.center[1];
				dest_z_x[event_count_x_dominant] = s.center[2];
				
				//NOTE!!!! When exchange the order of src and dest, be sure to flip the sign of TOF

				TOF_dist_x[event_count_x_dominant] = -TOF_dist_temp;

				snorm_x = sqrt(d.normal_1[0] * d.normal_1[0] + d.normal_1[1] * d.normal_1[1] + d.normal_1[2] * d.normal_1[2]);
				dnorm_x = sqrt(s.normal_1[0] * s.normal_1[0] + s.normal_1[1] * s.normal_1[1] + s.normal_1[2] * s.normal_1[2]);
				centm_x = sqrt((d.center[0] - s.center[0]) * (d.center[0] - s.center[0]) + (d.center[1] - s.center[1]) * (d.center[1] - s.center[1]) + (d.center[2] - s.center[2]) * (d.center[2] - s.center[2]));
				cos_s_x = ((d.normal_1[0] * (s.center[0] - d.center[0])) + (d.normal_1[1] * (s.center[1] - d.center[1])) + (d.normal_1[2] * (s.center[2] - d.center[2]))) / (snorm_x * centm_x);
				cos_d_x = ((s.normal_1[0] * (d.center[0] - s.center[0])) + (s.normal_1[1] * (d.center[1] - s.center[1])) + (s.normal_1[2] * (d.center[2] - s.center[2]))) / (dnorm_x * centm_x);
				angle_s_x = acos((cos_s_x)) * 180 / M_PI;
				angle_d_x = acos((cos_d_x)) * 180 / M_PI;

				//angle_s_x = 0;
				//angle_d_x = 0;

				if ((angle_s_x >= 0 && angle_s_x <= 75) && (angle_d_x >= 0 && angle_d_x <= 75)){
		          //if ((angle_s_x <= 1) || (angle_d_x <= 1)){
					coeffx = 1;
					
					//cout << angle_s_x << " " << angle_d_x << " " << cos_s_x << " " << cos_d_x << " " << centm_x << "\n";


				}
				else{
					coeffx = 0;
					
					//cout << angle_s_x << " " << angle_d_x << " " << cos_s_x << " " << cos_d_x << " " << centm_x << "\n";
					//cout << "#";

				}
			}

			coeff_x[event_count_x_dominant] = coeffx ;

			event_count_x_dominant++;




		}
		else if (fabs(dir[1]) > fabs(dir[0])){
			//vertical, predominant = 1
			//always make sure src has smaller y coordinate than dest


			snorm_y = 0, dnorm_y = 0, centm_y = 0, angle_s_y = 0, angle_d_y = 0, cos_s_y = 0, cos_d_y = 0;
			

			if (s.center[1] <= d.center[1]){
				src_x_y[event_count_y_dominant] = s.center[0];
				src_y_y[event_count_y_dominant] = s.center[1];
				src_z_y[event_count_y_dominant] = s.center[2];
				dest_x_y[event_count_y_dominant] = d.center[0];
				dest_y_y[event_count_y_dominant] = d.center[1];
				dest_z_y[event_count_y_dominant] = d.center[2];
				TOF_dist_y[event_count_y_dominant] = TOF_dist_temp;
				
				snorm_y = sqrt(s.normal_1[0] * s.normal_1[0] + s.normal_1[1] * s.normal_1[1] + s.normal_1[2] * s.normal_1[2]);
				dnorm_y = sqrt(d.normal_1[0] * d.normal_1[0] + d.normal_1[1] * d.normal_1[1] + d.normal_1[2] * d.normal_1[2]);
				centm_y = sqrt((d.center[0] - s.center[0]) * (d.center[0] - s.center[0]) + (d.center[1] - s.center[1]) * (d.center[1] - s.center[1]) + (d.center[2] - s.center[2]) * (d.center[2] - s.center[2]));
				cos_s_y = ((s.normal_1[0] * (d.center[0] - s.center[0])) + (s.normal_1[1] * (d.center[1] - s.center[1])) + (s.normal_1[2] * (d.center[2] - s.center[2]))) / (snorm_y * centm_y);
				cos_d_y = ((d.normal_1[0] * (s.center[0] - d.center[0])) + (d.normal_1[1] * (s.center[1] - d.center[1])) + (d.normal_1[2] * (s.center[2] - d.center[2]))) / (dnorm_y * centm_y);
				angle_s_y = acos((cos_s_y)) * 180 / M_PI;
				angle_d_y = acos((cos_d_y)) * 180 / M_PI;

				//angle_s_y = 0;
				//angle_d_y = 0;

				if ((angle_s_y >= 0 && angle_s_y <= 75) && (angle_d_y >= 0 && angle_d_y <= 75)){
					//if ((angle_s_y <= 1 ) || (angle_d_y <= 1 )){

					coeffy = 1;
					
					//cout << angle_s_y << " " << angle_d_y << " " << cos_s_y << " " << cos_d_y << " " << centm_y << "\n";
				}
				else{
					coeffy = 0;
				
					//cout << angle_s_y << " " << angle_d_y << " " << cos_s_y << " " << cos_d_y << " " << centm_y << "\n";
					//cout << "#";
				}




			}

			else {
				src_x_y[event_count_y_dominant] = d.center[0];
				src_y_y[event_count_y_dominant] = d.center[1];
				src_z_y[event_count_y_dominant] = d.center[2];
				dest_x_y[event_count_y_dominant] = s.center[0];
				dest_y_y[event_count_y_dominant] = s.center[1];
				dest_z_y[event_count_y_dominant] = s.center[2];
				//NOTE!!!! When exchange the order of src and dest, be sure to flip the sign of TOF
				TOF_dist_y[event_count_y_dominant] = -TOF_dist_temp;
				
				snorm_y = sqrt(d.normal_1[0] * d.normal_1[0] + d.normal_1[1] * d.normal_1[1] + d.normal_1[2] * d.normal_1[2]);
				dnorm_y = sqrt(s.normal_1[0] * s.normal_1[0] + s.normal_1[1] * s.normal_1[1] + s.normal_1[2] * s.normal_1[2]);
				centm_y = sqrt((d.center[0] - s.center[0]) * (d.center[0] - s.center[0]) + (d.center[1] - s.center[1]) * (d.center[1] - s.center[1]) + (d.center[2] - s.center[2]) * (d.center[2] - s.center[2]));
				cos_s_y = ((d.normal_1[0] * (s.center[0] - d.center[0])) + (d.normal_1[1] * (s.center[1] - d.center[1])) + (d.normal_1[2] * (s.center[2] - d.center[2]))) / (snorm_y * centm_y);
				cos_d_y = ((s.normal_1[0] * (d.center[0] - s.center[0])) + (s.normal_1[1] * (d.center[1] - s.center[1])) + (s.normal_1[2] * (d.center[2] - s.center[2]))) / (dnorm_y * centm_y);
				angle_s_y = acos((cos_s_y)) * 180 / M_PI;
				angle_d_y = acos((cos_d_y)) * 180 / M_PI;

				//angle_s_y = 0;
				//angle_d_y = 0;

				if ((angle_s_y >= 0 && angle_s_y <= 75) && (angle_d_y >= 0 && angle_d_y <= 75)){
				
					//if((angle_s_y <= 1 ) || (angle_d_y <= 1 )){


					coeffy = 1;
					
					//cout << angle_s_y << " " << angle_d_y << " " << cos_s_y << " " << cos_d_y << " " << centm_y << "\n";
				}
				else{
					coeffy = 0;
					
					//cout << angle_s_y << " " << angle_d_y << " " << cos_s_y << " " << cos_d_y << " " << centm_y << "\n";
					//cout << "#";
				}


			}
			coeff_y[event_count_y_dominant] = coeffy ;


			event_count_y_dominant++;

		}


	}

			
	


	

	LST_LORs events_x_host;
	LST_LORs events_y_host;

	cout << "Total number of x-predominant event : " << event_count_x_dominant << std::endl;
	cout << "Total number of y-predominant event : " << event_count_y_dominant << std::endl;

	

	events_x_host.num_lines = event_count_x_dominant;
	
	//checkCudaErrors(cudaSetDevice(device_ID));
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMalloc((void**)&events_x_host.src_x, events_x_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_x_host.src_y, events_x_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_x_host.src_z, events_x_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_x_host.dest_x, events_x_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_x_host.dest_y, events_x_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_x_host.dest_z, events_x_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_x_host.coeff, events_x_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_x_host.TOF_dist, events_x_host.num_lines*sizeof(float)));
	

	checkCudaErrors(cudaMemcpy(events_x_host.src_x, src_x_x, events_x_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_x_host.src_y, src_y_x, events_x_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_x_host.src_z, src_z_x, events_x_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_x_host.dest_x, dest_x_x, events_x_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_x_host.dest_y, dest_y_x, events_x_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_x_host.dest_z, dest_z_x, events_x_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_x_host.coeff, coeff_x, events_x_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_x_host.TOF_dist, TOF_dist_x, events_x_host.num_lines*sizeof(float), cudaMemcpyDefault));
	
	
	events_y_host.num_lines = event_count_y_dominant;

	checkCudaErrors(cudaMalloc((void**)&events_y_host.src_x, events_y_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_y_host.src_y, events_y_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_y_host.src_z, events_y_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_y_host.dest_x, events_y_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_y_host.dest_y, events_y_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_y_host.dest_z, events_y_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_y_host.coeff, events_y_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_y_host.TOF_dist, events_y_host.num_lines*sizeof(float)));

	//checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(events_y_host.src_x, src_x_y, events_y_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_y_host.src_y, src_y_y, events_y_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_y_host.src_z, src_z_y, events_y_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_y_host.dest_x, dest_x_y, events_y_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_y_host.dest_y, dest_y_y, events_y_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_y_host.dest_z, dest_z_y, events_y_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_y_host.coeff, coeff_y, events_y_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_y_host.TOF_dist, TOF_dist_y, events_y_host.num_lines*sizeof(float), cudaMemcpyDefault));

	
	checkCudaErrors(cudaMalloc((void**)&device_events_x_global_mem, sizeof(LST_LORs)));
	checkCudaErrors(cudaMalloc((void**)&device_events_y_global_mem, sizeof(LST_LORs)));
	
	checkCudaErrors(cudaMemcpy(device_events_x_global_mem, &events_x_host, sizeof(LST_LORs), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(device_events_y_global_mem, &events_y_host, sizeof(LST_LORs), cudaMemcpyDefault));

	//cudaMemcpyToSymbol(device_event_x, &events_x_host, sizeof(LST_LORs));
	//cudaMemcpyToSymbol(device_event_y, &events_y_host, sizeof(LST_LORs));

	int device_active;
	checkCudaErrors(cudaGetDevice(&device_active));
	size_t freebyte, total;
	cudaMemGetInfo(&freebyte, &total);
	
	/*cout << "On Device " << device_active << ":" << std::endl;
	cout << "Data list length = " << total_event_count << std::endl;
	cout << "Total number of x-predominant event copyed " << events_x_host.num_lines << std::endl;
	cout << "Total number of y-predominant event copyed " << events_y_host.num_lines << std::endl;
	cout << "Insert-Insert: " << _ii[device_ID] << std::endl;
	cout << "Insert-Scanner: " << _is[device_ID] << std::endl;
	cout << "Scanner-Scanner: " << _ss[device_ID] << std::endl;
	cout << "Total: " << total_event_count << endl << std::endl;
	cout << "Cuda memory Free " << freebyte / 1024 / 1024 << "MB out " << "of Total " << total / 1024 / 1024 << "MB" << endl;*/

	//free host memory
	free(src_x_x);	
	free(src_y_x);	
	free(src_z_x);	
	free(dest_x_x);	
	free(dest_y_x);
	free(dest_z_x);	
	free(TOF_dist_x);	
	free(coeff_x);	
	

	free(src_x_y); 
	free(src_y_y); 
	free(src_z_y); 
	free(dest_x_y); 
	free(dest_y_y); 
	free(dest_z_y); 
	free(TOF_dist_y); 
	free(coeff_y); 
	

}




void
cuda_em_recon::_Mem_allocation_for_LST_events_memcopy_scatter(PET_geometry& detector, PET_movement& movement, PET_data_scatter& data, int start_index, int end_index, PET_coincidence_type pet_coinc_type, LST_LORs_scatter*& device_events_x_global_mem, LST_LORs_scatter*& device_events_y_global_mem, int& event_count_x_dominant, int& event_count_y_dominant, int device_ID){
	float dir[3];
	float dist, TOF_dist_temp;
	int count;
	float sc_coeff;
	float *src_x_x, *src_y_x, *src_z_x, *dest_x_x, *dest_y_x, *dest_z_x;
	float *src_x_y, *src_y_y, *src_z_y, *dest_x_y, *dest_y_y, *dest_z_y;
	float *src_x_z, *src_y_z, *src_z_z, *dest_x_z, *dest_y_z, *dest_z_z;
	int *coeff_x, *coeff_y, *coeff_z;
	float *sc_coeff_x, *sc_coeff_y;
	float snorm_x, dnorm_x, centm_x, snorm_z, dnorm_z, centm_z, angle_s_x, angle_d_x, cos_s_x, cos_d_x;
	int coeffx, coeffy;
	float snorm_y, dnorm_y, centm_y, angle_s_y, angle_d_y, angle_s_z, angle_d_z, cos_s_y, cos_d_y, cos_s_z, cos_d_z, coeff_s_z, coeff_d_z;

	int position;
	box temp_s, temp_d;


	int lst_event_index;
	int total_event_count = end_index - start_index + 1;

	//allocation host memory to store the list mode events
	src_x_x = (float*)malloc(total_event_count*sizeof(float));
	src_y_x = (float*)malloc(total_event_count*sizeof(float));
	src_z_x = (float*)malloc(total_event_count*sizeof(float));
	dest_x_x = (float*)malloc(total_event_count*sizeof(float));
	dest_y_x = (float*)malloc(total_event_count*sizeof(float));
	dest_z_x = (float*)malloc(total_event_count*sizeof(float));
	coeff_x = (int*)malloc(total_event_count*sizeof(int));
	sc_coeff_x = (float*)malloc(total_event_count*sizeof(float));

	float *TOF_dist_x = (float*)malloc(total_event_count*sizeof(float));

	src_x_y = (float*)malloc(total_event_count*sizeof(float));
	src_y_y = (float*)malloc(total_event_count*sizeof(float));
	src_z_y = (float*)malloc(total_event_count*sizeof(float));
	dest_x_y = (float*)malloc(total_event_count*sizeof(float));
	dest_y_y = (float*)malloc(total_event_count*sizeof(float));
	dest_z_y = (float*)malloc(total_event_count*sizeof(float));
	coeff_y = (int*)malloc(total_event_count*sizeof(int));
	sc_coeff_y = (float*)malloc(total_event_count*sizeof(float));

	float *TOF_dist_y = (float*)malloc(total_event_count*sizeof(float));


	event_count_x_dominant = 0;
	event_count_y_dominant = 0;

	_ss[device_ID] = 0;
	_is[device_ID] = 0;
	_ii[device_ID] = 0;
	int m = 0;
	float sum_scatter = 0;
	float mean_sc = 0;

	for (lst_event_index = start_index; lst_event_index <= end_index; lst_event_index++){


		PET_LST_event_scatter& current_lst_event = data.PET_LST_event_list[lst_event_index];


		if (!current_lst_event.is_event_type(pet_coinc_type)){
			continue;
		}

		if (current_lst_event.is_event_type(SS)){
			_ss[device_ID]++;
		}
		else if (current_lst_event.is_event_type(II)){
			_ii[device_ID]++;
		}
		else{
			_is[device_ID]++;
		}

		//box &s = detector.detector_crystal_list.at(current_lst_event.src_id).geometry;
		//box &d = detector.detector_crystal_list.at(current_lst_event.dest_id).geometry;

		int timestep = movement.get_time_step();
		int diststep = movement.get_dist_step();
		int current_time = current_lst_event.t0;
		float shift_dist = current_lst_event.bed_position;

		//cout << "\n current_lst_event.t0: " << current_lst_event.t0 << " timestep: " << timestep << " diststep: " << diststep<<" Shift_dist : " << shift_dist;

		temp_s.center[0] = detector.detector_crystal_list.at(current_lst_event.src_id).geometry.center[0];
		temp_s.center[1] = detector.detector_crystal_list.at(current_lst_event.src_id).geometry.center[1];
		temp_s.center[2] = detector.detector_crystal_list.at(current_lst_event.src_id).geometry.center[2] - shift_dist;

		temp_d.center[0] = detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.center[0];
		temp_d.center[1] = detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.center[1];
		temp_d.center[2] = detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.center[2] - shift_dist;


		temp_s.dimension[0] = detector.detector_crystal_list.at(current_lst_event.src_id).geometry.dimension[0];
		temp_s.dimension[1] = detector.detector_crystal_list.at(current_lst_event.src_id).geometry.dimension[1];
		temp_s.dimension[2] = detector.detector_crystal_list.at(current_lst_event.src_id).geometry.dimension[2];

		temp_d.dimension[0] = detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.dimension[0];
		temp_d.dimension[1] = detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.dimension[1];
		temp_d.dimension[2] = detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.dimension[2];

		if (current_lst_event.src_id < NUM_SCANNER_CRYSTALS+NUM_INSERT_CRYSTALS){

			temp_s.normal_1[0] = (detector.detector_crystal_list.at(current_lst_event.src_id).geometry.normal_1[0]);
			temp_s.normal_1[1] = (detector.detector_crystal_list.at(current_lst_event.src_id).geometry.normal_1[1]);
			temp_s.normal_1[2] = (detector.detector_crystal_list.at(current_lst_event.src_id).geometry.normal_1[2]);
		}
		/*else if ((current_lst_event.src_id >= 60800 && current_lst_event.src_id < 126336)){
			temp_s.normal_1[0] = (detector.detector_crystal_list.at(current_lst_event.src_id).geometry.normal_1[0]);
			temp_s.normal_1[1] = (detector.detector_crystal_list.at(current_lst_event.src_id).geometry.normal_1[1]);
			temp_s.normal_1[2] = (detector.detector_crystal_list.at(current_lst_event.src_id).geometry.normal_1[2]);
		}*/

		if (current_lst_event.dest_id < NUM_SCANNER_CRYSTALS+NUM_INSERT_CRYSTALS){

			temp_d.normal_1[0] = (detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.normal_1[0]);
			temp_d.normal_1[1] = (detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.normal_1[1]);
			temp_d.normal_1[2] = (detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.normal_1[2]);
		}

		/*else if ((current_lst_event.dest_id >= 60800 && current_lst_event.dest_id < 126336)){
			temp_d.normal_1[0] = (detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.normal_1[0]);
			temp_d.normal_1[1] = (detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.normal_1[1]);
			temp_d.normal_1[2] = (detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.normal_1[2]);
		}*/












		//For IS type events, we consider applying insert movement before copying crystal coordinates to GPU
		/*if (pet_coinc_type == IS){

		if (current_lst_event.src_id >= 60800){
		//apply movement to temp_s

		TVec3<float> center(temp_s.center[0], temp_s.center[1], temp_s.center[2]);
		position = movement.get_position_index_by_time(current_lst_event.t0);
		//printf("Time %f corresponds to position %d\n", current_lst_event.t0, position);
		TVec3<float> new_center = movement.get_transform_matrix(position).rotation*center;
		new_center += movement.get_transform_matrix(position).translate;

		temp_s.center[0] = new_center[0];
		temp_s.center[1] = new_center[1];
		temp_s.center[2] = new_center[2];

		}
		else{
		//apply movement to temp_d

		TVec3<float> center(temp_d.center[0], temp_d.center[1], temp_d.center[2]);
		position = movement.get_position_index_by_time(current_lst_event.t0);
		//printf("Time %f corresponds to position %d\n", current_lst_event.t0, position);
		TVec3<float> new_center = movement.get_transform_matrix(position).rotation*center;
		new_center += movement.get_transform_matrix(position).translate;

		temp_d.center[0] = new_center[0];
		temp_d.center[1] = new_center[1];
		temp_d.center[2] = new_center[2];

		}

		}*/
		
		TOF_dist_temp = current_lst_event.TOF_dist;
		count = 1;
		sc_coeff = current_lst_event.sc_coeff;

		sum_scatter += sc_coeff;
		


		box &s = temp_s;
		box &d = temp_d;


		for (int dim = 0; dim < 3; dim++) {
			dir[dim] = d.center[dim] - s.center[dim];
		}
		/***********************************************************************************************************
		* Rewrtie this part into a dedicated function LOR_filter() to implement more complicated filtering strategy
		*/
		dist = sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
		if (dist < 60)
			continue;
		/**********************************************************************************************************/


		if (fabs(dir[0]) >= fabs(dir[1])){


			//horizontal , x predominant = 0
			//always make sure src has smaller x coordinate than dest

			snorm_x = 0, dnorm_x = 0, centm_x = 0, angle_s_x = 0, angle_d_x = 0, cos_s_x = 0, cos_d_x = 0;




			if (s.center[0] <= d.center[0]){
				src_x_x[event_count_x_dominant] = s.center[0];
				src_y_x[event_count_x_dominant] = s.center[1];
				src_z_x[event_count_x_dominant] = s.center[2];
				dest_x_x[event_count_x_dominant] = d.center[0];
				dest_y_x[event_count_x_dominant] = d.center[1];
				dest_z_x[event_count_x_dominant] = d.center[2];
				TOF_dist_x[event_count_x_dominant] = TOF_dist_temp;
				sc_coeff_x[event_count_x_dominant] = sc_coeff;

				snorm_x = sqrt(s.normal_1[0] * s.normal_1[0] + s.normal_1[1] * s.normal_1[1] + s.normal_1[2] * s.normal_1[2]);
				dnorm_x = sqrt(d.normal_1[0] * d.normal_1[0] + d.normal_1[1] * d.normal_1[1] + d.normal_1[2] * d.normal_1[2]);
				centm_x = sqrt((d.center[0] - s.center[0]) * (d.center[0] - s.center[0]) + (d.center[1] - s.center[1]) * (d.center[1] - s.center[1]) + (d.center[2] - s.center[2]) * (d.center[2] - s.center[2]));
				cos_s_x = ((s.normal_1[0] * (d.center[0] - s.center[0])) + (s.normal_1[1] * (d.center[1] - s.center[1])) + (s.normal_1[2] * (d.center[2] - s.center[2]))) / (snorm_x * centm_x);
				cos_d_x = ((d.normal_1[0] * (s.center[0] - d.center[0])) + (d.normal_1[1] * (s.center[1] - d.center[1])) + (d.normal_1[2] * (s.center[2] - d.center[2]))) / (dnorm_x * centm_x);
				angle_s_x = acos((cos_s_x)) * 180 / M_PI;
				angle_d_x = acos((cos_d_x)) * 180 / M_PI;




				if ((angle_s_x >= 0 && angle_s_x <= 75) && (angle_d_x >= 0 && angle_d_x <= 75)){

					coeffx = 1;

					//cout << angle_s_x << " " << angle_d_x <<" " << cos_s_x<< " " << cos_d_x<< " " <<centm_x<<"\n";


				}
				else{
					coeffx = 0;


					//cout << "#";

				}






			}
			else{
				src_x_x[event_count_x_dominant] = d.center[0];
				src_y_x[event_count_x_dominant] = d.center[1];
				src_z_x[event_count_x_dominant] = d.center[2];
				dest_x_x[event_count_x_dominant] = s.center[0];
				dest_y_x[event_count_x_dominant] = s.center[1];
				dest_z_x[event_count_x_dominant] = s.center[2];
				sc_coeff_x[event_count_x_dominant] = sc_coeff;

				//NOTE!!!! When exchange the order of src and dest, be sure to flip the sign of TOF

				TOF_dist_x[event_count_x_dominant] = -TOF_dist_temp;

				snorm_x = sqrt(d.normal_1[0] * d.normal_1[0] + d.normal_1[1] * d.normal_1[1] + d.normal_1[2] * d.normal_1[2]);
				dnorm_x = sqrt(s.normal_1[0] * s.normal_1[0] + s.normal_1[1] * s.normal_1[1] + s.normal_1[2] * s.normal_1[2]);
				centm_x = sqrt((d.center[0] - s.center[0]) * (d.center[0] - s.center[0]) + (d.center[1] - s.center[1]) * (d.center[1] - s.center[1]) + (d.center[2] - s.center[2]) * (d.center[2] - s.center[2]));
				cos_s_x = ((d.normal_1[0] * (s.center[0] - d.center[0])) + (d.normal_1[1] * (s.center[1] - d.center[1])) + (d.normal_1[2] * (s.center[2] - d.center[2]))) / (snorm_x * centm_x);
				cos_d_x = ((s.normal_1[0] * (d.center[0] - s.center[0])) + (s.normal_1[1] * (d.center[1] - s.center[1])) + (s.normal_1[2] * (d.center[2] - s.center[2]))) / (dnorm_x * centm_x);
				angle_s_x = acos((cos_s_x)) * 180 / M_PI;
				angle_d_x = acos((cos_d_x)) * 180 / M_PI;

				//angle_s_x = 0;
				//angle_d_x = 0;

				if ((angle_s_x >= 0 && angle_s_x <= 75) && (angle_d_x >= 0 && angle_d_x <= 75)){

					coeffx = 1;

					//cout << angle_s_x << " " << angle_d_x << " " << cos_s_x << " " << cos_d_x << " " << centm_x << "\n";


				}
				else{
					coeffx = 0;


					//cout << "#";

				}
			}

			coeff_x[event_count_x_dominant] = coeffx ;

			event_count_x_dominant++;




		}
		else if (fabs(dir[1]) > fabs(dir[0])){
			//vertical, predominant = 1
			//always make sure src has smaller y coordinate than dest


			snorm_y = 0, dnorm_y = 0, centm_y = 0, angle_s_y = 0, angle_d_y = 0, cos_s_y = 0, cos_d_y = 0;


			if (s.center[1] <= d.center[1]){
				src_x_y[event_count_y_dominant] = s.center[0];
				src_y_y[event_count_y_dominant] = s.center[1];
				src_z_y[event_count_y_dominant] = s.center[2];
				dest_x_y[event_count_y_dominant] = d.center[0];
				dest_y_y[event_count_y_dominant] = d.center[1];
				dest_z_y[event_count_y_dominant] = d.center[2];
				TOF_dist_y[event_count_y_dominant] = TOF_dist_temp;
				sc_coeff_y[event_count_y_dominant] = sc_coeff;

				snorm_y = sqrt(s.normal_1[0] * s.normal_1[0] + s.normal_1[1] * s.normal_1[1] + s.normal_1[2] * s.normal_1[2]);
				dnorm_y = sqrt(d.normal_1[0] * d.normal_1[0] + d.normal_1[1] * d.normal_1[1] + d.normal_1[2] * d.normal_1[2]);
				centm_y = sqrt((d.center[0] - s.center[0]) * (d.center[0] - s.center[0]) + (d.center[1] - s.center[1]) * (d.center[1] - s.center[1]) + (d.center[2] - s.center[2]) * (d.center[2] - s.center[2]));
				cos_s_y = ((s.normal_1[0] * (d.center[0] - s.center[0])) + (s.normal_1[1] * (d.center[1] - s.center[1])) + (s.normal_1[2] * (d.center[2] - s.center[2]))) / (snorm_y * centm_y);
				cos_d_y = ((d.normal_1[0] * (s.center[0] - d.center[0])) + (d.normal_1[1] * (s.center[1] - d.center[1])) + (d.normal_1[2] * (s.center[2] - d.center[2]))) / (dnorm_y * centm_y);
				angle_s_y = acos((cos_s_y)) * 180 / M_PI;
				angle_d_y = acos((cos_d_y)) * 180 / M_PI;

				//angle_s_y = 0;
				//angle_d_y = 0;

				if ((angle_s_y >= 0 && angle_s_y <= 75) && (angle_d_y >= 0 && angle_d_y <= 75)){

					coeffy = 1;

					//cout << angle_s_y << " " << angle_d_y << " " << cos_s_y << " " << cos_d_y << " " << centm_y << "\n";
				}
				else{
					coeffy = 0;


					//cout << "#";
				}




			}

			else {
				src_x_y[event_count_y_dominant] = d.center[0];
				src_y_y[event_count_y_dominant] = d.center[1];
				src_z_y[event_count_y_dominant] = d.center[2];
				dest_x_y[event_count_y_dominant] = s.center[0];
				dest_y_y[event_count_y_dominant] = s.center[1];
				dest_z_y[event_count_y_dominant] = s.center[2];
				sc_coeff_y[event_count_y_dominant] = sc_coeff;
				//NOTE!!!! When exchange the order of src and dest, be sure to flip the sign of TOF
				TOF_dist_y[event_count_y_dominant] = -TOF_dist_temp;

				snorm_y = sqrt(d.normal_1[0] * d.normal_1[0] + d.normal_1[1] * d.normal_1[1] + d.normal_1[2] * d.normal_1[2]);
				dnorm_y = sqrt(s.normal_1[0] * s.normal_1[0] + s.normal_1[1] * s.normal_1[1] + s.normal_1[2] * s.normal_1[2]);
				centm_y = sqrt((d.center[0] - s.center[0]) * (d.center[0] - s.center[0]) + (d.center[1] - s.center[1]) * (d.center[1] - s.center[1]) + (d.center[2] - s.center[2]) * (d.center[2] - s.center[2]));
				cos_s_y = ((d.normal_1[0] * (s.center[0] - d.center[0])) + (d.normal_1[1] * (s.center[1] - d.center[1])) + (d.normal_1[2] * (s.center[2] - d.center[2]))) / (snorm_y * centm_y);
				cos_d_y = ((s.normal_1[0] * (d.center[0] - s.center[0])) + (s.normal_1[1] * (d.center[1] - s.center[1])) + (s.normal_1[2] * (d.center[2] - s.center[2]))) / (dnorm_y * centm_y);
				angle_s_y = acos((cos_s_y)) * 180 / M_PI;
				angle_d_y = acos((cos_d_y)) * 180 / M_PI;

				//angle_s_y = 0;
				//angle_d_y = 0;

				if ((angle_s_y >= 0 && angle_s_y <= 75) && (angle_d_y >= 0 && angle_d_y <= 75)){


					coeffy = 1;

					//cout << angle_s_y << " " << angle_d_y << " " << cos_s_y << " " << cos_d_y << " " << centm_y << "\n";
				}
				else{
					coeffy = 0;


					//cout << "#";
				}


			}
			coeff_y[event_count_y_dominant] = coeffy ;


			event_count_y_dominant++;

		}


	}


	int num_coinc = event_count_x_dominant + event_count_y_dominant;
	mean_sc = sum_scatter / num_coinc;
	cout << "\n sum_scatter: " << sum_scatter << " mean sc : " << mean_sc<<"\n";



	LST_LORs_scatter events_x_host;
	LST_LORs_scatter events_y_host;

	cout << "Total number of x-predominant event : " << event_count_x_dominant << std::endl;
	cout << "Total number of y-predominant event : " << event_count_y_dominant << std::endl;



	events_x_host.num_lines = event_count_x_dominant;

	//checkCudaErrors(cudaSetDevice(device_ID));
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMalloc((void**)&events_x_host.src_x, events_x_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_x_host.src_y, events_x_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_x_host.src_z, events_x_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_x_host.dest_x, events_x_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_x_host.dest_y, events_x_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_x_host.dest_z, events_x_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_x_host.coeff, events_x_host.num_lines*sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&events_x_host.TOF_dist, events_x_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_x_host.sc_coeff, events_x_host.num_lines*sizeof(float)));


	checkCudaErrors(cudaMemcpy(events_x_host.src_x, src_x_x, events_x_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_x_host.src_y, src_y_x, events_x_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_x_host.src_z, src_z_x, events_x_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_x_host.dest_x, dest_x_x, events_x_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_x_host.dest_y, dest_y_x, events_x_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_x_host.dest_z, dest_z_x, events_x_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_x_host.coeff, coeff_x, events_x_host.num_lines*sizeof(int), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_x_host.TOF_dist, TOF_dist_x, events_x_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_x_host.sc_coeff, sc_coeff_x, events_x_host.num_lines*sizeof(float), cudaMemcpyDefault));


	events_y_host.num_lines = event_count_y_dominant;

	checkCudaErrors(cudaMalloc((void**)&events_y_host.src_x, events_y_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_y_host.src_y, events_y_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_y_host.src_z, events_y_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_y_host.dest_x, events_y_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_y_host.dest_y, events_y_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_y_host.dest_z, events_y_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_y_host.coeff, events_y_host.num_lines*sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&events_y_host.TOF_dist, events_y_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_y_host.sc_coeff, events_y_host.num_lines*sizeof(float)));

	//checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(events_y_host.src_x, src_x_y, events_y_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_y_host.src_y, src_y_y, events_y_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_y_host.src_z, src_z_y, events_y_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_y_host.dest_x, dest_x_y, events_y_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_y_host.dest_y, dest_y_y, events_y_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_y_host.dest_z, dest_z_y, events_y_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_y_host.coeff, coeff_y, events_y_host.num_lines*sizeof(int), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_y_host.TOF_dist, TOF_dist_y, events_y_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_y_host.sc_coeff, sc_coeff_y, events_y_host.num_lines*sizeof(float), cudaMemcpyDefault));


	checkCudaErrors(cudaMalloc((void**)&device_events_x_global_mem, sizeof(LST_LORs_scatter)));
	checkCudaErrors(cudaMalloc((void**)&device_events_y_global_mem, sizeof(LST_LORs_scatter)));

	checkCudaErrors(cudaMemcpy(device_events_x_global_mem, &events_x_host, sizeof(LST_LORs_scatter), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(device_events_y_global_mem, &events_y_host, sizeof(LST_LORs_scatter), cudaMemcpyDefault));

	//cudaMemcpyToSymbol(device_event_x, &events_x_host, sizeof(LST_LORs));
	//cudaMemcpyToSymbol(device_event_y, &events_y_host, sizeof(LST_LORs));

	int device_active;
	checkCudaErrors(cudaGetDevice(&device_active));
	size_t freebyte, total;
	cudaMemGetInfo(&freebyte, &total);

	/*cout << "On Device " << device_active << ":" << std::endl;
	cout << "Data list length = " << total_event_count << std::endl;
	cout << "Total number of x-predominant event copyed " << events_x_host.num_lines << std::endl;
	cout << "Total number of y-predominant event copyed " << events_y_host.num_lines << std::endl;
	cout << "Insert-Insert: " << _ii[device_ID] << std::endl;
	cout << "Insert-Scanner: " << _is[device_ID] << std::endl;
	cout << "Scanner-Scanner: " << _ss[device_ID] << std::endl;
	cout << "Total: " << total_event_count << endl << std::endl;
	cout << "Cuda memory Free " << freebyte / 1024 / 1024 << "MB out " << "of Total " << total / 1024 / 1024 << "MB" << endl;*/

	//free host memory
	free(src_x_x);
	free(src_y_x);
	free(src_z_x);
	free(dest_x_x);
	free(dest_y_x);
	free(dest_z_x);
	free(TOF_dist_x);
	free(coeff_x);
	free(sc_coeff_x);


	free(src_x_y);
	free(src_y_y);
	free(src_z_y);
	free(dest_x_y);
	free(dest_y_y);
	free(dest_z_y);
	free(TOF_dist_y);
	free(coeff_y);
	free(sc_coeff_y);


}



void
cuda_em_recon::_Mem_allocation_for_LST_events_memcopy_norm(PET_geometry& detector, PET_movement& movement, PET_data& data, int start_index, int end_index, PET_coincidence_type pet_coinc_type, LST_LORs*& device_events_x_global_mem, LST_LORs*& device_events_y_global_mem, int& event_count_x_dominant, int& event_count_y_dominant, int device_ID){
	float dir[3];
	float dist, TOF_dist_temp;
	int count;
	float *src_x_x, *src_y_x, *src_z_x, *dest_x_x, *dest_y_x, *dest_z_x;
	float *src_x_y, *src_y_y, *src_z_y, *dest_x_y, *dest_y_y, *dest_z_y;
	float *src_x_z, *src_y_z, *src_z_z, *dest_x_z, *dest_y_z, *dest_z_z;
	int *coeff_x, *coeff_y, *coeff_z;
	float snorm_x, dnorm_x, centm_x, snorm_z, dnorm_z, centm_z, angle_s_x, angle_d_x, cos_s_x, cos_d_x;
	int coeffx, coeffy;
	float snorm_y, dnorm_y, centm_y, angle_s_y, angle_d_y, angle_s_z, angle_d_z, cos_s_y, cos_d_y, cos_s_z, cos_d_z, coeff_s_z, coeff_d_z;

	int position;
	box temp_s, temp_d;


	int lst_event_index;
	int total_event_count = end_index - start_index + 1;

	//allocation host memory to store the list mode events
	src_x_x = (float*)malloc(total_event_count*sizeof(float));
	src_y_x = (float*)malloc(total_event_count*sizeof(float));
	src_z_x = (float*)malloc(total_event_count*sizeof(float));
	dest_x_x = (float*)malloc(total_event_count*sizeof(float));
	dest_y_x = (float*)malloc(total_event_count*sizeof(float));
	dest_z_x = (float*)malloc(total_event_count*sizeof(float));
	coeff_x = (int*)malloc(total_event_count*sizeof(int));

	float *TOF_dist_x = (float*)malloc(total_event_count*sizeof(float));

	src_x_y = (float*)malloc(total_event_count*sizeof(float));
	src_y_y = (float*)malloc(total_event_count*sizeof(float));
	src_z_y = (float*)malloc(total_event_count*sizeof(float));
	dest_x_y = (float*)malloc(total_event_count*sizeof(float));
	dest_y_y = (float*)malloc(total_event_count*sizeof(float));
	dest_z_y = (float*)malloc(total_event_count*sizeof(float));
	coeff_y = (int*)malloc(total_event_count*sizeof(int));

	float *TOF_dist_y = (float*)malloc(total_event_count*sizeof(float));


	event_count_x_dominant = 0;
	event_count_y_dominant = 0;

	_ss[device_ID] = 0;
	_is[device_ID] = 0;
	_ii[device_ID] = 0;
	int m = 0;


	for (lst_event_index = start_index; lst_event_index <= end_index; lst_event_index++){


		PET_LST_event& current_lst_event = data.PET_LST_event_list[lst_event_index];


		if (!current_lst_event.is_event_type(pet_coinc_type)){
			continue;
		}

		if (current_lst_event.is_event_type(SS)){
			_ss[device_ID]++;
		}
		else if (current_lst_event.is_event_type(II)){
			_ii[device_ID]++;
		}
		else{
			_is[device_ID]++;
		}

		//box &s = detector.detector_crystal_list.at(current_lst_event.src_id).geometry;
		//box &d = detector.detector_crystal_list.at(current_lst_event.dest_id).geometry;
		int timestep = movement.get_time_step();
		int diststep = movement.get_dist_step();
		int current_time = current_lst_event.t0;
		int shift_dist = (current_time / timestep)*diststep;

		temp_s.center[0] = detector.detector_crystal_list.at(current_lst_event.src_id).geometry.center[0];
		temp_s.center[1] = detector.detector_crystal_list.at(current_lst_event.src_id).geometry.center[1];
		temp_s.center[2] = detector.detector_crystal_list.at(current_lst_event.src_id).geometry.center[2] - shift_dist;

		temp_d.center[0] = detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.center[0];
		temp_d.center[1] = detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.center[1];
		temp_d.center[2] = detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.center[2] - shift_dist;

		temp_s.dimension[0] = detector.detector_crystal_list.at(current_lst_event.src_id).geometry.dimension[0];
		temp_s.dimension[1] = detector.detector_crystal_list.at(current_lst_event.src_id).geometry.dimension[1];
		temp_s.dimension[2] = detector.detector_crystal_list.at(current_lst_event.src_id).geometry.dimension[2];

		temp_d.dimension[0] = detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.dimension[0];
		temp_d.dimension[1] = detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.dimension[1];
		temp_d.dimension[2] = detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.dimension[2];

		if (current_lst_event.src_id < NUM_SCANNER_CRYSTALS+NUM_INSERT_CRYSTALS){

			temp_s.normal_1[0] =  (detector.detector_crystal_list.at(current_lst_event.src_id).geometry.normal_1[0]);
			temp_s.normal_1[1] =  (detector.detector_crystal_list.at(current_lst_event.src_id).geometry.normal_1[1]);
			temp_s.normal_1[2] =  (detector.detector_crystal_list.at(current_lst_event.src_id).geometry.normal_1[2]);
		}
		/*else if ((current_lst_event.src_id >= NUM_SCA && current_lst_event.src_id < 126336)){
			temp_s.normal_1[0] = (detector.detector_crystal_list.at(current_lst_event.src_id).geometry.normal_1[0]);
			temp_s.normal_1[1] = (detector.detector_crystal_list.at(current_lst_event.src_id).geometry.normal_1[1]);
			temp_s.normal_1[2] = (detector.detector_crystal_list.at(current_lst_event.src_id).geometry.normal_1[2]);
		}*/
		

		if (current_lst_event.dest_id < NUM_SCANNER_CRYSTALS+NUM_INSERT_CRYSTALS){

			temp_d.normal_1[0] =  (detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.normal_1[0]);
			temp_d.normal_1[1] =  (detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.normal_1[1]);
			temp_d.normal_1[2] =  (detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.normal_1[2]);
		}

		/*else if ((current_lst_event.dest_id >= 60800 && current_lst_event.dest_id < 126336)){
			temp_d.normal_1[0] = (detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.normal_1[0]);
			temp_d.normal_1[1] = (detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.normal_1[1]);
			temp_d.normal_1[2] = (detector.detector_crystal_list.at(current_lst_event.dest_id).geometry.normal_1[2]);
		}*/
		











		//For IS type events, we consider applying insert movement before copying crystal coordinates to GPU
		/*if (pet_coinc_type == IS){

			if (current_lst_event.src_id >= 60800){
				//apply movement to temp_s

				TVec3<float> center(temp_s.center[0], temp_s.center[1], temp_s.center[2]);
				position = movement.get_position_index_by_time(current_lst_event.t0);
				//printf("Time %f corresponds to position %d\n", current_lst_event.t0, position);
				TVec3<float> new_center = movement.get_transform_matrix(position).rotation*center;
				new_center += movement.get_transform_matrix(position).translate;

				temp_s.center[0] = new_center[0];
				temp_s.center[1] = new_center[1];
				temp_s.center[2] = new_center[2];

			}
			else{
				//apply movement to temp_d

				TVec3<float> center(temp_d.center[0], temp_d.center[1], temp_d.center[2]);
				position = movement.get_position_index_by_time(current_lst_event.t0);
				//printf("Time %f corresponds to position %d\n", current_lst_event.t0, position);
				TVec3<float> new_center = movement.get_transform_matrix(position).rotation*center;
				new_center += movement.get_transform_matrix(position).translate;

				temp_d.center[0] = new_center[0];
				temp_d.center[1] = new_center[1];
				temp_d.center[2] = new_center[2];

			}

		}*/

		TOF_dist_temp = current_lst_event.TOF_dist;
		count = current_lst_event.normfact;


		box &s = temp_s;
		box &d = temp_d;


		for (int dim = 0; dim < 3; dim++) {
			dir[dim] = d.center[dim] - s.center[dim];
		}
		/***********************************************************************************************************
		* Rewrtie this part into a dedicated function LOR_filter() to implement more complicated filtering strategy
		*/
		dist = sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
		if (dist < 60)
			continue;
		/**********************************************************************************************************/


		if (fabs(dir[0]) >= fabs(dir[1])){


			//horizontal , x predominant = 0
			//always make sure src has smaller x coordinate than dest

			snorm_x = 0, dnorm_x = 0, centm_x = 0, angle_s_x = 0, angle_d_x = 0, cos_s_x = 0, cos_d_x = 0;




			if (s.center[0] <= d.center[0]){
				src_x_x[event_count_x_dominant] = s.center[0];
				src_y_x[event_count_x_dominant] = s.center[1];
				src_z_x[event_count_x_dominant] = s.center[2];
				dest_x_x[event_count_x_dominant] = d.center[0];
				dest_y_x[event_count_x_dominant] = d.center[1];
				dest_z_x[event_count_x_dominant] = d.center[2];
				TOF_dist_x[event_count_x_dominant] = TOF_dist_temp;

				snorm_x = sqrt(s.normal_1[0] * s.normal_1[0] + s.normal_1[1] * s.normal_1[1] + s.normal_1[2] * s.normal_1[2]);
				dnorm_x = sqrt(d.normal_1[0] * d.normal_1[0] + d.normal_1[1] * d.normal_1[1] + d.normal_1[2] * d.normal_1[2]);
				centm_x = sqrt((d.center[0] - s.center[0]) * (d.center[0] - s.center[0]) + (d.center[1] - s.center[1]) * (d.center[1] - s.center[1]) + (d.center[2] - s.center[2]) * (d.center[2] - s.center[2]));
				cos_s_x = ((s.normal_1[0] * (d.center[0] - s.center[0])) + (s.normal_1[1] * (d.center[1] - s.center[1])) + (s.normal_1[2] * (d.center[2] - s.center[2]))) / (snorm_x * centm_x);
				cos_d_x = ((d.normal_1[0] * (s.center[0] - d.center[0])) + (d.normal_1[1] * (s.center[1] - d.center[1])) + (d.normal_1[2] * (s.center[2] - d.center[2]))) / (dnorm_x * centm_x);
				angle_s_x = acos((cos_s_x)) * 180 / M_PI;
				angle_d_x = acos((cos_d_x)) * 180 / M_PI;




				if ((angle_s_x >= 0 && angle_s_x <= 75) && (angle_d_x >= 0 && angle_d_x <= 75)){

					coeffx = count;

					//cout << angle_s_x << " " << angle_d_x <<" " << cos_s_x<< " " << cos_d_x<< " " <<centm_x<<"\n";


				}
				else{
					coeffx = 0;


					//cout << "#";

				}






			}
			else{
				src_x_x[event_count_x_dominant] = d.center[0];
				src_y_x[event_count_x_dominant] = d.center[1];
				src_z_x[event_count_x_dominant] = d.center[2];
				dest_x_x[event_count_x_dominant] = s.center[0];
				dest_y_x[event_count_x_dominant] = s.center[1];
				dest_z_x[event_count_x_dominant] = s.center[2];

				//NOTE!!!! When exchange the order of src and dest, be sure to flip the sign of TOF

				TOF_dist_x[event_count_x_dominant] = -TOF_dist_temp;

				snorm_x = sqrt(d.normal_1[0] * d.normal_1[0] + d.normal_1[1] * d.normal_1[1] + d.normal_1[2] * d.normal_1[2]);
				dnorm_x = sqrt(s.normal_1[0] * s.normal_1[0] + s.normal_1[1] * s.normal_1[1] + s.normal_1[2] * s.normal_1[2]);
				centm_x = sqrt((d.center[0] - s.center[0]) * (d.center[0] - s.center[0]) + (d.center[1] - s.center[1]) * (d.center[1] - s.center[1]) + (d.center[2] - s.center[2]) * (d.center[2] - s.center[2]));
				cos_s_x = ((d.normal_1[0] * (s.center[0] - d.center[0])) + (d.normal_1[1] * (s.center[1] - d.center[1])) + (d.normal_1[2] * (s.center[2] - d.center[2]))) / (snorm_x * centm_x);
				cos_d_x = ((s.normal_1[0] * (d.center[0] - s.center[0])) + (s.normal_1[1] * (d.center[1] - s.center[1])) + (s.normal_1[2] * (d.center[2] - s.center[2]))) / (dnorm_x * centm_x);
				angle_s_x = acos((cos_s_x)) * 180 / M_PI;
				angle_d_x = acos((cos_d_x)) * 180 / M_PI;

				//angle_s_x = 0;
				//angle_d_x = 0;

				if ((angle_s_x >= 0 && angle_s_x <= 75) && (angle_d_x >= 0 && angle_d_x <= 75)){

					coeffx = count;

					//cout << angle_s_x << " " << angle_d_x << " " << cos_s_x << " " << cos_d_x << " " << centm_x << "\n";


				}
				else{
					coeffx = 0;


					//cout << "#";

				}
			}

			coeff_x[event_count_x_dominant] = coeffx;
			//cout << "\n" << coeff_x[event_count_x_dominant];
			event_count_x_dominant++;
			



		}
		else if (fabs(dir[1]) > fabs(dir[0])){
			//vertical, predominant = 1
			//always make sure src has smaller y coordinate than dest


			snorm_y = 0, dnorm_y = 0, centm_y = 0, angle_s_y = 0, angle_d_y = 0, cos_s_y = 0, cos_d_y = 0;


			if (s.center[1] <= d.center[1]){
				src_x_y[event_count_y_dominant] = s.center[0];
				src_y_y[event_count_y_dominant] = s.center[1];
				src_z_y[event_count_y_dominant] = s.center[2];
				dest_x_y[event_count_y_dominant] = d.center[0];
				dest_y_y[event_count_y_dominant] = d.center[1];
				dest_z_y[event_count_y_dominant] = d.center[2];
				TOF_dist_y[event_count_y_dominant] = TOF_dist_temp;

				snorm_y = sqrt(s.normal_1[0] * s.normal_1[0] + s.normal_1[1] * s.normal_1[1] + s.normal_1[2] * s.normal_1[2]);
				dnorm_y = sqrt(d.normal_1[0] * d.normal_1[0] + d.normal_1[1] * d.normal_1[1] + d.normal_1[2] * d.normal_1[2]);
				centm_y = sqrt((d.center[0] - s.center[0]) * (d.center[0] - s.center[0]) + (d.center[1] - s.center[1]) * (d.center[1] - s.center[1]) + (d.center[2] - s.center[2]) * (d.center[2] - s.center[2]));
				cos_s_y = ((s.normal_1[0] * (d.center[0] - s.center[0])) + (s.normal_1[1] * (d.center[1] - s.center[1])) + (s.normal_1[2] * (d.center[2] - s.center[2]))) / (snorm_y * centm_y);
				cos_d_y = ((d.normal_1[0] * (s.center[0] - d.center[0])) + (d.normal_1[1] * (s.center[1] - d.center[1])) + (d.normal_1[2] * (s.center[2] - d.center[2]))) / (dnorm_y * centm_y);
				angle_s_y = acos((cos_s_y)) * 180 / M_PI;
				angle_d_y = acos((cos_d_y)) * 180 / M_PI;

				//angle_s_y = 0;
				//angle_d_y = 0;

				if ((angle_s_y >= 0 && angle_s_y <= 75) && (angle_d_y >= 0 && angle_d_y <= 75)){

					coeffy = count;

					//cout << angle_s_y << " " << angle_d_y << " " << cos_s_y << " " << cos_d_y << " " << centm_y << "\n";
				}
				else{
					coeffy = 0;


					//cout << "#";
				}




			}

			else {
				src_x_y[event_count_y_dominant] = d.center[0];
				src_y_y[event_count_y_dominant] = d.center[1];
				src_z_y[event_count_y_dominant] = d.center[2];
				dest_x_y[event_count_y_dominant] = s.center[0];
				dest_y_y[event_count_y_dominant] = s.center[1];
				dest_z_y[event_count_y_dominant] = s.center[2];
				//NOTE!!!! When exchange the order of src and dest, be sure to flip the sign of TOF
				TOF_dist_y[event_count_y_dominant] = -TOF_dist_temp;

				snorm_y = sqrt(d.normal_1[0] * d.normal_1[0] + d.normal_1[1] * d.normal_1[1] + d.normal_1[2] * d.normal_1[2]);
				dnorm_y = sqrt(s.normal_1[0] * s.normal_1[0] + s.normal_1[1] * s.normal_1[1] + s.normal_1[2] * s.normal_1[2]);
				centm_y = sqrt((d.center[0] - s.center[0]) * (d.center[0] - s.center[0]) + (d.center[1] - s.center[1]) * (d.center[1] - s.center[1]) + (d.center[2] - s.center[2]) * (d.center[2] - s.center[2]));
				cos_s_y = ((d.normal_1[0] * (s.center[0] - d.center[0])) + (d.normal_1[1] * (s.center[1] - d.center[1])) + (d.normal_1[2] * (s.center[2] - d.center[2]))) / (snorm_y * centm_y);
				cos_d_y = ((s.normal_1[0] * (d.center[0] - s.center[0])) + (s.normal_1[1] * (d.center[1] - s.center[1])) + (s.normal_1[2] * (d.center[2] - s.center[2]))) / (dnorm_y * centm_y);
				angle_s_y = acos((cos_s_y)) * 180 / M_PI;
				angle_d_y = acos((cos_d_y)) * 180 / M_PI;

				//angle_s_y = 0;
				//angle_d_y = 0;

				if ((angle_s_y >= 0 && angle_s_y <= 75) && (angle_d_y >= 0 && angle_d_y <= 75)){


					coeffy = count;

					//cout << angle_s_y << " " << angle_d_y << " " << cos_s_y << " " << cos_d_y << " " << centm_y << "\n";
				}
				else{
					coeffy = 0;


					//cout << "#";
				}


			}
			coeff_y[event_count_y_dominant] = coeffy;
			//cout << "\n" << coeff_y[event_count_y_dominant];

			event_count_y_dominant++;

		}


	}







	LST_LORs events_x_host;
	LST_LORs events_y_host;

   cout << "Total number of x-predominant event : " << event_count_x_dominant << std::endl;
   cout << "Total number of y-predominant event : " << event_count_y_dominant << std::endl;



	events_x_host.num_lines = event_count_x_dominant;

	//checkCudaErrors(cudaSetDevice(device_ID));
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMalloc((void**)&events_x_host.src_x, events_x_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_x_host.src_y, events_x_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_x_host.src_z, events_x_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_x_host.dest_x, events_x_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_x_host.dest_y, events_x_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_x_host.dest_z, events_x_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_x_host.coeff, events_x_host.num_lines*sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&events_x_host.TOF_dist, events_x_host.num_lines*sizeof(float)));


	checkCudaErrors(cudaMemcpy(events_x_host.src_x, src_x_x, events_x_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_x_host.src_y, src_y_x, events_x_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_x_host.src_z, src_z_x, events_x_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_x_host.dest_x, dest_x_x, events_x_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_x_host.dest_y, dest_y_x, events_x_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_x_host.dest_z, dest_z_x, events_x_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_x_host.coeff, coeff_x, events_x_host.num_lines*sizeof(int), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_x_host.TOF_dist, TOF_dist_x, events_x_host.num_lines*sizeof(float), cudaMemcpyDefault));


	events_y_host.num_lines = event_count_y_dominant;

	checkCudaErrors(cudaMalloc((void**)&events_y_host.src_x, events_y_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_y_host.src_y, events_y_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_y_host.src_z, events_y_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_y_host.dest_x, events_y_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_y_host.dest_y, events_y_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_y_host.dest_z, events_y_host.num_lines*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&events_y_host.coeff, events_y_host.num_lines*sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&events_y_host.TOF_dist, events_y_host.num_lines*sizeof(float)));

	//checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(events_y_host.src_x, src_x_y, events_y_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_y_host.src_y, src_y_y, events_y_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_y_host.src_z, src_z_y, events_y_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_y_host.dest_x, dest_x_y, events_y_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_y_host.dest_y, dest_y_y, events_y_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_y_host.dest_z, dest_z_y, events_y_host.num_lines*sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_y_host.coeff, coeff_y, events_y_host.num_lines*sizeof(int), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(events_y_host.TOF_dist, TOF_dist_y, events_y_host.num_lines*sizeof(float), cudaMemcpyDefault));


	checkCudaErrors(cudaMalloc((void**)&device_events_x_global_mem, sizeof(LST_LORs)));
	checkCudaErrors(cudaMalloc((void**)&device_events_y_global_mem, sizeof(LST_LORs)));

	checkCudaErrors(cudaMemcpy(device_events_x_global_mem, &events_x_host, sizeof(LST_LORs), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(device_events_y_global_mem, &events_y_host, sizeof(LST_LORs), cudaMemcpyDefault));

	//cudaMemcpyToSymbol(device_event_x, &events_x_host, sizeof(LST_LORs));
	//cudaMemcpyToSymbol(device_event_y, &events_y_host, sizeof(LST_LORs));

	int device_active;
	checkCudaErrors(cudaGetDevice(&device_active));
	size_t freebyte, total;
	cudaMemGetInfo(&freebyte, &total);

	/*cout << "On Device " << device_active << ":" << std::endl;
	cout << "Data list length = " << total_event_count << std::endl;
	cout << "Total number of x-predominant event copyed " << events_x_host.num_lines << std::endl;
	cout << "Total number of y-predominant event copyed " << events_y_host.num_lines << std::endl;
	cout << "Insert-Insert: " << _ii[device_ID] << std::endl;
	cout << "Insert-Scanner: " << _is[device_ID] << std::endl;
	cout << "Scanner-Scanner: " << _ss[device_ID] << std::endl;
	cout << "Total: " << total_event_count << endl << std::endl;
	cout << "Cuda memory Free " << freebyte / 1024 / 1024 << "MB out " << "of Total " << total / 1024 / 1024 << "MB" << endl;*/

	//free host memory
	free(src_x_x);
	free(src_y_x);
	free(src_z_x);
	free(dest_x_x);
	free(dest_y_x);
	free(dest_z_x);
	free(TOF_dist_x);
	free(coeff_x);


	free(src_x_y);
	free(src_y_y);
	free(src_z_y);
	free(dest_x_y);
	free(dest_y_y);
	free(dest_z_y);
	free(TOF_dist_y);
	free(coeff_y);


}



void
cuda_em_recon::_Mem_release_for_LST_events_uvm(LST_LORs_uvm*& device_events_x_uvm, LST_LORs_uvm*& device_events_y_uvm){
	cudaFree(device_events_x_uvm->src_x);
	cudaFree(device_events_x_uvm->src_y);
	cudaFree(device_events_x_uvm->src_z);
	cudaFree(device_events_x_uvm->dest_x);
	cudaFree(device_events_x_uvm->dest_y);
	cudaFree(device_events_x_uvm->dest_z);
	cudaFree(device_events_x_uvm->TOF_dist);
	cudaFree(device_events_x_uvm->coeff);

	delete device_events_x_uvm;

	cudaFree(device_events_y_uvm->src_x);
	cudaFree(device_events_y_uvm->src_y);
	cudaFree(device_events_y_uvm->src_z);
	cudaFree(device_events_y_uvm->dest_x);
	cudaFree(device_events_y_uvm->dest_y);
	cudaFree(device_events_y_uvm->dest_z);
	cudaFree(device_events_y_uvm->TOF_dist);
	cudaFree(device_events_y_uvm->coeff);

	delete device_events_y_uvm;

	
}


void
cuda_em_recon::_Mem_release_for_LST_events_memcopy(LST_LORs* device_events_x, LST_LORs* device_events_y){
	
	LST_LORs events_x_host;
	LST_LORs events_y_host;
	

	checkCudaErrors(cudaMemcpy(&events_x_host, device_events_x, sizeof(LST_LORs), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(&events_y_host, device_events_y, sizeof(LST_LORs), cudaMemcpyDefault));
	

	checkCudaErrors(cudaFree(events_x_host.src_x));
	checkCudaErrors(cudaFree(events_x_host.src_y));
	checkCudaErrors(cudaFree(events_x_host.src_z));
	checkCudaErrors(cudaFree(events_x_host.dest_x));
	checkCudaErrors(cudaFree(events_x_host.dest_y));
	checkCudaErrors(cudaFree(events_x_host.dest_z));
	checkCudaErrors(cudaFree(events_x_host.coeff));
	checkCudaErrors(cudaFree(events_x_host.TOF_dist));
	
	checkCudaErrors(cudaFree(device_events_x));

	
	checkCudaErrors(cudaFree(events_y_host.src_x));
	checkCudaErrors(cudaFree(events_y_host.src_y));
	checkCudaErrors(cudaFree(events_y_host.src_z));
	checkCudaErrors(cudaFree(events_y_host.dest_x));
	checkCudaErrors(cudaFree(events_y_host.dest_y));
	checkCudaErrors(cudaFree(events_y_host.dest_z));
	checkCudaErrors(cudaFree(events_y_host.coeff));
	checkCudaErrors(cudaFree(events_y_host.TOF_dist));
	
	checkCudaErrors(cudaFree(device_events_y));

	
}

void
cuda_em_recon::_Mem_release_for_LST_events_memcopy_scatter(LST_LORs_scatter* device_events_x, LST_LORs_scatter* device_events_y){

	LST_LORs_scatter events_x_host;
	LST_LORs_scatter events_y_host;


	checkCudaErrors(cudaMemcpy(&events_x_host, device_events_x, sizeof(LST_LORs_scatter), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(&events_y_host, device_events_y, sizeof(LST_LORs_scatter), cudaMemcpyDefault));


	checkCudaErrors(cudaFree(events_x_host.src_x));
	checkCudaErrors(cudaFree(events_x_host.src_y));
	checkCudaErrors(cudaFree(events_x_host.src_z));
	checkCudaErrors(cudaFree(events_x_host.dest_x));
	checkCudaErrors(cudaFree(events_x_host.dest_y));
	checkCudaErrors(cudaFree(events_x_host.dest_z));
	checkCudaErrors(cudaFree(events_x_host.coeff));
	checkCudaErrors(cudaFree(events_x_host.TOF_dist));
	checkCudaErrors(cudaFree(events_x_host.sc_coeff));

	checkCudaErrors(cudaFree(device_events_x));


	checkCudaErrors(cudaFree(events_y_host.src_x));
	checkCudaErrors(cudaFree(events_y_host.src_y));
	checkCudaErrors(cudaFree(events_y_host.src_z));
	checkCudaErrors(cudaFree(events_y_host.dest_x));
	checkCudaErrors(cudaFree(events_y_host.dest_y));
	checkCudaErrors(cudaFree(events_y_host.dest_z));
	checkCudaErrors(cudaFree(events_y_host.coeff));
	checkCudaErrors(cudaFree(events_y_host.TOF_dist));
	checkCudaErrors(cudaFree(events_y_host.sc_coeff));

	checkCudaErrors(cudaFree(device_events_y));


}


void
cuda_em_recon::_Mem_release_for_images(){
	int i;
	for (i = _num_gpu_start; i <= _num_gpu_end; i++){
		checkCudaErrors(cudaFree(_current_image_device[i]));
		//checkCudaErrors(cudaFree(_atten_image_device[i]));
		checkCudaErrors(cudaFree(_update_factor_device[i]));
		checkCudaErrors(cudaFree(_current2_image_device[i]));
		checkCudaErrors(cudaFree(_parameters_device[i]));
	}
	
	
}


void
cuda_em_recon::_Mem_allocation_for_fp_values(int device_ID){
	

	checkCudaErrors(cudaSetDevice(device_ID));
	//Allocate linear cuda memory to store the forward projection value, must be initialized to zero before use!!!!!!
	checkCudaErrors(cudaMalloc((void**)&_fp_value_x_device[device_ID], _event_count_x[device_ID] * sizeof(float)));
	//Allocate linear cuda memory to store the forward projection value, must be initialized to zero before use!!!!!!
	checkCudaErrors(cudaMalloc((void**)&_fp_value_y_device[device_ID], _event_count_y[device_ID] * sizeof(float)));
	//Allocate linear cuda memory to store the forward projection value, must be initialized to zero before use!!!!!!
	

	//Allocate linear cuda memory to store the forward projection value, must be initialized to zero before use!!!!!!
	checkCudaErrors(cudaMalloc((void**)&_fp_value2_x_device[device_ID], _event_count_x[device_ID] * sizeof(float)));
	//Allocate linear cuda memory to store the forward projection value, must be initialized to zero before use!!!!!!
	checkCudaErrors(cudaMalloc((void**)&_fp_value2_y_device[device_ID], _event_count_y[device_ID] * sizeof(float)));

}


void
cuda_em_recon::_Copy_images_to_GPU_memory(float** device, ImageArray<float>& host){
	
	cudaHostRegister(host._image, parameters_host.NUM_XYZ*sizeof(float),0);
	int i;
	for (i = _num_gpu_start; i <= _num_gpu_end; i++){
		checkCudaErrors(cudaMemcpyAsync(device[i], host._image, parameters_host.NUM_XYZ*sizeof(float), cudaMemcpyDefault));
	}
	cudaHostUnregister(host._image);
}


void
cuda_em_recon::Setup_data(PET_geometry& detector, PET_movement& movement, PET_data& data, int data_index_start, int data_index_end, PET_coincidence_type pet_coinc_type, int device_ID){
	
	//_Mem_allocation_for_LST_events_uvm(detector, data, _events_x_dominant_uvm, _events_y_dominant_uvm, _event_count_x, _event_count_y);
	checkCudaErrors(cudaSetDevice(device_ID));
	_Mem_allocation_for_LST_events_memcopy(detector, movement, data, data_index_start, data_index_end, pet_coinc_type, _events_x_dominant_global_mem[device_ID], _events_y_dominant_global_mem[device_ID], _event_count_x[device_ID], _event_count_y[device_ID], device_ID);
	//cout << "\n here";
	checkCudaErrors(cudaDeviceSynchronize());
	//cout << "Cuda data copy successfully for device " << device_ID << endl;

	_Mem_allocation_for_fp_values(device_ID);

	checkCudaErrors(cudaDeviceSynchronize());

	//cout << "Cuda fp value copy successfully for device " << device_ID << endl;
}



void
cuda_em_recon::Setup_data_min(PET_geometry& detector, PET_movement& movement, PET_data& data, int data_index_start, int data_index_end, PET_coincidence_type pet_coinc_type, int device_ID){
	
	//_Mem_allocation_for_LST_events_uvm(detector, data, _events_x_dominant_uvm, _events_y_dominant_uvm, _event_count_x, _event_count_y);
	checkCudaErrors(cudaSetDevice(device_ID));
	_Mem_allocation_for_min_LST_events_memcopy(detector, movement, data, data_index_start, data_index_end, pet_coinc_type, _events_x_dominant_global_mem[device_ID], _events_y_dominant_global_mem[device_ID], _event_count_x[device_ID], _event_count_y[device_ID], device_ID);
	//cout << "\n here";
	checkCudaErrors(cudaDeviceSynchronize());
	//cout << "Cuda data copy successfully for device " << device_ID << endl;

	_Mem_allocation_for_fp_values(device_ID);

	checkCudaErrors(cudaDeviceSynchronize());

	//cout << "Cuda fp value copy successfully for device " << device_ID << endl;
}



void
cuda_em_recon::Setup_data_scatter(PET_geometry& detector, PET_movement& movement, PET_data_scatter& data, int data_index_start, int data_index_end, PET_coincidence_type pet_coinc_type, int device_ID){

	//_Mem_allocation_for_LST_events_uvm(detector, data, _events_x_dominant_uvm, _events_y_dominant_uvm, _event_count_x, _event_count_y);
	checkCudaErrors(cudaSetDevice(device_ID));
	_Mem_allocation_for_LST_events_memcopy_scatter(detector, movement, data, data_index_start, data_index_end, pet_coinc_type, _events_x_dominant_global_mem_scatter[device_ID], _events_y_dominant_global_mem_scatter[device_ID], _event_count_x[device_ID], _event_count_y[device_ID], device_ID);
	//cout << "\n here";
	checkCudaErrors(cudaDeviceSynchronize());
	//cout << "Cuda data copy successfully for device " << device_ID << endl;

	_Mem_allocation_for_fp_values(device_ID);

	checkCudaErrors(cudaDeviceSynchronize());

	//cout << "Cuda fp value copy successfully for device " << device_ID << endl;
}


void
cuda_em_recon::Setup_data_norm(PET_geometry& detector, PET_movement& movement, PET_data& data, int data_index_start, int data_index_end, PET_coincidence_type pet_coinc_type, int device_ID){

	//_Mem_allocation_for_LST_events_uvm(detector, data, _events_x_dominant_uvm, _events_y_dominant_uvm, _event_count_x, _event_count_y);
	checkCudaErrors(cudaSetDevice(device_ID));
	_Mem_allocation_for_LST_events_memcopy_norm(detector, movement, data, data_index_start, data_index_end, pet_coinc_type, _events_x_dominant_global_mem[device_ID], _events_y_dominant_global_mem[device_ID], _event_count_x[device_ID], _event_count_y[device_ID], device_ID);
	//cout << "\n here";
	checkCudaErrors(cudaDeviceSynchronize());
	//cout << "Cuda data copy successfully for device " << device_ID << endl;

	_Mem_allocation_for_fp_values(device_ID);

	checkCudaErrors(cudaDeviceSynchronize());

	//cout << "Cuda fp value copy successfully for device " << device_ID << endl;
}
void
cuda_em_recon::Setup_parameters(parameters_t p){
	_Initialize_host_parameters(p);
}

void
cuda_em_recon::Setup_image(){
	
	//************************************Device Memoery Allocation*********************************//
	
	int i;
	for (i = _num_gpu_start; i <= _num_gpu_end; i++){
		checkCudaErrors(cudaSetDevice(i));

		//Allocate linear cuda memory to store the backward projection image
		checkCudaErrors(cudaMalloc((void**)&_update_factor_device[i], parameters_host.NUM_XYZ*sizeof(float)));
		//cout << "Cuda memory for storing update factor allocated" << endl;

		//Allocate linear cuda memory to store the backward projection image
		//checkCudaErrors(cudaMalloc((void**)&_atten_image_device[i], parameters_host.NUM_XYZ*sizeof(float)));
		//cout << "Cuda memory for storing update factor allocated" << endl;


		//Allocate linear cuda memory to store the current image
		checkCudaErrors(cudaMalloc((void**)&_current_image_device[i], parameters_host.NUM_XYZ *sizeof(float)));
		//cout << "Cuda memory for storing current images allocated successfully" << endl;

		//Allocate linear cuda memory to store the attenuation image
		checkCudaErrors(cudaMalloc((void**)&_current2_image_device[i], parameters_host.NUM_XYZ *sizeof(float)));
		//cout << "Cuda memory for storing current images allocated successfully" << endl;


		checkCudaErrors(cudaMalloc((void**)&_parameters_device[i], sizeof(PARAMETERS_IN_DEVICE_t)));
		//cout << "Cuda memory for storing parameters allocated successfully" << endl;
		checkCudaErrors(cudaDeviceSynchronize());
		
		size_t free, total;
		//printf("\n");
		cudaMemGetInfo(&free, &total);
		cout << "Cuda memory allocated successfully for device " << i << "Free " << free / 1024 / 1024 << "MB" << "of Total " << total / 1024 / 1024 << "MB" << endl;
	}
		

}


void
cuda_em_recon::Release(){

	Release_image();
	Release_data();
}

void
cuda_em_recon::Release_image(){
	_Mem_release_for_images();

}

void
cuda_em_recon::_Mem_release_for_fp_values(int device_ID){
	checkCudaErrors(cudaFree(_fp_value_x_device[device_ID]));
	checkCudaErrors(cudaFree(_fp_value_y_device[device_ID]));
	

	checkCudaErrors(cudaFree(_fp_value2_x_device[device_ID]));
	checkCudaErrors(cudaFree(_fp_value2_y_device[device_ID]));
}

void
cuda_em_recon::Release_data(){

	
	int i;
	for (i = _num_gpu_start; i <= _num_gpu_end; i++){
		checkCudaErrors(cudaFree(_fp_value_x_device[i]));
		checkCudaErrors(cudaFree(_fp_value_y_device[i]));
		
		checkCudaErrors(cudaFree(_fp_value2_x_device[i]));
		checkCudaErrors(cudaFree(_fp_value2_y_device[i]));

		_Mem_release_for_LST_events_memcopy(_events_x_dominant_global_mem[i], _events_y_dominant_global_mem[i]);
		_event_count_x[i] = 0;
		_event_count_y[i] = 0;
		

	}
	//_Mem_release_for_LST_events_uvm(_events_x_dominant_uvm, _events_y_dominant_uvm );

}

void
cuda_em_recon::Release_data_scatter(){


	int i;
	for (i = _num_gpu_start; i <= _num_gpu_end; i++){
		checkCudaErrors(cudaFree(_fp_value_x_device[i]));
		checkCudaErrors(cudaFree(_fp_value_y_device[i]));

		checkCudaErrors(cudaFree(_fp_value2_x_device[i]));
		checkCudaErrors(cudaFree(_fp_value2_y_device[i]));

		_Mem_release_for_LST_events_memcopy_scatter(_events_x_dominant_global_mem_scatter[i], _events_y_dominant_global_mem_scatter[i]);
		_event_count_x[i] = 0;
		_event_count_y[i] = 0;


	}
	//_Mem_release_for_LST_events_uvm(_events_x_dominant_uvm, _events_y_dominant_uvm );

}




void
cuda_em_recon::ComputeUpdateFactor(PET_geometry& detector, PET_movement& movement, PET_data& data, ImageArray<float>& Image, ImageArray<float>& UpdateFactor, ImageArray<float>& Attenuation_Image, float &nloglikelihood){

	cudaStream_t streamA[MAX_GPU], streamB[MAX_GPU], streamC[MAX_GPU], streamD[MAX_GPU];
	cudaEvent_t eventA[MAX_GPU], eventB[MAX_GPU], eventC[MAX_GPU], eventD[MAX_GPU];

	float milliseconds1 = 0.0f;
	float milliseconds2 = 0.0f;
	float milliseconds3 = 0.0f;

	dim3 dimBlock_f_x(BlockSize_forward_x);
	dim3 dimGrid_f_x(GridSize_forward_x);

	cudaEvent_t eventStart1[MAX_GPU], eventStart2[MAX_GPU], eventStart3[MAX_GPU], eventStop1[MAX_GPU], eventStop2[MAX_GPU], eventStop3[MAX_GPU];

	int device_id,device_data_length;
	int data_part_index;
	float* image_host;
	
	string filename;
	stringstream sstm;
	FILE *output_file;

	dim3 dimBlock_f_y(BlockSize_forward_y);
	dim3 dimGrid_f_y(GridSize_forward_y);

	dim3 dimBlock_b_x(BlockSize_backward_x);
	dim3 dimGrid_b_x(GridSize_backward_x);

	dim3 dimBlock_b_y(BlockSize_backward_y);
	dim3 dimGrid_b_y(GridSize_backward_y);

	

	dim3 dimBlock(1024);
	dim3 dimGrid(64);
	
	//copy image to all GPUs
	//cudaHostRegister(Image._image, parameters_host.NUM_XYZ*sizeof(float), cudaHostAllocPortable);
	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		checkCudaErrors(cudaSetDevice(device_id));

		
		checkCudaErrors(cudaMemcpy(_current_image_device[device_id], Image._image, parameters_host.NUM_XYZ*sizeof(float), cudaMemcpyDefault));
		
		checkCudaErrors(cudaMemcpy(_current2_image_device[device_id], Attenuation_Image._image, parameters_host.NUM_XYZ*sizeof(float), cudaMemcpyDefault));

		
		//setting the update factor to zero before projection
		setImageToZero_kernel << <dimGrid, dimBlock>> >(_update_factor_device[device_id], _parameters_device[device_id]);
		
	}
	//cudaHostUnregister(Image._image);
	
	float nloglikelihood_value_host = 0;
	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		checkCudaErrors(cudaSetDevice(device_id));
		checkCudaErrors(cudaMalloc((void**)&_nloglikelihood_value[device_id], sizeof(float)));
		checkCudaErrors(cudaMemcpy(_nloglikelihood_value[device_id], &nloglikelihood_value_host, sizeof(float), cudaMemcpyDefault));
	}



	printf("\n\n\n\n.......................Start SS data projection.........................\n");
	//SS events
			
		//copy data to all GPUs
	device_data_length = data.GetDataCount(ALL) / (_num_gpu_end - _num_gpu_start + 1);

	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		data_part_index = device_id - _num_gpu_start;
		if (device_id < _num_gpu_end){
			std::cout << "Copying data from " << data_part_index*device_data_length << " to " << (data_part_index + 1)*device_data_length - 1 << " to device " << device_id << std::endl;
			Setup_data(detector, movement, data, data_part_index*device_data_length, (data_part_index + 1)*device_data_length - 1, SS, device_id);
		}
		else{
			std::cout << "Copying data from " << data_part_index*device_data_length << " to " << data.GetDataCount(ALL) << " to device " << device_id << std::endl;
			Setup_data(detector, movement, data, data_part_index*device_data_length, data.GetDataCount(ALL) - 1, SS, device_id);
		}
	}
		printf("ALL SS data copied to device memory\n");

		parameters_host.FWHM_alpha = parameters_host.FWHM_alpha_ss;
		parameters_host.FWHM_sigma_inv = parameters_host.FWHM_sigma_inv_ss;
		parameters_host.RANGE1 = (int)(parameters_host.FWHM_ss / parameters_host.X_SAMP);
		parameters_host.RANGE2 = (int)(parameters_host.FWHM_ss / parameters_host.Z_SAMP);
		printf("H_CENTER_X = %f H_CENTER_Y = %f V_CENTER = %f\n", parameters_host.H_CENTER_X, parameters_host.H_CENTER_Y, parameters_host.V_CENTER);
		printf("RANGE1 = %d RANGE2 = %d FWHM_SS = %f\n", parameters_host.RANGE1, parameters_host.RANGE2, parameters_host.FWHM_ss);
		cudaHostRegister(&parameters_host, sizeof(PARAMETERS_IN_DEVICE_t), 0);

		for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
			checkCudaErrors(cudaSetDevice(device_id));
			checkCudaErrors(cudaDeviceSynchronize());


			checkCudaErrors(cudaStreamCreate(&streamA[device_id]));
			checkCudaErrors(cudaStreamCreate(&streamB[device_id]));
			checkCudaErrors(cudaStreamCreate(&streamC[device_id]));
			
			checkCudaErrors(cudaEventCreateWithFlags(&eventA[device_id], cudaEventDisableTiming));
			//checkCudaErrors(cudaEventCreate(&eventB[device_id]));
			//checkCudaErrors(cudaEventCreate(&eventC[device_id]));

			checkCudaErrors(cudaEventCreate(&eventStart1[device_id]));
			checkCudaErrors(cudaEventCreate(&eventStart2[device_id]));
			
			checkCudaErrors(cudaEventCreate(&eventStop1[device_id]));
			checkCudaErrors(cudaEventCreate(&eventStop2[device_id]));
			
		}

		for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
			cudaSetDevice(device_id);
			printf("\n\n\n\n.......................Start projection on Device %d.........................\n",device_id);
			


			checkCudaErrors(cudaMemcpyAsync(_parameters_device[device_id], &parameters_host, sizeof(PARAMETERS_IN_DEVICE_t), cudaMemcpyDefault, streamA[device_id]));
			checkCudaErrors(cudaEventRecord(eventA[device_id], streamA[device_id]));
			
			setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamB[device_id] >> >(_fp_value_x_device[device_id], _event_count_x[device_id]);
			setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamC[device_id] >> >(_fp_value_y_device[device_id], _event_count_y[device_id]);
			
			//if (atten_flag_fp){
				setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamB[device_id] >> >(_fp_value2_x_device[device_id], _event_count_x[device_id]);
				setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamC[device_id] >> >(_fp_value2_y_device[device_id], _event_count_y[device_id]);
			//}

			//if (export_likelihood_flag){
				//setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamB[device_id] >> >(_fp_value2_x_device[device_id], _event_count_x[device_id]);
				//setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamC[device_id] >> >(_fp_value2_y_device[device_id], _event_count_y[device_id]);
			//}
			
			if (_TOF_mode == TOF){
				//TOF version
				printf("starting TOF kernels.\n");

				checkCudaErrors(cudaStreamWaitEvent(streamB[device_id], eventA[device_id], 0));
				checkCudaErrors(cudaEventRecord(eventStart1[device_id], streamB[device_id]));

				if (atten_flag_fp){

					_fproj_atten_lst_cuda_x_kernel << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_current2_image_device[device_id], _fp_value2_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
					

					_TOF_fproj_lst_cuda_x_kernel_atten << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_current_image_device[device_id], _fp_value_x_device[device_id], _fp_value2_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
					if (export_likelihood_flag){
						//_fproj_atten_lst_cuda_x_kernel << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_atten_image_device[device_id], _fp_value2_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
						vector_sum_of_log << <dimGrid, dimBlock, 0, streamB[device_id] >> >(_fp_value_x_device[device_id], _event_count_x[device_id], _nloglikelihood_value[device_id]);
					}
				}

				else{
					_TOF_fproj_lst_cuda_x_kernel << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_current_image_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);

				}


				checkCudaErrors(cudaStreamWaitEvent(streamC[device_id], eventA[device_id], 0));
				checkCudaErrors(cudaEventRecord(eventStart2[device_id], streamC[device_id]));

				if (atten_flag_fp){

					_fproj_atten_lst_cuda_y_kernel << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_current2_image_device[device_id], _fp_value2_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
					

					_TOF_fproj_lst_cuda_y_kernel_atten << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_current_image_device[device_id], _fp_value_y_device[device_id], _fp_value2_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
					if (export_likelihood_flag){
						//_fproj_atten_lst_cuda_y_kernel << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_atten_image_device[device_id], _fp_value2_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
						vector_sum_of_log << <dimGrid, dimBlock, 0, streamC[device_id] >> >(_fp_value_y_device[device_id], _event_count_y[device_id], _nloglikelihood_value[device_id]);
					}
				}
				else{
					_TOF_fproj_lst_cuda_y_kernel << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_current_image_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
				}

				if (atten_flag_bp){
					_TOF_b_ratio_proj_lst_cuda_x_kernel_atten << <dimGrid_b_x, dimBlock_b_x, 0, streamB[device_id] >> >(_update_factor_device[device_id], _fp_value_x_device[device_id], _fp_value2_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
					//cudaEventRecord(eventB[device_id], streamB[device_id]);
					checkCudaErrors(cudaEventRecord(eventStop1[device_id], streamB[device_id]));

					_TOF_b_ratio_proj_lst_cuda_y_kernel_atten << <dimGrid_b_y, dimBlock_b_y, 0, streamC[device_id] >> >(_update_factor_device[device_id], _fp_value_y_device[device_id], _fp_value2_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
					//cudaEventRecord(eventC[device_id], streamC[device_id]);
					checkCudaErrors(cudaEventRecord(eventStop2[device_id], streamC[device_id]));
				}
				else{

					_TOF_b_ratio_proj_lst_cuda_x_kernel << <dimGrid_b_x, dimBlock_b_x, 0, streamB[device_id] >> >(_update_factor_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
					//cudaEventRecord(eventB[device_id], streamB[device_id]);
					checkCudaErrors(cudaEventRecord(eventStop1[device_id], streamB[device_id]));

					_TOF_b_ratio_proj_lst_cuda_y_kernel << <dimGrid_b_y, dimBlock_b_y, 0, streamC[device_id] >> >(_update_factor_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
					//cudaEventRecord(eventC[device_id], streamC[device_id]);
					checkCudaErrors(cudaEventRecord(eventStop2[device_id], streamC[device_id]));
				}
				
			}
			else{
				//NON-TOF version
				printf("starting NON-TOF kernels.\n");

				checkCudaErrors(cudaStreamWaitEvent(streamB[device_id], eventA[device_id], 0));
				checkCudaErrors(cudaEventRecord(eventStart1[device_id], streamB[device_id]));
				_fproj_lst_cuda_x_kernel << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_current_image_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
				if (export_likelihood_flag){
					//_fproj_atten_lst_cuda_x_kernel << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_atten_image_device[device_id], _fp_value2_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
					vector_sum_of_log << <dimGrid, dimBlock, 0, streamB[device_id] >> >(_fp_value_x_device[device_id], _event_count_x[device_id], _nloglikelihood_value[device_id]);
				}


				checkCudaErrors(cudaStreamWaitEvent(streamC[device_id], eventA[device_id], 0));
				checkCudaErrors(cudaEventRecord(eventStart2[device_id], streamC[device_id]));
				_fproj_lst_cuda_y_kernel << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_current_image_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
				if (export_likelihood_flag){
					//_fproj_atten_lst_cuda_y_kernel << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_atten_image_device[device_id], _fp_value2_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
					vector_sum_of_log << <dimGrid, dimBlock, 0, streamC[device_id] >> >(_fp_value_y_device[device_id], _event_count_y[device_id], _nloglikelihood_value[device_id]);
				}


				


				_b_ratio_proj_lst_cuda_x_kernel << <dimGrid_b_x, dimBlock_b_x, 0, streamB[device_id] >> >(_update_factor_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
				//cudaEventRecord(eventB[device_id], streamB[device_id]);
				checkCudaErrors(cudaEventRecord(eventStop1[device_id], streamB[device_id]));

				_b_ratio_proj_lst_cuda_y_kernel << <dimGrid_b_y, dimBlock_b_y, 0, streamC[device_id] >> >(_update_factor_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
				//cudaEventRecord(eventC[device_id], streamC[device_id]);
				checkCudaErrors(cudaEventRecord(eventStop2[device_id], streamC[device_id]));

			}

		}

		for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
			checkCudaErrors(cudaSetDevice(device_id));

			/*
			checkCudaErrors(cudaStreamSynchronize(streamA[device_id]));
			checkCudaErrors(cudaStreamSynchronize(streamB[device_id]));
			checkCudaErrors(cudaStreamSynchronize(streamC[device_id]));
			*/
			
			checkCudaErrors(cudaEventSynchronize(eventStart1[device_id]));
			checkCudaErrors(cudaEventSynchronize(eventStart2[device_id]));
			
			checkCudaErrors(cudaEventSynchronize(eventStop1[device_id]));
			checkCudaErrors(cudaEventSynchronize(eventStop2[device_id]));
			
			

			checkCudaErrors(cudaEventElapsedTime(&milliseconds1, eventStart1[device_id], eventStop1[device_id]));
			checkCudaErrors(cudaEventElapsedTime(&milliseconds2, eventStart2[device_id], eventStop2[device_id]));
			

			printf("\n.......................Wrapping up for device %d........................\n", device_id);
			printf("Time for X-dominant projections: %f seconds\n", milliseconds1 / 1000);
			printf("Time for Y-dominant projections: %f seconds\n", milliseconds2 / 1000);
			
			
			checkCudaErrors(cudaEventDestroy(eventA[device_id]));
			/*
			checkCudaErrors(cudaEventDestroy(eventB[device_id]));
			checkCudaErrors(cudaEventDestroy(eventC[device_id]));
			*/
			
			checkCudaErrors(cudaEventDestroy(eventStart1[device_id]));
			checkCudaErrors(cudaEventDestroy(eventStart2[device_id]));
			
			checkCudaErrors(cudaEventDestroy(eventStop1[device_id]));
			checkCudaErrors(cudaEventDestroy(eventStop2[device_id]));
			

			checkCudaErrors(cudaStreamDestroy(streamA[device_id]));
			checkCudaErrors(cudaStreamDestroy(streamB[device_id]));
			checkCudaErrors(cudaStreamDestroy(streamC[device_id]));
			

			//cudaDeviceSynchronize();
		}
		

		// Check forward projected values
		for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
			int size1 = _event_count_y[device_id];
			int size2 = _event_count_x[device_id];
			float sum_fp = 0, mean_fp = 0;
			float max_fp = 0;
			float* fpvalue1_hosty = new float[size1]();
			float* fpvalue1_hostx = new float[size2]();
			cudaSetDevice(device_id);
			cudaDeviceSynchronize();

			checkCudaErrors(cudaMemcpy(fpvalue1_hosty, _fp_value_y_device[device_id], size1*sizeof(float), cudaMemcpyDefault));
			checkCudaErrors(cudaMemcpy(fpvalue1_hostx, _fp_value_x_device[device_id], size2*sizeof(float), cudaMemcpyDefault));
			for (int i = 0; i < size1; i++){
				
				sum_fp += fpvalue1_hosty[i];
				if (max_fp < fpvalue1_hosty[i]){
					max_fp = fpvalue1_hosty[i];
				}
			}
			for (int j = 0; j < size2; j++){
				//cout << "\n fpvalues1 x " << fpvalue1_hostx[j];
				sum_fp += fpvalue1_hostx[j];
				if (max_fp < fpvalue1_hostx[j]){
					max_fp = fpvalue1_hostx[j];
				}
			}
			for (int i = 0; i < 100; i++){
				cout << "\n fpvalues x " << fpvalue1_hostx[i] << " fpvalues y " << fpvalue1_hosty[i];
			}
			mean_fp = sum_fp / (size1 + size2);
			cout << "\n Sum fp for device " << device_id << " : " << sum_fp << " Mean fp : " << mean_fp<<" Max fp : "<<max_fp;
		}

		for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
			int size1 = _event_count_y[device_id];
			int size2 = _event_count_x[device_id];
			float sum_fp = 0, mean_fp = 0;
			float* fpvalue2_hosty = new float[size1];
			float* fpvalue2_hostx = new float[size2];
			cudaSetDevice(device_id);
			cudaDeviceSynchronize();

			checkCudaErrors(cudaMemcpy(fpvalue2_hosty, _fp_value2_y_device[device_id], size1*sizeof(float), cudaMemcpyDefault));
			checkCudaErrors(cudaMemcpy(fpvalue2_hostx, _fp_value2_x_device[device_id], size2*sizeof(float), cudaMemcpyDefault));
			for (int i = 0; i < size1; i++){
				//cout << "\n fpvalues1 y " << fpvalue1_hosty[i];
				sum_fp += fpvalue2_hosty[i];
			}
			for (int j = 0; j < size2; j++){
				//cout << "\n fpvalues1 x " << fpvalue1_hostx[j];
				sum_fp += fpvalue2_hostx[j];
			}
			mean_fp = sum_fp / (size1 + size2);
			cout << "\n Sum fp2 for device " << device_id << " : " << sum_fp << " Mean fp : " << mean_fp;
		}

		cudaHostUnregister(&parameters_host);

		printf("\n... start clean up data\n\n");
		Release_data();
		printf("\n... done clean up data\n\n");



	printf("\n\n\n\n.......................Start IS data projection.........................\n");
	//IS events


	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		data_part_index = device_id - _num_gpu_start;
		if (device_id < _num_gpu_end){
			Setup_data(detector, movement, data, data_part_index*device_data_length, (data_part_index + 1)*device_data_length - 1, IS, device_id);
		}
		else{
			Setup_data(detector, movement, data, data_part_index*device_data_length, data.GetDataCount(ALL) - 1, IS, device_id);
		}
	}
		printf("IS data copied to device memory\n");
				


		parameters_host.FWHM_alpha = parameters_host.FWHM_alpha_is;
		parameters_host.FWHM_sigma_inv = parameters_host.FWHM_sigma_inv_is;
		parameters_host.RANGE1 = (int)(parameters_host.FWHM_is / parameters_host.X_SAMP);
		parameters_host.RANGE2 = (int)(parameters_host.FWHM_is / parameters_host.Z_SAMP);
		if (parameters_host.RANGE1 < 1)
			parameters_host.RANGE1 = 1;
		if (parameters_host.RANGE2 < 1)
			parameters_host.RANGE2 = 1;
		printf("H_CENTER_X = %f H_CENTER_Y = %f V_CENTER = %f\n", parameters_host.H_CENTER_X, parameters_host.H_CENTER_Y, parameters_host.V_CENTER);
		printf("RANGE1 = %d RANGE2 = %d FWHM_IS = %f\n", parameters_host.RANGE1, parameters_host.RANGE2, parameters_host.FWHM_is);
		cudaHostRegister(&parameters_host, sizeof(PARAMETERS_IN_DEVICE_t), 0);

		for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
			checkCudaErrors(cudaSetDevice(device_id));
			checkCudaErrors(cudaDeviceSynchronize());


			checkCudaErrors(cudaStreamCreate(&streamA[device_id]));
			checkCudaErrors(cudaStreamCreate(&streamB[device_id]));
			checkCudaErrors(cudaStreamCreate(&streamC[device_id]));
			checkCudaErrors(cudaEventCreateWithFlags(&eventA[device_id], cudaEventDisableTiming));
			//checkCudaErrors(cudaEventCreate(&eventB[device_id]));
			//checkCudaErrors(cudaEventCreate(&eventC[device_id]));

			checkCudaErrors(cudaEventCreate(&eventStart1[device_id]));
			checkCudaErrors(cudaEventCreate(&eventStart2[device_id]));
			checkCudaErrors(cudaEventCreate(&eventStop1[device_id]));
			checkCudaErrors(cudaEventCreate(&eventStop2[device_id]));
		}

		for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
			cudaSetDevice(device_id);
			printf("\n\n\n\n.......................Start projection on Device %d.........................\n", device_id);
						
			checkCudaErrors(cudaMemcpyAsync(_parameters_device[device_id], &parameters_host, sizeof(PARAMETERS_IN_DEVICE_t), cudaMemcpyDefault, streamA[device_id]));
			checkCudaErrors(cudaEventRecord(eventA[device_id], streamA[device_id]));

			setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamB[device_id] >> >(_fp_value_x_device[device_id], _event_count_x[device_id]);
			setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamC[device_id] >> >(_fp_value_y_device[device_id], _event_count_y[device_id]);
			if (export_likelihood_flag){
				setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamB[device_id] >> >(_fp_value2_x_device[device_id], _event_count_x[device_id]);
				setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamC[device_id] >> >(_fp_value2_y_device[device_id], _event_count_y[device_id]);
			}

			if (_TOF_mode == TOF){
				//TOF version
				printf("starting TOF kernels.\n");

				checkCudaErrors(cudaStreamWaitEvent(streamB[device_id], eventA[device_id], 0));
				checkCudaErrors(cudaEventRecord(eventStart1[device_id], streamB[device_id]));
				_TOF_fproj_lst_cuda_x_kernel << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_current_image_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
				if (export_likelihood_flag){
					//_fproj_atten_lst_cuda_x_kernel << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_atten_image_device[device_id], _fp_value2_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
					vector_sum_of_log << <dimGrid, dimBlock, 0, streamB[device_id] >> >(_fp_value_x_device[device_id], _event_count_x[device_id], _nloglikelihood_value[device_id]);
				}


				checkCudaErrors(cudaStreamWaitEvent(streamC[device_id], eventA[device_id], 0));
				checkCudaErrors(cudaEventRecord(eventStart2[device_id], streamC[device_id]));
				_TOF_fproj_lst_cuda_y_kernel << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_current_image_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
				if (export_likelihood_flag){
					//_fproj_atten_lst_cuda_y_kernel << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_atten_image_device[device_id], _fp_value2_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
					vector_sum_of_log << <dimGrid, dimBlock, 0, streamC[device_id] >> >(_fp_value_y_device[device_id], _event_count_y[device_id], _nloglikelihood_value[device_id]);
				}


				_TOF_b_ratio_proj_lst_cuda_x_kernel << <dimGrid_b_x, dimBlock_b_x, 0, streamB[device_id] >> >(_update_factor_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
				//cudaEventRecord(eventB[device_id], streamB[device_id]);
				checkCudaErrors(cudaEventRecord(eventStop1[device_id], streamB[device_id]));

				_TOF_b_ratio_proj_lst_cuda_y_kernel << <dimGrid_b_y, dimBlock_b_y, 0, streamC[device_id] >> >(_update_factor_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
				//cudaEventRecord(eventC[device_id], streamC[device_id]);
				checkCudaErrors(cudaEventRecord(eventStop2[device_id], streamC[device_id]));
			}
			else{
				//NON-TOF version
				printf("starting NON-TOF kernels.\n");

				checkCudaErrors(cudaStreamWaitEvent(streamB[device_id], eventA[device_id], 0));
				checkCudaErrors(cudaEventRecord(eventStart1[device_id], streamB[device_id]));
				_fproj_lst_cuda_x_kernel << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_current_image_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
				if (export_likelihood_flag){
					//_fproj_atten_lst_cuda_x_kernel << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_atten_image_device[device_id], _fp_value2_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
					vector_sum_of_log << <dimGrid, dimBlock, 0, streamB[device_id] >> >(_fp_value_x_device[device_id], _event_count_x[device_id], _nloglikelihood_value[device_id]);
				}


				checkCudaErrors(cudaStreamWaitEvent(streamC[device_id], eventA[device_id], 0));
				checkCudaErrors(cudaEventRecord(eventStart2[device_id], streamC[device_id]));
				_fproj_lst_cuda_y_kernel << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_current_image_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
				if (export_likelihood_flag){
					//_fproj_atten_lst_cuda_y_kernel << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_atten_image_device[device_id], _fp_value2_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
					vector_sum_of_log << <dimGrid, dimBlock, 0, streamC[device_id] >> >(_fp_value_y_device[device_id], _event_count_y[device_id], _nloglikelihood_value[device_id]);
				}


				_b_ratio_proj_lst_cuda_x_kernel << <dimGrid_b_x, dimBlock_b_x, 0, streamB[device_id] >> >(_update_factor_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
				//cudaEventRecord(eventB[device_id], streamB[device_id]);
				checkCudaErrors(cudaEventRecord(eventStop1[device_id], streamB[device_id]));

				_b_ratio_proj_lst_cuda_y_kernel << <dimGrid_b_y, dimBlock_b_y, 0, streamC[device_id] >> >(_update_factor_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
				//cudaEventRecord(eventC[device_id], streamC[device_id]);
				checkCudaErrors(cudaEventRecord(eventStop2[device_id], streamC[device_id]));
			}

		}

		for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
			checkCudaErrors(cudaSetDevice(device_id));

			/*
			checkCudaErrors(cudaStreamSynchronize(streamA[device_id]));
			checkCudaErrors(cudaStreamSynchronize(streamB[device_id]));
			checkCudaErrors(cudaStreamSynchronize(streamC[device_id]));
			*/
			
			checkCudaErrors(cudaEventSynchronize(eventStart1[device_id]));
			checkCudaErrors(cudaEventSynchronize(eventStart2[device_id]));
			checkCudaErrors(cudaEventSynchronize(eventStop1[device_id]));
			checkCudaErrors(cudaEventSynchronize(eventStop2[device_id]));
			

			checkCudaErrors(cudaEventElapsedTime(&milliseconds1, eventStart1[device_id], eventStop1[device_id]));
			checkCudaErrors(cudaEventElapsedTime(&milliseconds2, eventStart2[device_id], eventStop2[device_id]));

			printf("\n.......................Wrapping up for device %d........................\n", device_id);
			printf("Time for X-dominant projections: %f seconds\n", milliseconds1 / 1000);
			printf("Time for Y-dominant projections: %f seconds\n", milliseconds2 / 1000);

			checkCudaErrors(cudaEventDestroy(eventA[device_id]));
			/*
			checkCudaErrors(cudaEventDestroy(eventB[device_id]));
			checkCudaErrors(cudaEventDestroy(eventC[device_id]));
			*/

			
			checkCudaErrors(cudaEventDestroy(eventStart1[device_id]));
			checkCudaErrors(cudaEventDestroy(eventStart2[device_id]));
			checkCudaErrors(cudaEventDestroy(eventStop1[device_id]));
			checkCudaErrors(cudaEventDestroy(eventStop2[device_id]));

			checkCudaErrors(cudaStreamDestroy(streamA[device_id]));
			checkCudaErrors(cudaStreamDestroy(streamB[device_id]));
			checkCudaErrors(cudaStreamDestroy(streamC[device_id]));

			//cudaDeviceSynchronize();
		}
		cudaHostUnregister(&parameters_host);
		
		printf("\n... start clean up data\n\n");
		Release_data();
		printf("\n... done clean up data\n\n");

		

		printf("\n\n\n\n.......................Start II data projection.........................\n");
		//II events


		for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
			data_part_index = device_id - _num_gpu_start;
			if (device_id < _num_gpu_end){
				Setup_data(detector, movement, data, data_part_index*device_data_length, (data_part_index + 1)*device_data_length - 1, II, device_id);
			}
			else{
				Setup_data(detector, movement, data, data_part_index*device_data_length, data.GetDataCount(ALL) - 1, II, device_id);
			}
		}
		printf("II data copied to device memory\n");



		parameters_host.FWHM_alpha = parameters_host.FWHM_alpha_ii;
		parameters_host.FWHM_sigma_inv = parameters_host.FWHM_sigma_inv_ii;
		parameters_host.RANGE1 = (int)(parameters_host.FWHM_ii / parameters_host.X_SAMP);
		parameters_host.RANGE2 = (int)(parameters_host.FWHM_ii / parameters_host.Z_SAMP);
		if (parameters_host.RANGE1 < 1)
			parameters_host.RANGE1 = 1;
		if (parameters_host.RANGE2 < 1)
			parameters_host.RANGE2 = 1;
		printf("H_CENTER_X = %f H_CENTER_Y = %f V_CENTER = %f\n", parameters_host.H_CENTER_X, parameters_host.H_CENTER_Y, parameters_host.V_CENTER);
		printf("RANGE1 = %d RANGE2 = %d FWHM_II = %f\n", parameters_host.RANGE1, parameters_host.RANGE2, parameters_host.FWHM_ii);
		cudaHostRegister(&parameters_host, sizeof(PARAMETERS_IN_DEVICE_t), 0);

		for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
			checkCudaErrors(cudaSetDevice(device_id));
			checkCudaErrors(cudaDeviceSynchronize());


			checkCudaErrors(cudaStreamCreate(&streamA[device_id]));
			checkCudaErrors(cudaStreamCreate(&streamB[device_id]));
			checkCudaErrors(cudaStreamCreate(&streamC[device_id]));
			checkCudaErrors(cudaEventCreateWithFlags(&eventA[device_id], cudaEventDisableTiming));
			//checkCudaErrors(cudaEventCreate(&eventB[device_id]));
			//checkCudaErrors(cudaEventCreate(&eventC[device_id]));

			checkCudaErrors(cudaEventCreate(&eventStart1[device_id]));
			checkCudaErrors(cudaEventCreate(&eventStart2[device_id]));
			checkCudaErrors(cudaEventCreate(&eventStop1[device_id]));
			checkCudaErrors(cudaEventCreate(&eventStop2[device_id]));
		}

		for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
			cudaSetDevice(device_id);
			printf("\n\n\n\n.......................Start projection on Device %d.........................\n", device_id);

			checkCudaErrors(cudaMemcpyAsync(_parameters_device[device_id], &parameters_host, sizeof(PARAMETERS_IN_DEVICE_t), cudaMemcpyDefault, streamA[device_id]));
			checkCudaErrors(cudaEventRecord(eventA[device_id], streamA[device_id]));

			setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamB[device_id] >> >(_fp_value_x_device[device_id], _event_count_x[device_id]);
			setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamC[device_id] >> >(_fp_value_y_device[device_id], _event_count_y[device_id]);
			if (export_likelihood_flag){
				//setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamB[device_id] >> >(_fp_value2_x_device[device_id], _event_count_x[device_id]);
				//setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamC[device_id] >> >(_fp_value2_y_device[device_id], _event_count_y[device_id]);
			}

			if (_TOF_mode == TOF){
				//TOF version
				printf("starting TOF kernels.\n");

				checkCudaErrors(cudaStreamWaitEvent(streamB[device_id], eventA[device_id], 0));
				checkCudaErrors(cudaEventRecord(eventStart1[device_id], streamB[device_id]));
				_TOF_fproj_lst_cuda_x_kernel << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_current_image_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
				if (export_likelihood_flag){
					//_fproj_atten_lst_cuda_x_kernel << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_atten_image_device[device_id], _fp_value2_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
					vector_sum_of_log << <dimGrid, dimBlock, 0, streamB[device_id] >> >(_fp_value_x_device[device_id], _event_count_x[device_id], _nloglikelihood_value[device_id]);
				}


				checkCudaErrors(cudaStreamWaitEvent(streamC[device_id], eventA[device_id], 0));
				checkCudaErrors(cudaEventRecord(eventStart2[device_id], streamC[device_id]));
				_TOF_fproj_lst_cuda_y_kernel << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_current_image_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
				if (export_likelihood_flag){
					//_fproj_atten_lst_cuda_y_kernel << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_atten_image_device[device_id], _fp_value2_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
					vector_sum_of_log << <dimGrid, dimBlock, 0, streamC[device_id] >> >(_fp_value_y_device[device_id], _event_count_y[device_id], _nloglikelihood_value[device_id]);
				}


				_TOF_b_ratio_proj_lst_cuda_x_kernel << <dimGrid_b_x, dimBlock_b_x, 0, streamB[device_id] >> >(_update_factor_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
				//cudaEventRecord(eventB[device_id], streamB[device_id]);
				checkCudaErrors(cudaEventRecord(eventStop1[device_id], streamB[device_id]));

				_TOF_b_ratio_proj_lst_cuda_y_kernel << <dimGrid_b_y, dimBlock_b_y, 0, streamC[device_id] >> >(_update_factor_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
				//cudaEventRecord(eventC[device_id], streamC[device_id]);
				checkCudaErrors(cudaEventRecord(eventStop2[device_id], streamC[device_id]));
			}
			else{
				//NON-TOF version
				printf("starting NON-TOF kernels.\n");

				checkCudaErrors(cudaStreamWaitEvent(streamB[device_id], eventA[device_id], 0));
				checkCudaErrors(cudaEventRecord(eventStart1[device_id], streamB[device_id]));
				_fproj_lst_cuda_x_kernel << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_current_image_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
				if (export_likelihood_flag){
					//_fproj_atten_lst_cuda_x_kernel << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_atten_image_device[device_id], _fp_value2_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
					vector_sum_of_log << <dimGrid, dimBlock, 0, streamB[device_id] >> >(_fp_value_x_device[device_id], _event_count_x[device_id], _nloglikelihood_value[device_id]);
				}


				checkCudaErrors(cudaStreamWaitEvent(streamC[device_id], eventA[device_id], 0));
				checkCudaErrors(cudaEventRecord(eventStart2[device_id], streamC[device_id]));
				_fproj_lst_cuda_y_kernel << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_current_image_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
				if (export_likelihood_flag){
					//_fproj_atten_lst_cuda_y_kernel << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_atten_image_device[device_id], _fp_value2_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
					vector_sum_of_log << <dimGrid, dimBlock, 0, streamC[device_id] >> >(_fp_value_y_device[device_id], _event_count_y[device_id], _nloglikelihood_value[device_id]);
				}


				_b_ratio_proj_lst_cuda_x_kernel << <dimGrid_b_x, dimBlock_b_x, 0, streamB[device_id] >> >(_update_factor_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
				//cudaEventRecord(eventB[device_id], streamB[device_id]);
				checkCudaErrors(cudaEventRecord(eventStop1[device_id], streamB[device_id]));

				_b_ratio_proj_lst_cuda_y_kernel << <dimGrid_b_y, dimBlock_b_y, 0, streamC[device_id] >> >(_update_factor_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
				//cudaEventRecord(eventC[device_id], streamC[device_id]);
				checkCudaErrors(cudaEventRecord(eventStop2[device_id], streamC[device_id]));
			}

		}

		for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
			checkCudaErrors(cudaSetDevice(device_id)); 

				/*
				checkCudaErrors(cudaStreamSynchronize(streamA[device_id]));
				checkCudaErrors(cudaStreamSynchronize(streamB[device_id]));
				checkCudaErrors(cudaStreamSynchronize(streamC[device_id]));
				*/

				checkCudaErrors(cudaEventSynchronize(eventStart1[device_id]));
			checkCudaErrors(cudaEventSynchronize(eventStart2[device_id]));
			checkCudaErrors(cudaEventSynchronize(eventStop1[device_id]));
			checkCudaErrors(cudaEventSynchronize(eventStop2[device_id]));


			checkCudaErrors(cudaEventElapsedTime(&milliseconds1, eventStart1[device_id], eventStop1[device_id]));
			checkCudaErrors(cudaEventElapsedTime(&milliseconds2, eventStart2[device_id], eventStop2[device_id]));

			printf("\n.......................Wrapping up for device %d........................\n", device_id);
			printf("Time for X-dominant projections: %f seconds\n", milliseconds1 / 1000);
			printf("Time for Y-dominant projections: %f seconds\n", milliseconds2 / 1000);

			checkCudaErrors(cudaEventDestroy(eventA[device_id]));
			/*
			checkCudaErrors(cudaEventDestroy(eventB[device_id]));
			checkCudaErrors(cudaEventDestroy(eventC[device_id]));
			*/


			checkCudaErrors(cudaEventDestroy(eventStart1[device_id]));
			checkCudaErrors(cudaEventDestroy(eventStart2[device_id]));
			checkCudaErrors(cudaEventDestroy(eventStop1[device_id]));
			checkCudaErrors(cudaEventDestroy(eventStop2[device_id]));

			checkCudaErrors(cudaStreamDestroy(streamA[device_id]));
			checkCudaErrors(cudaStreamDestroy(streamB[device_id]));
			checkCudaErrors(cudaStreamDestroy(streamC[device_id]));

			//cudaDeviceSynchronize();
		}
		cudaHostUnregister(&parameters_host);

		printf("\n... start clean up data\n\n");
		Release_data();
		printf("\n... done clean up data\n\n");



		UpdateFactor.SetValue(0.0f);
		image_host = (float*)malloc(parameters_host.NUM_XYZ*sizeof(float));
		//copy the update factor back to host memory
		for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
			cudaSetDevice(device_id);
			cudaDeviceSynchronize();
			cudaMemcpy(image_host, _update_factor_device[device_id], parameters_host.NUM_XYZ*sizeof(float), cudaMemcpyDefault);
			UpdateFactor.AddFromMem(image_host);
		}

		if (export_likelihood_flag){

			
			float temp;
			for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
				cudaSetDevice(device_id);
				cudaDeviceSynchronize();
				cudaMemcpy(&temp, _nloglikelihood_value[device_id], sizeof(float), cudaMemcpyDefault);
				cudaFree(_nloglikelihood_value[device_id]);
				cudaDeviceSynchronize();
				nloglikelihood_value_host += temp;
			}

			nloglikelihood = -nloglikelihood_value_host;
		}
		else{
			nloglikelihood = 0;
		}


		//Put the image_host (update factor) values to UpdateFactor array
		
		printf("\n\n\n\n.......................End projection.........................\n");
	

	free(image_host);


}


void
cuda_em_recon::ComputeUpdateFactorScatter(PET_geometry& detector, PET_movement& movement, PET_data_scatter& data, ImageArray<float>& Image, ImageArray<float>& UpdateFactor, ImageArray<float>& Attenuation_Image, float &nloglikelihood){


	cudaStream_t streamA[MAX_GPU], streamB[MAX_GPU], streamC[MAX_GPU], streamD[MAX_GPU];
	cudaEvent_t eventA[MAX_GPU], eventB[MAX_GPU], eventC[MAX_GPU], eventD[MAX_GPU];

	float milliseconds1 = 0.0f;
	float milliseconds2 = 0.0f;
	float milliseconds3 = 0.0f;

	dim3 dimBlock_f_x(BlockSize_forward_x);
	dim3 dimGrid_f_x(GridSize_forward_x);

	cudaEvent_t eventStart1[MAX_GPU], eventStart2[MAX_GPU], eventStart3[MAX_GPU], eventStop1[MAX_GPU], eventStop2[MAX_GPU], eventStop3[MAX_GPU];

	int device_id, device_data_length;
	int data_part_index;
	float* image_host;

	string filename;
	stringstream sstm;
	FILE *output_file;

	dim3 dimBlock_f_y(BlockSize_forward_y);
	dim3 dimGrid_f_y(GridSize_forward_y);

	dim3 dimBlock_b_x(BlockSize_backward_x);
	dim3 dimGrid_b_x(GridSize_backward_x);

	dim3 dimBlock_b_y(BlockSize_backward_y);
	dim3 dimGrid_b_y(GridSize_backward_y);



	dim3 dimBlock(1024);
	dim3 dimGrid(64);

	//copy image to all GPUs
	//cudaHostRegister(Image._image, parameters_host.NUM_XYZ*sizeof(float), cudaHostAllocPortable);
	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		checkCudaErrors(cudaSetDevice(device_id));

		checkCudaErrors(cudaMemcpy(_current_image_device[device_id], Image._image, parameters_host.NUM_XYZ*sizeof(float), cudaMemcpyDefault));

		checkCudaErrors(cudaMemcpy(_current2_image_device[device_id], Attenuation_Image._image, parameters_host.NUM_XYZ*sizeof(float), cudaMemcpyDefault)); 

		//setting the update factor to zero before projection
		setImageToZero_kernel <<<dimGrid, dimBlock >>>(_update_factor_device[device_id], _parameters_device[device_id]);

	}
	//cudaHostUnregister(Image._image);

	float nloglikelihood_value_host = 0;
	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		checkCudaErrors(cudaSetDevice(device_id));
		checkCudaErrors(cudaMalloc((void**)&_nloglikelihood_value[device_id], sizeof(float)));
		checkCudaErrors(cudaMemcpy(_nloglikelihood_value[device_id], &nloglikelihood_value_host, sizeof(float), cudaMemcpyDefault));
	}



	printf("\n\n\n\n.......................Start SS data projection.........................\n");
	//SS events

	//copy data to all GPUs
	device_data_length = data.GetDataCount(ALL) / (_num_gpu_end - _num_gpu_start + 1);

	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		data_part_index = device_id - _num_gpu_start;
		if (device_id < _num_gpu_end){
			std::cout << "Copying data from " << data_part_index*device_data_length << " to " << (data_part_index + 1)*device_data_length - 1 << " to device " << device_id << std::endl;
			Setup_data_scatter(detector, movement, data, data_part_index*device_data_length, (data_part_index + 1)*device_data_length - 1, SS, device_id);
			cout << "\n SS";
		}
		else{
			std::cout << "Copying data from " << data_part_index*device_data_length << " to " << data.GetDataCount(ALL) << " to device " << device_id << std::endl;
			Setup_data_scatter(detector, movement, data, data_part_index*device_data_length, data.GetDataCount(ALL) - 1, SS, device_id);
			cout << "\n ALL";
		}
	}
	printf("ALL SS data copied to device memory\n");

	parameters_host.FWHM_alpha = parameters_host.FWHM_alpha_ss;
	parameters_host.FWHM_sigma_inv = parameters_host.FWHM_sigma_inv_ss;
	parameters_host.RANGE1 = (int)(parameters_host.FWHM_ss / parameters_host.X_SAMP);
	parameters_host.RANGE2 = (int)(parameters_host.FWHM_ss / parameters_host.Z_SAMP);
	printf("H_CENTER_X = %f H_CENTER_Y = %f V_CENTER = %f\n", parameters_host.H_CENTER_X, parameters_host.H_CENTER_Y, parameters_host.V_CENTER);
	printf("RANGE1 = %d RANGE2 = %d FWHM_SS = %f\n", parameters_host.RANGE1, parameters_host.RANGE2, parameters_host.FWHM_ss);
	cudaHostRegister(&parameters_host, sizeof(PARAMETERS_IN_DEVICE_t), 0);

	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		checkCudaErrors(cudaSetDevice(device_id));
		checkCudaErrors(cudaDeviceSynchronize());


		checkCudaErrors(cudaStreamCreate(&streamA[device_id]));
		checkCudaErrors(cudaStreamCreate(&streamB[device_id]));
		checkCudaErrors(cudaStreamCreate(&streamC[device_id]));

		checkCudaErrors(cudaEventCreateWithFlags(&eventA[device_id], cudaEventDisableTiming));
		//checkCudaErrors(cudaEventCreate(&eventB[device_id]));
		//checkCudaErrors(cudaEventCreate(&eventC[device_id]));

		checkCudaErrors(cudaEventCreate(&eventStart1[device_id]));
		checkCudaErrors(cudaEventCreate(&eventStart2[device_id]));

		checkCudaErrors(cudaEventCreate(&eventStop1[device_id]));
		checkCudaErrors(cudaEventCreate(&eventStop2[device_id]));

	}

	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		cudaSetDevice(device_id);
		printf("\n\n\n\n.......................Start projection on Device %d.........................\n", device_id);



		checkCudaErrors(cudaMemcpyAsync(_parameters_device[device_id], &parameters_host, sizeof(PARAMETERS_IN_DEVICE_t), cudaMemcpyDefault, streamA[device_id]));
		checkCudaErrors(cudaEventRecord(eventA[device_id], streamA[device_id]));

		setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamB[device_id] >> >(_fp_value_x_device[device_id], _event_count_x[device_id]);
		setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamC[device_id] >> >(_fp_value_y_device[device_id], _event_count_y[device_id]);

		//if (atten_flag_fp){

			setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamB[device_id] >> >(_fp_value2_x_device[device_id], _event_count_x[device_id]);
			setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamC[device_id] >> >(_fp_value2_y_device[device_id], _event_count_y[device_id]);

		//}


		if (export_likelihood_flag){
			//setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamB[device_id] >> >(_fp_value2_x_device[device_id], _event_count_x[device_id]);
			//setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamC[device_id] >> >(_fp_value2_y_device[device_id], _event_count_y[device_id]);
		}

		if (_TOF_mode == TOF){
			//TOF version
			printf("starting TOF kernels.\n");

			checkCudaErrors(cudaStreamWaitEvent(streamB[device_id], eventA[device_id], 0));
			checkCudaErrors(cudaEventRecord(eventStart1[device_id], streamB[device_id]));

			if (atten_flag_fp){
				_fproj_atten_lst_cuda_x_kernel_scatter << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_current2_image_device[device_id], _fp_value2_x_device[device_id], _events_x_dominant_global_mem_scatter[device_id], _parameters_device[device_id]);
				_TOF_fproj_lst_cuda_x_kernel_scatter_atten << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_current_image_device[device_id], _fp_value_x_device[device_id], _fp_value2_x_device[device_id], _events_x_dominant_global_mem_scatter[device_id], _parameters_device[device_id]);
				if (export_likelihood_flag){
					//_fproj_atten_lst_cuda_x_kernel << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_atten_image_device[device_id], _fp_value2_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
					vector_sum_of_log << <dimGrid, dimBlock, 0, streamB[device_id] >> >(_fp_value_x_device[device_id], _event_count_x[device_id], _nloglikelihood_value[device_id]);
				}
			}
			else{
				_TOF_fproj_lst_cuda_x_kernel_scatter << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_current_image_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem_scatter[device_id], _parameters_device[device_id]);
				
			}


			checkCudaErrors(cudaStreamWaitEvent(streamC[device_id], eventA[device_id], 0));
			checkCudaErrors(cudaEventRecord(eventStart2[device_id], streamC[device_id]));

			if (atten_flag_fp){
				_fproj_atten_lst_cuda_y_kernel_scatter << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_current2_image_device[device_id], _fp_value2_y_device[device_id], _events_y_dominant_global_mem_scatter[device_id], _parameters_device[device_id]);

				_TOF_fproj_lst_cuda_y_kernel_scatter_atten << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_current_image_device[device_id], _fp_value_y_device[device_id], _fp_value2_y_device[device_id], _events_y_dominant_global_mem_scatter[device_id], _parameters_device[device_id]);
				if (export_likelihood_flag){
					//_fproj_atten_lst_cuda_y_kernel << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_atten_image_device[device_id], _fp_value2_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
					vector_sum_of_log << <dimGrid, dimBlock, 0, streamC[device_id] >> >(_fp_value_y_device[device_id], _event_count_y[device_id], _nloglikelihood_value[device_id]);
				}
			}
			else{

				_TOF_fproj_lst_cuda_y_kernel_scatter<< <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_current_image_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem_scatter[device_id], _parameters_device[device_id]);
			

			}

			if (atten_flag_bp){

				_TOF_b_ratio_proj_lst_cuda_x_kernel_scatter_atten << <dimGrid_b_x, dimBlock_b_x, 0, streamB[device_id] >> >(_update_factor_device[device_id], _fp_value_x_device[device_id], _fp_value2_x_device[device_id], _events_x_dominant_global_mem_scatter[device_id], _parameters_device[device_id]);
				//cudaEventRecord(eventB[device_id], streamB[device_id]);
				checkCudaErrors(cudaEventRecord(eventStop1[device_id], streamB[device_id]));

				_TOF_b_ratio_proj_lst_cuda_y_kernel_scatter_atten << <dimGrid_b_y, dimBlock_b_y, 0, streamC[device_id] >> >(_update_factor_device[device_id], _fp_value_y_device[device_id], _fp_value2_y_device[device_id], _events_y_dominant_global_mem_scatter[device_id], _parameters_device[device_id]);
				//cudaEventRecord(eventC[device_id], streamC[device_id]);
				checkCudaErrors(cudaEventRecord(eventStop2[device_id], streamC[device_id]));
			}
			else{


				_TOF_b_ratio_proj_lst_cuda_x_kernel_scatter << <dimGrid_b_x, dimBlock_b_x, 0, streamB[device_id] >> >(_update_factor_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem_scatter[device_id], _parameters_device[device_id]);
				//cudaEventRecord(eventB[device_id], streamB[device_id]);
				checkCudaErrors(cudaEventRecord(eventStop1[device_id], streamB[device_id]));

				_TOF_b_ratio_proj_lst_cuda_y_kernel_scatter << <dimGrid_b_y, dimBlock_b_y, 0, streamC[device_id] >> >(_update_factor_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem_scatter[device_id], _parameters_device[device_id]);
				//cudaEventRecord(eventC[device_id], streamC[device_id]);
				checkCudaErrors(cudaEventRecord(eventStop2[device_id], streamC[device_id]));
			}


		}
		else{
			//NON-TOF version
			printf("starting NON-TOF kernels.\n");

			checkCudaErrors(cudaStreamWaitEvent(streamB[device_id], eventA[device_id], 0));
			checkCudaErrors(cudaEventRecord(eventStart1[device_id], streamB[device_id]));
			_fproj_lst_cuda_x_kernel << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_current_image_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
			if (export_likelihood_flag){
				//_fproj_atten_lst_cuda_x_kernel << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_atten_image_device[device_id], _fp_value2_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
				vector_sum_of_log << <dimGrid, dimBlock, 0, streamB[device_id] >> >(_fp_value_x_device[device_id], _event_count_x[device_id], _nloglikelihood_value[device_id]);
			}


			checkCudaErrors(cudaStreamWaitEvent(streamC[device_id], eventA[device_id], 0));
			checkCudaErrors(cudaEventRecord(eventStart2[device_id], streamC[device_id]));
			_fproj_lst_cuda_y_kernel << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_current_image_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
			if (export_likelihood_flag){
				//_fproj_atten_lst_cuda_y_kernel << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_atten_image_device[device_id], _fp_value2_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
				vector_sum_of_log << <dimGrid, dimBlock, 0, streamC[device_id] >> >(_fp_value_y_device[device_id], _event_count_y[device_id], _nloglikelihood_value[device_id]);
			}





			_b_ratio_proj_lst_cuda_x_kernel << <dimGrid_b_x, dimBlock_b_x, 0, streamB[device_id] >> >(_update_factor_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
			//cudaEventRecord(eventB[device_id], streamB[device_id]);
			checkCudaErrors(cudaEventRecord(eventStop1[device_id], streamB[device_id]));

			_b_ratio_proj_lst_cuda_y_kernel << <dimGrid_b_y, dimBlock_b_y, 0, streamC[device_id] >> >(_update_factor_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
			//cudaEventRecord(eventC[device_id], streamC[device_id]);
			checkCudaErrors(cudaEventRecord(eventStop2[device_id], streamC[device_id]));

		}

	}

	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		checkCudaErrors(cudaSetDevice(device_id));

		/*
		checkCudaErrors(cudaStreamSynchronize(streamA[device_id]));
		checkCudaErrors(cudaStreamSynchronize(streamB[device_id]));
		checkCudaErrors(cudaStreamSynchronize(streamC[device_id]));
		*/

		checkCudaErrors(cudaEventSynchronize(eventStart1[device_id]));
		checkCudaErrors(cudaEventSynchronize(eventStart2[device_id]));

		checkCudaErrors(cudaEventSynchronize(eventStop1[device_id]));
		checkCudaErrors(cudaEventSynchronize(eventStop2[device_id]));



		checkCudaErrors(cudaEventElapsedTime(&milliseconds1, eventStart1[device_id], eventStop1[device_id]));
		checkCudaErrors(cudaEventElapsedTime(&milliseconds2, eventStart2[device_id], eventStop2[device_id]));


		printf("\n.......................Wrapping up for device %d........................\n", device_id);
		printf("Time for X-dominant projections: %f seconds\n", milliseconds1 / 1000);
		printf("Time for Y-dominant projections: %f seconds\n", milliseconds2 / 1000);


		checkCudaErrors(cudaEventDestroy(eventA[device_id]));
		/*
		checkCudaErrors(cudaEventDestroy(eventB[device_id]));
		checkCudaErrors(cudaEventDestroy(eventC[device_id]));
		*/

		checkCudaErrors(cudaEventDestroy(eventStart1[device_id]));
		checkCudaErrors(cudaEventDestroy(eventStart2[device_id]));

		checkCudaErrors(cudaEventDestroy(eventStop1[device_id]));
		checkCudaErrors(cudaEventDestroy(eventStop2[device_id]));


		checkCudaErrors(cudaStreamDestroy(streamA[device_id]));
		checkCudaErrors(cudaStreamDestroy(streamB[device_id]));
		checkCudaErrors(cudaStreamDestroy(streamC[device_id]));


		//cudaDeviceSynchronize();
	}

	// Check forward projected values
	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		int size1 = _event_count_y[device_id];
		int size2 = _event_count_x[device_id];
		float sum_fp = 0, mean_fp = 0;
		float max_fp = 0;
		float* fpvalue1_hosty = new float[size1]();
		
		
		float* fpvalue1_hostx = new float[size2]();
		
		cudaSetDevice(device_id);
		cudaDeviceSynchronize();
		
		cudaMemcpy(fpvalue1_hosty, _fp_value_y_device[device_id], size1*sizeof(float), cudaMemcpyDefault);
		cudaMemcpy(fpvalue1_hostx, _fp_value_x_device[device_id], size2*sizeof(float), cudaMemcpyDefault);
		for (int i = 0; i < size1; i++){
			//cout << "\n fpvalues1 y " << fpvalue1_hosty[i];
			sum_fp += fpvalue1_hosty[i];
			if (max_fp < fpvalue1_hosty[i]){
				max_fp = fpvalue1_hosty[i];
			}
		}
		for (int j = 0; j < size2; j++){
			//cout << "\n fpvalues1 x " << fpvalue1_hostx[j];
			sum_fp += fpvalue1_hostx[j];
			if (max_fp < fpvalue1_hostx[j]){
				max_fp = fpvalue1_hostx[j];
			}
		}
		for (int i = 0; i < 100; i++){
			cout << "\n fpvalues x " << fpvalue1_hostx[i]<< " fpvalues y " << fpvalue1_hosty[i];
		}
		mean_fp = sum_fp / (size1 + size2);
		cout << "\n Sum fp for device " << device_id << " : " << sum_fp << " Mean fp : " << mean_fp << " Max fp : " << max_fp;
	}

	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		int size1 = _event_count_y[device_id];
		int size2 = _event_count_x[device_id];
		float sum_fp = 0, mean_fp = 0;
		float* fpvalue2_hosty = new float[size1];
		float* fpvalue2_hostx = new float[size2];
		cudaSetDevice(device_id);
		cudaDeviceSynchronize();

		cudaMemcpy(fpvalue2_hosty, _fp_value2_y_device[device_id], size1*sizeof(float), cudaMemcpyDefault);
		cudaMemcpy(fpvalue2_hostx, _fp_value2_x_device[device_id], size2*sizeof(float), cudaMemcpyDefault);
		for (int i = 0; i < size1; i++){
			//cout << "\n fpvalues1 y " << fpvalue1_hosty[i];
			sum_fp += fpvalue2_hosty[i];
		}
		for (int j = 0; j < size2; j++){
			//cout << "\n fpvalues1 x " << fpvalue1_hostx[j];
			sum_fp += fpvalue2_hostx[j];
		}
		mean_fp = sum_fp / (size1 + size2);
		cout << "\n Sum fp2 for device " << device_id << " : " << sum_fp << " Mean fp : " << mean_fp;
	}

	cudaHostUnregister(&parameters_host);

	printf("\n... start clean up data\n\n");
	Release_data_scatter();
	printf("\n... done clean up data\n\n");



	printf("\n\n\n\n.......................Start IS data projection.........................\n");
	//IS events


	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		data_part_index = device_id - _num_gpu_start;
		if (device_id < _num_gpu_end){
			Setup_data_scatter(detector, movement, data, data_part_index*device_data_length, (data_part_index + 1)*device_data_length - 1, IS, device_id);
		}
		else{
			Setup_data_scatter(detector, movement, data, data_part_index*device_data_length, data.GetDataCount(ALL) - 1, IS, device_id);
		}
	}
	printf("IS data copied to device memory\n");



	parameters_host.FWHM_alpha = parameters_host.FWHM_alpha_is;
	parameters_host.FWHM_sigma_inv = parameters_host.FWHM_sigma_inv_is;
	parameters_host.RANGE1 = (int)(parameters_host.FWHM_is / parameters_host.X_SAMP);
	parameters_host.RANGE2 = (int)(parameters_host.FWHM_is / parameters_host.Z_SAMP);
	if (parameters_host.RANGE1 < 1)
		parameters_host.RANGE1 = 1;
	if (parameters_host.RANGE2 < 1)
		parameters_host.RANGE2 = 1;
	printf("H_CENTER_X = %f H_CENTER_Y = %f V_CENTER = %f\n", parameters_host.H_CENTER_X, parameters_host.H_CENTER_Y, parameters_host.V_CENTER);
	printf("RANGE1 = %d RANGE2 = %d FWHM_IS = %f\n", parameters_host.RANGE1, parameters_host.RANGE2, parameters_host.FWHM_is);
	cudaHostRegister(&parameters_host, sizeof(PARAMETERS_IN_DEVICE_t), 0);

	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		checkCudaErrors(cudaSetDevice(device_id));
		checkCudaErrors(cudaDeviceSynchronize());


		checkCudaErrors(cudaStreamCreate(&streamA[device_id]));
		checkCudaErrors(cudaStreamCreate(&streamB[device_id]));
		checkCudaErrors(cudaStreamCreate(&streamC[device_id]));
		checkCudaErrors(cudaEventCreateWithFlags(&eventA[device_id], cudaEventDisableTiming));
		//checkCudaErrors(cudaEventCreate(&eventB[device_id]));
		//checkCudaErrors(cudaEventCreate(&eventC[device_id]));

		checkCudaErrors(cudaEventCreate(&eventStart1[device_id]));
		checkCudaErrors(cudaEventCreate(&eventStart2[device_id]));
		checkCudaErrors(cudaEventCreate(&eventStop1[device_id]));
		checkCudaErrors(cudaEventCreate(&eventStop2[device_id]));
	}

	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		cudaSetDevice(device_id);
		printf("\n\n\n\n.......................Start projection on Device %d.........................\n", device_id);

		checkCudaErrors(cudaMemcpyAsync(_parameters_device[device_id], &parameters_host, sizeof(PARAMETERS_IN_DEVICE_t), cudaMemcpyDefault, streamA[device_id]));
		checkCudaErrors(cudaEventRecord(eventA[device_id], streamA[device_id]));

		setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamB[device_id] >> >(_fp_value_x_device[device_id], _event_count_x[device_id]);
		setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamC[device_id] >> >(_fp_value_y_device[device_id], _event_count_y[device_id]);
		setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamB[device_id] >> >(_fp_value2_x_device[device_id], _event_count_x[device_id]);
		setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamC[device_id] >> >(_fp_value2_y_device[device_id], _event_count_y[device_id]);

		if (export_likelihood_flag){
			setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamB[device_id] >> >(_fp_value2_x_device[device_id], _event_count_x[device_id]);
			setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamC[device_id] >> >(_fp_value2_y_device[device_id], _event_count_y[device_id]);
		}

		if (_TOF_mode == TOF){
			//TOF version
			printf("starting TOF kernels.\n");

			checkCudaErrors(cudaStreamWaitEvent(streamB[device_id], eventA[device_id], 0));
			checkCudaErrors(cudaEventRecord(eventStart1[device_id], streamB[device_id]));

			_fproj_atten_lst_cuda_x_kernel_scatter << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_current2_image_device[device_id], _fp_value2_x_device[device_id], _events_x_dominant_global_mem_scatter[device_id], _parameters_device[device_id]);
			_TOF_fproj_lst_cuda_x_kernel_scatter_atten << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_current_image_device[device_id], _fp_value_x_device[device_id], _fp_value2_x_device[device_id], _events_x_dominant_global_mem_scatter[device_id], _parameters_device[device_id]);

			//_TOF_fproj_lst_cuda_x_kernel_scatter << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_current_image_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem_scatter[device_id], _parameters_device[device_id]);
			if (export_likelihood_flag){
				//_fproj_atten_lst_cuda_x_kernel << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_atten_image_device[device_id], _fp_value2_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
				vector_sum_of_log << <dimGrid, dimBlock, 0, streamB[device_id] >> >(_fp_value_x_device[device_id], _event_count_x[device_id], _nloglikelihood_value[device_id]);
			}


			checkCudaErrors(cudaStreamWaitEvent(streamC[device_id], eventA[device_id], 0));
			checkCudaErrors(cudaEventRecord(eventStart2[device_id], streamC[device_id]));

			_fproj_atten_lst_cuda_y_kernel_scatter << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_current2_image_device[device_id], _fp_value2_y_device[device_id], _events_y_dominant_global_mem_scatter[device_id], _parameters_device[device_id]);
			_TOF_fproj_lst_cuda_y_kernel_scatter_atten << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_current_image_device[device_id], _fp_value_y_device[device_id], _fp_value2_y_device[device_id], _events_y_dominant_global_mem_scatter[device_id], _parameters_device[device_id]);

			if (export_likelihood_flag){
				//_fproj_atten_lst_cuda_y_kernel << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_atten_image_device[device_id], _fp_value2_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
				vector_sum_of_log << <dimGrid, dimBlock, 0, streamC[device_id] >> >(_fp_value_y_device[device_id], _event_count_y[device_id], _nloglikelihood_value[device_id]);
			}

			_TOF_b_ratio_proj_lst_cuda_x_kernel_scatter_atten << <dimGrid_b_x, dimBlock_b_x, 0, streamB[device_id] >> >(_update_factor_device[device_id], _fp_value_x_device[device_id], _fp_value2_x_device[device_id], _events_x_dominant_global_mem_scatter[device_id], _parameters_device[device_id]);
			//cudaEventRecord(eventB[device_id], streamB[device_id]);
			checkCudaErrors(cudaEventRecord(eventStop1[device_id], streamB[device_id]));

			_TOF_b_ratio_proj_lst_cuda_y_kernel_scatter_atten << <dimGrid_b_y, dimBlock_b_y, 0, streamC[device_id] >> >(_update_factor_device[device_id], _fp_value_y_device[device_id], _fp_value2_y_device[device_id], _events_y_dominant_global_mem_scatter[device_id], _parameters_device[device_id]);
			//cudaEventRecord(eventC[device_id], streamC[device_id]);
			checkCudaErrors(cudaEventRecord(eventStop2[device_id], streamC[device_id]));
		}
		else{
			//NON-TOF version
			printf("starting NON-TOF kernels.\n");

			checkCudaErrors(cudaStreamWaitEvent(streamB[device_id], eventA[device_id], 0));
			checkCudaErrors(cudaEventRecord(eventStart1[device_id], streamB[device_id]));
			_fproj_lst_cuda_x_kernel << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_current_image_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
			if (export_likelihood_flag){
				//_fproj_atten_lst_cuda_x_kernel << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_atten_image_device[device_id], _fp_value2_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
				vector_sum_of_log << <dimGrid, dimBlock, 0, streamB[device_id] >> >(_fp_value_x_device[device_id], _event_count_x[device_id], _nloglikelihood_value[device_id]);
			}


			checkCudaErrors(cudaStreamWaitEvent(streamC[device_id], eventA[device_id], 0));
			checkCudaErrors(cudaEventRecord(eventStart2[device_id], streamC[device_id]));
			_fproj_lst_cuda_y_kernel << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_current_image_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
			if (export_likelihood_flag){
				//_fproj_atten_lst_cuda_y_kernel << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_atten_image_device[device_id], _fp_value2_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
				vector_sum_of_log << <dimGrid, dimBlock, 0, streamC[device_id] >> >(_fp_value_y_device[device_id], _event_count_y[device_id], _nloglikelihood_value[device_id]);
			}


			_b_ratio_proj_lst_cuda_x_kernel << <dimGrid_b_x, dimBlock_b_x, 0, streamB[device_id] >> >(_update_factor_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
			//cudaEventRecord(eventB[device_id], streamB[device_id]);
			checkCudaErrors(cudaEventRecord(eventStop1[device_id], streamB[device_id]));

			_b_ratio_proj_lst_cuda_y_kernel << <dimGrid_b_y, dimBlock_b_y, 0, streamC[device_id] >> >(_update_factor_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
			//cudaEventRecord(eventC[device_id], streamC[device_id]);
			checkCudaErrors(cudaEventRecord(eventStop2[device_id], streamC[device_id]));
		}

	}

	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		checkCudaErrors(cudaSetDevice(device_id));

		/*
		checkCudaErrors(cudaStreamSynchronize(streamA[device_id]));
		checkCudaErrors(cudaStreamSynchronize(streamB[device_id]));
		checkCudaErrors(cudaStreamSynchronize(streamC[device_id]));
		*/

		checkCudaErrors(cudaEventSynchronize(eventStart1[device_id]));
		checkCudaErrors(cudaEventSynchronize(eventStart2[device_id]));
		checkCudaErrors(cudaEventSynchronize(eventStop1[device_id]));
		checkCudaErrors(cudaEventSynchronize(eventStop2[device_id]));


		checkCudaErrors(cudaEventElapsedTime(&milliseconds1, eventStart1[device_id], eventStop1[device_id]));
		checkCudaErrors(cudaEventElapsedTime(&milliseconds2, eventStart2[device_id], eventStop2[device_id]));

		printf("\n.......................Wrapping up for device %d........................\n", device_id);
		printf("Time for X-dominant projections: %f seconds\n", milliseconds1 / 1000);
		printf("Time for Y-dominant projections: %f seconds\n", milliseconds2 / 1000);

		checkCudaErrors(cudaEventDestroy(eventA[device_id]));
		/*
		checkCudaErrors(cudaEventDestroy(eventB[device_id]));
		checkCudaErrors(cudaEventDestroy(eventC[device_id]));
		*/


		checkCudaErrors(cudaEventDestroy(eventStart1[device_id]));
		checkCudaErrors(cudaEventDestroy(eventStart2[device_id]));
		checkCudaErrors(cudaEventDestroy(eventStop1[device_id]));
		checkCudaErrors(cudaEventDestroy(eventStop2[device_id]));

		checkCudaErrors(cudaStreamDestroy(streamA[device_id]));
		checkCudaErrors(cudaStreamDestroy(streamB[device_id]));
		checkCudaErrors(cudaStreamDestroy(streamC[device_id]));

		//cudaDeviceSynchronize();
	}
	cudaHostUnregister(&parameters_host);

	printf("\n... start clean up data\n\n");
	Release_data_scatter();
	printf("\n... done clean up data\n\n");



	printf("\n\n\n\n.......................Start II data projection.........................\n");
	//II events


	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		data_part_index = device_id - _num_gpu_start;
		if (device_id < _num_gpu_end){
			Setup_data_scatter(detector, movement, data, data_part_index*device_data_length, (data_part_index + 1)*device_data_length - 1, II, device_id);
		}
		else{
			Setup_data_scatter(detector, movement, data, data_part_index*device_data_length, data.GetDataCount(ALL) - 1, II, device_id);
		}
	}
	printf("II data copied to device memory\n");



	parameters_host.FWHM_alpha = parameters_host.FWHM_alpha_ii;
	parameters_host.FWHM_sigma_inv = parameters_host.FWHM_sigma_inv_ii;
	parameters_host.RANGE1 = (int)(parameters_host.FWHM_ii / parameters_host.X_SAMP);
	parameters_host.RANGE2 = (int)(parameters_host.FWHM_ii / parameters_host.Z_SAMP);
	if (parameters_host.RANGE1 < 1)
		parameters_host.RANGE1 = 1;
	if (parameters_host.RANGE2 < 1)
		parameters_host.RANGE2 = 1;
	printf("H_CENTER_X = %f H_CENTER_Y = %f V_CENTER = %f\n", parameters_host.H_CENTER_X, parameters_host.H_CENTER_Y, parameters_host.V_CENTER);
	printf("RANGE1 = %d RANGE2 = %d FWHM_IS = %f\n", parameters_host.RANGE1, parameters_host.RANGE2, parameters_host.FWHM_is);
	cudaHostRegister(&parameters_host, sizeof(PARAMETERS_IN_DEVICE_t), 0);

	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		checkCudaErrors(cudaSetDevice(device_id));
		checkCudaErrors(cudaDeviceSynchronize());


		checkCudaErrors(cudaStreamCreate(&streamA[device_id]));
		checkCudaErrors(cudaStreamCreate(&streamB[device_id]));
		checkCudaErrors(cudaStreamCreate(&streamC[device_id]));
		checkCudaErrors(cudaEventCreateWithFlags(&eventA[device_id], cudaEventDisableTiming));
		//checkCudaErrors(cudaEventCreate(&eventB[device_id]));
		//checkCudaErrors(cudaEventCreate(&eventC[device_id]));

		checkCudaErrors(cudaEventCreate(&eventStart1[device_id]));
		checkCudaErrors(cudaEventCreate(&eventStart2[device_id]));
		checkCudaErrors(cudaEventCreate(&eventStop1[device_id]));
		checkCudaErrors(cudaEventCreate(&eventStop2[device_id]));
	}

	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		cudaSetDevice(device_id);
		printf("\n\n\n\n.......................Start projection on Device %d.........................\n", device_id);

		checkCudaErrors(cudaMemcpyAsync(_parameters_device[device_id], &parameters_host, sizeof(PARAMETERS_IN_DEVICE_t), cudaMemcpyDefault, streamA[device_id]));
		checkCudaErrors(cudaEventRecord(eventA[device_id], streamA[device_id]));

		setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamB[device_id] >> >(_fp_value_x_device[device_id], _event_count_x[device_id]);
		setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamC[device_id] >> >(_fp_value_y_device[device_id], _event_count_y[device_id]);
		setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamB[device_id] >> >(_fp_value2_x_device[device_id], _event_count_x[device_id]);
		setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamC[device_id] >> >(_fp_value2_y_device[device_id], _event_count_y[device_id]);

		if (export_likelihood_flag){
			//setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamB[device_id] >> >(_fp_value2_x_device[device_id], _event_count_x[device_id]);
			//setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamC[device_id] >> >(_fp_value2_y_device[device_id], _event_count_y[device_id]);
		}

		if (_TOF_mode == TOF){
			//TOF version
			printf("starting TOF kernels.\n");

			checkCudaErrors(cudaStreamWaitEvent(streamB[device_id], eventA[device_id], 0));
			checkCudaErrors(cudaEventRecord(eventStart1[device_id], streamB[device_id]));
			_fproj_atten_lst_cuda_x_kernel_scatter << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_current2_image_device[device_id], _fp_value2_x_device[device_id], _events_x_dominant_global_mem_scatter[device_id], _parameters_device[device_id]);
			_TOF_fproj_lst_cuda_x_kernel_scatter_atten << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_current_image_device[device_id], _fp_value_x_device[device_id], _fp_value2_x_device[device_id], _events_x_dominant_global_mem_scatter[device_id], _parameters_device[device_id]);
			if (export_likelihood_flag){
				//_fproj_atten_lst_cuda_x_kernel << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_atten_image_device[device_id], _fp_value2_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
				vector_sum_of_log << <dimGrid, dimBlock, 0, streamB[device_id] >> >(_fp_value_x_device[device_id], _event_count_x[device_id], _nloglikelihood_value[device_id]);
			}


			checkCudaErrors(cudaStreamWaitEvent(streamC[device_id], eventA[device_id], 0));
			checkCudaErrors(cudaEventRecord(eventStart2[device_id], streamC[device_id]));
			_fproj_atten_lst_cuda_y_kernel_scatter << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_current2_image_device[device_id], _fp_value2_y_device[device_id], _events_y_dominant_global_mem_scatter[device_id], _parameters_device[device_id]);
			_TOF_fproj_lst_cuda_y_kernel_scatter_atten << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_current_image_device[device_id], _fp_value_y_device[device_id], _fp_value2_y_device[device_id], _events_y_dominant_global_mem_scatter[device_id], _parameters_device[device_id]);
			if (export_likelihood_flag){
				//_fproj_atten_lst_cuda_y_kernel << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_atten_image_device[device_id], _fp_value2_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
				vector_sum_of_log << <dimGrid, dimBlock, 0, streamC[device_id] >> >(_fp_value_y_device[device_id], _event_count_y[device_id], _nloglikelihood_value[device_id]);
			}


			_TOF_b_ratio_proj_lst_cuda_x_kernel_scatter_atten << <dimGrid_b_x, dimBlock_b_x, 0, streamB[device_id] >> >(_update_factor_device[device_id], _fp_value_x_device[device_id], _fp_value2_x_device[device_id], _events_x_dominant_global_mem_scatter[device_id], _parameters_device[device_id]);
			//cudaEventRecord(eventB[device_id], streamB[device_id]);
			checkCudaErrors(cudaEventRecord(eventStop1[device_id], streamB[device_id]));

			_TOF_b_ratio_proj_lst_cuda_y_kernel_scatter_atten << <dimGrid_b_y, dimBlock_b_y, 0, streamC[device_id] >> >(_update_factor_device[device_id], _fp_value_y_device[device_id], _fp_value2_y_device[device_id], _events_y_dominant_global_mem_scatter[device_id], _parameters_device[device_id]);
			//cudaEventRecord(eventC[device_id], streamC[device_id]);
			checkCudaErrors(cudaEventRecord(eventStop2[device_id], streamC[device_id]));
		}
		else{
			//NON-TOF version
			printf("starting NON-TOF kernels.\n");

			checkCudaErrors(cudaStreamWaitEvent(streamB[device_id], eventA[device_id], 0));
			checkCudaErrors(cudaEventRecord(eventStart1[device_id], streamB[device_id]));
			_fproj_lst_cuda_x_kernel << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_current_image_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
			if (export_likelihood_flag){
				//_fproj_atten_lst_cuda_x_kernel << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_atten_image_device[device_id], _fp_value2_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
				vector_sum_of_log << <dimGrid, dimBlock, 0, streamB[device_id] >> >(_fp_value_x_device[device_id], _event_count_x[device_id], _nloglikelihood_value[device_id]);
			}


			checkCudaErrors(cudaStreamWaitEvent(streamC[device_id], eventA[device_id], 0));
			checkCudaErrors(cudaEventRecord(eventStart2[device_id], streamC[device_id]));
			_fproj_lst_cuda_y_kernel << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_current_image_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
			if (export_likelihood_flag){
				//_fproj_atten_lst_cuda_y_kernel << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_atten_image_device[device_id], _fp_value2_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
				vector_sum_of_log << <dimGrid, dimBlock, 0, streamC[device_id] >> >(_fp_value_y_device[device_id], _event_count_y[device_id], _nloglikelihood_value[device_id]);
			}


			_b_ratio_proj_lst_cuda_x_kernel << <dimGrid_b_x, dimBlock_b_x, 0, streamB[device_id] >> >(_update_factor_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
			//cudaEventRecord(eventB[device_id], streamB[device_id]);
			checkCudaErrors(cudaEventRecord(eventStop1[device_id], streamB[device_id]));

			_b_ratio_proj_lst_cuda_y_kernel << <dimGrid_b_y, dimBlock_b_y, 0, streamC[device_id] >> >(_update_factor_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
			//cudaEventRecord(eventC[device_id], streamC[device_id]);
			checkCudaErrors(cudaEventRecord(eventStop2[device_id], streamC[device_id]));
		}

	}

	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		checkCudaErrors(cudaSetDevice(device_id));

		/*
		checkCudaErrors(cudaStreamSynchronize(streamA[device_id]));
		checkCudaErrors(cudaStreamSynchronize(streamB[device_id]));
		checkCudaErrors(cudaStreamSynchronize(streamC[device_id]));
		*/

		checkCudaErrors(cudaEventSynchronize(eventStart1[device_id]));
		checkCudaErrors(cudaEventSynchronize(eventStart2[device_id]));
		checkCudaErrors(cudaEventSynchronize(eventStop1[device_id]));
		checkCudaErrors(cudaEventSynchronize(eventStop2[device_id]));


		checkCudaErrors(cudaEventElapsedTime(&milliseconds1, eventStart1[device_id], eventStop1[device_id]));
		checkCudaErrors(cudaEventElapsedTime(&milliseconds2, eventStart2[device_id], eventStop2[device_id]));

		printf("\n.......................Wrapping up for device %d........................\n", device_id);
		printf("Time for X-dominant projections: %f seconds\n", milliseconds1 / 1000);
		printf("Time for Y-dominant projections: %f seconds\n", milliseconds2 / 1000);

		checkCudaErrors(cudaEventDestroy(eventA[device_id]));
		/*
		checkCudaErrors(cudaEventDestroy(eventB[device_id]));
		checkCudaErrors(cudaEventDestroy(eventC[device_id]));
		*/


		checkCudaErrors(cudaEventDestroy(eventStart1[device_id]));
		checkCudaErrors(cudaEventDestroy(eventStart2[device_id]));
		checkCudaErrors(cudaEventDestroy(eventStop1[device_id]));
		checkCudaErrors(cudaEventDestroy(eventStop2[device_id]));

		checkCudaErrors(cudaStreamDestroy(streamA[device_id]));
		checkCudaErrors(cudaStreamDestroy(streamB[device_id]));
		checkCudaErrors(cudaStreamDestroy(streamC[device_id]));

		//cudaDeviceSynchronize();
	}
	cudaHostUnregister(&parameters_host);

	printf("\n... start clean up data\n\n");
	Release_data_scatter();
	printf("\n... done clean up data\n\n");



	UpdateFactor.SetValue(0.0f);
	image_host = (float*)malloc(parameters_host.NUM_XYZ*sizeof(float));
	//copy the update factor back to host memory
	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		cudaSetDevice(device_id);
		cudaDeviceSynchronize();
		cudaMemcpy(image_host, _update_factor_device[device_id], parameters_host.NUM_XYZ*sizeof(float), cudaMemcpyDefault);
		UpdateFactor.AddFromMem(image_host);
	}

	if (export_likelihood_flag){


		float temp;
		for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
			cudaSetDevice(device_id);
			cudaDeviceSynchronize();
			cudaMemcpy(&temp, _nloglikelihood_value[device_id], sizeof(float), cudaMemcpyDefault);
			cudaFree(_nloglikelihood_value[device_id]);
			cudaDeviceSynchronize();
			nloglikelihood_value_host += temp;
		}

		nloglikelihood = -nloglikelihood_value_host;
	}
	else{
		nloglikelihood = 0;
	}


	//Put the image_host (update factor) values to UpdateFactor array

	printf("\n\n\n\n.......................End projection.........................\n");


	free(image_host);


}





void
cuda_em_recon::ComputeUpdateFactorSS(PET_geometry& detector, PET_movement& movement, PET_data& data, ImageArray<float>& Image, ImageArray<float>& UpdateFactor){


	cudaStream_t streamA[MAX_GPU], streamB[MAX_GPU], streamC[MAX_GPU];
	cudaEvent_t eventA[MAX_GPU], eventB[MAX_GPU], eventC[MAX_GPU];

	cudaEvent_t eventStart1[MAX_GPU], eventStart2[MAX_GPU], eventStop1[MAX_GPU], eventStop2[MAX_GPU];

	int device_id, device_data_length;
	int data_part_index;
	float* image_host;
	//float* fp_x_host;
	//float* fp_y_host;
	string filename;
	stringstream sstm;
	FILE *output_file;

	float milliseconds1 = 0.0f;
	float milliseconds2 = 0.0f;

	dim3 dimBlock_f_x(BlockSize_forward_x);
	dim3 dimGrid_f_x(GridSize_forward_x);

	dim3 dimBlock_f_y(BlockSize_forward_y);
	dim3 dimGrid_f_y(GridSize_forward_y);

	dim3 dimBlock_b_x(BlockSize_backward_x);
	dim3 dimGrid_b_x(GridSize_backward_x);

	dim3 dimBlock_b_y(BlockSize_backward_y);
	dim3 dimGrid_b_y(GridSize_backward_y);

	dim3 dimBlock(1024);
	dim3 dimGrid(64);

	//copy image to all GPUs
	//cudaHostRegister(Image._image, parameters_host.NUM_XYZ*sizeof(float), cudaHostAllocPortable);
	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		checkCudaErrors(cudaSetDevice(device_id));
		checkCudaErrors(cudaMemcpy(_current_image_device[device_id], Image._image, parameters_host.NUM_XYZ*sizeof(float), cudaMemcpyDefault));
		//setting the update factor to zero before projection
		setImageToZero_kernel << <dimGrid, dimBlock >> >(_update_factor_device[device_id], _parameters_device[device_id]);
	}
	//cudaHostUnregister(Image._image);

	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		//checkCudaErrors(cudaSetDevice(device_id));
	}

	printf("\n\n\n\n.......................Start SS data projection.........................\n");
	//SS events

	//copy data to all GPUs
	device_data_length = data.GetDataCount(ALL) / (_num_gpu_end - _num_gpu_start + 1);

	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		data_part_index = device_id - _num_gpu_start;
		if (device_id < _num_gpu_end){
			std::cout << "Copying data from " << data_part_index*device_data_length << " to " << (data_part_index + 1)*device_data_length - 1 << " to device " << device_id << std::endl;
			Setup_data(detector, movement, data, data_part_index*device_data_length, (data_part_index + 1)*device_data_length - 1, SS, device_id);
		}
		else{
			std::cout << "Copying data from " << data_part_index*device_data_length << " to " << data.GetDataCount(ALL) << " to device " << device_id << std::endl;
			Setup_data(detector, movement, data, data_part_index*device_data_length, data.GetDataCount(ALL) - 1, SS, device_id);
		}
	}
	printf("ALL SS data copied to device memory\n");

	parameters_host.FWHM_alpha = parameters_host.FWHM_alpha_ss;
	parameters_host.FWHM_sigma_inv = parameters_host.FWHM_sigma_inv_ss;
	parameters_host.RANGE1 = (int)(parameters_host.FWHM_ss / parameters_host.X_SAMP);
	parameters_host.RANGE2 = (int)(parameters_host.FWHM_ss / parameters_host.Z_SAMP);
	printf("H_CENTER_X = %f H_CENTER_Y = %f V_CENTER = %f\n", parameters_host.H_CENTER_X, parameters_host.H_CENTER_Y, parameters_host.V_CENTER);
	printf("RANGE1 = %d RANGE2 = %d FWHM_SS = %f\n", parameters_host.RANGE1, parameters_host.RANGE2, parameters_host.FWHM_ss);

	cudaHostRegister(&parameters_host, sizeof(PARAMETERS_IN_DEVICE_t), 0);

	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){


		size_t free, total;
		printf("\n");
		cudaMemGetInfo(&free, &total);
		cout << "Before projection for device " << device_id << "Free " << free / 1024 / 1024 << "MB" << "of Total " << total / 1024 / 1024 << "MB" << endl;

		checkCudaErrors(cudaSetDevice(device_id));
		checkCudaErrors(cudaDeviceSynchronize());


		checkCudaErrors(cudaStreamCreate(&streamA[device_id]));
		checkCudaErrors(cudaStreamCreate(&streamB[device_id]));
		checkCudaErrors(cudaStreamCreate(&streamC[device_id]));
		checkCudaErrors(cudaEventCreateWithFlags(&eventA[device_id], cudaEventDisableTiming));
		//checkCudaErrors(cudaEventCreate(&eventB[device_id]));
		//checkCudaErrors(cudaEventCreate(&eventC[device_id]));

		checkCudaErrors(cudaEventCreate(&eventStart1[device_id]));
		checkCudaErrors(cudaEventCreate(&eventStart2[device_id]));
		checkCudaErrors(cudaEventCreate(&eventStop1[device_id]));
		checkCudaErrors(cudaEventCreate(&eventStop2[device_id]));
	}

	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		cudaSetDevice(device_id);
		printf("\n\n\n\n.......................Start projection on Device %d.........................\n", device_id);

		checkCudaErrors(cudaMemcpyAsync(_parameters_device[device_id], &parameters_host, sizeof(PARAMETERS_IN_DEVICE_t), cudaMemcpyDefault, streamA[device_id]));
		checkCudaErrors(cudaEventRecord(eventA[device_id], streamA[device_id]));

		setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamB[device_id] >> >(_fp_value_x_device[device_id], _event_count_x[device_id]);

		setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamC[device_id] >> >(_fp_value_y_device[device_id], _event_count_y[device_id]);

		if (_TOF_mode == TOF){
			//TOF version
			printf("starting TOF kernels.\n");

			checkCudaErrors(cudaStreamWaitEvent(streamB[device_id], eventA[device_id], 0));
			checkCudaErrors(cudaEventRecord(eventStart1[device_id], streamB[device_id]));
			_TOF_fproj_lst_cuda_x_kernel << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_current_image_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);

			checkCudaErrors(cudaStreamWaitEvent(streamC[device_id], eventA[device_id], 0));
			checkCudaErrors(cudaEventRecord(eventStart2[device_id], streamC[device_id]));
			_TOF_fproj_lst_cuda_y_kernel << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_current_image_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);

			_TOF_b_ratio_proj_lst_cuda_x_kernel << <dimGrid_b_x, dimBlock_b_x, 0, streamB[device_id] >> >(_update_factor_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
			//cudaEventRecord(eventB[device_id], streamB[device_id]);
			checkCudaErrors(cudaEventRecord(eventStop1[device_id], streamB[device_id]));

			_TOF_b_ratio_proj_lst_cuda_y_kernel << <dimGrid_b_y, dimBlock_b_y, 0, streamC[device_id] >> >(_update_factor_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
			//cudaEventRecord(eventC[device_id], streamC[device_id]);
			checkCudaErrors(cudaEventRecord(eventStop2[device_id], streamC[device_id]));
		}
		else{
			//NON-TOF version
			printf("starting NON-TOF kernels.\n");

			checkCudaErrors(cudaStreamWaitEvent(streamB[device_id], eventA[device_id], 0));
			checkCudaErrors(cudaEventRecord(eventStart1[device_id], streamB[device_id]));
			_fproj_lst_cuda_x_kernel << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_current_image_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);

			checkCudaErrors(cudaStreamWaitEvent(streamC[device_id], eventA[device_id], 0));
			checkCudaErrors(cudaEventRecord(eventStart2[device_id], streamC[device_id]));
			_fproj_lst_cuda_y_kernel << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_current_image_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);

			_b_ratio_proj_lst_cuda_x_kernel << <dimGrid_b_x, dimBlock_b_x, 0, streamB[device_id] >> >(_update_factor_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
			//cudaEventRecord(eventB[device_id], streamB[device_id]);
			checkCudaErrors(cudaEventRecord(eventStop1[device_id], streamB[device_id]));

			_b_ratio_proj_lst_cuda_y_kernel << <dimGrid_b_y, dimBlock_b_y, 0, streamC[device_id] >> >(_update_factor_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
			//cudaEventRecord(eventC[device_id], streamC[device_id]);
			checkCudaErrors(cudaEventRecord(eventStop2[device_id], streamC[device_id]));
		}

	}

	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		checkCudaErrors(cudaSetDevice(device_id));

		/*
		checkCudaErrors(cudaStreamSynchronize(streamA[device_id]));
		checkCudaErrors(cudaStreamSynchronize(streamB[device_id]));
		checkCudaErrors(cudaStreamSynchronize(streamC[device_id]));
		*/

		checkCudaErrors(cudaEventSynchronize(eventStart1[device_id]));
		checkCudaErrors(cudaEventSynchronize(eventStart2[device_id]));
		checkCudaErrors(cudaEventSynchronize(eventStop1[device_id]));
		checkCudaErrors(cudaEventSynchronize(eventStop2[device_id]));


		checkCudaErrors(cudaEventElapsedTime(&milliseconds1, eventStart1[device_id], eventStop1[device_id]));
		checkCudaErrors(cudaEventElapsedTime(&milliseconds2, eventStart2[device_id], eventStop2[device_id]));

		printf("\n.......................Wrapping up for device %d........................\n", device_id);
		printf("Time for X-dominant projections: %f seconds\n", milliseconds1 / 1000);
		printf("Time for Y-dominant projections: %f seconds\n", milliseconds2 / 1000);

		checkCudaErrors(cudaEventDestroy(eventA[device_id]));
		/*
		checkCudaErrors(cudaEventDestroy(eventB[device_id]));
		checkCudaErrors(cudaEventDestroy(eventC[device_id]));
		*/

		checkCudaErrors(cudaEventDestroy(eventStart1[device_id]));
		checkCudaErrors(cudaEventDestroy(eventStart2[device_id]));
		checkCudaErrors(cudaEventDestroy(eventStop1[device_id]));
		checkCudaErrors(cudaEventDestroy(eventStop2[device_id]));

		checkCudaErrors(cudaStreamDestroy(streamA[device_id]));
		checkCudaErrors(cudaStreamDestroy(streamB[device_id]));
		checkCudaErrors(cudaStreamDestroy(streamC[device_id]));

		//cudaDeviceSynchronize();
	}

	cudaHostUnregister(&parameters_host);

	printf("\n... start clean up data\n\n");
	Release_data();
	printf("\n... done clean up data\n\n");


	UpdateFactor.SetValue(0.0f);
	image_host = (float*)malloc(parameters_host.NUM_XYZ*sizeof(float));
	//copy the update factor back to host memory
	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		cudaSetDevice(device_id);
		cudaDeviceSynchronize();
		cudaMemcpy(image_host, _update_factor_device[device_id], parameters_host.NUM_XYZ*sizeof(float), cudaMemcpyDefault);
		UpdateFactor.AddFromMem(image_host);
	}

	//Put the image_host (update factor) values to UpdateFactor array

	printf("\n\n\n\n.......................End projection.........................\n");


	free(image_host);


}





void
cuda_em_recon::ComputeUpdateFactorIS(PET_geometry& detector, PET_movement& movement, PET_data& data, ImageArray<float>& Image, ImageArray<float>& UpdateFactor){

	printf("8716");

	cudaStream_t streamA[MAX_GPU], streamB[MAX_GPU], streamC[MAX_GPU];
	cudaEvent_t eventA[MAX_GPU], eventB[MAX_GPU], eventC[MAX_GPU];

	cudaEvent_t eventStart1[MAX_GPU], eventStart2[MAX_GPU], eventStop1[MAX_GPU], eventStop2[MAX_GPU];

	int device_id, device_data_length;
	int data_part_index;
	float* image_host;
	//float* fp_x_host;
	//float* fp_y_host;
	string filename;
	stringstream sstm;
	FILE *output_file;

	float milliseconds1 = 0.0f;
	float milliseconds2 = 0.0f;

	dim3 dimBlock_f_x(BlockSize_forward_x);
	dim3 dimGrid_f_x(GridSize_forward_x);

	dim3 dimBlock_f_y(BlockSize_forward_y);
	dim3 dimGrid_f_y(GridSize_forward_y);

	dim3 dimBlock_b_x(BlockSize_backward_x);
	dim3 dimGrid_b_x(GridSize_backward_x);

	dim3 dimBlock_b_y(BlockSize_backward_y);
	dim3 dimGrid_b_y(GridSize_backward_y);

	dim3 dimBlock(1024);
	dim3 dimGrid(64);

	//copy image to all GPUs
	//cudaHostRegister(Image._image, parameters_host.NUM_XYZ*sizeof(float), cudaHostAllocPortable);
	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		checkCudaErrors(cudaSetDevice(device_id));
		checkCudaErrors(cudaMemcpy(_current_image_device[device_id], Image._image, parameters_host.NUM_XYZ*sizeof(float), cudaMemcpyDefault));
		//setting the update factor to zero before projection
		setImageToZero_kernel << <dimGrid, dimBlock >> >(_update_factor_device[device_id], _parameters_device[device_id]);
	}
	//cudaHostUnregister(Image._image);

	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		//checkCudaErrors(cudaSetDevice(device_id));
	}


	//copy data to all GPUs
	device_data_length = data.GetDataCount(ALL) / (_num_gpu_end - _num_gpu_start + 1);


	printf("\n\n\n\n.......................Start IS data projection.........................\n");
	//IS events


	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		data_part_index = device_id - _num_gpu_start;
		if (device_id < _num_gpu_end){
			Setup_data(detector, movement, data, data_part_index*device_data_length, (data_part_index + 1)*device_data_length - 1, IS, device_id);
		}
		else{
			Setup_data(detector, movement, data, data_part_index*device_data_length, data.GetDataCount(ALL) - 1, IS, device_id);
		}
	}
	printf("IS data copied to device memory\n");

	parameters_host.FWHM_alpha = parameters_host.FWHM_alpha_is;
	parameters_host.FWHM_sigma_inv = parameters_host.FWHM_sigma_inv_is;
	parameters_host.RANGE1 = (int)(parameters_host.FWHM_is / parameters_host.X_SAMP);
	parameters_host.RANGE2 = (int)(parameters_host.FWHM_is / parameters_host.Z_SAMP);
	if (parameters_host.RANGE1 < 1)
		parameters_host.RANGE1 = 1;
	if (parameters_host.RANGE2 < 1)
		parameters_host.RANGE2 = 1;
	printf("H_CENTER_X = %f H_CENTER_Y = %f V_CENTER = %f\n", parameters_host.H_CENTER_X, parameters_host.H_CENTER_Y, parameters_host.V_CENTER);
	printf("RANGE1 = %d RANGE2 = %d FWHM_IS = %f\n", parameters_host.RANGE1, parameters_host.RANGE2, parameters_host.FWHM_is);

	cudaHostRegister(&parameters_host, sizeof(PARAMETERS_IN_DEVICE_t), 0);

	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		checkCudaErrors(cudaSetDevice(device_id));
		checkCudaErrors(cudaDeviceSynchronize());


		checkCudaErrors(cudaStreamCreate(&streamA[device_id]));
		checkCudaErrors(cudaStreamCreate(&streamB[device_id]));
		checkCudaErrors(cudaStreamCreate(&streamC[device_id]));
		checkCudaErrors(cudaEventCreateWithFlags(&eventA[device_id], cudaEventDisableTiming));
		//checkCudaErrors(cudaEventCreate(&eventB[device_id]));
		//checkCudaErrors(cudaEventCreate(&eventC[device_id]));

		checkCudaErrors(cudaEventCreate(&eventStart1[device_id]));
		checkCudaErrors(cudaEventCreate(&eventStart2[device_id]));
		checkCudaErrors(cudaEventCreate(&eventStop1[device_id]));
		checkCudaErrors(cudaEventCreate(&eventStop2[device_id]));
	}

	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		cudaSetDevice(device_id);
		printf("\n\n\n\n.......................Start projection on Device %d.........................\n", device_id);

		checkCudaErrors(cudaMemcpyAsync(_parameters_device[device_id], &parameters_host, sizeof(PARAMETERS_IN_DEVICE_t), cudaMemcpyDefault, streamA[device_id]));
		checkCudaErrors(cudaEventRecord(eventA[device_id], streamA[device_id]));

		setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamB[device_id] >> >(_fp_value_x_device[device_id], _event_count_x[device_id]);

		setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamC[device_id] >> >(_fp_value_y_device[device_id], _event_count_y[device_id]);

		if (_TOF_mode == TOF){
			//TOF version
			printf("starting TOF kernels.\n");

			checkCudaErrors(cudaStreamWaitEvent(streamB[device_id], eventA[device_id], 0));
			checkCudaErrors(cudaEventRecord(eventStart1[device_id], streamB[device_id]));
			_TOF_fproj_lst_cuda_x_kernel << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_current_image_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);

			checkCudaErrors(cudaStreamWaitEvent(streamC[device_id], eventA[device_id], 0));
			checkCudaErrors(cudaEventRecord(eventStart2[device_id], streamC[device_id]));
			_TOF_fproj_lst_cuda_y_kernel << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_current_image_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);

			_TOF_b_ratio_proj_lst_cuda_x_kernel << <dimGrid_b_x, dimBlock_b_x, 0, streamB[device_id] >> >(_update_factor_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
			//cudaEventRecord(eventB[device_id], streamB[device_id]);
			checkCudaErrors(cudaEventRecord(eventStop1[device_id], streamB[device_id]));

			_TOF_b_ratio_proj_lst_cuda_y_kernel << <dimGrid_b_y, dimBlock_b_y, 0, streamC[device_id] >> >(_update_factor_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
			//cudaEventRecord(eventC[device_id], streamC[device_id]);
			checkCudaErrors(cudaEventRecord(eventStop2[device_id], streamC[device_id]));
		}
		else{
			//NON-TOF version
			printf("starting NON-TOF kernels.\n");

			checkCudaErrors(cudaStreamWaitEvent(streamB[device_id], eventA[device_id], 0));
			checkCudaErrors(cudaEventRecord(eventStart1[device_id], streamB[device_id]));
			_fproj_lst_cuda_x_kernel << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_current_image_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);

			checkCudaErrors(cudaStreamWaitEvent(streamC[device_id], eventA[device_id], 0));
			checkCudaErrors(cudaEventRecord(eventStart2[device_id], streamC[device_id]));
			_fproj_lst_cuda_y_kernel << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_current_image_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);

			_b_ratio_proj_lst_cuda_x_kernel << <dimGrid_b_x, dimBlock_b_x, 0, streamB[device_id] >> >(_update_factor_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
			//cudaEventRecord(eventB[device_id], streamB[device_id]);
			checkCudaErrors(cudaEventRecord(eventStop1[device_id], streamB[device_id]));

			_b_ratio_proj_lst_cuda_y_kernel << <dimGrid_b_y, dimBlock_b_y, 0, streamC[device_id] >> >(_update_factor_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
			//cudaEventRecord(eventC[device_id], streamC[device_id]);
			checkCudaErrors(cudaEventRecord(eventStop2[device_id], streamC[device_id]));
		}

	}

	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		checkCudaErrors(cudaSetDevice(device_id));

		/*
		checkCudaErrors(cudaStreamSynchronize(streamA[device_id]));
		checkCudaErrors(cudaStreamSynchronize(streamB[device_id]));
		checkCudaErrors(cudaStreamSynchronize(streamC[device_id]));
		*/

		checkCudaErrors(cudaEventSynchronize(eventStart1[device_id]));
		checkCudaErrors(cudaEventSynchronize(eventStart2[device_id]));
		checkCudaErrors(cudaEventSynchronize(eventStop1[device_id]));
		checkCudaErrors(cudaEventSynchronize(eventStop2[device_id]));


		checkCudaErrors(cudaEventElapsedTime(&milliseconds1, eventStart1[device_id], eventStop1[device_id]));
		checkCudaErrors(cudaEventElapsedTime(&milliseconds2, eventStart2[device_id], eventStop2[device_id]));

		printf("\n.......................Wrapping up for device %d........................\n", device_id);
		printf("Time for X-dominant projections: %f seconds\n", milliseconds1 / 1000);
		printf("Time for Y-dominant projections: %f seconds\n", milliseconds2 / 1000);

		checkCudaErrors(cudaEventDestroy(eventA[device_id]));
		/*
		checkCudaErrors(cudaEventDestroy(eventB[device_id]));
		checkCudaErrors(cudaEventDestroy(eventC[device_id]));
		*/

		checkCudaErrors(cudaEventDestroy(eventStart1[device_id]));
		checkCudaErrors(cudaEventDestroy(eventStart2[device_id]));
		checkCudaErrors(cudaEventDestroy(eventStop1[device_id]));
		checkCudaErrors(cudaEventDestroy(eventStop2[device_id]));

		checkCudaErrors(cudaStreamDestroy(streamA[device_id]));
		checkCudaErrors(cudaStreamDestroy(streamB[device_id]));
		checkCudaErrors(cudaStreamDestroy(streamC[device_id]));

		//cudaDeviceSynchronize();
	}
	cudaHostUnregister(&parameters_host);

	printf("\n... start clean up data\n\n");
	Release_data();
	printf("\n... done clean up data\n\n");




	UpdateFactor.SetValue(0.0f);
	image_host = (float*)malloc(parameters_host.NUM_XYZ*sizeof(float));
	//copy the update factor back to host memory
	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		cudaSetDevice(device_id);
		cudaDeviceSynchronize();
		cudaMemcpy(image_host, _update_factor_device[device_id], parameters_host.NUM_XYZ*sizeof(float), cudaMemcpyDefault);
		UpdateFactor.AddFromMem(image_host);
	}

	//Put the image_host (update factor) values to UpdateFactor array

	printf("\n\n\n\n.......................End projection.........................\n");


	free(image_host);


}


















void
cuda_em_recon::Backward_Projection_Attenuation_norm(PET_geometry& detector, PET_movement& movement, PET_data& data, ImageArray<float>& Atten_Image, ImageArray<float>& Norm_Image,  ImageArray<float>& Emission_Image){


	cudaStream_t streamA[MAX_GPU], streamB[MAX_GPU], streamC[MAX_GPU], streamD[MAX_GPU];
	cudaEvent_t eventA[MAX_GPU], eventB[MAX_GPU], eventC[MAX_GPU], eventD[MAX_GPU];

	cudaEvent_t eventStart1[MAX_GPU], eventStart2[MAX_GPU], eventStart3[MAX_GPU], eventStop1[MAX_GPU], eventStop2[MAX_GPU], eventStop3[MAX_GPU];

	int device_id, device_data_length;
	int data_part_index;
	float* image_host;
	//float* fp_x_host;
	//float* fp_y_host;
	string filename;
	stringstream sstm;
	FILE *output_file;

	float milliseconds1 = 0.0f;
	float milliseconds2 = 0.0f;
	float milliseconds3 = 0.0f;

	dim3 dimBlock_f_x(BlockSize_forward_x);
	dim3 dimGrid_f_x(GridSize_forward_x);

	dim3 dimBlock_f_y(BlockSize_forward_y);
	dim3 dimGrid_f_y(GridSize_forward_y);

	dim3 dimBlock_b_x(BlockSize_backward_x);
	dim3 dimGrid_b_x(GridSize_backward_x);

	dim3 dimBlock_b_y(BlockSize_backward_y);
	dim3 dimGrid_b_y(GridSize_backward_y);

	

	dim3 dimBlock(1024);
	dim3 dimGrid(64);

	//copy image to all GPUs
	//cudaHostRegister(Image._image, parameters_host.NUM_XYZ*sizeof(float), cudaHostAllocPortable);
	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		checkCudaErrors(cudaSetDevice(device_id));
		checkCudaErrors(cudaMemcpy(_current_image_device[device_id], Atten_Image._image, parameters_host.NUM_XYZ*sizeof(float), cudaMemcpyDefault));
		checkCudaErrors(cudaMemcpy(_current2_image_device[device_id], Emission_Image._image, parameters_host.NUM_XYZ*sizeof(float), cudaMemcpyDefault));
		//setting the update factor to zero before projection
		setImageToZero_kernel << <dimGrid, dimBlock >> >(_update_factor_device[device_id], _parameters_device[device_id]);
	}
	//cudaHostUnregister(Image._image);

	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		//checkCudaErrors(cudaSetDevice(device_id));
	}

	//printf("\n\n\n\n.......................Start SS data projection.........................\n");
	//SS events

	//copy data to all GPUs
	device_data_length = data.GetDataCount(ALL) / (_num_gpu_end - _num_gpu_start + 1);

	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		data_part_index = device_id - _num_gpu_start;
		if (device_id < _num_gpu_end){
			Setup_data_norm(detector, movement, data, data_part_index*device_data_length, (data_part_index + 1)*device_data_length - 1, SS, device_id);
		}
		else{
			Setup_data_norm(detector, movement, data, data_part_index*device_data_length, data.GetDataCount(ALL) - 1, SS, device_id);
		}
	}
	//printf("ALL SS data copied to device memory\n");

	parameters_host.FWHM_alpha = parameters_host.FWHM_alpha_ss;
	parameters_host.FWHM_sigma_inv = parameters_host.FWHM_sigma_inv_ss;

	parameters_host.RANGE1 = (int)(parameters_host.FWHM_ss / parameters_host.X_SAMP);
	parameters_host.RANGE2 = (int)(parameters_host.FWHM_ss / parameters_host.Z_SAMP);
	parameters_host.RANGE_atten_1 = (int)(parameters_host.FWHM_ss / parameters_host.X_SAMP);
	parameters_host.RANGE_atten_2 = (int)(parameters_host.FWHM_ss / parameters_host.Z_SAMP);
	//printf("H_CENTER_X = %f H_CENTER_Y = %f V_CENTER = %f\n", parameters_host.H_CENTER_X, parameters_host.H_CENTER_Y, parameters_host.V_CENTER);
	//printf("RANGE_atten_1 = %d RANGE_atten_2 = %d FWHM_SS = %f\n", parameters_host.RANGE_atten_1, parameters_host.RANGE_atten_2, parameters_host.FWHM_ss);


	cudaHostRegister(&parameters_host, sizeof(PARAMETERS_IN_DEVICE_t), 0);

	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		checkCudaErrors(cudaSetDevice(device_id));
		checkCudaErrors(cudaDeviceSynchronize());


		checkCudaErrors(cudaStreamCreate(&streamA[device_id]));
		checkCudaErrors(cudaStreamCreate(&streamB[device_id]));
		checkCudaErrors(cudaStreamCreate(&streamC[device_id]));
	
		checkCudaErrors(cudaEventCreateWithFlags(&eventA[device_id], cudaEventDisableTiming));
		//checkCudaErrors(cudaEventCreate(&eventB[device_id]));
		//checkCudaErrors(cudaEventCreate(&eventC[device_id]));

		checkCudaErrors(cudaEventCreate(&eventStart1[device_id]));
		checkCudaErrors(cudaEventCreate(&eventStart2[device_id]));
		
		checkCudaErrors(cudaEventCreate(&eventStop1[device_id]));
		checkCudaErrors(cudaEventCreate(&eventStop2[device_id]));
	
	}

	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		cudaSetDevice(device_id);
		printf("\n\n\n\n.......................Start projection on Device %d.........................\n", device_id);

		checkCudaErrors(cudaMemcpyAsync(_parameters_device[device_id], &parameters_host, sizeof(PARAMETERS_IN_DEVICE_t), cudaMemcpyDefault, streamA[device_id]));
		checkCudaErrors(cudaEventRecord(eventA[device_id], streamA[device_id]));

		setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamB[device_id] >> >(_fp_value_x_device[device_id], _event_count_x[device_id]);

		setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamC[device_id] >> >(_fp_value_y_device[device_id], _event_count_y[device_id]);


		setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamB[device_id] >> >(_fp_value2_x_device[device_id], _event_count_x[device_id]);

		setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamC[device_id] >> >(_fp_value2_y_device[device_id], _event_count_y[device_id]);

		

		if (_TOF_mode == TOF){
			//TOF version
			printf("starting TOF kernels.\n");

			checkCudaErrors(cudaStreamWaitEvent(streamB[device_id], eventA[device_id], 0));
			checkCudaErrors(cudaEventRecord(eventStart1[device_id], streamB[device_id]));
			_fproj_atten_lst_cuda_x_kernel << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_current_image_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);

			_fproj_lst_cuda_x_kernel << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_current2_image_device[device_id], _fp_value2_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
			//__global__ _fproj_lst_cuda_x_kernel(float* image, float* fp_x, LST_LORs *_events_x_dominant_uvm, PARAMETERS_IN_DEVICE_t* parameters_device){


			checkCudaErrors(cudaStreamWaitEvent(streamC[device_id], eventA[device_id], 0));
			checkCudaErrors(cudaEventRecord(eventStart2[device_id], streamC[device_id]));
			_fproj_atten_lst_cuda_y_kernel << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_current_image_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);

			_fproj_lst_cuda_y_kernel << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_current2_image_device[device_id], _fp_value2_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);


			_bproj_atten_lst_cuda_x_kernel_norm << <dimGrid_b_x, dimBlock_b_x, 0, streamB[device_id] >> >(_update_factor_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id], _fp_value2_x_device[device_id]);
			//cudaEventRecord(eventB[device_id], streamB[device_id]);
			checkCudaErrors(cudaEventRecord(eventStop1[device_id], streamB[device_id]));

			_bproj_atten_lst_cuda_y_kernel_norm << <dimGrid_b_y, dimBlock_b_y, 0, streamC[device_id] >> >(_update_factor_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id], _fp_value2_y_device[device_id]);
			//cudaEventRecord(eventC[device_id], streamC[device_id]);
			checkCudaErrors(cudaEventRecord(eventStop2[device_id], streamC[device_id]));

			
		}
		else{
			//NON-TOF version
			printf("starting NON-TOF kernels.\n");

			checkCudaErrors(cudaStreamWaitEvent(streamB[device_id], eventA[device_id], 0));
			checkCudaErrors(cudaEventRecord(eventStart1[device_id], streamB[device_id]));
		
			_fproj_atten_lst_cuda_x_kernel << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_current2_image_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
			
			
			
			checkCudaErrors(cudaStreamWaitEvent(streamC[device_id], eventA[device_id], 0));
			checkCudaErrors(cudaEventRecord(eventStart2[device_id], streamC[device_id]));
		
			_fproj_atten_lst_cuda_y_kernel << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_current2_image_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
		
	

			
			_bproj_atten_lst_cuda_x_kernel << <dimGrid_b_x, dimBlock_b_x, 0, streamB[device_id] >> >(_update_factor_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
			//cudaEventRecord(eventB[device_id], streamB[device_id]);
			checkCudaErrors(cudaEventRecord(eventStop1[device_id], streamB[device_id]));
			_bproj_atten_lst_cuda_y_kernel << <dimGrid_b_y, dimBlock_b_y, 0, streamC[device_id] >> >(_update_factor_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
			//cudaEventRecord(eventC[device_id], streamC[device_id]);
			checkCudaErrors(cudaEventRecord(eventStop2[device_id], streamC[device_id]));
			
		}

	}

	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		checkCudaErrors(cudaSetDevice(device_id));

		/*
		checkCudaErrors(cudaStreamSynchronize(streamA[device_id]));
		checkCudaErrors(cudaStreamSynchronize(streamB[device_id]));
		checkCudaErrors(cudaStreamSynchronize(streamC[device_id]));
		*/

		checkCudaErrors(cudaEventSynchronize(eventStart1[device_id]));
		checkCudaErrors(cudaEventSynchronize(eventStart2[device_id]));
		
		checkCudaErrors(cudaEventSynchronize(eventStop1[device_id]));
		checkCudaErrors(cudaEventSynchronize(eventStop2[device_id]));
		

		checkCudaErrors(cudaEventElapsedTime(&milliseconds1, eventStart1[device_id], eventStop1[device_id]));
		checkCudaErrors(cudaEventElapsedTime(&milliseconds2, eventStart2[device_id], eventStop2[device_id]));
		
		printf("\n.......................Wrapping up for device %d........................\n", device_id);
		printf("Time for X-dominant projections: %f seconds\n", milliseconds1 / 1000);
		printf("Time for Y-dominant projections: %f seconds\n", milliseconds2 / 1000);
		

		checkCudaErrors(cudaEventDestroy(eventA[device_id]));
		/*
		checkCudaErrors(cudaEventDestroy(eventB[device_id]));
		checkCudaErrors(cudaEventDestroy(eventC[device_id]));
		*/

		checkCudaErrors(cudaEventDestroy(eventStart1[device_id]));
		checkCudaErrors(cudaEventDestroy(eventStart2[device_id]));
		
		checkCudaErrors(cudaEventDestroy(eventStop1[device_id]));
		checkCudaErrors(cudaEventDestroy(eventStop2[device_id]));
		

		checkCudaErrors(cudaStreamDestroy(streamA[device_id]));
		checkCudaErrors(cudaStreamDestroy(streamB[device_id]));
		checkCudaErrors(cudaStreamDestroy(streamC[device_id]));
		
		//cudaDeviceSynchronize();
	}

	/*
	// Check forward projected values
	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		int size1 = _event_count_y[device_id];
		int size2 = _event_count_x[device_id];

		float* fpvalue2_hosty = new float[size1];
		float* fpvalue2_hostx = new float[size2];

		float* fpvalue1_hosty = new float[size1];
		float* fpvalue1_hostx = new float[size2];
		cudaSetDevice(device_id);
		cudaDeviceSynchronize();
		cudaMemcpy(fpvalue2_hosty, _fp_value2_y_device[device_id], size1*sizeof(float), cudaMemcpyDefault);
		cudaMemcpy(fpvalue2_hostx, _fp_value2_x_device[device_id], size2*sizeof(float), cudaMemcpyDefault);
		cudaMemcpy(fpvalue1_hosty, _fp_value_y_device[device_id], size1*sizeof(float), cudaMemcpyDefault);
		cudaMemcpy(fpvalue1_hostx, _fp_value_x_device[device_id], size2*sizeof(float), cudaMemcpyDefault);
		for (int i = 0; i < size1; i++){
			cout << "\n fpvalues2 y " << fpvalue2_hosty[i] <<  " fpvalues1 y " << fpvalue1_hosty[i];
		}
		for (int j = 0; j < size2; j++){
			cout << "\n fpvalues2 x : " << fpvalue2_hostx[j] << " fpvalues1 x " << fpvalue1_hostx[j];
		}
	}
	*/
	


	cudaHostUnregister(&parameters_host);

	//printf("\n... start clean up data\n\n");
	Release_data();
	printf("\n... done clean up data\n\n");



	/*printf("\n\n\n\n.......................Start II data projection.........................\n");
	//IS events


	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		data_part_index = device_id - _num_gpu_start;
		if (device_id < _num_gpu_end){
			Setup_data(detector, movement, data, data_part_index*device_data_length, (data_part_index + 1)*device_data_length - 1, II, device_id);
		}
		else{
			Setup_data(detector, movement, data, data_part_index*device_data_length, data.GetDataCount(ALL) - 1, II, device_id);
		}
	}
	printf("II data copied to device memory\n");


	parameters_host.FWHM_alpha = parameters_host.FWHM_alpha_ii;
	parameters_host.FWHM_sigma_inv = parameters_host.FWHM_sigma_inv_ii;
	parameters_host.RANGE_atten_1 = (int)(parameters_host.FWHM_ii / parameters_host.X_SAMP);
	parameters_host.RANGE_atten_2 = (int)(parameters_host.FWHM_ii / parameters_host.Z_SAMP);
	if (parameters_host.RANGE_atten_1 < 1)
		parameters_host.RANGE_atten_1 = 1;
	if (parameters_host.RANGE_atten_2 < 1)
		parameters_host.RANGE_atten_2 = 1;
	printf("H_CENTER_X = %f H_CENTER_Y = %f V_CENTER = %f\n", parameters_host.H_CENTER_X, parameters_host.H_CENTER_Y, parameters_host.V_CENTER);
	printf("RANGE_atten_1 = %d RANGE_atten_2 = %d FWHM_II = %f\n", parameters_host.RANGE_atten_1, parameters_host.RANGE_atten_2, parameters_host.FWHM_ii);

	cudaHostRegister(&parameters_host, sizeof(PARAMETERS_IN_DEVICE_t), 0);


	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		checkCudaErrors(cudaSetDevice(device_id));
		checkCudaErrors(cudaDeviceSynchronize());


		checkCudaErrors(cudaStreamCreate(&streamA[device_id]));
		checkCudaErrors(cudaStreamCreate(&streamB[device_id]));
		checkCudaErrors(cudaStreamCreate(&streamC[device_id]));
		checkCudaErrors(cudaEventCreateWithFlags(&eventA[device_id], cudaEventDisableTiming));
		//checkCudaErrors(cudaEventCreate(&eventB[device_id]));
		//checkCudaErrors(cudaEventCreate(&eventC[device_id]));

		checkCudaErrors(cudaEventCreate(&eventStart1[device_id]));
		checkCudaErrors(cudaEventCreate(&eventStart2[device_id]));
		checkCudaErrors(cudaEventCreate(&eventStop1[device_id]));
		checkCudaErrors(cudaEventCreate(&eventStop2[device_id]));
	}

	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		cudaSetDevice(device_id);
		printf("\n\n\n\n.......................Start projection on Device %d.........................\n", device_id);

		checkCudaErrors(cudaMemcpyAsync(_parameters_device[device_id], &parameters_host, sizeof(PARAMETERS_IN_DEVICE_t), cudaMemcpyDefault, streamA[device_id]));
		checkCudaErrors(cudaEventRecord(eventA[device_id], streamA[device_id]));

		setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamB[device_id] >> >(_fp_value_x_device[device_id], _event_count_x[device_id]);

		setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamC[device_id] >> >(_fp_value_y_device[device_id], _event_count_y[device_id]);

		if (_TOF_mode == TOF){
			//TOF version
			printf("starting TOF kernels.\n");

			checkCudaErrors(cudaStreamWaitEvent(streamB[device_id], eventA[device_id], 0));
			checkCudaErrors(cudaEventRecord(eventStart1[device_id], streamB[device_id]));
			_fproj_atten_lst_cuda_x_kernel << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_current_image_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);

			checkCudaErrors(cudaStreamWaitEvent(streamC[device_id], eventA[device_id], 0));
			checkCudaErrors(cudaEventRecord(eventStart2[device_id], streamC[device_id]));
			_fproj_atten_lst_cuda_y_kernel << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_current_image_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);

			_bproj_atten_lst_cuda_x_kernel << <dimGrid_b_x, dimBlock_b_x, 0, streamB[device_id] >> >(_update_factor_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
			//cudaEventRecord(eventB[device_id], streamB[device_id]);
			checkCudaErrors(cudaEventRecord(eventStop1[device_id], streamB[device_id]));

			_bproj_atten_lst_cuda_y_kernel << <dimGrid_b_y, dimBlock_b_y, 0, streamC[device_id] >> >(_update_factor_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
			//cudaEventRecord(eventC[device_id], streamC[device_id]);
			checkCudaErrors(cudaEventRecord(eventStop2[device_id], streamC[device_id]));
		}
		else{
			//NON-TOF version
			printf("starting NON-TOF kernels.\n");

			checkCudaErrors(cudaStreamWaitEvent(streamB[device_id], eventA[device_id], 0));
			checkCudaErrors(cudaEventRecord(eventStart1[device_id], streamB[device_id]));
			_fproj_atten_lst_cuda_x_kernel << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_current_image_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);

			checkCudaErrors(cudaStreamWaitEvent(streamC[device_id], eventA[device_id], 0));
			checkCudaErrors(cudaEventRecord(eventStart2[device_id], streamC[device_id]));
			_fproj_atten_lst_cuda_y_kernel << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_current_image_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);

			_bproj_atten_lst_cuda_x_kernel << <dimGrid_b_x, dimBlock_b_x, 0, streamB[device_id] >> >(_update_factor_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
			//cudaEventRecord(eventB[device_id], streamB[device_id]);
			checkCudaErrors(cudaEventRecord(eventStop1[device_id], streamB[device_id]));

			_bproj_atten_lst_cuda_y_kernel << <dimGrid_b_y, dimBlock_b_y, 0, streamC[device_id] >> >(_update_factor_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
			//cudaEventRecord(eventC[device_id], streamC[device_id]);
			checkCudaErrors(cudaEventRecord(eventStop2[device_id], streamC[device_id]));
		}

	}

	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		checkCudaErrors(cudaSetDevice(device_id));
		
		
		checkCudaErrors(cudaStreamSynchronize(streamA[device_id]));
		checkCudaErrors(cudaStreamSynchronize(streamB[device_id]));
		checkCudaErrors(cudaStreamSynchronize(streamC[device_id]));
		

		checkCudaErrors(cudaEventSynchronize(eventStart1[device_id]));
		checkCudaErrors(cudaEventSynchronize(eventStart2[device_id]));
		checkCudaErrors(cudaEventSynchronize(eventStop1[device_id]));
		checkCudaErrors(cudaEventSynchronize(eventStop2[device_id]));


		checkCudaErrors(cudaEventElapsedTime(&milliseconds1, eventStart1[device_id], eventStop1[device_id]));
		checkCudaErrors(cudaEventElapsedTime(&milliseconds2, eventStart2[device_id], eventStop2[device_id]));

		printf("\n.......................Wrapping up for device %d........................\n", device_id);
		printf("Time for X-dominant projections: %f seconds\n", milliseconds1 / 1000);
		printf("Time for Y-dominant projections: %f seconds\n", milliseconds2 / 1000);

		checkCudaErrors(cudaEventDestroy(eventA[device_id]));
		
		checkCudaErrors(cudaEventDestroy(eventB[device_id]));
		checkCudaErrors(cudaEventDestroy(eventC[device_id]));
		

		checkCudaErrors(cudaEventDestroy(eventStart1[device_id]));
		checkCudaErrors(cudaEventDestroy(eventStart2[device_id]));
		checkCudaErrors(cudaEventDestroy(eventStop1[device_id]));
		checkCudaErrors(cudaEventDestroy(eventStop2[device_id]));

		checkCudaErrors(cudaStreamDestroy(streamA[device_id]));
		checkCudaErrors(cudaStreamDestroy(streamB[device_id]));
		checkCudaErrors(cudaStreamDestroy(streamC[device_id]));

		//cudaDeviceSynchronize();
	}
	cudaHostUnregister(&parameters_host);

	printf("\n... start clean up data\n\n");
	Release_data();
	printf("\n... done clean up data\n\n");

	*/


	Norm_Image.SetValue(0.0f);
	image_host = (float*)malloc(parameters_host.NUM_XYZ*sizeof(float));
	//copy the update factor back to host memory
	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		cudaSetDevice(device_id);
		cudaDeviceSynchronize();
		cudaMemcpy(image_host, _update_factor_device[device_id], parameters_host.NUM_XYZ*sizeof(float), cudaMemcpyDefault);
		Norm_Image.AddFromMem(image_host);
		/*
		sstm.str("");
		sstm << _output_path << _output_filename_prefix << "_device_"<< device_id<<".img";
		Atten_Image.ReadFromMem(image_host);
		Atten_Image.WriteToFile(sstm.str());
		*/
	}



	

	printf("\n\n\n\n.......................End projection.........................\n");


	free(image_host);


}


void
cuda_em_recon::Backward_Projection_Attenuation(PET_geometry& detector, PET_movement& movement, PET_data& data, ImageArray<float>& Atten_Image, ImageArray<float>& Norm_Image){


	cudaStream_t streamA[MAX_GPU], streamB[MAX_GPU], streamC[MAX_GPU], streamD[MAX_GPU];
	cudaEvent_t eventA[MAX_GPU], eventB[MAX_GPU], eventC[MAX_GPU], eventD[MAX_GPU];

	cudaEvent_t eventStart1[MAX_GPU], eventStart2[MAX_GPU], eventStart3[MAX_GPU], eventStop1[MAX_GPU], eventStop2[MAX_GPU], eventStop3[MAX_GPU];

	int device_id, device_data_length;
	int data_part_index;
	float* image_host;
	//float* fp_x_host;
	//float* fp_y_host;
	string filename;
	stringstream sstm;
	FILE *output_file;

	float milliseconds1 = 0.0f;
	float milliseconds2 = 0.0f;
	float milliseconds3 = 0.0f;

	dim3 dimBlock_f_x(BlockSize_forward_x);
	dim3 dimGrid_f_x(GridSize_forward_x);

	dim3 dimBlock_f_y(BlockSize_forward_y);
	dim3 dimGrid_f_y(GridSize_forward_y);

	dim3 dimBlock_b_x(BlockSize_backward_x);
	dim3 dimGrid_b_x(GridSize_backward_x);

	dim3 dimBlock_b_y(BlockSize_backward_y);
	dim3 dimGrid_b_y(GridSize_backward_y);



	dim3 dimBlock(1024);
	dim3 dimGrid(64);

	//copy image to all GPUs
	//cudaHostRegister(Image._image, parameters_host.NUM_XYZ*sizeof(float), cudaHostAllocPortable);
	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		checkCudaErrors(cudaSetDevice(device_id));
		checkCudaErrors(cudaMemcpy(_current_image_device[device_id], Atten_Image._image, parameters_host.NUM_XYZ*sizeof(float), cudaMemcpyDefault));
		//checkCudaErrors(cudaMemcpy(_current2_image_device[device_id], Emission_Image._image, parameters_host.NUM_XYZ*sizeof(float), cudaMemcpyDefault));
		//setting the update factor to zero before projection
		setImageToZero_kernel << <dimGrid, dimBlock >> >(_update_factor_device[device_id], _parameters_device[device_id]);
	}
	//cudaHostUnregister(Image._image);

	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		//checkCudaErrors(cudaSetDevice(device_id));
	}

	printf("\n\n\n\n.......................Start SS data projection.........................\n");
	//SS events

	//copy data to all GPUs
	device_data_length = data.GetDataCount(ALL) / (_num_gpu_end - _num_gpu_start + 1);

	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		data_part_index = device_id - _num_gpu_start;
		if (device_id < _num_gpu_end){
			Setup_data(detector, movement, data, data_part_index*device_data_length, (data_part_index + 1)*device_data_length - 1, SS, device_id);
		}
		else{
			Setup_data(detector, movement, data, data_part_index*device_data_length, data.GetDataCount(ALL) - 1, SS, device_id);
		}
	}
	//printf("ALL SS data copied to device memory\n");

	parameters_host.FWHM_alpha = parameters_host.FWHM_alpha_ss;
	parameters_host.FWHM_sigma_inv = parameters_host.FWHM_sigma_inv_ss;

	parameters_host.RANGE1 = (int)(parameters_host.FWHM_ss / parameters_host.X_SAMP);
	parameters_host.RANGE2 = (int)(parameters_host.FWHM_ss / parameters_host.Z_SAMP);
	parameters_host.RANGE_atten_1 = (int)(parameters_host.FWHM_ss / parameters_host.X_SAMP);
	parameters_host.RANGE_atten_2 = (int)(parameters_host.FWHM_ss / parameters_host.Z_SAMP);
	//printf("H_CENTER_X = %f H_CENTER_Y = %f V_CENTER = %f\n", parameters_host.H_CENTER_X, parameters_host.H_CENTER_Y, parameters_host.V_CENTER);
	//printf("RANGE_atten_1 = %d RANGE_atten_2 = %d FWHM_SS = %f\n", parameters_host.RANGE_atten_1, parameters_host.RANGE_atten_2, parameters_host.FWHM_ss);


	cudaHostRegister(&parameters_host, sizeof(PARAMETERS_IN_DEVICE_t), 0);

	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		checkCudaErrors(cudaSetDevice(device_id));
		checkCudaErrors(cudaDeviceSynchronize());


		checkCudaErrors(cudaStreamCreate(&streamA[device_id]));
		checkCudaErrors(cudaStreamCreate(&streamB[device_id]));
		checkCudaErrors(cudaStreamCreate(&streamC[device_id]));

		checkCudaErrors(cudaEventCreateWithFlags(&eventA[device_id], cudaEventDisableTiming));
		//checkCudaErrors(cudaEventCreate(&eventB[device_id]));
		//checkCudaErrors(cudaEventCreate(&eventC[device_id]));

		checkCudaErrors(cudaEventCreate(&eventStart1[device_id]));
		checkCudaErrors(cudaEventCreate(&eventStart2[device_id]));

		checkCudaErrors(cudaEventCreate(&eventStop1[device_id]));
		checkCudaErrors(cudaEventCreate(&eventStop2[device_id]));

	}

	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		cudaSetDevice(device_id);
		printf("\n\n\n\n.......................Start projection on Device %d.........................\n", device_id);

		checkCudaErrors(cudaMemcpyAsync(_parameters_device[device_id], &parameters_host, sizeof(PARAMETERS_IN_DEVICE_t), cudaMemcpyDefault, streamA[device_id]));
		checkCudaErrors(cudaEventRecord(eventA[device_id], streamA[device_id]));

		setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamB[device_id] >> >(_fp_value_x_device[device_id], _event_count_x[device_id]);

		setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamC[device_id] >> >(_fp_value_y_device[device_id], _event_count_y[device_id]);


		//setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamB[device_id] >> >(_fp_value2_x_device[device_id], _event_count_x[device_id]);

		//setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamC[device_id] >> >(_fp_value2_y_device[device_id], _event_count_y[device_id]);



		if (_TOF_mode == TOF){
			//TOF version
			//printf("starting TOF kernels.\n");

			checkCudaErrors(cudaStreamWaitEvent(streamB[device_id], eventA[device_id], 0));
			checkCudaErrors(cudaEventRecord(eventStart1[device_id], streamB[device_id]));
			_fproj_atten_lst_cuda_x_kernel << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_current_image_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
			//_fproj_lst_cuda_x_kernel << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_current_image_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);

			//_fproj_lst_cuda_x_kernel << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_current2_image_device[device_id], _fp_value2_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);

			checkCudaErrors(cudaStreamWaitEvent(streamC[device_id], eventA[device_id], 0));
			checkCudaErrors(cudaEventRecord(eventStart2[device_id], streamC[device_id]));
			_fproj_atten_lst_cuda_y_kernel << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_current_image_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
			//_fproj_lst_cuda_y_kernel << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_current_image_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);

			//_fproj_lst_cuda_y_kernel << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_current2_image_device[device_id], _fp_value2_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);

			_bproj_atten_lst_cuda_x_kernel << <dimGrid_b_x, dimBlock_b_x, 0, streamB[device_id] >> >(_update_factor_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
			//cudaEventRecord(eventB[device_id], streamB[device_id]);
			checkCudaErrors(cudaEventRecord(eventStop1[device_id], streamB[device_id]));

			_bproj_atten_lst_cuda_y_kernel << <dimGrid_b_y, dimBlock_b_y, 0, streamC[device_id] >> >(_update_factor_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
			//cudaEventRecord(eventC[device_id], streamC[device_id]);
			checkCudaErrors(cudaEventRecord(eventStop2[device_id], streamC[device_id]));


		}
		else{
			//NON-TOF version
			//printf("starting NON-TOF kernels.\n");

			checkCudaErrors(cudaStreamWaitEvent(streamB[device_id], eventA[device_id], 0));
			checkCudaErrors(cudaEventRecord(eventStart1[device_id], streamB[device_id]));

			_fproj_atten_lst_cuda_x_kernel << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_current_image_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);



			checkCudaErrors(cudaStreamWaitEvent(streamC[device_id], eventA[device_id], 0));
			checkCudaErrors(cudaEventRecord(eventStart2[device_id], streamC[device_id]));

			_fproj_atten_lst_cuda_y_kernel << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_current_image_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);




			_bproj_atten_lst_cuda_x_kernel << <dimGrid_b_x, dimBlock_b_x, 0, streamB[device_id] >> >(_update_factor_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
			//cudaEventRecord(eventB[device_id], streamB[device_id]);
			checkCudaErrors(cudaEventRecord(eventStop1[device_id], streamB[device_id]));
			_bproj_atten_lst_cuda_y_kernel << <dimGrid_b_y, dimBlock_b_y, 0, streamC[device_id] >> >(_update_factor_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
			//cudaEventRecord(eventC[device_id], streamC[device_id]);
			checkCudaErrors(cudaEventRecord(eventStop2[device_id], streamC[device_id]));

		}

	}

	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		checkCudaErrors(cudaSetDevice(device_id));

		/*
		checkCudaErrors(cudaStreamSynchronize(streamA[device_id]));
		checkCudaErrors(cudaStreamSynchronize(streamB[device_id]));
		checkCudaErrors(cudaStreamSynchronize(streamC[device_id]));
		*/

		checkCudaErrors(cudaEventSynchronize(eventStart1[device_id]));
		checkCudaErrors(cudaEventSynchronize(eventStart2[device_id]));

		checkCudaErrors(cudaEventSynchronize(eventStop1[device_id]));
		checkCudaErrors(cudaEventSynchronize(eventStop2[device_id]));


		checkCudaErrors(cudaEventElapsedTime(&milliseconds1, eventStart1[device_id], eventStop1[device_id]));
		checkCudaErrors(cudaEventElapsedTime(&milliseconds2, eventStart2[device_id], eventStop2[device_id]));

		printf("\n.......................Wrapping up for device %d........................\n", device_id);
		printf("Time for X-dominant projections: %f seconds\n", milliseconds1 / 1000);
		printf("Time for Y-dominant projections: %f seconds\n", milliseconds2 / 1000);


		checkCudaErrors(cudaEventDestroy(eventA[device_id]));
		/*
		checkCudaErrors(cudaEventDestroy(eventB[device_id]));
		checkCudaErrors(cudaEventDestroy(eventC[device_id]));
		*/

		checkCudaErrors(cudaEventDestroy(eventStart1[device_id]));
		checkCudaErrors(cudaEventDestroy(eventStart2[device_id]));

		checkCudaErrors(cudaEventDestroy(eventStop1[device_id]));
		checkCudaErrors(cudaEventDestroy(eventStop2[device_id]));


		checkCudaErrors(cudaStreamDestroy(streamA[device_id]));
		checkCudaErrors(cudaStreamDestroy(streamB[device_id]));
		checkCudaErrors(cudaStreamDestroy(streamC[device_id]));

		//cudaDeviceSynchronize();
	}

	cudaHostUnregister(&parameters_host);

	//printf("\n... start clean up data\n\n");
	Release_data();
	printf("\n... done clean up data\n\n");


	printf("\n\n\n\n.......................Start OS data projection.........................\n");
	//OS events


	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		data_part_index = device_id - _num_gpu_start;
		if (device_id < _num_gpu_end){
			Setup_data(detector, movement, data, data_part_index*device_data_length, (data_part_index + 1)*device_data_length - 1, IS, device_id);
		}
		else{
			Setup_data(detector, movement, data, data_part_index*device_data_length, data.GetDataCount(ALL) - 1, IS, device_id);
		}
	}
	//printf("ALL SS data copied to device memory\n");

	parameters_host.FWHM_alpha = parameters_host.FWHM_alpha_is;
	parameters_host.FWHM_sigma_inv = parameters_host.FWHM_sigma_inv_is;

	parameters_host.RANGE1 = (int)(parameters_host.FWHM_is / parameters_host.X_SAMP);
	parameters_host.RANGE2 = (int)(parameters_host.FWHM_is / parameters_host.Z_SAMP);
	parameters_host.RANGE_atten_1 = (int)(parameters_host.FWHM_is / parameters_host.X_SAMP);
	parameters_host.RANGE_atten_2 = (int)(parameters_host.FWHM_is / parameters_host.Z_SAMP);
	if (parameters_host.RANGE_atten_1 < 1)
		parameters_host.RANGE_atten_1 = 1;
	if (parameters_host.RANGE_atten_2 < 1)
		parameters_host.RANGE_atten_2 = 1;
	//printf("H_CENTER_X = %f H_CENTER_Y = %f V_CENTER = %f\n", parameters_host.H_CENTER_X, parameters_host.H_CENTER_Y, parameters_host.V_CENTER);
	//printf("RANGE_atten_1 = %d RANGE_atten_2 = %d FWHM_SS = %f\n", parameters_host.RANGE_atten_1, parameters_host.RANGE_atten_2, parameters_host.FWHM_ss);


	cudaHostRegister(&parameters_host, sizeof(PARAMETERS_IN_DEVICE_t), 0);

	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		checkCudaErrors(cudaSetDevice(device_id));
		checkCudaErrors(cudaDeviceSynchronize());


		checkCudaErrors(cudaStreamCreate(&streamA[device_id]));
		checkCudaErrors(cudaStreamCreate(&streamB[device_id]));
		checkCudaErrors(cudaStreamCreate(&streamC[device_id]));

		checkCudaErrors(cudaEventCreateWithFlags(&eventA[device_id], cudaEventDisableTiming));
		//checkCudaErrors(cudaEventCreate(&eventB[device_id]));
		//checkCudaErrors(cudaEventCreate(&eventC[device_id]));

		checkCudaErrors(cudaEventCreate(&eventStart1[device_id]));
		checkCudaErrors(cudaEventCreate(&eventStart2[device_id]));

		checkCudaErrors(cudaEventCreate(&eventStop1[device_id]));
		checkCudaErrors(cudaEventCreate(&eventStop2[device_id]));

	}

	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		cudaSetDevice(device_id);
		printf("\n\n\n\n.......................Start projection on Device %d.........................\n", device_id);

		checkCudaErrors(cudaMemcpyAsync(_parameters_device[device_id], &parameters_host, sizeof(PARAMETERS_IN_DEVICE_t), cudaMemcpyDefault, streamA[device_id]));
		checkCudaErrors(cudaEventRecord(eventA[device_id], streamA[device_id]));

		setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamB[device_id] >> >(_fp_value_x_device[device_id], _event_count_x[device_id]);

		setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamC[device_id] >> >(_fp_value_y_device[device_id], _event_count_y[device_id]);


		//setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamB[device_id] >> >(_fp_value2_x_device[device_id], _event_count_x[device_id]);

		//setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamC[device_id] >> >(_fp_value2_y_device[device_id], _event_count_y[device_id]);



		if (_TOF_mode == TOF){
			//TOF version
			//printf("starting TOF kernels.\n");

			checkCudaErrors(cudaStreamWaitEvent(streamB[device_id], eventA[device_id], 0));
			checkCudaErrors(cudaEventRecord(eventStart1[device_id], streamB[device_id]));
			_fproj_atten_lst_cuda_x_kernel << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_current_image_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);

			//_fproj_lst_cuda_x_kernel << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_current2_image_device[device_id], _fp_value2_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);

			checkCudaErrors(cudaStreamWaitEvent(streamC[device_id], eventA[device_id], 0));
			checkCudaErrors(cudaEventRecord(eventStart2[device_id], streamC[device_id]));
			_fproj_atten_lst_cuda_y_kernel << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_current_image_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);

			//_fproj_lst_cuda_y_kernel << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_current2_image_device[device_id], _fp_value2_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);

			_bproj_atten_lst_cuda_x_kernel << <dimGrid_b_x, dimBlock_b_x, 0, streamB[device_id] >> >(_update_factor_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
			//cudaEventRecord(eventB[device_id], streamB[device_id]);
			checkCudaErrors(cudaEventRecord(eventStop1[device_id], streamB[device_id]));

			_bproj_atten_lst_cuda_y_kernel << <dimGrid_b_y, dimBlock_b_y, 0, streamC[device_id] >> >(_update_factor_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
			//cudaEventRecord(eventC[device_id], streamC[device_id]);
			checkCudaErrors(cudaEventRecord(eventStop2[device_id], streamC[device_id]));


		}
		else{
			//NON-TOF version
			//printf("starting NON-TOF kernels.\n");

			checkCudaErrors(cudaStreamWaitEvent(streamB[device_id], eventA[device_id], 0));
			checkCudaErrors(cudaEventRecord(eventStart1[device_id], streamB[device_id]));

			_fproj_atten_lst_cuda_x_kernel << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_current_image_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);



			checkCudaErrors(cudaStreamWaitEvent(streamC[device_id], eventA[device_id], 0));
			checkCudaErrors(cudaEventRecord(eventStart2[device_id], streamC[device_id]));

			_fproj_atten_lst_cuda_y_kernel << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_current_image_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);




			_bproj_atten_lst_cuda_x_kernel << <dimGrid_b_x, dimBlock_b_x, 0, streamB[device_id] >> >(_update_factor_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
			//cudaEventRecord(eventB[device_id], streamB[device_id]);
			checkCudaErrors(cudaEventRecord(eventStop1[device_id], streamB[device_id]));
			_bproj_atten_lst_cuda_y_kernel << <dimGrid_b_y, dimBlock_b_y, 0, streamC[device_id] >> >(_update_factor_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
			//cudaEventRecord(eventC[device_id], streamC[device_id]);
			checkCudaErrors(cudaEventRecord(eventStop2[device_id], streamC[device_id]));

		}

	}

	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		checkCudaErrors(cudaSetDevice(device_id));

		/*
		checkCudaErrors(cudaStreamSynchronize(streamA[device_id]));
		checkCudaErrors(cudaStreamSynchronize(streamB[device_id]));
		checkCudaErrors(cudaStreamSynchronize(streamC[device_id]));
		*/

		checkCudaErrors(cudaEventSynchronize(eventStart1[device_id]));
		checkCudaErrors(cudaEventSynchronize(eventStart2[device_id]));

		checkCudaErrors(cudaEventSynchronize(eventStop1[device_id]));
		checkCudaErrors(cudaEventSynchronize(eventStop2[device_id]));


		checkCudaErrors(cudaEventElapsedTime(&milliseconds1, eventStart1[device_id], eventStop1[device_id]));
		checkCudaErrors(cudaEventElapsedTime(&milliseconds2, eventStart2[device_id], eventStop2[device_id]));

		printf("\n.......................Wrapping up for device %d........................\n", device_id);
		printf("Time for X-dominant projections: %f seconds\n", milliseconds1 / 1000);
		printf("Time for Y-dominant projections: %f seconds\n", milliseconds2 / 1000);


		checkCudaErrors(cudaEventDestroy(eventA[device_id]));
		/*
		checkCudaErrors(cudaEventDestroy(eventB[device_id]));
		checkCudaErrors(cudaEventDestroy(eventC[device_id]));
		*/

		checkCudaErrors(cudaEventDestroy(eventStart1[device_id]));
		checkCudaErrors(cudaEventDestroy(eventStart2[device_id]));

		checkCudaErrors(cudaEventDestroy(eventStop1[device_id]));
		checkCudaErrors(cudaEventDestroy(eventStop2[device_id]));


		checkCudaErrors(cudaStreamDestroy(streamA[device_id]));
		checkCudaErrors(cudaStreamDestroy(streamB[device_id]));
		checkCudaErrors(cudaStreamDestroy(streamC[device_id]));

		//cudaDeviceSynchronize();
	}

	cudaHostUnregister(&parameters_host);

	//printf("\n... start clean up data\n\n");
	Release_data();
	printf("\n... done clean up data\n\n");


	printf("\n\n\n\n.......................Start II data projection.........................\n");
	//IS events


	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		data_part_index = device_id - _num_gpu_start;
		if (device_id < _num_gpu_end){
			Setup_data(detector, movement, data, data_part_index*device_data_length, (data_part_index + 1)*device_data_length - 1, II, device_id);
		}
		else{
			Setup_data(detector, movement, data, data_part_index*device_data_length, data.GetDataCount(ALL) - 1, II, device_id);
		}
	}
	//printf("ALL SS data copied to device memory\n");

	parameters_host.FWHM_alpha = parameters_host.FWHM_alpha_ii;
	parameters_host.FWHM_sigma_inv = parameters_host.FWHM_sigma_inv_ii;

	parameters_host.RANGE1 = (int)(parameters_host.FWHM_ii / parameters_host.X_SAMP);
	parameters_host.RANGE2 = (int)(parameters_host.FWHM_ii / parameters_host.Z_SAMP);
	parameters_host.RANGE_atten_1 = (int)(parameters_host.FWHM_ii / parameters_host.X_SAMP);
	parameters_host.RANGE_atten_2 = (int)(parameters_host.FWHM_ii / parameters_host.Z_SAMP);
	if (parameters_host.RANGE_atten_1 < 1)
		parameters_host.RANGE_atten_1 = 1;
	if (parameters_host.RANGE_atten_2 < 1)
		parameters_host.RANGE_atten_2 = 1;
	//printf("H_CENTER_X = %f H_CENTER_Y = %f V_CENTER = %f\n", parameters_host.H_CENTER_X, parameters_host.H_CENTER_Y, parameters_host.V_CENTER);
	//printf("RANGE_atten_1 = %d RANGE_atten_2 = %d FWHM_SS = %f\n", parameters_host.RANGE_atten_1, parameters_host.RANGE_atten_2, parameters_host.FWHM_ss);


	cudaHostRegister(&parameters_host, sizeof(PARAMETERS_IN_DEVICE_t), 0);

	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		checkCudaErrors(cudaSetDevice(device_id));
		checkCudaErrors(cudaDeviceSynchronize());


		checkCudaErrors(cudaStreamCreate(&streamA[device_id]));
		checkCudaErrors(cudaStreamCreate(&streamB[device_id]));
		checkCudaErrors(cudaStreamCreate(&streamC[device_id]));

		checkCudaErrors(cudaEventCreateWithFlags(&eventA[device_id], cudaEventDisableTiming));
		//checkCudaErrors(cudaEventCreate(&eventB[device_id]));
		//checkCudaErrors(cudaEventCreate(&eventC[device_id]));

		checkCudaErrors(cudaEventCreate(&eventStart1[device_id]));
		checkCudaErrors(cudaEventCreate(&eventStart2[device_id]));

		checkCudaErrors(cudaEventCreate(&eventStop1[device_id]));
		checkCudaErrors(cudaEventCreate(&eventStop2[device_id]));

	}

	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		cudaSetDevice(device_id);
		printf("\n\n\n\n.......................Start projection on Device %d.........................\n", device_id);

		checkCudaErrors(cudaMemcpyAsync(_parameters_device[device_id], &parameters_host, sizeof(PARAMETERS_IN_DEVICE_t), cudaMemcpyDefault, streamA[device_id]));
		checkCudaErrors(cudaEventRecord(eventA[device_id], streamA[device_id]));

		setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamB[device_id] >> >(_fp_value_x_device[device_id], _event_count_x[device_id]);

		setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamC[device_id] >> >(_fp_value_y_device[device_id], _event_count_y[device_id]);


		//setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamB[device_id] >> >(_fp_value2_x_device[device_id], _event_count_x[device_id]);

		//setForwardProjectionValueToZero_kernel << <dimGrid, dimBlock, 0, streamC[device_id] >> >(_fp_value2_y_device[device_id], _event_count_y[device_id]);



		if (_TOF_mode == TOF){
			//TOF version
			//printf("starting TOF kernels.\n");

			checkCudaErrors(cudaStreamWaitEvent(streamB[device_id], eventA[device_id], 0));
			checkCudaErrors(cudaEventRecord(eventStart1[device_id], streamB[device_id]));
			_fproj_atten_lst_cuda_x_kernel << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_current_image_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);

			//_fproj_lst_cuda_x_kernel << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_current2_image_device[device_id], _fp_value2_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);

			checkCudaErrors(cudaStreamWaitEvent(streamC[device_id], eventA[device_id], 0));
			checkCudaErrors(cudaEventRecord(eventStart2[device_id], streamC[device_id]));
			_fproj_atten_lst_cuda_y_kernel << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_current_image_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);

			//_fproj_lst_cuda_y_kernel << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_current2_image_device[device_id], _fp_value2_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);

			_bproj_atten_lst_cuda_x_kernel << <dimGrid_b_x, dimBlock_b_x, 0, streamB[device_id] >> >(_update_factor_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
			//cudaEventRecord(eventB[device_id], streamB[device_id]);
			checkCudaErrors(cudaEventRecord(eventStop1[device_id], streamB[device_id]));

			_bproj_atten_lst_cuda_y_kernel << <dimGrid_b_y, dimBlock_b_y, 0, streamC[device_id] >> >(_update_factor_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
			//cudaEventRecord(eventC[device_id], streamC[device_id]);
			checkCudaErrors(cudaEventRecord(eventStop2[device_id], streamC[device_id]));


		}
		else{
			//NON-TOF version
			//printf("starting NON-TOF kernels.\n");

			checkCudaErrors(cudaStreamWaitEvent(streamB[device_id], eventA[device_id], 0));
			checkCudaErrors(cudaEventRecord(eventStart1[device_id], streamB[device_id]));

			_fproj_atten_lst_cuda_x_kernel << <dimGrid_f_x, dimBlock_f_x, 0, streamB[device_id] >> >(_current_image_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);



			checkCudaErrors(cudaStreamWaitEvent(streamC[device_id], eventA[device_id], 0));
			checkCudaErrors(cudaEventRecord(eventStart2[device_id], streamC[device_id]));

			_fproj_atten_lst_cuda_y_kernel << <dimGrid_f_y, dimBlock_f_y, 0, streamC[device_id] >> >(_current_image_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);




			_bproj_atten_lst_cuda_x_kernel << <dimGrid_b_x, dimBlock_b_x, 0, streamB[device_id] >> >(_update_factor_device[device_id], _fp_value_x_device[device_id], _events_x_dominant_global_mem[device_id], _parameters_device[device_id]);
			//cudaEventRecord(eventB[device_id], streamB[device_id]);
			checkCudaErrors(cudaEventRecord(eventStop1[device_id], streamB[device_id]));
			_bproj_atten_lst_cuda_y_kernel << <dimGrid_b_y, dimBlock_b_y, 0, streamC[device_id] >> >(_update_factor_device[device_id], _fp_value_y_device[device_id], _events_y_dominant_global_mem[device_id], _parameters_device[device_id]);
			//cudaEventRecord(eventC[device_id], streamC[device_id]);
			checkCudaErrors(cudaEventRecord(eventStop2[device_id], streamC[device_id]));

		}

	}

	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		checkCudaErrors(cudaSetDevice(device_id));

		/*
		checkCudaErrors(cudaStreamSynchronize(streamA[device_id]));
		checkCudaErrors(cudaStreamSynchronize(streamB[device_id]));
		checkCudaErrors(cudaStreamSynchronize(streamC[device_id]));
		*/

		checkCudaErrors(cudaEventSynchronize(eventStart1[device_id]));
		checkCudaErrors(cudaEventSynchronize(eventStart2[device_id]));

		checkCudaErrors(cudaEventSynchronize(eventStop1[device_id]));
		checkCudaErrors(cudaEventSynchronize(eventStop2[device_id]));


		checkCudaErrors(cudaEventElapsedTime(&milliseconds1, eventStart1[device_id], eventStop1[device_id]));
		checkCudaErrors(cudaEventElapsedTime(&milliseconds2, eventStart2[device_id], eventStop2[device_id]));

		printf("\n.......................Wrapping up for device %d........................\n", device_id);
		printf("Time for X-dominant projections: %f seconds\n", milliseconds1 / 1000);
		printf("Time for Y-dominant projections: %f seconds\n", milliseconds2 / 1000);


		checkCudaErrors(cudaEventDestroy(eventA[device_id]));
		/*
		checkCudaErrors(cudaEventDestroy(eventB[device_id]));
		checkCudaErrors(cudaEventDestroy(eventC[device_id]));
		*/

		checkCudaErrors(cudaEventDestroy(eventStart1[device_id]));
		checkCudaErrors(cudaEventDestroy(eventStart2[device_id]));

		checkCudaErrors(cudaEventDestroy(eventStop1[device_id]));
		checkCudaErrors(cudaEventDestroy(eventStop2[device_id]));


		checkCudaErrors(cudaStreamDestroy(streamA[device_id]));
		checkCudaErrors(cudaStreamDestroy(streamB[device_id]));
		checkCudaErrors(cudaStreamDestroy(streamC[device_id]));

		//cudaDeviceSynchronize();
	}

	cudaHostUnregister(&parameters_host);

	//printf("\n... start clean up data\n\n");
	Release_data();
	printf("\n... done clean up data\n\n");


	

	Norm_Image.SetValue(0.0f);
	image_host = (float*)malloc(parameters_host.NUM_XYZ*sizeof(float));
	//copy the update factor back to host memory
	for (device_id = _num_gpu_start; device_id <= _num_gpu_end; device_id++){
		cudaSetDevice(device_id);
		cudaDeviceSynchronize();
		cudaMemcpy(image_host, _update_factor_device[device_id], parameters_host.NUM_XYZ*sizeof(float), cudaMemcpyDefault);
		Norm_Image.AddFromMem(image_host);
		/*
		sstm.str("");
		sstm << _output_path << _output_filename_prefix << "_device_"<< device_id<<".img";
		Atten_Image.ReadFromMem(image_host);
		Atten_Image.WriteToFile(sstm.str());
		*/
	}


	printf("\n\n\n\n.......................End projection.........................\n");


	free(image_host);


}


