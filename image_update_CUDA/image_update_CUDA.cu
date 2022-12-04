
#define _USE_MATH_DEFINES
#include <math.h>
#include <omp.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cuda.h>
#include "image_update_CUDA.cuh"
#include "../common/inc/helper_functions.h"
#include "../common/inc/helper_cuda.h"


__global__ void SetImageValue(float* image, int size, float value){
	int ind;
	for (ind = threadIdx.x + blockDim.x*blockIdx.x; ind < size; ind = ind + blockDim.x*gridDim.x){
		image[ind] = value;
	}
}

__global__ void MLImageUpdate_CUDA(float* new_image, float* current_image, float* update_factor_image, float* mask_image, float* sensitivity_image, parameters_t para, PET_geometry g, int z_start, int z_end, float* function_image, float* gradient_image, float* hessian_image, float* th_up, float* th_lo){
	int ind;
	for (ind = threadIdx.x + blockDim.x*blockIdx.x + (z_start - 1)*g.NUM_XY; ind < z_end*g.NUM_XY; ind = ind + blockDim.x*gridDim.x){
		if (mask_image[ind] > 0.5f){

			new_image[ind] = current_image[ind] * update_factor_image[ind] / sensitivity_image[ind];

			if (mask_image[ind] <0.5f){
				//new_image[ind] = 0.0f;
			}
		}
	}
}

__global__ void ComputeDecomposedPenaltyFunction(float* penalty, float* new_image, float* current_image, float* mask_image, parameters_t* para, PET_geometry* g){

	// For 26 neighbours, there are 7 types of neighbour distance, in a simplified case, there are only 3 types assuming using cubic voxels
	float weights[4] = { 0.0f };
	int ind, nind;
	int ind_x, ind_y, ind_z;
	int dx, dy, dz;
	int nx, ny, nz;
	float diff, val;
	float weight;
	float temp;
	float sum_thread;
	__shared__ float sum;

	if (threadIdx.x == 0){
		sum = 0.0f;
	}
	__syncthreads();
	//weights[0] for center voxel itself
	weights[0] = 0.0f;

	weights[1] = g->INV_X_SAMP;

	weights[2] = g->INV_X_SAMP* __frsqrt_rn(2.0f);

	weights[3] = g->INV_X_SAMP* __frsqrt_rn(3.0f);

	for (ind = threadIdx.x + blockDim.x*blockIdx.x; ind < g->NUM_XYZ; ind = ind + blockDim.x*gridDim.x){
		
		sum_thread = 0.0f;

		if (mask_image[ind] > 0.5f){

			val = 2 * new_image[ind] - current_image[ind];
			ind_z = ind / g->NUM_XY;
			ind_y = (ind - ind_z*g->NUM_XY) / g->NUM_X;
			ind_x = ind - ind_z*g->NUM_XY - ind_y*g->NUM_X;

			for (dz = -1; dz <= 1; dz++){
				for (dy = -1; dy <= 1; dy++){
					for (dx = -1; dx <= 1; dx++){

						nx = ind_x + dx;
						ny = ind_y + dy;
						nz = ind_z + dz;

						//test if this voxel is within the range
						if (nx >= 0 && nx < g->NUM_X && ny >= 0 && ny < g->NUM_Y && nz >= 0 && nz < g->NUM_Z){
							nind = nz*g->NUM_XY + ny*g->NUM_X + nx;

							if (mask_image[nind] > 0.5f){
								diff = (val - current_image[nind]) / para->prior_delta;
								//determine the weights
								weight = weights[__sad(dz, 0, __sad(dy, 0, __sad(dx, 0, 0)))];

								if (diff < 20.0f){
									temp = logf((__expf(diff) + __expf(-diff)) / 2);
								}
								else{
									temp = diff - logf(2.0f);
								}
								sum_thread += weight*temp;
							}
						}
					}
				}
			}
		}
		atomicAdd(&sum, sum_thread);
	}
	__syncthreads();

	if (threadIdx.x == 0){
		sum = sum*para->prior_beta*para->prior_delta*para->prior_delta*0.5f;
		atomicAdd(penalty, sum);
	}

}

__global__ void ComputeOriginalPenaltyFunction(float* penalty, float* new_image, float* mask_image, parameters_t* para, PET_geometry* g){
	
	// For 26 neighbours, there are 7 types of neighbour distance, in a simplified case, there are only 3 types assuming using cubic voxels
	float weights[4] = { 0.0f };
	int ind,nind;
	int ind_x, ind_y, ind_z;
	int dx, dy, dz;
	int nx, ny, nz;
	float diff,val;
	float weight;
	float temp;
	float sum_thread;
	__shared__ float sum;

	if (threadIdx.x == 0){
		sum = 0.0f;
	}
	__syncthreads();
	//weights[0] for center voxel itself
	weights[0] = 0.0f;

	weights[1] = g->INV_X_SAMP;

	weights[2] = g->INV_X_SAMP* __frsqrt_rn(2.0f);

	weights[3] = g->INV_X_SAMP* __frsqrt_rn(3.0f);

	for (ind = threadIdx.x + blockDim.x*blockIdx.x; ind < g->NUM_XYZ; ind = ind + blockDim.x*gridDim.x){
		
		sum_thread = 0.0f;

		if (mask_image[ind] > 0.5f){
		
			val = new_image[ind];
			ind_z = ind / g->NUM_XY;
			ind_y = (ind - ind_z*g->NUM_XY) / g->NUM_X;
			ind_x = ind - ind_z*g->NUM_XY - ind_y*g->NUM_X;
		
			for (dz = -1; dz <= 1; dz++){
				for (dy = -1; dy <= 1; dy++){
					for (dx = -1; dx <= 1; dx++){

						nx = ind_x + dx;
						ny = ind_y + dy;
						nz = ind_z + dz;

						//test if this voxel is within the range
						if (nx >= 0 && nx < g->NUM_X && ny >= 0 && ny < g->NUM_Y && nz >= 0 && nz < g->NUM_Z){
							nind = nz*g->NUM_XY + ny*g->NUM_X + nx;

							if (mask_image[nind] > 0.5f){
								diff = (val - new_image[nind]) / para->prior_delta;
								//determine the weights
								weight = weights[__sad(dz, 0, __sad(dy, 0, __sad(dx, 0, 0)))];
								
								if (diff < 20.0f){
									temp = logf((__expf(diff) + __expf(-diff)) / 2);
								}
								else{
									temp = diff - logf(2.0f);
								}
								sum_thread += weight*temp;
							}
						}
					}
				}
			}
		}
		atomicAdd(&sum, sum_thread);
	}
	__syncthreads();

	if (threadIdx.x == 0){
		sum = sum*para->prior_beta*para->prior_delta*para->prior_delta;
		atomicAdd(penalty, sum);
	}
}

__global__ void ComputeTotalFGH(float* penalty, float* new_image, float* current_image, float* update_factor_image, float* mask_image, float* sensitivity_image, parameters_t para, PET_geometry g, int z_start, int z_end, float* function_image, float* gradient_image, float* hessian_image){
	
	//This function computes the element-wise Function, Gradient and Hessian of both the data fitting term and penalty term
	// For 26 neighbours, there are 7 types of neighbour distance, in a simplified case, there are only 3 types assuming using cubic voxels
	float weights[4] = { 0.0f };
	int ind, nind;
	int ind_x, ind_y, ind_z;
	int dx, dy, dz;
	int nx, ny, nz;
	float diff, val;
	float weight;
	float temp;
	float new_val;
	float sum_thread;
	float function_val, gradient_val, hessian_val;
	float p_function_val, p_gradient_val, p_hessian_val;
	__shared__ float sum;

	if (threadIdx.x == 0){
		sum = 0.0f;
	}
	__syncthreads();
	//weights[0] for center voxel itself
	weights[0] = 0.0f;

	weights[1] = g.INV_X_SAMP;

	weights[2] = g.INV_X_SAMP* __frsqrt_rn(2.0f);

	weights[3] = g.INV_X_SAMP* __frsqrt_rn(3.0f);


	for (ind = threadIdx.x + blockDim.x*blockIdx.x + (z_start-1)*g.NUM_XY  ; ind < z_end*g.NUM_XY; ind = ind + blockDim.x*gridDim.x){
		if (mask_image[ind] > 0.5f){
			sum_thread = 0.0f;
			new_val = new_image[ind];
			temp = current_image[ind] * update_factor_image[ind];
			
			function_val = sensitivity_image[ind] * new_val - temp*logf(new_val);
			gradient_val = new_val - temp / new_val;
			hessian_val = temp / (new_val*new_val);

			p_function_val = 0.0f;
			p_gradient_val = 0.0f;
			p_hessian_val = 0.0f;

			val = 2*new_val - current_image[ind];
			ind_z = ind / g.NUM_XY;
			ind_y = (ind - ind_z*g.NUM_XY) / g.NUM_X;
			ind_x = ind - ind_z*g.NUM_XY - ind_y*g.NUM_X;

			for (dz = -1; dz <= 1; dz++){
				for (dy = -1; dy <= 1; dy++){
					for (dx = -1; dx <= 1; dx++){

						nx = ind_x + dx;
						ny = ind_y + dy;
						nz = ind_z + dz;

						//test if this voxel is within the range
						if (nx >= 0 && nx < g.NUM_X && ny >= 0 && ny < g.NUM_Y && nz >= 0 && nz < g.NUM_Z){
							nind = nz*g.NUM_XY + ny*g.NUM_X + nx;

							if (mask_image[nind] > 0.5f){
								diff = (val - current_image[nind]) / para.prior_delta;
								//determine the weights
								weight = weights[__sad(dz, 0, __sad(dy, 0, __sad(dx, 0, 0)))];

								//large diff will cause the cosh(diff) loose precision or go beyond the limits
								if (diff < 20.0f){
									new_val = (__expf(diff) + __expf(-diff)) / 2.0f;
									temp = logf(new_val);
									new_val = (__expf(diff) - __expf(-diff)) / (2.0f*new_val);
								}
								else{
									temp = diff - logf(2.0f);
									new_val = 1.0f;
								}
								p_function_val += weight*temp;
								p_gradient_val += weight * new_val;
								p_hessian_val += weight * (1 - new_val*new_val);

							}
						}
					}
				}
			}

			atomicAdd(&sum, p_function_val);
			
			function_image[ind] = function_val + p_function_val*para.prior_beta*para.prior_delta*para.prior_delta*0.5f;
			gradient_image[ind] = gradient_val + p_gradient_val*para.prior_beta*para.prior_delta;
			hessian_image[ind] = hessian_val + p_hessian_val*para.prior_beta*2.0f;
		}
	}
	__syncthreads();

	if (threadIdx.x == 0){
		//sum = sum*para.prior_beta*para.prior_delta*para.prior_delta*0.5f;
		atomicAdd(penalty, sum);
	}

}

__device__ float ComputePenaltyFunctionVal(float*& current_image, float*& mask_image, float& new_image_val, int& ind, parameters_t& para, PET_geometry& g){
	//ind should be already validated for the effective region of image, so no need to validate here
	float weights[4] = { 0.0f };
	int nind;
	int ind_x, ind_y, ind_z;
	int dx, dy, dz;
	int nx, ny, nz;
	float val, diff, weight, temp;
	float function_val;

	function_val = 0.0f;

	weights[0] = 0.0f;
	weights[1] = g.INV_X_SAMP;
	weights[2] = g.INV_X_SAMP* __frsqrt_rn(2.0f);
	weights[3] = g.INV_X_SAMP* __frsqrt_rn(3.0f);

	val = 2 * new_image_val - current_image[ind];
	ind_z = ind / g.NUM_XY;
	ind_y = (ind - ind_z*g.NUM_XY) / g.NUM_X;
	ind_x = ind - ind_z*g.NUM_XY - ind_y*g.NUM_X;

	for (dz = -1; dz <= 1; dz++){
		for (dy = -1; dy <= 1; dy++){
			for (dx = -1; dx <= 1; dx++){

				nx = ind_x + dx;
				ny = ind_y + dy;
				nz = ind_z + dz;

				//test if this voxel is within the range
				if (nx >= 0 && nx < g.NUM_X && ny >= 0 && ny < g.NUM_Y && nz >= 0 && nz < g.NUM_Z){
					nind = nz*g.NUM_XY + ny*g.NUM_X + nx;

					if (mask_image[nind] > 0.5f){
						diff = (val - current_image[nind]) / para.prior_delta;
						//determine the weights
						weight = weights[__sad(dz, 0, __sad(dy, 0, __sad(dx, 0, 0)))];

						if (diff < 20.0f){
							temp = logf((__expf(diff) + __expf(-diff)) / 2);
						}
						else{
							temp = diff - logf(2.0f);
						}
						function_val += weight*temp;
					}
				}
			}
		}
	}
	return function_val*para.prior_delta*para.prior_delta*para.prior_beta*0.5f;
}

__device__ void ComputeVoxelFGH(float*& current_image, float*& update_factor_image, float*& mask_image, float*& sensitivity_image, float& new_image_val, int& ind, parameters_t& para, PET_geometry& g, float& function_val, float& gradient_val, float& hessian_val){
	//ind should be already validated for the effective region of image, so no need to validate here
	float weights[4] = { 0.0f };
	int nind;
	int ind_x, ind_y, ind_z;
	int dx, dy, dz;
	int nx, ny, nz;
	float new_val, val, diff, weight, temp;
	float p_function_val, p_gradient_val, p_hessian_val;


	weights[0] = 0.0f;
	weights[1] = g.INV_X_SAMP;
	weights[2] = g.INV_X_SAMP* __frsqrt_rn(2.0f);
	weights[3] = g.INV_X_SAMP* __frsqrt_rn(3.0f);

	new_val = new_image_val;
	temp = current_image[ind] * update_factor_image[ind];

	function_val = sensitivity_image[ind] * new_val - temp*logf(new_val);
	gradient_val = new_val - temp / new_val;
	hessian_val = temp / (new_val*new_val);

	p_function_val = 0.0f;
	p_gradient_val = 0.0f;
	p_hessian_val = 0.0f;

	val = 2 * new_val - current_image[ind];
	ind_z = ind / g.NUM_XY;
	ind_y = (ind - ind_z*g.NUM_XY) / g.NUM_X;
	ind_x = ind - ind_z*g.NUM_XY - ind_y*g.NUM_X;

	for (dz = -1; dz <= 1; dz++){
		for (dy = -1; dy <= 1; dy++){
			for (dx = -1; dx <= 1; dx++){

				nx = ind_x + dx;
				ny = ind_y + dy;
				nz = ind_z + dz;

				//test if this voxel is within the range
				if (nx >= 0 && nx < g.NUM_X && ny >= 0 && ny < g.NUM_Y && nz >= 0 && nz < g.NUM_Z){
					nind = nz*g.NUM_XY + ny*g.NUM_X + nx;

					if (mask_image[nind] > 0.5f){
						diff = (val - current_image[nind]) / para.prior_delta;
						//determine the weights
						weight = weights[__sad(dz, 0, __sad(dy, 0, __sad(dx, 0, 0)))];

						if (diff < 20.0f){
							//new_val = (expf(diff) + expf(-diff)) / 2.0f;
							temp = logf(coshf(diff));
							//new_val = (expf(diff) - expf(-diff)) / (2.0f*new_val);
						}
						else{
							temp = diff - logf(2.0f);
							//new_val = 1.0f;
						}
						new_val = tanhf(diff);
						p_function_val += weight*temp;
						p_gradient_val += weight * new_val;
						p_hessian_val += weight * (1 - new_val*new_val);
					}
				}
			}
		}
	}

	function_val += p_function_val*para.prior_beta*para.prior_delta*para.prior_delta*0.5f;
	gradient_val += p_gradient_val*para.prior_beta*para.prior_delta;
	hessian_val += p_hessian_val*para.prior_beta*2.0f;
}

__device__ float ComputeVoxelFGH2(float*& current_image, float*& update_factor_image, float*& mask_image, float*& sensitivity_image, float& new_image_val, int& ind, parameters_t& para, PET_geometry& g){
	//ind should be already validated for the effective region of image, so no need to validate here
	float weights[4] = { 0.0f };
	int nind;
	int ind_x, ind_y, ind_z;
	int dx, dy, dz;
	int nx, ny, nz;
	float new_val, val, diff, weight, temp;
	float p_function_val, p_gradient_val, p_hessian_val;
	float function_val, gradient_val, hessian_val;

	weights[0] = 0.0f;
	weights[1] = g.INV_X_SAMP;
	weights[2] = g.INV_X_SAMP* __frsqrt_rn(2.0f);
	weights[3] = g.INV_X_SAMP* __frsqrt_rn(3.0f);

	new_val = new_image_val;
	temp = current_image[ind] * update_factor_image[ind];

	function_val = sensitivity_image[ind] * new_val - temp*logf(new_val);
	gradient_val = new_val - temp / new_val;
	hessian_val = temp / (new_val*new_val);

	p_function_val = 0.0f;
	p_gradient_val = 0.0f;
	p_hessian_val = 0.0f;

	val = 2 * new_val - current_image[ind];
	ind_z = ind / g.NUM_XY;
	ind_y = (ind - ind_z*g.NUM_XY) / g.NUM_X;
	ind_x = ind - ind_z*g.NUM_XY - ind_y*g.NUM_X;

	for (dz = -1; dz <= 1; dz++){
		for (dy = -1; dy <= 1; dy++){
			for (dx = -1; dx <= 1; dx++){

				nx = ind_x + dx;
				ny = ind_y + dy;
				nz = ind_z + dz;

				//test if this voxel is within the range
				if (nx >= 0 && nx < g.NUM_X && ny >= 0 && ny < g.NUM_Y && nz >= 0 && nz < g.NUM_Z){
					nind = nz*g.NUM_XY + ny*g.NUM_X + nx;

					if (mask_image[nind] > 0.5f){
						diff = (val - current_image[nind]) / para.prior_delta;
						//determine the weights
						weight = weights[__sad(dz, 0, __sad(dy, 0, __sad(dx, 0, 0)))];

						if (diff < 20.0f){
							//new_val = (expf(diff) + expf(-diff)) / 2.0f;
							temp = logf(coshf(diff));
							//new_val = (expf(diff) - expf(-diff)) / (2.0f*new_val);
						}
						else{
							temp = diff - logf(2.0f);
							//new_val = 1.0f;
						}
						new_val = tanhf(diff);
						p_function_val += weight*temp;
						p_gradient_val += weight * new_val;
						p_hessian_val += weight * (1 - new_val*new_val);
					}
				}
			}
		}
	}

	function_val += p_function_val*para.prior_beta*para.prior_delta*para.prior_delta*0.5f;
	gradient_val += p_gradient_val*para.prior_beta*para.prior_delta;
	hessian_val += p_hessian_val*para.prior_beta*2.0f;

	return -gradient_val / hessian_val;
}


__global__ void TrustRegionNewtonMethodImageUpdate_CUDA(float* new_image, float* current_image, float* update_factor_image, float* mask_image, float* sensitivity_image, parameters_t para, PET_geometry g, int z_start, int z_end, float* function_image, float* gradient_image, float* hessian_image, float* th_up, float* th_lo){

	int ind;
	float delta;
	int hit_trust_region;
	float th;
	float function, gradient, hessian;
	//float obj_function_quadratic_approximate;
	float obj_function_rsm_new_estimate;
	float rsm_new_estimate, new_estimate;
	float obj_approximate_diff;
	float obj_diff;
	float r_val;

	
	for (ind = threadIdx.x + blockDim.x*blockIdx.x + (z_start - 1)*g.NUM_XY; ind < z_end*g.NUM_XY; ind = ind + blockDim.x*gridDim.x){
		if (mask_image[ind] > 0.5f){
			
			hit_trust_region = 0;
			th = th_up[ind];
			new_estimate = new_image[ind];
			function = function_image[ind];
			gradient = gradient_image[ind];
			hessian = hessian_image[ind];

			//The TH must bond the image value to be positive
			if (new_estimate - th < IMAGE_SMALLEST_ALLOWED){
				th = new_estimate - IMAGE_SMALLEST_ALLOWED;
			}

			delta = -gradient / hessian;

			if (delta < -th){
				delta = -th;
				hit_trust_region = 1;
			}
			else if (delta > th){
				delta = th;
				hit_trust_region = 1;
			}
			
			rsm_new_estimate = new_estimate + delta;

			//obj_function_quadratic_approximate = function + gradient*delta + 0.5f*hessian*delta*delta;
			obj_function_rsm_new_estimate = -current_image[ind] * update_factor_image[ind] * logf(rsm_new_estimate) + sensitivity_image[ind] * rsm_new_estimate + ComputePenaltyFunctionVal(current_image, mask_image, rsm_new_estimate, ind, para, g);

			obj_diff = function - obj_function_rsm_new_estimate;
			//obj_approximate_diff = function - obj_function_quadratic_approximate;
			obj_approximate_diff = -(gradient*delta + 0.5f*hessian*delta*delta);
			r_val = obj_diff / obj_approximate_diff;


			//if r_vals>0.75, then the quadratic approximates the function well.
			//and if the delta is within the restricted region, then the h_val is good enough
			
			/*
			if (r_val > 0.75 && !hit_trust_region){
				th = th;
			}
			*/

			//if the delta is at the boundary of the region, then we need to
			//increase the region
			if (r_val > 0.75f && hit_trust_region){
				th = 2.0f * th;
			}
			//if r_vals<0.25, bad approximation, the curvature is too flat, we
			//need to use a smaller restricted region
			if (r_val < 0.25f){
				th = 0.25f * fabsf(delta);
			}
			//else, h remains the same

			//if r_vals>0, rsm_new_estimate accepted.
			if (r_val > 0.0f){
				new_estimate = rsm_new_estimate;
			}

			new_image[ind] = new_estimate;
			th_up[ind] = th;
			//th_lo[ind] = th;
		}
	}
}

__global__ void TrustRegionNewtonMethodImageUpdate_CUDA2(float* new_image, float* current_image, float* update_factor_image, float* mask_image, float* sensitivity_image, parameters_t para, PET_geometry g, int z_start, int z_end, float th_up, float th_lo){

	int iter;
	int ind;
	float delta;
	int hit_trust_region;
	float th;
	float function, gradient, hessian;
	//float obj_function_quadratic_approximate;
	float obj_function_rsm_new_estimate;
	float rsm_new_estimate, new_estimate;
	float obj_approximate_diff;
	float obj_diff;
	float r_val;


	for (ind = threadIdx.x + blockDim.x*blockIdx.x + (z_start - 1)*g.NUM_XY; ind < z_end*g.NUM_XY; ind = ind + blockDim.x*gridDim.x){
		if (mask_image[ind] > 0.5f){

			new_estimate = current_image[ind];
			th = th_up;

			for (iter = 1; iter <= MAX_NEWTONS_ITER; iter++){

				hit_trust_region = 0;


				//The TH must bond the image value to be positive
				if (new_estimate - th < IMAGE_SMALLEST_ALLOWED){
					th = new_estimate - IMAGE_SMALLEST_ALLOWED;
				}

				ComputeVoxelFGH(current_image, update_factor_image, mask_image, sensitivity_image, new_estimate, ind, para, g, function, gradient, hessian);

				/*
				if (blockIdx.x == 0 && threadIdx.x == 0){
				printf("function = %f\n", hessian);
				}
				*/

				delta = -gradient / hessian;

				if (delta < -th){
					delta = -th;
					hit_trust_region = 1;
				}
				else if (delta > th){
					delta = th;
					hit_trust_region = 1;
				}

				rsm_new_estimate = new_estimate + delta;

				//obj_function_quadratic_approximate = function + gradient*delta + 0.5f*hessian*delta*delta;
				obj_function_rsm_new_estimate = -current_image[ind] * update_factor_image[ind] * logf(rsm_new_estimate) + sensitivity_image[ind] * rsm_new_estimate + ComputePenaltyFunctionVal(current_image, mask_image, rsm_new_estimate, ind, para, g);

				obj_diff = function - obj_function_rsm_new_estimate;
				//obj_approximate_diff = function - obj_function_quadratic_approximate;
				obj_approximate_diff = -(gradient*delta + 0.5f*hessian*delta*delta);
				r_val = obj_diff / obj_approximate_diff;


				//if r_vals>0.75, then the quadratic approximates the function well.
				//and if the delta is within the restricted region, then the h_val is good enough

				/*
				if (r_val > 0.75 && !hit_trust_region){
				th = th;
				}
				*/

				//if the delta is at the boundary of the region, then we need to
				//increase the region
				if (r_val > 0.75f && hit_trust_region){
					th = 2.0f * th;
				}
				//if r_vals<0.25, bad approximation, the curvature is too flat, we
				//need to use a smaller restricted region
				if (r_val < 0.25f){
					th = 0.25f * fabsf(delta);
				}
				//else, h remains the same

				//if r_vals>0, rsm_new_estimate accepted.
				if (r_val > 0.0f){
					new_estimate = rsm_new_estimate;
				}

			}

			if (fabsf(new_estimate - current_image[ind]) < 0.0001){
				if (blockIdx.x == 0 && threadIdx.x == 0){
					printf("almost no update: %f\n", delta);
				}
			}

			new_image[ind] = new_estimate;
		}
	}
}








void
image_update_CUDA::ImageUpdateML_CUDA(ImageArray<float>& new_image, ImageArray<float>& update_factor, ImageArray<float>& current_image, ImageArray<float>&mask_image, ImageArray<float>& sensitivity_image, PET_geometry& g)
{
	// update all slices
	DoImageUpdateML_CUDA(new_image, update_factor, current_image, mask_image, sensitivity_image, g, 1, g.NUM_Z);
}

void
image_update_CUDA::ImageUpdatePL_CUDA(ImageArray<float>& new_image, ImageArray<float>& update_factor, ImageArray<float>& current_image, ImageArray<float>&mask_image, ImageArray<float>& sensitivity_image, parameters_t& para, PET_geometry& g)
{
	// update first and last slices with ML to avoid calculating the penalty on slices reconstructed with missing data

	// update all slices by ML
	DoImageUpdateML_CUDA(new_image, update_factor, current_image, mask_image, sensitivity_image, g, 1, g.NUM_Z);

	if (g.NUM_Z > 2 * NUM_ML_SLICES){
		// update selected slices by PL
		DoImageUpdatePL_CUDA(new_image, update_factor, current_image, mask_image, sensitivity_image, para, g, NUM_ML_SLICES + 1, g.NUM_Z - NUM_ML_SLICES);
	}

}

void
image_update_CUDA::DoImageUpdateML_CUDA(ImageArray<float>& new_image, ImageArray<float>& update_factor, ImageArray<float>& current_image, ImageArray<float>&mask_image, ImageArray<float>& sensitivity_image, PET_geometry& g, int z_start, int z_end)
{
	int z_ind;
	int xy_ind;
	int xyz_ind;

	/*
	omp_set_num_threads(16);
#pragma omp parallel for schedule(dynamic) collapse(2) private(xy_ind, xyz_ind) schedule(dynamic)
	*/


	for (z_ind = z_start; z_ind <= z_end; z_ind++){

		for (xy_ind = 0; xy_ind < g.NUM_XY; xy_ind++){

			xyz_ind = (z_ind - 1)*g.NUM_XY + xy_ind;

			if (mask_image[xyz_ind] > 0.5f){
				new_image[xyz_ind] = current_image[xyz_ind] * update_factor[xyz_ind] / sensitivity_image[xyz_ind];
				

				if (isinf(new_image[xyz_ind])){
					new_image[xyz_ind] = 0.0f;
				}


				if (isnan(new_image[xyz_ind])){

					printf("Nan value detected at X %d Y %d Z %d , %f x %f div %10f, mask %f\n", xy_ind % g.NUM_X, xy_ind / g.NUM_X, z_ind, current_image[xyz_ind], update_factor[xyz_ind], sensitivity_image[xyz_ind], mask_image[xyz_ind]);
				}

				if (isinf(new_image[xyz_ind])){

					printf("Inf value detected at X %d Y %d Z %d , %f x %f div %10f, mask %f\n", xy_ind % g.NUM_X, xy_ind / g.NUM_X, z_ind, current_image[xyz_ind], update_factor[xyz_ind], sensitivity_image[xyz_ind], mask_image[xyz_ind]);
				}


			}
			else{
				new_image[xyz_ind] = 0.0f;
			}
		}
	}
	printf("Image updated by ML");
}

void
image_update_CUDA::DoImageUpdatePL_CUDA(ImageArray<float>& new_image, ImageArray<float>& update_factor, ImageArray<float>& current_image, ImageArray<float>&mask_image, ImageArray<float>& sensitivity_image, parameters_t& para, PET_geometry& g, int z_start, int z_end)
{
	size_t free_mem, total_mem;
	int iter;
	int ind;
	float* host_new_image;

	float host_penalty;
	float* device_penalty;
	float* device_new_image;
	float* device_current_image;
	float* device_update_factor;
	float* device_mask_image;
	float* device_sensitivity_image;
	float* device_function_image;
	float* device_gradient_image;
	float* device_hessian_image;
	float* device_th_up;
	float* device_th_lo;
	size_t image_size;

	dim3 dimBlock(1024);
	dim3 dimGrid(256);

	dim3 dimBlock_update(1024);
	dim3 dimGrid_update(256);
	
	image_size = g.NUM_XYZ*sizeof(float);
	printf("size of image is %d\n", image_size);
		
	host_new_image = (float*)malloc(image_size);

	cuMemGetInfo(&free_mem, &total_mem);
	printf("At this point, free memory = %d MB, Total memory = %d MB\n\n", ((int)free_mem) / 1048576, ((int)total_mem) / 1048576);

	cudaMalloc((void**)&device_penalty, sizeof(float));
	
	
	cudaMalloc((void**)&device_new_image, image_size);
	cudaMalloc((void**)&device_current_image, image_size);
	cudaMalloc((void**)&device_update_factor, image_size);
	cudaMalloc((void**)&device_mask_image, image_size);
	cudaMalloc((void**)&device_sensitivity_image, image_size);

	cudaMemcpy(device_new_image, new_image._image, image_size, cudaMemcpyHostToDevice);
	cudaMemcpy(device_current_image, current_image._image, image_size, cudaMemcpyHostToDevice);
	cudaMemcpy(device_update_factor, update_factor._image, image_size, cudaMemcpyHostToDevice);
	cudaMemcpy(device_mask_image, mask_image._image, image_size, cudaMemcpyHostToDevice);
	cudaMemcpy(device_sensitivity_image, sensitivity_image._image, image_size, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	

	cudaMalloc((void**)&device_function_image, image_size);
	cudaMalloc((void**)&device_gradient_image, image_size);
	cudaMalloc((void**)&device_hessian_image, image_size);
	cudaMalloc((void**)&device_th_up, image_size);
	cudaMalloc((void**)&device_th_lo, image_size);


	
	SetImageValue << <dimGrid, dimBlock >> >(device_function_image, g.NUM_XYZ, 0.0f);
	SetImageValue << <dimGrid, dimBlock >> >(device_gradient_image, g.NUM_XYZ, 0.0f);
	SetImageValue << <dimGrid, dimBlock >> >(device_hessian_image, g.NUM_XYZ, 0.0f);
	SetImageValue << <dimGrid, dimBlock >> >(device_th_up, g.NUM_XYZ, 0.5f);
	SetImageValue << <dimGrid, dimBlock >> >(device_th_lo, g.NUM_XYZ, 0.5f);
	cudaDeviceSynchronize();
	

	
	
	for (iter = 1; iter <= MAX_NEWTONS_ITER; iter++){

		printf("...Now at Newton's iteration = %d\n", iter);

		host_penalty = 0.0;
		cudaMemcpy(device_penalty, &host_penalty, sizeof(float), cudaMemcpyHostToDevice);
		ComputeTotalFGH << <dimGrid_update, dimBlock_update >> >(device_penalty, device_new_image, device_current_image, device_update_factor, device_mask_image, device_sensitivity_image, para, g, z_start, z_end, device_function_image, device_gradient_image, device_hessian_image);
		cudaDeviceSynchronize();
		cudaMemcpy(&host_penalty, device_penalty, sizeof(float), cudaMemcpyDeviceToHost);
		printf("......current penalty value = %f\n\n\n", host_penalty);
		
		TrustRegionNewtonMethodImageUpdate_CUDA << <dimGrid_update, dimBlock_update >> >(device_new_image, device_current_image, device_update_factor, device_mask_image, device_sensitivity_image, para, g, z_start, z_end, device_function_image, device_gradient_image, device_hessian_image, device_th_up, device_th_lo);
		cudaDeviceSynchronize();

	}
	
	
	cudaMemcpy(host_new_image, device_new_image, image_size, cudaMemcpyDeviceToHost);
	for (ind = (z_start - 1)*g.NUM_XY; ind <= z_end*g.NUM_XY; ind++){
		if (host_new_image[ind] > SMALLEST_ALLOWED){
			//printf("%f ", host_new_image[ind]);
		}
		new_image[ind] = host_new_image[ind];
	}

	free(host_new_image);
	
	cudaFree(device_penalty);
	cudaFree(device_new_image);
	
	cudaFree(device_current_image);
	cudaFree(device_update_factor);
	cudaFree(device_mask_image);
	cudaFree(device_sensitivity_image);
	
	cudaFree(device_function_image);
	cudaFree(device_gradient_image);
	cudaFree(device_hessian_image);
	cudaFree(device_th_up);
	cudaFree(device_th_lo);
}

void
image_update_CUDA::DoImageUpdatePL_CUDA2(ImageArray<float>& new_image, ImageArray<float>& update_factor, ImageArray<float>& current_image, ImageArray<float>&mask_image, ImageArray<float>& sensitivity_image, parameters_t& para, PET_geometry& g, int z_start, int z_end)
{
	size_t free_mem, total_mem;
	//int iter;
	int ind;
	int th_up, th_lo;
	float* host_new_image;

	//float host_penalty;
	float* device_penalty;
	float* device_new_image;
	float* device_current_image;
	float* device_update_factor;
	float* device_mask_image;
	float* device_sensitivity_image;

	size_t image_size;

	dim3 dimBlock(1024);
	dim3 dimGrid(256);

	dim3 dimBlock_update(512);
	dim3 dimGrid_update(256);

	image_size = g.NUM_XYZ*sizeof(float);
	printf("size of image is %d\n", image_size);

	host_new_image = (float*)malloc(image_size);

	cuMemGetInfo(&free_mem, &total_mem);
	printf("At this point, free memory = %d MB, Total memory = %d MB\n\n", ((int)free_mem) / 1048576, ((int)total_mem) / 1048576);

	cudaMalloc((void**)&device_penalty, sizeof(float));


	cudaMalloc((void**)&device_new_image, image_size);
	cudaMalloc((void**)&device_current_image, image_size);
	cudaMalloc((void**)&device_update_factor, image_size);
	cudaMalloc((void**)&device_mask_image, image_size);
	cudaMalloc((void**)&device_sensitivity_image, image_size);

	cudaMemcpy(device_new_image, new_image._image, image_size, cudaMemcpyHostToDevice);
	cudaMemcpy(device_current_image, current_image._image, image_size, cudaMemcpyHostToDevice);
	cudaMemcpy(device_update_factor, update_factor._image, image_size, cudaMemcpyHostToDevice);
	cudaMemcpy(device_mask_image, mask_image._image, image_size, cudaMemcpyHostToDevice);
	cudaMemcpy(device_sensitivity_image, sensitivity_image._image, image_size, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	th_up = 0.5f;
	th_lo = 0.5f;


	printf("...Now begins Trust Region Newton's update\n");


	TrustRegionNewtonMethodImageUpdate_CUDA2 << <dimGrid_update, dimBlock_update >> >(device_new_image, device_current_image, device_update_factor, device_mask_image, device_sensitivity_image, para, g, z_start, z_end, th_up, th_lo);
	cudaDeviceSynchronize();

	printf("...End of Trust Region Newton's update\n");

	cudaMemcpy(host_new_image, device_new_image, image_size, cudaMemcpyDeviceToHost);
	for (ind = (z_start - 1)*g.NUM_XY; ind <= z_end*g.NUM_XY; ind++){
		if (host_new_image[ind] > SMALLEST_ALLOWED){
			//printf("%f ", host_new_image[ind]);
		}
		new_image[ind] = host_new_image[ind];
	}

	

	free(host_new_image);

	cudaFree(device_penalty);
	cudaFree(device_new_image);
	cudaFree(device_current_image);
	cudaFree(device_update_factor);
	cudaFree(device_mask_image);
	cudaFree(device_sensitivity_image);


}
