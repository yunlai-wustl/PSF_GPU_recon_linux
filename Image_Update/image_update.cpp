#include "image_update.h"

void
image_update::ImageUpdateML(ImageArray<float>& new_image, ImageArray<float>& update_factor, ImageArray<float>& current_image, ImageArray<float>&mask_image, ImageArray<float>& sensitivity_image, PET_geometry& g)
{
	// update all slices
	DoImageUpdateML(new_image, update_factor, current_image, mask_image, sensitivity_image, g, 1, g.NUM_Z);
}

void
image_update::ImageUpdatePL(ImageArray<float>& new_image, ImageArray<float>& update_factor, ImageArray<float>& current_image, ImageArray<float>&mask_image, ImageArray<float>& sensitivity_image, Penalty& PenaltyTerm, PET_geometry& g)
{
	// update first and last slices with ML to avoid calculating the penalty on slices reconstructed with missing data

	if (g.NUM_Z > 2 * NUM_ML_SLICES)
	{
		// update first slices
		DoImageUpdateML(new_image, update_factor, current_image, mask_image, sensitivity_image, g, 1, NUM_ML_SLICES);

		// update central slices
		DoImageUpdatePL(new_image, update_factor, current_image, mask_image, sensitivity_image, PenaltyTerm, g, NUM_ML_SLICES + 1, g.NUM_Z - NUM_ML_SLICES);

		// update last slices
		DoImageUpdateML(new_image, update_factor, current_image, mask_image, sensitivity_image, g, g.NUM_Z - NUM_ML_SLICES + 1, g.NUM_Z);
	}
	else
	{
		// not enough slices to perform PL, so just use ML for all slices
		DoImageUpdateML(new_image, update_factor, current_image, mask_image, sensitivity_image, g, 1, g.NUM_Z);
	}
}

void
image_update::DoImageUpdateML(ImageArray<float>& new_image, ImageArray<float>& update_factor, ImageArray<float>& current_image, ImageArray<float>&mask_image, ImageArray<float>& sensitivity_image, PET_geometry& g, int z_start, int z_end)
{
	int z_ind;
	int xy_ind;
	int xyz_ind, voxel_base;
	
	for (z_ind = z_start; z_ind <= z_end; z_ind++){

		voxel_base = (z_ind - 1)*g.NUM_XY;
		for (xy_ind = 0; xy_ind < g.NUM_XY; xy_ind++){

			xyz_ind = voxel_base + xy_ind;

			if (mask_image[xyz_ind] > 0.5f){
				new_image[xyz_ind] = current_image[xyz_ind] * update_factor[xyz_ind] / sensitivity_image[xyz_ind];
			}
			else{
				new_image[xyz_ind] = 0.0f;
			}
		}
	}
}

void
image_update::DoImageUpdatePL(ImageArray<float>& new_image, ImageArray<float>& update_factor, ImageArray<float>& current_image, ImageArray<float>&mask_image, ImageArray<float>& sensitivity_image, Penalty& PenaltyTerm, PET_geometry& g, int z_start, int z_end)
{
#pragma omp parallel for schedule(dynamic)
	int z_ind;
	int xy_ind;
	int xyz_ind, voxel_base;
	int newtons_iter;
	float function_data, first_deriv_data, second_deriv_data;
	float function_pen, first_deriv_pen, second_deriv_pen;
	
	float obj_function, obj_gradient, obj_heissian;
	float obj_function_2;
	float delta_raw, delta;
	float th;
	float th_upper, th_lower;

	float obj_function_quadratic_approximate;
	float rsm_new_estimate, new_estimate;
	float obj_approximate_diff;
	float obj_diff;
	float r_val;


	for (z_ind = z_start; z_ind <= z_end; z_ind++)
	{

		printf("z: %d\n", z_ind);

		voxel_base = (z_ind - 1)*g.NUM_XY;
		for (xy_ind = 0; xy_ind < g.NUM_XY; xy_ind++)
		{
			
			xyz_ind = voxel_base + xy_ind;

			if (mask_image[xyz_ind] > 0.5f){
				// Do Trust Region Newton's update


				// initialize free variable in Newton's method to current image value
				new_estimate = current_image[xyz_ind]*update_factor[xyz_ind]/sensitivity_image[xyz_ind];

				// run Newton's method on the current voxel
				th = 0.1;
				th_lower = -0.1;
				th_upper = 0.1;
				for (newtons_iter = 0; newtons_iter < MAX_NEWTONS_ITER; newtons_iter++) // very few of these iterations should hopefully be needed
				{
					// calculate derivatives for the data term
					
					Data_fit::ComputeNewtonMethodValues(current_image[xyz_ind], update_factor[xyz_ind], sensitivity_image[xyz_ind], new_estimate, function_data, first_deriv_data, second_deriv_data);

					// calculate derivatives for the penalty term
					
					PenaltyTerm.ComputeNewtonMethodValues(current_image, new_estimate, xyz_ind, function_pen, first_deriv_pen, second_deriv_pen);

					// perform Newton's update; break out of loop if update is less than 1 HU

					obj_function = function_data + function_pen;
					obj_gradient = first_deriv_data + first_deriv_pen;
					obj_heissian = second_deriv_data + second_deriv_pen;

					delta_raw = -obj_gradient/obj_heissian;

					if (delta_raw < -th){
						delta = -th;
					}
					else if (delta_raw > th){
						delta = th;
					}
					else{
						delta = delta_raw;
					}

					obj_function_quadratic_approximate = obj_function + obj_gradient*delta + 0.5*obj_heissian*delta*delta;
					//evaluate the obj_function at new estimate and compute r
					rsm_new_estimate = new_estimate + delta;

					obj_function_2 = -current_image[xyz_ind] * update_factor[xyz_ind] * log(rsm_new_estimate) + sensitivity_image[xyz_ind] * rsm_new_estimate + PenaltyTerm.ComputePenaltyFunctionVal(current_image, rsm_new_estimate, xyz_ind);

					//compute r
					obj_diff = obj_function - obj_function_2;
					obj_approximate_diff = obj_function - obj_function_quadratic_approximate;
					r_val = obj_diff / obj_approximate_diff;


					//if r_vals>0.75, then the quadratic approximates the function well.
					//and if the delta is within the restricted region, then the h_val is good enough
					if (r_val > 0.75 && delta_raw >= -th && delta_raw <= th){
						th = th;
					}
					//if the delta is at the boundary of the region, then we need to
					//increase the region
					if (r_val > 0.75 && (delta_raw < -th || delta_raw > th)){
						th = 2*th;
					}
					//if r_vals<0.25, bad approximation, the curvature is too flat, we
					//need to use a smaller restricted region
					if (r_val < 0.25){
						th = 0.25 * fabs(delta);
					}
					//else, h remains the same

					//if r_vals>0, rsm_new_estimate accepted.
					if (r_val > 0){
						new_estimate = rsm_new_estimate;
					}
					
				}
				new_image[xyz_ind] = new_estimate;
				
			}
			else{
				//0
				new_image[xyz_ind] = 0.0f;
			}




		}
	}
}
