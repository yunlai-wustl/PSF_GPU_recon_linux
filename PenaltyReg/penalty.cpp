#include "penalty.h"

Penalty::Penalty(parameters_t global_parameters, ImageArray<float>& Mask_Image, bool is_penalty_3d, const PET_geometry& g){
	
	_g = g;

	// set up the valid voxel region based on the mask image

	int i;
	// initialize _VoxInsidePenalty to all false
	_VoxInsidePenalty.Setup(_g);
	_VoxInsidePenalty.SetValue(false);

	// fill in _VoxInsidePenalty mask based on xy_indices list, i.e., remove those xy indices that are on the xy image border to eliminate out-of-bounds accesses
	for (i=0; i<(int)Mask_Image.GetSize() ; i++)
	{


		if (Mask_Image._image[i]>0.5f) {
			
			_VoxInsidePenalty[i] = true;
		}
	}

	// initialize valid z range to eliminate out-of-bounds accesses during Penalty calculations
	_start_z = 2;
	_end_z = _g.NUM_Z - 1;




	// set up penalty strength
	_strength = global_parameters.prior_beta / global_parameters.num_OSEM_subsets;
	_delta = global_parameters.prior_delta;
	_inv_delta = 1 / _delta;
	_delta_sq = _delta*_delta;


	// set up 2D/3D penalty
	if (is_penalty_3d)
		_num_neighbors = 26;
	else
		_num_neighbors = 8;

	// 26 neighbors for a given voxel in 3D (3^3-1 = 26)
	// weights are based on inverse distance between neighboring voxel centers

	// === xy-plane orthogonal neighbor index offsets and weights ===

	_vox_offset[0] = 1; // x+1
	_vox_offset[1] = -1; // x-1
	_vox_offset[2] = _g.NUM_X; // y+1
	_vox_offset[3] = -_g.NUM_X; // y-1

	_weight[0] = _weight[1] = _g.INV_X_SAMP;
	_weight[2] = _weight[3] = _g.INV_Y_SAMP;

	// === xy-plane non-corner diagonal neighbor index offsets and weights ===

	_vox_offset[4] = _g.NUM_X + 1; // x+1, y+1
	_vox_offset[5] = _g.NUM_X - 1; // x-1, y+1
	_vox_offset[6] = -_g.NUM_X + 1; // x+1, y-1
	_vox_offset[7] = -_g.NUM_X - 1; // x-1, y-1

	_weight[4] = _weight[5] = _weight[6] = _weight[7] = _g.INV_X_SAMP / sqrt(2.0); //Assume we have same sampling distance for X and Y

	// === non-xy-plane orthogonal neighbor index offsets and weights ===

	_vox_offset[8] = _g.NUM_XY; // z+1
	_vox_offset[9] = -_g.NUM_XY; // z-1

	_weight[8] = _weight[9] = _g.INV_Z_SAMP;

	// === non-xy-plane non-corner diagonal neighbor index offsets and weights ===

	_vox_offset[10] = _g.NUM_XY + 1;   // x+1, z+1
	_vox_offset[11] = _g.NUM_XY - 1;   // x-1, z+1
	_vox_offset[12] = _g.NUM_XY + _g.NUM_X; // y+1, z+1
	_vox_offset[13] = _g.NUM_XY - _g.NUM_X; // y-1, z+1
	_vox_offset[14] = -_g.NUM_XY + 1;   // x+1, z-1
	_vox_offset[15] = -_g.NUM_XY - 1;   // x-1, z-1
	_vox_offset[16] = -_g.NUM_XY + _g.NUM_X; // y+1, z-1
	_vox_offset[17] = -_g.NUM_XY - _g.NUM_X; // y-1, z-1

	_weight[10] = _weight[11] = _weight[12] = _weight[13] = 1.0 / sqrt(_g.X_SAMP * _g.X_SAMP + _g.Z_SAMP * _g.Z_SAMP); //Assume we have same sampling distance for X and Y
	_weight[14] = _weight[15] = _weight[16] = _weight[17] = _weight[10];

	// === non-xy-plane corner diagonal neighbor index offsets and weights ===

	_vox_offset[18] = _g.NUM_XY + _g.NUM_X + 1; // x+1, y+1, z+1
	_vox_offset[19] = _g.NUM_XY + _g.NUM_X - 1; // x-1, y+1, z+1
	_vox_offset[20] = _g.NUM_XY - _g.NUM_X + 1; // x+1, y-1, z+1
	_vox_offset[21] = _g.NUM_XY - _g.NUM_X - 1; // x-1, y-1, z+1
	_vox_offset[22] = -_g.NUM_XY + _g.NUM_X + 1; // x+1, y+1, z-1
	_vox_offset[23] = -_g.NUM_XY + _g.NUM_X - 1; // x-1, y+1, z-1
	_vox_offset[24] = -_g.NUM_XY - _g.NUM_X + 1; // x+1, y-1, z-1
	_vox_offset[25] = -_g.NUM_XY - _g.NUM_X - 1; // x-1, y-1, z-1

	_weight[18] = _weight[19] = _weight[20] = _weight[21] = 1.0 / sqrt(_g.X_SAMP * _g.X_SAMP + _g.Y_SAMP * _g.Y_SAMP + _g.Z_SAMP * _g.Z_SAMP);
	_weight[22] = _weight[23] = _weight[24] = _weight[25] = _weight[18];

}





float
Penalty::ComputeFullPenalty(ImageArray<float>& Image)
{
	double penalty = 0.0f;
	int z_ind, xy_ind, base_xyz_ind;
	int i;
	float t_inv_delta;

	//we do NOT compute penalty for the 1st and last z-slice
	for (z_ind = _start_z; z_ind <= _end_z; z_ind++)
	{
		int slice_start_ind = (z_ind - 1) * _g.NUM_XY;
		for (xy_ind = 0; xy_ind < _g.NUM_XY; xy_ind++)
		{
			base_xyz_ind = slice_start_ind + xy_ind;

			if (_VoxInsidePenalty[base_xyz_ind])
			{
				for (i = 0; i < _num_neighbors; i++)
				{
					if (_VoxInsidePenalty[base_xyz_ind + _vox_offset[i]])
					{
						t_inv_delta = (Image[base_xyz_ind] - Image[base_xyz_ind + _vox_offset[i]]) * _inv_delta;
						penalty += _weight[i] * log(cosh(t_inv_delta));
					}
				}
			}
		}
	}

	return (_strength * _delta_sq * penalty);
}



void
Penalty::ComputeNewtonMethodValues(ImageArray<float>& Image, float mu_free, int xyz_ind, float& function_val, float& first_deriv, float& second_deriv)
{
	int z_ind;
	float bracketterm_temp, bracketterm;
	float temp;

	z_ind = xyz_ind / _g.NUM_XY + 1;


	function_val = 0.0f;
	first_deriv = 0.0f;
	second_deriv = 0.0f;
	if (_VoxInsidePenalty[xyz_ind] && (z_ind >= _start_z) && (z_ind <= _end_z))
	{
		
		bracketterm_temp = 2 * mu_free - Image[xyz_ind];

		for (int i = 0; i < _num_neighbors; i++)
		{
			// t_hat is the argument to the first and second derivative functions
			bracketterm = bracketterm_temp - Image[xyz_ind + _vox_offset[i]];

			// temp is common to first and second derivative calculations
			temp = bracketterm * _inv_delta;

			function_val += _weight[i] * log(cosh(temp));
			first_deriv += 2 * _weight[i] * tanh(temp);
			second_deriv += 4 * _weight[i] * cosh(temp);
		}

		function_val *= 0.5*_strength*_delta*_delta;
		first_deriv *= 0.5 * _strength * _delta;
		second_deriv *= 0.5 * _strength;
	}
}


float
Penalty::ComputePenaltyFunctionVal(ImageArray<float>& Image, float mu_free, int xyz_ind)
{
	int z_ind;
	float bracketterm_temp, bracketterm;
	float temp;



	float function_val = 0.0f;


	bracketterm_temp = 2 * mu_free - Image[xyz_ind];

	for (int i = 0; i < _num_neighbors; i++)
	{
		// t_hat is the argument to the first and second derivative functions
		bracketterm = bracketterm_temp - Image[xyz_ind + _vox_offset[i]];
		// temp is common to first and second derivative calculations
		temp = bracketterm * _inv_delta;
		function_val += _weight[i] * log(cosh(temp));
	}
	function_val *= 0.5*_strength*_delta*_delta;

	return function_val;
}




float
Penalty::FunctionVal(float t)
{
	float t_inv_delta;
	t_inv_delta = t * _inv_delta;
	
	return _delta_sq * log(cosh(t_inv_delta));
}


float
Penalty::FirstDeriv(float t)
{
	return _delta * tanh(t * _inv_delta);
}

