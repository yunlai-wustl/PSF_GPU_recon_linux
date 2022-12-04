#ifndef PENALTY_H
#define PENALTY_H

#include "../Solution_Items/ImageArray.h"
#include "../Solution_Items/PET_geometry.h"
#include "../Solution_Items/global.h"

class Penalty
{
public:
	
	float ComputeFullPenalty(ImageArray<float>& Image);
	
	void ComputeNewtonMethodValues(ImageArray<float>& Image, float mu_free, int xyz_ind, float& function_val, float& first_deriv, float& second_deriv);
	float ComputePenaltyFunctionVal(ImageArray<float>& Image, float mu_free, int xyz_ind);

	void FindMinMaxNeighborValues(ImageArray<float>& Image, int z_ind, int xy_ind, float& min_val, float& max_val);
	Penalty(parameters_t global_parameters, ImageArray<float>& Mask_Image, bool is_penalty_3d, const PET_geometry& g);

protected:
	float FunctionVal(float t);
	float FirstDeriv(float t);
	float SecondDerivAt0();

	void ComputeSurrogateParams(float x_j, float x_k, float u_min, float u_max, float& a, float& b);

	float _strength;
	int _num_neighbors;
	
	//the z index of an image stack starts from 1, not 0
	int _start_z;
	int _end_z;
	
	int _vox_offset[26];
	float _weight[26];

	ImageArray<bool> _VoxInsidePenalty; // array element is true if the voxel indexed by xy_ind should have its penalty actually calculated, false otherwise
	
	PET_geometry _g;


	float _delta;
	float _inv_delta;
	float _delta_sq;

};

#endif