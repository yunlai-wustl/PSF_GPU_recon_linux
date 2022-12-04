#pragma once

#include "../PenaltyReg/penalty.h"
#include "../PenaltyReg/data_fit.h"
#include "../Solution_Items/ImageArray.h"
#include "../Solution_Items/PET_geometry.h"
#include "../Solution_Items/global.h"

#define NUM_ML_SLICES 15
#define MAX_NEWTONS_ITER 10

class image_update {

public:

	static void ImageUpdateML(ImageArray<float>& new_image, ImageArray<float>& update_factor, ImageArray<float>& current_image, ImageArray<float>&mask_image, ImageArray<float>& sensitivity_image, PET_geometry& g);
	static void ImageUpdatePL(ImageArray<float>& new_image, ImageArray<float>& update_factor, ImageArray<float>& current_image, ImageArray<float>&mask_image, ImageArray<float>& sensitivity_image, Penalty& PenaltyTerm, PET_geometry& g);

private:

	static void DoImageUpdateML(ImageArray<float>& new_image, ImageArray<float>& update_factor, ImageArray<float>& current_image, ImageArray<float>&mask_image, ImageArray<float>& sensitivity_image, PET_geometry& g, int z_start, int z_end);
	static void DoImageUpdatePL(ImageArray<float>& new_image, ImageArray<float>& update_factor, ImageArray<float>& current_image, ImageArray<float>&mask_image, ImageArray<float>& sensitivity_image, Penalty& PenaltyTerm, PET_geometry& g, int z_start, int z_end);

};
