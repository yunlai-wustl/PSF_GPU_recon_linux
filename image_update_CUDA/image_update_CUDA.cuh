#include "../PenaltyReg/penalty.h"
#include "../PenaltyReg/data_fit.h"
#include "../Solution_Items/ImageArray.h"
#include "../Solution_Items/PET_geometry.h"
#include "../Solution_Items/global.h"

#define NUM_ML_SLICES 0
#define MAX_NEWTONS_ITER 10
#define IMAGE_SMALLEST_ALLOWED 1e-8

class image_update_CUDA {

public:

	static void ImageUpdateML_CUDA(ImageArray<float>& new_image, ImageArray<float>& update_factor, ImageArray<float>& current_image, ImageArray<float>&mask_image, ImageArray<float>& sensitivity_image, PET_geometry& g);
	static void ImageUpdatePL_CUDA(ImageArray<float>& new_image, ImageArray<float>& update_factor, ImageArray<float>& current_image, ImageArray<float>&mask_image, ImageArray<float>& sensitivity_image, parameters_t& para, PET_geometry& g);

private:

	static void DoImageUpdateML_CUDA(ImageArray<float>& new_image, ImageArray<float>& update_factor, ImageArray<float>& current_image, ImageArray<float>&mask_image, ImageArray<float>& sensitivity_image, PET_geometry& g, int z_start, int z_end);
	static void DoImageUpdatePL_CUDA(ImageArray<float>& new_image, ImageArray<float>& update_factor, ImageArray<float>& current_image, ImageArray<float>&mask_image, ImageArray<float>& sensitivity_image, parameters_t& para, PET_geometry& g, int z_start, int z_end);
	static void DoImageUpdatePL_CUDA2(ImageArray<float>& new_image, ImageArray<float>& update_factor, ImageArray<float>& current_image, ImageArray<float>&mask_image, ImageArray<float>& sensitivity_image, parameters_t& para, PET_geometry& g, int z_start, int z_end);

};
