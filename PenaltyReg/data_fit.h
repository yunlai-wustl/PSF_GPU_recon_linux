#pragma once

#include "../Solution_Items/ImageArray.h"
#include "../Solution_Items/PET_geometry.h"
#include "../Solution_Items/global.h"

namespace Data_fit {

	void ComputeNewtonMethodValues(float image_val, float update_factor_val, float sensitivity, float mu_free, float& function_val, float& first_deriv, float& second_deriv);

};
