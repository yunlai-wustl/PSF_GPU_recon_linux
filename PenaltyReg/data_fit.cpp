#include "data_fit.h"

void
Data_fit::ComputeNewtonMethodValues(float image_val, float update_factor_val, float sensitivity, float mu_free, float& function_val, float& first_deriv, float& second_deriv)
{

	float last_image_scaled = image_val*update_factor_val;

	function_val = sensitivity*mu_free - last_image_scaled*log(mu_free);

	first_deriv = mu_free - last_image_scaled / mu_free;

	second_deriv = last_image_scaled / (mu_free*mu_free);
}
