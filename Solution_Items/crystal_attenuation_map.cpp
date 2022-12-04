
#include "global.h"
#include "config.h"
#include "ImageArray.h"
#include "PET_geometry.h"

class crystal_attenuation_map{

public:
	static void generate_crystal_attenuation_map(PET_geometry geometry, ImageArray<float> &image, std::vector<int> crystal_list);
private:
	static void write_cubic_crystal_to_image(box crystal, ImageArray<float> &image);
};




void
crystal_attenuation_map::generate_crystal_attenuation_map(PET_geometry geometry, ImageArray<float> &image, std::vector<int> crystal_list){

	int total_number_crystals;
	int i;


	total_number_crystals = crystal_list.size();
	std::cout << "Generating attenuation map for " << total_number_crystals << " crystals." << std::endl;


	for (i = 0; i < total_number_crystals; i++){

		int crystal_index;
		box crystal;

		crystal_index = crystal_list.at(i);
		crystal = geometry.detector_crystal_list.at(crystal_index).geometry;








	}

}

void
crystal_attenuation_map::write_cubic_crystal_to_image(box crystal, ImageArray<float> &image){

}


