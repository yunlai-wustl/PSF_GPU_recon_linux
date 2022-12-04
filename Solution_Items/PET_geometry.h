#ifndef PET_GEOMETRY_H
#define PET_GEOMETRY_H


#include "global.h"
#include "config.h"
#include "PET_movement.h"

class PET_geometry{
public:
	PET_geometry();
	PET_geometry(const std::string config_file_name, const std::string detectors_file_name);
	~PET_geometry();

	std::vector<Detector_crystal> detector_crystal_list;
	int num_detectors;

	int NUM_X;
	int NUM_Y;
	int NUM_Z;
	float X_SAMP;
	float Y_SAMP;
	float Z_SAMP;
	float X_OFFSET;
	float Y_OFFSET;
	float Z_OFFSET;
	float TOF_res;
	float R_ACTIVITY;
	
	int NUM_XY;
	int NUM_XYZ;
	float INV_X_SAMP;
	float INV_Y_SAMP;
	float INV_Z_SAMP;
	
	
	int decode_LOR_ID_1(int LOR_index);
	int decode_LOR_ID_2(int LOR_index);
	int get_num_detector_crystals();
	void apply_movement(PET_movement &movement, int crystal_start, int crystal_end, int position);
	void apply_global_adjustment(PET_movement &movement, int crystal_start, int crystal_end, int insert);
	void print_crystal_centers(string filename);

	void shift_all_crystal(float length);
private:

	int read_detectors_file(const std::string detectors_file_name);
	void shift_crystal_center(float* center, float distance, float* vector);
	bool is_LOR_valid(int LOR_index);
	bool is_Coincidence_Pair_valid(int id_1, int id_2);

};

#endif
