#define MAX_NUM_CRYSTALS 600000

#include "PET_geometry.h"

int PET_geometry::get_num_detector_crystals(){
	return (int)detector_crystal_list.size();
}

bool PET_geometry::is_Coincidence_Pair_valid(int id_1, int id_2){
	return true;
}

PET_geometry::PET_geometry(){
}

PET_geometry::PET_geometry(const std::string config_file_name, const std::string detectors_file_name){
	Config config(config_file_name);
	config.GetValue<int>("NUMX", NUM_X);
	config.GetValue<int>("NUMY", NUM_Y);
	config.GetValue<int>("NUMZ", NUM_Z);
	config.GetValue<float>("XSAMP", X_SAMP);
	config.GetValue<float>("YSAMP", Y_SAMP);
	config.GetValue<float>("ZSAMP", Z_SAMP);
	config.GetValue<float>("XOFFSET", X_OFFSET);
	config.GetValue<float>("YOFFSET", Y_OFFSET);
	config.GetValue<float>("ZOFFSET", Z_OFFSET);
	config.GetValue<float>("TOF resolution", TOF_res);
	config.GetValue<float>("Reconstruction Radius", R_ACTIVITY);

	NUM_XY = NUM_X*NUM_Y;
	NUM_XYZ = NUM_XY*NUM_Z;
	INV_X_SAMP = 1 / X_SAMP;
	INV_Y_SAMP = 1 / Y_SAMP;
	INV_Z_SAMP = 1 / Z_SAMP;

	//read the detectors file into detector_crystal_list
	detector_crystal_list.reserve(MAX_NUM_CRYSTALS);
	read_detectors_file(detectors_file_name);
	num_detectors = (int)detector_crystal_list.size();
	printf("number of crystals in geometry file: %d.\n", num_detectors);

	detector_crystal_list.shrink_to_fit();
}

PET_geometry::~PET_geometry(){

}

int
PET_geometry::read_detectors_file(const std::string filename){
	FILE *geometry_file;
	int block;
	int i;
	int total_num_crystals = 0;
	float vec[3];
	geometry_file = fopen(filename.c_str(), "r");
	Detector_crystal current_detector;
	for (block = 0; block < MAX_NUM_CRYSTALS && !feof(geometry_file); block++) {
		// read in center point 
		if (fscanf(geometry_file, "%f %f %f ", &current_detector.geometry.center[0], &current_detector.geometry.center[1], &current_detector.geometry.center[2]) == EOF) {
			break;
		}
		// read in normal vectors
		for (i = 0; i<3; i++){
			fscanf(geometry_file, "%f", &current_detector.geometry.normal_0[i]);
		}
		for (i = 0; i<3; i++){
			fscanf(geometry_file, "%f", &current_detector.geometry.normal_1[i]);
		}
		for (i = 0; i<3; i++){
			fscanf(geometry_file, "%f", &current_detector.geometry.normal_2[i]);
		}
		/*
		for(d=0; d<3; d++){// dimensions
		for(i=0; i<3; i++){// norm_x,norm_y,norm_z;
		fscanf(geometry_file,"%f",&blocks[block].geometry.normal[i][d]);
		}
		}*/
		// repeat elements
		fscanf(geometry_file, "%f %f %f", &current_detector.geometry.dimension[0], &current_detector.geometry.dimension[1], &current_detector.geometry.dimension[2]);
		fscanf(geometry_file, "%d %d %d", &current_detector.layer[0], &current_detector.layer[1], &current_detector.layer[2]);
		//fscanf(geometry_file,"%d %d %d",&blocks[block].sub_crystal_cut[0], &blocks[block].sub_crystal_cut[1], &blocks[block].sub_crystal_cut[2]);
		
		detector_crystal_list.push_back(current_detector);
		total_num_crystals++;
	}

	for (block = 0; block < NUM_SCANNER_CRYSTALS; block++){
		vec[0] = (detector_crystal_list.at(block).geometry.normal_1[0]);
		vec[1] = (detector_crystal_list.at(block).geometry.normal_1[1]);
		vec[2] = (detector_crystal_list.at(block).geometry.normal_1[2]);

		shift_crystal_center(detector_crystal_list.at(block).geometry.center, 0.0, vec);//3.74
	}

	/*for (block = 17600; block < 40368; block++){
		vec[0] =  -1*(detector_crystal_list.at(block).geometry.normal_2[0]);
		vec[1] =  -1*(detector_crystal_list.at(block).geometry.normal_2[1]);
		vec[2] =  -1*(detector_crystal_list.at(block).geometry.normal_2[2]);

		shift_crystal_center(detector_crystal_list.at(block).geometry.center, 0.37, vec);
	}

	for (block = 52536; block < 84984; block++){
		vec[0] = -1 * (detector_crystal_list.at(block).geometry.normal_2[0]);
		vec[1] = -1 * (detector_crystal_list.at(block).geometry.normal_2[1]);
		vec[2] = -1 * (detector_crystal_list.at(block).geometry.normal_2[2]);

		shift_crystal_center(detector_crystal_list.at(block).geometry.center, 0.37, vec);
	}*/

	for (block = NUM_SCANNER_CRYSTALS; block < NUM_SCANNER_CRYSTALS+NUM_INSERT_CRYSTALS; block++){
		vec[0] =  (detector_crystal_list.at(block).geometry.normal_1[0]);
		vec[1] =  (detector_crystal_list.at(block).geometry.normal_1[1]);
		vec[2] =  (detector_crystal_list.at(block).geometry.normal_1[2]);

		shift_crystal_center(detector_crystal_list.at(block).geometry.center, 0.0, vec);//1.0
	}

	/*for (block = 93568; block < 126336; block++){
		vec[0] = -(detector_crystal_list.at(block).geometry.normal_1[0]);
		vec[1] = -(detector_crystal_list.at(block).geometry.normal_1[1]);
		vec[2] = -(detector_crystal_list.at(block).geometry.normal_1[2]);

		shift_crystal_center(detector_crystal_list.at(block).geometry.center, 1.0, vec);
	}*/

	/*for (block = 84984; block < 126336; block++){
		vec[0] = (detector_crystal_list.at(block).geometry.normal_2[0]);
		vec[1] =  (detector_crystal_list.at(block).geometry.normal_2[1]);
		vec[2] =  (detector_crystal_list.at(block).geometry.normal_2[2]);

		shift_crystal_center(detector_crystal_list.at(block).geometry.center, 0.37, vec);

	}*/
	

	fclose(geometry_file);
	return total_num_crystals;   // return number of blocks read
}

void
PET_geometry::shift_crystal_center(float* center, float distance, float* vector){

	center[0] = center[0] + distance*vector[0];
	center[1] = center[1] + distance*vector[1];
	center[2] = center[2] + distance*vector[2];

}

void
PET_geometry::apply_movement(PET_movement &movement, int crystal_start, int crystal_end, int position){
	int i;
	TVec3<float> vec;

	printf("apply movement of position %d \n", position);
	printf("rotation matrix is:\n");
	vec = movement.get_transform_matrix(position).rotation[0];
	printf("%f %f %f\n", vec[0], vec[1], vec[2]);
	vec = movement.get_transform_matrix(position).rotation[1];
	printf("%f %f %f\n", vec[0], vec[1], vec[2]);
	vec = movement.get_transform_matrix(position).rotation[2];
	printf("%f %f %f\n", vec[0], vec[1], vec[2]);


	for (i = crystal_start; i < crystal_end; i++){

		TVec3<float> center(detector_crystal_list[i].geometry.center[0], detector_crystal_list[i].geometry.center[1], detector_crystal_list[i].geometry.center[2]);

		TVec3<float> new_center = movement.get_transform_matrix(position).rotation*center;
		new_center += movement.get_transform_matrix(position).translate;

		detector_crystal_list[i].geometry.center[0] = new_center[0];
		detector_crystal_list[i].geometry.center[1] = new_center[1];
		detector_crystal_list[i].geometry.center[2] = new_center[2];

	}

}


void
PET_geometry::apply_global_adjustment(PET_movement &movement, int crystal_start, int crystal_end, int insert){
	int i;
	TVec3<float> vec;

	printf("apply global adjustment of insert %d \n", insert);
	printf("rotation matrix is:\n");
	vec = movement.get_global_adjustment_transform_matrix(insert).rotation[0];
	printf("%f %f %f\n", vec[0], vec[1], vec[2]);
	vec = movement.get_global_adjustment_transform_matrix(insert).rotation[1];
	printf("%f %f %f\n", vec[0], vec[1], vec[2]);
	vec = movement.get_global_adjustment_transform_matrix(insert).rotation[2];
	printf("%f %f %f\n", vec[0], vec[1], vec[2]);





	for (i = crystal_start; i < crystal_end; i++){

		TVec3<float> center(detector_crystal_list[i].geometry.center[0], detector_crystal_list[i].geometry.center[1], detector_crystal_list[i].geometry.center[2]);

		//first, rotate -90 degree along the x-axis to bring the panel to horizontal position
		TVec3<float> t1(0.0, 0.0, 0.0);
		TVec3<float> r1(0.0, 0.0, -90.0);
		PET_movement forward;
		forward._add_one_global_adjustment(t1, r1);
		TVec3<float> new_center = forward.get_global_adjustment_transform_matrix(0).rotation*center;
		//new_center += forward.get_global_adjustment_transform_matrix(0).translate;
		
		//now perform the real adjustment for insert
		new_center = movement.get_global_adjustment_transform_matrix(insert).rotation*new_center;
		new_center += movement.get_global_adjustment_transform_matrix(insert).translate;

		//now reverse the panel back to vertical position
		TVec3<float> t2(0.0, 0.0, 0.0);
		TVec3<float> r2(0.0, 0.0, 90.0);
		PET_movement backward;
		backward._add_one_global_adjustment(t2, r2);
		new_center = backward.get_global_adjustment_transform_matrix(0).rotation*new_center;
		//new_center += backward.get_global_adjustment_transform_matrix(0).translate;



		detector_crystal_list[i].geometry.center[0] = new_center[0];
		detector_crystal_list[i].geometry.center[1] = new_center[1];
		detector_crystal_list[i].geometry.center[2] = new_center[2];


	}

}


void
PET_geometry::print_crystal_centers(string filename){
	FILE *fid;
	fid = fopen(filename.c_str(), "a");

	int i;
	for (i = 0; i < num_detectors; i++){

		fprintf(fid, "%f %f %f\n", detector_crystal_list[i].geometry.center[0], detector_crystal_list[i].geometry.center[1], detector_crystal_list[i].geometry.center[2]);

	}
	fclose(fid);

}

void
PET_geometry::shift_all_crystal(float length){
	int n = detector_crystal_list.size();
	for (int i = 0; i < n; i++){
		detector_crystal_list[i].geometry.center[2] += length;
	}
}