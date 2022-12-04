#include "PET_movement.h"
#include "config.h"

#define MAX_NUM_MOVEMENT 10000

PET_movement::PET_movement(){

}

PET_movement::~PET_movement(){

}

PET_movement::PET_movement(const std::string config_file_name){

	Config config(config_file_name);
	config.GetValue<string>("Insert Movement file", movement_filename);
	
	_read_movement_file(movement_filename);

	list_of_global_adjustment_transform_matrix.reserve(num_insert);
	list_of_transform_matrix.reserve(num_positions);

	_compute_transform_matrix_from_euler_angles(list_of_global_adjustment_euler_tranform, list_of_global_adjustment_transform_matrix);
	_compute_transform_matrix_from_euler_angles(list_of_euler_transform, list_of_transform_matrix);

	//_show_movement_information();

}

int
PET_movement::_add_one_global_adjustment(TVec3<float> translation, TVec3<float> rotation){
	num_insert = 1;
	num_positions = 0;
	list_of_global_adjustment_transform_matrix.reserve(num_insert);

	euler_transform eul;
	eul.XYZ = translation;
	eul.ABC = rotation;
	list_of_global_adjustment_euler_tranform.push_back(eul);
	
	_compute_transform_matrix_from_euler_angles(list_of_global_adjustment_euler_tranform, list_of_global_adjustment_transform_matrix);

	return 0;
}

int
PET_movement::_read_movement_file(const std::string filename){


	FILE *movement_file;
	int npositions;
	int ninsert;
	int i;
	float XYZ[3];
	float ABC[3];
	float time[2];

	movement_file = fopen(filename.c_str(), "r");
	//read in number of insert
	fscanf(movement_file, "%d", &ninsert);
	//read in number of movement positions
	fscanf(movement_file, "%d", &npositions);

	npositions = MAX_NUM_MOVEMENT < npositions ? MAX_NUM_MOVEMENT : npositions;

	//read in global adjustment
	for (i = 0; i < ninsert && !feof(movement_file); i++) {
		fscanf(movement_file, "%f %f %f %f %f %f", &XYZ[0], &XYZ[1], &XYZ[2], &ABC[0], &ABC[1], &ABC[2]);
		_add_global_adjustment(XYZ, ABC);
	}
	
	for (i = 0; i < npositions && !feof(movement_file); i++) {
		fscanf(movement_file, "%f %f %f %f %f %f %f %f", &XYZ[0], &XYZ[1], &XYZ[2], &ABC[0], &ABC[1], &ABC[2], &time[0], &time[1]);
		_add_movement_position(XYZ, ABC, time);

	}
	num_insert = ninsert;
	num_positions = npositions;

	return 0;
}


void
PET_movement::_add_global_adjustment(float *vec1, float *vec2){
	euler_transform eul;
	eul.XYZ = TVec3<float>(vec1[0], vec1[1], vec1[2]);
	eul.ABC = TVec3<float>(vec2[0], vec2[1], vec2[2]);
	list_of_global_adjustment_euler_tranform.push_back(eul);
}



void
PET_movement::_add_movement_position(float *vec1, float *vec2, float* tp){
	euler_transform eul;
	eul.XYZ = TVec3<float>(vec1[0], vec1[1], vec1[2]);
	eul.ABC = TVec3<float>(vec2[0], vec2[1], vec2[2]);
	list_of_euler_transform.push_back(eul);

	time_period timeperiod;
	timeperiod.t_start = tp[0];
	timeperiod.t_end = tp[1];
	list_of_time_period.push_back(timeperiod);
}


void
PET_movement::_show_movement_information(){
	int i;
	euler_transform eul;
	TVec3<float> vec;
	time_period tp;
	printf("\n Number of insert = %d\n", num_insert);
	printf("\n Number of movement positions = %d\n", num_positions);
	printf("\n\n");
	for (i = 0; i < num_insert; i++){
		eul = list_of_global_adjustment_euler_tranform.at(i);
		printf("\n  Global adjustment for insert %d = %f %f %f %f %f %f\n", i+1, eul.XYZ[0], eul.XYZ[1], eul.XYZ[2], eul.ABC[0], eul.ABC[1], eul.ABC[2]);
		printf("rotation matrix is:\n");
		vec = get_global_adjustment_transform_matrix(i).rotation[0];
		printf("%f %f %f\n", vec[0], vec[1], vec[2]);
		vec = get_global_adjustment_transform_matrix(i).rotation[1];
		printf("%f %f %f\n", vec[0], vec[1], vec[2]);
		vec = get_global_adjustment_transform_matrix(i).rotation[2];
		printf("%f %f %f\n", vec[0], vec[1], vec[2]);
		printf("\n\n");

	}
	printf("\n\n");
	for (i = 0; i < num_positions; i++){
		eul = list_of_euler_transform.at(i);
		tp = list_of_time_period.at(i);
		printf("\n  Position %d = %f %f %f %f %f %f %f %f\n", i+1, eul.XYZ[0], eul.XYZ[1], eul.XYZ[2], eul.ABC[0], eul.ABC[1], eul.ABC[2], tp.t_start, tp.t_end);
		printf("rotation matrix is:\n");
		vec = get_transform_matrix(i).rotation[0];
		printf("%f %f %f\n", vec[0], vec[1], vec[2]);
		vec = get_transform_matrix(i).rotation[1];
		printf("%f %f %f\n", vec[0], vec[1], vec[2]);
		vec = get_transform_matrix(i).rotation[2];
		printf("%f %f %f\n", vec[0], vec[1], vec[2]);
		printf("\n\n");
	}

}

int
PET_movement::_compute_transform_matrix_from_euler_angles(std::vector<euler_transform> &eul, std::vector<transform_matrix> &matrix_transform){

	int i;

	for (i = 0; i < eul.size(); i++){


		matrix_transform[i].translate = eul[i].XYZ;


		matrix_transform[i].rotation = _euler_angle_to_rotation_matrix(eul[i].ABC);

	}
	return 0;

}

TMat3<float>
PET_movement::_euler_angle_to_rotation_matrix(TVec3<float> euler_ABC){

	//Rz
	float A = euler_ABC[0] / 180.0 * M_PI;
	TVec3<float> row11(cos(A), -sin(A), 0);
	TVec3<float> row12(sin(A), cos(A), 0);
	TVec3<float> row13(0, 0, 1);
	TMat3<float> Rz(row11, row12, row13);

	//Ry
	float B = euler_ABC[1] / 180.0 * M_PI;
	TVec3<float> row21(cos(B), 0, sin(B));
	TVec3<float> row22(0, 1, 0);
	TVec3<float> row23(-sin(B), 0, cos(B));
	TMat3<float> Ry(row21, row22, row23);

	//Rx
	float C = euler_ABC[2] / 180.0 * M_PI;
	TVec3<float> row31(1, 0, 0);
	TVec3<float> row32(0, cos(C), -sin(C));
	TVec3<float> row33(0, sin(C), cos(C));
	TMat3<float> Rx(row31, row32, row33);

	return Rz*Ry*Rx;
}



int
PET_movement::get_dist_step(){
	return num_positions;
}


int
PET_movement::get_time_step(){
	return num_insert;
}

int
PET_movement::get_position_index_by_time(float t){
	//performe binary search on list_of_time_period to find the index of the time slot of the given time
	
	int index;

	int min = 0;
	int max = list_of_time_period.size() - 1;
	
	while (1){

		//break condition
		if (list_of_time_period[min] > t || list_of_time_period[max] < t){
			//no position contains time t
			break;
		}

		//test position at median index
		index = (max + min) / 2;
		if (list_of_time_period[index] == t){
			return index;
		}
		else if (list_of_time_period[index] < t){
			if (min == index)
				min++;
			else
				min = index;
		}
		else{
			if (max == index)
				max--;
			else
				max = index;
		}

	}
	return -1;
}

transform_matrix
PET_movement::get_transform_matrix(int index){
	return list_of_transform_matrix[index];
}

transform_matrix
PET_movement::get_global_adjustment_transform_matrix(int index){
	return list_of_global_adjustment_transform_matrix[index];
}

time_period
PET_movement::get_time_period(int index){
	return list_of_time_period[index];
}
