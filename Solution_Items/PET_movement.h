#ifndef PET_MOVEMENT_H
#define PET_MOVEMENT_H
#define _USE_MATH_DEFINES

#include "global.h"
#include "../numeric/mat3.h"
#include "../numeric/vec3.h"
#include "time_period.h"

typedef struct{
	TVec3<float> XYZ;
	TVec3<float> ABC;
}euler_transform;

typedef struct{
	TVec3<float> translate;
	TMat3<float> rotation;
}transform_matrix;


class PET_movement{
public:
	PET_movement();
	PET_movement(const std::string config_file_name);
	~PET_movement();

	int _add_one_global_adjustment(TVec3<float> translation, TVec3<float> rotation);
	int get_time_step();
	int get_dist_step();
	int get_position_index_by_time(float time);
	transform_matrix get_transform_matrix(int index);
	transform_matrix get_global_adjustment_transform_matrix(int index);
	time_period get_time_period(int index);


private:
	std::string movement_filename;
	int num_positions;
	int num_insert;

	std::vector<euler_transform> list_of_euler_transform;
	std::vector<transform_matrix> list_of_transform_matrix;
	std::vector<time_period> list_of_time_period;

	std::vector<euler_transform> list_of_global_adjustment_euler_tranform;
	std::vector<transform_matrix> list_of_global_adjustment_transform_matrix;

	int _read_movement_file(const std::string filename);
	void _add_global_adjustment(float *vec1, float *vec2);
	void _add_movement_position(float *vec1, float *vec2, float* tp);
	void _show_movement_information();

	TMat3<float> _euler_angle_to_rotation_matrix(TVec3<float>);
	int _compute_transform_matrix_from_euler_angles(std::vector<euler_transform> &eul, std::vector<transform_matrix> &matrix_transform);
};

#endif
