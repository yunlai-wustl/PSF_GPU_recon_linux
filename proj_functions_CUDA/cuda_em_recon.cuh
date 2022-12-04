#ifndef CUDA_EM_RECON_CUH
#define CUDA_EM_RECON_CUH


#include "cuda_common_header.cuh"

#include "../Solution_Items/PET_data.h"
#include "../Solution_Items/PET_DATA_scatter.h"
#include "../Solution_Items/PET_geometry.h"
#include "../Solution_Items/ImageArray.h"
#include "../numeric/mat3.h"
#include "../numeric/vec3.h"


class cuda_em_recon {
public:
	cuda_em_recon(parameters_t p, string output_prefix, string output_path);
	~cuda_em_recon();
		
	void Setup_data(PET_geometry& detector, PET_movement& movement, PET_data& data, int data_index_start, int data_index_end, PET_coincidence_type pet_coinc_type, int device_ID);
	void Setup_data_min(PET_geometry& detector, PET_movement& movement, PET_data& data, int data_index_start, int data_index_end, PET_coincidence_type pet_coinc_type, int device_ID);

	void Setup_data_scatter(PET_geometry& detector, PET_movement& movement, PET_data_scatter& data, int data_index_start, int data_index_end, PET_coincidence_type pet_coinc_type, int device_ID);
	void Setup_parameters(parameters_t p);
	void Setup_data_norm(PET_geometry& detector, PET_movement& movement, PET_data& data, int data_index_start, int data_index_end, PET_coincidence_type pet_coinc_type, int device_ID);
	void Setup_image();
	void Release();
	void Release_image();
	void Release_data();
	void Release_data_scatter();


	void ComputeUpdateFactor(PET_geometry& detector, PET_movement& movement, PET_data& data, ImageArray<float>& Image, ImageArray<float>& UpdateFactor, ImageArray<float>& Attenuation_Image, float &nloglikelihood);
	void ComputeUpdateFactorScatter(PET_geometry& detector, PET_movement& movement, PET_data_scatter& data, ImageArray<float>& Image, ImageArray<float>& UpdateFactor, ImageArray<float>& Attenuation_Image, float &nloglikelihood);
	void ComputeUpdateFactorSS(PET_geometry& detector, PET_movement& movement, PET_data& data, ImageArray<float>& Image, ImageArray<float>& UpdateFactor);
	void ComputeUpdateFactorIS(PET_geometry& detector, PET_movement& movement, PET_data& data, ImageArray<float>& Image, ImageArray<float>& UpdateFactor);

	void Backward_Projection(PET_geometry& detector, PET_data& data);
	void Backward_Projection_Attenuation(PET_geometry& detector, PET_movement& movement, PET_data& data, ImageArray<float>& Image, ImageArray<float>& Norm_Image);
	void Backward_Projection_Attenuation_norm(PET_geometry& detector, PET_movement& movement, PET_data& data, ImageArray<float>& Image, ImageArray<float>& Norm_Image, ImageArray<float>& Emission_Image);


	
private:
	int _num_gpu_start;
	int _num_gpu_end;
	//int _num_gpus;

	string _output_path;
	string _output_filename_prefix;

	int _ii[MAX_GPU];
	int _is[MAX_GPU];
	int _ss[MAX_GPU];
	int _event_count_x[MAX_GPU];
	int _event_count_y[MAX_GPU];
	

	TOF_MODE _TOF_mode;
	int export_likelihood_flag;
	int atten_flag_fp;
	int atten_flag_bp;
	//pointers to the POD object that stores list mode LORs on the unified virtual memory
	//needs to be allocated using new()
	LST_LORs_uvm *_events_x_dominant_uvm[MAX_GPU];
	LST_LORs_uvm *_events_y_dominant_uvm[MAX_GPU];
	LST_LORs_uvm *_events_z_dominant_uvm[MAX_GPU];


	//pointers to the POD object that stores list mode LORs on the traditional GPU global memory
	//needs to be allocated using cudaMalloc()
	LST_LORs *_events_x_dominant_global_mem[MAX_GPU];
	LST_LORs *_events_y_dominant_global_mem[MAX_GPU];
	LST_LORs *_events_z_dominant_global_mem[MAX_GPU];


	//pointers to the POD object that stores list mode LORs on the traditional GPU global memory
	//needs to be allocated using cudaMalloc()
	LST_LORs_scatter *_events_x_dominant_global_mem_scatter[MAX_GPU];
	LST_LORs_scatter *_events_y_dominant_global_mem_scatter[MAX_GPU];
	LST_LORs_scatter *_events_z_dominant_global_mem_scatter[MAX_GPU];

	//pointers to device memory, stores the images
	//needs to be allocated by cudaMalloc()
	float* _atten_image_device[MAX_GPU];
	float* _current_image_device[MAX_GPU];
	float* _update_factor_device[MAX_GPU];
	float* _current2_image_device[MAX_GPU];

	PARAMETERS_IN_DEVICE_t* _parameters_device[MAX_GPU];

	float* _nloglikelihood_value[MAX_GPU];

	//pointers to device memory, stores the list mode forward projection value
	//needs to be allocated by cudaMalloc()
	float* _fp_value_x_device[MAX_GPU];
	float* _fp_value_y_device[MAX_GPU];
	float* _fp_value_z_device[MAX_GPU];

	float* _fp_value2_x_device[MAX_GPU];
	float* _fp_value2_y_device[MAX_GPU];
	float* _fp_value2_z_device[MAX_GPU];

	PARAMETERS_IN_DEVICE_t parameters_host;
	
	void _Initialize_host_parameters(parameters_t p);
	void _Initialize_device_parameters();
	void _Mem_allocation_for_LST_events_uvm(PET_geometry& detector, PET_data& data, LST_LORs_uvm*& device_events_x_uvm, LST_LORs_uvm*& device_events_y_uvm,  int& event_count_x_dominant, int& event_count_y_dominant);
	void _Mem_allocation_for_LST_events_memcopy(PET_geometry& detector, PET_movement& movement, PET_data& data, int start_index, int end_index, PET_coincidence_type pet_coinc_type, LST_LORs*& device_events_x_global_mem, LST_LORs*& device_events_y_global_mem,  int& event_count_x_dominant, int& event_count_y_dominant,  int device_ID);
	void _Mem_release_for_LST_events_uvm(LST_LORs_uvm*& device_events_x_uvm, LST_LORs_uvm*& device_events_y_uvm);
	void _Mem_release_for_LST_events_memcopy(LST_LORs* device_events_x, LST_LORs* device_events_y);
	void _Mem_allocation_for_LST_events_memcopy_norm(PET_geometry& detector, PET_movement& movement, PET_data& data, int start_index, int end_index, PET_coincidence_type pet_coinc_type, LST_LORs*& device_events_x_global_mem, LST_LORs*& device_events_y_global_mem, int& event_count_x_dominant, int& event_count_y_dominant, int device_ID);
	void _Mem_allocation_for_LST_events_memcopy_scatter(PET_geometry& detector, PET_movement& movement, PET_data_scatter& data, int start_index, int end_index, PET_coincidence_type pet_coinc_type, LST_LORs_scatter*& device_events_x_global_mem, LST_LORs_scatter*& device_events_y_global_mem, int& event_count_x_dominant, int& event_count_y_dominant, int device_ID);
	void _Mem_allocation_for_min_LST_events_memcopy(PET_geometry& detector, PET_movement& movement, PET_data& data, int start_index, int end_index, PET_coincidence_type pet_coinc_type, LST_LORs*& device_events_x_global_mem, LST_LORs*& device_events_y_global_mem, int& event_count_x_dominant, int& event_count_y_dominant, int device_ID);

	void _Mem_release_for_LST_events_memcopy_scatter(LST_LORs_scatter* device_events_x, LST_LORs_scatter* device_events_y);

	void _Mem_allocation_for_images();
	void _Mem_allocation_for_fp_values(int device_ID);
	void _Mem_release_for_images();
	void _Mem_release_for_fp_values(int device_ID);
	void _Copy_images_to_GPU_memory(float** device, ImageArray<float>& host);

};


#endif
