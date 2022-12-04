//Generate sensitivity image
// Uses LOR based normalization
//Written by Suranjana Samanta

#define _CRT_SECURE_NO_DEPRECATE
#include <stdio.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include "../Solution_Items/global.h"
#include "../Solution_Items/command_line.h"
#include "../Solution_Items/ImageArray.h"
#include "../Solution_Items/config.h"
#include "../Solution_Items/PET_data.h"
#include "../Solution_Items/PET_geometry.h"
#include "../Solution_Items/PET_movement.h"
#include "../proj_functions_CUDA/cuda_em_recon.cuh"
#include "../image_update_CUDA/image_update_CUDA.cuh"

#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#include <math.h>

void Usage(char argv0[])
{
	string program_name = CommandLine::GetProgramName(argv0);
	printf("Usage: %s <config_file>\n", program_name.c_str());
}

int main(int argc, char *argv[]) {

	// path and filename
	string data_path, img_path, norm_path;

	string initial_image_filename;
	string output_image_filename_prefix;
	string geometry_filename;
	string config_filename;
	string emission_data_filename;
	string scattered_data_filename;
	string atten_image_filename;
	string sensitivity_image_filename;
	string emission_image_filename;
	string normalization_emission_image_filename;
	string normalization_emission_data_filename;
	string blank_emission_data_filename;
	string transmission_emission_data_filename;

	// image and backprojection arrays
	ImageArray<float> atten_image;
	ImageArray<float> emission_image;
	ImageArray<float> sensitivity_image;
	ImageArray<float> initial_image;
	ImageArray<float> current_image;
	ImageArray<float> mask_image;
	ImageArray<float> update_factor_image;
	ImageArray<float> sub_sensitivity_image;
	ImageArray<float> sub_sensitivity_image1;
	ImageArray<float> sensitivity_image1;
	ImageArray<float> new_image;

	int i;

	// data pointers 
	float time_start, time_stop;
	int time_gap;

	float time_start_ss, time_stop_ss;
	float time_start_is, time_stop_is;
	float time_start_ii, time_stop_ii;

	float fwhm_ss, fwhm_is, fwhm_ii;
	float time;

	// reconstruction parameters
	int num_cores;
	int start_num_gpu, end_num_gpu;
	int num_iter, start_iter;
	int OSEM_subsets = 1;
	int image_write_freq;
	//penalty
	float prior_beta = 0, prior_C1 = 1, prior_C2 = 1, prior_delta = 1;

	//voxel representation
	float spherical_voxel_ratio = 1;

	//CBM
	float CBM_distance_total = 0.0;
	int n_bed_positions = 1;
	float bed_movement_per_step = 0.0;

	// ============================= variables for list mode =============================
	int total_event_count;

	// flags
	int initial_image_flag = 0;
	int randoms_flag = 0;
	int scatter_flag = 0;
	int normalization_flag = 0;
	int TOF_flag = 0;
	int penalty_flag = 0;
	int compute_convergencecurves = 0;
	int CBM_flag = 0;
	TOF_MODE usingTOF;

	parameters_t global_parameters;

	// ============================= variables for input and output =============================
	stringstream sstm;


	// ============================= process command line arguments ============================= 
	if (argc != 2)
	{
		Usage(argv[0]);
		return 0;
	}

	// ============================= read in config file ============================= 
	config_filename = argv[1];
	Config config(config_filename);
	config.GetValue<string>("DAT_PATH", data_path);
	config.GetValue<string>("IMG_PATH", img_path);
	config.GetValue<string>("NORM_PATH", norm_path);

	config.GetValue<string>("Detector description file", geometry_filename);
	config.GetValue<string>("Emission data file", emission_data_filename);
	config.GetValue<string>("Scattered data file", scattered_data_filename, false);
	config.GetValue<string>("Initial image file", initial_image_filename, false);
	config.GetValue<string>("Attenuation image file", atten_image_filename, false);
	config.GetValue<string>("Sensitivity image file", sensitivity_image_filename);
	config.GetValue<string>("Normalization Emission image file", normalization_emission_image_filename);
	config.GetValue<string>("Normalization Emission data file", normalization_emission_data_filename);
	config.GetValue<string>("Output image prefix", output_image_filename_prefix);

	config.GetValue<int>("Number of cores", num_cores);
	config.GetValue<int>("Start number of GPU", start_num_gpu);
	config.GetValue<int>("End number of GPU", end_num_gpu);

	config.GetValue<int>("Number of iterations", num_iter);
	config.GetValue<int>("Starting iteration", start_iter);//start_iter =0 means starting from all ones image or user specified initial image
	config.GetValue<int>("Number of subsets", OSEM_subsets);
	config.GetValue<int>("Image write frequency", image_write_freq);

	config.GetValue<float>("Time start", time_start);
	config.GetValue<float>("Time stop", time_stop);
	config.GetValue<int>("Time gap", time_gap);

	config.GetValue<float>("Time start SS", time_start_ss);
	config.GetValue<float>("Time stop SS", time_stop_ss);

	config.GetValue<float>("Time start IS", time_start_is);
	config.GetValue<float>("Time stop IS", time_stop_is);

	config.GetValue<float>("Time start II", time_start_ii);
	config.GetValue<float>("Time stop II", time_stop_ii);


	config.GetValue<float>("FWHM SS", fwhm_ss);
	config.GetValue<float>("FWHM IS", fwhm_is);
	config.GetValue<float>("FWHM II", fwhm_ii);

	config.GetValue<float>("Time for Sensitivity Image", time);

	config.GetValue<int>("Reconstruction use initial image", initial_image_flag);
	config.GetValue<int>("Reconstruction use randoms", randoms_flag);
	config.GetValue<int>("Reconstruction use scatter", scatter_flag);
	config.GetValue<int>("Reconstruction use normalization", normalization_flag);
	config.GetValue<int>("Reconstruction use TOF", TOF_flag);
	config.GetValue<int>("Reconstruction use CBM", CBM_flag);

	config.GetValue<int>("Reconstruction using penalty", penalty_flag);
	config.GetValue<float>("Beta", prior_beta);
	config.GetValue<float>("Delta", prior_delta);
	config.GetValue<float>("C1", prior_C1);
	config.GetValue<float>("C2", prior_C2);

	config.GetValue<float>("Spherical Voxel Ratio", spherical_voxel_ratio);
	if (CBM_flag)
	{
		config.GetValue<float>("Bed moving distance", CBM_distance_total);
		config.GetValue<float>("Bed movement per step", bed_movement_per_step);
		n_bed_positions = (int)(CBM_distance_total / bed_movement_per_step) + 1;
	}

	printf("=== Parameter listing ===\n\n");

	if (penalty_flag) {
		printf("Using Penalized ML reconstruction with beta = %f and delta = %f\n", prior_beta, prior_delta);
	}
	else {
		printf("Using ML reconstruction\n");
	}
	printf("Spherical voxel filling ratio: %f\n", spherical_voxel_ratio);

	printf("Starting from iteration: %d\n", start_iter);
	printf("Number of iterations: %d\n", num_iter);
	printf("Number of subsets: %d\n", OSEM_subsets);
	printf("Initial image file: %s\n", (initial_image_filename == "") ? "none provided" : initial_image_filename.c_str());
	printf("Image write frequency: %d\n", image_write_freq);
	printf("Output image prefix: %s\n", output_image_filename_prefix.c_str());
	printf("Emission data file: %s\n", emission_data_filename.c_str());
	printf("Detector description file: %s\n", geometry_filename.c_str());

	printf("\n=== End of parameter listing ===\n\n");

	if (initial_image_flag == 1){
		if (initial_image_filename == ""){
			printf("Image reconstruction will use an initial image, starting from iteration %d. However, an initial image is not specified, please choose an initial image as the image for iteration %d.\n", start_iter, start_iter - 1);
			throw 1;
		}
		else{
			printf("Image reconstruction will use an initial image, starting from iteration %d. Using file %s as the image for iteration %d.\n", start_iter, initial_image_filename.c_str(), start_iter - 1);
		}
	}
	else{
		printf("Image reconstruction will use an ALL-ONEs image, starting from iteration %d.\n", start_iter);
	}


	// ============================= read in geometry file ============================= 
	PET_geometry geometry(config_filename, geometry_filename);
	global_parameters.NUM_X = geometry.NUM_X;
	global_parameters.NUM_Y = geometry.NUM_Y;
	global_parameters.NUM_Z = geometry.NUM_Z;
	global_parameters.X_SAMP = geometry.X_SAMP;
	global_parameters.Y_SAMP = geometry.Y_SAMP;
	global_parameters.Z_SAMP = geometry.Z_SAMP;
	global_parameters.X_OFFSET = geometry.X_OFFSET;
	global_parameters.Y_OFFSET = geometry.Y_OFFSET;
	global_parameters.Z_OFFSET = geometry.Z_OFFSET;
	global_parameters.TOF_on = TOF_flag;
	global_parameters.TOF_res = geometry.TOF_res;
	global_parameters.num_iter = num_iter;
	global_parameters.num_gpu_start = start_num_gpu;
	global_parameters.num_gpu_end = end_num_gpu;
	global_parameters.num_OSEM_subsets = OSEM_subsets;
	global_parameters.start_iter = start_iter;
	global_parameters.write_freq = image_write_freq;
	global_parameters.FWHM = fwhm_ss;
	global_parameters.FWHM_SS = fwhm_ss;
	global_parameters.FWHM_IS = fwhm_is;
	global_parameters.FWHM_II = fwhm_ii;
	global_parameters.prior_beta = prior_beta;
	global_parameters.prior_delta = prior_delta;
	global_parameters.spherical_voxel_ratio = spherical_voxel_ratio;

	// ============================= read in movement file ============================= 
	PET_movement movement(config_filename);

	//*****************************Getting GPU information******************************//
	char* pciBusID[MAX_GPU];
	struct cudaDeviceProp cudaDeviceProp[MAX_GPU];
	int GPU_N;
	int device;
	int pciBusNameLen;
	pciBusNameLen = 256;

	cudaGetDeviceCount(&GPU_N);

	if (GPU_N - 1 < global_parameters.num_gpu_end)
	{
		global_parameters.num_gpu_end = GPU_N;
	}
	printf("\n");
	printf("CUDA-capable device count: %d\n", GPU_N);
	printf("Using from device %d to device %d\n", global_parameters.num_gpu_start, global_parameters.num_gpu_end);

	for (device = global_parameters.num_gpu_start; device <= global_parameters.num_gpu_end; device++){
		cudaDeviceReset();
		pciBusID[device] = (char*)malloc(pciBusNameLen * sizeof(char));
		cudaDeviceGetPCIBusId(pciBusID[device], pciBusNameLen, device);
		cudaGetDeviceProperties(&cudaDeviceProp[device], device);
		printf("\n");
		printf("...Device %d at PCI BUS: %s, %s at %d\n", device, pciBusID[device], cudaDeviceProp[device].name, cudaDeviceProp[device].pciBusID);
		free(pciBusID[device]);
	}
	cudaGetDevice(&device);
	printf("...Current device: %d\n", device);



	//*****************************Getting CPU multi-threading information******************************//



	// ============================= Reading in Image Arrays ============================= 
	// image, backprojection, and image update normalization arrays
	atten_image.Setup(geometry);
	// initialize the size of attenuation image from the config file
	emission_image.Setup(geometry);
	sensitivity_image.Setup(geometry); // size of sensitivity image will be equal to attenuation image
	sub_sensitivity_image.Setup(geometry);
	atten_image.ReadFromFile(img_path + atten_image_filename);

	// if CBM_flag turned on, shift all crystals to negative end
	if (CBM_flag)
	{
		geometry.shift_all_crystal(-CBM_distance_total / 2.0f); //shift all the LORs to negative end; then shift towards positive end step by step
	}

	// ============================= Reading in Normalization list mode data for block coefficients ============================= 
	// bck 0, r1 17600, f1 66752, r2 85184, f2 134336, total 126336
	if (normalization_flag){

		cout << "\n Block Based Normalization Turned On.  \n";
		int r = 0;
		int block_crystal1 = 200; // Number of crystals in one module of volume 1
		int block_crystal2 = 200; // Number of crystals in one module of volume 2
		int sn1 = 0; // Starting crystal number of volume 1
		int sn2 = 0; // Starting crystal number of volume 2
		int fn1 = 60800;
		int fn2 = 60800;
		int bn = (fn1 - sn1) / block_crystal1; // number of block modules for detector volume 1
		int bm = (fn2 - sn2) / block_crystal2; // number of block modules for detector volume 2

		int** tra = new int*[bn];
		for (int i = 0; i < bn; i++)
			tra[i] = new int[bm];

		for (int i = 0; i < bn; i++){
			for (int j = 0; j < bm; j++){
				tra[i][j] = 0;
			}
		}


		//Read Emission File and calculate number of events per Block


		int event_ptr;
		mini_coinc_event *compact_coinc_data;
		struct _stat64 file_status;
		size_t t;
		long long int size;
		int num_coincidence;
		FILE *inputfile;

		const std::string inputfile_name(data_path + emission_data_filename);


		if (_stat64(inputfile_name.c_str(), &file_status) < 0) {
			printf("Read file error: %s \n", inputfile_name.c_str());
			return 0;
		}

		printf("Size of mini_coinc_event structure is %d\n", (int)sizeof(mini_coinc_event));
		printf("Following coincidence files are being processed:\n");
		num_coincidence = 0;
		size = file_status.st_size;
		size /= 52;
		if (size >= INT_MAX){
			printf("File \"%s\" length is tooooooooooo long. This program only supports coincidence events fewer than %d. Current file has %lld. Program stopped\n", inputfile_name.c_str(), INT_MAX, size);
			return 0;
		}
		num_coincidence = (int)size;
		printf("%d coincidence events in file: %s\n", num_coincidence, inputfile_name.c_str());

		//Coincidence data memory
		compact_coinc_data = (mini_coinc_event*)malloc(num_coincidence*sizeof(mini_coinc_event));

		//Read in coincidence data
		printf("Opening list mode file, %s \n", inputfile_name.c_str());
		if ((inputfile = fopen(inputfile_name.c_str(), "rb")) == NULL) {
			printf("Error: Could not read input file. \n");
			return 0;
		}

		t = fread(compact_coinc_data, sizeof(mini_coinc_event), num_coincidence, inputfile);
		if (num_coincidence == (int)t){
			printf("%s read successfully.\n", inputfile_name.c_str());
		}
		else{
			printf("%s read error. %d elements were read\n", inputfile_name.c_str(), (int)t);
			return 0;
		}
		fclose(inputfile);

		for (int i = 0; i < num_coincidence; i++){
			if ((compact_coinc_data[i].crystal_index_1 >= sn1 && compact_coinc_data[i].crystal_index_1 < fn1) && (compact_coinc_data[i].crystal_index_2 >= sn2 && compact_coinc_data[i].crystal_index_2 < fn2)){
				int j = 0; int k = 0; int inter = 0;
				//cout << "\n coinc data: " << compact_coinc_data[i].crystal_index_1 << " " << compact_coinc_data[i].crystal_index_2;
				inter = (compact_coinc_data[i].crystal_index_1 - sn1) / block_crystal1;

				//cout << "\n inter: " << inter;
				if (inter > bn - 1 || inter < 0){
					j = (int)floor((compact_coinc_data[i].crystal_index_2 - sn1) / block_crystal1);
					k = (int)floor((compact_coinc_data[i].crystal_index_1 - sn2) / block_crystal2);

				}
				else{
					j = (int)floor((compact_coinc_data[i].crystal_index_1 - sn1) / block_crystal1);
					k = (int)floor((compact_coinc_data[i].crystal_index_2 - sn2) / block_crystal2);


				}
				tra[j][k] = tra[j][k] + 1;

			}
		}

		free(compact_coinc_data);





		/*sensitivity_image1.Setup(geometry);
		sensitivity_image1.SetValue(0.0f);

		sub_sensitivity_image1.Setup(geometry);
		sub_sensitivity_image1.SetValue(0.0f);*/

		//for (int idx = 0; idx < 16; idx = idx + 4){
		PET_data ss_events;

		ss_events.Setup(config_filename, PROMPT);

		std::cout << "\n Events are being created. ";
		//ss_events.CreateFullCoincidenceData(geometry, SS, cut[idx], cut[idx + 1], cut[idx + 2], cut[idx + 3], det1, det2);
		ss_events.CreateFullCoincidenceData_normblock(geometry, SS, sn1, fn1, sn2, fn2, time, tra);
		std::cout << " Events from " << sn1 << " to " << fn1 << "and" << sn2 << "to" << fn2 << " are created" << "\n";
		//cout << "\n Events from group " << idx + 1 << " are created. ";
		cuda_em_recon recon(global_parameters, output_image_filename_prefix, img_path);

		recon.Setup_image();


		for (int n = 1; n <= 1; n++){


			PET_data sub_ss_events(ss_events, 1, n);

			recon.Backward_Projection_Attenuation(geometry, movement, sub_ss_events, atten_image, sub_sensitivity_image);
			sensitivity_image.AddFromImageArray(sub_sensitivity_image);

		}


		//temp_image3.AddFromImageArray(sensitivity_image1);
		recon.Release_image();
		//}
		sstm.str(" ");
		sstm << img_path << output_image_filename_prefix << "brainPET_sensitvity.img";
		sensitivity_image.WriteToFile(sstm.str());


	}
	// Outsert Normalization
	/*if (normalization_flag){

	emission_image.ReadFromFile(img_path + normalization_emission_image_filename);
	sensitivity_image.ReadFromFile(img_path + sensitivity_image_filename);

	//float coeff[32768][32768] = { 0.0 };
	ImageArray<float> temp_image;
	temp_image.Setup(geometry);
	temp_image.SetValue(0.0f);

	/*ImageArray<float> temp_image2;
	temp_image2.Setup(geometry);
	temp_image2.SetValue(0.0f);

	ImageArray<float> temp_image3;
	temp_image3.Setup(geometry);
	temp_image3.SetValue(0.0f);*/



	//SS, 60800, 93568, 93568, 126336

	// Outsert Normalization
	/*int r = 0;
	int block_crystal1 = 1; // Number of crystals in one module of volume 1
	int block_crystal2 = 1; // Number of crystals in one module of volume 2
	int sn1 = 60800; // Starting crystal number of volume 1
	int sn2 = 93568; // Starting crystal number of volume 2
	int fn1 = 93568;
	int fn2 = 126336;
	int bn = fn1 - sn1;; // number of block modules for detector volume 1
	int bm = fn2 - sn2;; // number of block modules for detector volume 2
	/*int* Cut_idx_1 = new int[bn+1]; // for sensitivity image generation only
	for (int i = 0; i <= bn; i++){
	Cut_idx_1[i] = sn1 + i* block_crystal1;
	}
	int* Cut_idx_2 = new int[bm+1]; // for sensitivity image generation only
	for (int i = 0; i <= bm; i++){
	Cut_idx_2[i] = sn2 + i* block_crystal2;
	}*/
	// Outsert Normalization
	/*int* det1 = new int[bn];
	for (int i = 0; i < bn; i++){
	det1[i] = 0;
	}
	int* det2 = new int[bm];
	for (int i = 0; i < bm; i++){
	det2[i] = 0;
	}


	int** tra = new int*[bn];
	for (int i = 0; i < bn; i++)
	tra[i] = new int[bm];

	for (int i = 0; i < bn; i++){
	for (int j = 0; j < bm; j++){
	tra[i][j] = 0;
	}
	}






	int cut[16] = { 0, 30400, 60800, 93568, 0, 30400, 93568, 126336, 30400, 60800, 60800, 93568, 30400, 60800, 93568, 126336 };

	/*for (int idx = 0; idx < 15; idx = idx + 4){
	//cout << "here";
	PET_data ss_events;

	ss_events.Setup(config_filename, PROMPT);

	cout << " Events from " << cut[idx] + r << " to " << cut[idx + 1] + r << "and" << cut[idx + 2] << "to" << cut[idx + 3] << " are being created" << "\n";
	ss_events.CreateFullCoincidenceData(geometry, SS, cut[idx], cut[idx + 1], cut[idx + 2], cut[idx + 3], tra);
	//ss_events.CreateFullCoincidenceData(geometry, SS, sn1, fn1, sn2, fn2, tra);
	cout << " Events from " << cut[idx] + r << " to " << cut[idx + 1] + r << "and" << cut[idx + 2] << "to" << cut[idx+3] << " are created" << "\n";
	//cout << "\n Events from group " << idx + 1 << " are created. ";
	cuda_em_recon recon(global_parameters, output_image_filename_prefix, img_path);

	recon.Setup_image();

	int P1 = 5;
	for (int n = 1; n <= P1; n++){


	PET_data sub_ss_events(ss_events, P1, n);

	recon.Backward_Projection_Attenuation(geometry, movement, sub_ss_events, atten_image, sub_sensitivity_image, emission_image);
	sensitivity_image.AddFromImageArray(sub_sensitivity_image);
	cout << "\n\n Next subgroup \n";
	}

	//recon.Backward_Projection_Attenuation(geometry, movement, ss_events, atten_image, sensitivity_image, emission_image);




	//temp_image2.AddFromImageArray(sensitivity_image);
	recon.Release_image();
	}*/

	// Outsert Normalization
	/*temp_image.AddFromImageArray(emission_image);
	//temp_image.MultiplyBy(emission_image);
	//temp_image.ScaledBy(emission_image);
	temp_image.DivideBy(sensitivity_image);
	temp_image.ScaledBy(35184.46f);



	/* sstm.str(" ");
	sstm << img_path << output_image_filename_prefix << "_sensitivity_OS_xnorm_real.img";
	sensitivity_image.WriteToFile(sstm.str());*/

	// Outsert Normalization	
	/*sstm.str(" ");
	sstm << img_path << output_image_filename_prefix << "_temp_OO_xnorm_real.img";
	temp_image.WriteToFile(sstm.str());

	//Normalization Part


	int event_ptr;
	mini_coinc_event *compact_coinc_data;
	struct _stat64 file_status;
	size_t t;
	long long int size;
	int num_coincidence;
	FILE *inputfile;

	const std::string inputfile_name(data_path + normalization_emission_data_filename);


	if (_stat64(inputfile_name.c_str(), &file_status) < 0) {
	printf("Read file error: %s \n", inputfile_name.c_str());
	return 0;
	}

	printf("Size of mini_coinc_event structure is %d\n", (int)sizeof(mini_coinc_event));
	printf("Following coincidence files are being processed:\n");
	num_coincidence = 0;
	size = file_status.st_size;
	size /= 52;
	if (size >= INT_MAX){
	printf("File \"%s\" length is tooooooooooo long. This program only supports coincidence events fewer than %d. Current file has %lld. Program stopped\n", inputfile_name.c_str(), INT_MAX, size);
	return 0;
	}
	num_coincidence = (int)size;
	printf("%d coincidence events in file: %s\n", num_coincidence, inputfile_name.c_str());

	//Coincidence data memory
	compact_coinc_data = (mini_coinc_event*)malloc(num_coincidence*sizeof(mini_coinc_event));

	//Read in coincidence data
	printf("Opening list mode file, %s \n", inputfile_name.c_str());
	if ((inputfile = fopen(inputfile_name.c_str(), "rb")) == NULL) {
	printf("Error: Could not read input file. \n");
	return 0;
	}

	t = fread(compact_coinc_data, sizeof(mini_coinc_event), num_coincidence, inputfile);
	if (num_coincidence == (int)t){
	printf("%s read successfully.\n", inputfile_name.c_str());
	}
	else{
	printf("%s read error. %d elements were read\n", inputfile_name.c_str(), (int)t);
	return 0;
	}
	fclose(inputfile);

	int oo = 0;
	for (int i = 0; i < num_coincidence; i++){

	//if (((compact_coinc_data[i].crystal_index_1 >= sn1 && compact_coinc_data[i].crystal_index_1< fn1) && (compact_coinc_data[i].crystal_index_2 >= sn2 && compact_coinc_data[i].crystal_index_2 < fn2)) || ((compact_coinc_data[i].crystal_index_2 >= sn1 && compact_coinc_data[i].crystal_index_2< fn1) && (compact_coinc_data[i].crystal_index_1 >= sn2 && compact_coinc_data[i].crystal_index_1 < fn2))){
	oo = oo + 1;

	int j = 0; int k = 0; int inter = 0;
	//cout << "\n coinc data: " << compact_coincidence_data[i].crystal_index_1 <<" "<< compact_coincidence_data[i].crystal_index_2;
	inter = (compact_coinc_data[i].crystal_index_1 - sn1) / block_crystal1;

	//cout << "\n inter: " << inter;
	if (inter > bn - 1 || inter < 0){
	j = (int)floor((compact_coinc_data[i].crystal_index_2 - sn1) );
	k = (int)floor((compact_coinc_data[i].crystal_index_1 - sn2) );

	}
	else{
	j = (int)floor((compact_coinc_data[i].crystal_index_1 - sn1) );
	k = (int)floor((compact_coinc_data[i].crystal_index_2 - sn2) );


	}
	det1[j] = det1[j] + 1;
	det2[k] = det2[k] + 1;


	/*if ((j > 60800 || j < 0) || (k>65536 || k < 0) || (tra[j][k] > 5 || tra[j][k] < 1)){

	cout << "\n j " << j << " k " << k << " value " << tra[j][k] << "\n";
	}*/



	//}
	// Outsert Normalization
	/*if (i % 10000000 == 0) cout << "\n Current event number " << i;
	}

	int sum = 0; int min = det1[0];
	for (int j = 0; j < bn; j++){

	sum = sum + det1[j];
	if (min > det1[j]){
	min = det1[j];
	}


	}
	cout << "\n mean : " << sum / bn << " Max element : " << *max_element(det1, det1 + bn) << " Min element : " << min;
	for (int j = 0; j < bm; j++){

	sum = sum + det2[j];

	}
	cout << "\n Sum : " << sum << " oo events : " << oo << " num_coinc : " << num_coincidence<<" corner : " <<det1[0] << " center: " <<det1[7168];

	free(compact_coinc_data);

	sensitivity_image1.Setup(geometry);
	sensitivity_image1.SetValue(0.0f);

	sub_sensitivity_image1.Setup(geometry);
	sub_sensitivity_image1.SetValue(0.0f);

	//for (int idx = 0; idx < 16; idx = idx + 4){
	PET_data ss_events;

	ss_events.Setup(config_filename, PROMPT);

	//cout << "\n Events from group " << idx + 1 << " are being created. ";
	//ss_events.CreateFullCoincidenceData(geometry, SS, cut[idx], cut[idx + 1], cut[idx + 2], cut[idx + 3], det1, det2);
	ss_events.CreateFullCoincidenceData_norm(geometry, SS, sn1, fn1, sn2, fn2, time, det1, det2);
	//cout << " Events from " << Cut_idx_1[j] + r << " to " << Cut_idx_1[j + 1] + r << "and" << Cut_idx_2[k] << "to" << Cut_idx_2[k + 1] << " are created" << "\n";
	//cout << "\n Events from group " << idx + 1 << " are created. ";
	cuda_em_recon recon(global_parameters, output_image_filename_prefix, img_path);

	recon.Setup_image();



	for (int n = 1; n <= 5; n++){


	PET_data sub_ss_events(ss_events, 5, n);

	recon.Backward_Projection_Attenuation_norm(geometry, movement, sub_ss_events, atten_image, sub_sensitivity_image1, temp_image);
	sensitivity_image1.AddFromImageArray(sub_sensitivity_image1);
	cout << "\n\n next";
	}


	//temp_image3.AddFromImageArray(sensitivity_image1);
	recon.Release_image();
	//}
	sstm.str(" ");
	sstm << img_path << output_image_filename_prefix << "_sensitivity_OO_norm_timestamp"<<time<<".img";
	sensitivity_image1.WriteToFile(sstm.str());

	free(compact_coinc_data);

	}*/

	//SS, 60800, 93568, 93568, 126336,
	else {

		cout << "\n Normalization Turned Off.  \n";

		int sn1 = 0; // Starting crystal number of volume 1
		int sn2 = 0; // Starting crystal number of volume 2
		int fn1 = NUM_SCANNER_CRYSTALS;
		int fn2 = NUM_SCANNER_CRYSTALS;
		int tot_crystals1 = fn1 - sn1;
		int tot_crystals2 = fn2 - sn2;
		int num_steps1 = 2;
		int num_steps2 = 2;
		int step1 = tot_crystals1 / num_steps1;
		int step2 = tot_crystals2 / num_steps2;
		int* idx1 = new int[num_steps1];
		for (int i = 0; i < num_steps1; i++){
			idx1[i] = sn1 + step1*i;
		}
		int* idx2 = new int[num_steps1];
		for (int i = 0; i < num_steps1; i++){
			idx2[i] = sn1 + step1*(i + 1);
		}
		int* idx3 = new int[num_steps2];
		for (int i = 0; i < num_steps2; i++){
			idx3[i] = sn2 + step2*i;
		}
		int* idx4 = new int[num_steps2];
		for (int i = 0; i < num_steps2; i++){
			idx4[i] = sn2 + step2*(i + 1);
		}
		ImageArray<float> temp_image;
		temp_image.Setup(geometry);

		if (sn1 == sn2 && fn1 == fn2){
			cout << "\n Generating Sensitivity Image for SS data without normalization...\n";
			for (int n_bed = 0; n_bed < n_bed_positions; n_bed++){
				cout << "\n Generating Sensitivity Image for bed position "<<n_bed<<"...\n";
				geometry.shift_all_crystal(bed_movement_per_step); //for each bed position, shift one step

				for (int i = 0; i < num_steps1; i++){
					for (int j = i; j < num_steps1; j++){
						PET_data ss_events;
						ss_events.Setup(config_filename, PROMPT);
						ss_events.CreateFullCoincidenceData(geometry, SS, idx1[i], idx2[i], idx1[j], idx2[j]);
						cout << "\n Events from " << idx1[i] << " --> " << idx2[i] << " and " << idx3[j] << " --> " << idx4[j] << " are created" << "\n";


						cuda_em_recon recon(global_parameters, output_image_filename_prefix, img_path);

						recon.Setup_image();

						int GPU_step = 5;
						for (int n = 1; n <= GPU_step; n++){


							PET_data sub_ss_events(ss_events, GPU_step, n);

							recon.Backward_Projection_Attenuation(geometry, movement, sub_ss_events, atten_image, sub_sensitivity_image);
							sensitivity_image.AddFromImageArray(sub_sensitivity_image);

						}

						sstm.str("");
						sstm << img_path << output_image_filename_prefix << "_nonorm_sensitivity_SS_bed_position_" << n_bed << ".img";
						sensitivity_image.WriteToFile(sstm.str());


						recon.Release_image();


					}
				}
			}
		}
		else{
			cout << "\n Generating Sensitivity Image for OS/OO data without normalization...\n";
			for (int i = 0; i < num_steps1; i++){
				for (int j = 0; j < num_steps2; j++){
					PET_data ss_events;
					ss_events.Setup(config_filename, PROMPT);
					ss_events.CreateFullCoincidenceData(geometry, SS, idx1[i], idx2[i], idx3[j], idx4[j]);
					cout << "\n Events from " << idx1[i] << " --> " << idx2[i] << "and" << idx3[j] << " --> " << idx4[j] << " are created" << "\n";


					cuda_em_recon recon(global_parameters, output_image_filename_prefix, img_path);

					recon.Setup_image();

					int GPU_step = 5;
					for (int n = 1; n <= GPU_step; n++){


						PET_data sub_ss_events(ss_events, GPU_step, n);

						recon.Backward_Projection_Attenuation(geometry, movement, sub_ss_events, atten_image, sub_sensitivity_image);
						sensitivity_image.AddFromImageArray(sub_sensitivity_image);

					}


					//recon.Backward_Projection_Attenuation(geometry, movement, ss_events, atten_image, sensitivity_image);
					sstm.str("");
					sstm << img_path << output_image_filename_prefix << "_nonorm_sensitivity_OO_timestamp_" << time << ".img";
					sensitivity_image.WriteToFile(sstm.str());
					recon.Release_image();
				}
			}
		}

	}
	/*PET_data ss_events;
	ss_events.Setup(config_filename, PROMPT);

	ss_events.CreateFullCoincidenceData(geometry, SS, 0, 3000, 30400, 33400, time);
	cout << "...here...";
	cuda_em_recon recon(global_parameters, output_image_filename_prefix, img_path);

	recon.Setup_image();

	int P1 = 5;
	for (int n = 1; n <= P1; n++){


	PET_data sub_ss_events(ss_events, P1, n);

	recon.Backward_Projection_Attenuation(geometry, movement, sub_ss_events, atten_image, sub_sensitivity_image);
	sensitivity_image.AddFromImageArray(sub_sensitivity_image);
	cout << "\n\n next";
	}


	//recon.Backward_Projection_Attenuation(geometry, movement, ss_events, atten_image, sensitivity_image);
	sstm.str("");
	sstm << img_path << output_image_filename_prefix << "_sensitivity_test.img";
	sensitivity_image.WriteToFile(sstm.str());



	//geometry.print_crystal_centers("crystal_centers_position_" + to_string(0) + "of" + to_string(0) + ".txt");
	printf("Sensitivity image of fr has been generated. \n");

	recon.Release_image();
	}

	*/


	//Data is already released
	//recon.Release_image();

	//cudaProfilerStop();


	// ============================ CLEANUP MEMORY =============================

	// free data list


	// free image arrays

	cout << "Freeing up image arrays" << endl;



	atten_image.Reset();
	//sensitivity_image.Reset();

	cout << "End of program" << endl;

	return 0;
}
