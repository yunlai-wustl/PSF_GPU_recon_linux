// EM reconstruction algorithm
#include <omp.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <fstream>
#include <time.h>

#include "../Solution_Items/global.h"
#include "../Solution_Items/command_line.h"
#include "../Solution_Items/ImageArray.h"
#include "../Solution_Items/config.h"
#include "../Solution_Items/PET_data.h"
#include "../Solution_Items/PET_DATA_scatter.h"
#include "../Solution_Items/PET_geometry.h"
#include "../proj_functions_CUDA/cuda_em_recon.cuh"
#include "../image_update_CUDA/image_update_CUDA.cuh"


#define DTTMFMT "%Y-%m-%d-%H-%M-%S "
#define DTTMSZ 21
static char *getDtTm(char *buff) {
	time_t t = time(0);
	strftime(buff, DTTMSZ, DTTMFMT, localtime(&t));
	return buff;
}


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

	// image and backprojection arrays
	ImageArray<float> atten_image;
	ImageArray<float> sensitivity_image;
	ImageArray<float> initial_image;
	ImageArray<float> current_image;
	ImageArray<float> mask_image;
	ImageArray<float> update_factor_image;
	ImageArray<float> new_image;

	int i;

	// data pointers
	float time_start, time_stop;
	int time_gap;
	float time_start_ss, time_stop_ss;
	float time_start_is, time_stop_is;
	float time_start_ii, time_stop_ii;

	float fwhm_ss, fwhm_is, fwhm_ii;

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
	int nloglikelihood_flag = 0;
	int atten_recon_fp = 0;
	int atten_recon_bp = 0;
	int exporting_precision = 10;
	TOF_MODE NONTOF;

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


	//config.GetValue<float>("Time for Sensitivity Image", time);

	config.GetValue<int>("Reconstruction use initial image", initial_image_flag);
	config.GetValue<int>("Reconstruction use randoms", randoms_flag);
	config.GetValue<int>("Reconstruction use scatter", scatter_flag);
	config.GetValue<int>("Reconstruction use normalization", normalization_flag);
	config.GetValue<int>("Reconstruction use TOF", TOF_flag);

	config.GetValue<int>("Reconstruction using penalty", penalty_flag);
	config.GetValue<float>("Beta", prior_beta);
	config.GetValue<float>("Delta", prior_delta);
	config.GetValue<float>("C1", prior_C1);
	config.GetValue<float>("C2", prior_C2);

	config.GetValue<float>("Spherical Voxel Ratio", spherical_voxel_ratio);
	config.GetValue<int>("Export Negative Log Likelihood", nloglikelihood_flag);
	config.GetValue<int>("Export Precision", exporting_precision);
	config.GetValue<int>("Forward Projection attenuation", atten_recon_fp);
	config.GetValue<int>("Backward Projection attenuation", atten_recon_bp);

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
	printf("detector geometry file read successfully\n");

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
	global_parameters.export_negative_log_likelihood = nloglikelihood_flag;
	global_parameters.attenuation_correction_fp = atten_recon_fp;
	global_parameters.attenuation_correction_bp = atten_recon_bp;
	// ============================= read in movement file ============================= 
	PET_movement movement(config_filename);
	printf("movement file read successfully\n");



	//Normalization routine, needs to be rewritten into a seperate program
	/*if (normalization_flag){

		int num_ss, num_is, num_ii;
		float coeff_ss, coeff_is;
		ImageArray<float> sens_ss;
		ImageArray<float> sens_is;
		ImageArray<float> emission_image;

		sens_ss.Setup(geometry);
		sens_is.Setup(geometry);
		emission_image.Setup(geometry);
		sens_ss.ReadFromFile(img_path + "sensitivity_ss_atten_unscaled.img");
		sens_is.ReadFromFile(img_path + "sensitivity_is_atten_unscaled.img");
		emission_image.ReadFromFile(img_path + normalization_emission_image_filename);


		PET_data prompts_norm(config_filename, PROMPT);
		prompts_norm.ReadFromFile(data_path + normalization_emission_data_filename, geometry);
		total_event_count = prompts_norm.GetDataListLength();
		num_ss = prompts_norm.GetDataCount(SS);
		num_is = prompts_norm.GetDataCount(IS);
		num_ii = prompts_norm.GetDataCount(II);

		printf("\n\nss is ii = %d %d %d\n\n", num_ss, num_is, num_ii);

		ImageArray<float> temp_image;
		temp_image.Setup(geometry);

		temp_image.CopyFromImageArray(sens_ss);
		temp_image.MultiplyBy(emission_image);
		coeff_ss = (float)num_ss / temp_image.GetSum();

		temp_image.CopyFromImageArray(sens_is);
		temp_image.MultiplyBy(emission_image);
		coeff_is = (float)num_is / temp_image.GetSum();

		printf("coeff ss = %f\n", coeff_ss);
		printf("coeff is = %f\n", coeff_is);


		sens_ss.ScaledBy(coeff_ss);
		sens_is.ScaledBy(coeff_is);
		temp_image.CopyFromImageArray(sens_ss);
		temp_image.AddFromImageArray(sens_is);

		temp_image.WriteToFile(img_path + "sensitivity_image_ss_is.img");
		sens_ss.WriteToFile(img_path + "sensitivity_image_ss.img");
		sens_is.WriteToFile(img_path + "sensitivity_image_is.img");

	}*/
	//End of Normalization routine, needs to be rewritten into a seperate program




	//Time weighting routine
	/*
	int num_ss, num_is, num_ii;
	float coeff_ss, coeff_is;
	coeff_ss = 40678;
	coeff_is = 2978;

	ImageArray<float> sens_ss;
	ImageArray<float> sens_is;
	ImageArray<float> temp_image;

	sens_ss.Setup(geometry);
	sens_is.Setup(geometry);
	temp_image.Setup(geometry);

	sens_is.SetValue(0.0f);

	sens_ss.ReadFromFile(img_path + output_image_filename_prefix + "_sensitivity_ss.img");


	int number_of_insert_positions_temp = movement.get_num_positions();
	float total_time = 0.0f;
	float time_at_current_position;
	string ss_filename;
	for (int i = 0; i < number_of_insert_positions_temp; i++){
	time_at_current_position = movement.get_time_period(i).t_end - movement.get_time_period(i).t_start;
	time_at_current_position = 200;

	total_time += time_at_current_position;
	ss_filename = output_image_filename_prefix + "_sensitivity_is_position_" + to_string(i + 1) + "of" + to_string(number_of_insert_positions_temp) + ".img";
	std::cout << "Now processing file " << i + 1 << " of " << number_of_insert_positions_temp << ": " << ss_filename << endl;
	temp_image.ReadFromFile(img_path + ss_filename);
	temp_image.ScaledBy(time_at_current_position);
	std::cout << "Sensitivity is scaled by scan time duration at current location of " << time_at_current_position << endl << endl;
	sens_is.AddFromImageArray(temp_image);
	}
	sens_is.ScaledBy(1.0 / total_time);
	std::cout << "Finally sensitivity_is is scaled by total scan time duration of " << total_time << endl << endl;

	sens_ss.ScaledBy(1 / number_of_insert_positions_temp);

	sens_ss.ScaledBy(coeff_ss);
	sens_is.ScaledBy(coeff_is);

	temp_image.SetValue(0.0f);
	temp_image.AddFromImageArray(sens_ss);
	temp_image.AddFromImageArray(sens_is);

	sens_ss.WriteToFile(img_path + "sensitivity_image_ss_normalized.img");
	sens_is.WriteToFile(img_path + "sensitivity_image_is_normalized.img");
	temp_image.WriteToFile(img_path + "sensitivity_image_ss+is_normalized.img");
	*/
	//End of Time weighting routine





	//*****************************Getting GPU information******************************//
	char* pciBusID[MAX_GPU];
	struct cudaDeviceProp cudaDeviceProp[MAX_GPU];
	int GPU_N;
	int device;
	int pciBusNameLen;
	pciBusNameLen = 256;
	int runtimeVersion;
	int driverVersion;

	cudaGetDeviceCount(&GPU_N);
	cudaRuntimeGetVersion(&runtimeVersion); //Returns in *runtimeVersion the version number of the current CUDA Runtime instance. The version is returned as (1000 major + 10 minor). For example, CUDA 9.2 would be represented by 9020.
	cudaDriverGetVersion(&driverVersion); //Returns in *driverVersion the latest version of CUDA supported by the driver.The version is returned as(1000 major + 10 minor).For example, CUDA 9.2 would be represented by 9020. If no driver is installed, then 0 is returned as the driver version.

	if (GPU_N - 1 < global_parameters.num_gpu_end)
	{
		global_parameters.num_gpu_end = GPU_N;
	}
	printf("\n");
	printf("CUDA runtime version: %d\n", runtimeVersion);
	printf("CUDA driver version: %d\n", driverVersion);
	printf("CUDA-capable device count: %d\n", GPU_N);
	printf("Using from device %d to device %d\n", global_parameters.num_gpu_start, global_parameters.num_gpu_end);

	for (device = global_parameters.num_gpu_start; device <= global_parameters.num_gpu_end; device++){
		cudaSetDevice(device);
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
	sensitivity_image.Setup(geometry);
	initial_image.Setup(geometry);
	current_image.Setup(geometry);
	update_factor_image.Setup(geometry);
	new_image.Setup(geometry);

	atten_image.ReadFromFile(img_path + atten_image_filename);
	sensitivity_image.ReadFromFile(img_path + sensitivity_image_filename);
	if (initial_image_flag){
		initial_image.ReadFromFile(img_path + initial_image_filename);
		current_image.CopyFromImageArray(initial_image); 
	}
	else{
		current_image.SetValue(1.0f);
	}

	//creat a MASK to identify 0 elements in sensitivity images
	mask_image.Setup(geometry);

	mask_image.SetValue(1.0f);

	//mask_image.ReadFromFile(img_path + "mask_image.img");


	for (i = 0; i < global_parameters.NUM_X*global_parameters.NUM_Y*global_parameters.NUM_Z; i++){
		if (sensitivity_image._image[i] < 0.0f){
			std::cout << "Sensitivity image contains negative value " << sensitivity_image._image[i] << " at index " << i << ", ignored and treated as 0" << endl;
			sensitivity_image._image[i] = 0.0f;
			mask_image._image[i] = 0.0f;
		}
		else if (sensitivity_image._image[i] == 0.0f){
			sensitivity_image._image[i] = 0.0f;
			mask_image._image[i] = 0.0f;
		}
		else if (sensitivity_image._image[i] < SMALLEST_ALLOWED){
			//cout << "Sensitivity image contains too small value " << sensitivity_image._image[i] << " at index " << i << ", ignored and treated as 0" << endl;
			sensitivity_image._image[i] = 0.0f;
			mask_image._image[i] = 0.0f;
		}
	}
	mask_image.WriteToFile(img_path + "mask_image.img");

	sensitivity_image.ScaledBy(1.0f / ((float)(global_parameters.num_OSEM_subsets)));


	//set the update factor image to be zero
	update_factor_image.SetValue(0.0f);

	ImageArray<float> temp_image_for_likelihood_calculation;
	temp_image_for_likelihood_calculation.Setup(geometry);

	if (scatter_flag == 0){
		// =================== READ IN THE FULLY 3D UNCOMPRESSED PROMPT List Mode Data =================
		PET_data prompts(config_filename, PROMPT);
		prompts.ReadFromFile(data_path + emission_data_filename, geometry);
		total_event_count = prompts.GetDataListLength();

		/*if (randoms_flag){
			PET_data randoms(config_filename, DELAYED);
			randoms.ReadFromFile(data_path + emission_data_filename, geometry);
		}

		if (scatter_flag){
			PET_data scatters(config_filename, BODY_SCATTERED);
			scatters.ReadFromFile(data_path + scattered_data_filename, geometry);
		}
		*/
		/*PET_geometry g_adjusted = geometry;
		g_adjusted.apply_global_adjustment(movement, 126336, 0, 0);

		int number_of_insert_positions = movement.get_num_positions();
		for (int i = 0; i < number_of_insert_positions; i++){

		PET_geometry g_temp = g_adjusted;
		g_temp.apply_movement(movement, 126336, 0, i);
		g_temp.print_crystal_centers("crystal_centers_position_" + to_string(i + 1) + "of" + to_string(number_of_insert_positions) + ".txt");
		}

		geometry.print_crystal_centers("crystal_centers_position_" + to_string(0) + "of" + to_string(0) + ".txt");*/


		PET_data prompts_is(prompts, time_start_is, time_stop_is, IS, time_gap);
		PET_data prompts_ss(prompts, time_start_ss, time_stop_ss, SS, time_gap);
		PET_data prompts_ii(prompts, time_start_ii, time_stop_ii, II, time_gap);
		PET_data prompts_combo(prompts_ss, time_start_ss, time_stop_ss, ALL, time_gap);
		prompts_combo.add_data(prompts_is, time_start_is, time_stop_is, ALL);
		prompts_combo.add_data(prompts_ii, time_start_ii, time_stop_ii, ALL);


		// ================================= Performe EM iterations ==========================
		char buff[DTTMSZ];
		std::ofstream likelihood_outfile;
		sstm.str("");
		sstm << "negative_log_likelihood_values_" << getDtTm(buff) << ".txt";
		likelihood_outfile.open(sstm.str(), std::ofstream::out | std::ofstream::app);
		likelihood_outfile.precision(exporting_precision);

		float nloglikelihood_1st_term = 0.0f;
		float nloglikelihood_2nd_term = 0.0f;

		std::cout << "Starting EM iterations" << std::endl;
		cudaProfilerStart();

		cuda_em_recon recon(global_parameters, output_image_filename_prefix, img_path);
		recon.Setup_image();

		int full_iter, subset_ind, iter;
		float subset_time_start, subset_time_stop;

		new_image.SetValue(0.0f);

		for (full_iter = global_parameters.start_iter; full_iter <= global_parameters.num_iter; full_iter++){

			for (subset_ind = 1; subset_ind <= global_parameters.num_OSEM_subsets; subset_ind++){

				iter = (full_iter - 1)*global_parameters.num_OSEM_subsets + (subset_ind - 1) + 1;

				getDtTm(buff);

				std::cout << "Full iteration = " << full_iter << ",  Sub iteration =  " << subset_ind << " :" << std::endl;
				subset_time_start = time_start + (time_stop - time_start) / global_parameters.num_OSEM_subsets*(subset_ind - 1);
				subset_time_stop = time_start + (time_stop - time_start) / global_parameters.num_OSEM_subsets*subset_ind;

				PET_data prompts_subset(prompts_combo, subset_time_start, subset_time_stop, ALL, time_gap);
				cout << ".....";
				std::cout << "Start to compute update factor " << std::endl;

				recon.ComputeUpdateFactor(geometry, movement, prompts_subset, current_image, update_factor_image, atten_image, nloglikelihood_2nd_term);

				//recon.ComputeUpdateFactorSS(geometry, movement, prompts_subset, current_image, update_factor_image);

				//update_factor_image.WriteToFile(img_path + "update_image.img");

				if (nloglikelihood_flag){
					temp_image_for_likelihood_calculation.CopyFromImageArray(sensitivity_image);
					temp_image_for_likelihood_calculation.MultiplyBy(current_image);
					nloglikelihood_1st_term = temp_image_for_likelihood_calculation.GetSum();
				}

				likelihood_outfile << buff << "\t" << "iter=" << iter - 1 << "\t" << nloglikelihood_1st_term << "\t" << nloglikelihood_2nd_term << "\t" << nloglikelihood_1st_term + nloglikelihood_2nd_term << std::endl;
				std::cout << "......Perform image update......" << endl;




				if (!penalty_flag || prior_beta < 1e-10 || full_iter <= 2){
					cout << "\n mlem updated  ";
					//treated as pure ML-EM image update
					image_update_CUDA::ImageUpdateML_CUDA(new_image, update_factor_image, current_image, mask_image, sensitivity_image, geometry);
				}
				else{
					cout << "\n ---------";
					//MAP-EM image 
					global_parameters.prior_beta = global_parameters.prior_beta / global_parameters.num_OSEM_subsets;
					image_update_CUDA::ImageUpdatePL_CUDA(new_image, update_factor_image, current_image, mask_image, sensitivity_image, global_parameters, geometry);


				}



				current_image.CopyFromImageArray(new_image);
				new_image.SetValue(0.0f);
				cout << "\n here  ";
				std::cout << "......Completed image update......" << endl;



				if ((iter % image_write_freq) == 0){



					/*sstm.str("");
					sstm << img_path << output_image_filename_prefix << "_UPDATE_cuda_iter_" << iter << ".img";

					update_factor_image.WriteToFile(sstm.str());*/





					sstm.str("");
					sstm << img_path << output_image_filename_prefix << "_RECON_cuda_iter_" << iter << ".img";


					current_image.WriteToFile(sstm.str());

				}


			}


		}

		likelihood_outfile.close();

		//Data is already released
		recon.Release_image();

		cudaProfilerStop();


		// ============================ CLEANUP MEMORY =============================

		// free data list

		//deonstructor takes care of mem release of these images

		new_image.Reset();
		atten_image.Reset();
		sensitivity_image.Reset();
		initial_image.Reset();
		current_image.Reset();
		mask_image.Reset();
		update_factor_image.Reset();



		std::cout << "End of program" << endl;

		return 0;
	}
	else if (scatter_flag == 1){

		// =================== READ IN THE FULLY 3D UNCOMPRESSED PROMPT List Mode Data =================
		PET_data_scatter prompts(config_filename, PROMPT);
		prompts.ReadFromFile(data_path + emission_data_filename, geometry);
		total_event_count = prompts.GetDataListLength();

		/*PET_geometry g_adjusted = geometry;
		g_adjusted.apply_global_adjustment(movement, 126336, 0, 0);

		int number_of_insert_positions = movement.get_num_positions();
		for (int i = 0; i < number_of_insert_positions; i++){

		PET_geometry g_temp = g_adjusted;
		g_temp.apply_movement(movement, 126336, 0, i);
		g_temp.print_crystal_centers("crystal_centers_position_" + to_string(i + 1) + "of" + to_string(number_of_insert_positions) + ".txt");
		}

		geometry.print_crystal_centers("crystal_centers_position_" + to_string(0) + "of" + to_string(0) + ".txt");*/


		PET_data_scatter prompts_is(prompts, time_start_is, time_stop_is, IS, time_gap);
		PET_data_scatter prompts_ss(prompts, time_start_ss, time_stop_ss, SS, time_gap);
		PET_data_scatter prompts_ii(prompts, time_start_ii, time_stop_ii, II, time_gap);

		PET_data_scatter prompts_combo(prompts_ss, time_start_ss, time_stop_ss, ALL, time_gap);
		prompts_combo.add_data(prompts_is, time_start_is, time_stop_is, ALL);
		prompts_combo.add_data(prompts_ii, time_start_ii, time_stop_ii, ALL);


		// ================================= Performe EM iterations ==========================
		char buff[DTTMSZ];
		std::ofstream likelihood_outfile;
		sstm.str("");
		sstm << "negative_log_likelihood_values_" << getDtTm(buff) << ".txt";
		likelihood_outfile.open(sstm.str(), std::ofstream::out | std::ofstream::app);
		likelihood_outfile.precision(exporting_precision);

		float nloglikelihood_1st_term = 0.0f;
		float nloglikelihood_2nd_term = 0.0f;

		std::cout << "Starting EM iterations" << std::endl;
		cudaProfilerStart();

		cuda_em_recon recon(global_parameters, output_image_filename_prefix, img_path);
		recon.Setup_image();

		int full_iter, subset_ind, iter;
		float subset_time_start, subset_time_stop;

		new_image.SetValue(0.0f);

		for (full_iter = global_parameters.start_iter; full_iter <= global_parameters.num_iter; full_iter++){

			for (subset_ind = 1; subset_ind <= global_parameters.num_OSEM_subsets; subset_ind++){

				iter = (full_iter - 1)*global_parameters.num_OSEM_subsets + (subset_ind - 1) + 1;

				getDtTm(buff);

				std::cout << "Full iteration = " << full_iter << ",  Sub iteration =  " << subset_ind << " :" << std::endl;
				subset_time_start = time_start + (time_stop - time_start) / global_parameters.num_OSEM_subsets*(subset_ind - 1);
				subset_time_stop = time_start + (time_stop - time_start) / global_parameters.num_OSEM_subsets*subset_ind;

				PET_data_scatter prompts_subset(prompts_combo, subset_time_start, subset_time_stop, ALL, time_gap);
				cout << ".....";
				std::cout << "Start to compute update factor " << std::endl;

				recon.ComputeUpdateFactorScatter(geometry, movement, prompts_subset, current_image, update_factor_image, atten_image, nloglikelihood_2nd_term);

				//recon.ComputeUpdateFactorSS(g_adjusted, movement, prompts_subset, current_image, update_factor_image);

				//update_factor_image.WriteToFile(img_path + "update_image.img");

				if (nloglikelihood_flag){
					temp_image_for_likelihood_calculation.CopyFromImageArray(sensitivity_image);
					temp_image_for_likelihood_calculation.MultiplyBy(current_image);
					nloglikelihood_1st_term = temp_image_for_likelihood_calculation.GetSum();
				}

				likelihood_outfile << buff << "\t" << "iter=" << iter - 1 << "\t" << nloglikelihood_1st_term << "\t" << nloglikelihood_2nd_term << "\t" << nloglikelihood_1st_term + nloglikelihood_2nd_term << std::endl;
				std::cout << "......Perform image update......" << endl;




				if (!penalty_flag || prior_beta < 1e-10 || full_iter <= 2){
					cout << "\n mlem updated  ";
					//treated as pure ML-EM image update
					image_update_CUDA::ImageUpdateML_CUDA(new_image, update_factor_image, current_image, mask_image, sensitivity_image, geometry);
				}
				else{
					cout << "\n ---------";
					//MAP-EM image 
					global_parameters.prior_beta = global_parameters.prior_beta / global_parameters.num_OSEM_subsets;
					image_update_CUDA::ImageUpdatePL_CUDA(new_image, update_factor_image, current_image, mask_image, sensitivity_image, global_parameters, geometry);


				}



				current_image.CopyFromImageArray(new_image);
				new_image.SetValue(0.0f);
				cout << "\n here  ";
				std::cout << "......Completed image update......" << endl;



				if ((iter % image_write_freq) == 0){



					/*sstm.str("");
					sstm << img_path << output_image_filename_prefix << "_UPDATE_cuda_iter_" << iter << ".img";

					update_factor_image.WriteToFile(sstm.str());
					*/




					sstm.str("");
					sstm << img_path << output_image_filename_prefix << "_RECON_cuda_iter_" << iter << ".img";


					current_image.WriteToFile(sstm.str());

				}


			}


		}

		//likelihood_outfile.close();

		//Data is already released
		recon.Release_image();

		cudaProfilerStop();


		// ============================ CLEANUP MEMORY =============================

		// free data list

		//deonstructor takes care of mem release of these images

		new_image.Reset();
		atten_image.Reset();
		sensitivity_image.Reset();
		initial_image.Reset();
		current_image.Reset();
		mask_image.Reset();
		update_factor_image.Reset();



		std::cout << "End of program" << endl;

		return 0;

	}
	
}