#include <sys/stat.h>

#include "..\Solution_Items\global.h"

#include "..\Solution_Items\command_line.h"
#include "..\Solution_Items\ImageArray.h"
#include "..\Solution_Items\config.h"
#include "..\Solution_Items\PET_data.h"
#include "..\Solution_Items\PET_geometry.h"
#include "..\Solution_Items\MC_data_analysis.h"

#define SIZE_LIMIT 4000


void Usage(char argv0[])
{
	string program_name = CommandLine::GetProgramName(argv0);
	printf("Usage: %s <config_file>\n", program_name.c_str());
}

int encode_crystal_mCT_Insert(volume_id crystal_id){
	int crystal_index;
	int block_index;

	if (crystal_id.system == 0){//belongs to Scanner
		block_index = crystal_id.module + 4 * crystal_id.sector;
		crystal_index = 169 * block_index + crystal_id.crystal;
	}
	else{//belongs to insert
		block_index = crystal_id.module;
		crystal_index = 256 * block_index + crystal_id.crystal + 126336;
	}

	return crystal_index;
}

float calculate_distance(pos_32 crystal_center, pos_32 hit){
	float distance;

	distance = sqrt((crystal_center.x - hit.x)*(crystal_center.x - hit.x) + (crystal_center.y - hit.y)*(crystal_center.y - hit.y) + (crystal_center.z - hit.z)*(crystal_center.z - hit.z));

	return distance;
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

	float fwhm_ss, fwhm_is, fwhm_ii;

	// reconstruction parameters
	int num_cores;
	int start_num_gpu, end_num_gpu;
	int num_iter, start_iter;
	int OSEM_subsets = 1;
	int image_write_freq;
	//penalty
	float prior_beta = 0, prior_C1 = 1, prior_C2 = 1, prior_delta = 1;

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

	config.GetValue<float>("FWHM SS", fwhm_ss);
	config.GetValue<float>("FWHM IS", fwhm_is);
	config.GetValue<float>("FWHM II", fwhm_ii);

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


	printf("=== Parameter listing ===\n\n");

	if (penalty_flag) {
		printf("Using Penalized ML reconstruction with beta = %f and delta = %f\n", prior_beta, prior_delta);
	}
	else {
		printf("Using ML reconstruction\n");
	}
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




	// =================== READ IN THE FULLY 3D UNCOMPRESSED PROMPT List Mode Data =================
	FILE* inputfile;
	FILE* outputfile;
	int num_singles;
	int size;
	int offset;
	float error_distance;
	struct stat64 file_status;
	string inputfile_name = data_path + emission_data_filename;
	string outputfile_name = "singles event list.dat";
	GATE_singles_event* singles_data;
	pos_32 crystal_center;

	if (stat64(inputfile_name.c_str(), &file_status) < 0) {
		printf("Read file error: %s \n", inputfile_name.c_str());
		return 0;
	}

	printf("Size of mini_coinc_event structure is %d\n", (int)sizeof(mini_coinc_event));
	printf("Following coincidence files are being processed:\n");
	num_singles = 0;
	size = file_status.st_size;
	size /= sizeof(GATE_singles_event);
	if (size >= INT_MAX){
		printf("File \"%s\" length is tooooooooooo long. This program only supports coincidence events fewer than %d. Current file has %lld. Program stopped\n", inputfile_name.c_str(), INT_MAX, size);
		return 0;
	}
	num_singles = (int)size;
	printf("%d Singles events events in file: %s\n", num_singles, inputfile_name.c_str());


	printf("Opening list mode file, %s \n", inputfile_name.c_str());
	if ((inputfile = fopen(inputfile_name.c_str(), "rb")) == NULL) {
		printf("Error: Could not read input file. \n");
		return 0;
	}


	singles_data = (GATE_singles_event*)malloc(num_singles * sizeof(GATE_singles_event));
	fread(singles_data, sizeof(GATE_singles_event), num_singles, inputfile);
	fclose(inputfile);

	simple_singles_event* event_list;
	event_list = (simple_singles_event*)malloc(num_singles*sizeof(simple_singles_event));
	error_distance = 0.0;

	for (i = 0; i < num_singles; i++){

		event_list[i].crystal_index_1 = encode_crystal_mCT_Insert(singles_data[i].single_ID);
		event_list[i].id_1 = singles_data[i].single_ID;
		event_list[i].hit_pos.x = singles_data[i].single_position.x;
		event_list[i].hit_pos.y = singles_data[i].single_position.y;
		event_list[i].hit_pos.z = singles_data[i].single_position.z;
		
		crystal_center.x = geometry.detector_crystal_list.at(event_list[i].crystal_index_1).geometry.center[0];
		crystal_center.y = geometry.detector_crystal_list.at(event_list[i].crystal_index_1).geometry.center[1];
		crystal_center.z = geometry.detector_crystal_list.at(event_list[i].crystal_index_1).geometry.center[2];

		error_distance += calculate_distance(crystal_center, event_list[i].hit_pos);

	}

	free(singles_data);

	error_distance /= (float)num_singles;

	printf("Average error distance = %f\n\n", error_distance);


	int limit;
	limit = SIZE_LIMIT / int(sizeof(simple_singles_event)) * 1000 * 1000;
	outputfile = fopen(outputfile_name.c_str(), "wb");
	offset = 0;
	while (num_singles > limit){
		fwrite(event_list + offset, sizeof(simple_singles_event), limit, outputfile);
		num_singles -= limit;
		offset += limit;
	}
	fwrite(event_list, sizeof(simple_singles_event), num_singles, outputfile);

	fclose(outputfile);

	free(event_list);



	// ============================ CLEANUP MEMORY =============================


	cout << "End of program" << endl;

	return 0;
}






