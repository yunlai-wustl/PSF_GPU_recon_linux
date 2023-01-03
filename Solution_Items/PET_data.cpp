#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <stdlib.h>


#include "PET_data.h"
#include <math.h>


PET_data::PET_data(){
	PET_LST_event_list.reserve(MAX_NUM_LST_EVENT);
	//PET_LST_events_pos_list.reserve(MAX_NUM_LST_EVENT);
	//PET_LST_events_pos_list_1.reserve(MAX_NUM_LST_EVENT);
	//PET_LST_events_pos_list_2.reserve(MAX_NUM_LST_EVENT);
}

PET_data::PET_data(int length){
	PET_LST_event_list.reserve(length);
}


PET_data::PET_data(const std::string config_file_name, PET_data_type type){
	Setup(config_file_name, type);

	PET_LST_event_list.reserve(MAX_NUM_LST_EVENT);
	//PET_LST_events_pos_list.reserve(MAX_NUM_LST_EVENT);
	//PET_LST_events_pos_list_1.reserve(MAX_NUM_LST_EVENT);
	//PET_LST_events_pos_list_2.reserve(MAX_NUM_LST_EVENT);


}

PET_data::PET_data(PET_data &data_from, float time_start, float time_end, PET_coincidence_type t, int time_gap){
	int i;
	float time;
	
	_acq_time_start = time_start;
	_acq_time_end = time_end;
	_acq_time_length = time_end - time_start;
	_acq_time_gap = time_gap;

	if (_acq_time_start<data_from._acq_time_start || _acq_time_end>data_from._acq_time_end){
		printf("\n Subset data time range outside the original data time range, using original instead\n");
		_acq_time_start = data_from._acq_time_start;
		_acq_time_end = data_from._acq_time_end;
		_acq_time_length = data_from._acq_time_length;
		_acq_time_gap = data_from._acq_time_gap;
	}


	_source = data_from._source;
	_data_type = data_from._data_type;
	
	_num_ss_event = 0;
	_num_ii_event = 0;
	_num_is_event = 0;
	_num_event = 0;

	for (i = 0; i < data_from._num_event; i++){
		time = data_from.PET_LST_event_list.at(i).t0;
			if (time >= time_start && time <= time_end && data_from.PET_LST_event_list.at(i).is_event_type(t)){
			

			if (_source == INVEON || _source == NON_TOF_MINIMAL || _source == TOF_MINIMAL){
				PET_LST_event_list.push_back(data_from.PET_LST_event_list.at(i));
			}
			else if (_source == COMPACT_COINCIDENCE){
				PET_LST_event_list.push_back(data_from.PET_LST_event_list.at(i));
				//PET_LST_events_pos_list.push_back(data_from.PET_LST_events_pos_list.at(i));
			}
			else if (_source == XYZ_COINCIDENCE){
				PET_LST_events_pos_list_1.push_back(data_from.PET_LST_events_pos_list_1.at(i));
				PET_LST_events_pos_list_2.push_back(data_from.PET_LST_events_pos_list_2.at(i));
			}
			else if (_source == CBM){
				PET_LST_event_list.push_back(data_from.PET_LST_event_list.at(i));
			}
			_num_event++;

			if (data_from.PET_LST_event_list[i].is_event_type(SS)){
				_num_ss_event++;
			}
			else if (data_from.PET_LST_event_list[i].is_event_type(II)){
				_num_ii_event++;
			}
			else{
				_num_is_event++;
			}
		}
	}

	printf("...Subset includes list mode event from time %f to %f\n", time_start, time_end);
	printf("...Total counts = %d, SS = %d, IS = %d, II = %d \n", _num_event, _num_ss_event, _num_is_event, _num_ii_event);
}

PET_data::PET_data(PET_data &data_from, int num_subsets, int subset){
	int i;

	if (!(subset > 0 && subset <= num_subsets)){
		fputs("Invalid subset and number of subsets pair.\n", stderr);
		exit(1);
	}
	
	//_acq_time_start = data_from._acq_time_start;
	//_acq_time_end = data_from._acq_time_end;
	//_acq_time_length = data_from._acq_time_length;
	//_source = data_from._source;
	//_data_type = data_from._data_type;
	_num_event = (data_from._num_event) / num_subsets;
	//_source = data_from._source;

	for (i = (subset - 1)*_num_event; i < subset*_num_event; i++){
		PET_LST_event_list.push_back(data_from.PET_LST_event_list.at(i));
	}
	printf("Subset %d includes events from %d to %d\nA total of %d events\n", subset, i - _num_event, i, _num_event);
}


PET_data::~PET_data(){

}


const unsigned char
PET_data::graycode[11] = { 0x00, 0x10, 0x30, 0x20, 0x60, 0x70, 0x50, 0x40, 0x00, 0x10, 0x30 };



void
PET_data::add_data(PET_data &data_from, float time_start, float time_end, PET_coincidence_type t){
	int i;
	float time;

	_acq_time_start = time_start;
	_acq_time_end = time_end;
	_acq_time_length = time_end - time_start;

	if (_acq_time_start<data_from._acq_time_start || _acq_time_end>data_from._acq_time_end){
		printf("\n Add data time range outside the original data time range, using original instead\n");
		_acq_time_start = data_from._acq_time_start;
		_acq_time_end = data_from._acq_time_end;
		_acq_time_length = data_from._acq_time_length;
	}


	if (_source != data_from._source){
		printf("Source type does not match!\n");
		exit(1);
	};
	
	if (_data_type != data_from._data_type){
		printf("Data type does not match!\n");
		exit(1);

	}



	for (i = 0; i < data_from._num_event; i++){
		time = data_from.PET_LST_event_list.at(i).t0;
		if (time >= time_start && time <= time_end && data_from.PET_LST_event_list.at(i).is_event_type(t)){


			if (_source == INVEON || _source == NON_TOF_MINIMAL || _source == TOF_MINIMAL){
				PET_LST_event_list.push_back(data_from.PET_LST_event_list.at(i));
			}
			else if (_source == COMPACT_COINCIDENCE){
				PET_LST_event_list.push_back(data_from.PET_LST_event_list.at(i));
				//PET_LST_events_pos_list.push_back(data_from.PET_LST_events_pos_list.at(i));
			}
			else if (_source == XYZ_COINCIDENCE){
				PET_LST_events_pos_list_1.push_back(data_from.PET_LST_events_pos_list_1.at(i));
				PET_LST_events_pos_list_2.push_back(data_from.PET_LST_events_pos_list_2.at(i));
			}

			_num_event++;

			if (data_from.PET_LST_event_list[i].is_event_type(SS)){
				_num_ss_event++;
			}
			else if (data_from.PET_LST_event_list[i].is_event_type(II)){
				_num_ii_event++;
			}
			else{
				_num_is_event++;
			}
		}
	}

	printf("...Combined data includes list mode event from time %f to %f\n", time_start, time_end);
	printf("...Total counts = %d, SS = %d, IS = %d, II = %d \n", _num_event, _num_ss_event, _num_is_event, _num_ii_event);
}




int
PET_data::GetDataListLength(){
	return (int)PET_LST_event_list.size();
}


void
PET_data::Setup(const std::string config_file_name, PET_data_type type){
	Config config(config_file_name);
	config.GetValue<float>("Time start", _acq_time_start);
	config.GetValue<float>("Time stop", _acq_time_end);
	config.GetValue<int>("Time gap", _acq_time_gap);
	config.GetValue<string>("Data source", _data_source);
	config.GetValue<string>("Scan protocol", _scan_protocol);

	_acq_time_length = _acq_time_end - _acq_time_start;
	if (_acq_time_length < 0.001){
		printf("Time stop must be greater than Time start.\n");
		throw 1;
	}
	else{
		//printf("%s data selected from %f to %f, a total of %f seconds.\n", _data_source.c_str(), _acq_time_start, _acq_time_end, _acq_time_length);
	}
	if (_scan_protocol == "Static"){
		_protocol = STATIC;
	}
	if (_scan_protocol == "CBM"){
		_protocol = CBM;
	}
	if (_scan_protocol == "Step and Shoot"){
		_protocol = STEP_AND_SHOOT;
	}




	if (_data_source == "Inveon"){
		_source = INVEON;
	}
	if (_data_source == "GATE Coincidence"){
		_source = GATE_COINCIDENCE;
	}
	if (_data_source == "Compact Coincidence"){
		_source = COMPACT_COINCIDENCE;
	}
	if (_data_source == "GATE Singles"){
		_source = GATE_SINGLES;
	}
	if (_data_source == "XYZ Coincidence"){
		_source = XYZ_COINCIDENCE;
	}
	if (_data_source == "Non TOF Minimal"){
		_source = NON_TOF_MINIMAL;
	}
	if (_data_source == "TOF Minimal"){
		_source = TOF_MINIMAL;
	}

	_data_type = type;
}

void
PET_data::ReadFromFile(const std::string filename, PET_geometry geometry){
	
	if (_source == INVEON){
		_read_Inveon_LST_data(filename, _num_event, _data_type);
	}
	else if (_source == COMPACT_COINCIDENCE){
		_read_compact_coincidence_LST_data(filename, _num_event, _data_type, geometry);
	}
	else if (_source == NON_TOF_MINIMAL){
		_read_non_TOF_LST_data(filename, _num_event, _data_type);
	}
	else if (_source == TOF_MINIMAL){
		_read_TOF_LST_data(filename, _num_event, _data_type);
	}
	else{
		printf("Support for this kind of source file is not implemented yet!\n");
	}

	PET_LST_event_list.shrink_to_fit();
	//PET_LST_events_pos_list.shrink_to_fit();
	//PET_LST_events_pos_list_1.shrink_to_fit();
	//PET_LST_events_pos_list_2.shrink_to_fit();

	printf("%d coincidence events have been read into the data list.\n\n", _num_event);
	
}



int
PET_data::_read_non_TOF_LST_data(const std::string inputfile_name, int &total_prompt_event, PET_data_type mode){
	PET_LST_event_type event_mode;
	int event_ptr;
	nontof_coinc_event *coincidence_data;
	double time_tag;
	int i;
	struct stat64 file_status;
	long long int size;
	size_t t;
	int num_coincidence;
	FILE *inputfile;


	if (mode == DELAYED){
		event_mode = eDELAYED;
	}
	else if (mode == PROMPT){
		event_mode = ePROMPT;
	}
	else{
		printf("Only Prompt and Delayed events are allowed for ");
		return 0;
	}


	if (stat64(inputfile_name.c_str(), &file_status) < 0) {
		printf("Read file error: %s \n", inputfile_name.c_str());
		return 0;
	}

	printf("Following coincidence files are being processed:\n");
	num_coincidence = 0;
	size = file_status.st_size;
	size /= sizeof(nontof_coinc_event);
	if (size >= INT_MAX){
		printf("File \"%s\" length is tooooooooooo long. This program only supports coincidence events fewer than %d. Current file has %lld. Program stopped\n", inputfile_name.c_str(), INT_MAX, size);
		return 0;
	}
	num_coincidence = (int)size;
	printf("%d coincidence events in file: %s\n", num_coincidence, inputfile_name.c_str());

	//Coincidence data memory
	coincidence_data = (nontof_coinc_event*)malloc(num_coincidence*sizeof(nontof_coinc_event));

	//Read in coincidence data
	printf("Opening list mode file, %s \n", inputfile_name.c_str());
	if ((inputfile = fopen(inputfile_name.c_str(), "rb")) == NULL) {
		printf("Error: Could not read input file. \n");
		return 0;
	}

	t = fread(coincidence_data, sizeof(nontof_coinc_event), num_coincidence, inputfile);
	if (num_coincidence == (int)t){
		printf("%s read successfully.\n", inputfile_name.c_str());
	}
	else{
		printf("%s read error. %d elements were read\n", inputfile_name.c_str(), (int)t);
		return 0;
	}
	fclose(inputfile);

	//


	event_ptr = 0;
	_num_ss_event = 0;
	_num_is_event = 0;
	_num_ii_event = 0;
	int r1r1 = 0; int r1r2 = 0; int r2r2 = 0; int f1r1 = 0; int f1r2 = 0; int f2r1 = 0; int f2r2 = 0; int r1b = 0; int r2b = 0; int f1b = 0; int f2b = 0;

	for (i = 0; i<num_coincidence; i++){
		time_tag = coincidence_data[i].time_1;

		if (MAX_NUM_LST_EVENT > event_ptr && time_tag >= _acq_time_start && time_tag <= _acq_time_end){


			PET_LST_event current_event(coincidence_data[i].crystal_index_1, coincidence_data[i].crystal_index_2, 0.0f, (float)(coincidence_data[i].time_1), 1);

			//printf("%d %d\n", compact_coincidence_data[i].crystal_index_1, compact_coincidence_data[i].crystal_index_2);
			PET_LST_event_list.push_back(current_event);

			if (coincidence_data[i].crystal_index_1 < NUM_SCANNER_CRYSTALS && coincidence_data[i].crystal_index_2 < NUM_SCANNER_CRYSTALS){

				_num_ss_event++;

			}
			else if (coincidence_data[i].crystal_index_1 >= NUM_SCANNER_CRYSTALS && coincidence_data[i].crystal_index_2 >= NUM_SCANNER_CRYSTALS){
				_num_ii_event++;
			}
			else{
				_num_is_event++;
			}

			event_ptr++;
		}
	}

	total_prompt_event = event_ptr;

	printf("Select time slot from %f seconds to %f seconds with a gap of %d seconds in between.\n", _acq_time_start, _acq_time_end, _acq_time_gap);
	printf("Total acquisition time is %f seconds.\n", time_tag);
	printf("%d events selected from %d total events.\n", event_ptr, num_coincidence);



	free(coincidence_data);

	return 1;

}


int
PET_data::_read_TOF_LST_data(const std::string inputfile_name, int &total_prompt_event, PET_data_type mode){
	PET_LST_event_type event_mode;
	int event_ptr;
	tof_coinc_event *coincidence_data;
	double time_tag;
	int i;
	struct stat64 file_status;
	long long int size;
	size_t t;
	int num_coincidence;
	FILE *inputfile;


	if (mode == DELAYED){
		event_mode = eDELAYED;
	}
	else if (mode == PROMPT){
		event_mode = ePROMPT;
	}
	else{
		printf("Only Prompt and Delayed events are allowed for ");
		return 0;
	}


	if (stat64(inputfile_name.c_str(), &file_status) < 0) {
		printf("Read file error: %s \n", inputfile_name.c_str());
		return 0;
	}

	printf("Following coincidence files are being processed:\n");
	num_coincidence = 0;
	size = file_status.st_size;
	size /= sizeof(tof_coinc_event);
	if (size >= INT_MAX){
		printf("File \"%s\" length is tooooooooooo long. This program only supports coincidence events fewer than %d. Current file has %lld. Program stopped\n", inputfile_name.c_str(), INT_MAX, size);
		return 0;
	}
	num_coincidence = (int)size;
	printf("%d coincidence events in file: %s\n", num_coincidence, inputfile_name.c_str());

	//Coincidence data memory
	coincidence_data = (tof_coinc_event*)malloc(num_coincidence*sizeof(tof_coinc_event));

	//Read in coincidence data
	printf("Opening list mode file, %s \n", inputfile_name.c_str());
	if ((inputfile = fopen(inputfile_name.c_str(), "rb")) == NULL) {
		printf("Error: Could not read input file. \n");
		return 0;
	}

	t = fread(coincidence_data, sizeof(tof_coinc_event), num_coincidence, inputfile);
	if (num_coincidence == (int)t){
		printf("%s read successfully.\n", inputfile_name.c_str());
	}
	else{
		printf("%s read error. %d elements were read\n", inputfile_name.c_str(), (int)t);
		return 0;
	}
	fclose(inputfile);

	//


	event_ptr = 0;
	_num_ss_event = 0;
	_num_is_event = 0;
	_num_ii_event = 0;
	int r1r1 = 0; int r1r2 = 0; int r2r2 = 0; int f1r1 = 0; int f1r2 = 0; int f2r1 = 0; int f2r2 = 0; int r1b = 0; int r2b = 0; int f1b = 0; int f2b = 0;

	for (i = 0; i<num_coincidence; i++){
		time_tag = coincidence_data[i].time_1;

		if (MAX_NUM_LST_EVENT > event_ptr && time_tag >= _acq_time_start && time_tag <= _acq_time_end){


			PET_LST_event current_event(coincidence_data[i].crystal_index_1, coincidence_data[i].crystal_index_2, coincidence_data[i].diff_time * 1000, (float)(coincidence_data[i].time_1), 1);

			//printf("%d %d\n", compact_coincidence_data[i].crystal_index_1, compact_coincidence_data[i].crystal_index_2);
			PET_LST_event_list.push_back(current_event);

			if (coincidence_data[i].crystal_index_1 < 126336 && coincidence_data[i].crystal_index_2 < 126336){

					_num_ss_event++;
			}
			else if (coincidence_data[i].crystal_index_1 >= 126336 && coincidence_data[i].crystal_index_2 >= 126336){
				_num_ii_event++;
			}
			else{
				_num_is_event++;
			}

			event_ptr++;
		}
	}

	total_prompt_event = event_ptr;

	printf("Select time slot from %f seconds to %f seconds.\n", _acq_time_start, _acq_time_end);
	printf("Total acquisition time is %f seconds.\n", time_tag);
	printf("%d events selected from %d total events.\n", event_ptr, num_coincidence);

	free(coincidence_data);

	return 1;

}

int
PET_data::_read_TOF_LST_data_CBM(const std::string inputfile_name, int &total_prompt_event, PET_data_type mode){
	PET_LST_event_type event_mode;
	int event_ptr;
	tof_coinc_event_cbm *coincidence_data;
	double time_tag;
	int i;
	struct stat64 file_status;
	long long int size;
	size_t t;
	int num_coincidence;
	FILE *inputfile;


	if (mode == DELAYED){
		event_mode = eDELAYED;
	}
	else if (mode == PROMPT){
		event_mode = ePROMPT;
	}
	else{
		printf("Only Prompt and Delayed events are allowed for ");
		return 0;
	}


	if (stat64(inputfile_name.c_str(), &file_status) < 0) {
		printf("Read file error: %s \n", inputfile_name.c_str());
		return 0;
	}

	printf("Following coincidence files are being processed:\n");
	num_coincidence = 0;
	size = file_status.st_size;
	size /= sizeof(tof_coinc_event_cbm);
	if (size >= INT_MAX){
		printf("File \"%s\" length is tooooooooooo long. This program only supports coincidence events fewer than %d. Current file has %lld. Program stopped\n", inputfile_name.c_str(), INT_MAX, size);
		return 0;
	}
	num_coincidence = (int)size;
	printf("%d coincidence events in file: %s\n", num_coincidence, inputfile_name.c_str());

	//Coincidence data memory
	coincidence_data = (tof_coinc_event_cbm*)malloc(num_coincidence*sizeof(tof_coinc_event_cbm));

	//Read in coincidence data
	printf("Opening list mode file, %s \n", inputfile_name.c_str());
	if ((inputfile = fopen(inputfile_name.c_str(), "rb")) == NULL) {
		printf("Error: Could not read input file. \n");
		return 0;
	}

	t = fread(coincidence_data, sizeof(tof_coinc_event_cbm), num_coincidence, inputfile);
	if (num_coincidence == (int)t){
		printf("%s read successfully.\n", inputfile_name.c_str());
	}
	else{
		printf("%s read error. %d elements were read\n", inputfile_name.c_str(), (int)t);
		return 0;
	}
	fclose(inputfile);

	//


	event_ptr = 0;
	_num_ss_event = 0;
	_num_is_event = 0;
	_num_ii_event = 0;

	for (i = 0; i<num_coincidence; i++){
		time_tag = coincidence_data[i].time_1;

		if (MAX_NUM_LST_EVENT > event_ptr && time_tag >= _acq_time_start && time_tag <= _acq_time_end){
				PET_LST_event current_event(coincidence_data[i].crystal_index_1, coincidence_data[i].crystal_index_2, 1.5e11*coincidence_data[i].diff_time, (float)(coincidence_data[i].time_1),  1, coincidence_data[i].bed_position);
				//PET_LST_event current_event(coincidence_data[i].crystal_index_1, coincidence_data[i].crystal_index_2, 0, (float)(coincidence_data[i].time_1), 511, event_mode, 1, coincidence_data[i].bed_position);

			//printf("%d %d\n", compact_coincidence_data[i].crystal_index_1, compact_coincidence_data[i].crystal_index_2);
			PET_LST_event_list.push_back(current_event);

			if (coincidence_data[i].crystal_index_1 < NUM_SCANNER_CRYSTALS && coincidence_data[i].crystal_index_2 < NUM_SCANNER_CRYSTALS){

				_num_ss_event++;
			}
			else if (coincidence_data[i].crystal_index_1 >= NUM_SCANNER_CRYSTALS && coincidence_data[i].crystal_index_2 >= NUM_SCANNER_CRYSTALS){
				_num_ii_event++;
			}
			else{
				_num_is_event++;
			}

			event_ptr++;
		}
	}

	total_prompt_event = event_ptr;

	printf("Select time slot from %f seconds to %f seconds.\n", _acq_time_start, _acq_time_end);
	printf("Total acquisition time is %f seconds.\n", time_tag);
	printf("%d events selected from %d total events.\n", event_ptr, num_coincidence);

	free(coincidence_data);

	return 1;

}



int
PET_data::_read_XYZ_coincidence_LST_data(const std::string inputfile_name, int &total_prompt_event, PET_data_type mode){

	FILE *data_file;
	int event_index;
	int total_num_coincidence = 0;
	data_file = fopen(inputfile_name.c_str(), "r");
	pos_32 XYZ_1;
	pos_32 XYZ_2;
	float dumb;

	for (event_index = 0; event_index < MAX_NUM_LST_EVENT && !feof(data_file); event_index++) {
		
		// read in first coincidence
		if (fscanf(data_file, "%f %f %f %f %f %f %f %f", &XYZ_1.x, &XYZ_1.y, &XYZ_1.z, &dumb, &XYZ_2.x, &XYZ_2.y, &XYZ_2.z, &dumb) == EOF) {
			break;
		}
		
		PET_LST_events_pos_list_1.push_back(XYZ_1);
		PET_LST_events_pos_list_2.push_back(XYZ_2);

		total_num_coincidence++;
	}

	total_prompt_event = total_num_coincidence;

	return total_num_coincidence;

}

void
PET_data::CreateFullCoincidenceData_norm(PET_geometry geometry, PET_coincidence_type type, int lower1, int upper1, int lower2, int upper2, float time, int* det1, int* det2){
	_create_full_coincidence_LST_data_norm(geometry, time, _num_event, type, lower1, upper1, lower2, upper2, det1, det2 );
}

void
PET_data::CreateFullCoincidenceData_normblock(PET_geometry geometry, PET_coincidence_type type, int lower1, int upper1, int lower2, int upper2, float time, int** det1){
	_create_full_coincidence_LST_data_normblock(geometry, time, _num_event, type, lower1, upper1, lower2, upper2, det1);
}

void
PET_data::CreateFullCoincidenceData(PET_geometry geometry, PET_coincidence_type type, int lower1, int upper1, int lower2, int upper2, float time){
	_create_full_coincidence_LST_data(geometry, time, _num_event, type, lower1, upper1, lower2, upper2);
}

void
PET_data::CreateFullCoincidenceData(PET_geometry geometry, PET_coincidence_type type, int lower1, int upper1, int lower2, int upper2){
	_create_min_coincidence_LST_data(geometry, _num_event, type, lower1, upper1, lower2, upper2);
}

int
PET_data::_read_compact_coincidence_LST_data(const std::string inputfile_name, int &total_prompt_event, PET_data_type mode, PET_geometry &detector){
	PET_LST_event_type event_mode;
	int event_ptr;
	mini_coinc_event *compact_coincidence_data;
	double time_tag;
	int i;
	struct stat64 file_status;
	long long int size;
	size_t t;
	int num_coincidence;
	FILE *inputfile;


	if (mode == DELAYED){
		event_mode = eDELAYED;
	}
	else if (mode == PROMPT){
		event_mode = ePROMPT;
	}
	else{
		printf("Only Prompt and Delayed events are allowed for ");
		return 0;
	}


	if (stat64(inputfile_name.c_str(), &file_status) < 0) {
		printf("Read file error: %s \n", inputfile_name.c_str());
		return 0;
	}

	printf("Size of mini_coinc_event structure is %d\n", (int)sizeof(mini_coinc_event));
	printf("Following coincidence files are being processed:\n");
	num_coincidence = 0;
	size = file_status.st_size;
	size /= sizeof(mini_coinc_event);
	if (size >= INT_MAX){
		printf("File \"%s\" length is tooooooooooo long. This program only supports coincidence events fewer than %d. Current file has %lld. Program stopped\n", inputfile_name.c_str(), INT_MAX, size);
		return 0;
	}
	num_coincidence = (int)size;
	printf("%d coincidence events in file: %s\n", num_coincidence, inputfile_name.c_str());

	//Coincidence data memory
	compact_coincidence_data = (mini_coinc_event*)malloc(num_coincidence*sizeof(mini_coinc_event));

	//Read in coincidence data
	printf("Opening list mode file, %s \n", inputfile_name.c_str());
	if ((inputfile = fopen(inputfile_name.c_str(), "rb")) == NULL) {
		printf("Error: Could not read input file. \n");
		return 0;
	}

	t = fread(compact_coincidence_data, sizeof(mini_coinc_event), num_coincidence, inputfile);
	if (num_coincidence == (int)t){
		printf("%s read successfully.\n", inputfile_name.c_str());
	}
	else{
		printf("%s read error. %d elements were read\n", inputfile_name.c_str(), (int)t);
		return 0;
	}
	fclose(inputfile);

	//

	usleep(10000);
	cout << "\n Now going through events.....";
	event_ptr = 0;
	_num_ss_event = 0;
	_num_is_event = 0;
	_num_ii_event = 0;
	float _current_bed_poistion = 0.0f;
	box s, d;

	//cout << "\n " << compact_coincidence_data[305746166].crystal_index_1 << " " << compact_coincidence_data[305746166].crystal_index_2 << " " << compact_coincidence_data[305746166].time_1 << " " << compact_coincidence_data[305746166].diff_time;

	for (i = 0; i<num_coincidence; i++){
		time_tag = compact_coincidence_data[i].time_1;
		
		if (MAX_NUM_LST_EVENT > event_ptr && time_tag >= _acq_time_start && time_tag <= _acq_time_end && (int(time_tag) % _acq_time_gap) < 1 ){
			if (_protocol != STATIC)
				_current_bed_poistion = compact_coincidence_data[i].bed_position;
			PET_LST_event current_event(compact_coincidence_data[i].crystal_index_1, compact_coincidence_data[i].crystal_index_2, (float)(1.5e11 * compact_coincidence_data[i].diff_time), (float)(compact_coincidence_data[i].time_1), 1, _current_bed_poistion);
			//printf("%d %d\n", compact_coincidence_data[i].crystal_index_1, compact_coincidence_data[i].crystal_index_2);
			if (compact_coincidence_data[i].crystal_index_1 < 0 || compact_coincidence_data[i].crystal_index_2 < 0|| \
			compact_coincidence_data[i].crystal_index_1 >= NUM_SCANNER_CRYSTALS+NUM_INSERT_CRYSTALS \
			||compact_coincidence_data[i].crystal_index_2 >= NUM_SCANNER_CRYSTALS+NUM_INSERT_CRYSTALS)
				continue;
			PET_LST_event_list.push_back(current_event);
			if (compact_coincidence_data[i].crystal_index_1 < NUM_SCANNER_CRYSTALS && compact_coincidence_data[i].crystal_index_2 < NUM_SCANNER_CRYSTALS){
				_num_ss_event++;
			}
			else if (compact_coincidence_data[i].crystal_index_1 >= NUM_SCANNER_CRYSTALS && compact_coincidence_data[i].crystal_index_2 >= NUM_SCANNER_CRYSTALS){
				_num_ii_event++;
			}
			else{
				_num_is_event++;
			}

			event_ptr++;
		}
	}

	total_prompt_event = event_ptr;
	//coincnumber = f1b;

	printf("Select time slot from %f seconds to %f seconds with %d seconds gap in between.\n", _acq_time_start, _acq_time_end, _acq_time_gap);
	printf("Total acquisition time is %f seconds.\n", time_tag);
	printf("%d events selected from %d total events.\n", event_ptr, num_coincidence);
	cout << "\n ss " <<_num_ss_event << "\n";
	cout << "\n oo =" << _num_ii_event << "\n";
	cout << "\n os =" <<_num_is_event << "\n";
	

	
	//int tot = 0;
	//tot = r1b + r2b + f1b + f2b + r1r1 + r1r2 + r2r2 + f1r1 + f1r2 + f2r1 + f2r2 + bb + f1f2 + f1f1;
	//cout << " total difference = " << num_coincidence - tot << "\n";



	free(compact_coincidence_data);

	return 1;

}

int
PET_data::_create_full_coincidence_LST_data_norm(PET_geometry &geometry, float &time, int &total_prompt_event, PET_coincidence_type type, int &lower1, int &upper1, int &lower2, int &upper2, int* &det1, int* &det2){
	int crystal_index_1, crystal_index_2;
	//double time_tag;
	int i, j;
	int event_ptr;
	int num_crystals;
	int num_scanner_crystals;
	int num_insert_crystals;
	int count = 0;

	num_crystals = geometry.get_num_detector_crystals();

	num_scanner_crystals = NUM_SCANNER_CRYSTALS;
	num_insert_crystals = NUM_INSERT_CRYSTALS;

	if (num_crystals != num_scanner_crystals + num_insert_crystals){
		printf("\n\n\n .......num_crystals does not equal num_scaner_crystals + num_insert_crystals \n\n\n");
		return 0;
	}
	
	
	//Original
	if (type == SS){
		event_ptr = 0;
        for (i = lower1; i < upper1; i++){
			for (j = lower2; j < upper2; j++){
				//if (lower1 != lower2 && upper1 != upper2){
					crystal_index_1 = i;
					crystal_index_2 = j;
					
				    count = sqrt( det1[(i - 60800)]*det2[(j - 93568)]);
					//cout << "\n" << count; //hardcoded needs to be changed for oo
					//if(count < 0 || count > 20000) cout << "\n" << count;

					PET_LST_event current_event(crystal_index_1, crystal_index_2, 0.0f, time, count);
					PET_LST_event_list.push_back(current_event);
					event_ptr++;
				//}
			}//cout << "\n" << i;
		}

		total_prompt_event = event_ptr;
	}
	else if (type == IS){
		event_ptr = 0;
		for (i = lower1; i < upper1; i++){
			for (j = lower2; j < upper2; j++){
				crystal_index_1 = i;
				crystal_index_2 = j;
				//Select only scanner crystals in the lower half of the ring.
				if (crystal_index_2 < num_scanner_crystals / 2){
					PET_LST_event current_event(crystal_index_1, crystal_index_2, 0.0f, time, count);
					PET_LST_event_list.push_back(current_event);
					event_ptr++;
				}
			}
		}
		total_prompt_event = event_ptr;
	}
	else if (type == II){
		event_ptr = 0;
		for (i = lower1; i < upper1; i++){
			for (j = lower2; j < upper2; j++){
				crystal_index_1 = i;
				crystal_index_2 = j;
				PET_LST_event current_event(crystal_index_1, crystal_index_2, 0.0f, time, count);
				PET_LST_event_list.push_back(current_event);
				event_ptr++;
			}
		}
		total_prompt_event = event_ptr;
	}
	else if (type == ALL){
		//All
		event_ptr = 0;
		for (i = 0; i < num_crystals; i++){
			for (j = i + 1; j < num_crystals; j++){
				crystal_index_1 = i;
				crystal_index_2 = j;
				PET_LST_event current_event(crystal_index_1, crystal_index_2, 0.0f, time, count);
				PET_LST_event_list.push_back(current_event);
				event_ptr++;
			}
		}
		total_prompt_event = event_ptr;
	}
	else{
		//All
		event_ptr = 0;
		for (i = 0; i < num_crystals; i++){
			for (j = i + 1; j < num_crystals; j++){
				crystal_index_1 = i;
				crystal_index_2 = j;
				PET_LST_event current_event(crystal_index_1, crystal_index_2, 0.0f, time, count);
				PET_LST_event_list.push_back(current_event);
				event_ptr++;
			}
		}
		total_prompt_event = event_ptr;
	}

	
	//printf("%d events created.\n", event_ptr);
	//cout << "\n here; ";
	
	
}

int
PET_data::_create_full_coincidence_LST_data_normblock(PET_geometry &geometry, float &time, int &total_prompt_event, PET_coincidence_type type, int &lower1, int &upper1, int &lower2, int &upper2, int** &det1){
	int crystal_index_1, crystal_index_2;
	//double time_tag;
	int i, j;
	int event_ptr;
	int num_crystals;
	int num_scanner_crystals;
	int num_insert_crystals;
	int count = 0;

	num_crystals = geometry.get_num_detector_crystals();

	num_scanner_crystals = NUM_SCANNER_CRYSTALS;
	num_insert_crystals = NUM_INSERT_CRYSTALS;

	if (num_crystals != num_scanner_crystals + num_insert_crystals){
		printf("\n\n\n .......num_crystals does not equal num_scaner_crystals + num_insert_crystals \n\n\n");
		return 0;
	}


	//Original
	if (type == SS){
		event_ptr = 0;
		int i1 = 0; int j1 = 0;
		for (i = lower1; i < upper1; i++){
			for (j = i+1; j < upper2; j++){
				//if (lower1 != lower2 && upper1 != upper2){
				crystal_index_1 = i;
				crystal_index_2 = j;
				i1 = int(i / 200);
				j1 = int(j / 200);
				count = det1[i1][j1]/40000;
				//cout << "\n" << count; //hardcoded needs to be changed for oo
				//if(count < 0 || count > 20000) cout << "\n" << count;

				PET_LST_event current_event(crystal_index_1, crystal_index_2, 0.0f, time, count);
				PET_LST_event_list.push_back(current_event);
				event_ptr++;
				//}
			}//cout << "\n" << i;
		}

		total_prompt_event = event_ptr;
	}
	else if (type == IS){
		event_ptr = 0;
		for (i = lower1; i < upper1; i++){
			for (j = lower2; j < upper2; j++){
				crystal_index_1 = i;
				crystal_index_2 = j;
				//Select only scanner crystals in the lower half of the ring.
				if (crystal_index_2 < num_scanner_crystals / 2){
					PET_LST_event current_event(crystal_index_1, crystal_index_2, 0.0f, time, count);
					PET_LST_event_list.push_back(current_event);
					event_ptr++;
				}
			}
		}
		total_prompt_event = event_ptr;
	}
	else if (type == II){
		event_ptr = 0;
		for (i = lower1; i < upper1; i++){
			for (j = lower2; j < upper2; j++){
				crystal_index_1 = i;
				crystal_index_2 = j;
				PET_LST_event current_event(crystal_index_1, crystal_index_2, 0.0f, time, count);
				PET_LST_event_list.push_back(current_event);
				event_ptr++;
			}
		}
		total_prompt_event = event_ptr;
	}
	else if (type == ALL){
		//All
		event_ptr = 0;
		for (i = 0; i < num_crystals; i++){
			for (j = i + 1; j < num_crystals; j++){
				crystal_index_1 = i;
				crystal_index_2 = j;
				PET_LST_event current_event(crystal_index_1, crystal_index_2, 0.0f, time, count);
				PET_LST_event_list.push_back(current_event);
				event_ptr++;
			}
		}
		total_prompt_event = event_ptr;
	}
	else{
		//All
		event_ptr = 0;
		for (i = 0; i < num_crystals; i++){
			for (j = i + 1; j < num_crystals; j++){
				crystal_index_1 = i;
				crystal_index_2 = j;
				PET_LST_event current_event(crystal_index_1, crystal_index_2, 0.0f, time, count);
				PET_LST_event_list.push_back(current_event);
				event_ptr++;
			}
		}
		total_prompt_event = event_ptr;
	}


	//printf("%d events created.\n", event_ptr);
	//cout << "\n here; ";


}

int
PET_data::_create_full_coincidence_LST_data(PET_geometry &geometry, float &time, int &total_prompt_event, PET_coincidence_type type, int &lower1, int &upper1, int &lower2, int &upper2){
	int crystal_index_1, crystal_index_2;
	//double time_tag;
	int i, j;
	int event_ptr;
	int num_crystals;
	int num_scanner_crystals;
	int num_insert_crystals;
	int count = 1;

	num_crystals = geometry.get_num_detector_crystals();

	num_scanner_crystals = NUM_SCANNER_CRYSTALS;
	num_insert_crystals = NUM_INSERT_CRYSTALS;

	if (num_crystals != num_scanner_crystals + num_insert_crystals){
		printf("\n\n\n .......num_crystals does not equal num_scaner_crystals + num_insert_crystals \n\n\n");
		return 0;
	}


	//Original
	if (type == SS){
		event_ptr = 0;
		if (lower1 != lower2 && upper1 != upper2){
			for (i = lower1; i < upper1; i++){
				for (j = lower2; j < upper2; j++){
					crystal_index_1 = i;
					crystal_index_2 = j;
					PET_LST_event current_event(crystal_index_1, crystal_index_2, 0.0f, time, count,0.0f);
					PET_LST_event_list.push_back(current_event);
					event_ptr++;
					
				}
			}
		}
		else{
			for (i = lower1; i < upper1; i++){
				for (j = i+1; j < upper2; j++){
					crystal_index_1 = i;
					crystal_index_2 = j;
					PET_LST_event current_event(crystal_index_1, crystal_index_2, 0.0f, time, count,0.0f);
					PET_LST_event_list.push_back(current_event);
					event_ptr++;
					
				}
			}
		}

		total_prompt_event = event_ptr;
	}
	else if (type == IS){
		event_ptr = 0;
		if (lower1 != lower2 && upper1 != upper2){
			for (i = lower1; i < upper1; i++){
				for (j = lower2; j < upper2; j++){
					crystal_index_1 = i;
					crystal_index_2 = j;
					PET_LST_event current_event(crystal_index_1, crystal_index_2, 0.0f, time, count,0.0f);
					PET_LST_event_list.push_back(current_event);
					event_ptr++;

				}
			}
		}
		else{
			for (i = lower1; i < upper1; i++){
				for (j = i + 1; j < upper2; j++){
					crystal_index_1 = i;
					crystal_index_2 = j;
					PET_LST_event current_event(crystal_index_1, crystal_index_2, 0.0f, time, count,0.0f);
					PET_LST_event_list.push_back(current_event);
					event_ptr++;

				}
			}
		}

		total_prompt_event = event_ptr;
	}
	else if (type == II){
		event_ptr = 0;
		if (lower1 != lower2 && upper1 != upper2){
			for (i = lower1; i < upper1; i++){
				for (j = lower2; j < upper2; j++){
					crystal_index_1 = i;
					crystal_index_2 = j;
					PET_LST_event current_event(crystal_index_1, crystal_index_2, 0.0f, time, count, 0.0f);
					PET_LST_event_list.push_back(current_event);
					event_ptr++;

				}
			}
		}
		else{
			for (i = lower1; i < upper1; i++){
				for (j = i + 1; j < upper2; j++){
					crystal_index_1 = i;
					crystal_index_2 = j;
					PET_LST_event current_event(crystal_index_1, crystal_index_2, 0.0f, time, count, 0.0f);
					PET_LST_event_list.push_back(current_event);
					event_ptr++;

				}
			}
		}

		total_prompt_event = event_ptr;
	}
	else if (type == ALL){
		//All
		event_ptr = 0;
		if (lower1 != lower2 && upper1 != upper2){
			for (i = lower1; i < upper1; i++){
				for (j = lower2; j < upper2; j++){
					crystal_index_1 = i;
					crystal_index_2 = j;
					PET_LST_event current_event(crystal_index_1, crystal_index_2, 0.0f, time, count,0.0f);
					PET_LST_event_list.push_back(current_event);
					event_ptr++;

				}
			}
		}
		else{
			for (i = lower1; i < upper1; i++){
				for (j = i + 1; j < upper2; j++){
					crystal_index_1 = i;
					crystal_index_2 = j;
					PET_LST_event current_event(crystal_index_1, crystal_index_2, 0.0f, time, count,0.0f);
					PET_LST_event_list.push_back(current_event);
					event_ptr++;

				}
			}
		}

		total_prompt_event = event_ptr;
	}
	else{
		//All
		event_ptr = 0;
		if (lower1 != lower2 && upper1 != upper2){
			for (i = lower1; i < upper1; i++){
				for (j = lower2; j < upper2; j++){
					crystal_index_1 = i;
					crystal_index_2 = j;
					PET_LST_event current_event(crystal_index_1, crystal_index_2, 0.0f, time, count,0.0f);
					PET_LST_event_list.push_back(current_event);
					event_ptr++;

				}
			}
		}
		else{
			for (i = lower1; i < upper1; i++){
				for (j = i + 1; j < upper2; j++){
					crystal_index_1 = i;
					crystal_index_2 = j;
					PET_LST_event current_event(crystal_index_1, crystal_index_2, 0.0f, time, count,0.0f);
					PET_LST_event_list.push_back(current_event);
					event_ptr++;

				}
			}
		}

		total_prompt_event = event_ptr;
	}


	printf("%d events created.\n", event_ptr);
}


int
PET_data::_create_min_coincidence_LST_data(PET_geometry &geometry, int &total_prompt_event, PET_coincidence_type type, int &lower1, int &upper1, int &lower2, int &upper2){
	int crystal_index_1, crystal_index_2;
	int i, j;
	int event_ptr;
	int num_crystals;
	int num_scanner_crystals;
	int num_insert_crystals;
	int count = 1;

	num_crystals = geometry.get_num_detector_crystals();

	num_scanner_crystals = NUM_SCANNER_CRYSTALS;
	num_insert_crystals = NUM_INSERT_CRYSTALS;

	if (num_crystals != num_scanner_crystals + num_insert_crystals){
		printf("\n\n\n .......num_crystals does not equal num_scaner_crystals + num_insert_crystals \n\n\n");
		return 0;
	}


	//Original
	if (type == SS){
		event_ptr = 0;
		if (lower1 != lower2 && upper1 != upper2){
			for (i = lower1; i < upper1; i++){
				for (j = lower2; j < upper2; j++){
					crystal_index_1 = i;
					crystal_index_2 = j;
					PET_LST_event current_event(crystal_index_1, crystal_index_2);
					PET_LST_event_list.push_back(current_event);
					event_ptr++;

				}
			}
		}
		else{
			for (i = lower1; i < upper1; i++){
				for (j = i + 1; j < upper2; j++){
					crystal_index_1 = i;
					crystal_index_2 = j;
					PET_LST_event current_event(crystal_index_1, crystal_index_2);
					PET_LST_event_list.push_back(current_event);
					event_ptr++;

				}
			}
		}

		total_prompt_event = event_ptr;
	}
	else if (type == IS){
		event_ptr = 0;
		if (lower1 != lower2 && upper1 != upper2){
			for (i = lower1; i < upper1; i++){
				for (j = lower2; j < upper2; j++){
					crystal_index_1 = i;
					crystal_index_2 = j;
					PET_LST_event current_event(crystal_index_1, crystal_index_2);
					PET_LST_event_list.push_back(current_event);
					event_ptr++;

				}
			}
		}
		else{
			for (i = lower1; i < upper1; i++){
				for (j = i + 1; j < upper2; j++){
					crystal_index_1 = i;
					crystal_index_2 = j;
					PET_LST_event current_event(crystal_index_1, crystal_index_2);
					PET_LST_event_list.push_back(current_event);
					event_ptr++;

				}
			}
		}

		total_prompt_event = event_ptr;
	}
	else if (type == II){
		event_ptr = 0;
		if (lower1 != lower2 && upper1 != upper2){
			for (i = lower1; i < upper1; i++){
				for (j = lower2; j < upper2; j++){
					crystal_index_1 = i;
					crystal_index_2 = j;
					PET_LST_event current_event(crystal_index_1, crystal_index_2);
					PET_LST_event_list.push_back(current_event);
					event_ptr++;

				}
			}
		}
		else{
			for (i = lower1; i < upper1; i++){
				for (j = i + 1; j < upper2; j++){
					crystal_index_1 = i;
					crystal_index_2 = j;
					PET_LST_event current_event(crystal_index_1, crystal_index_2);
					PET_LST_event_list.push_back(current_event);
					event_ptr++;

				}
			}
		}

		total_prompt_event = event_ptr;
	}
	else if (type == ALL){
		//All
		event_ptr = 0;
		if (lower1 != lower2 && upper1 != upper2){
			for (i = lower1; i < upper1; i++){
				for (j = lower2; j < upper2; j++){
					crystal_index_1 = i;
					crystal_index_2 = j;
					PET_LST_event current_event(crystal_index_1, crystal_index_2);
					PET_LST_event_list.push_back(current_event);
					event_ptr++;

				}
			}
		}
		else{
			for (i = lower1; i < upper1; i++){
				for (j = i + 1; j < upper2; j++){
					crystal_index_1 = i;
					crystal_index_2 = j;
					PET_LST_event current_event(crystal_index_1, crystal_index_2);
					PET_LST_event_list.push_back(current_event);
					event_ptr++;

				}
			}
		}

		total_prompt_event = event_ptr;
	}
	else{
		//All
		event_ptr = 0;
		if (lower1 != lower2 && upper1 != upper2){
			for (i = lower1; i < upper1; i++){
				for (j = lower2; j < upper2; j++){
					crystal_index_1 = i;
					crystal_index_2 = j;
					PET_LST_event current_event(crystal_index_1, crystal_index_2);
					PET_LST_event_list.push_back(current_event);
					event_ptr++;

				}
			}
		}
		else{
			for (i = lower1; i < upper1; i++){
				for (j = i + 1; j < upper2; j++){
					crystal_index_1 = i;
					crystal_index_2 = j;
					PET_LST_event current_event(crystal_index_1, crystal_index_2);
					PET_LST_event_list.push_back(current_event);
					event_ptr++;

				}
			}
		}

		total_prompt_event = event_ptr;
	}


	printf("%d events created.\n", event_ptr);
}







int
PET_data::GetDataCount(PET_coincidence_type type){
	if (type == SS){
		return _num_ss_event;
	}
	else if (type == IS){
		return _num_is_event;
	}
	else if (type == II){
		return _num_ii_event;
	}
	else if (type == ALL){
		return _num_event;
	}
	else{
		return _num_event;
	}
}







int
PET_data::_read_Inveon_LST_data(const std::string inputfile_name, int &total_prompt_event, PET_data_type mode){
	
	PET_LST_event_type event_mode;
		
	int event_ptr;
	int verbose = 1;
	int SinglesBuffLen = 10;

	unsigned int next_pos = 0, src_pos = 0, gc_counter = 0, i, type = 0, header_flag = 0, detector_id = 0, board_id = 0, pkt_type = 0, event_type = 0;
	unsigned sub_typ1 = 0, sub_typ2 = 0;

	int prompt_flag = 0;
	int prompt_cnt = 0;
	int time_tag = 0;
	FILE *inputfile;
	struct stat64 file_status;
	int CID1, CID2, DID1, DID2, SID1, SID2, TDiff;
	int CoinEvtCnt = 0, SingleEvtCnt = 0, Undef1Cnt = 0, CntTagCnt = 0, Undef2Cnt = 0, Undef3Cnt = 0, ExtPakCnt = 0, McTagCnt = 0;
	int IOSBTCnt[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

	int selected_event;

	// Allocate memory
	in_buf = (unsigned char*)calloc(BUFSIZE, sizeof(unsigned char));
	
	// Initialize variables
	OutOfSync = 0;
	TotalBytes = 0;
	FileSize = 0;
	cur_buf_size = 0;

	if (mode == DELAYED){
		selected_event = DELAYED_EVENT;
		event_mode = eDELAYED;
	}
	else if (mode == PROMPT){
		selected_event = PROMPT_EVENT;
		event_mode = ePROMPT;
	}
	else{
		printf("Only Prompt and Delayed events are allowed for ");
		return 0;
	}

	//printf("Sorting : %s \n", selected_event == DELAYED_EVENT ? "Delayed" : "Prompt");

	printf("Opening list mode file, %s \n", inputfile_name.c_str());
	if ((inputfile = fopen(inputfile_name.c_str(), "rb")) == NULL) {
		printf("Error: Could not read input file. \n");
		return 0;
	}
	if (stat64(inputfile_name.c_str(), &file_status) < 0) {
		printf("Read file error. \n");
		return 0;
	}

	FileSize = file_status.st_size;
	printf("input filesize:%ld\n", FileSize);
	
	read_to_buffer(inputfile);
	
	gc_counter = synchronize(&src_pos);
	
	event_ptr = 0;
	// Loop through the data
	while (1) {
		if (graycode[gc_counter] == (in_buf[src_pos + 5] & 0xf0)) {
			type = in_buf[src_pos + 5] & 0x0f;
			gc_counter = (gc_counter + 1) % 8;
			switch (type) {

			case  0x0e:	// extended event
				ExtPakCnt++;
				header_flag = in_buf[src_pos + 3] & 0x80;
				pkt_type = (in_buf[src_pos + 3] & 0x78) >> 3;
				if (pkt_type == selected_event && header_flag) {  // timing mode data
					if ((src_pos + 6) >BUFSIZE) break;
					header_flag = in_buf[src_pos + 9] & 0x80;
					event_type = in_buf[src_pos + 11] & 0x0f;
					sub_typ1 = (in_buf[src_pos + 4] & 0xf0) >> 4;
					sub_typ2 = (in_buf[src_pos + 10] & 0xf0) >> 4;
					if (header_flag || sub_typ1 != 0x01 || sub_typ2 != 0x01) break;
					CoinEvtCnt++;
					prompt_flag = in_buf[src_pos + 3] & 0x08;
					if (prompt_flag == 0) prompt_cnt++;

					CID1 = in_buf[src_pos] | ((in_buf[src_pos + 1] & 0x07) << 8);
					CID2 = in_buf[src_pos + 6] | ((in_buf[src_pos + 7] & 0x07) << 8);
					DID1 = (in_buf[src_pos + 1] & 0x18) >> 3;
					DID2 = (in_buf[src_pos + 7] & 0x18) >> 3;
					SID1 = (in_buf[src_pos + 1] & 0xE0) >> 5 | ((in_buf[src_pos + 2] & 0x1f) << 3);
					SID2 = (in_buf[src_pos + 7] & 0xE0) >> 5 | ((in_buf[src_pos + 8] & 0x1f) << 3);
					TDiff = (in_buf[src_pos + 8] & 0xE0) >> 5 | ((in_buf[src_pos + 9] & 0x7f) << 3);


					if (MAX_NUM_LST_EVENT > event_ptr && time_tag >= 5000 * _acq_time_start && time_tag <= 5000 * _acq_time_end){
						
						PET_LST_event current_event(_get_crystal_id(SID1, DID1, CID1), _get_crystal_id(SID2, DID2, CID2), 0.0f, (float)(time_tag / 5000.0), 1);
						
						PET_LST_event_list.push_back(current_event);

						event_ptr++;
					}
				}
				break;

			case  0x0a:   // Counter Tag (Time Mark, singles, etc)
				pkt_type = in_buf[src_pos + 4] & 0x0f;
				if (pkt_type == 0x00)
				{
					//time_tag = (in_buf[src_pos]) | (in_buf[src_pos+1]<<8) | (in_buf[src_pos+2]<<16) | (in_buf[src_pos+3]<<24);
					//fwrite(&time_tag,1,sizeof(int),outputtimetag);
					time_tag++;
				}
				CntTagCnt++;
				break;
			case  0x0b://process as undefined2
				Undef2Cnt++;
				break;
			case 0x0c: // process as ios tags
				pkt_type = (in_buf[src_pos + 4] & 0xf0) >> 4;
				IOSBTCnt[pkt_type]++;
				break;
			case 0x0d:// process as undefined3
				Undef3Cnt++;
				break;
			case 0x0f: // process as micro controller tags
				McTagCnt++;
				break;
			case 0x00: // process as coincidence event
			case 0x01:
			case 0x02:
			case 0x03:
			case 0x04:
			case 0x05:
			case 0x06:
			case 0x07:
				CoinEvtCnt++;
				break;
			case 0x08:
				SingleEvtCnt++;
				break;
			case 0x09:
				Undef1Cnt++;
				break;

			}
			src_pos += 6;

		}
		else {
			// Syncronize based on gray code
			OutOfSync++;
			gc_counter = synchronize(&src_pos);
		}

		/* if source position moves past buffer size,
		* read in more data.
		*/
		if ((int)src_pos >= cur_buf_size) {
			// show the progress bar
			loadBar(TotalBytes, FileSize, 100, 20);
			if (TotalBytes >= FileSize) {
				break;
			}
			else {
				next_pos = (next_pos + 1) % SinglesBuffLen;

				/* if there is enough data to fill the buffer,
				* read till buffer is full.
				* else, read remaining bytes in file.
				*/
				read_to_buffer(inputfile);

				src_pos = 0;
			}

		} // (src_pos >= cur_buf_size)
	} // while loop

	fclose(inputfile);

	total_prompt_event = event_ptr;

	if (verbose) {
		printf("CoinEvtCnt = %d \nSingleEvtCnt = %d \nUndef1Cnt = %d \nCntTagCnt = %d \n", CoinEvtCnt, SingleEvtCnt, Undef1Cnt, CntTagCnt);
		printf("Undef2Cnt = %d \nUndef3Cnt = %d \nExtPakCnt = %d \nMcTagCnt = %d \nOutOfSync = %d \n", Undef2Cnt, Undef3Cnt, ExtPakCnt, McTagCnt, OutOfSync);
		printf("\nPrompt = %d \n", prompt_cnt);
		printf("IOSBT : ");
		for (i = 0; i<16; i++)
			printf("%d \t", IOSBTCnt[i]);
		printf("\n");
	}

	printf("Select time slot from %f seconds to %f seconds.\n", (float)_acq_time_start, (float)_acq_time_end);
	printf("Total acquisition time is %f seconds.\n", time_tag*0.0002);
	printf("%d events selected from %d total events.\n", event_ptr, prompt_cnt);

	// Free memory
	free(in_buf);

	return 1;
}


inline void
PET_data::loadBar(long int x, long int n, int r, int w)
{
	/* Code to display a progress bar.
	* From: http://www.rosshemsley.co.uk/2011/02/creating-a-progress-bar-in-c-or-any-other-console-app/
	*/
	// Only update r times.
	// printf(" x: %d n: %d \n", x, n);
	// if ( x % (n/r) != 0 ) return;

	int i;
	// Calculuate the ratio of complete-to-incomplete.
	float ratio = x / (float)n;
	int   c = (int)(ratio * w);

	// Show the percentage complete.
	printf("%3d%% [", (int)(ratio * 100));

	// Show the load bar.
	for (i = 0; i<c; i++)
		printf("=");

	for (i = c; i<w; i++)
		printf(" ");

	// ANSI Control codes to go back to the
	// previous line and clear it.
	printf("]\n\033[F\033[J");
}


int
PET_data::synchronize(unsigned int *src_pos_p) {
	/* Synchronizes the reading pointer, based on
	* known sync points in data stream.
	*/
	int i, j;
	unsigned int offset, gc_counter = 0;

	// Syncronize based on gray code
	OutOfSync++;
	offset = 0;
	for (i = 0; i < 6; i++) {	// scan byte-by-byte
		for (j = 0; j < 8; j++) {
			if (((in_buf[*src_pos_p + i] & 0xf0) == graycode[j]) &&
				((in_buf[*src_pos_p + i + 6] & 0xf0) == graycode[j + 1]) &&
				((in_buf[*src_pos_p + i + 12] & 0xf0) == graycode[j + 2])) {
				offset = i;
			}
		}
	}
	// printf("offset: %d", offset);

	//if (offset ==0) offset = 1;
	*src_pos_p = *src_pos_p + offset + 1;
	for (i = 0; i < 8; i++)
		if (graycode[i] == (in_buf[*src_pos_p + 5] & 0xf0)) {
			gc_counter = i;
			break;
		}
	// printf("gc_counter: %d", gc_counter);
	return gc_counter;

}


long int
PET_data::read_to_buffer(FILE *inputfile) {
	/* if there is enough data to fill the buffer,
	* read till buffer is full.
	* else, read remaining bytes in file.
	*/

	if (FileSize>(BUFSIZE + TotalBytes)) {
		cur_buf_size = BUFSIZE;
		fread(in_buf, 1, BUFSIZE, inputfile);
		TotalBytes += BUFSIZE;
	}
	else {
		cur_buf_size = FileSize - TotalBytes;
		fread(in_buf, 1, cur_buf_size, inputfile);
		TotalBytes = FileSize;
	}

	return TotalBytes;
}


int
PET_data::_get_crystal_id(int sx, int dx, int cx){
	return _get_crystal_id_MicroInsert(sx, dx, cx);
}

int
PET_data::_get_crystal_id_PlantPET(int sx, int dx, int cx){
	int temp;
	int sector, block_index, in_block, unit_index, in_unit;
	int crystal_id;
	int CableLUT[32] = { 0, 1, 31, 31, 2, 3, 4, 31, 5, 6, 7, 31, 8, 9, 31, 31, 10, 11, 12, 31, 13, 14, 15, 31, 16, 17, 18, 31, 19, 20, 31, 31 };
	
	//find the CORRECT cable channel according to cable LUT
	if(sx >= 8){
		temp = (sx - 8) * 4 + dx;
		temp = CableLUT[temp];
		sx = temp / 4 + 8;
		dx = temp % 4;
	}

	
	if (sx<8){//belongs to Inveon
		block_index = 8 * dx + sx;
		unit_index = 20 * block_index + (19 - cx / 20);
		in_unit = cx % 20;
		crystal_id = ((20 * unit_index + in_unit) + 5376);
	}
	else{//belongs to R4
		sector = 4 * (sx - 8) + dx;
		if (cx<128){//A,B
			in_block = cx % 8;
			block_index = ((cx / 8) % 2) * 21 + sector;
			in_unit = (cx / 8) / 2;//consider reverse this order
		}
		else{//C,D
			in_block = cx % 8;
			block_index = ((cx / 8) % 2 + 2) * 21 + sector;
			in_unit = 7 - (cx / 8 - 16) / 2;
		}
		unit_index = 8 * block_index + in_block;
		crystal_id = (8 * unit_index + in_unit);
	}
	return crystal_id;
}

int
PET_data::_get_crystal_id_MicroInsert(int sx, int dx, int cx){

	int sector, block_index, in_block, unit_index, in_unit;
	int crystal_id;

	if (cx >= 512 && cx<912) {
		cx = cx - 512;
	}
	else{
		//sx = sx + 16;
	}
	
	if (sx<16){//belongs to insert
		sector = (sx / 4) * 3 + sx % 4;

		//in_block = cx/20;
		//in_unit = 19 - cx%20;

		in_unit = 19 - cx / 20;
		in_block = 19 - cx % 20;
		dx = 3 - dx;
		crystal_id = (sector * 4 + dx) * 400 + 20 * in_block + in_unit;
		if (crystal_id >= 19200)
			printf("%d\n", crystal_id);
	}
	else{//belongs to scanner
		in_unit = 19 - cx % 20;
		in_block = cx / 20;
		sector = sx - 16;
		sector = 15 - sector;
		crystal_id = (sector * 4 + dx) * 400 + 20 * in_block + in_unit + 19200;
		if (crystal_id<19200)
			printf("%d\n", crystal_id);
	}
	return crystal_id;
}