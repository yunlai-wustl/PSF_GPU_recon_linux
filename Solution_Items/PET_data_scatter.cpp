#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdlib.h>
#include <unistd.h>

#include "PET_DATA_scatter.h"
#include <math.h>


PET_data_scatter::PET_data_scatter(){
	PET_LST_event_list.reserve(MAX_NUM_LST_EVENT);
	//PET_LST_events_pos_list.reserve(MAX_NUM_LST_EVENT);
	//PET_LST_events_pos_list_1.reserve(MAX_NUM_LST_EVENT);
	//PET_LST_events_pos_list_2.reserve(MAX_NUM_LST_EVENT);
}

PET_data_scatter::PET_data_scatter(int length){
	PET_LST_event_list.reserve(length);
}


PET_data_scatter::PET_data_scatter(const std::string config_file_name, PET_data_type type){
	Setup(config_file_name, type);

	PET_LST_event_list.reserve(MAX_NUM_LST_EVENT);
	//PET_LST_events_pos_list.reserve(MAX_NUM_LST_EVENT);
	//PET_LST_events_pos_list_1.reserve(MAX_NUM_LST_EVENT);
	//PET_LST_events_pos_list_2.reserve(MAX_NUM_LST_EVENT);


}

PET_data_scatter::PET_data_scatter(PET_data_scatter &data_from, float time_start, float time_end, PET_coincidence_type t, int time_gap){
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
				//PET_LST_events_pos_list_2.push_back(data_from.PET_LST_events_pos_list_2.at(i));
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

PET_data_scatter::PET_data_scatter(PET_data_scatter &data_from, int num_subsets, int subset){
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


PET_data_scatter::~PET_data_scatter(){

}


const unsigned char
PET_data_scatter::graycode[11] = { 0x00, 0x10, 0x30, 0x20, 0x60, 0x70, 0x50, 0x40, 0x00, 0x10, 0x30 };



void
PET_data_scatter::add_data(PET_data_scatter &data_from, float time_start, float time_end, PET_coincidence_type t){
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
PET_data_scatter::GetDataListLength(){
	return (int)PET_LST_event_list.size();
}


void
PET_data_scatter::Setup(const std::string config_file_name, PET_data_type type){
	Config config(config_file_name);
	config.GetValue<float>("Time start", _acq_time_start);
	config.GetValue<float>("Time stop", _acq_time_end);
	config.GetValue<int>("Time gap", _acq_time_gap);
	config.GetValue<string>("Data source", _data_source);

	_acq_time_length = _acq_time_end - _acq_time_start;
	if (_acq_time_length < 0.001){
		printf("Time stop must be greater than Time start.\n");
		throw 1;
	}
	else{
		//printf("%s data selected from %f to %f, a total of %f seconds.\n", _data_source.c_str(), _acq_time_start, _acq_time_end, _acq_time_length);
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
PET_data_scatter::ReadFromFile(const std::string filename, PET_geometry geometry){

	
	if (_source == COMPACT_COINCIDENCE){
		_read_compact_coincidence_LST_data(filename, _num_event, _data_type, geometry);
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
PET_data_scatter::_read_compact_coincidence_LST_data(const std::string inputfile_name, int &total_prompt_event, PET_data_type mode, PET_geometry &detector){
	PET_LST_event_type event_mode;
	int event_ptr;
	mini_coinc_event_scatter *compact_coincidence_data;
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

	printf("Size of mini_coinc_event structure is %d\n", (int)sizeof(mini_coinc_event_scatter));
	printf("Following coincidence files are being processed:\n");
	num_coincidence = 0;
	size = file_status.st_size;
	int size_per_event = (int)sizeof(mini_coinc_event_scatter);
	size /= size_per_event;
	if (size >= INT_MAX){
		printf("File \"%s\" length is tooooooooooo long. This program only supports coincidence events fewer than %d. Current file has %lld. Program stopped\n", inputfile_name.c_str(), INT_MAX, size);
		return 0;
	}
	num_coincidence = (int)size;
	printf("%d coincidence events in file: %s\n", num_coincidence, inputfile_name.c_str());

	//Coincidence data memory
	compact_coincidence_data = (mini_coinc_event_scatter*)calloc(num_coincidence,sizeof(mini_coinc_event_scatter));

	//Read in coincidence data
	printf("Opening list mode file, %s \n", inputfile_name.c_str());
	if ((inputfile = fopen(inputfile_name.c_str(), "rb")) == NULL) {
		printf("Error: Could not read input file. \n");
		return 0;
	}

	t = fread(compact_coincidence_data, sizeof(mini_coinc_event_scatter), num_coincidence, inputfile);
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

	box s, d;
	double maxtime = 0;
	
	cout << "\n " << compact_coincidence_data[num_coincidence - 1].crystal_index_1 << " " << compact_coincidence_data[num_coincidence - 1].crystal_index_2 << " " << compact_coincidence_data[num_coincidence - 1].time_1 << " " << compact_coincidence_data[num_coincidence - 1].diff_time<<"\n";
	//cout << "\n Time for last event: " << compact_coincidence_data[num_coincidence - 1].time_1 << "\n";
	for (i = 0; i<num_coincidence; i++){
		time_tag = compact_coincidence_data[i].time_1;
		maxtime = max(maxtime, time_tag);
		
		if (MAX_NUM_LST_EVENT > event_ptr && time_tag >= _acq_time_start && time_tag <= _acq_time_end && (int(time_tag) % _acq_time_gap) < 1){


			PET_LST_event_scatter current_event(compact_coincidence_data[i].crystal_index_1, compact_coincidence_data[i].crystal_index_2, (float)(1.5e11 * compact_coincidence_data[i].diff_time), (float)(compact_coincidence_data[i].time_1), compact_coincidence_data[i].bed_position, 1, compact_coincidence_data[i].scatter_coeff);

			//printf("%d %d\n", compact_coincidence_data[i].crystal_index_1, compact_coincidence_data[i].crystal_index_2);
			PET_LST_event_list.push_back(current_event);

			//PET_LST_events_pos_list.push_back(compact_coincidence_data[i].source_pos);

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
	
	printf("Select time slot from %f seconds to %f seconds with %d seconds gap in between.\n", _acq_time_start, _acq_time_end, _acq_time_gap);
	printf("Total acquisition time is %f seconds and max time is %f seconds.\n", time_tag, maxtime);
	printf("%d events selected from %d total events.\n", event_ptr, num_coincidence);
	cout << "\n ss " << _num_ss_event << "\n";
	cout << "\n oo =" << _num_ii_event << "\n";
	cout << "\n os =" << _num_is_event << "\n";
	



	
	free(compact_coincidence_data);

	return 1;

}


int
PET_data_scatter::GetDataCount(PET_coincidence_type type){
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



inline void
PET_data_scatter::loadBar(long int x, long int n, int r, int w)
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
PET_data_scatter::synchronize(unsigned int *src_pos_p) {
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
PET_data_scatter::read_to_buffer(FILE *inputfile) {
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


