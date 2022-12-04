/*
#define __REDIRECT
#define __USE_LARGEFILE64
#define __USE_FILE_OFFSET64
#define _LARGEFILE_SOURCE
#define _LARGEFILE64_SOURCE
*/
#define MAX_SINGLES_IN_RUN 4000000
#define MAX_NUM_FILE 10

#define GATE_COINCIDENCE_ENTRY_LENGTH 264
#define GATE_SINGLE_ENTRY_LENGTH 132


struct pos {
	double	x;
	double	y;
	double	z;
};


struct volume_id {
	int system;
	int sector;
	int module;
	int submodule;
	int crystal;
	int layer;
};

#pragma pack(4)
struct singles_event {
	int Run;
	int eventID;
	int sourceID;
	pos source;
	volume_id single_ID;
	double time; //time in sec
	double energy;	//energy in MeV
	pos single_position;
	int compton_scatter_in_Phantom;
	int compton_scatter_in_Detector;
	int rayleigh_scatter_in_Phantom;
	int rayleigh_scatter_in_Detector;
	char name_compton_scatter_Phantom[8];
	char name_rayleigh_scatter_Phantom[8];
};

#pragma pack(4)
struct gate_coinc_event {
	int Run_1;
	int eventID_1;
	int sourceID_1;
	pos source_1;
	double time_1; //time in sec
	double energy_1;	//energy in MeV
	pos single_position_1;
	volume_id single_ID_1;
	int compton_scatter_in_Phantom_1;
	int compton_scatter_in_Detector_1;
	int rayleigh_scatter_in_Phantom_1;
	int rayleigh_scatter_in_Detector_1;
	//char name_compton_scatter_Phantom_1[8];
	//char name_rayleigh_scatter_Phantom_1[8];
	double scanner_axial_pos_1;
	double scanner_angular_pos_1;

	int Run_2;
	int eventID_2;
	int sourceID_2;
	pos source_2;
	double time_2; //time in sec
	double energy_2;	//energy in MeV
	pos single_position_2;
	volume_id single_ID_2;
	int compton_scatter_in_Phantom_2;
	int compton_scatter_in_Detector_2;
	int rayleigh_scatter_in_Phantom_2;
	int rayleigh_scatter_in_Detector_2;
	//char name_compton_scatter_Phantom_1[8];
	//char name_rayleigh_scatter_Phantom_1[8];
	double scanner_axial_pos_2;
	double scanner_angular_pos_2;
};

#pragma pack(4)
struct coinc_event{
	//first single
	int Run_1;
	int eventID_1;
	int sourceID_1;
	pos source_1;
	double	time_1; //time in sec
	double	energy_1;	//energy in MeV
	pos		single_position_1;
	volume_id single_ID_1;

	//second single
	int Run_2;
	int eventID_2;
	int sourceID_2;
	pos source_2;
	double	time_2; //time in sec
	double	energy_2;	//energy in MeV
	pos		single_position_2;
	volume_id single_ID_2;
};


#pragma pack(4)
struct mini_coinc_event{
	int event_id;
	//volume_id single_ID_1;
	//volume_id single_ID_2;
	int crystal_index_1;
	int crystal_index_2;
	double time_1;//time of event 1
	double diff_time;//t2-t1
	pos source_pos;
};


int coinc_sorter_2singles_by_time_window(singles_event **data, int *length, int num_file, int run_id, mini_coinc_event **coinc_data);
int coinc_sorter_2singles_by_event_ID(singles_event **data, int *length, int num_file, int run_id, mini_coinc_event **coinc_data);
int conversion_gate_list_mode(gate_coinc_event *data, int length, mini_coinc_event **coinc_data);

int record_mini_coinc_from_gate_coincidence(gate_coinc_event *d_ptr, mini_coinc_event *coinc_ptr);
int record_mini_coinc(singles_event *d_ptr1, singles_event *d_ptr2, mini_coinc_event *coinc_ptr);

int encode_crystal_plantPET(volume_id crystal_id);
int encode_crystal_pocPET(volume_id crystal_id);
int encode_crystal_pocPET002(volume_id crystal_id);
int encode_crystal_mCT(volume_id crystal_id);
