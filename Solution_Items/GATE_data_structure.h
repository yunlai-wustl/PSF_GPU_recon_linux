#ifndef GATE_DATA_STRUCTURE_H
#define GATE_DATA_STRUCTURE_H

#pragma pack(push)

typedef struct{
	double	x;
	double	y;
	double	z;
}pos;

typedef struct{
	float x;
	float y;
	float z;
}pos_32;

typedef struct{
	int system;
	int sector;
	int module;
	int submodule;
	int crystal;
	int layer;
}volume_id;

#pragma pack(4)
typedef struct{
	//132 bytes
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
}GATE_singles_event;

#pragma pack(4)
typedef struct{
	//first single
	int Run_1;
	int eventID_1;
	int sourceID_1;
	pos source_1;
	double	time_1; //time in sec
	double	energy_1;	//energy in MeV
	pos		single_position_1;
	volume_id single_ID_1;
	int compton_scatter_in_Phantom_1;
	int compton_scatter_in_Detector_1;
	int rayleigh_scatter_in_Phantom_1;
	int rayleigh_scatter_in_Detector_1;
	double scanner_axial_1;
	double scanner_angular_1;

	//second single
	int Run_2;
	int eventID_2;
	int sourceID_2;
	pos source_2;
	double	time_2; //time in sec
	double	energy_2;	//energy in MeV
	pos		single_position_2;
	volume_id single_ID_2;
	int compton_scatter_in_Phantom_2;
	int compton_scatter_in_Detector_2;
	int rayleigh_scatter_in_Phantom_2;
	int rayleigh_scatter_in_Detector_2;
	double scanner_axial_2;
	double scanner_angular_2;

}GATE_coincidence_event;

#pragma pack(4)
typedef struct{
	int crystal_index_1;
	int crystal_index_2;
	float time_1;//time of event 1
	double diff_time;//t2-t1
	float bed_position; // in mm and scanner coordinate (scanner center is 0)
}mini_coinc_event;

#pragma pack(4)
typedef struct{
	int crystal_index_1;
	int crystal_index_2;
	float time_1;//time of event 1
	double diff_time;//t2-t1
	float bed_position; //in mm
	float scatter_coeff; // in number of decay/LOR over entire scan
}mini_coinc_event_scatter;

#pragma pack(4)
typedef struct{
	int event_id;
	//volume_id single_ID_1;
	//volume_id single_ID_2;
	int crystal_index_1;
	int crystal_index_2;
	double time_1;//time of event 1
	double diff_time;//t2-t1
	pos source_pos;
	float norm_coeff;
}mini_coinc_event_norm;


#pragma pack(4)
typedef struct{
	int crystal_index_1;
	int crystal_index_2;
	float time_1;//time of event 1
	double diff_time;//t2-t1
}tof_coinc_event;


#pragma pack(4)
typedef struct{
	int crystal_index_1;
	int crystal_index_2;
	float time_1;//time of event 1
	double diff_time;//t2-t1
	float bed_position; //bed position, in mm
}tof_coinc_event_cbm;


#pragma pack(4)
typedef struct{
	int crystal_index_1;
	int crystal_index_2;
	float source_x; // in mm
	float source_y;
	float source_z;
}decay_event;


#pragma pack(4)
typedef struct{
	int crystal_index_1;
	int crystal_index_2;
	float time_1;//time of event 1
}nontof_coinc_event;

#pragma pack(pop)


#endif