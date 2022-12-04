#ifndef PET_DATA_H
#define PET_DATA_H

#include <string>
#include <array>
#include <vector>

#include "global.h"
#include "config.h"
#include "PET_LST_event.h"
#include "PET_geometry.h"


#define BUFSIZE 6*1024*1024
#define PROMPT_EVENT 0x02
#define DELAYED_EVENT 0x03


class PET_data{
public:
	PET_data();
	PET_data(int length);
	PET_data(const std::string config_file_name, PET_data_type type);
	PET_data(PET_data &data_from, float time_start, float time_end, PET_coincidence_type t, int time_gap);
	PET_data(PET_data &data_from, int num_subsets, int subset);
	~PET_data();
	int GetDataListLength();
	void add_data(PET_data &data_from, float time_start, float time_end, PET_coincidence_type t);
	void Setup(const std::string config_file_name, PET_data_type type);
	void ReadFromFile(const std::string filename, PET_geometry geometry);
	void CreateFullCoincidenceData(PET_geometry geometry, PET_coincidence_type type,int lower1, int upper1,int lower2, int upper2, float time);
	void CreateFullCoincidenceData(PET_geometry geometry, PET_coincidence_type type, int lower1, int upper1, int lower2, int upper2);
	void CreateFullCoincidenceData_norm(PET_geometry geometry, PET_coincidence_type type, int lower1, int upper1, int lower2, int upper2, float time, int* det1, int* det2);
	void CreateFullCoincidenceData_normblock(PET_geometry geometry, PET_coincidence_type type, int lower1, int upper1, int lower2, int upper2, float time, int** det1);
	int GetDataCount(PET_coincidence_type type);
	vector<PET_LST_event> PET_LST_event_list;

	vector<pos> PET_LST_events_pos_list;
	

	
	vector<pos_32> PET_LST_events_pos_list_1;
	vector<pos_32> PET_LST_events_pos_list_2;
	PET_data_source _source;
	PET_protocol_type _protocol;

private:
	PET_coincidence_type _coincidence_type;
	PET_data_type _data_type;
	int _num_event;
	int _num_ss_event;
	int _num_is_event;
	int _num_ii_event;
	

	float _acq_time_start;
	float _acq_time_end;
	float _acq_time_length;
	int _acq_time_gap;
	string _data_source;
	string _scan_protocol;

	long int TotalBytes;
	long int FileSize;
	int cur_buf_size;
	unsigned OutOfSync;
	unsigned char *in_buf;

	static const unsigned char graycode[11];

	int _read_XYZ_coincidence_LST_data(const std::string inputfile_name, int &total_prompt_event, PET_data_type mode);
	int _read_Inveon_LST_data(const std::string inputfile_name, int &total_prompt_event, PET_data_type mode);
	int _read_compact_coincidence_LST_data(const std::string inputfile_name, int &total_prompt_event, PET_data_type mode, PET_geometry &geometry);
	int _read_non_TOF_LST_data(const std::string inputfile_name, int &total_prompt_event, PET_data_type mode);
	int _read_TOF_LST_data(const std::string inputfile_name, int &total_prompt_event, PET_data_type mode);
	int _read_TOF_LST_data_CBM(const std::string inputfile_name, int &total_prompt_event, PET_data_type mode);

	int _create_full_coincidence_LST_data(PET_geometry &geometry, float &time, int &total_prompt_event, PET_coincidence_type type, int &lower1, int &upper1, int &lower2, int &upper2);
	int	_create_min_coincidence_LST_data(PET_geometry &geometry, int& total_prompt_event, PET_coincidence_type type, int &lower1, int &upper1, int &lower2, int &upper2);

	int _create_full_coincidence_LST_data_norm(PET_geometry &geometry, float &time, int &total_prompt_event, PET_coincidence_type type, int &lower1, int &upper1, int &lower2, int &upper2, int* &det1, int* &det2);
	int _create_full_coincidence_LST_data_normblock(PET_geometry &geometry, float &time, int &total_prompt_event, PET_coincidence_type type, int &lower1, int &upper1, int &lower2, int &upper2, int** &det1);
    int get_listmode_segmentation_file(const std::string inputfile_name);
	
	static inline void loadBar(long int x, long int n, int r, int w);
	int synchronize(unsigned int *src_pos_p);
	long int read_to_buffer(FILE *inputfile);
	
	int _get_crystal_id(int sx, int dx, int cx);
	int _get_crystal_id_PlantPET(int sx, int dx, int cx);
	int _get_crystal_id_MicroInsert(int sx, int dx, int cx);
};

#endif