#ifndef PETDATA_H
#define PETDATA_H

#include <string>
#include <array>
#include <vector>

#include "global.h"
#include "config.h"
#include "PET_LST_event.h"
#include "PET_geometry.h"

class PETData{

private:
	string inputfilename;
	string outputfilename;
	PET_data_source _source;

public:
	PETData();
	~PETData();
	
	void set_input();
	void set_output();

	virtual int ReadDataFromFile(const std::string inputfile_name, int &total_prompt_event, PET_data_type mode);
	virtual int ReadTimeSegmentFile(const std::string inputfile_name);
	virtual void CreateFullCoincidenceData(PET_geometry geometry, PET_coincidence_type type, int lower1, int upper1, int lower2, int upper2, int** normfact);



	int GetDataCount(PET_coincidence_type type);
	int SetDataSource(PET_data_source source);

protected:
	vector<PET_LST_event> PET_LST_event_list;


private:
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

	virtual int _get_crystal_id(int sx, int dx, int cx);




	int _get_crystal_id_PlantPET(int sx, int dx, int cx);
	int _get_crystal_id_MicroInsert(int sx, int dx, int cx);
};

#endif