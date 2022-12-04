#ifndef PETDATALISTGATECOINCIDENCE_H
#define PETDATALISTGATECOINCIDENCE_H

#include "PETDataList.h"
#include "petreconSetGet.h"

class PETDataListGateCoincidence : public PETDataList{
private:
	PET_coincidence_type _coincidence_type;

	int read(const std::string inputfile_name, int &total_prompt_event, PET_data_type mode);


};


#endif