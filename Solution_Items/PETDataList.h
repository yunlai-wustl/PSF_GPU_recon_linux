#ifndef PETDATALIST_H
#define PETDATALIST_H

#include "PETData.h"


class PETDataList : public PETData{

private:
	int ListDataLength;

public:

	int GetListDataLength();
	void SetListDataLength();


};


#endif