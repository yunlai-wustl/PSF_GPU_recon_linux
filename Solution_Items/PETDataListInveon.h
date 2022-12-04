#ifndef PETDATALISTINVEON_H
#define PETDATALISTINVEON_H

#include "PETDataList.h"


class PETDataListInveon : public PETDataList{
public:





private:

	long int TotalBytes;
	long int FileSize;
	int cur_buf_size;
	unsigned OutOfSync;
	unsigned char *in_buf;

	static const unsigned char graycode[11];
	static inline void loadBar(long int x, long int n, int r, int w);
	int synchronize(unsigned int *src_pos_p);
	long int read_to_buffer(FILE *inputfile);

};


#endif