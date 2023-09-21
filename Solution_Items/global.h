#ifndef GLOBAL_H
#define GLOBAL_H


#include <cmath>
#include <limits.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>

#include <string>
#include <vector>
#include <array>

#define MAX_NUM_LST_LENGTH_COPY_TO_ONE_GPU 200000000
#define MAX_NUM_LST_EVENT 1200000000
#define SMALLEST_ALLOWED 1e-8
#define BIGGEST_ALLOWED 10000

#define MAX_GPU 8
#define NUM_SCANNER_CRYSTALS 60800
#define NUM_INSERT_CRYSTALS 57600

#define SCANNER_CRYTAL_SHIFT_DISTANCE_CENTER_TO_SURFACE 0
#define OUTSERT_CRYTAL_SHIFT_DISTANCE_CENTER_TO_SURFACE 0


#define CRYSTAL_THICKNESS_OUTSERT 10
#define CRYSTAL_THICKNESS_SCANNER 20


#define CRYSTAL_WIDTH_OUTSERT 1
#define CRYSTAL_WIDTH_SCANNER 3.2

typedef enum { FORWARD, BACKWARD }proj_direction;
typedef enum { ATTENUATION, EMISSION }LUT_MODE;
typedef enum { TOF, NONTOF }TOF_MODE;
typedef enum { SS, IS, II, ALL } PET_coincidence_type;
typedef enum {STATIC,CBM,STEP_AND_SHOOT} PET_protocol_type;
typedef enum { ePROMPT, eDELAYED, eBODY_SCATTERED, eINSERT_SCATTERED } PET_LST_event_type;
typedef enum { PROMPT, DELAYED, BODY_SCATTERED, INSERT_SCATTERED } PET_data_type;
typedef enum { COMPACT_COINCIDENCE, INVEON, GATE_COINCIDENCE, GATE_SINGLES, XYZ_COINCIDENCE, NON_TOF_MINIMAL, TOF_MINIMAL } PET_data_source;


class box {
public:
	float dimension[3]; // vertices
	float normal_0[3]; // unit normal vectors
	float normal_1[3];
	float normal_2[3];
	float center[3]; // center point
};

class Detector_crystal{
public:
	/*	Detector_crystal(box thebox, int * thelayer){
	geometry = thebox;
	layer[0] = thelayer[0];
	layer[1] = thelayer[1];
	layer[2] = thelayer[2];
	}
	*/
	
	box geometry;
	int layer[3];
};


typedef struct {
	int NUM_X;
	int NUM_Y;
	int NUM_Z;

	float X_OFFSET;
	float Y_OFFSET;
	float Z_OFFSET;
	
	float X_SAMP;
	float Y_SAMP;
	float Z_SAMP;

	int TOF_on;
	float TOF_res;

	float FWHM;
	float FWHM_SS;
	float FWHM_IS;
	float FWHM_II;


	int num_gpu_start;
	int num_gpu_end;
	int num_iter;
	int start_iter;
	int num_OSEM_subsets;
	int write_freq;

	float prior_beta;
	float prior_delta;
	
	float spherical_voxel_ratio;
	int export_negative_log_likelihood;
	int attenuation_correction_fp;
	int attenuation_correction_bp;
}parameters_t;


#endif
