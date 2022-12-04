#ifndef PET_LST_EVENT_SCATTER_H
#define PET_LST_EVENT_SCATTER_H

#include "global.h"
#include "GATE_data_structure.h"


class PET_LST_event_scatter
{
public:
	PET_LST_event_scatter(int id1, int id2, float tof, float t, float bed_position, int norm, float scatter_coeff) : src_id(id1), dest_id(id2), TOF_dist(tof), t0(t), bed_position(bed_position), sc_coeff(scatter_coeff){}
	int src_id;
	int dest_id;
	float TOF_dist;		//0.5*c*t in mm
	float t0;		//time of arrival in seconds
	int energy;
	float bed_position;
	PET_LST_event_type type;
	int normfact;
	float sc_coeff;

	bool is_event_type(PET_coincidence_type t){

		if (t == SS){
			return ((src_id < NUM_SCANNER_CRYSTALS) && (dest_id < NUM_SCANNER_CRYSTALS));
		}

		if (t == IS){
			return (((src_id < NUM_SCANNER_CRYSTALS) && (dest_id >= NUM_SCANNER_CRYSTALS)) || ((src_id >= NUM_SCANNER_CRYSTALS) && (dest_id < NUM_SCANNER_CRYSTALS)));
		}

		if (t == II){
			return ((src_id >= NUM_SCANNER_CRYSTALS) && (dest_id >= NUM_SCANNER_CRYSTALS));
		}

		if (t == ALL){
			return true;
		}
		else
			return false;

	}

};





#endif