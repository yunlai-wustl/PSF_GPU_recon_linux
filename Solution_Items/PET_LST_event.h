//
// PET_LST_event.h: header file for PET_LST_even
//
// Completion Date:  03/13/2022
//
// Description of the role or purpose of objects of this class goes here
//
// Constructors:
//      constructor prototype 1
//          PET_LST_event(int id1, int id2, float tof, float t, int e, PET_LST_event_type event_type, int norm) 
//						: src_id(id1), dest_id(id2), TOF_dist(tof), t0(t), energy(e), type(event_type), normfact(norm){}
//          Description of constructor: default constructor
//
//      constructor prototype 2
//          Description of constructor, e.g., copy constructor
//
// Public Member Functions:
//
//      is_event_type(PET_coincidence_type t)
//        precondition:
//        postcondition:
//          input: a coincidence type, can be 'SS','IS','II'
//          output: if current PET_LST_event is a type of t
//
//      prototype 2
//        precondition:
//        postcondition:
//          Description of what the function does, what input it expects,
//              what output it generates, what it returns.
//


#ifndef PET_LST_EVENT_H
#define PET_LST_EVENT_H

#include "global.h"
#include "GATE_data_structure.h"


class PET_LST_event
{
public:
	PET_LST_event(int id1, int id2, float tof, float t,  int norm) : src_id(id1), dest_id(id2), TOF_dist(tof), t0(t), normfact(norm){}
	
	PET_LST_event(int id1, int id2, float tof, float t,   int norm, float bed_position ) : \
		src_id(id1), dest_id(id2), TOF_dist(tof), t0(t),  normfact(norm), bed_position(bed_position){}
	// constructor for CBM reconstruction 
	PET_LST_event(int id1, int id2) :src_id(id1), dest_id(id2){}; //for sensitivity image calculation; minimize memory requirement 

	float bed_position;
	int src_id;
	int dest_id;
	double TOF_dist;		//0.5*c*t in mm
	float t0;		//time of arrival in seconds
	PET_LST_event_type type;
	int normfact;

	bool is_event_type(PET_coincidence_type t){
	
		if (t == SS){
			return ((src_id < NUM_SCANNER_CRYSTALS) && (dest_id < NUM_SCANNER_CRYSTALS));
		}
		
		if (t == IS){
			return (((src_id < NUM_SCANNER_CRYSTALS) && (dest_id >=NUM_SCANNER_CRYSTALS)) || ((src_id >= NUM_SCANNER_CRYSTALS) && (dest_id < NUM_SCANNER_CRYSTALS)));
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