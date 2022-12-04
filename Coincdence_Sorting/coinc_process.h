/*
#define __REDIRECT
#define __USE_LARGEFILE64
#define __USE_FILE_OFFSET64
#define _LARGEFILE_SOURCE
#define _LARGEFILE64_SOURCE
*/
#include "../Solution_Items/GATE_data_structure.h"
#include <map>

#define MAX_SINGLES_IN_RUN 10000000
#define MAX_NUM_FILE 10


int coinc_sorter_2singles_by_time_window_scatter(GATE_singles_event **data, int *length, int num_file, int run_id, mini_coinc_event*& coinc_data);
int coinc_sorter_2singles_by_event_ID(GATE_singles_event **data, int *length, int num_file, int run_id, mini_coinc_event*& coinc_data, int protocol, float& initial_bed_position, float& speed, std::map<int, float>& bed_position_lut);
int coinc_sorter_2singles_by_event_ID_no_scatter(GATE_singles_event **data, int *length, int num_file, int run_id, mini_coinc_event*& coinc_data, int protocol, float& initial_bed_position, float& speed, std::map<int, float>& bed_position_lut);
int coinc_sorter_2singles_by_time_window(GATE_singles_event **data, int *length, int num_file, int run_id, mini_coinc_event*& coinc_data, int& protocol, float& initial_bed_position, float& speed, std::map<int, float>& bed_position_lut);
int coinc_sorter_2singles_by_event_ID_no_scatter_decay(GATE_singles_event **data, int *length, int num_file, int run_id, decay_event*& decay_data);
int record_mini_coinc(GATE_singles_event *d_ptr1, GATE_singles_event *d_ptr2, mini_coinc_event * coinc_ptr);
int record_mini_coinc(GATE_singles_event *d_ptr1, GATE_singles_event *d_ptr2, mini_coinc_event * coinc_ptr, float& start_bed_position, float& speed);
int record_mini_coinc(GATE_singles_event *d_ptr1, GATE_singles_event *d_ptr2, mini_coinc_event * coinc_ptr, std::map<int, float>& bed_posiiton_lut);
int record_mini_coinc_scatter(GATE_singles_event *d_ptr1, GATE_singles_event *d_ptr2, mini_coinc_event *coinc_ptr);
int record_decay(GATE_singles_event *d_ptr1, GATE_singles_event *d_ptr2, decay_event *decay_ptr);
int encode_crystal_mCT_flat(volume_id crystal_id);
int encode_crystal_mCT_cyl(volume_id crystal_id);
int encode_crystal_mCT_flat_geom(volume_id crystal_id);
int encode_crystal_plantPET(volume_id crystal_id);
int encode_crystal_pocPET(volume_id crystal_id);
int encode_crystal_pocPET002(volume_id crystal_id);
int encode_crystal_mCT(volume_id crystal_id);
int encode_crystal_mCT_Insert(volume_id crystal_id);
int encode_crystal_mCT_Insert_doi(volume_id crystal_id);
int encode_crystal_USPET(volume_id crystal_id);
int encode_crystal_USPET_pos1(volume_id crystal_id);
int encode_crystal_USPET_pos2(volume_id crystal_id);
int encode_crystal_USPET_pos3(volume_id crystal_id);
int encode_crystal_lungPET_pos1(volume_id crystal_id);
int encode_crystal_lungPET_pos2(volume_id crystal_id);
int encode_crystal_lungPET_pos3(volume_id crystal_id);
int encode_crystal_vision(volume_id crystal_id);
int encode_crystal_TBPET_vision(volume_id crystal_id);
int encode_crystal_outsert(volume_id crystal_id);
int encode_crystal_USPET_modified(volume_id crystal_id);
int encode_brain_PET(volume_id crystal_id);
