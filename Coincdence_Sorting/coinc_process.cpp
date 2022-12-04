/*
 * coinc_process.cc
 *
 *  Created on: Feb 10, 2014
 *      Author: keli
 */
#include "coinc_process.h"

#include <iostream>
#include <fstream>
#include <string.h>

#include <stdio.h>
#include <cmath>
#include <math.h>
#include <stdlib.h>
#include <limits.h>
#include <sys/stat.h>

using namespace std;


int main(int argc, char **argv)
{
	long signed int offset;
	int total_num_coincidence;
	int num_run = 0;
	int *current_length;
	int run_id;
	int temp;
	int num_run_in_each_file[MAX_NUM_FILE];
	int total_singles_data_length;
	int i, j, k;
	int num_single_file;
	char **single_file_names_ptr;
	GATE_singles_event **singles_data;
	int **seg_singles_in_run;
	int *singles_data_length;
	struct  stat64 file_status;
	long long int size;
	FILE *current_file;
	FILE *output;
	size_t t;
	char coin_filename[128] = "coinc_joint.lst";
	int sorting_method;// 0 for true, 1 for prompt, 2 for delayed
	mini_coinc_event *mini_coinc_data;
	decay_event* decay_data;
	int num_coincidence;
	int remainder;
	int protocol=0;
	float speed=0; //in mm/s
	float initial_bed_position=0; // initial bed position in scanner coordinate; in mm
	std::map<int, float> bed_position_lut;
//	printf("mini_coinc_event size: %d\n",sizeof(mini_coinc_event));
//	printf("size of long is %d\n",(int)sizeof(offset));
//	printf("Max Int is %d\n",INT_MAX);
//	printf("Max Long is %ld\n",LONG_MAX);
//	printf("Max Long Long is %ld\n",LLONG_MAX);


	//*****************************program argument*******************************************//
	if (argc <= 2){
		printf("Not enough arguments, please use -help option.\n");
		return 0;
	}

	if (strcmp(argv[1], "-help") == 0){
		printf("Here describes how to run this program.");
		printf("coincdence_sorting true/prompt static/CBM/step_and_shoot singles1.dat singles2.dat bed_position_lut(Step and shoot)/intial_bed_position(CBM)  speed(CBM) \n");
		printf("all distance units are mm; time unit is second\n");
		return 0;
	}
	else if (strcmp(argv[1], "true") == 0){
		sorting_method = 0;
		printf("Using \"true\" sorting method.\n");
	}
	else if (strcmp(argv[1], "prompt") == 0){
		sorting_method = 1;
		printf("Using \"prompt\" sorting method.\n");
	}
	else if (strcmp(argv[1], "scatter_timestamp") == 0){
		sorting_method = 2;
		printf("Using \"scatter\" sorting method.\n");
	}
	else if (strcmp(argv[1], "scatter_eventid") == 0){
		sorting_method = 3;
		printf("Using \"scatter\" sorting method.\n");
	}
	else if (strcmp(argv[1], "decay") == 0){
		sorting_method = 4;
		printf("writing out decay positions.\n");
	}
	else{
		//use default sorting method
		sorting_method = 0;
		printf("Unrecognized sorting method, using \"true\" instead.\n");
	}

	//check protocol (Static/CBM/step_and_shoot)
	if (strcmp(argv[2], "static") == 0){
		protocol = 0;
		printf("Using static protocl.\n");
	}
	else if (strcmp(argv[2], "CBM") == 0){
		protocol = 1;
		if (argc != 7)
			printf("not enough arguments for CBM! \n");
		initial_bed_position = atof(argv[5]);
		speed = atof(argv[6]);
		cout<<"Using \"CBM\" protocol"<<",initial position="<<initial_bed_position<<"mm"<<", speed ="<<speed<<"mm/s"<<endl;
	}
	else if (strcmp(argv[2], "step_and_shoot") == 0){
		protocol = 2;
		printf("Using \"step_and_shoot\" \n");
		ifstream bedPositionLut;
		int run_index;
		float bed_position_that_run;
		bedPositionLut.open(argv[5]);
		cout << "bedPositionLut" << endl;
		while (!bedPositionLut.eof()) {
			bedPositionLut >> run_index >> bed_position_that_run;
			bed_position_lut[run_index] = bed_position_that_run;
			std::cout << bed_position_that_run << std::endl;
		}
		cout << "bed position lookup table read successfully \n" << endl;
		for (auto b : bed_position_lut)
			cout << "run id:" << b.first << ", bed position:" << b.second << "mm" << endl;
	}
	else{
		//use default sorting method
		sorting_method = 0;
		printf("Unrecognized protocol, using \"static\" instead.\n");
	}


	//*****************************END*****program argument*******************************************//

	//*****************************check file length***********************************************//
	num_single_file = 2;
	single_file_names_ptr = argv;
	singles_data_length = (int*)calloc(num_single_file, sizeof(int));
	printf("Following single files are being processed:\n");
	single_file_names_ptr = (char**)calloc(num_single_file, sizeof(char*));
	total_singles_data_length = 0;
	for (i = 0; i<num_single_file; i++){
		single_file_names_ptr[i] = argv[i + 3];
		cout << single_file_names_ptr[i] << endl;
		stat64(single_file_names_ptr[i], &file_status);
		size = file_status.st_size;
		cout << "size of this singles file: " << size << endl;
		cout << "size of Gate_Singles:" << sizeof(GATE_singles_event) << endl;
		remainder = -1 * (size % sizeof(GATE_singles_event));
		cout << "Remainder: " << remainder<<"\n";
		size /= sizeof(GATE_singles_event);
		if (size >= INT_MAX){
			printf("File \"%s\" length is tooooooooooo long. This program only supports singles event fewer than %d. Current file has %lld. Program stopped\n", single_file_names_ptr[i], INT_MAX, size);
			return 0;
		}
		singles_data_length[i] = (int)size;
		total_singles_data_length += singles_data_length[i];
		printf("%d singles event in file \"%s\"\n", singles_data_length[i], single_file_names_ptr[i]);
	}





	//check the num_run in each singles file is consistent
	for (i = 0; i<num_single_file; i++){
		current_file = fopen(single_file_names_ptr[i], "rb");
		if (remainder == 0)
			fseek(current_file, -132, SEEK_END);
		else
			fseek(current_file, remainder, SEEK_END);
		fread(&num_run, sizeof(int), 1, current_file);
		num_run_in_each_file[i] = num_run + 1;
		fclose(current_file);
	}


	num_run = num_run_in_each_file[0];
	for (i = 0; i<num_single_file; i++){
		printf("%d\n", num_run_in_each_file[i]);
		if (num_run != num_run_in_each_file[i]){
			cout <<" 1st file has "<< num_run <<" runs, the other has " <<num_run_in_each_file[i]<<" runs" << endl;
			printf("Number of runs in different singles files are not consistent, check file");
			return 0;
		}
	}
	printf("All singles files have a number of run %d\n", num_run);
	if (protocol == 1)  {	   //CBM ignore run
		printf("using CBM protocol, ignore runs, set number of run = 1 \n");
		num_run = 1;
		seg_singles_in_run = (int**)calloc(num_single_file, sizeof(int*));
		for (i = 0; i < num_single_file; i++){
			current_file = fopen(single_file_names_ptr[i], "rb");
			seg_singles_in_run[i] = (int*)calloc(2, sizeof(int));//starting pos of each run and total length
			seg_singles_in_run[i][0] = 0;
			seg_singles_in_run[i][1] = singles_data_length[i];
			fclose(current_file);
			printf("Segmentation of singles in each run for file %d is:\n", i);

			for (j = 0; j < 2; j++){
				printf("\t%d", seg_singles_in_run[i][j]);
				printf("\n");
			}
		}
		printf("\n");
	}
	else{
		 	//find number of singles in each run, for each file
		seg_singles_in_run = (int**)calloc(num_single_file, sizeof(int*));
		for (i = 0; i < num_single_file; i++){
			current_file = fopen(single_file_names_ptr[i], "rb");
			seg_singles_in_run[i] = (int*)calloc(num_run + 1, sizeof(int));//starting pos of each run and total length
			j = 0;

			for (k = 0; k < singles_data_length[i]; k++){
				fread(&temp, sizeof(int), 1, current_file);
				if (temp == j){
					seg_singles_in_run[i][j] = k;
					j++;
					if (j == num_run)
						break;
				}
				fseek(current_file, 128, SEEK_CUR);
			}
			seg_singles_in_run[i][num_run] = singles_data_length[i];
			fclose(current_file);
			printf("Segmentation of singles in each run for file %d is:\n", i);

			for (j = 0; j < num_run + 1; j++){
				printf("\t%d", seg_singles_in_run[i][j]);
				printf("\n");

			}
		}
	}



	//***********************************Read data in one run each time and process****************************************//
	//Singles data memory
	singles_data = (GATE_singles_event**)calloc(num_single_file, sizeof(GATE_singles_event*));
	current_length = (int*)calloc(num_single_file, sizeof(int));
	output = fopen(coin_filename, "wb");

	/*
	ofstream myfile2;
	char str2[64];
	strcpy(str2,coin_filename);
	strcat(str2,".txt");
	myfile2.open(str2);
	*/

	ofstream myfile2;
	char str2[64];
	strcpy(str2, coin_filename);
	strcat(str2, ".txt");
	myfile2.open(str2);

	total_num_coincidence = 0;
	for (run_id = 0; run_id<num_run; run_id++){
		printf("\nNow doing coincidence processing for run #%d\n", run_id);

		//Read in singles data in each run at a time
		for (i = 0; i<num_single_file; i++){
			current_length[i] = seg_singles_in_run[i][run_id + 1] - seg_singles_in_run[i][run_id];//num of singles in current run

			printf("......reading single file #%d of %d, containing %d singles, from %d to %d.\n", i, num_single_file, current_length[i], seg_singles_in_run[i][run_id], seg_singles_in_run[i][run_id + 1]);
			singles_data[i] = (GATE_singles_event*)calloc(current_length[i], sizeof(GATE_singles_event));
			current_file = fopen(single_file_names_ptr[i], "rb");
			offset = 132LL * (long long)seg_singles_in_run[i][run_id];
			if (0 != fseek(current_file, offset, SEEK_SET)){
				printf("fseek for run %d file %i failed.\n", run_id, i);
			}
			else{
				printf("......current position for file %d is %ld\n", i, offset);
			}

			t = fread(singles_data[i], sizeof(GATE_singles_event), current_length[i], current_file);
			if (current_length[i] == (int)t){
				printf("......%s read successfully.\n",single_file_names_ptr[i]);
			}
			else{
				printf("......%s read error. %d elements were read\n", single_file_names_ptr[i], (int)t);
			}
			fclose(current_file);
		}

		num_coincidence = 0;

		//processing coincidence sorting for the current run
		//***************************************************sort the singles to get coincidences********************************************************//
		if (sorting_method == 0){
			printf("sorting events using event id......\n");

			if (protocol==1)
				cout << "initial position=" << initial_bed_position << "mm" << ", speed =" << speed << "mm/s" << endl;

			num_coincidence = coinc_sorter_2singles_by_event_ID(singles_data, current_length, num_single_file, run_id, mini_coinc_data, protocol,initial_bed_position,speed,bed_position_lut);

			printf("......%d coincidence events were returned.\n", num_coincidence);
			t = fwrite(mini_coinc_data, sizeof(mini_coinc_event), num_coincidence, output);
			if (num_coincidence == (int)t){
				total_num_coincidence += num_coincidence;
				//printf("%s containing %d coincidence events was written successfully.\n",coin_filename, num_coincidence);
			}
			else{
				printf("%s written with error. %d elements were written.\n", coin_filename, (int)t);
			}
			for (j = 0; j<100; j++){
				myfile2 << "  " << mini_coinc_data[j].crystal_index_1 << "  " << mini_coinc_data[j].crystal_index_2 << "  " << mini_coinc_data[j].time_1 << "        " << mini_coinc_data[j].diff_time << "    " << mini_coinc_data[j] .bed_position<< endl;
			}

			free(mini_coinc_data);


		}//end sorting method 0

		if (sorting_method == 1){//sorting by time window
			//combine all the singles files with their time of event in ascending order
			printf("sorting events using time window......\n");

			num_coincidence = coinc_sorter_2singles_by_time_window(singles_data, current_length, num_single_file, run_id, mini_coinc_data, protocol, initial_bed_position, speed, bed_position_lut);
			printf("%d coincidence events were returned.\n", num_coincidence);

			///added by jiang
			t = fwrite(mini_coinc_data, sizeof(mini_coinc_event), num_coincidence, output);
			if (num_coincidence == (int)t){
				total_num_coincidence += num_coincidence;
				//printf("%s containing %d coincidence events was written successfully.\n",coin_filename, num_coincidence);
			}
			else{
				printf("%s written with error. %d elements were written.\n", coin_filename, (int)t);
			}

			for (j = 0; j<100; j++){
				myfile2 << "  " << mini_coinc_data[j].crystal_index_1 << "  " << mini_coinc_data[j].crystal_index_2 << "  " << mini_coinc_data[j].time_1 << "        " << mini_coinc_data[j].diff_time << "     "<<mini_coinc_data[j].bed_position << endl;
			}

			free(mini_coinc_data);  // added by jiang

		}


		if (sorting_method == 2){//sorting by time window, then writing only scatter events
			//combine all the singles files with their time of event in ascending order
			num_coincidence = coinc_sorter_2singles_by_time_window_scatter(singles_data, current_length, num_single_file, run_id, mini_coinc_data);
			printf("%d coincidence events were returned.\n", num_coincidence);


			t = fwrite(mini_coinc_data, sizeof(mini_coinc_event), num_coincidence, output);
			if (num_coincidence == (int)t){
				total_num_coincidence += num_coincidence;
				//printf("%s containing %d coincidence events was written successfully.\n",coin_filename, num_coincidence);
			}
			else{
				printf("%s written with error. %d elements were written.\n", coin_filename, (int)t);
			}

			for (j = 0; j<100; j++){
				myfile2 << "  " << mini_coinc_data[j].crystal_index_1 << "  " << mini_coinc_data[j].crystal_index_2 << "  " << mini_coinc_data[j].time_1 << "        " << mini_coinc_data[j].diff_time << "        " << mini_coinc_data[j].bed_position << endl;
			}


			free(mini_coinc_data);  // added by suranjana

		}

		if (sorting_method == 3){//sorting by event_id, then writing only scatter events
			num_coincidence = coinc_sorter_2singles_by_event_ID_no_scatter(singles_data, current_length, num_single_file, run_id, mini_coinc_data, protocol, initial_bed_position, speed, bed_position_lut);
			printf("......%d coincidence events were returned.\n", num_coincidence);


			t = fwrite(mini_coinc_data, sizeof(mini_coinc_event), num_coincidence, output);
			if (num_coincidence == (int)t){
				total_num_coincidence += num_coincidence;
				//printf("%s containing %d coincidence events was written successfully.\n",coin_filename, num_coincidence);
			}
			else{
				printf("%s written with error. %d elements were written.\n", coin_filename, (int)t);
			}


			for (j = 0; j<100; j++){
				// if(mini_coinc_data[j].crystal_index_1>126336 || mini_coinc_data[j].crystal_index_2>126336){
				myfile2 << "  " << mini_coinc_data[j].crystal_index_1 << "  " << mini_coinc_data[j].crystal_index_2 << "  " << mini_coinc_data[j].time_1 << "        " << mini_coinc_data[j].diff_time << "        " << mini_coinc_data[j].bed_position << endl;
				//}
			}

			free(mini_coinc_data);
		}//end sorting method 3 added by suranjana, Yunlai


		if (sorting_method == 4){//sorting by event_id, no scatter, and write out decay
			num_coincidence = coinc_sorter_2singles_by_event_ID_no_scatter_decay(singles_data, current_length, num_single_file, run_id, decay_data);
			printf("......%d coincidence events were returned.\n", num_coincidence);
			t = fwrite(decay_data, sizeof(decay_event), num_coincidence, output);
			if (num_coincidence == (int)t){
				total_num_coincidence += num_coincidence;
				//printf("%s containing %d coincidence events was written successfully.\n",coin_filename, num_coincidence);
			}
			else{
				printf("%s written with error. %d elements were written.\n", coin_filename, (int)t);
			}
			for (j = 0; j<100; j++){
				// if(mini_coinc_data[j].crystal_index_1>126336 || mini_coinc_data[j].crystal_index_2>126336){
				myfile2 << "  " << decay_data[j].crystal_index_1 << "  " << decay_data[j].crystal_index_2 << "  " << decay_data[j].source_x << "        " << decay_data[j].source_x << "        " << decay_data[j].source_x << endl;
				//}
			}
			free(decay_data);
		}//end sorting method 4 added by Yunlai



		for (i = 0; i<num_single_file; i++){
			free(singles_data[i]);
		}




	}

	printf("\nSorting finished! %d coincidence events in total.\n", total_num_coincidence);

	//calc_number_events(output, total_num_coincidence);
	fclose(output);
	myfile2.close();

	free(current_length);

	for (i = 0; i<num_single_file; i++){
		free(seg_singles_in_run[i]);
	}
	free(singles_data);
	free(seg_singles_in_run);






	//***********************************free memory****************************************//
	free(single_file_names_ptr);


	return 0;

}

int coinc_sorter_2singles_by_time_window(GATE_singles_event **data, int *length, int num_file, int run_id, mini_coinc_event*& coinc_data, int& protocol, float& initial_bed_position, float& speed, map<int,float>& bed_position_lut){
	/*int coinc_ind;
	int num_true;
	coinc_ind = 0;
	return coinc_ind;*/

	//// added by jiang
	GATE_singles_event *data_ptr1, *data_ptr2;
	GATE_singles_event *bigger, *smaller;

	int total_num_singles;
	int total_num_coinc, coinc_ind;
	int i, small_ind, big_ind;
	int index[2];


	//printf("\n");
	//printf("Begin coincidence sorting.\n");
	if (num_file != 2){
		printf("This function for sorting 2 singles file only.\n");
		printf("Only the first 2 singles file will be sorted, others will be ignored.\n");
	}

	num_file = 2;
	total_num_singles = 0;
	for (i = 0; i<num_file; i++){
		total_num_singles += length[i];
	}
	//assume a 15% coincidenc/singles ratio, allocate memory for coincidence
	total_num_coinc = (int)(0.80*total_num_singles);
	coinc_data = (mini_coinc_event*)calloc(total_num_coinc, sizeof(mini_coinc_event));
	printf("......Total number of singles (in %d singles files) is %d, allocating %d entries for coincidence.\n", num_file, total_num_singles, total_num_coinc);

	//For each run group, sort the data from 2 files by their interaction time

	index[0] = 0;
	index[1] = 0;

	coinc_ind = 0;

	while (coinc_ind<total_num_coinc){

		

		data_ptr1 = data[0] + index[0];
		data_ptr2 = data[1] + index[1];

		smaller = data_ptr1;
		bigger = data_ptr2;
		
		//First, compare between the two event
		if (((double)(abs(data_ptr1->time - data_ptr2->time)*pow(10, 9)))<4.7){
			switch (protocol)
			{
			case 0:	record_mini_coinc(data_ptr1, data_ptr2, coinc_data + coinc_ind); break;//static
			case 1: record_mini_coinc(data_ptr1, data_ptr2, coinc_data + coinc_ind, initial_bed_position, speed); break;//CBM
			case 2: record_mini_coinc(data_ptr1, data_ptr2, coinc_data + coinc_ind, bed_position_lut); break;//step and shoot
			}

			coinc_ind++;

			//if this is the end for both list, done
			if (index[0] == (length[0] - 1) && index[1] == (length[1] - 1))
				break;

			//if anyone is not at the end, increase it and continue
			if (index[0]<length[0] - 1)
				index[0]++;
			if (index[1]<length[1] - 1)
				index[1]++;
			continue;
		}
		
		else if (data_ptr1->time < data_ptr2->time){
			smaller = data_ptr1;
			bigger = data_ptr2;
			small_ind = 0;
			big_ind = 1;
		}
		else if (data_ptr1->time > data_ptr2->time){
			smaller = data_ptr2;
			bigger = data_ptr1;
			small_ind = 1;
			big_ind = 0;
		}

		//if we reach the end of current run for both lists, then skip the following part, current run is done;
		if (index[0] >= (length[0] - 1) && index[1] >= (length[1] - 1))
			break;
		

		//scan the smaller ptr, till it exceeds the bigger
		while (index[small_ind] < (length[small_ind] - 1)){//notice -1 here!
			if (((double)(abs(smaller->time - (smaller + 1)->time)*pow(10, 9)))<4.5){
				switch (protocol)
				{
				case 0:	record_mini_coinc(smaller, smaller + 1, coinc_data + coinc_ind); break;//static
				case 1: record_mini_coinc(smaller, smaller + 1, coinc_data + coinc_ind, initial_bed_position, speed); break;//CBM
				case 2: record_mini_coinc(smaller, smaller + 1, coinc_data + coinc_ind, bed_position_lut); break;//step and shoot
				}
				smaller = smaller + 1;
				index[small_ind]++;
				coinc_ind++;
			}
			else{
				smaller++;
				index[small_ind]++;
			}

			if (smaller->time >= bigger->time){
				break;
			}
		}

		

		//if the smaller ptr reaches the end, then scan the bigger list till end
		//this mean the previous while loop is broken when the samller list reaches the end
		if (smaller->time < bigger->time){
			
			while (index[big_ind] < (length[big_ind] - 1)){
				if (((double)(abs(bigger->time - (bigger + 1)->time)*pow(10, 9)))<4.5){
					//cout << "\n coinc # : " << coinc_ind;
					switch (protocol)
					{
					case 0:	record_mini_coinc(bigger, bigger + 1, coinc_data + coinc_ind); break;//static
					case 1: record_mini_coinc(bigger, bigger + 1, coinc_data + coinc_ind, initial_bed_position, speed); break;//CBM
					case 2: record_mini_coinc(bigger, bigger + 1, coinc_data + coinc_ind, bed_position_lut); break;//step and shoot
					}

					bigger = bigger + 1;
					index[big_ind]++;
					coinc_ind++;
				}
				else{
					bigger++;
					index[big_ind]++;
				}

			}
		}
		
	}

	return coinc_ind;

}



int coinc_sorter_2singles_by_time_window_scatter(GATE_singles_event **data, int *length, int num_file, int run_id, mini_coinc_event*& coinc_data){
	/*int coinc_ind;
	int num_true;
	coinc_ind = 0;
	return coinc_ind;*/

	
	GATE_singles_event *data_ptr1, *data_ptr2;
	GATE_singles_event *bigger, *smaller;

	int total_num_singles;
	int total_num_coinc, coinc_ind;
	int i, small_ind, big_ind;
	int index[2];


	//printf("\n");
	//printf("Begin coincidence sorting.\n");
	if (num_file != 2){
		printf("This function for sorting 2 singles file only.\n");
		printf("Only the first 2 singles file will be sorted, others will be ignored.\n");
	}

	num_file = 2;
	total_num_singles = 0;
	for (i = 0; i<num_file; i++){
		total_num_singles += length[i];
	}
	//assume a 15% coincidenc/singles ratio, allocate memory for coincidence
	total_num_coinc = (int)(0.80*total_num_singles);
	coinc_data = (mini_coinc_event*)calloc(total_num_coinc, sizeof(mini_coinc_event));
	printf("......Total number of singles (in %d singles files) is %d, allocating %d entries for coincidence.\n", num_file, total_num_singles, total_num_coinc);

	//For each run group, sort the data from 2 files by their interaction time

	index[0] = 0;
	index[1] = 0;

	coinc_ind = 0;

	while (coinc_ind<total_num_coinc){

		data_ptr1 = data[0] + index[0];
		data_ptr2 = data[1] + index[1];

		smaller = data_ptr1;
		bigger = data_ptr2;

		//First, compare between the two event
		if (((double)(abs(data_ptr1->time - data_ptr2->time)*pow(10, 9)))<4.5){

			record_mini_coinc_scatter(data_ptr1, data_ptr2, coinc_data + coinc_ind);
			coinc_ind++;


			//if this is the end for both list, done
			if (index[0] == (length[0] - 1) && index[1] == (length[1] - 1))
				break;

			//if anyone is not at the end, increase it and continue
			if (index[0]<length[0] - 1)
				index[0]++;
			if (index[1]<length[1] - 1)
				index[1]++;
			continue;
		}
		else if (data_ptr1->time < data_ptr2->time){
			smaller = data_ptr1;
			bigger = data_ptr2;
			small_ind = 0;
			big_ind = 1;
		}
		else if (data_ptr1->time > data_ptr2->time){
			smaller = data_ptr2;
			bigger = data_ptr1;
			small_ind = 1;
			big_ind = 0;
		}

		//if we reach the end of current run for both lists, then skip the following part, current run is done;
		if (index[0] >= (length[0] - 1) && index[1] >= (length[1] - 1))
			break;


		//scan the smaller ptr, till it exceeds the bigger
		while (index[small_ind] < (length[small_ind] - 1)){//notice -1 here!
			if (((double)(abs(smaller->time - (smaller + 1)->time)*pow(10, 9)))<4.5){

				record_mini_coinc_scatter(smaller, smaller + 1, coinc_data + coinc_ind);
				smaller = smaller + 1;
				index[small_ind]++;
				coinc_ind++;
			}
			else{
				smaller++;
				index[small_ind]++;
			}

			if (smaller->time >= bigger->time){
				break;
			}
		}

		//if the smaller ptr reaches the end, then scan the bigger list till end
		//this mean the previous while loop is broken when the samller list reaches the end
		if (smaller->time < bigger->time){

			while (index[big_ind] < (length[big_ind] - 1)){
				if (((double)(abs(bigger->time - (bigger + 1)->time)*pow(10, 9)))<4.5){

					record_mini_coinc_scatter(bigger, bigger + 1, coinc_data + coinc_ind);
					bigger = bigger + 1;
					index[big_ind]++;
					coinc_ind++;
				}
				else{
					bigger++;
					index[big_ind]++;
				}

			}
		}
	}

	return coinc_ind;

}

int coinc_sorter_2singles_by_event_ID(GATE_singles_event **data, int *length, int num_file, int run_id, mini_coinc_event*& coinc_data, int protocol, float& initial_bed_position, float& speed, map<int, float>& bed_position_lut){
	GATE_singles_event *data_ptr1, *data_ptr2;
	GATE_singles_event *bigger, *smaller;
	printf("protocol = %d", protocol);
	int total_num_singles;
	int total_num_coinc, coinc_ind;
	int i, small_ind, big_ind;
	int index[2];


	//printf("\n");
	//printf("Begin coincidence sorting.\n");
	if (num_file != 2){
		printf("This function for sorting 2 singles file only.\n");
		printf("Only the first 2 singles file will be sorted, others will be ignored.\n");
	}

	num_file = 2;
	total_num_singles = 0;
	for (i = 0; i<num_file; i++){
		total_num_singles += length[i];
	}
	//assume a 15% coincidenc/singles ratio, allocate memory for coincidence
	total_num_coinc = (int)(0.8*total_num_singles);
	coinc_data = (mini_coinc_event*)calloc(total_num_coinc, sizeof(mini_coinc_event));
	printf("......Total number of singles (in %d singles files) is %d, allocating %d entries for coincidence.\n", num_file, total_num_singles, total_num_coinc);

	//For each run group, sort the data from 2 files by there event ID

	index[0] = 0;
	index[1] = 0;

	coinc_ind = 0;

	while (coinc_ind<total_num_coinc){

		data_ptr1 = data[0] + index[0];
		data_ptr2 = data[1] + index[1];


		//First, compare between the two event
		if (data_ptr1->eventID == data_ptr2->eventID){
			switch (protocol)
			{
			case 0:	record_mini_coinc(data_ptr1, data_ptr2, coinc_data + coinc_ind); break;//static
			case 1: record_mini_coinc(data_ptr1, data_ptr2, coinc_data + coinc_ind, initial_bed_position, speed); break;//CBM
			case 2: record_mini_coinc(data_ptr1, data_ptr2, coinc_data + coinc_ind, bed_position_lut); break;//step and shoot
			}
			coinc_ind++;

			//if this is the end for both list, done
			if (index[0] == (length[0] - 1) && index[1] == (length[1] - 1))
				break;

			//if anyone is not at the end, increase it and continue
			if (index[0]<length[0] - 1)
				index[0]++;
			if (index[1]<length[1] - 1)
				index[1]++;
			continue;
		}
		else if (data_ptr1->eventID < data_ptr2->eventID){
			smaller = data_ptr1;
			bigger = data_ptr2;
			small_ind = 0;
			big_ind = 1;
		}
		else{
			smaller = data_ptr2;
			bigger = data_ptr1;
			small_ind = 1;
			big_ind = 0;
		}

		//if we reach the end of current run for both lists, then skip the following part, current run is done;
		if (index[0] >= (length[0] - 1) && index[1] >= (length[1] - 1))
			break;


		//scan the smaller ptr, till it exceeds the bigger
		while (index[small_ind] < (length[small_ind] - 1)){//notice -1 here!
			if (smaller->eventID == (smaller + 1)->eventID){
				switch (protocol)
				{
				case 0:
					record_mini_coinc(smaller, smaller + 1, coinc_data + coinc_ind);//static
					break;//static
				case 1: 
					record_mini_coinc(smaller, smaller + 1, coinc_data + coinc_ind, initial_bed_position, speed);//CBM
					break;
				case 2: 
					record_mini_coinc(smaller, smaller + 1, coinc_data + coinc_ind, bed_position_lut);//step and shoot
					break;
				}
				smaller = smaller + 1;
				index[small_ind]++;
				coinc_ind++;
			}
			else{
				smaller++;
				index[small_ind]++;
			}

			if (smaller->eventID >= bigger->eventID){
				break;
			}
		}

		//if the smaller ptr reaches the end, then scan the bigger list till end
		//this mean the previous while loop is broken when the samller list reaches the end
		if (smaller->eventID < bigger->eventID){

			while (index[big_ind] < (length[big_ind] - 1)){
				if (bigger->eventID == (bigger + 1)->eventID){
					switch (protocol)
					{
						case 0:	
							record_mini_coinc(bigger, bigger + 1, coinc_data + coinc_ind);
							break;//static
						case 1:
							record_mini_coinc(bigger, bigger + 1, coinc_data + coinc_ind, initial_bed_position, speed);
							break;//CBM
						case 2:
							record_mini_coinc(bigger, bigger + 1, coinc_data + coinc_ind, bed_position_lut);//step and shoot
							break;
					}
					bigger = bigger + 1;
					index[big_ind]++;
					coinc_ind++;
				}
				else{
					bigger++;
					index[big_ind]++;
				}

			}
		}
	}

	return coinc_ind;
}




int coinc_sorter_2singles_by_event_ID_no_scatter(GATE_singles_event **data, int *length, int num_file, int run_id, mini_coinc_event*& coinc_data, int protocol, float& initial_bed_position, float& speed, std::map<int, float>& bed_position_lut){
	GATE_singles_event *data_ptr1, *data_ptr2;
	GATE_singles_event *bigger, *smaller;
	printf("protocol = %d", protocol);
	int total_num_singles;
	int total_num_coinc, coinc_ind;
	int i, small_ind, big_ind;
	int index[2];


	//printf("\n");
	//printf("Begin coincidence sorting.\n");
	if (num_file != 2){
		printf("This function for sorting 2 singles file only.\n");
		printf("Only the first 2 singles file will be sorted, others will be ignored.\n");
	}

	num_file = 2;
	total_num_singles = 0;
	for (i = 0; i<num_file; i++){
		total_num_singles += length[i];
	}
	//assume a 15% coincidenc/singles ratio, allocate memory for coincidence
	total_num_coinc = (int)(0.8*total_num_singles);
	coinc_data = (mini_coinc_event*)calloc(total_num_coinc, sizeof(mini_coinc_event));
	printf("......Total number of singles (in %d singles files) is %d, allocating %d entries for coincidence.\n", num_file, total_num_singles, total_num_coinc);

	//For each run group, sort the data from 2 files by there event ID

	index[0] = 0;
	index[1] = 0;

	coinc_ind = 0;

	while (coinc_ind<total_num_coinc){

		data_ptr1 = data[0] + index[0];
		data_ptr2 = data[1] + index[1];


		//First, compare between the two event
		if (data_ptr1->eventID == data_ptr2->eventID){
			if((data_ptr1->compton_scatter_in_Phantom == 0) && (data_ptr2->compton_scatter_in_Phantom == 0)){
				switch (protocol)
				{
				case 0:	record_mini_coinc(data_ptr1, data_ptr2, coinc_data + coinc_ind); break;//static
				case 1: record_mini_coinc(data_ptr1, data_ptr2, coinc_data + coinc_ind, initial_bed_position, speed); break;//CBM
				case 2: record_mini_coinc(data_ptr1, data_ptr2, coinc_data + coinc_ind, bed_position_lut); break;//step and shoot
				}
				coinc_ind++;
			}
			//if this is the end for both list, done
			if (index[0] == (length[0] - 1) && index[1] == (length[1] - 1))
				break;
			//if anyone is not at the end, increase it and continue
			if (index[0]<length[0] - 1)
				index[0]++;
			if (index[1]<length[1] - 1)
				index[1]++;
			continue;
		}
		else if (data_ptr1->eventID < data_ptr2->eventID){
			smaller = data_ptr1;
			bigger = data_ptr2;
			small_ind = 0;
			big_ind = 1;
		}
		else{
			smaller = data_ptr2;
			bigger = data_ptr1;
			small_ind = 1;
			big_ind = 0;
		}

		//if we reach the end of current run for both lists, then skip the following part, current run is done;
		if (index[0] >= (length[0] - 1) && index[1] >= (length[1] - 1))
			break;


		//scan the smaller ptr, till it exceeds the bigger
		while (index[small_ind] < (length[small_ind] - 1)){//notice -1 here!
			if ((smaller->eventID == (smaller + 1)->eventID)){
				if ((smaller->compton_scatter_in_Phantom == 0) && ((smaller + 1)->compton_scatter_in_Phantom == 0)){
					switch (protocol){
					case 0:	
						record_mini_coinc(smaller, smaller + 1, coinc_data + coinc_ind);//static
						break;//static
					case 1:
						record_mini_coinc(smaller, smaller + 1, coinc_data + coinc_ind, initial_bed_position, speed);//CBM
						break;
					case 2:
						record_mini_coinc(smaller, smaller + 1, coinc_data + coinc_ind, bed_position_lut);//step and shoot
						break;
					}
					coinc_ind++;
				}
				smaller = smaller + 1;
				index[small_ind]++;
			}
			else{
				smaller++;
				index[small_ind]++;
			}

			if (smaller->eventID >= bigger->eventID){
				break;
			}
		}

		//if the smaller ptr reaches the end, then scan the bigger list till end
		//this mean the previous while loop is broken when the samller list reaches the end
		if ((smaller->eventID < bigger->eventID)&&smaller){

			while (index[big_ind] < (length[big_ind] - 1)){
				if (bigger->eventID == (bigger + 1)->eventID){
					if ((bigger->compton_scatter_in_Phantom == 0) && ((bigger + 1)->compton_scatter_in_Phantom == 0)){
						switch (protocol)
						{
						case 0:
							record_mini_coinc(bigger, bigger + 1, coinc_data + coinc_ind);
							break;//static
						case 1:
							record_mini_coinc(bigger, bigger + 1, coinc_data + coinc_ind, initial_bed_position, speed);
							break;//CBM
						case 2:
							record_mini_coinc(bigger, bigger + 1, coinc_data + coinc_ind, bed_position_lut);//step and shoot
							break;
						}
						coinc_ind++;
					}
					bigger = bigger + 1;
					index[big_ind]++;
				}
				else{
					bigger++;
					index[big_ind]++;
				}

			}
		}
	}

	return coinc_ind;
}





int coinc_sorter_2singles_by_event_ID_no_scatter_decay(GATE_singles_event **data, int *length, int num_file, int run_id, decay_event*& decay_data){
	GATE_singles_event *data_ptr1, *data_ptr2;
	GATE_singles_event *bigger, *smaller;
	int total_num_singles;
	int total_num_coinc, coinc_ind;
	int i, small_ind, big_ind;
	int index[2];


	//printf("\n");
	//printf("Begin coincidence sorting.\n");
	if (num_file != 2){
		printf("This function for sorting 2 singles file only.\n");
		printf("Only the first 2 singles file will be sorted, others will be ignored.\n");
	}

	num_file = 2;
	total_num_singles = 0;
	for (i = 0; i<num_file; i++){
		total_num_singles += length[i];
	}
	//assume a 15% coincidenc/singles ratio, allocate memory for coincidence
	total_num_coinc = (int)(0.8*total_num_singles);
	decay_data = (decay_event*)calloc(total_num_coinc, sizeof(decay_event));
	printf("......Total number of singles (in %d singles files) is %d, allocating %d entries for coincidence.\n", num_file, total_num_singles, total_num_coinc);

	//For each run group, sort the data from 2 files by there event ID

	index[0] = 0;
	index[1] = 0;

	coinc_ind = 0;

	while (coinc_ind<total_num_coinc){

		data_ptr1 = data[0] + index[0];
		data_ptr2 = data[1] + index[1];


		//First, compare between the two event
		if (data_ptr1->eventID == data_ptr2->eventID){
			if ((data_ptr1->compton_scatter_in_Phantom == 0) && (data_ptr2->compton_scatter_in_Phantom == 0)){
				record_decay(data_ptr1, data_ptr2, decay_data + coinc_ind); 
				coinc_ind++;
			}
			//if this is the end for both list, done
			if (index[0] == (length[0] - 1) && index[1] == (length[1] - 1))
				break;
			//if anyone is not at the end, increase it and continue
			if (index[0]<length[0] - 1)
				index[0]++;
			if (index[1]<length[1] - 1)
				index[1]++;
			continue;
		}
		else if (data_ptr1->eventID < data_ptr2->eventID){
			smaller = data_ptr1;
			bigger = data_ptr2;
			small_ind = 0;
			big_ind = 1;
		}
		else{
			smaller = data_ptr2;
			bigger = data_ptr1;
			small_ind = 1;
			big_ind = 0;
		}

		//if we reach the end of current run for both lists, then skip the following part, current run is done;
		if (index[0] >= (length[0] - 1) && index[1] >= (length[1] - 1))
			break;


		//scan the smaller ptr, till it exceeds the bigger
		while (index[small_ind] < (length[small_ind] - 1)){//notice -1 here!
			if ((smaller->eventID == (smaller + 1)->eventID)){
				if ((smaller->compton_scatter_in_Phantom == 0) && ((smaller + 1)->compton_scatter_in_Phantom == 0)){
					record_decay(smaller, smaller + 1, decay_data + coinc_ind);				
					coinc_ind++;
				}
				smaller = smaller + 1;
				index[small_ind]++;
			}
			else{
				smaller++;
				index[small_ind]++;
			}

			if (smaller->eventID >= bigger->eventID){
				break;
			}
		}

		//if the smaller ptr reaches the end, then scan the bigger list till end
		//this mean the previous while loop is broken when the samller list reaches the end
		if ((smaller->eventID < bigger->eventID) && smaller){

			while (index[big_ind] < (length[big_ind] - 1)){
				if (bigger->eventID == (bigger + 1)->eventID){
					if ((bigger->compton_scatter_in_Phantom == 0) && ((bigger + 1)->compton_scatter_in_Phantom == 0)){
						record_decay(bigger, bigger + 1, decay_data + coinc_ind);//step and shoot						
						coinc_ind++;
					}
					bigger = bigger + 1;
					index[big_ind]++;
				}
				else{
					bigger++;
					index[big_ind]++;
				}

			}
		}
	}

	return coinc_ind;
}




//*************************************over loading function record_mini_coinc based on input parameters***************************************//
//static 
int record_mini_coinc(GATE_singles_event *d_ptr1, GATE_singles_event *d_ptr2, mini_coinc_event *coinc_ptr){
	//record this coinc

	int a, b, c, d;
	a = encode_crystal_outsert(d_ptr1->single_ID);
	b = encode_crystal_outsert(d_ptr2->single_ID);
	coinc_ptr->crystal_index_1 = a;
	coinc_ptr->crystal_index_2 = b;
	coinc_ptr->time_1 = d_ptr1->time;
	coinc_ptr->diff_time = d_ptr2->time - d_ptr1->time;
	coinc_ptr->bed_position = 0.0f;
	//recorded
	return 0;
}
//continous bed motion
int record_mini_coinc(GATE_singles_event *d_ptr1, GATE_singles_event *d_ptr2, mini_coinc_event *coinc_ptr,float& start_bed_position, float& speed){
	//record this coinc
	int a, b, c, d;
	a = encode_crystal_outsert(d_ptr1->single_ID);
	b = encode_crystal_outsert(d_ptr2->single_ID);
	coinc_ptr->crystal_index_1 = a;
	coinc_ptr->crystal_index_2 = b;
	coinc_ptr->time_1 = d_ptr1->time; //in second
	coinc_ptr->diff_time = d_ptr2->time - d_ptr1->time; // in second
	coinc_ptr->bed_position = start_bed_position + d_ptr1->time*speed; //speed in mm/s 
	//recorded
	return 0;
}

//step and shoot
int record_mini_coinc(GATE_singles_event *d_ptr1, GATE_singles_event *d_ptr2, mini_coinc_event *coinc_ptr, std::map<int,float>& bed_posiiton_lut){
	//record this coinc
	int a, b, c, d;
	a = encode_crystal_outsert(d_ptr1->single_ID);
	b = encode_crystal_outsert(d_ptr2->single_ID);
	coinc_ptr->crystal_index_1 = a;
	coinc_ptr->crystal_index_2 = b;
	coinc_ptr->time_1 = d_ptr1->time; //in second
	coinc_ptr->diff_time = d_ptr2->time - d_ptr1->time; // in second
	coinc_ptr->bed_position = bed_posiiton_lut[d_ptr1->Run]; //run
	//recorded
	return 0;
}


int record_decay(GATE_singles_event *d_ptr1, GATE_singles_event *d_ptr2, decay_event *decay_ptr){
	//record this coinc
	int a, b, c, d;
	a = encode_crystal_outsert(d_ptr1->single_ID);
	b = encode_crystal_outsert(d_ptr2->single_ID);
	decay_ptr->crystal_index_1 = a;
	decay_ptr->crystal_index_2 = b;
	decay_ptr->source_x = (float)(d_ptr1->source).x;
	decay_ptr->source_y = (float)(d_ptr1->source).y;
	decay_ptr->source_z = (float)(d_ptr1->source).z;
	//recorded
	return 0;
}







//*****************************************************************
int record_mini_coinc_scatter(GATE_singles_event *d_ptr1, GATE_singles_event *d_ptr2, mini_coinc_event *coinc_ptr){
	//record this coinc

	int a, b, c, d, e, f;
	a = encode_crystal_outsert(d_ptr1->single_ID);
	b = encode_crystal_outsert(d_ptr2->single_ID);
	c = d_ptr1->compton_scatter_in_Phantom;
	d = d_ptr2->compton_scatter_in_Phantom;
	e = d_ptr1->compton_scatter_in_Detector;
	f = d_ptr2->compton_scatter_in_Detector;
	if (c != 0 || d != 0){
		coinc_ptr->crystal_index_1 = a;
		coinc_ptr->crystal_index_2 = b;
		coinc_ptr->time_1 = d_ptr1->time;
		coinc_ptr->diff_time = d_ptr2->time - d_ptr1->time;
	}
	else{
		coinc_ptr->crystal_index_1 = 0;
		coinc_ptr->crystal_index_2 = 0;
		coinc_ptr->time_1 = 0;
		coinc_ptr->diff_time = 0;
	}

	//recorded
	return 0;
}
int encode_crystal_plantPET(volume_id crystal_id){
	int crystal_index;
	int block_index;

	if (crystal_id.system == 1){//belongs to Inveon
		block_index = 8 * crystal_id.module + crystal_id.sector;
		crystal_index = 400 * block_index + crystal_id.crystal + 5376;
	}
	else{//belongs to R4
		block_index = 21 * crystal_id.module + crystal_id.sector;
		crystal_index = 64 * block_index + crystal_id.crystal;
	}

	return crystal_index;
}

int encode_crystal_pocPET(volume_id crystal_id){
	int crystal_index;
	int block_index;

	if (crystal_id.system == 1){//belongs to BackPanel
		block_index = crystal_id.module + 4 * crystal_id.sector;
		crystal_index = 256 * block_index + crystal_id.crystal;
	}
	else{//belongs to Front
		block_index = crystal_id.module;
		crystal_index = 256 * block_index + crystal_id.crystal + 6144;
	}

	return crystal_index;
}


int encode_crystal_pocPET002(volume_id crystal_id){
	int crystal_index;
	int block_index;

	if (crystal_id.system == 0){//belongs to FrontPanel
		block_index = crystal_id.module;
		crystal_index = 256 * block_index + crystal_id.crystal;
	}
	else{//belongs to Back
		block_index = crystal_id.module;
		crystal_index = 256 * block_index + crystal_id.crystal + 4096;
	}

	return crystal_index;
}

int encode_crystal_mCT(volume_id crystal_id){
	int crystal_index;
	int block_index;

	block_index = crystal_id.module + 4 * crystal_id.sector;
	crystal_index = 169 * block_index + crystal_id.crystal;

	return crystal_index;
}

int encode_brain_PET(volume_id crystal_id){
	int crystal_index;
	int block_index;

	block_index = crystal_id.module + 8 * crystal_id.sector;
	crystal_index = 32 * block_index + crystal_id.crystal;

	return crystal_index;
}

int encode_crystal_vision(volume_id crystal_id){
	int crystal_index;
	int block_index;

	block_index = crystal_id.module + 8 * crystal_id.sector;
	crystal_index = 200 * block_index + crystal_id.crystal;

	return crystal_index;
}

int encode_crystal_outsert(volume_id crystal_id){
	int crystal_index;
	int block_index;
	if (crystal_id.system == 0){
		block_index = crystal_id.module + 8 * crystal_id.sector;
		crystal_index = 200 * block_index + crystal_id.crystal;
	}
	else{//belongs to outsert
		block_index = crystal_id.module + 8 * crystal_id.sector;
		crystal_index = 2048 * block_index + crystal_id.crystal + 60800;
	}


	return crystal_index;
}

int encode_crystal_mCT_cyl(volume_id crystal_id){
	int crystal_index;
	int block_index;

	block_index = crystal_id.module + 2 * crystal_id.sector;
	crystal_index = 676 * block_index + crystal_id.crystal;

	return crystal_index;
}

/*int encode_crystal_mCT(volume_id crystal_id){
int crystal_index;
int block_index;

block_index = crystal_id.sector;
crystal_index = 2000 * block_index + crystal_id.crystal;

return crystal_index;
}
*/
int encode_crystal_mCT_Insert_prev(volume_id crystal_id){
	int crystal_index;
	int block_index;

	if (crystal_id.system == 0){//belongs to rr1
		block_index = (crystal_id.sector);
		crystal_index = 2000 * block_index + crystal_id.crystal;
	}
	else if (crystal_id.system == 1){//belongs to rr2
		block_index = (crystal_id.sector);
		crystal_index = 2000 * block_index + crystal_id.crystal + 18000;
	}
	else if (crystal_id.system == 2){//belongs to rr3
		block_index = (crystal_id.sector);
		crystal_index = 2000 * block_index + crystal_id.crystal + 36000;
	}
	else if (crystal_id.system == 3){//belongs to rr4
		block_index = (crystal_id.sector);
		crystal_index = 2000 * block_index + crystal_id.crystal + 42000;
	}
	else if (crystal_id.system == 4){//belongs to front
		block_index = (crystal_id.sector);
		crystal_index = 2000 * block_index + crystal_id.crystal + 48000;
	}
	else if (crystal_id.system == 5){//belongs to back
		block_index = (crystal_id.sector);
		crystal_index = 975 * block_index + crystal_id.crystal + 62000;
	}

	return crystal_index;

}

int encode_crystal_mCT_Insert(volume_id crystal_id){
	int crystal_index;
	int block_index;

	if (crystal_id.system == 0){//belongs to back panel
		block_index = (crystal_id.sector);
		crystal_index = 720 * block_index + crystal_id.crystal;
	}
	else if (crystal_id.system == 1){//belongs to front and racetrack
		block_index = (crystal_id.sector);
		crystal_index = 676 * block_index + crystal_id.crystal + 6480;
	}


	return crystal_index;

}

int encode_crystal_TBPET_vision(volume_id crystal_id){
	int crystal_index;
	int block_index;

	if (crystal_id.system == 0){//belongs to back panel
		block_index = crystal_id.module + 8 * crystal_id.sector;
		crystal_index = 200 * block_index + crystal_id.crystal;
	}
	else if (crystal_id.system == 1){//belongs to front and racetrack
		block_index = (crystal_id.sector);
		crystal_index = 1024 * block_index + crystal_id.crystal + 17600;
	}


	return crystal_index;

}

int encode_crystal_USPET(volume_id crystal_id){
	int crystal_index;
	int block_index;

	if (crystal_id.system == 0){//belongs to back panel
		block_index = crystal_id.module + 8 * crystal_id.sector;
		crystal_index = 288 * block_index + crystal_id.crystal;
	}
	else if (crystal_id.system == 1){//belongs to front panel

		block_index = crystal_id.module + 8 * crystal_id.sector;
		crystal_index = 288 * block_index + crystal_id.crystal + 2304;
	}
	else if (crystal_id.system == 2){//belongs to front panel

		block_index = crystal_id.module + 2 * crystal_id.sector;
		crystal_index = 576 * block_index + crystal_id.crystal + 4608;
	}
	else if (crystal_id.system == 3){//belongs to front panel

		block_index = crystal_id.module + 2 * crystal_id.sector;
		crystal_index = 576 * block_index + crystal_id.crystal + 5760;
	}


	return crystal_index;


}

int encode_crystal_USPET_pos1(volume_id crystal_id){
	int crystal_index;
	int block_index;

	if (crystal_id.system == 0){//belongs to back panel
		block_index = crystal_id.module + 8 * crystal_id.sector;
		crystal_index = 288 * block_index + crystal_id.crystal;
	}
	else if (crystal_id.system == 1){//belongs to front panel

		block_index = crystal_id.module + 8 * crystal_id.sector;
		crystal_index = 288 * block_index + crystal_id.crystal + 2304;
	}
	else if (crystal_id.system == 2){//belongs to front panel

		block_index = crystal_id.module + 8 * crystal_id.sector;
		crystal_index = 288 * block_index + crystal_id.crystal + 4608;
	}
	else if (crystal_id.system == 3){//belongs to front panel

		block_index = crystal_id.module + 2 * crystal_id.sector;
		crystal_index = 576 * block_index + crystal_id.crystal + 6912;
	}

	return crystal_index;


}

int encode_crystal_USPET_pos2(volume_id crystal_id){
	int crystal_index;
	int block_index;

	if (crystal_id.system == 0){//belongs to back panel
		block_index = crystal_id.module + 8 * crystal_id.sector;
		crystal_index = 288 * block_index + crystal_id.crystal;
	}
	else if (crystal_id.system == 1){//belongs to front panel

		block_index = crystal_id.module + 8 * crystal_id.sector;
		crystal_index = 288 * block_index + crystal_id.crystal + 2304;
	}
	else if (crystal_id.system == 2){//belongs to front panel

		block_index = crystal_id.module + 8 * crystal_id.sector;
		crystal_index = 288 * block_index + crystal_id.crystal + 4608;
	}
	else if (crystal_id.system == 3){//belongs to front panel

		block_index = crystal_id.module + 2 * crystal_id.sector;
		crystal_index = 576 * block_index + crystal_id.crystal + 9216;
	}

	return crystal_index;


}


int encode_crystal_USPET_pos3(volume_id crystal_id){
	int crystal_index;
	int block_index;

	if (crystal_id.system == 0){//belongs to back panel
		block_index = crystal_id.module + 8 * crystal_id.sector;
		crystal_index = 288 * block_index + crystal_id.crystal;
	}
	else if (crystal_id.system == 1){//belongs to front panel

		block_index = crystal_id.module + 8 * crystal_id.sector;
		crystal_index = 288 * block_index + crystal_id.crystal + 2304;
	}
	else if (crystal_id.system == 2){//belongs to front panel

		block_index = crystal_id.module + 8 * crystal_id.sector;
		crystal_index = 288 * block_index + crystal_id.crystal + 4608;
	}
	else if (crystal_id.system == 3){//belongs to front panel

		block_index = crystal_id.module + 2 * crystal_id.sector;
		crystal_index = 576 * block_index + crystal_id.crystal + 11520;
	}

	return crystal_index;


}

int encode_crystal_lungPET_pos1(volume_id crystal_id){
	int crystal_index;
	int block_index;

	if (crystal_id.system == 0){//belongs to back panel
		block_index = crystal_id.module;
		crystal_index = 256 * block_index + crystal_id.crystal;
	}
	else if (crystal_id.system == 1){//belongs to front panel

		block_index = crystal_id.module;
		crystal_index = 256 * block_index + crystal_id.crystal + 18432;
	}

	return crystal_index;


}

int encode_crystal_lungPET_pos2(volume_id crystal_id){
	int crystal_index;
	int block_index;

	if (crystal_id.system == 0){//belongs to back panel
		block_index = crystal_id.module;
		crystal_index = 256 * block_index + crystal_id.crystal;
	}
	else if (crystal_id.system == 1){//belongs to front panel

		block_index = crystal_id.module;
		crystal_index = 256 * block_index + crystal_id.crystal + 24576;
	}

	return crystal_index;


}

int encode_crystal_lungPET_pos3(volume_id crystal_id){
	int crystal_index;
	int block_index;

	if (crystal_id.system == 0){//belongs to back panel
		block_index = crystal_id.module;
		crystal_index = 256 * block_index + crystal_id.crystal;
	}
	else if (crystal_id.system == 1){//belongs to front panel

		block_index = crystal_id.module;
		crystal_index = 256 * block_index + crystal_id.crystal + 30720;
	}

	return crystal_index;


}


int encode_crystal_mCT_Insert_doi(volume_id crystal_id){
	int crystal_index;
	int block_index;

	if (crystal_id.system == 0){//belongs to back panel
		block_index = (crystal_id.sector);
		crystal_index = 720 * block_index + crystal_id.crystal;
	}
	else if (crystal_id.system == 1){//belongs to front and racetrack
		block_index = (crystal_id.sector);
		crystal_index = 676 * block_index + crystal_id.crystal + 7920;
	}


	return crystal_index;

}

int encode_crystal_mCT_flat(volume_id crystal_id){
	int crystal_index;
	int block_index;

	//belongs to front and racetrack
	block_index = (crystal_id.sector);
	crystal_index = 676 * block_index + crystal_id.crystal;



	return crystal_index;

}

int encode_crystal_mCT_flat_geom(volume_id crystal_id){
	int crystal_index;
	int block_index;

	if (crystal_id.system == 0){//belongs to top panel at 50

		crystal_index = crystal_id.crystal;
	}
	else if (crystal_id.system == 1){//belongs to bottom panel at -50

		crystal_index = crystal_id.crystal + 250;
	}

	return crystal_index;

}


int encode_crystal_mCT_Insert_ori(volume_id crystal_id){
	int crystal_index;
	int block_index;

	if (crystal_id.system == 0){//belongs to Scanner
		block_index = crystal_id.module + 4 * crystal_id.sector;
		crystal_index = 169 * block_index + crystal_id.crystal;
	}
	else{//belongs to insert
		block_index = crystal_id.module;
		crystal_index = 256 * block_index + crystal_id.crystal + 70775;
	}

	return crystal_index;
}

int encode_crystal_USPET_modified(volume_id crystal_id){
	int crystal_index;
	int block_index;

	if (crystal_id.system == 0){//belongs to back panel
		block_index = crystal_id.module;
		crystal_index = 64 * block_index + crystal_id.crystal;
	}
	else if (crystal_id.system == 1){//belongs to front panel

		block_index = crystal_id.module;
		crystal_index = 64 * block_index + crystal_id.crystal + 256;
	}
	else if (crystal_id.system == 2){//belongs to front panel

		block_index = crystal_id.module;
		crystal_index = 64 * block_index + crystal_id.crystal + 512;
	}
	else if (crystal_id.system == 3){//belongs to front panel

		block_index = crystal_id.module;
		crystal_index = 64 * block_index + crystal_id.crystal + 768;
	}

	return crystal_index;


}
