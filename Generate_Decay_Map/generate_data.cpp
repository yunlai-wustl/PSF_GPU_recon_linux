/*
* generate_final list mode file with number_of_coincevents_for each LOR for normalization
*
*  
*  Author: Suranjana Samanta
*/
#include "generate_data.h"

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

int record_IS(mini_coinc_event *coinc_ptr_1, mini_coinc_event *coinc_ptr_2){
	//record this coinc
	coinc_ptr_2->event_id = coinc_ptr_1->event_id;
	coinc_ptr_2->crystal_index_1 = coinc_ptr_1->crystal_index_1;
	coinc_ptr_2->crystal_index_2 = coinc_ptr_1->crystal_index_2;
	coinc_ptr_2->time_1 = coinc_ptr_1->time_1;
	coinc_ptr_2->diff_time = coinc_ptr_1->diff_time;
	//coinc_ptr_2->source_pos = coinc_ptr_1->source_pos;
	//recorded
	return 0;
}

int main(int argc, char **argv)
{
	int sorting_method;
	int x, y, z;
	mini_coinc_event *compact_coinc_data, *scatter_data;
	mini_coinc_event *temp;
	pos *pos_list;
	size_t t, t1;
	long long int size, size1;

	int i;

	char *file_names_ptr, *file_names_ptr1;

	int data_length, data_length1;
	int true_coincidence;

	struct stat64 st, st1;

	FILE *current_file;
	FILE *fp;
	char coin_filename[128] = "number_events_perLOR";
	int size_of_each_write = 10;
	int num_coincidence, num_coincidence1;
	int entry_length, entry_length1;
	int NUM_MAX = 5000;
	if (argc<3){
		printf("Not enough arguments, please use -help option.\n");
		return 0;
	}

	if (strcmp(argv[1], "-help") == 0){
		printf("Here describes how to run this program.\n");
		printf("mini for mini coincidence file input.\n");
		printf("gate for GATE original coincidence file input.\n");
		return 0;
	}
	else if (strcmp(argv[1], "mini") == 0){
		sorting_method = 0;
		printf("Using \"mini\" file input.\n");
	}
	else if (strcmp(argv[1], "gate") == 0){
		sorting_method = 1;
		printf("Using \"prompt\" sorting method.\n");
	}
	else{
		//use default sorting method
		sorting_method = 0;
		printf("Unrecognized sorting method, using \"true\" instead.\n");
	}
	//*****************************END*****program argument*******************************************//

	//*****************************check file length***********************************************//
	entry_length = sizeof(mini_coinc_event);

	printf("Following coincidence files are being processed:\n");

	file_names_ptr = argv[2];
	stat64(file_names_ptr, &st);
	size = st.st_size;
	size /= entry_length;
	if (size >= INT_MAX){
		printf("File \"%s\" length is tooooooooooo long. This program only supports singles event fewer than %d. Current file has %I64d. Program stopped\n", file_names_ptr, INT_MAX, size);
		return 0;
	}
	data_length = size;
	printf("%d coincidence event in file \"%s\"\n", data_length, file_names_ptr);

	num_coincidence = data_length;

	//***********************************Read data and generate decay map****************************************//
	compact_coinc_data = (mini_coinc_event*)calloc(num_coincidence,sizeof(mini_coinc_event));
	//int	total_num_coinc_IS = (int)(0.25*num_coincidence);
	temp = (mini_coinc_event*)calloc((num_coincidence/size_of_each_write),sizeof(mini_coinc_event));

	current_file = fopen(file_names_ptr, "rb");
	t = fread(compact_coinc_data, sizeof(mini_coinc_event), num_coincidence, current_file);
	if (num_coincidence == (int)t){
		printf("......%s read successfully.\n", file_names_ptr);
	}
	else{
		printf("......%s read error. %d elements were read\n", file_names_ptr, (int)t);
	}
	fclose(current_file);


	/////////////////////////////////////////////////////////////////////////////////////////////////////////////
	entry_length1 = sizeof(mini_coinc_event);

	file_names_ptr1 = argv[3];
	stat64(file_names_ptr1, &st1);
	size1 = st1.st_size;
	size1 /= entry_length1;
	if (size1 >= INT_MAX){
		printf("File \"%s\" length is tooooooooooo long. This program only supports singles event fewer than %d. Current file has %I64d. Program stopped\n", file_names_ptr1, INT_MAX, size1);
		return 0;
	}
	data_length1 = size1;
	printf("%d coincidence event in file \"%s\"\n", data_length1, file_names_ptr1);

	num_coincidence1 = data_length1;
	scatter_data = (mini_coinc_event*)calloc(num_coincidence1, sizeof(mini_coinc_event));
	//int	total_num_coinc_IS = (int)(0.25*num_coincidence);


	current_file = fopen(file_names_ptr1, "rb");
	t1 = fread(scatter_data, sizeof(mini_coinc_event), num_coincidence1, current_file);
	if (num_coincidence1 == (int)t1){
		printf("......%s read successfully.\n", file_names_ptr1);
	}
	else{
		printf("......%s read error. %d elements were read\n", file_names_ptr1, (int)t1);
	}
	fclose(current_file);



	/*ofstream myfile2;
	char str2[64];
	strcpy(str2,coin_filename);
	strcat(str2,".txt");
	myfile2.open(str2);
	for (int j = 0; j < num_coincidence; j++){
	//if (mini_coinc_data[j].time_1 <= 10){
	myfile2<<mini_coinc_data[j].event_id<<"  "<<mini_coinc_data[j].crystal_index_1<<"  "<<mini_coinc_data[j].crystal_index_2<<"  "<<mini_coinc_data[j].time_1<<"  "<<mini_coinc_data[j].diff_time<<endl;
	//myfile2 << mini_coinc_data[j].diff_time << endl;
	//}
	}
	myfile2.close();*/
// 0,60800,93568,126336
	int r = 0;
	int block_crystal1 = 1; // Number of crystals in one module of volume 1
	int block_crystal2 = 1; // Number of crystals in one module of volume 2
	int sn1 = 0; // Starting crystal number of volume 1
	int fn1 = 60800; // Ending crystal number of volume 1

	int sn2 = 0; // Starting crystal number of volume 2
	int fn2 = 60800; // Ending crystal number of volume 2

	int bn = fn1 - sn1; // number of block modules for detector volume 1
	int bm = fn2 - sn2; // number of block modules for detector volume 2
	/*int* Cut_idx_1 = new int[bn+1]; // for sensitivity image generation only
	for (int i = 0; i <= bn; i++){
		Cut_idx_1[i] = sn1 + i* block_crystal1;
	}
	int* Cut_idx_2 = new int[bm+1]; // for sensitivity image generation only
	for (int i = 0; i <= bm; i++){
		Cut_idx_2[i] = sn2 + i* block_crystal2;
	}

	int** tra = new int*[bn];
	for (int i = 0; i < bn; i++)
		tra[i] = new int[bm];*/
	
	
	/*float **tra = (float **)calloc(32768, sizeof(float *));
	for (int i = 0; i < 5; i++)
		tra[i] = (float *)calloc(32768, sizeof(float));*/

	//int tra[32768][8192] = { { 0 } };
	//cout << "\n here";

	/*for (int i = 0; i < bn; i++){
		for (int j = 0; j < bm; j++){
			tra[i][j] = 0;
		}
	}
	


	cout << "Total coincidences = "<<num_coincidence;

	int oo = 0;
	for (int i = 0; i < num_coincidence; i++){

		if (((compact_coinc_data[i].crystal_index_1 >= sn1 && compact_coinc_data[i].crystal_index_1< Cut_idx_1[bn]) && (compact_coinc_data[i].crystal_index_2 >= sn2 && compact_coinc_data[i].crystal_index_2 < Cut_idx_2[bm])) || ((compact_coinc_data[i].crystal_index_2 >= sn1 && compact_coinc_data[i].crystal_index_2< Cut_idx_1[bn]) && (compact_coinc_data[i].crystal_index_1 >= sn2 && compact_coinc_data[i].crystal_index_1 < Cut_idx_2[bm]))){
			
			oo = oo + 1;
			int j = 0; int k = 0; int inter = 0;
			//cout << "\n coinc data: " << compact_coincidence_data[i].crystal_index_1 <<" "<< compact_coincidence_data[i].crystal_index_2;
			inter = (compact_coinc_data[i].crystal_index_1 - sn1) / block_crystal1;
		
			//cout << "\n inter: " << inter;
			if (inter > bn - 1 || inter < 0){
				j = (int)floor((compact_coinc_data[i].crystal_index_2 - sn1) / block_crystal1);
				k = (int)floor((compact_coinc_data[i].crystal_index_1 - sn2) / block_crystal2);

			}
			else{
				j = (int)floor((compact_coinc_data[i].crystal_index_1 - sn1) / block_crystal1);
				k = (int)floor((compact_coinc_data[i].crystal_index_2 - sn2) / block_crystal2);


			}
			
			tra[j][k] = tra[j][k] + 1;
			if ((j > 60800 || j < 0) || (k>65536 || k < 0) || (tra[j][k] > 4 || tra[j][k] < 1)){

				cout << "\n j " << j << " k " << k << " value " << tra[j][k] << "\n";
			}

		}
	if(i%50000000 == 0) cout << "\n Current event number " << i;
	}

	int sum = 0;
	for (int j = 0; j < bn; j++){
		for (int k = 0; k < bm; k++){
			sum = sum + tra[j][k];
		}
	}
	cout << "\n Sum : " << sum << " oo events : "<<oo<<" num_coinc : " << num_coincidence;*/
	/*
	for (int i = 0; i < num_coincidence; i++){

		int ci1 = compact_coinc_data[i].crystal_index_1; int ci2 = compact_coinc_data[i].crystal_index_2;

		if ((ci1 > sn1 && ci1< Cut_idx_1[bn]) && (ci2 > sn2 && ci2 < Cut_idx_2[bm])){
			temp[i].event_id = compact_coinc_data[i].event_id;
			temp[i].crystal_index_1 = ci1;
			temp[i].crystal_index_2 = ci2;
			temp[i].time_1 = compact_coinc_data[i].time_1;
			temp[i].diff_time = compact_coinc_data[i].diff_time;
			temp[i].source_pos = compact_coinc_data[i].source_pos;
			temp[i].norm_coeff = tra[(ci1 - sn1)/block_crystal1][(ci2 - sn2)/block_crystal2];

		}
		else if ((ci1 > sn2 && ci1< Cut_idx_2[bm]) && (ci2 > sn1 && ci2 < Cut_idx_1[bn])){
			temp[i].event_id = compact_coinc_data[i].event_id;
			temp[i].crystal_index_1 = ci1;
			temp[i].crystal_index_2 = ci2;
			temp[i].time_1 = compact_coinc_data[i].time_1;
			temp[i].diff_time = compact_coinc_data[i].diff_time;
			temp[i].source_pos = compact_coinc_data[i].source_pos;
			temp[i].norm_coeff = tra[(ci2 - sn1) / block_crystal1][(ci1 - sn2) / block_crystal2];
		}
	}*/

	/*ofstream myfile2;
	char str2[64];
	strcpy(str2,coin_filename);
	strcat(str2,".txt");
	myfile2.open(str2);
	for (int j = 0; j < bn; j++){
		for (int k = 0; k < bm; k++){

			myfile2 << tra[j][k]<<"\t";
		}
	myfile2 << std::endl;
	}
	myfile2.close();*/


	int num_coinc3 = 0;
	int step = 0;
	int cutidx[11] = { 0 }; // Change this with size_of_each_write
	cutidx[10] = num_coincidence;
	for (int i = 0; i < size_of_each_write; i++){
		step = floor(num_coincidence / size_of_each_write);
		cutidx[i] = i*step;
		cout << "\n Pointer at: " << cutidx[i];

	}

	/*for (int i = 0; i < size_of_each_write; i++){
		int num_coinc2 = 0;
		cout << "\n Now scanning events from " << cutidx[i] << " to " << cutidx[i + 1];
		for (int j = cutidx[i]; j < cutidx[i + 1]; j++){
			//if (compact_coinc_data[j].time_1 < 120 && compact_coinc_data[j].diff_time >= 0){
			//if (((compact_coinc_data[j].crystal_index_1 >= sn1 && compact_coinc_data[j].crystal_index_1< fn1) && (compact_coinc_data[j].crystal_index_2 >= sn2 && compact_coinc_data[j].crystal_index_2 < fn2)) || ((compact_coinc_data[j].crystal_index_2 >= sn1 && compact_coinc_data[j].crystal_index_2< fn1) && (compact_coinc_data[j].crystal_index_1 >= sn2 && compact_coinc_data[j].crystal_index_1 < fn2))){
			//if (compact_coinc_data[i].crystal_index_1 != 0 && compact_coinc_data[i].crystal_index_2 != 0 && compact_coinc_data[i].time_1 !=0) {
			//if ((compact_coinc_data[j].crystal_index_1 != 0 || compact_coinc_data[j].crystal_index_2 != 0 || compact_coinc_data[j].time_1 != 0) && compact_coinc_data[j].crystal_index_2 <= 126336 ) {
				 
			temp[num_coinc2] = compact_coinc_data[j];
				num_coinc2++;
				num_coinc3++;
			}

		}
		printf("......%d coincidence events were returned in %d iteration.\n", num_coinc2, i);
		fp = fopen("scatter.lst", "ab+");
		fwrite(temp, sizeof(mini_coinc_event), num_coinc2, fp);
		fclose(fp);
	}
	*/

	double timestamp = 0;
		timestamp = compact_coinc_data[num_coincidence - 1].time_1;
	cout << " \n Time of last event: " << timestamp;
	
	//cout << "\n Total events written: " << num_coinc3;
	
	

//int num_coinc2 = 0;

	/*for (int i = 0; i < num_coincidence; i++){
		if (compact_coinc_data[i].crystal_index_1 > 126336 || compact_coinc_data[i].crystal_index_2 >126336 || compact_coinc_data[i].crystal_index_1 <0  || compact_coinc_data[i].crystal_index_2 < 0) {
		//if (compact_coinc_data[i].event_id == 54){
			num_coinc2++;
		cout << "\n eventID: " << compact_coinc_data[i].event_id << " ID1: " << compact_coinc_data[i].crystal_index_1 << " ID2: " << compact_coinc_data[i].crystal_index_2;
		}
	}
	cout << "\n " << num_coinc2 << " bad events from a total of " << num_coincidence << " events!!";
	*/

	/*cout << "\n First event :  eventID: " << compact_coinc_data[0].event_id << " ID1: " << compact_coinc_data[0].crystal_index_1 << " ID2: " << compact_coinc_data[0].crystal_index_2;
	cout << "\n Second event :  eventID: " << compact_coinc_data[1].event_id << " ID1: " << compact_coinc_data[1].crystal_index_1 << " ID2: " << compact_coinc_data[1].crystal_index_2;
	cout << "\n Third event :  eventID: " << compact_coinc_data[2].event_id << " ID1: " << compact_coinc_data[2].crystal_index_1 << " ID2: " << compact_coinc_data[2].crystal_index_2;

	cout << "\n Event at n=21254 :  eventID: " << compact_coinc_data[21254].event_id << " ID1: " << compact_coinc_data[21254].crystal_index_1 << " ID2: " << compact_coinc_data[21254].crystal_index_2;
	cout << "\n Event at n=21255 :  eventID: " << compact_coinc_data[21255].event_id << " ID1: " << compact_coinc_data[21255].crystal_index_1 << " ID2: " << compact_coinc_data[21255].crystal_index_2;
	cout << "\n Event at n=5354 :  eventID: " << compact_coinc_data[5354].event_id << " ID1: " << compact_coinc_data[5354].crystal_index_1 << " ID2: " << compact_coinc_data[5354].crystal_index_2;
	cout << "\n Event at n=5355 :  eventID: " << compact_coinc_data[5355].event_id << " ID1: " << compact_coinc_data[5355].crystal_index_1 << " ID2: " << compact_coinc_data[5355].crystal_index_2;
	cout << "\n Event at n=5356 :  eventID: " << compact_coinc_data[5356].event_id << " ID1: " << compact_coinc_data[5356].crystal_index_1 << " ID2: " << compact_coinc_data[5356].crystal_index_2;
	cout << "\n Event at n=5357 :  eventID: " << compact_coinc_data[5357].event_id << " ID1: " << compact_coinc_data[5357].crystal_index_1 << " ID2: " << compact_coinc_data[5357].crystal_index_2;
	*/
	

	/*for (int i = 0; i < 50; i++){

		//cout << "\n src_pos x: " << compact_coinc_data[i].source_pos.x << " src_pos y: " << compact_coinc_data[i].source_pos.y << " src_pos z: " << compact_coinc_data[i].source_pos.z;
		if ((compact_coinc_data[i].source_pos.x >= -0.5 && compact_coinc_data[i].source_pos.x <= 0.5) && (compact_coinc_data[i].source_pos.y >= -.5 && compact_coinc_data[i].source_pos.y <= .5) && (compact_coinc_data[i].source_pos.z >= -.5 && compact_coinc_data[i].source_pos.z <= .5)){
			num_coinc2++;
		}
			
	}
	cout << "\n " << num_coinc2 << " events in central voxel out of " << num_coincidence << " events!!";*/

	/*int num_events = 50;
	int j = 0;
	for (int i = 0; i < num_coincidence; i++){
		if ((compact_coinc_data[i].crystal_index_1 >= 17180 && compact_coinc_data[i].crystal_index_1 <= 17220 && compact_coinc_data[i].crystal_index_2 >= 47580 && compact_coinc_data[i].crystal_index_2 <= 47620) || (compact_coinc_data[i].crystal_index_2 >= 17180 && compact_coinc_data[i].crystal_index_2 <= 17220 && compact_coinc_data[i].crystal_index_1 >= 47580 && compact_coinc_data[i].crystal_index_1 <= 47620)){
			temp[j] = compact_coinc_data[i];
			cout << "\n " << temp[j].crystal_index_1 << "   " << temp[j].crystal_index_2;
			j++;
		}
	}

	fp = fopen("central_LOR_events_y.lst", "ab+");
	fwrite(temp, sizeof(mini_coinc_event), j, fp);
	fclose(fp);

	cout << "\n " << j << " events recorded in file out of " << num_coincidence << " events!!"; 

	*/


	



	/*printf("......%d coincidence events were returned.\n", num_coincidence);
	fp = fopen("OO_with_norm_factors.lst", "ab+");
	fwrite(temp, sizeof(mini_coinc_event_norm), num_coincidence, fp);
	fclose(fp);*/



	//free(temp);

	for (int i = 0; i < size_of_each_write; i++){
	int num_coinc2 = 0;
	cout << "\n Now scanning events from " << cutidx[i] << " to " << cutidx[i + 1];
	for (int j = cutidx[i]; j < cutidx[i + 1]; j++){
		if (scatter_data[j].crystal_index_1 == 0 && scatter_data[j].crystal_index_2 == 0 && scatter_data[j].time_1 == 0){
		temp[num_coinc2] = compact_coinc_data[j];
		num_coinc2++;
		num_coinc3++;
	}

}
printf("......%d coincidence events were returned in %d iteration.\n", num_coinc2, i);
fp = fopen("true.lst", "ab+");
fwrite(temp, sizeof(mini_coinc_event), num_coinc2, fp);
fclose(fp);
}

	cout << "\n Total events written: " << num_coinc3;



	free(compact_coinc_data);
	free(temp);
	free(scatter_data);
	return 0;
}



