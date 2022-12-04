/*
* Read Scatter Coefficient file and generate new listmode data file
*  Author: Suranjana Samanta
*  05/15/2022 add the part reading random/random and scatter
*/

#define _CRT_SECURE_NO_WARNINGS
#include "process_scatter.h"
#include <iostream>
#include <fstream>
#include <string.h>
#include <vector>
#include <stdio.h>
#include <cmath>
#include <math.h>
#include <stdlib.h>
#include <limits.h>
#include <sys/stat.h>

#define MAX_COINC 10000000
#define TOF_BIN_SIZE 13.021
#define TOF_WINDOW 4710.0f
#define SCATTER_SCALE_SS 150.0f
#define SCATTER_SCALE_OS 1500.0f
#define SCATTER_SCALE_OO 15000.0f
#define NUM_CRYSTAL_SCANNER 60800
using namespace std;
void record_scatter_coinc(vector<tof_coinc_event_cbm_scatter>& mini_scatter_coinc_event, int& num_coinc2, vector<vector<float>>& randomMatrix, float* crystal_efficency, tof_coinc_event_cbm* coinc_data);


int main(int argc, char **argv)
{
	int method;
	tof_coinc_event_cbm*coinc_data;
	tof_coinc_event_cbm_scatter *temp;
	double *scatter_data;
	size_t t, t1;
	long long int size, size1;
	int event_count = 0;
	int i;
	int Xsize;
	char *file_names_ptr, *file_names_ptr1;

	int data_length, data_length1;
	int n_SS_0 = 0;
	int n_SS = 0;
	int n_OS = 0;
	vector<int> n_OS_0(4,0);
	int n_OO_0 = 0;
	int n_OO = 0;
	struct stat64 st, st1;
	ofstream output("scatter_added_file.lst", ios_base::binary);

	FILE *current_file;
	FILE *fp;
	char coin_filename[128] = "check_file";
	float* crystal_efficency = (float*)malloc(60800 * sizeof(float));
	vector<vector<float>> randomMatrix(304, vector<float>(304));

	int num_coincidence, num_coincidence1;
	int entry_length, entry_length1;
	int NUM_MAX = 5000;
	if (argc <= 3){
		printf("Not enough arguments, please use -help option.\n");
		return 0;
	}

	if (strcmp(argv[1], "-help") == 0){
		printf("Here describes how to run this program.\n");
		printf("scatter for adding scatter only.\n");
		printf("scatter_random for adding scatter and random coefficients.\n");
		return 0;
	}
	else if (strcmp(argv[1], "scatter") == 0){
		method = 0;
		printf("Adding scatter coefficients only!\n");
	}
	else if (strcmp(argv[1], "random") == 0){
		method = 1;
		printf("Adding random coefficients only!\n");
		ifstream fileRandomMatrix;
		fileRandomMatrix.open(argv[3]);
		cout << "reading random matrix" << endl;
		while (!fileRandomMatrix.eof()) {
			for (int i = 0; i<304; i++){
				for (int j = 0; j<304; j++){
					fileRandomMatrix >> randomMatrix[i][j];
				}
			}
		}
		FILE* crystalEffciencyFile = fopen(argv[4], "rb");
		t1 = fread(crystal_efficency, sizeof(float), NUM_CRYSTAL_SCANNER, crystalEffciencyFile);
		if ((int)t1 == NUM_CRYSTAL_SCANNER){
			printf("......%s read successfully.\n", argv[4]);
		}
		else{
			printf("......%s read error. %d elements were read\n", argv[4], (int)t1);
		}
		fclose(crystalEffciencyFile);
	}
	else if (strcmp(argv[1], "random_scatter") == 0){
		method = 2;
		printf("Adding random and scatter coefficients !\n");
		cout << "scaling coeff for SS scatter coincidence: " << SCATTER_SCALE_SS << endl;
		cout << "scaling coeff for OS scatter coincidence: " << SCATTER_SCALE_OS << endl;
		cout << "scaling coeff for OO scatter coincidence: " << SCATTER_SCALE_OO << endl;
		cout << "time bin for random coincidence: " << TOF_BIN_SIZE << "ps" << endl;
		ifstream fileRandomMatrix;
		fileRandomMatrix.open(argv[3]);
		cout << "reading random matrix" << endl;
		while (!fileRandomMatrix.eof()) {
			for (int i = 0; i<304; i++){
				for (int j = 0; j<304; j++){
					fileRandomMatrix >> randomMatrix[i][j];
				}
			}
		}
		FILE* crystalEffciencyFile = fopen(argv[4], "rb");
		t1 = fread(crystal_efficency, sizeof(float), 60800, crystalEffciencyFile);
		if ((int)t1 == 60800){
			printf("......%s read successfully.\n", argv[4]);
		}
		else{
			printf("......%s read error. %d elements were read\n", argv[4], (int)t1);
		}
		fclose(crystalEffciencyFile);
	}
	else{
		//use default i.e. adds scatter coefficients only
		method = 0;
		printf("Unrecognized  method, using default (adding scatter coefficients only).\n");
	}
	//*****************************END*****program argument*******************************************//

	//*****************************check file length***********************************************//

	//**************** FILE 1 : ListMode File *********************************
	entry_length = sizeof(tof_coinc_event_cbm);

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

	//***********************************Read data ****************************************//
	coinc_data = (tof_coinc_event_cbm*)malloc(num_coincidence*sizeof(tof_coinc_event_cbm));
	//int	total_num_coinc_IS = (int)(0.25*num_coincidence);
	vector<tof_coinc_event_cbm_scatter> coinc_event_list;
	temp = (tof_coinc_event_cbm_scatter*)calloc(num_coincidence / 10, sizeof(tof_coinc_event_cbm_scatter));

	current_file = fopen(file_names_ptr, "rb");
	t = fread(coinc_data, sizeof(tof_coinc_event_cbm), num_coincidence, current_file);
	if (num_coincidence == (int)t){
		printf("......%s read successfully.\n", file_names_ptr);
	}
	else{
		printf("......%s read error. %d elements were read\n", file_names_ptr, (int)t);
	}
	fclose(current_file);

	double time_tag = 0, maxtime = 0;
	for (i = 0; i < num_coincidence; i++){
		time_tag = coinc_data[i].time_1;

	}
	maxtime = coinc_data[num_coincidence - 1].time_1;
	cout << "\n Time tag : " << time_tag << " maxtime: " << maxtime << "\n";


	if (method == 0){
		//*****************FILE 2 : SCATTER DAT FILE*****************************
		entry_length1 = sizeof(double);
		printf("Following coincidence files are being processed:\n");
		file_names_ptr1 = argv[3];
		if (stat64(file_names_ptr1, &st1) < 0) {
			printf("Read file error: %s \n", file_names_ptr1);
			return 0;
		}
		size1 = st1.st_size;
		size1 /= entry_length1;
		if (size1 >= INT_MAX){
			printf("File \"%s\" length is tooooooooooo long. This program only supports singles event fewer than %d. Current file has %I64d. Program stopped\n", file_names_ptr1, INT_MAX, size1);
			return 0;
		}
		data_length1 = size1;
		printf("%d coincidence event in file \"%s\"\n", data_length1, file_names_ptr1);

		cout << "\n Data_length for coinc file : " << data_length << " Data_length for scatter file : " << data_length1;

		if (data_length1 != data_length + 1){ // Last entry of scatter file is 0... Need to check this !!!
			cout << "\n Number of coinc events in listmode file and scatter data file doesn't match!!!!";
			return 0;
		}
		num_coincidence1 = data_length1;
		//***********************************Read data ****************************************//
		scatter_data = (double*)calloc(num_coincidence1, sizeof(double));
		current_file = fopen(file_names_ptr1, "rb");
		t1 = fread(scatter_data, sizeof(double), num_coincidence1, current_file);
		if (num_coincidence1 == (int)t1){
			printf("......%s read successfully.\n", file_names_ptr1);
		}
		else{
			printf("......%s read error. %d elements were read\n", file_names_ptr1, (int)t1);
		}
		fclose(current_file);
		for (int i = 0; i < 100; i++)
			printf("line %d: %.17g \n",i,scatter_data[i]);


		int step = 0;
		int cutidx[11] = { 0 };
		double time_ptr = 0;
		cutidx[10] = num_coincidence;
		for (int i = 0; i < 10; i++){
			step = floor(num_coincidence / 10);
			cutidx[i] = i*step;
			cout << "\n Pointer at: " << cutidx[i];			
		}
		float sum_scatter = 0;
		float mean_scatter = 0;
		float maxs = 0, mins = scatter_data[0] * SCATTER_SCALE_SS;
		int crystal1 = 0;
		int crystal2 = 0;
		//read and write 10 times
		for (int i = 0; i < 10; i++){
			int num_coinc2 = 0;
			cout << "\n Now scanning events from " << cutidx[i] << " to " << cutidx[i + 1];
			for (int j = cutidx[i]; j < cutidx[i + 1]; j++){
				temp[num_coinc2].crystal_index_1 = coinc_data[j].crystal_index_1;
				temp[num_coinc2].crystal_index_2 = coinc_data[j].crystal_index_2;
				temp[num_coinc2].time_1 = coinc_data[j].time_1;
				temp[num_coinc2].diff_time = coinc_data[j].diff_time;
				//check coincidence type and scale scatter accordingly 
				if (temp[num_coinc2].crystal_index_1 < NUM_CRYSTAL_SCANNER && temp[num_coinc2].crystal_index_2 < NUM_CRYSTAL_SCANNER)
				{
					n_SS++;
					temp[num_coinc2].scatter_coeff = float(scatter_data[j])*SCATTER_SCALE_SS;
					if (scatter_data[j] == 0)
						n_SS_0++;
				}
				else if (temp[num_coinc2].crystal_index_1 >= NUM_CRYSTAL_SCANNER && temp[num_coinc2].crystal_index_2 >= NUM_CRYSTAL_SCANNER)
				{
					n_OO++;
					temp[num_coinc2].scatter_coeff = float(scatter_data[j])*SCATTER_SCALE_OO;
					if (scatter_data[j] == 0)
						n_OO_0++;
				}
				else
				{
					n_OS++;
					temp[num_coinc2].scatter_coeff = float(scatter_data[j])*SCATTER_SCALE_OS;
					if (scatter_data[j] == 0)
						n_OS_0[(temp[num_coinc2].crystal_index_2 - 60800) / 16384]++;
				}

				if (temp[num_coinc2].scatter_coeff > maxs){
					maxs = temp[num_coinc2].scatter_coeff;
				}
				if (temp[num_coinc2].scatter_coeff < mins){
					mins = temp[num_coinc2].scatter_coeff;
				}
				sum_scatter += temp[num_coinc2].scatter_coeff;
				num_coinc2++;
			}

			cout << "\n Time for last written event : " << time_ptr << "\n";
			printf("......%d coincidence events were returned in %d iteration.\n", num_coinc2, i);
			fp = fopen("File_with_scatter.lst", "ab+");
			fwrite(temp, sizeof(tof_coinc_event_cbm_scatter), num_coinc2, fp);
			fclose(fp);
		}
		mean_scatter = sum_scatter / num_coincidence;
		cout << "\n Mean scatter : " << mean_scatter;
		cout << "\n Max scatter : " << maxs;
		cout << "\n Min scatter : " << mins;
		cout << "\n Total number of SS events:"<<n_SS<<"number of SS events with scatter coeff =0 " << n_SS_0 << endl;
		cout << "\n Total number of OS events:" << n_OS << endl;
		cout << "\n number of OS events with scatter coeff =0, detector 0 (60800-77183) "<< n_OS_0[0] << endl;
		cout << "\n number of OS events with scatter coeff =0, detector 1 (77184-93567) " << n_OS_0[1] << endl;
		cout << "\n number of OS events with scatter coeff =0, detector 2 (93568-109951) " << n_OS_0[2] << endl;
		cout << "\n number of OS events with scatter coeff =0, detector 3 (109952-126336) " << n_OS_0[3] << endl;
		cout << "\n Total number of OO events:" << n_OS << endl;
		cout << "\n number of OO events with scatter coeff =0," << n_OO_0 << endl;

		cout << "\n Total events written: " << num_coincidence;
		// Check how output file looks like using last few events
		ofstream myfile2;
		char str2[64];
		strcpy(str2, coin_filename);
		strcat(str2, ".txt");
		myfile2.open(str2);
		for (int j = 0; j < 100; j++){
			myfile2 << temp[j].crystal_index_1 << "  " << temp[j].crystal_index_2 << "  " << temp[j].time_1 << "  " << temp[j].diff_time << " " << temp[j].bed_position<<" "<< temp[j].scatter_coeff << endl;
		}
		myfile2.close();
		free(coinc_data);
		free(scatter_data);
		free(temp);
		return 0;
	}
	else if (method == 1){
		//---------already read random matrix and crystal efficiency ------------------------//
		cout << "processing data" << endl;
		while (event_count < num_coincidence){
			coinc_event_list.reserve(MAX_COINC);
			float sum_scatter = 0;
			float mean_scatter = 0;
			Xsize = coinc_event_list.size();
			if (Xsize < MAX_COINC){
				//record_cbm_coinc(coinc_event_list, crystal1, crystal2, time_elapsed, (float)tof, current_bed_position);
				record_scatter_coinc(coinc_event_list, event_count, randomMatrix, crystal_efficency, coinc_data);
				event_count++;
			}
			//if the count reaches max count, write out buffer and reset the vector
			else
			{
				cout << "writing out " << Xsize << " coincidence events" << endl;
				output.write(reinterpret_cast<char *>(&coinc_event_list[0]), coinc_event_list.size()*sizeof(tof_coinc_event_cbm_scatter));
				coinc_event_list.clear();
			}
		}
		Xsize = coinc_event_list.size();
		cout << "writing out " << Xsize << " coincidence events" << endl;
		//output.write(reinterpret_cast<char *>(&coinc_event_list[0]), coinc_event_list.size()*sizeof(tof_coinc_event_cbm));
		output.write(reinterpret_cast<char *>(&coinc_event_list[0]), coinc_event_list.size()*sizeof(tof_coinc_event_cbm_scatter));

		output.close();
		//std::vector<tof_coinc_event_cbm>().swap(coinc_event_list); //free memory
		std::vector<tof_coinc_event_cbm_scatter>().swap(coinc_event_list); //free memory

		return 0;
	}
	else if (method == 2){
		//*****************FILE 2 : SCATTER DAT FILE*****************************
		entry_length1 = sizeof(double);
		printf("Following coincidence files are being processed:\n");
		file_names_ptr1 = argv[5];
		if (stat64(file_names_ptr1, &st1) < 0) {
			printf("Read file error: %s \n", file_names_ptr1);
			return 0;
		}
		size1 = st1.st_size;
		size1 /= entry_length1;
		if (size1 >= INT_MAX){
			printf("File \"%s\" length is tooooooooooo long. This program only supports singles event fewer than %d. Current file has %I64d. Program stopped\n", file_names_ptr1, INT_MAX, size1);
			return 0;
		}
		data_length1 = size1;
		printf("%d coincidence event in file \"%s\"\n", data_length1, file_names_ptr1);

		cout << "\n Data_length for coinc file : " << data_length << " Data_length for scatter file : " << data_length1;

		if (data_length1 != data_length + 1){ // Last entry of scatter file is 0... Need to check this !!!
			cout << "\n Number of coinc events in listmode file and scatter data file doesn't match!!!!";
			return 0;
		}
		num_coincidence1 = data_length1;
		//***********************************Read data ****************************************//
		scatter_data = (double*)calloc(num_coincidence1, sizeof(double));
		current_file = fopen(file_names_ptr1, "rb");
		t1 = fread(scatter_data, sizeof(double), num_coincidence1, current_file);
		if (num_coincidence1 == (int)t1){
			printf("......%s read successfully.\n", file_names_ptr1);
		}
		else{
			printf("......%s read error. %d elements were read\n", file_names_ptr1, (int)t1);
		}
		fclose(current_file);
		int step = 0;
		int cutidx[11] = { 0 };
		double time_ptr = 0;
		cutidx[10] = num_coincidence;
		for (int i = 0; i < 10; i++){
			step = floor(num_coincidence / 10);
			cutidx[i] = i*step;
			cout << "\n Pointer at: " << cutidx[i];
		}
		float sum_scatter = 0;
		float mean_scatter = 0;
		float maxs = 0, mins = scatter_data[0] * SCATTER_SCALE_SS;
		int crystal1 = 0;
		int crystal2 = 0;
		int block1, block2;
		//read and write 10 times
		for (int i = 0; i < 10; i++){
			int num_coinc2 = 0;
			cout << "\n Now scanning events from " << cutidx[i] << " to " << cutidx[i + 1];
			for (int j = cutidx[i]; j < cutidx[i + 1]; j++){
				temp[num_coinc2].crystal_index_1 = coinc_data[j].crystal_index_1;
				temp[num_coinc2].crystal_index_2 = coinc_data[j].crystal_index_2;
				temp[num_coinc2].time_1 = coinc_data[j].time_1;
				temp[num_coinc2].diff_time = coinc_data[j].diff_time;
				block1 = coinc_data[j].crystal_index_1 / 200;
				block2 = coinc_data[j].crystal_index_2 / 200;
				if (temp[num_coinc2].crystal_index_1<NUM_CRYSTAL_SCANNER && temp[num_coinc2].crystal_index_2<NUM_CRYSTAL_SCANNER)
					temp[num_coinc2].scatter_coeff = float(scatter_data[j])*SCATTER_SCALE_SS;
				else if (temp[num_coinc2].crystal_index_1 >= NUM_CRYSTAL_SCANNER && temp[num_coinc2].crystal_index_2 >= NUM_CRYSTAL_SCANNER)
					temp[num_coinc2].scatter_coeff = float(scatter_data[j])*SCATTER_SCALE_OO;
				else
					temp[num_coinc2].scatter_coeff = float(scatter_data[j])*SCATTER_SCALE_OS;
				//add random
				temp[num_coinc2].scatter_coeff += randomMatrix[block1][block2] / 40000.0f*crystal_efficency[coinc_data[num_coinc2].crystal_index_1] * crystal_efficency[coinc_data[num_coinc2].crystal_index_2] * TOF_BIN_SIZE / TOF_WINDOW;
				if (temp[num_coinc2].scatter_coeff > maxs){
					maxs = temp[num_coinc2].scatter_coeff;
				}
				if (temp[num_coinc2].scatter_coeff < mins){
					mins = temp[num_coinc2].scatter_coeff;
				}
				sum_scatter += temp[num_coinc2].scatter_coeff;
				num_coinc2++;
			}
			cout << "\n Time for last written event : " << time_ptr << "\n";
			printf("......%d coincidence events were returned in %d iteration.\n", num_coinc2, i);
			fp = fopen("File_with_scatter.lst", "ab+");
			fwrite(temp, sizeof(tof_coinc_event_cbm_scatter), num_coinc2, fp);
			fclose(fp);
		}
		mean_scatter = sum_scatter / num_coincidence;
		cout << "\n Mean scatter : " << mean_scatter;
		cout << "\n Max scatter : " << maxs;
		cout << "\n Min scatter : " << mins;
		cout << "\n Total events written: " << num_coincidence;
		// Check how output file looks like using last few events
		ofstream myfile2;
		char str2[64];
		strcpy(str2, coin_filename);
		strcat(str2, ".txt");
		myfile2.open(str2);
		for (int j = 0; j < 100; j++){
			myfile2 << temp[j].crystal_index_1 << "  " << temp[j].crystal_index_2 << "  " << temp[j].time_1 << "  " << temp[j].diff_time << " " << temp[j].bed_position<<" "<< temp[j].scatter_coeff << endl;
		}
		myfile2.close();
		free(coinc_data);
		free(scatter_data);
		free(temp);
		return 0;
	}



}


void record_scatter_coinc(vector<tof_coinc_event_cbm_scatter>& mini_scatter_coinc_event, int& num_coinc2, vector<vector<float>>& randomMatrix, float* crystal_efficency, tof_coinc_event_cbm* coinc_data)
{
	int block1, block2;
	tof_coinc_event_cbm_scatter new_event;
	new_event.crystal_index_1 = coinc_data[num_coinc2].crystal_index_1;
	new_event.crystal_index_2 = coinc_data[num_coinc2].crystal_index_2;
	new_event.time_1 = coinc_data[num_coinc2].time_1;
	new_event.diff_time = coinc_data[num_coinc2].diff_time;
	new_event.bed_position = coinc_data[num_coinc2].bed_position;
	block1 = coinc_data[num_coinc2].crystal_index_1 / 200;
	block2 = coinc_data[num_coinc2].crystal_index_2 / 200;
	new_event.scatter_coeff = randomMatrix[block1][block2] / 40000.0f*crystal_efficency[coinc_data[num_coinc2].crystal_index_1] * crystal_efficency[coinc_data[num_coinc2].crystal_index_2] * TOF_BIN_SIZE / TOF_WINDOW;
	mini_scatter_coinc_event.push_back(new_event);
}
