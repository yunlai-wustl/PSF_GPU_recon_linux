/*
 * generate_decay_map.cc
 *
 *  Created on: Jan 13, 2015
 *      Author: keli
 */
#define _CRT_SECURE_NO_WARNINGS
#include "generate_decay_map.h"

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
	int sorting_method;
	int image_size_x,image_size_y,image_size_z;
	float image_samp_x, image_samp_y, image_samp_z;
	int image_size, voxel_index;
	int x,y,z;
	int *decay_map_SS, *decay_map_OS, *decay_map_OO, *decay_map_all;
	decay_event *decay_data;
	vector<pos32> pos_list_SS, pos_list_OO, pos_list_OS;
	size_t t;
	long long int size = 0;

	int i;
	
	char *file_names_ptr;
	
	int data_length;
	int true_coincidence;
	int ss, os, oo;
	
	struct stat64 st;
	
	FILE *current_file;
	FILE *output;
	
	char decay_map_SS_filename[128] = "decay_map_SS.img";
	char decay_map_OS_filename[128] = "decay_map_OS.img";
	char decay_map_OO_filename[128] = "decay_map_OO.img";
	char decay_map_All_filename[128] = "decay_map_all.img";

	int num_coincidence;
	int entry_length;

//	printf("decay_event size: %d\n",sizeof(decay_event));
//	printf("size of long os %d\n",(int)sizeof(offset));
//	printf("Max Int os %d\n",INT_MAX);
//	printf("Max Long os %ld\n",LONG_MAX);
//	printf("Max Long Long os %ld\n",LLONG_MAX);


//*****************************program argument*******************************************//
	if(argc<3){
		printf("Not enough arguments, please use -help option.\n");
		return 0;
	}

	if (strcmp(argv[1], "-help") == 0){
		printf("Here describes how to run thos program.\n");
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


	image_size_x = 600;
	image_size_y = 600;
	image_size_z = 400;
	image_samp_x = 1.0;
	image_samp_y = 1.0;
	image_samp_z = 1.0;
	image_size = image_size_x*image_size_y*image_size_z;

//*****************************check file length***********************************************//
	entry_length = sizeof(decay_event);

	printf("Following coincidence files are being processed:\n");

	file_names_ptr=argv[2];
	stat64(file_names_ptr, &st);
	size = st.st_size;
	cout << "\n " << size;
	size /=entry_length;
	cout << "\n " << size;
		if (size >= LLONG_MAX){
		printf("File \"%s\" length os tooooooooooo long. Thos program only supports singles event fewer than %d. Current file has %I64d. Program stopped\n", file_names_ptr, LLONG_MAX, size);
		return 0;
	}
	data_length=size;
	printf("%d coincidence event in file \"%s\"\n",data_length, file_names_ptr);

	num_coincidence = data_length;
	
//***********************************Read data and generate decay map****************************************//
	decay_data = (decay_event*)malloc(num_coincidence*sizeof(decay_event));
	current_file = fopen(file_names_ptr,"rb");
	t = fread(decay_data,sizeof(decay_event),num_coincidence,current_file);
	if (num_coincidence==(int)t){
		printf("......%s read successfully.\n",file_names_ptr);
	}else{
		printf("......%s read error. %d elements were read\n",file_names_ptr,(int)t);
	}
	fclose(current_file);
	
	
	true_coincidence = 0;
	ss=0;
	oo=0;
	os=0;
	pos_list_SS.reserve(data_length);
	pos_list_OS.reserve(data_length);
	pos_list_OO.reserve(data_length);
	for(i=0;i<data_length;i++){
		true_coincidence++;
		if (decay_data[i].crystal_index_1 <60800 && decay_data[i].crystal_index_2 < 60800){
			ss++;
			pos_list_SS.push_back(decay_data[i].source_position);
		}
		else if (decay_data[i].crystal_index_1 >= 60800 && decay_data[i].crystal_index_2 >= 60800){
			oo++;
			pos_list_OO.push_back(decay_data[i].source_position);
		}
		else{
			os++;
			pos_list_OS.push_back(decay_data[i].source_position);
		}
	}
	free(decay_data);

	decay_map_SS = (int*)calloc(image_size,sizeof(int));
	decay_map_OS = (int*)calloc(image_size, sizeof(int));
	decay_map_OO = (int*)calloc(image_size, sizeof(int));
	decay_map_all = (int*)calloc(image_size, sizeof(int));

	printf("%d true coincidence found....Starting to generate decay map\n",true_coincidence);
	printf("ss = %d, oo = %d, os = %d\n",ss,oo,os);
	
	//***************put pos_list into decay_map*******************//
	for(i=0;i<true_coincidence;i++){
		x = (int)(pos_list_SS[i].x / image_samp_x + 1000.0) - 1000 + image_size_x / 2;
		y = (int)(pos_list_SS[i].y / image_samp_y + 1000.0) - 1000 + image_size_y / 2;
		z = (int)(pos_list_SS[i].z / image_samp_z + 1000.0) - 1000 + image_size_z / 2 - 68.5;
		if (i<10)
			cout << "x=" << x << ", y=" << y << ", z=" << z << endl;
		if(x>=0 && x<image_size_x && y>=0 && y<image_size_y && z>=0 && z<image_size_z){
			 voxel_index = x + y*image_size_x + z*image_size_x*image_size_y;
			 decay_map_SS[voxel_index]++;
			 decay_map_all[voxel_index]++;
		 }
	}
	printf("....SS Decay map generated\n");
	/**********************Os decay map *****************/
	for (i = 0; i<true_coincidence; i++){
		x = (int)(pos_list_OS[i].x / image_samp_x + 1000.0) - 1000 + image_size_x / 2;
		y = (int)(pos_list_OS[i].y / image_samp_y + 1000.0) - 1000 + image_size_y / 2;
		z = (int)(pos_list_OS[i].z / image_samp_z + 1000.0) - 1000 + image_size_z / 2 - 68.5;
		if (i<10)
			cout << "x=" << x << ", y=" << y << ", z=" << z << endl;
		if (x >= 0 && x<image_size_x && y >= 0 && y<image_size_y && z >= 0 && z<image_size_z){
			voxel_index = x + y*image_size_x + z*image_size_x*image_size_y;
			decay_map_OS[voxel_index]++;
			decay_map_all[voxel_index]++;
		}
	}
	printf("....OS Decay map generated\n");
/**********************OO decay map *****************/
	for (i = 0; i<true_coincidence; i++){
		x = (int)(pos_list_OO[i].x / image_samp_x + 1000.0) - 1000 + image_size_x / 2;
		y = (int)(pos_list_OO[i].y / image_samp_y + 1000.0) - 1000 + image_size_y / 2;
		z = (int)(pos_list_OO[i].z / image_samp_z + 1000.0) - 1000 + image_size_z / 2 - 68.5;
		if (i<10)
			cout << "x=" << x << ", y=" << y << ", z=" << z << endl;
		if (x >= 0 && x<image_size_x && y >= 0 && y<image_size_y && z >= 0 && z<image_size_z){
			voxel_index = x + y*image_size_x + z*image_size_x*image_size_y;
			decay_map_OO[voxel_index]++;
			decay_map_all[voxel_index]++;
		}
	}
	printf("....OO Decay map generated\n");


	vector<pos32>().swap(pos_list_SS); //free memory
	vector<pos32>().swap(pos_list_OS); //free memory
	vector<pos32>().swap(pos_list_OO); //free memory
//*****************************write decay map*********************************//
	output = fopen(decay_map_SS_filename,"wb");
	t=fwrite(decay_map_SS, sizeof(int), image_size, output);
	if (image_size==(int)t){
		printf("Done writing SS decay map\n");
	}else{
		printf("%s written with error. %d elements were written.\n",decay_map_SS_filename,(int)t);
	}
	fclose(output);
	free(decay_map_SS);

	output = fopen(decay_map_OS_filename, "wb");
	t = fwrite(decay_map_OS, sizeof(int), image_size, output);
	if (image_size == (int)t){
		printf("Done writing OS decay map\n");
	}
	else{
		printf("%s written with error. %d elements were written.\n", decay_map_OS_filename, (int)t);
	}
	fclose(output);
	free(decay_map_OS);

	output = fopen(decay_map_OO_filename, "wb");
	t = fwrite(decay_map_OO, sizeof(int), image_size, output);
	if (image_size == (int)t){
		printf("Done writing OO decay map\n");
	}
	else{
		printf("%s written with error. %d elements were written.\n", decay_map_OO_filename, (int)t);
	}
	fclose(output);
	free(decay_map_OO);

	output = fopen(decay_map_All_filename, "wb");
	t = fwrite(decay_map_all, sizeof(int), image_size, output);
	if (image_size == (int)t){
		printf("Done writing OO decay map\n");
	}
	else{
		printf("%s written with error. %d elements were written.\n", decay_map_All_filename, (int)t);
	}
	fclose(output);
	free(decay_map_all);
	return 0;
}

void record_decay_map(vector<pos32>& l, int* map, int image_samp_x, int image_samp_y, int image_samp_z, int image_size_x, int image_size_y, int image_size_z, int center){
	int x, y, z, voxel_index;
	for (int i = 0; i<l.size(); i++){
		x = (int)(l[i].x / image_samp_x + 1000.0) - 1000 + image_size_x / 2;
		y = (int)(l[i].y / image_samp_y + 1000.0) - 1000 + image_size_y / 2;
		z = (int)(l[i].z / image_samp_z + 1000.0) - 1000 + image_size_z / 2 - center;
		if (x >= 0 && x<image_size_x && y >= 0 && y<image_size_y && z >= 0 && z<image_size_z){
			voxel_index = x + y*image_size_x + z*image_size_x*image_size_y;
			map[voxel_index]++;
		}
	}
}