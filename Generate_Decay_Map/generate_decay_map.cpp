/*
 * generate_decay_map.cc
 *
 *  Created on: Jan 13, 2015
 *      Author: keli
 */
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
	int *decay_map;
	mini_coinc_event *mini_coinc_data;
	pos *pos_list;
	size_t t;
	long long int size;

	int i;
	
	char *file_names_ptr;
	
	int data_length;
	int true_coincidence;
	int ss, is, ii;
	
	struct _stat64 st;
	
	FILE *current_file;
	FILE *output;
	
	char decay_map_filename[128] = "decay_map.img";
	int num_coincidence;
	int entry_length;

//	printf("mini_coinc_event size: %d\n",sizeof(mini_coinc_event));
//	printf("size of long is %d\n",(int)sizeof(offset));
//	printf("Max Int is %d\n",INT_MAX);
//	printf("Max Long is %ld\n",LONG_MAX);
//	printf("Max Long Long is %ld\n",LLONG_MAX);


//*****************************program argument*******************************************//
	if(argc<3){
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


	image_size_x = 600;
	image_size_y = 600;
	image_size_z = 224;
	image_samp_x = 1.0;
	image_samp_y = 1.0;
	image_samp_z = 1.0;
	image_size = image_size_x*image_size_y*image_size_z;

//*****************************check file length***********************************************//
	entry_length = sizeof(mini_coinc_event);

	printf("Following coincidence files are being processed:\n");

	file_names_ptr=argv[1];
	_stat64(file_names_ptr, &st);
	size = st.st_size;
	size /=entry_length;
	if(size >= INT_MAX){
		printf("File \"%s\" length is tooooooooooo long. This program only supports singles event fewer than %d. Current file has %I64d. Program stopped\n",file_names_ptr, INT_MAX, size);
		return 0;
	}
	data_length=size;
	printf("%d coincidence event in file \"%s\"\n",data_length, file_names_ptr);

	num_coincidence = data_length;
	
//***********************************Read data and generate decay map****************************************//
	mini_coinc_data = (mini_coinc_event*)malloc(num_coincidence*sizeof(mini_coinc_event));
	current_file = fopen(file_names_ptr,"rb");
	t = fread(mini_coinc_data,sizeof(mini_coinc_event),num_coincidence,current_file);
	if (num_coincidence==(int)t){
		printf("......%s read successfully.\n",file_names_ptr);
	}else{
		printf("......%s read error. %d elements were read\n",file_names_ptr,(int)t);
	}
	fclose(current_file);
	
	pos_list = (pos*)malloc(num_coincidence*sizeof(pos));
	true_coincidence = 0;

	ss=0;
	ii=0;
	is=0;

	for(i=0;i<data_length;i++){
		if(mini_coinc_data[i].event_id >= 0){
			pos_list[true_coincidence] = mini_coinc_data[i].source_pos;
			true_coincidence++;

			if (mini_coinc_data[i].crystal_index_1 < 32448 && mini_coinc_data[i].crystal_index_2 < 32448){
				ss++;
			}
			else if (mini_coinc_data[i].crystal_index_1 >= 32448 && mini_coinc_data[i].crystal_index_2 >= 32448){
				ii++;
			}else{
				is++;
			}

		}
	}
	free(mini_coinc_data);
	decay_map = (int*)calloc(image_size,sizeof(int));
	printf("%d true coincidence found....Starting to generate decay map\n",true_coincidence);
	printf("ss = %d, ii = %d, is = %d\n",ss,ii,is);
	
	//***************put pos_list into decay_map*******************//
	for(i=0;i<true_coincidence;i++){
		
		 x = (int)(pos_list[i].x / image_samp_x + 1000.0)-1000 +  image_size_x/2;
		 y = (int)(pos_list[i].y / image_samp_y + 1000.0)-1000 +  image_size_y/2;
		 z = (int)(pos_list[i].z / image_samp_z + 1000.0)-1000 +  image_size_z/2;
		 
		 if(x>=0 && x<image_size_x && y>=0 && y<image_size_y && z>=0 && z<image_size_z){
			 voxel_index = x + y*image_size_x + z*image_size_x*image_size_y;
			 decay_map[voxel_index]++;
		 }
		 
	}
	printf("....Decay map generated\n");
//*****************************write decay map*********************************//
	output = fopen(decay_map_filename,"wb");
	t=fwrite(decay_map, sizeof(int), image_size, output);
	if (image_size==(int)t){
		printf("Done writing decay map\n");
	}else{
		printf("%s written with error. %d elements were written.\n",decay_map_filename,(int)t);
	}
	fclose(output);

	free(pos_list);
	free(decay_map);
	return 0;

}

