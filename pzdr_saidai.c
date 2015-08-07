/* author gumboshi <gumboshi@gmail.com> */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "patterns.h"
#include "pzdr_def.h"
#include "pzdr_saidai.h"

double gettimeofday_sec()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + (double)tv.tv_usec*1e-6;
}


int main(int argc, char* argv[]){

  int width = 6;
  int hight = 5;
  int start = 0;
  int end = width*hight;
  int specified_end = end;
  int reverse_length;
  int half_size;
  int LS = 0;
  int isStrong = 0;
  int simuave = 0;
  int i;
  int useCUDA = 0;
#ifdef CUDA
  useCUDA = 1;
#endif
  int line = 0;
  int way = 0;
  int table_size = NORMAL_TABLE;
  if (argc != 1) {
    for (i = 1; i < argc; i++) {
      if (strcmp(argv[i], "-s") == 0) {
        i++;
        start = atoi(argv[i]);
      }
      else if (strcmp(argv[i], "-e") == 0) {
        i++;
        specified_end = atoi(argv[i]);
      }
      else if (strcmp(argv[i], "-l") == 0) {
        i++;
        line = atoi(argv[i]);
      }
      else if (strcmp(argv[i], "-w") == 0) {
        i++;
        way = atoi(argv[i]);
      }
      else if (strcmp(argv[i], "-small") == 0) {
	table_size = SMALL_TABLE;
      }
      else if (strcmp(argv[i], "-normal") == 0) {
	table_size = NORMAL_TABLE;
      }
      else if (strcmp(argv[i], "-big") == 0) {
	table_size = BIG_TABLE;
      }
      else if (strcmp(argv[i], "-strong") == 0) {
	isStrong = 1;
      }
      else if (strcmp(argv[i], "-laku") == 0 || strcmp(argv[i], "-paru") == 0) {
	LS = LAKU_PARU;
      }
      else if (strcmp(argv[i], "-krishna") == 0) {
	LS = KRISHNA;
      }
      else if (strcmp(argv[i], "-hero") == 0) {
	LS = HERO;
      }
      else if (strcmp(argv[i], "-sonia") == 0) {
	LS = SONIA;
      }
      else if (strcmp(argv[i], "-noLS") == 0) {
	LS = NOLS;
      }
      else if (strcmp(argv[i], "-ave") == 0) {
	simuave = 1;
	srand((unsigned) time(NULL));
      }
#ifdef CUDA
      else if (strcmp(argv[i], "-gpu") == 0) {
	useCUDA = 1;
      }
      else if (strcmp(argv[i], "-cpu") == 0) {
	useCUDA = 0;
      }
#endif
      else if (strcmp(argv[i], "-ID2TN") == 0) {
	i++;
	unsigned long long ID = atoll(argv[i]);
	ID2table(ID, 6, 5);
        fprintf(stdout,"please use ID2table.jar (GUI interface)\n");
	exit(1);
      }
      else if (strcmp(argv[i], "-ID2TB") == 0) {
	i++;
	unsigned long long ID = atoll(argv[i]);
	ID2table(ID, 7, 6);
        fprintf(stdout,"please see ID2table.jar (GUI interface)\n");
	exit(1);
      }
      else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
        fprintf(stdout,"options\n");
        fprintf(stdout,"-s <arg>    : Start number of main color. (default=0)\n");
        fprintf(stdout,"-e <arg>    : End number of main color. (default=TABLE_SIZE)\n");
        fprintf(stdout,"-small      : Simulate with 4x5 table\n");
        fprintf(stdout,"-normal     : Simulate with 5x6 table\n");
        fprintf(stdout,"-big        : Simulate with 6x7 table\n");
        fprintf(stdout,"-l <arg>    : Number of \"LINE\"s \n");
        fprintf(stdout,"-w <arg>    : Number of \"2WAY\"s \n");
        fprintf(stdout,"-strong     : Use Enhanced Orbs. (drop kyouka)\n");
        fprintf(stdout,"-laku, -paru: Simulate with Lakshmi or Parvati leader skill mode. \n");
        fprintf(stdout,"-krishna    : Simulate with krishna leader skill mode. \n");
        fprintf(stdout,"-hero       : Simulate with Helo leader skill mode. \n");
        fprintf(stdout,"-sonia      : Simulate with Red Sonia mode. \n");
        fprintf(stdout,"-noLS       : Simulate with fixed leader skill. \n");
        fprintf(stdout,"-ave        : execute 10000 times OCHIKON simulation. \n");
        fprintf(stdout,"-gpu        : execute by gpu. (To enable gpu mode, type command `make gpu`) \n");
        fprintf(stdout,"-cpu        : execute by cpu. \n");
        fprintf(stdout,"-ID2TN <arg>: print normal size table equivalent of input ID (unsigned long long value) \n");
        fprintf(stdout,"-ID2TB <arg>: print big size table equivalent of input ID (unsigned long long value) \n");
        fprintf(stdout,"\n *********** example ************ \n");
        fprintf(stdout,"(1): to simulate 13-17 ~ 18-12, \n");
        fprintf(stdout,"$(exefile) -normal -s 13 -e 18\n\n");
        fprintf(stdout,"(2): to simulate Parvati LS with 2way*2, \n");
        fprintf(stdout,"$(exefile) -normal -paru -w 2\n\n");
        fprintf(stdout,"(3): to simulate with large table, \n");
        fprintf(stdout,"$(exefile) -big \n\n");
        fprintf(stdout,"(4): OCHIKON simulation, \n");
        fprintf(stdout,"$(exefile) -ave \n\n");
        fprintf(stdout,"(5): to check the output ID, \n");
        fprintf(stdout,"$(exefile) -ID2TN <ID> \n\n");
        exit(1);
      }
      else {
        fprintf(stderr,"unknown option. See --help.\n");
        exit(1);
      }
    }
  }
  
  unsigned long long *num_patterns;
  int *num_patterns_half;
/*   void (*gt)(long long, long long*); */
/*   int (*os)(long long*, int*, int*, int*, int, int); */
  int combo_length;
  switch(table_size){
  case SMALL_TABLE: 
    width = 5;
    hight = 4;
    num_patterns = num_patterns20;
    num_patterns_half = num_patterns10;
/*     gt = generate_table_small; */
/*     os = one_step_small; */
    combo_length = 7;
    break;
  case NORMAL_TABLE: 
    width = 6;
    hight = 5;
    num_patterns = num_patterns30;
    num_patterns_half = num_patterns15;
/*     gt = generate_table_normal; */
/*     os = one_step_normal; */
    combo_length = 10;
    break;
  case BIG_TABLE: 
    width = 7;
    hight = 6;
    num_patterns = num_patterns42;
    num_patterns_half = num_patterns21;
/*     gt = generate_table_big; */
/*     os = one_step_big; */
    combo_length = 14;
    break;
  default:
    fprintf(stderr,"unknown\n");
    exit(1);
  }
    
  if(end == specified_end){
    end = width * hight;
  }else{
    end = specified_end;
  }
  

  half_size = width*hight/2;
  reverse_length = 1 << width;
  int bit_count_table[256];
  int reversed_bit_table[reverse_length];
  init_reversed_bit_table(reversed_bit_table, width);

  int *tableID_half_table;
  int *tableID_half_prefix;
  tableID_half_table  = (int*)malloc(sizeof(int)*(1L << half_size));
  tableID_half_prefix = (int*)malloc(sizeof(int)*half_size);
  
  init_bit_count_table(bit_count_table);
  create_half_tableID(tableID_half_table, tableID_half_prefix, bit_count_table, num_patterns_half, half_size);

  if(useCUDA){
#ifdef CUDA
    //simulate_all_cuda(gt, os, table_size, start, end, bit_count_table, reversed_bit_table, tableID_half_table, tableID_half_prefix, num_patterns, num_patterns_half, width, hight, combo_length, LS, isStrong, line, way, simuave);
    simulate_all_cuda(table_size, start, end, /*bit_count_table, */reversed_bit_table, tableID_half_table, tableID_half_prefix, /*num_patterns,*/ num_patterns_half, width, hight, combo_length, LS, isStrong, line, way, simuave);
#endif
  }else{
    //simulate_all(gt, os, start, end, bit_count_table, reversed_bit_table, tableID_half_table, tableID_half_prefix, num_patterns, num_patterns_half, width, hight, combo_length, LS, isStrong, line, way, simuave);
    simulate_all(table_size, start, end, /*bit_count_table, */reversed_bit_table, tableID_half_table, tableID_half_prefix, num_patterns, num_patterns_half, width, hight, combo_length, LS, isStrong, line, way, simuave);
  }

  free(tableID_half_table);
  free(tableID_half_prefix);

  return 0;

}

/* void init_combo_info(int *color_combo, int *num_drops_combo, int *isLine_combo, int combo_length){ */
/*   int i; */
/*   for(i = 0;i < combo_length;i++){ */
/*     color_combo[i] = 0; */
/*     num_drops_combo[i] = 0; */
/*     isLine_combo[i] = 0; */
/*   } */
/* } */

void init_bit_count_table(int *bit_count_table){
  int i, j;
  for(i = 0;i < 256;i++){
    int count = 0;
    int compared_num = i;
    for(j = 0;j < 8;j++){
      if((compared_num >> j & 1) == 1){
	count++;
      }
    }
    bit_count_table[i] = count;
  }
}

void init_reversed_bit_table(int *reversed_bit_table, const int width){
  int i, j;

  for(i = 0;i < (1 << width);i++){
    int ii[width];
    for(j = 0;j < width;j++){
      ii[j] = ((i >> j) & 1) << (width-1 - j);
    }
    int sum = 0;
    for(j = 0;j < width;j++){
      sum += ii[j];
    }
    reversed_bit_table[i] = sum;
  }
}


void create_half_tableID(int *tableID_half_table, int *tableID_half_prefix, int * const bit_count_table, int * const num_patterns, const int half_table_size){

  int num_attacks;
  int max_tableID = 1 << half_table_size;
  int num_chunks = (half_table_size-1)/8+1;
  int sum = 0;
  for(num_attacks = 0;num_attacks <= half_table_size;num_attacks++){
    tableID_half_prefix[num_attacks] = sum;
    sum = sum + num_patterns[num_attacks];
  }
  for(num_attacks = 0;num_attacks <= half_table_size;num_attacks++){
    int chunk[num_chunks];
    int tableID = 0;
    int index = 0;
    while(tableID <= max_tableID){
      int count = 0;
      int i;
      for(i = 0;i < num_chunks;i++){
	chunk[i] = (tableID >> 8*i ) & 255;
	count += bit_count_table[chunk[i]];
      }    
      if(count == num_attacks){
	tableID_half_table[index+tableID_half_prefix[num_attacks]] = tableID;
	index++;
      }
      tableID++;
    }
  }
}
 
#define WID 7
void generate_table_small(unsigned long long tableID, unsigned long long *color_table){

  unsigned long long ID = tableID;
  unsigned long long b0 = ID & 31;
  unsigned long long b1 = (ID >> 5 ) & 31;
  unsigned long long b2 = (ID >> 10) & 31;
  unsigned long long b3 = (ID >> 15) & 31;
  color_table[0] = (b0 << (WID+1)) | (b1 << (WID*2+1))
    | (b2 << (WID*3+1)) | (b3 << (WID*4+1));
  ID = ~ID;
  b0 = ID & 31;
  b1 = (ID >> 5 ) & 31;
  b2 = (ID >> 10) & 31;
  b3 = (ID >> 15) & 31;
  color_table[1] = (b0 << (WID+1)) | (b1 << (WID*2+1))
    | (b2 << (WID*3+1)) | (b3 << (WID*4+1));

}
#undef WID

#define WID 8
void generate_table_normal(unsigned long long tableID, unsigned long long *color_table){

  unsigned long long ID = tableID;
  unsigned long long b0 = ID & 63;
  unsigned long long b1 = (ID >> 6 ) & 63;
  unsigned long long b2 = (ID >> 12) & 63;
  unsigned long long b3 = (ID >> 18) & 63;
  unsigned long long b4 = (ID >> 24) & 63;
  color_table[0] = (b0 << (WID+1)) | (b1 << (WID*2+1))
    | (b2 << (WID*3+1)) | (b3 << (WID*4+1)) | (b4 << (WID*5+1));
  ID = ~ID;
  b0 = ID & 63;
  b1 = (ID >> 6 ) & 63;
  b2 = (ID >> 12) & 63;
  b3 = (ID >> 18) & 63;
  b4 = (ID >> 24) & 63;
  color_table[1] = (b0 << (WID+1)) | (b1 << (WID*2+1))
    | (b2 << (WID*3+1)) | (b3 << (WID*4+1)) | (b4 << (WID*5+1));

}
#undef WID

#define WID 9
void generate_table_big(unsigned long long tableID, unsigned long long *color_table){
   
  unsigned long long b0, b1, b2, b3, b4, b5;
  unsigned long long ID = tableID;
  b0 = ID & 127;
  b1 = (ID >> 7 ) & 127;
  b2 = (ID >> 14) & 127;
  b3 = (ID >> 21) & 127;
  b4 = (ID >> 28) & 127;
  b5 = (ID >> 35) & 127;
  color_table[0] = (b0 << (WID+1)) | (b1 << (WID*2+1))
    | (b2 << (WID*3+1)) | (b3 << (WID*4+1)) 
    | (b4 << (WID*5+1)) | (b5 << (WID*6+1));
  ID = ~ID;
  b0 = ID & 127;
  b1 = (ID >> 7 ) & 127;
  b2 = (ID >> 14) & 127;
  b3 = (ID >> 21) & 127;
  b4 = (ID >> 28) & 127;
  b5 = (ID >> 35) & 127;
  color_table[1] = (b0 << (WID+1)) | (b1 << (WID*2+1))
    | (b2 << (WID*3+1)) | (b3 << (WID*4+1)) 
    | (b4 << (WID*5+1)) | (b5 << (WID*6+1));

}
#undef WID


void simulate_all(const int table_size, const int start, const int end, /*int * const bit_count_table,*/ int * const reversed_bit_table, int * const tableID_half_table, int * const tableID_half_prefix, unsigned long long * const num_patterns, int * const num_patterns_half, const int width, const int hight, const int combo_length, const int LS, const int isStrong, const int line, const int way, const int simuave){

  const float pline = (float)line;
  const float pway = pow(1.5,way);
  const unsigned long long filter = (1L << (width*hight))-1;

  int num_threads = 1;
#ifdef _OPENMP
  num_threads = omp_get_max_threads();
#endif

  unsigned long long tableID = 0;
  //int rank = MIN(RANKINGLENGTH, num_patterns[num_attacks]);
  const int rank = RANKINGLENGTH;
  unsigned long long max_powerID[num_threads][2][rank];
  float max_power[num_threads][2][rank];
  unsigned long long final_MID[42][rank];
  float final_MP[42][rank];

  int i, j, k, m;
  int color_combo[combo_length];
  int num_drops_combo[combo_length];
  int isLine_combo[combo_length];
  const int half_table_size = width*hight/2;
  const int reverse_length = 1 << width;
  int num_attacks;

  for(num_attacks = start;num_attacks <= end;num_attacks++){
    printf("calculating %2d-%2d & %2d-%2d ...\n", num_attacks, width*hight-num_attacks, width*hight-num_attacks, num_attacks);
    if(half_table_size < num_attacks && num_attacks <= width*hight-start) break;

    if(num_attacks == half_table_size){
      for(i = 0;i < num_threads;i++){
	for(j = 0;j < rank;j++){
	  max_powerID[i][0][j] = 0;
	  max_power[i][0][j] = 0;
	  max_powerID[i][1][j] = 0;
	  max_power[i][1][j] = 0;
	}
      }

#pragma omp parallel private(i,j,k,tableID, color_combo, num_drops_combo, isLine_combo)
      {
	int thread_num = 0;
#ifdef _OPENMP
	thread_num = omp_get_thread_num();
#endif
	int u, l, uu, ll;
	for(u = 0;u <= num_attacks;u++){
	  l = num_attacks - u;
	  int uoffset = tableID_half_prefix[u];
	  int loffset = tableID_half_prefix[l];
#pragma omp for 
	  for(uu = 0;uu < num_patterns_half[u];uu++){
	    for(ll = 0;ll < num_patterns_half[l];ll++){
	      unsigned long long upperID = (long long)tableID_half_table[uu+uoffset];
	      unsigned long long lowerID = (long long)tableID_half_table[ll+loffset];
	      tableID = (upperID << half_table_size) | lowerID;
	      unsigned long long reversed = 0;
	      int reverse_bit[width];
	      for(i = 0;i < hight; i++){
		reverse_bit[i] = (tableID >> width*i ) & (reverse_length-1);
		reversed = reversed | (((long long)reversed_bit_table[reverse_bit[i]]) << width*i);
	      }
	      unsigned long long inversed = (~tableID) & filter;
	      if(tableID <= reversed && tableID <= inversed){
		//init_combo_info(color_combo, num_drops_combo, isLine_combo, combo_length);
		int combo_counter = 0;
		unsigned long long color_table[NUM_COLORS];
		int num_c;
		for(num_c = 0;num_c < NUM_COLORS;num_c++){
		  color_table[num_c] = 0;
		}
		switch(table_size){
		case SMALL_TABLE:
		  generate_table_small(tableID, color_table);
		  break;
		case NORMAL_TABLE:
		  generate_table_normal(tableID, color_table);
		  break;
		case BIG_TABLE:
		  generate_table_big(tableID, color_table);
		  break;
		default:
		  fprintf(stderr, "unknown table size\n");
		  exit(1);
		}
		
		int returned_combo_counter = 0;
		do{
		  combo_counter = returned_combo_counter;
		  
		  switch(table_size){
		  case SMALL_TABLE:
		    returned_combo_counter = one_step_small(color_table, color_combo, num_drops_combo, isLine_combo, combo_counter, NUM_COLORS);
		    break;
		  case NORMAL_TABLE:
		    returned_combo_counter = one_step_normal(color_table, color_combo, num_drops_combo, isLine_combo, combo_counter, NUM_COLORS);
		    break;
		  case BIG_TABLE:
		    returned_combo_counter = one_step_big(color_table, color_combo, num_drops_combo, isLine_combo, combo_counter, NUM_COLORS);
		    break;
		  }
		}while(returned_combo_counter != combo_counter);
		//float power = return_attack(combo_counter, color_combo, num_drops_combo, isLine_combo, LS, isStrong, pline, pway);
		float power[2];
		return_attack_double(power, combo_counter, color_combo, num_drops_combo, isLine_combo, LS, isStrong, pline, pway);
		
		if(max_power[thread_num][0][rank-1] < power[0]){
		  for(j = 0;j < rank;j++){
		    if(max_power[thread_num][0][j] < power[0]){
		      for(k = rank-2;k >= j;k--){
			max_powerID[thread_num][0][k+1] = max_powerID[thread_num][0][k];
			max_power[thread_num][0][k+1] = max_power[thread_num][0][k];
		      }
		      max_powerID[thread_num][0][j] = tableID;
		      max_power[thread_num][0][j] = power[0];
		      break;
		    }
		  }
		}
		if(max_power[thread_num][1][rank-1] < power[1]){
		  for(j = 0;j < rank;j++){
		    if(max_power[thread_num][1][j] < power[1]){
		      for(k = rank-2;k >= j;k--){
			max_powerID[thread_num][1][k+1] = max_powerID[thread_num][1][k];
			max_power[thread_num][1][k+1] = max_power[thread_num][1][k];
		      }
		      max_powerID[thread_num][1][j] = (~tableID) & filter;
		      max_power[thread_num][1][j] = power[1];
		      break;
		    }
		  }
		}

	      }
	    }
	  }
	}

      } //omp end

      float MP[rank];
      unsigned long long MID[rank];
      int ms;
      for(i = 0;i < rank;i++){
	MP[i] = 0.0;
	MID[i]= 0;
      }
      
      for(i = 0;i < num_threads;i++){
	for(ms = 0; ms < 2; ms++){
	  for(j = 0;j < rank;j++){
	    float power = max_power[i][ms][j];
	    tableID = max_powerID[i][ms][j];
	    if(MP[rank-1] < power){
	      for(k = 0;k < rank;k++){
		if(MP[k] < power){
		  for(m = rank-2;m >= k;m--){
		    MID[m+1] = MID[m];
		    MP[m+1] = MP[m];
		  }
		  MID[k] = tableID;
		  MP[k] = power;
		  break;
		}
	      }
	    }
	  }
	}
      }
      for(i = 0;i < rank;i++){
	float power = MP[i];
	unsigned long long tmp = MID[i];
	unsigned long long minID = tmp;
	int index = i;
	for(j = i+1;j < rank;j++){
	  if(power == MP[j]){
	    if(minID > MID[j]){
	      minID = MID[j];
	      index = j;
	    }
	  }else{
	    break;
	  }
	}
	MID[index] = tmp;
	MID[i] = minID;
      }
      for(i = 0;i < rank;i++){
	final_MID[num_attacks][i] = MID[i];
	final_MP [num_attacks][i] = MP [i];
      }

    }else{
      for(i = 0;i < num_threads;i++){
	for(j = 0;j < rank;j++){
	  max_powerID[i][0][j] = 0;
	  max_power[i][0][j] = 0;
	  max_powerID[i][1][j] = 0;
	  max_power[i][1][j] = 0;
	}
      }

#pragma omp parallel private(i,j,k,tableID, color_combo, num_drops_combo, isLine_combo)
      {
	int thread_num = 0;
#ifdef _OPENMP
	thread_num = omp_get_thread_num();
#endif
	int u, l, uu, ll;
	for(u = 0;u <= num_attacks;u++){
	  l = num_attacks - u;
	  if(u <= half_table_size && l <= half_table_size){
	    int uoffset = tableID_half_prefix[u];
	    int loffset = tableID_half_prefix[l];
#pragma omp for 
	    for(uu = 0;uu < num_patterns_half[u];uu++){
	      for(ll = 0;ll < num_patterns_half[l];ll++){
		unsigned long long upperID = (long long)tableID_half_table[uu+uoffset];
		unsigned long long lowerID = (long long)tableID_half_table[ll+loffset];
		tableID = (upperID << half_table_size) | lowerID;
		unsigned long long reversed = 0;
		int reverse_bit[width];
		for(i = 0;i < hight; i++){
		  reverse_bit[i] = (tableID >> width*i ) & (reverse_length-1);
		  reversed = reversed | (((long long)reversed_bit_table[reverse_bit[i]]) << width*i);
		}
		if(tableID <= reversed){
		  //init_combo_info(color_combo, num_drops_combo, isLine_combo, combo_length);
		  int combo_counter = 0;
		  unsigned long long color_table[NUM_COLORS];
		  int num_c;
		  for(num_c = 0;num_c < NUM_COLORS;num_c++){
		    color_table[num_c] = 0;
		  }
		  switch(table_size){
		  case SMALL_TABLE:
		    generate_table_small(tableID, color_table);
		    break;
		  case NORMAL_TABLE:
		    generate_table_normal(tableID, color_table);
		    break;
		  case BIG_TABLE:
		    generate_table_big(tableID, color_table);
		    break;
		  default:
		    fprintf(stderr, "unknown table size\n");
		    exit(1);
		  }
		
		  int returned_combo_counter = 0;
		  do{
		    combo_counter = returned_combo_counter;
		  
		    switch(table_size){
		    case SMALL_TABLE:
		      returned_combo_counter = one_step_small(color_table, color_combo, num_drops_combo, isLine_combo, combo_counter, NUM_COLORS);
		      break;
		    case NORMAL_TABLE:
		      returned_combo_counter = one_step_normal(color_table, color_combo, num_drops_combo, isLine_combo, combo_counter, NUM_COLORS);
		      break;
		    case BIG_TABLE:
		      returned_combo_counter = one_step_big(color_table, color_combo, num_drops_combo, isLine_combo, combo_counter, NUM_COLORS);
		      break;
		    }
		  }while(returned_combo_counter != combo_counter);
		  //float power = return_attack(combo_counter, color_combo, num_drops_combo, isLine_combo, LS, isStrong, pline, pway);
		  float power[2];
		  return_attack_double(power, combo_counter, color_combo, num_drops_combo, isLine_combo, LS, isStrong, pline, pway);
		
		  if(max_power[thread_num][0][rank-1] < power[0]){
		    for(j = 0;j < rank;j++){
		      if(max_power[thread_num][0][j] < power[0]){
			for(k = rank-2;k >= j;k--){
			  max_powerID[thread_num][0][k+1] = max_powerID[thread_num][0][k];
			  max_power[thread_num][0][k+1] = max_power[thread_num][0][k];
			}
			max_powerID[thread_num][0][j] = tableID;
			max_power[thread_num][0][j] = power[0];
			break;
		      }
		    }
		  }
		  if(max_power[thread_num][1][rank-1] < power[1]){
		    for(j = 0;j < rank;j++){
		      if(max_power[thread_num][1][j] < power[1]){
			for(k = rank-2;k >= j;k--){
			  max_powerID[thread_num][1][k+1] = max_powerID[thread_num][1][k];
			  max_power[thread_num][1][k+1] = max_power[thread_num][1][k];
			}
			max_powerID[thread_num][1][j] = (~tableID) & filter;
			max_power[thread_num][1][j] = power[1];
			break;
		      }
		    }
		  }

		}
	      }
	    }
	  }
	}

      } //omp end

      float MP[2][rank];
      unsigned long long MID[2][rank];
      int ms;
      for(ms = 0; ms < 2; ms++){
	for(i = 0;i < rank;i++){
	  MP[ms][i] = 0.0;
	  MID[ms][i]= 0;
	}
      
	for(i = 0;i < num_threads;i++){
	  for(j = 0;j < rank;j++){
	    float power = max_power[i][ms][j];
	    tableID = max_powerID[i][ms][j];
	    if(MP[ms][rank-1] < power){
	      for(k = 0;k < rank;k++){
		if(MP[ms][k] < power){
		  for(m = rank-2;m >= k;m--){
		    MID[ms][m+1] = MID[ms][m];
		    MP[ms][m+1] = MP[ms][m];
		  }
		  MID[ms][k] = tableID;
		  MP[ms][k] = power;
		  break;
		}
	      }
	    }
	  }
	}
	for(i = 0;i < rank;i++){
	  float power = MP[ms][i];
	  unsigned long long tmp = MID[ms][i];
	  unsigned long long minID = tmp;
	  int index = i;
	  for(j = i+1;j < rank;j++){
	    if(power == MP[ms][j]){
	      if(minID > MID[ms][j]){
		minID = MID[ms][j];
		index = j;
	      }
	    }else{
	      break;
	    }
	  }
	  MID[ms][index] = tmp;
	  MID[ms][i] = minID;
	}
	for(i = 0;i < rank;i++){
	  final_MID[num_attacks][i] = MID[0][i];
	  final_MP [num_attacks][i] = MP [0][i];
	  final_MID[width*hight-num_attacks][i] = MID[1][i];
	  final_MP [width*hight-num_attacks][i] = MP [1][i];
	}
      }
    }
  }
  for(num_attacks = start;num_attacks <= end;num_attacks++){
    printf("%2d-%2d, line %d, way %d\n", num_attacks, width*hight-num_attacks, line, way);

    if(simuave){
      simulate_average(table_size, final_MID[num_attacks], final_MP[num_attacks], num_attacks, width, hight, LS, isStrong, pline, pway);
      
    }else{
      for(i = 0;i < rank;i++){
	printf("%d,max ID,%lld,power,%f\n",i,final_MID[num_attacks][i],final_MP[num_attacks][i]);
      }
    }
  }
}

#define WID 7
int one_step_small(unsigned long long *color_table, int *color_combo, int *num_drops_combo, int *isLine_combo, int finish, int num_colors){
  // 0 → width
  // ↓
  // hight
  // 000000000
  // 000000000
  // 000000000
  // 000000000
  // 000000000
  // 000000010
  // 000000111
  // 000000010
  
  unsigned long long isErase_tables[num_colors];
  int combo_counter = finish;
  int num_c;
  unsigned long long tmp, tmp2;

  for(num_c = 0;num_c < num_colors;num_c++){
    
    unsigned long long color = color_table[num_c];
    //自身の上下シフト・左右シフトとビット積をとる。その上下・左右が消すべきビット
    unsigned long long n, w, s, e;
    n = color >> WID;
    w = color >> 1;
    s = color << WID;
    e = color << 1;
    tmp  = (color & n & s);
    tmp  = tmp  | (tmp  >> WID) | (tmp  << WID);
    tmp2 = (color & w & e);
    tmp2 = tmp2 | (tmp2 >> 1  ) | (tmp2 << 1  );
    isErase_tables[num_c] = (color & tmp) | (color & tmp2);
  }

// #if num_colors==2
/*   if(isErase_tables[0] == isErase_tables[1])  */
/*     return combo_counter; */
//   // isErase_table[0~N] == 0, つまりは消えるドロップがないなら以降の処理は必要ない。
//   // が、しかしおそらくWarp divergenceの関係で、ない方が速い。(少なくともGPUでは) (CPUでもない方が速いことを確認。分岐予測の方が優秀)
//   // とすれば、isEraseをtableにしてループ分割する必要はないが、おそらく最適化の関係で分割した方が速い。
// #endif

  for(num_c = 0;num_c < num_colors;num_c++){
    unsigned long long isErase_table = isErase_tables[num_c];
    color_table[num_c] = color_table[num_c] & (~isErase_table);

    unsigned long long p = 1L << (WID+1);
    while(isErase_table) {
      while(!(isErase_table & p)){
	p = p << 1;
      }
      
      tmp = p;
      color_combo[combo_counter] = num_c;
      unsigned long long tmp_old;
      do{
	// ひとかたまりで消えるドロップは必ず隣接しているので、上下左右の隣接bitを探索する。
	// 消去ドロップの仕様変更のおかげで可能になった
	tmp_old = tmp;
	tmp = (tmp | (tmp << 1) | (tmp >> 1) | (tmp << WID) | (tmp >> WID)) & isErase_table;
      }while(tmp_old != tmp);
      isErase_table = isErase_table & (~tmp);

      // tmp の立ってるbit数を数えることで、ひとかたまりのドロップ数を数える
      // いわゆるpopcnt。

//       int b1, b2, b3, b4, b5, b6;
//       b1 = tmp >> (WID*1+1) & 127;
//       b2 = tmp >> (WID*2+1) & 127;
//       b3 = tmp >> (WID*3+1) & 127;
//       b4 = tmp >> (WID*4+1) & 127;
//       b5 = tmp >> (WID*5+1) & 127;
//       b6 = tmp >> (WID*6+1) & 127;
//       num_drops_combo[combo_counter] = bit_count_table[b1] + bit_count_table[b2] 
// 	+ bit_count_table[b3] + bit_count_table[b4] + bit_count_table[b5] + bit_count_table[b6];
      unsigned long long bits = tmp;
      bits = (bits & 0x5555555555555555LU) + ((bits >> 1) & 0x5555555555555555LU);
      bits = (bits & 0x3333333333333333LU) + ((bits >> 2) & 0x3333333333333333LU);
      bits = (bits + (bits >> 4)) & 0x0F0F0F0F0F0F0F0FLU;
      bits = bits + (bits >> 8);
      bits = bits + (bits >> 16);
      bits = (bits + (bits >> 32)) & 0x0000007F;
      num_drops_combo[combo_counter] = bits;
//       num_drops_combo[combo_counter] = __popcnt(tmp);

      isLine_combo[combo_counter] = ((tmp >> (WID  +1)) & 31) == 31
	|| ((tmp >> (WID*2+1)) & 31) == 31
	|| ((tmp >> (WID*3+1)) & 31) == 31
	|| ((tmp >> (WID*4+1)) & 31) == 31;
//       bits = tmp;
//       bits = bits & (bits >> 1);
//       bits = bits & (bits >> 2);
//       bits = bits & (bits >> 3);
//       isLine_combo[combo_counter] = ((bits & 36099303471055872L) != 0);
      combo_counter++;
    }
  }

  if(finish != combo_counter){
    unsigned long long exist_table = color_table[0];
    for(num_c = 1;num_c < num_colors;num_c++){
      exist_table = exist_table | color_table[num_c];
    }
    
    unsigned long long exist_org;
    do{
      exist_org = exist_table;
      
      unsigned long long exist_u = (exist_table >> WID) | 16642998272L;
      for(num_c = 0;num_c < num_colors;num_c++){
	unsigned long long color = color_table[num_c];
	unsigned long long color_u = color & exist_u;
	unsigned long long color_d = (color << WID) & (~exist_table) & (~2130303778816L); 
	color_table[num_c] = color_u | color_d;
      }
      exist_table = color_table[0];
      for(num_c = 1;num_c < num_colors;num_c++){
	exist_table = exist_table | color_table[num_c];
      }
    }while(exist_org != exist_table);
  }
//     if(threadIdx.x == 0 && blockIdx.x == 0){
//      print_table(color_table);
//      printf("%lld, %lld\n", exist_org, exist_table);
//     }
  return combo_counter;
}
#undef WID


#define WID 8
int one_step_normal(unsigned long long *color_table, int *color_combo, int *num_drops_combo, int *isLine_combo, int finish, int num_colors){
  // 0 → width
  // ↓
  // hight
  // 000000000
  // 000000000
  // 000000000
  // 000000000
  // 000000000
  // 000000010
  // 000000111
  // 000000010
  
  unsigned long long isErase_tables[num_colors];
  int combo_counter = finish;
  int num_c;
  unsigned long long tmp, tmp2;

  for(num_c = 0;num_c < num_colors;num_c++){
    
    unsigned long long color = color_table[num_c];
    //自身の上下シフト・左右シフトとビット積をとる。その上下・左右が消すべきビット
    unsigned long long n, w, s, e;
    n = color >> WID;
    w = color >> 1;
    s = color << WID;
    e = color << 1;
    tmp  = (color & n & s);
    tmp  = tmp  | (tmp  >> WID) | (tmp  << WID);
    tmp2 = (color & w & e);
    tmp2 = tmp2 | (tmp2 >> 1  ) | (tmp2 << 1  );
    isErase_tables[num_c] = (color & tmp) | (color & tmp2);
  }

// #if num_colors==2
/*   if(isErase_tables[0] == isErase_tables[1])  */
/*     return combo_counter; */
//   // isErase_table[0~N] == 0, つまりは消えるドロップがないなら以降の処理は必要ない。
//   // が、しかしおそらくWarp divergenceの関係で、ない方が速い。(少なくともGPUでは) (CPUでもない方が速いことを確認。分岐予測の方が優秀)
//   // とすれば、isEraseをtableにしてループ分割する必要はないが、おそらく最適化の関係で分割した方が速い。
// #endif

  for(num_c = 0;num_c < num_colors;num_c++){
    unsigned long long isErase_table = isErase_tables[num_c];
    color_table[num_c] = color_table[num_c] & (~isErase_table);

    unsigned long long p = 1L << (WID+1);
    while(isErase_table) {
      while(!(isErase_table & p)){
	p = p << 1;
      }
      
      tmp = p;
      color_combo[combo_counter] = num_c;
      unsigned long long tmp_old;
      do{
	// ひとかたまりで消えるドロップは必ず隣接しているので、上下左右の隣接bitを探索する。
	// 消去ドロップの仕様変更のおかげで可能になった
	tmp_old = tmp;
	tmp = (tmp | (tmp << 1) | (tmp >> 1) | (tmp << WID) | (tmp >> WID)) & isErase_table;
      }while(tmp_old != tmp);
      isErase_table = isErase_table & (~tmp);

      // tmp の立ってるbit数を数えることで、ひとかたまりのドロップ数を数える
      // いわゆるpopcnt。

//       int b1, b2, b3, b4, b5, b6;
//       b1 = tmp >> (WID*1+1) & 127;
//       b2 = tmp >> (WID*2+1) & 127;
//       b3 = tmp >> (WID*3+1) & 127;
//       b4 = tmp >> (WID*4+1) & 127;
//       b5 = tmp >> (WID*5+1) & 127;
//       b6 = tmp >> (WID*6+1) & 127;
//       num_drops_combo[combo_counter] = bit_count_table[b1] + bit_count_table[b2] 
// 	+ bit_count_table[b3] + bit_count_table[b4] + bit_count_table[b5] + bit_count_table[b6];
      unsigned long long bits = tmp;
      bits = (bits & 0x5555555555555555LU) + ((bits >> 1) & 0x5555555555555555LU);
      bits = (bits & 0x3333333333333333LU) + ((bits >> 2) & 0x3333333333333333LU);
      bits = (bits + (bits >> 4)) & 0x0F0F0F0F0F0F0F0FLU;
      bits = bits + (bits >> 8);
      bits = bits + (bits >> 16);
      bits = (bits + (bits >> 32)) & 0x0000007F;
      num_drops_combo[combo_counter] = bits;
//       num_drops_combo[combo_counter] = __popcnt(tmp);

      isLine_combo[combo_counter] = ((tmp >> (WID  +1)) & 63) == 63
	|| ((tmp >> (WID*2+1)) & 63) == 63
	|| ((tmp >> (WID*3+1)) & 63) == 63
	|| ((tmp >> (WID*4+1)) & 63) == 63
	|| ((tmp >> (WID*5+1)) & 63) == 63;
//       bits = tmp;
//       bits = bits & (bits >> 1);
//       bits = bits & (bits >> 2);
//       bits = bits & (bits >> 3);
//       isLine_combo[combo_counter] = ((bits & 36099303471055872L) != 0);
      combo_counter++;
    }
  }

  if(finish != combo_counter){
    unsigned long long exist_table = color_table[0];
    for(num_c = 1;num_c < num_colors;num_c++){
      exist_table = exist_table | color_table[num_c];
    }
    
    unsigned long long exist_org;
    do{
      exist_org = exist_table;
      
      unsigned long long exist_u = (exist_table >> WID) | 138538465099776L;
      for(num_c = 0;num_c < num_colors;num_c++){
	unsigned long long color = color_table[num_c];
	unsigned long long color_u = color & exist_u;
	unsigned long long color_d = (color << WID) & (~exist_table) & (~35465847065542656L); // color << WIDが諸悪の根源。非常に扱いに気をつけるべき。bit_tableだとオーバーフローで消えるので(~354...)はいらない。
	color_table[num_c] = color_u | color_d;
      }
      exist_table = color_table[0];
      for(num_c = 1;num_c < num_colors;num_c++){
	exist_table = exist_table | color_table[num_c];
      }
    }while(exist_org != exist_table);
  }
//     if(threadIdx.x == 0 && blockIdx.x == 0){
//      print_table(color_table);
//      printf("%lld, %lld\n", exist_org, exist_table);
//     }
  return combo_counter;
}
#undef WID


#define WID 9
int one_step_big(unsigned long long *color_table, int *color_combo, int *num_drops_combo, int *isLine_combo, int finish, int num_colors){
  // 0 → width
  // ↓
  // hight
  // 000000000
  // 000000000
  // 000000000
  // 000000000
  // 000000000
  // 000000010
  // 000000111
  // 000000010
  
  unsigned long long isErase_tables[num_colors];
  //unsigned long long isErase_table;
  int combo_counter = finish;
  int num_c;
  unsigned long long tmp, tmp2;

  for(num_c = 0;num_c < num_colors;num_c++){
    
    unsigned long long color = color_table[num_c];

    unsigned long long n, w, s, e;
    n = color >> WID;
    w = color >> 1;
    s = color << WID;
    e = color << 1;
    tmp  = (color & n & s);
    tmp  = tmp  | (tmp  >> WID) | (tmp  << WID);
    tmp2 = (color & w & e);
    tmp2 = tmp2 | (tmp2 >> 1  ) | (tmp2 << 1  );
    isErase_tables[num_c] = (color & tmp) | (color & tmp2);
    //isErase_table = (color & tmp) | (color & tmp2);

  }

/* #if num_colors==2 */
/*   if(isErase_tables[0] == isErase_tables[1]) */
/*     return combo_counter; */
/* #endif */

  for(num_c = 0;num_c < num_colors;num_c++){
    unsigned long long isErase_table = isErase_tables[num_c];
    color_table[num_c] = color_table[num_c] & (~isErase_table);

    unsigned long long p = 1L << (WID+1);
    while(isErase_table) {
      while(!(isErase_table & p)){
	p = p << 1;
      }
      
      tmp = p;
      color_combo[combo_counter] = num_c;
      unsigned long long tmp_old;
      do{
	tmp_old = tmp;
	tmp = (tmp | (tmp << 1) | (tmp >> 1) | (tmp << WID) | (tmp >> WID)) & isErase_table;
      }while(tmp_old != tmp);
      isErase_table = isErase_table & (~tmp);
//       int b1, b2, b3, b4, b5, b6;
//       b1 = tmp >> (WID*1+1) & 127;
//       b2 = tmp >> (WID*2+1) & 127;
//       b3 = tmp >> (WID*3+1) & 127;
//       b4 = tmp >> (WID*4+1) & 127;
//       b5 = tmp >> (WID*5+1) & 127;
//       b6 = tmp >> (WID*6+1) & 127;
//       num_drops_combo[combo_counter] = bit_count_table[b1] + bit_count_table[b2] 
// 	+ bit_count_table[b3] + bit_count_table[b4] + bit_count_table[b5] + bit_count_table[b6];
      unsigned long long bits = tmp;
      bits = (bits & 0x5555555555555555LU) + ((bits >> 1) & 0x5555555555555555LU);
      bits = (bits & 0x3333333333333333LU) + ((bits >> 2) & 0x3333333333333333LU);
      bits = (bits + (bits >> 4)) & 0x0F0F0F0F0F0F0F0FLU;
      bits = bits + (bits >> 8);
      bits = bits + (bits >> 16);
      bits = (bits + (bits >> 32)) & 0x0000007F;
      num_drops_combo[combo_counter] = bits;
//       num_drops_combo[combo_counter] = __popcll(tmp);

      isLine_combo[combo_counter] = ((tmp >> (WID  +1)) & 127) == 127
	|| ((tmp >> (WID*2+1)) & 127) == 127
	|| ((tmp >> (WID*3+1)) & 127) == 127
	|| ((tmp >> (WID*4+1)) & 127) == 127
	|| ((tmp >> (WID*5+1)) & 127) == 127
	|| ((tmp >> (WID*6+1)) & 127) == 127;
//       bits = tmp;
//       bits = bits & (bits >> 1);
//       bits = bits & (bits >> 2);
//       bits = bits & (bits >> 3);
//       isLine_combo[combo_counter] = ((bits & 36099303471055872L) != 0);
      
      combo_counter++;
    }
  }
  
  if(finish != combo_counter){
    unsigned long long exist_table = color_table[0];
    for(num_c = 1;num_c < num_colors;num_c++){
      exist_table = exist_table | color_table[num_c];
    }
    
    unsigned long long exist_org;
    do{
      exist_org = exist_table;
      
      unsigned long long exist_u = (exist_table >> WID) | 4575657221408423936L;
      
      for(num_c = 0;num_c < num_colors;num_c++){
	unsigned long long color = color_table[num_c];
	unsigned long long color_u = color & exist_u;
	unsigned long long color_d = (color << WID) & (~exist_table);
	color_table[num_c] = color_u | color_d;
      }
      exist_table = color_table[0];
      for(num_c = 1;num_c < num_colors;num_c++){
	exist_table = exist_table | color_table[num_c];
      }
    }while(exist_org != exist_table);
  }

  return combo_counter;
}
#undef WID


void print_table(const unsigned long long *color_table, const int width, const int hight){

  int i, j;
  for(i = 1;i <= hight;i++){
    for(j = 1;j <= width;j++){
      unsigned long long p = (1L << ((width+2)*i+j));
      if((color_table[0] & p) == p)
	printf("G ");
      else if((color_table[1] & p) == p)
	printf("Y ");
      else
	printf("? ");
    }
    putchar('\n');
  }
  putchar('\n');
}


void print_table2(const unsigned long long color_table, const int width, const int hight){
  int i, j;
  for(i = 1;i <= hight;i++){
    for(j = 1;j <= width;j++){
      unsigned long long p = (1L << ((width+2)*i+j));
      printf("%d ", (color_table & p) == p);
    }
    putchar('\n');
  }
  putchar('\n');
}

void ID2table(const unsigned long long ID, const int width, const int hight){

  int i, j;
  for(i = 0;i < hight;i++){
    for(j = 0;j < width;j++){
      unsigned long long p = (1L << ((width)*i+j));
      if((ID & p) == p)
	printf("G ");
      else 
	printf("Y ");
    }
    putchar('\n');
  }
  putchar('\n');
}


float return_attack(const int combo_counter, int *const color_combo, int *const num_drops_combo, int *const isLine_combo, const int LS, const int strong, const float line, const float way){
  // used for simulation mode
  // [FIXME] check only Green attack
  const float AT = 1.0;
  int num_line = 0;
  float attack = 0;
  float l = 1.0;
  int i;
  for(i = 0;i < combo_counter;i++){
    int color = color_combo[i];
    float drop_pwr;
    switch(color){
    case MAINCOLOR:
      drop_pwr = num_drops_combo[i]==4 ? (1+0.25*(num_drops_combo[i]-3))*way : 1+0.25*(num_drops_combo[i]-3); 
      if(strong)
	drop_pwr = drop_pwr * (1+0.06*num_drops_combo[i]);
      attack += drop_pwr; 
      if(isLine_combo[i]) num_line++;
      break;
    default:
      break;
    }
  }

  int heroLS6;
  int heroLS7;
  int heroLS8;
  int count;
  switch(LS){
  case HERO: 
    heroLS6 = 0;
    heroLS7 = 0;
    heroLS8 = 0;
    for(i = 0;i < combo_counter;i++){
      if(MAINCOLOR == color_combo[i]){
	int num_drops = num_drops_combo[i];
	if(num_drops == 6)
	  heroLS6 = 1;
	if(num_drops == 7)
	  heroLS7 = 1;
	if(num_drops >= 8)
	  heroLS8 = 1;
      }
    }
    if(heroLS8)
      l = 16;
    else if(heroLS7)
      l = 12.25;
    else if(heroLS6)
      l = 9;
    else
      l = 1;
    break;
  case SONIA:
    if(combo_counter < 6)
      l = 6.25;
    else
      l = 2.75*2.75;
    break;
  case KRISHNA:
    count = 0;
    for(i = 0;i < combo_counter;i++){
      if(MAINCOLOR == color_combo[i]){
	count++;
	int num_drops = num_drops_combo[i];
	if(num_drops == 5)
	  l = 2.25;
      }
    }
    if(count == 2)
      l = l * 3 * 3;
    else if(count >= 3)
      l = l * 4.5 * 4.5;
    else
      l = 1;
    break;
  case BASTET:
    if(combo_counter == 5)
      l = 3.0*3.0;
    else if(combo_counter == 6)
      l = 3.5*3.5;
    else if(combo_counter >= 7)
      l = 4.0*4.0;
    else
      l = 1.0;
    break;
  case LAKU_PARU:
    l = 6.25;
    for(i = 0;i < combo_counter;i++){
      if(MAINCOLOR != color_combo[i]){
	int num_drops = num_drops_combo[i];
	if(num_drops >= 5)
	  l = 25;
      }
    }
    break;
  default:
    break;
  }
    
  attack = attack * (1+0.25*(combo_counter-1)) * AT * l * (1+0.1*line*num_line) ;
  return attack;
}


void return_attack_double(float *power, const int combo_counter, int *const color_combo, int *const num_drops_combo, int *const isLine_combo, const int LS, const int strong, const float line, const float way){
  // used for simulation mode
  // [FIXME] check only Green attack
  const float AT = 1.0;
  int num_line_m = 0;
  float attack_m = 0;
  int num_line_s = 0;
  float attack_s = 0;
  float l_m = 1.0;
  float l_s = 1.0;
  int i;
  float drop_pwr;
  for(i = 0;i < combo_counter;i++){
    int color = color_combo[i];
    switch(color){
    case MAINCOLOR:
      drop_pwr = num_drops_combo[i]==4 ? (1+0.25*(num_drops_combo[i]-3))*way : 1+0.25*(num_drops_combo[i]-3); 
      if(strong)
	drop_pwr = drop_pwr * (1+0.06*num_drops_combo[i]);
      attack_m += drop_pwr; 
      if(isLine_combo[i]) num_line_m++;
      break;
    case SUBCOLOR:
      drop_pwr = num_drops_combo[i]==4 ? (1+0.25*(num_drops_combo[i]-3))*way : 1+0.25*(num_drops_combo[i]-3); 
      if(strong)
	drop_pwr = drop_pwr * (1+0.06*num_drops_combo[i]);
      attack_s += drop_pwr; 
      if(isLine_combo[i]) num_line_s++;
      break;
    default:
      break;
    }
  }

  int count_m;
  int count_s;
  switch(LS){
  case HERO: 
    for(i = 0;i < combo_counter;i++){
      if(MAINCOLOR == color_combo[i]){
	int num_drops = num_drops_combo[i];
	if(num_drops >= 8){
	  l_m = 16;
	}else if(num_drops == 7 && l_m < 12.25){
	  l_m = 12.25;
	}else if(num_drops == 6 && l_m < 9){
	  l_m = 9;
	}
      }
      if(SUBCOLOR == color_combo[i]){
	int num_drops = num_drops_combo[i];
	if(num_drops >= 8){
	  l_s = 16;
	}else if(num_drops == 7 && l_s < 12.25){
	  l_s = 12.25;
	}else if(num_drops == 6 && l_s < 9){
	  l_s = 9;
	}
      }
    }
    break;
  case SONIA:
    if(combo_counter < 6){
      l_m = 6.25;
      l_s = 6.25;
    }else{
      l_m = 2.75*2.75;
      l_s = 2.75*2.75;
    }
    break;
  case KRISHNA:
    count_m = 0;
    for(i = 0;i < combo_counter;i++){
      if(MAINCOLOR == color_combo[i]){
	count_m++;
	int num_drops = num_drops_combo[i];
	if(num_drops == 5)
	  l_m = 2.25;
      }
    }
    if(count_m == 2)
      l_m = l_m * 3 * 3;
    else if(count_m >= 3)
      l_m = l_m * 4.5 * 4.5;
    else
      l_m = 1;
    count_s = 0;
    for(i = 0;i < combo_counter;i++){
      if(SUBCOLOR == color_combo[i]){
	count_s++;
	int num_drops = num_drops_combo[i];
	if(num_drops == 5)
	  l_s = 2.25;
      }
    }
    if(count_s == 2)
      l_s = l_s * 3 * 3;
    else if(count_s >= 3)
      l_s = l_s * 4.5 * 4.5;
    else
      l_s = 1;
    break;
  case BASTET:
    if(combo_counter == 5){
      l_m = 3.0*3.0;
      l_s = 3.0*3.0;
    }else if(combo_counter == 6){
      l_m = 3.5*3.5;
      l_s = 3.5*3.5;
    }else if(combo_counter >= 7){
      l_m = 4.0*4.0;
      l_s = 4.0*4.0;
    }else{
      l_m = 1.0;
      l_s = 1.0;
    }
    break;
  case LAKU_PARU:
    l_m = 6.25;
    l_s = 6.25;
    for(i = 0;i < combo_counter;i++){
      if(SUBCOLOR == color_combo[i]){
	int num_drops = num_drops_combo[i];
	if(num_drops >= 5)
	  l_m = 25;
      }
      if(MAINCOLOR == color_combo[i]){
	int num_drops = num_drops_combo[i];
	if(num_drops >= 5)
	  l_s = 25;
      }
    }
    break;
  default:
    break;
  }
    
  power[0] = attack_m * (1+0.25*(combo_counter-1)) * AT * l_m * (1+0.1*line*num_line_m);
  power[1] = attack_s * (1+0.25*(combo_counter-1)) * AT * l_s * (1+0.1*line*num_line_m);
}

void fill_random(unsigned long long *color_table, const int width, const int hight){
  int i, j, k;
  for(i = 1;i <= hight;i++){
    for(j = 1;j <= width;j++){
      unsigned long long p = (1L << ((width+2)*i+j));
      int flag = 1;
      for(k = 0;k < 6;k++){
	if((color_table[k] & p) == p){
	  flag = 0;
	}
      }
      if(flag){
	int color = rand()%6;
	color_table[color] = color_table[color] | p;
      }
    }
  }
}

void simulate_average(const int table_size, unsigned long long * const MID, float * const MP, const int num_attacks, const int width, const int hight, const int LS, const int isStrong, const float line, const float way){

  int combo_length = 100;
  int rank = RANKINGLENGTH;
  int i, j;
  int color_combo[combo_length];
  int num_drops_combo[combo_length];
  int isLine_combo[combo_length];
  float average_power[RANKINGLENGTH];
  float min_power[RANKINGLENGTH];
  float average_combo[RANKINGLENGTH];
  int   min_combo[RANKINGLENGTH];

#pragma omp parallel for private(color_combo, num_drops_combo, isLine_combo, j) 
  for(i = 0;i < rank;i++){
    unsigned long long tableID = MID[i];
    float pave = 0.0;
    float cave = 0.0;
    float pmin = 1000000000.0;
    int cmin = combo_length;
    for(j = 0;j < 10000;j++){
      //init_combo_info(color_combo, num_drops_combo, isLine_combo, combo_length);
      int combo_counter = 0;
      unsigned long long color_table[6];
      int num_c;
      for(num_c = 0;num_c < 6;num_c++){
	color_table[num_c] = 0;
      }
      switch(table_size){
      case SMALL_TABLE:
	generate_table_small(tableID, color_table);
	break;
      case NORMAL_TABLE:
	generate_table_normal(tableID, color_table);
	break;
      case BIG_TABLE:
	generate_table_big(tableID, color_table);
	break;
      default:
	fprintf(stderr, "unknown table size\n");
	exit(1);
      }
      int returned_combo_counter = 0;
      do{
	combo_counter = returned_combo_counter;
	switch(table_size){
	case SMALL_TABLE:
	  returned_combo_counter = one_step_small(color_table, color_combo, num_drops_combo, isLine_combo, combo_counter, 6);
	  break;
	case NORMAL_TABLE:
	  returned_combo_counter = one_step_normal(color_table, color_combo, num_drops_combo, isLine_combo, combo_counter, 6);
	  break;
	case BIG_TABLE:
	  returned_combo_counter = one_step_big(color_table, color_combo, num_drops_combo, isLine_combo, combo_counter, 6);
	  break;
	}
	fill_random(color_table, width, hight);
      }while(returned_combo_counter != combo_counter);
      float power = return_attack(combo_counter, color_combo, num_drops_combo, isLine_combo, LS, isStrong, line, way);
      if(power < pmin){
	pmin = power;
      }
      if(combo_counter < cmin){
	cmin = combo_counter;
      }
      pave += power;
      cave += combo_counter;
    }
    average_combo[i] = cave/10000.0;
    average_power[i] = pave/10000.0;
    min_combo[i] = cmin;
    min_power[i] = pmin;
  } 
  printf("rank,tableID      ,power     ,ave power ,min power ,ave combo ,min combo\n");
  for(i = 0;i < rank;i++){
    printf("%4d,%13lld,%10.3f,%10.3f,%10.3f,%10.3f,%6d\n",i,MID[i],MP[i],average_power[i],min_power[i],average_combo[i],min_combo[i]);
  }
}
