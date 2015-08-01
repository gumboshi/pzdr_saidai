#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

/* #define R 1 */
/* #define B 2 */
/* #define G 3 */
/* #define Y 4 */
/* #define P 5 */
/* #define PI 6 */
#define G 0 
#define Y 1
#define width  7
#define hight  6
#define WID (width+2)
#define HIG (hight+2)
#define COMBO_LENGTH 14
#define REVERSE_LENGTH 128
#define MAXID 4398046511104
#define MIN(A,B) ((A) < (B) ? (A) : (B))
#define MIN3(A,B,C) (MIN(C,MIN(A,B)))
#define MAX(A,B) ((A) > (B) ? (A) : (B))
#define MAX3(A,B,C) (MAX(C,MAX(A,B)))
#define RANKINGLENGTH 10

#define LS 1
#define ATT 1
#define LINE 6
#define POWER_2WAY 2.25
#define STRONG_DROP 0
#define HERO 0
#define SONIA 0
#define KRISHNA 0
#define BASTET 0
#define MYCOLOR G
#define NUM_COLORS 2

#define NUM_BLOCK 52
#define NUM_THREAD 256

#define CUDA_SAFE_CALL(func)			\
  do {						\
    cudaError_t err = (func);						\
    if (err != cudaSuccess) {						\
      fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
      exit(err);							\
    }									\
  } while(0)


float line = LINE;
float way = POWER_2WAY;

long long num_patterns[43] = {
  1,
  42,
  861,
  11480,
  111930,
  850668,
  5245786,
  26978328,
  118030185,
  445891810,
  1471442973,
  4280561376,
  11058116888,
  25518731280,
  52860229080,
  98672427616,
  166509721602,
  254661927156,
  353697121050,
  446775310800,
  513791607420,
  538257874440,
  513791607420,
  446775310800,
  353697121050,
  254661927156,
  166509721602,
  98672427616,
  52860229080,
  25518731280,
  11058116888,
  4280561376,
  1471442973,
  445891810,
  118030185,
  26978328,
  5245786,
  850668,
  111930,
  11480,
  861,
  42,
  1
};


long long num_patterns_omitted[43] = {
  1,
  24,
  447,
  5804,
  56184,
  425976,
  2624584,
  13493196,
  59023899,
  222963704,
  735754917,
  2140339440,
  5529155344,
  12759516192,
  26430335472,
  49336520624,
  83255264874,
  127331468784,
  176849160982,
  223388334312,
  256896534336,
  269129685968,
  256896534336,
  223388334312,
  176849160982,
  127331468784,
  83255264874,
  49336520624,
  26430335472,
  12759516192,
  5529155344,
  2140339440,
  735754917,
  222963704,
  59023899,
  13493196,
  2624584,
  425976,
  56184,
  5804,
  447,
  24,
  1
};

int num_patterns21[22] = {
  1,
  21,
  210,
  1330,
  5985,
  20349,
  54264,
  116280,
  203490,
  293930,
  352716,
  352716,
  293930,
  203490,
  116280,
  54264,
  20349,
  5985,
  1330,
  210,
  21,
  1
};

int table_prefix21[22] = {
  0,
  1,
  22,
  232,
  1562,
  7547,
  27896,
  82160,
  198440,
  401930,
  695860,
  1048576,
  1401292,
  1695222,
  1898712,
  2014992,
  2069256,
  2089605,
  2095590,
  2096920,
  2097130,
  2097151
};

__device__ void init_combo_info(int color_combo[COMBO_LENGTH], int num_drops_combo[COMBO_LENGTH], int isLine_combo[COMBO_LENGTH]);
__device__ void generate_table(long long tableID, long long color_table[NUM_COLORS]);
__device__ int one_step(long long color_table[NUM_COLORS], int color_combo[COMBO_LENGTH], int num_drops_combo[COMBO_LENGTH], int isLine_combo[COMBO_LENGTH], int finish);
__device__ int one_step_opt(long long color_table[NUM_COLORS], int color_combo[COMBO_LENGTH], int num_drops_combo[COMBO_LENGTH], int isLine_combo[COMBO_LENGTH], int finish);
__device__ int one_step_opt2(long long color_table[NUM_COLORS], int color_combo[COMBO_LENGTH], int num_drops_combo[COMBO_LENGTH], int isLine_combo[COMBO_LENGTH], int finish);
__device__ int one_step_opt3(long long color_table[NUM_COLORS], int color_combo[COMBO_LENGTH], int num_drops_combo[COMBO_LENGTH], int isLine_combo[COMBO_LENGTH], const int *bit_hash_table, int finish);
__device__ int one_step_opt4(long long color_table[NUM_COLORS], int color_combo[COMBO_LENGTH], int num_drops_combo[COMBO_LENGTH], int isLine_combo[COMBO_LENGTH], const int *bit_hash_table, int finish);
__device__ int one_step_opt5(long long color_table[NUM_COLORS], int color_combo[COMBO_LENGTH], int num_drops_combo[COMBO_LENGTH], int isLine_combo[COMBO_LENGTH], int finish);
__device__ int one_step_opt_c2(long long *color_table, int color_combo[COMBO_LENGTH], int num_drops_combo[COMBO_LENGTH], int isLine_combo[COMBO_LENGTH], int finish);
__device__ float return_attack(int combo_counter, int color_combo[COMBO_LENGTH], int num_drops_combo[COMBO_LENGTH], int isLine_combo[COMBO_LENGTH], float line, float way);
__device__ void print_table(long long color_table[NUM_COLORS]);
__device__ void print_table2(long long color_table);

void init_bit_hash_table(int bit_hash_table[256]);
void init_reversed_bit_table(int bit_hash_table[REVERSE_LENGTH]);
long long find_next_tableID(long long tableID, int num_attacks);
void simulate_all(int num_attacks, int *bit_hash_table_dev, int *reversed_bit_table_dev, int *tableID_table21_dev, int *tableID_prefix21_dev, int *num_patterns21_dev);
void create_tableID_21(int *tableID_table21, int *tableID_prefix21, int bit_hash_table[256]);


double gettimeofday_sec()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + (double)tv.tv_usec*1e-6;
}

int main(int argc, char* argv[]){


  int start = 0;
  int end = width*hight;
  int i;
  if (argc != 1) {
    for (i = 1; i < argc; i++) {
      if (strcmp(argv[i], "-s") == 0) {
        i++;
        start = atoi(argv[i]);
      }
      else if (strcmp(argv[i], "-e") == 0) {
        i++;
        end = atoi(argv[i]);
      }
      else if (strcmp(argv[i], "-l") == 0) {
        i++;
        line = atoi(argv[i]);
      }
      else if (strcmp(argv[i], "-w") == 0) {
        i++;
        way = pow(1.5,atoi(argv[i]));
      }else {
        fprintf(stderr,"unknown option\n");
        exit(1);
      }
    }
  }
    
  int bit_hash_table[256];
  int reversed_bit_table[REVERSE_LENGTH];

  int *tableID_table21;
  tableID_table21 = (int*)malloc(sizeof(int)*2097152);
  int tableID_prefix21[22];

  init_bit_hash_table(bit_hash_table);
  init_reversed_bit_table(reversed_bit_table);
  create_tableID_21(tableID_table21,tableID_prefix21,bit_hash_table);

  int *tableID_table21_dev, *tableID_prefix21_dev, *num_patterns21_dev;
  int *bit_hash_table_dev, *reversed_bit_table_dev;
  CUDA_SAFE_CALL(cudaMalloc((void**)&bit_hash_table_dev,    sizeof(int) * 256));
  CUDA_SAFE_CALL(cudaMalloc((void**)&reversed_bit_table_dev,sizeof(int) * REVERSE_LENGTH));
  CUDA_SAFE_CALL(cudaMalloc((void**)&tableID_table21_dev,   sizeof(int) * 2097152));
  CUDA_SAFE_CALL(cudaMalloc((void**)&num_patterns21_dev,    sizeof(int) * 22));
  CUDA_SAFE_CALL(cudaMalloc((void**)&tableID_prefix21_dev,  sizeof(int) * 22));
  CUDA_SAFE_CALL(cudaMemcpy(bit_hash_table_dev,    bit_hash_table,     sizeof(int) * 256, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(reversed_bit_table_dev,reversed_bit_table, sizeof(int) * REVERSE_LENGTH, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(tableID_table21_dev,   tableID_table21,    sizeof(int) * 2097152, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(num_patterns21_dev,    num_patterns21,     sizeof(int) * 22, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(tableID_prefix21_dev,  tableID_prefix21,   sizeof(int) * 22, cudaMemcpyHostToDevice));

  int num_attacks;
  
  for(num_attacks = start;num_attacks <= end;num_attacks++){
    printf("num attack drops = %d, line %f, way %f\n", num_attacks, line, way);
    simulate_all(num_attacks, bit_hash_table_dev, reversed_bit_table_dev, tableID_table21_dev, tableID_prefix21_dev, num_patterns21_dev);
  }

  free(tableID_table21);
  CUDA_SAFE_CALL(cudaFree(tableID_table21_dev));
  CUDA_SAFE_CALL(cudaFree(num_patterns21_dev));
  CUDA_SAFE_CALL(cudaFree(tableID_prefix21_dev));
  CUDA_SAFE_CALL(cudaFree(bit_hash_table_dev));
  CUDA_SAFE_CALL(cudaFree(reversed_bit_table_dev));
  return 0;

}


__device__ inline void init_combo_info(int color_combo[COMBO_LENGTH], int num_drops_combo[COMBO_LENGTH], int isLine_combo[COMBO_LENGTH]){
  int i;
  for(i = 0;i < COMBO_LENGTH;i++){
    color_combo[i] = 0;
    num_drops_combo[i] = 0;
    isLine_combo[i] = 0;
  }
}

void init_bit_hash_table(int bit_hash_table[256]){
  int i, j;
  for(i = 0;i < 256;i++){
    int count = 0;
    int compared_num = i;
    for(j = 0;j < 8;j++){
      if((compared_num >> j & 1) == 1){
	count++;
      }
    }
    bit_hash_table[i] = count;
  }
}

void init_reversed_bit_table(int reversed_bit_table[REVERSE_LENGTH]){
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

long long find_next_tableID(long long tableID, int num_attacks){

  long long return_tableID = tableID;
  int i;
  while(tableID <= MAXID){
    int count = 0;
    long long compared_num = return_tableID;
    for(i = 0;i < width*hight; i++){
      if(((compared_num >> i) & 1) == 1){
	count++;
      }
    }
    if(count == num_attacks){
      return return_tableID;
    }
    return_tableID++;
  }
  return 0;
}


long long find_next_tableID_opt(long long tableID, int num_attacks, int bit_hash_table[256]){

  long long return_tableID = tableID;
  while(return_tableID <= MAXID){
    int compared_num0 = return_tableID & 255;
    int compared_num1 = (return_tableID >> 8 ) & 255;
    int compared_num2 = (return_tableID >> 16) & 255;
    int compared_num3 = (return_tableID >> 24) & 255;
    int compared_num4 = (return_tableID >> 32) & 255;
    int compared_num5 = (return_tableID >> 40) & 255;

    int count = bit_hash_table[compared_num0]
      +   bit_hash_table[compared_num1]
      +   bit_hash_table[compared_num2]
      +   bit_hash_table[compared_num3]
      +   bit_hash_table[compared_num4]
      +   bit_hash_table[compared_num5];
    if(count == num_attacks){
      return return_tableID;
    }
    return_tableID++;
  }
  return 0;
}


long long find_next_tableID_omit_flip_horizontal(long long tableID, int num_attacks, int bit_hash_table[256], int reversed_bit_table[REVERSE_LENGTH]){

  long long return_tableID = tableID;
  int i;
  int compared_num[(width*hight-1)/8+1];
  int bit_num[hight];
  while(return_tableID <= MAXID){
    int count = 0;
    for(i = 0;i < (width*hight-1)/8+1;i++){
      compared_num[i] = (return_tableID >> 8*i ) & 255;
      count += bit_hash_table[compared_num[i]];
    }
    if(count == num_attacks){
      long long reversed = 0;
      for(i = 0;i < hight; i++){
        bit_num[i] = (return_tableID >> width*i ) & (REVERSE_LENGTH-1);
        reversed += ((long long)reversed_bit_table[bit_num[i]]) << width*i;
      }
      if(return_tableID <= reversed){
        return return_tableID;
      }
    }
    return_tableID++;
  }
  return 0;
}

void create_tableID_21(int *tableID_table21, int *tableID_prefix21, int bit_hash_table[256]){

  int num_attacks;
  int sum = 0;
  for(num_attacks = 0;num_attacks <= 21;num_attacks++){
    tableID_prefix21[num_attacks] = sum;
    sum = sum + num_patterns21[num_attacks];
  }
  for(num_attacks = 0;num_attacks <= 21;num_attacks++){
    int compared_num[3];
    int tableID = 0;
    int index = 0;
    while(tableID <= 2097152){
      int count = 0;
      int i;
      for(i = 0;i < 3;i++){
	compared_num[i] = (tableID >> 8*i ) & 255;
	count += bit_hash_table[compared_num[i]];
      }    
      if(count == num_attacks){
	tableID_table21[index+tableID_prefix21[num_attacks]] = tableID;
	index++;
      }
      tableID++;
    }
  }
}


#if NUM_COLORS==2
/* static inline void generate_table(unsigned int tableID, long long color_table[NUM_COLORS]){ */

/*   long long ID = (long long) tableID; */
/*   long long b0 = ID & 63; */
/*   long long b1 = (ID >> 6 ) & 63; */
/*   long long b2 = (ID >> 12) & 63; */
/*   long long b3 = (ID >> 18) & 63; */
/*   long long b4 = (ID >> 24) & 63; */
/*   color_table[0] = (b0 << (WID+1)) | (b1 << (WID*2+1)) */
/*     | (b2 << (WID*3+1)) | (b3 << (WID*4+1)) | (b4 << (WID*5+1)); */
/*   color_table[1] = ((~b0 & 63) << (WID+1)) | ((~b1 & 63) << (WID*2+1)) */
/*     | ((~b2 & 63) << (WID*3+1)) | ((~b3 & 63) << (WID*4+1)) | ((~b4 & 63) << (WID*5+1)); */
/* } */

__device__ inline void generate_table(long long tableID, long long color_table[NUM_COLORS]){

  long long b0, b1, b2, b3, b4, b5;
  long long ID = tableID;
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

/* static inline void generate_table(unsigned int tableID, long long color_table[NUM_COLORS]){ */

/*   long long ID = (long long) tableID; */
/*   long long b0 = ID & 63L; */
/*   long long b1 = ID & 4032L; */
/*   long long b2 = ID & 258048L; */
/*   long long b3 = ID & 16515072L; */
/*   long long b4 = ID & 1056964608L; */
/*   color_table[0] = (b0 << 1) | (b1 << 3) | (b2 << 5) | (b3 << 7) | (b4 << 9); */
/*   ID = ~ID; */
/*   b0 = ID & 63L; */
/*   b1 = ID & 4032L; */
/*   b2 = ID & 258048L; */
/*   b3 = ID & 16515072L; */
/*   b4 = ID & 1056964608L; */
/*   color_table[1] = (b0 << 1) | (b1 << 3) | (b2 << 5) | (b3 << 7) | (b4 << 9); */
/*   //printf("c1 %ld \n",color_table[0]); */
/*   //printf("c2 %ld \n",color_table[1]); */
/* } */
#endif


__global__ void simulate_all_kernel(int num_attacks, const int * __restrict__ num_patterns21, long long *maxID, float *maxPower, float line, float way, const int * __restrict__ tableID_prefix21, int *tableID_table21, const int * __restrict__ bit_hash_table, const int * __restrict__ reversed_bit_table){

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int gdim = gridDim.x;
  int bdim = blockDim.x;
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  int step = blockDim.x * gridDim.x;
  int color_combo[COMBO_LENGTH];
  int num_drops_combo[COMBO_LENGTH];
  int isLine_combo[COMBO_LENGTH];
  int i,j,k;
  int rank = RANKINGLENGTH;
  float MP[RANKINGLENGTH];
  long long MID[RANKINGLENGTH];
  long long tableID = 0;

  for(i = 0;i < rank;i++){
    MID[i] = 0;
    MP[i] = 0.0;
  }

  int u, l, uu, ll;
  int compared_num[(width*hight-1)/8+1];
  int bit_num[hight];
  for(u = 0;u <= num_attacks;u++){
    l = num_attacks - u;
    if(u <= 21 && l <= 21){
      int uoffset = tableID_prefix21[u];
      int loffset = tableID_prefix21[l];
      for(uu = bid;uu < num_patterns21[u];uu+=gdim){
	long long upperID = (long long)tableID_table21[uu+uoffset];
        for(ll = tid;ll < num_patterns21[l];ll+=bdim){
          long long lowerID = (long long)tableID_table21[ll+loffset];
          tableID = (upperID << 21) | lowerID;

          int count = 0;
          for(i = 0;i < (width*hight-1)/8+1;i++){
            compared_num[i] = (tableID >> 8*i ) & 255;
            count += bit_hash_table[compared_num[i]];
          }
          if(count == num_attacks){
            long long reversed = 0;
            for(i = 0;i < hight; i++){
              //bit_num[i] = (tableID >> (width*i) ) & (REVERSE_LENGTH-1);
              bit_num[i] = (int)((tableID >> (width*i) ) & (REVERSE_LENGTH-1));
              reversed += ((long long)reversed_bit_table[bit_num[i]]) << (width*i);
            }
	    // if(tableID == 1103874885640L) printf("%lld, %lld\n", tableID,reversed);
	    if(tableID <= reversed){
	      //if(tableID == 1103874885640L) printf(" %lld\n", tableID);
	      init_combo_info(color_combo, num_drops_combo, isLine_combo);
	      int combo_counter = 0;
	      long long color_table[NUM_COLORS];
	      int num_c;
	      for(num_c = 0;num_c < NUM_COLORS;num_c++){
		color_table[num_c] = 0;
	      }
	      //tableID = 1103874885640L;
	      //tableID = 42656280L;
	      generate_table(tableID, color_table);
	      int returned_combo_counter = 0;
	      do{
		//if(tableID == 1933312L){
		// if(id == 0){
		//   printf("ID %lld\n",tableID);
		//   print_table(color_table);
		//   print_table2(color_table[0]);
		//   print_table2(color_table[1]);
		// }
		combo_counter = returned_combo_counter;
		//returned_combo_counter = one_step(color_table, color_combo, num_drops_combo, isLine_combo, combo_counter);
		returned_combo_counter = one_step_opt(color_table, color_combo, num_drops_combo, isLine_combo, combo_counter);
		//printf("combo = %d\n", returned_combo_counter);
	      }while(returned_combo_counter != combo_counter);
	      float power = return_attack(combo_counter, color_combo, num_drops_combo, isLine_combo, line, way);
	      // if(id == 0){
	      // 	printf("power = %f\n", power);
	      // 	for(j = 0;j < COMBO_LENGTH;j++)
	      // 	  printf("color_combo = %d, num_d = %d, isL = %d\n", color_combo[j], num_drops_combo[j], isLine_combo[j]);
	      // }
	      //return;
	      if(MP[rank-1] < power){
		for(j = 0;j < rank;j++){
		  if(MP[j] < power){
		    for(k = rank-2;k >= j;k--){
		      MID[k+1] = MID[k];
		      MP[k+1] = MP[k];
		    }
		    MID[j] = tableID;
		    MP[j] = power;
		    break;
		  }
		}
	      }
            }
	  }
        }
      }
    }
  }

  for(i = 0;i < rank;i++){
    maxPower[id + step*i] = MP[i];
    maxID[id + step*i] = MID[i];
  }
}


__global__ void simulate_all_kernel_opt(int num_attacks, const int * __restrict__ num_patterns21, long long *maxID, float *maxPower, float line, float way, const int * __restrict__ tableID_prefix21, int *tableID_table21, const int * __restrict__ reversed_bit_table, const int * __restrict__ bit_hash_table){

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int gdim = gridDim.x;
  int bdim = blockDim.x;
  int color_combo[COMBO_LENGTH];
  int num_drops_combo[COMBO_LENGTH];
  int isLine_combo[COMBO_LENGTH];
  int i,j,k;
  int rank = RANKINGLENGTH;
  float MP[RANKINGLENGTH];
  long long MID[RANKINGLENGTH];
  long long tableID = 0;

  for(i = 0;i < rank;i++){
    MID[i] = 0;
    MP[i] = 0.0;
  }

  int u, l, uu, ll;
  int bit_num[hight];
  for(u = 0;u <= num_attacks;u++){
    l = num_attacks - u;
    if(u <= 21 && l <= 21){
      int uoffset = tableID_prefix21[u];
      int loffset = tableID_prefix21[l];
      for(uu = bid;uu < num_patterns21[u];uu+=gdim){
	long long upperID = (long long)tableID_table21[uu+uoffset];
        for(ll = tid;ll < num_patterns21[l];ll+=bdim){
          long long lowerID = (long long)tableID_table21[ll+loffset];
          tableID = (upperID << 21) | lowerID;

	  long long reversed = 0;
	  for(i = 0;i < hight; i++){
	    //bit_num[i] = (tableID >> (width*i) ) & (REVERSE_LENGTH-1);
	    bit_num[i] = (int)((tableID >> (width*i) ) & (REVERSE_LENGTH-1));
	    reversed += ((long long)reversed_bit_table[bit_num[i]]) << (width*i);
	  }
	  // if(tableID == 1103874885640L) printf("%lld, %lld\n", tableID,reversed);
	  if(tableID <= reversed){
	    //if(tableID == 1103874885640L) printf(" %lld\n", tableID);
	    init_combo_info(color_combo, num_drops_combo, isLine_combo);
	    int combo_counter = 0;
	    long long color_table[NUM_COLORS];
	    int num_c;
	    for(num_c = 0;num_c < NUM_COLORS;num_c++){
	      color_table[num_c] = 0;
	    }
	    //tableID = 1103874885640L;
	    //tableID = 42656280L;
	    generate_table(tableID, color_table);
	    int returned_combo_counter = 0;
	    do{
	      //if(tableID == 1933312L){
// 	      if(blockDim.x * blockIdx.x + threadIdx.x == 0){
// 	        printf("ID %lld\n",tableID);
// 	        print_table(color_table);
// 	        print_table2(color_table[0]);
// 	        print_table2(color_table[1]);
// 	      }
	      combo_counter = returned_combo_counter;
	      //returned_combo_counter = one_step(color_table, color_combo, num_drops_combo, isLine_combo, combo_counter);
	      //returned_combo_counter = one_step_opt2(color_table, color_combo, num_drops_combo, isLine_combo, combo_counter);
	      //returned_combo_counter = one_step_opt3(color_table, color_combo, num_drops_combo, isLine_combo, bit_hash_table, combo_counter);
	      //returned_combo_counter = one_step_opt4(color_table, color_combo, num_drops_combo, isLine_combo, bit_hash_table, combo_counter);
	      returned_combo_counter = one_step_opt5(color_table, color_combo, num_drops_combo, isLine_combo, combo_counter);
	      //returned_combo_counter = one_step_opt_c2(color_table, color_combo, num_drops_combo, isLine_combo, combo_counter);
	      //printf("combo = %d\n", returned_combo_counter);
	    }while(returned_combo_counter != combo_counter);
	    float power = return_attack(combo_counter, color_combo, num_drops_combo, isLine_combo, line, way);
	    // if(id == 0){
	    // 	printf("power = %f\n", power);
	    // 	for(j = 0;j < COMBO_LENGTH;j++)
	    // 	  printf("color_combo = %d, num_d = %d, isL = %d\n", color_combo[j], num_drops_combo[j], isLine_combo[j]);
	    // }
	    //return;
	    if(MP[rank-1] < power){
	      for(j = 0;j < rank;j++){
		if(MP[j] < power){
		  for(k = rank-2;k >= j;k--){
		    MID[k+1] = MID[k];
		    MP[k+1] = MP[k];
		  }
		  MID[j] = tableID;
		  MP[j] = power;
		  break;
		}
	      }
	    }
	  }
        }
      }
    }
  }

  int id = blockDim.x * blockIdx.x + threadIdx.x;
  int step = blockDim.x * gridDim.x;

  for(i = 0;i < rank;i++){
    maxPower[id + step*i] = MP[i];
    maxID[id + step*i] = MID[i];
  }
//   int tid = threadIdx.x;
//   int bid = blockIdx.x;
//   int gdim = gridDim.x;
//   int bdim = blockDim.x;
//   int id = blockDim.x * blockIdx.x + threadIdx.x;
//   int step = blockDim.x * gridDim.x;
//   int color_combo[COMBO_LENGTH];
//   int num_drops_combo[COMBO_LENGTH];
//   int isLine_combo[COMBO_LENGTH];
//   int i,j,k;
//   int rank = RANKINGLENGTH;
//   float MP[RANKINGLENGTH];
//   long long MID[RANKINGLENGTH];
//   long long tableID = 0;

//   for(i = 0;i < rank;i++){
//     MID[i] = 0;
//     MP[i] = 0.0;
//   }

//   int u, l, uu, ll;
//   int bit_num[hight];
//   for(u = 0;u <= num_attacks;u++){
//     l = num_attacks - u;
//     if(u <= 21 && l <= 21){
//       int uoffset = tableID_prefix21[u];
//       int loffset = tableID_prefix21[l];
//       for(uu = bid;uu < num_patterns21[u];uu+=gdim){
// 	long long upperID = (long long)tableID_table21[uu+uoffset];
//         //for(ll = tid;ll < num_patterns21[l];ll+=bdim){
//         for(ll = tid;ll < num_patterns21[l];){
// 	  long long reversed;
// 	  do{
// 	    long long lowerID = (long long)tableID_table21[ll+loffset];
// 	    tableID = (upperID << 21) | lowerID;
// 	    reversed = 0;
// 	    for(i = 0;i < hight; i++){
// 	      //bit_num[i] = (tableID >> (width*i) ) & (REVERSE_LENGTH-1);
// 	      bit_num[i] = (int)((tableID >> (width*i) ) & (REVERSE_LENGTH-1));
// 	      reversed += ((long long)reversed_bit_table[bit_num[i]]) << (width*i);
// 	    }
// 	    ll+=bdim;
// 	  }while(tableID > reversed && ll < num_patterns21[l]);
// 	  // if(tableID == 1103874885640L) printf("%lld, %lld\n", tableID,reversed);
// 	  //if(tableID == 1103874885640L) printf(" %lld\n", tableID);
// 	  init_combo_info(color_combo, num_drops_combo, isLine_combo);
// 	  int combo_counter = 0;
// 	  long long color_table[NUM_COLORS];
// 	  int num_c;
// 	  for(num_c = 0;num_c < NUM_COLORS;num_c++){
// 	    color_table[num_c] = 0;
// 	  }
// 	  //tableID = 1103874885640L;
// 	  //tableID = 42656280L;
// 	  generate_table(tableID, color_table);
// 	  int returned_combo_counter = 0;
// 	  do{
// 	    //if(tableID == 1933312L){
// 	    // if(id == 0){
// 	    //   printf("ID %lld\n",tableID);
// 	    //   print_table(color_table);
// 	    //   print_table2(color_table[0]);
// 	    //   print_table2(color_table[1]);
// 	    // }
// 	    combo_counter = returned_combo_counter;
// 	    //returned_combo_counter = one_step(color_table, color_combo, num_drops_combo, isLine_combo, combo_counter);
// 	    returned_combo_counter = one_step_opt(color_table, color_combo, num_drops_combo, isLine_combo, combo_counter);
// 	    //printf("combo = %d\n", returned_combo_counter);
// 	  }while(returned_combo_counter != combo_counter);
// 	  float power = return_attack(combo_counter, color_combo, num_drops_combo, isLine_combo, line, way);
// 	  // if(id == 0){
// 	  // 	printf("power = %f\n", power);
// 	  // 	for(j = 0;j < COMBO_LENGTH;j++)
// 	  // 	  printf("color_combo = %d, num_d = %d, isL = %d\n", color_combo[j], num_drops_combo[j], isLine_combo[j]);
// 	  // }
// 	  //return;
// 	  if(MP[rank-1] < power){
// 	    for(j = 0;j < rank;j++){
// 	      if(MP[j] < power){
// 		for(k = rank-2;k >= j;k--){
// 		  MID[k+1] = MID[k];
// 		  MP[k+1] = MP[k];
// 		}
// 		MID[j] = tableID;
// 		MP[j] = power;
// 		break;
// 	      }
// 	    }
// 	  }
// 	}
//       }
//     }
//   }

//   for(i = 0;i < rank;i++){
//     maxPower[id + step*i] = MP[i];
//     maxID[id + step*i] = MID[i];
//   }
}

void simulate_all(int num_attacks, int *bit_hash_table_dev, int *reversed_bit_table_dev, int *tableID_table21_dev, int *tableID_prefix21_dev, int *num_patterns21_dev){

  long long tableID = 0;
  int rank = RANKINGLENGTH * 10;
  int i, j, k;
  long long *max_powerID_dev;
  float *max_power_dev;
  int tsize = NUM_THREAD;
  //int gsize = ((num_patterns_omitted[num_attacks]-1)/128+1);
  int gsize = NUM_BLOCK;
  long long max_powerID[tsize*gsize*RANKINGLENGTH];
  float max_power[tsize*gsize*RANKINGLENGTH];

  CUDA_SAFE_CALL(cudaMalloc((void**)&max_powerID_dev, sizeof(long long) * tsize * gsize * RANKINGLENGTH));;
  CUDA_SAFE_CALL(cudaMalloc((void**)&max_power_dev,   sizeof(float)    * tsize * gsize * RANKINGLENGTH));;

  dim3 grid(gsize,1,1);
  dim3 block(tsize,1,1);
  cudaDeviceSynchronize();
  double t1 = gettimeofday_sec();
  //simulate_all_kernel<<<grid, block>>>(num_attacks, num_patterns21_dev, max_powerID_dev, max_power_dev, line, way, tableID_prefix21_dev, tableID_table21_dev, bit_hash_table_dev, reversed_bit_table_dev);
  simulate_all_kernel_opt<<<grid, block>>>(num_attacks, num_patterns21_dev, max_powerID_dev, max_power_dev, line, way, tableID_prefix21_dev, tableID_table21_dev, reversed_bit_table_dev, bit_hash_table_dev);
  cudaDeviceSynchronize();
  double t2 = gettimeofday_sec();
  printf("num %d,time,%f\n",num_attacks,t2-t1);
  //return;
  cudaMemcpy(max_powerID, max_powerID_dev, sizeof(long long) * tsize * gsize * RANKINGLENGTH, cudaMemcpyDeviceToHost);
  cudaMemcpy(max_power  , max_power_dev  , sizeof(float)       * tsize * gsize * RANKINGLENGTH, cudaMemcpyDeviceToHost);

  float    MP [rank];
  long long MID[rank];
  for(i = 0;i < rank;i++){
    MP[i] = 0.0;
    MID[i]= 0;
  }

  for(i = 0;i < gsize*tsize*RANKINGLENGTH;i++){
    float power = max_power[i];
    tableID = max_powerID[i];
    //printf("ID %15u, power %f\n",tableID, power);
    if(MP[rank-1] < power){
      for(j = 0;j < rank;j++){
	if(MP[j] < power){
	  for(k = rank-2;k >= j;k--){
	    MID[k+1] = MID[k];
	    MP[k+1] = MP[k];
	  }
	  MID[j] = tableID;
	  MP[j] = power;
	  // for(k = 0;k <= 10;k++){
	  //   printf("MID[%d] = %lld\n",k,MID[k]);
	  // }
	  break;
	}
      }
    }
  }
  for(i = 0;i < rank;i++){
    float power = MP[i];
    long long tmp = MID[i];
    long long minID = tmp;
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
    printf("%d,max ID,%lld,power,%f\n",i,MID[i],MP[i]);
  }
  CUDA_SAFE_CALL(cudaFree(max_powerID_dev));
  CUDA_SAFE_CALL(cudaFree(max_power_dev));

}


#if NUM_COLORS==2
__device__ inline int one_step(long long color_table[NUM_COLORS], int color_combo[COMBO_LENGTH], int num_drops_combo[COMBO_LENGTH], int isLine_combo[COMBO_LENGTH], int finish){
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
  
  //long long isErase_table[NUM_COLORS];
  int combo_counter = finish;
  /* int line_flag[hight]; */
  int i,j,ii,jj;
  int num_c;

  for(num_c = 0;num_c < NUM_COLORS;num_c++){
    
    //isErase_table[num_c] = 0;
    long long isErase_table = 0;
    long long color = color_table[num_c];

    long long f1, f2, f3, f4, f5;
    f1 = 14L;
    // 011100000
    // 000000000
    // 000000000
    // 000000000
    // 000000000
    // 000000000
    // 000000000
    // 000000000
    f2 = 28L;
    // 001110000
    // 000000000
    // 000000000
    // 000000000
    // 000000000
    // 000000000
    // 000000000
    // 000000000
    f3 = 56L;
    // 000111000
    // 000000000
    // 000000000
    // 000000000
    // 000000000
    // 000000000
    // 000000000
    // 000000000
    f4 = 112L;
    // 000011100
    // 000000000
    // 000000000
    // 000000000
    // 000000000
    // 000000000
    // 000000000
    // 000000000
    f5 = 224L;
    // 000001110
    // 000000000
    // 000000000
    // 000000000
    // 000000000
    // 000000000
    // 000000000
    // 000000000
    for(i = 1;i <= hight;i++){
      f1 = f1 << WID;
      f2 = f2 << WID;
      f3 = f3 << WID;
      f4 = f4 << WID;
      f5 = f5 << WID;
      isErase_table = (color & f1) == f1 ? isErase_table | f1 : isErase_table;
      isErase_table = (color & f2) == f2 ? isErase_table | f2 : isErase_table;
      isErase_table = (color & f3) == f3 ? isErase_table | f3 : isErase_table;
      isErase_table = (color & f4) == f4 ? isErase_table | f4 : isErase_table;
      isErase_table = (color & f5) == f5 ? isErase_table | f5 : isErase_table;
    }

    f1 = 134480384L;
    // 000000000
    // 100000000
    // 100000000
    // 100000000
    // 000000000
    // 000000000
    // 000000000
    // 000000000
    f2 = 68853956608L;
    // 000000000
    // 000000000
    // 100000000
    // 100000000
    // 100000000
    // 000000000
    // 000000000
    // 000000000
    f3 = 35253225783296L;
    // 000000000
    // 000000000
    // 000000000
    // 100000000
    // 100000000
    // 100000000
    // 000000000
    // 000000000
    f4 = 18049651601047552L;
    // 000000000
    // 000000000
    // 000000000
    // 000000000
    // 100000000
    // 100000000
    // 100000000
    // 000000000
    for(j = 1;j <= width;j++){
      f1 = f1 << 1;
      f2 = f2 << 1;
      f3 = f3 << 1;
      f4 = f4 << 1;
      isErase_table = (color & f1) == f1 ? isErase_table | f1 : isErase_table;
      isErase_table = (color & f2) == f2 ? isErase_table | f2 : isErase_table;
      isErase_table = (color & f3) == f3 ? isErase_table | f3 : isErase_table;
      isErase_table = (color & f4) == f4 ? isErase_table | f4 : isErase_table;
    }

    //print_table2(isErase_table[num_c]);
    
    /// count drops ///
    // filter 527874L
    // 2進数 (左上が1桁目)
    // 010000000
    // 111000000
    // 010000000
    // 000000000
    // 000000000
    // 000000000
    // 000000000
    // 000000000
    // 上の左シフトと比較
    // tmpとisEraseと上で&してtmpに|すれば良い? ← ちょっと違う
#if 0
    for(i = 1;i <= hight;i++){
      for(j = 1;j <= width;j++){
	long long tmp = (1L << (WID*i+j));
	//if(isErase_table[num_c] & tmp){
	if(isErase_table & tmp){
	  color_combo[combo_counter] = num_c;
	  long long tmp_old;
	  do{
	    long long filter = 527874L << (WID*(i-1));
	    long long p = (1L << (WID*i+1));
	    tmp_old = tmp;
	    for(ii = i;ii <= hight;ii++,filter = filter << 2,p = p << 2){
	      for(jj = 1;jj <= width;jj++,filter = filter << 1,p = p << 1){
		//if(isErase_table[num_c] & p){
		if(isErase_table & p){
		  if(tmp & filter){
		    tmp = tmp | p;
		    num_drops_combo[combo_counter]++;
		    color = color & (~p);
		    //isErase_table[num_c] = isErase_table[num_c] & (~p);
		    isErase_table = isErase_table & (~p);
		  }
		}
		//printf("%d, %d \n", ii, jj);
		//print_table2(filter);
	      }
	    }
	  }while(tmp_old != tmp);
	  //print_table2(tmp);
	  isLine_combo[combo_counter] = ((tmp >> (WID  +1)) & 127) == 127
	    || ((tmp >> (WID*2+1)) & 127) == 127
	    || ((tmp >> (WID*3+1)) & 127) == 127
	    || ((tmp >> (WID*4+1)) & 127) == 127
	    || ((tmp >> (WID*5+1)) & 127) == 127
	    || ((tmp >> (WID*6+1)) & 127) == 127;
	  combo_counter++;
	}
      }
    }
#else
    long long tmp;
    int flag;
    do {
      flag = 0;
      for(i = 1;i <= hight;i++){
	for(j = 1;j <= width;j++){
	  tmp = (1L << (WID*i+j));
	  if(isErase_table & tmp){
	    flag = 1;
	    break;
	  }
	}
	if(flag) break;
      }
      if(flag){
	color_combo[combo_counter] = num_c;
	long long tmp_old;
	do{
	  long long filter = 527874L << (WID*(i-1));
	  long long p = (1L << (WID*i+1));
	  tmp_old = tmp;
	  for(ii = i;ii <= hight;ii++,filter = filter << 2,p = p << 2){
	    for(jj = 1;jj <= width;jj++,filter = filter << 1,p = p << 1){
	      if(isErase_table & p){
		if(tmp & filter){
		  tmp = tmp | p;
		  num_drops_combo[combo_counter]++;
		  color = color & (~p);
		  isErase_table = isErase_table & (~p);
		}
	      }
	    }
	  }
	}while(tmp_old != tmp);
	isLine_combo[combo_counter] = ((tmp >> (WID  +1)) & 127) == 127
	  || ((tmp >> (WID*2+1)) & 127) == 127
	  || ((tmp >> (WID*3+1)) & 127) == 127
	  || ((tmp >> (WID*4+1)) & 127) == 127
	  || ((tmp >> (WID*5+1)) & 127) == 127
	  || ((tmp >> (WID*6+1)) & 127) == 127;
	combo_counter++;
      }
    }while(flag);
#endif

    color_table[num_c] = color;
  }
  
  /// drop event ///
  // 真理値表から考えると
  // 上の行 | 下の行 || 解上  | 解下     
  //     0  |  0     || 0     | 0  
  //     0  |  1     || 0     | 1  
  //     1  |  0     || 0     | 1  
  //     1  |  1     || 1     | 1
  // 解上 = 上の行 & 下の行
  // 解下 = 上の行 | 下の行
  // 上の行をそれぞれの色のtableにして、
  // 下の行を複合した存在するかどうかのtableにする
  // 上の行 color_u = color_u & exist_d
  // 下の行 color_d += (exist_d | color_u) ^ exist_d
  // exist_u = exist_u & exist_d
  // exist_d = exist_u | exist_d

  //print_table(color_table);
  long long exist_table = color_table[0];
  for(num_c = 1;num_c < NUM_COLORS;num_c++){
    exist_table = exist_table | color_table[num_c];
  }

  long long exist_org;
  int upper = 2;
  do{
    exist_org = exist_table;
    for(i = hight;i >= upper;i--){
      long long exist_u = exist_table & (254L << (WID*(i-1)));
      long long exist_d = exist_table & (254L << (WID*(i)));
      //print_table2(exist_u);
      //print_table2(exist_d);
      for(num_c = 0;num_c < NUM_COLORS;num_c++){
	long long color = color_table[num_c];
	long long color_u = color & (254L << (WID*(i-1)));
	color = (color & ~(254L << (WID*(i-1)))) | (color_u & (exist_d >> WID));
	color_table[num_c] = color | ((exist_d | (color_u << WID)) ^ exist_d) ;
      }
      exist_table = (exist_table & ~(254L << (WID*(i-1)))) | (exist_u & (exist_d >> WID));
      exist_table = (exist_table & ~(254L << (WID*(i)))) | ((exist_u << WID) | exist_d);
      //print_table2(exist_table);
    }
    upper++;
  }while(exist_org != exist_table);
  // if(threadIdx.x == 0 && blockIdx.x == 0)
  //   print_table(color_table);
  //print_table2(exist_table);

  /* isLine_combo[combo_counter] = (isErase_table[num_c] >> (WID  +1)) & 63 == 63 */
  /*   || (isErase_table[num_c] >> (WID*2+1)) & 63 == 63 */
  /*   || (isErase_table[num_c] >> (WID*3+1)) & 63 == 63 */
  /*   || (isErase_table[num_c] >> (WID*4+1)) & 63 == 63 */
  /*   || (isErase_table[num_c] >> (WID*5+1)) & 63 == 63;  */
  //printf("combo_counter %d\n", combo_counter);

  return combo_counter;
}

#if 0
__device__ inline int one_step_opt(long long color_table[NUM_COLORS], int color_combo[COMBO_LENGTH], int num_drops_combo[COMBO_LENGTH], int isLine_combo[COMBO_LENGTH], int finish){
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
  
  //long long isErase_table[NUM_COLORS];
  int combo_counter = finish;
  /* int line_flag[hight]; */
  int i,j,ii,jj;
  int num_c;

  for(num_c = 0;num_c < NUM_COLORS;num_c++){
    
    //isErase_table[num_c] = 0;
    long long isErase_table = 0;
    long long color = color_table[num_c];

    //自身の上下左右シフトと比較
    long long n, w, s, e;
    n = color >> WID;
    w = color >> 1;
    s = color << WID;
    e = color << 1;
    long long tmp, tmp2;
    tmp  = (color & n & s);
    tmp  = tmp  | (tmp  >> WID) | (tmp  << WID);
    tmp2 = (color & w & e);
    tmp2 = tmp2 | (tmp2 >> 1  ) | (tmp2 << 1  );
    isErase_table = (color & tmp) | (color & tmp2);
    color = color & (~isErase_table);

    // if(threadIdx.x == 0 && blockIdx.x == 0)
    //   print_table2(isErase_table);
    
    /// count drops ///
    // filter 527874L
    // 2進数 (左上が1桁目)
    // 010000000
    // 111000000
    // 010000000
    // 000000000
    // 000000000
    // 000000000
    // 000000000
    // 000000000
    // 上の左シフトと比較
    // tmpとisEraseと上で&してtmpに|すれば良い? ← ちょっと違う
    int flag;
    do {
      flag = 0;
//       for(i = 1;i <= hight;i++){
// 	for(j = 1;j <= width;j++){
// 	  tmp = (1L << (WID*i+j));
// 	  if(isErase_table & tmp){
// 	    flag = 1;
// 	    break;
// 	  }
// 	}
// 	if(flag) break;
//       }
      tmp = 1L << (WID+1);
      for(i = 1;i <= hight;i++, tmp = tmp << 2){
	for(j = 1;j <= width;j++, tmp = tmp << 1){
	  if(isErase_table & tmp){
	    flag = 1;
	    break;
	  }
	}
	if(flag) break;
      }
      if(flag){
	color_combo[combo_counter] = num_c;
	long long tmp_old;
	do{
	  tmp_old = tmp;
#if 1
	  long long p = (1L << (WID*i+1));
	  long long filter = 527874L << (WID*(i-1));
	  //#pragma unroll
	  for(ii = i;ii <= hight;ii++,filter = filter << 2,p = p << 2){
	    //for(ii = i-1;ii < hight;ii++,p = p << 2){
	    //#pragma unroll
	    for(jj = 1;jj <= width;jj++,filter = filter << 1,p = p << 1){
	      //for(jj = 0;jj < width;jj++,p = p << 1){
	      if(isErase_table & p){
		//filter = 527874L << (WID*(ii)+(jj));
		if(tmp & filter){
		  tmp = tmp | p;
		  num_drops_combo[combo_counter]++;
		  //color = color & (~p);
		  isErase_table = isErase_table & (~p);
		}
	      }
	    }
	  }
#else
	  ii = i;
	  jj = j;
	  int flag2;
	  long long p = (1L << (WID*i+j));
	  long long filter = 527874L << (WID*(i-1)+(j-1));
	  do{
	    flag2 = 0;
	    for(;ii <= hight;ii++,filter = filter << 2,p = p << 2){
	      for(;jj <= width;jj++,filter = filter << 1,p = p << 1){
		if(isErase_table & p){
		  if(tmp & filter){
		    flag2 = 1;
		    break;
		  }
		}
	      }
	      if(flag2) break;
	      jj = 1;
	    }
	    if(flag2){
	      tmp = tmp | p;
	      num_drops_combo[combo_counter]++;
	      isErase_table = isErase_table & (~p);
	    }
	  }while(ii <= hight);
#endif

	}while(tmp_old != tmp);
	isLine_combo[combo_counter] = ((tmp >> (WID  +1)) & 127) == 127
	  || ((tmp >> (WID*2+1)) & 127) == 127
	  || ((tmp >> (WID*3+1)) & 127) == 127
	  || ((tmp >> (WID*4+1)) & 127) == 127
	  || ((tmp >> (WID*5+1)) & 127) == 127
	  || ((tmp >> (WID*6+1)) & 127) == 127;
	combo_counter++;
      }
    }while(flag);

    color_table[num_c] = color;
  }
  
  /// drop event ///
  // 真理値表から考えると
  // 上の行 | 下の行 || 解上  | 解下     
  //     0  |  0     || 0     | 0  
  //     0  |  1     || 0     | 1  
  //     1  |  0     || 0     | 1  
  //     1  |  1     || 1     | 1
  // 解上 = 上の行 & 下の行
  // 解下 = 上の行 | 下の行
  // 上の行をそれぞれの色のtableにして、
  // 下の行を複合した存在するかどうかのtableにする
  // 上の行 color_u = color_u & exist_d
  // 下の行 color_d += (exist_d | color_u) ^ exist_d
  // exist_u = exist_u & exist_d
  // exist_d = exist_u | exist_d

  //print_table(color_table);
  long long exist_table = color_table[0];
  for(num_c = 1;num_c < NUM_COLORS;num_c++){
    exist_table = exist_table | color_table[num_c];
  }

  long long exist_org;
  int upper = 2;
//   do{
//     exist_org = exist_table;
//     for(i = hight;i >= upper;i--){
//       long long exist_u = exist_table & (254L << (WID*(i-1)));
//       long long exist_d = exist_table & (254L << (WID*(i)));
//       //print_table2(exist_u);
//       //print_table2(exist_d);
//       for(num_c = 0;num_c < NUM_COLORS;num_c++){
// 	long long color = color_table[num_c];
// 	long long color_u = color & (254L << (WID*(i-1)));
// 	color = (color & ~(254L << (WID*(i-1)))) | (color_u & (exist_d >> WID));
// 	color_table[num_c] = color | ((exist_d | (color_u << WID)) ^ exist_d) ;
//       }
//       exist_table = (exist_table & ~(254L << (WID*(i-1)))) | (exist_u & (exist_d >> WID));
//       exist_table = (exist_table & ~(254L << (WID*(i)))) | ((exist_u << WID) | exist_d);
//       //print_table2(exist_table);
//     }
//     upper++;
//   }while(exist_org != exist_table);

  long long up, dp;
  do{
    exist_org = exist_table;
    up = (254L << (WID*(hight-1)));
    dp = (254L << (WID*(hight  )));
    for(i = hight;i >= upper;i--,up = up >> WID, dp = dp >> WID){
      long long exist_u = exist_table & up;
      long long exist_d = exist_table & dp;
      //print_table2(exist_u);
      //print_table2(exist_d);
      for(num_c = 0;num_c < NUM_COLORS;num_c++){
	long long color = color_table[num_c];
	long long color_u = color & up;
	color = (color & (~up)) | (color_u & (exist_d >> WID));
	color_table[num_c] = color | ((exist_d | (color_u << WID)) ^ exist_d) ;
      }
      exist_table = (exist_table & (~up)) | (exist_u & (exist_d >> WID));
      exist_table = (exist_table & (~dp)) | ((exist_u << WID) | exist_d);
      //print_table2(exist_table);
    }
    upper++;
  }while(exist_org != exist_table);


  // if(threadIdx.x == 0 && blockIdx.x == 0)
  //   print_table(color_table);
  //print_table2(exist_table);

  /* isLine_combo[combo_counter] = (isErase_table[num_c] >> (WID  +1)) & 63 == 63 */
  /*   || (isErase_table[num_c] >> (WID*2+1)) & 63 == 63 */
  /*   || (isErase_table[num_c] >> (WID*3+1)) & 63 == 63 */
  /*   || (isErase_table[num_c] >> (WID*4+1)) & 63 == 63 */
  /*   || (isErase_table[num_c] >> (WID*5+1)) & 63 == 63;  */
  //printf("combo_counter %d\n", combo_counter);

  return combo_counter;
}

#else
__device__ inline int one_step_opt(long long color_table[NUM_COLORS], int color_combo[COMBO_LENGTH], int num_drops_combo[COMBO_LENGTH], int isLine_combo[COMBO_LENGTH], int finish){
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
  
  long long isErase_tables[NUM_COLORS];
  int combo_counter = finish;
  /* int line_flag[hight]; */
  int i,j,ii,jj;
  int num_c;
  long long tmp, tmp2;

  for(num_c = 0;num_c < NUM_COLORS;num_c++){
    
    long long color = color_table[num_c];

    //自身の上下左右シフトと比較
    long long n, w, s, e;
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

#if NUM_COLORS==2
  if(isErase_tables[0] == isErase_tables[1])
    return combo_counter;

#endif
  for(num_c = 0;num_c < NUM_COLORS;num_c++){
    long long isErase_table = isErase_tables[num_c];
    color_table[num_c] = color_table[num_c] & (~isErase_table);

    // if(threadIdx.x == 0 && blockIdx.x == 0)
    //   print_table2(isErase_table);
    
    /// count drops ///
    // filter 527874L
    // 2進数 (左上が1桁目)
    // 010000000
    // 111000000
    // 010000000
    // 000000000
    // 000000000
    // 000000000
    // 000000000
    // 000000000
    // 上の左シフトと比較
    // tmpとisEraseと上で&してtmpに|すれば良い? ← ちょっと違う
    int flag;
    do {
      flag = 0;
//       for(i = 1;i <= hight;i++){
// 	for(j = 1;j <= width;j++){
// 	  tmp = (1L << (WID*i+j));
// 	  if(isErase_table & tmp){
// 	    flag = 1;
// 	    break;
// 	  }
// 	}
// 	if(flag) break;
//       }
      tmp = 1L << (WID+1);
      for(i = 1;i <= hight;i++, tmp = tmp << 2){
	for(j = 1;j <= width;j++, tmp = tmp << 1){
	  if(isErase_table & tmp){
	    flag = 1;
	    break;
	  }
	}
	if(flag) break;
      }
      if(flag){
	color_combo[combo_counter] = num_c;
	long long tmp_old;
	do{
	  tmp_old = tmp;
	  long long p = (1L << (WID*i+1));
	  long long filter = 527874L << (WID*(i-1));
	  //#pragma unroll
	  for(ii = i;ii <= hight;ii++,filter = filter << 2,p = p << 2){
	    //for(ii = i-1;ii < hight;ii++,p = p << 2){
	    //#pragma unroll
	    for(jj = 1;jj <= width;jj++,filter = filter << 1,p = p << 1){
	      //for(jj = 0;jj < width;jj++,p = p << 1){
	      if(isErase_table & p){
		//filter = 527874L << (WID*(ii)+(jj));
		if(tmp & filter){
		  tmp = tmp | p;
		  num_drops_combo[combo_counter]++;
		  //color = color & (~p);
		  isErase_table = isErase_table & (~p);
		}
	      }
	    }
	  }

	}while(tmp_old != tmp);
	isLine_combo[combo_counter] = ((tmp >> (WID  +1)) & 127) == 127
	  || ((tmp >> (WID*2+1)) & 127) == 127
	  || ((tmp >> (WID*3+1)) & 127) == 127
	  || ((tmp >> (WID*4+1)) & 127) == 127
	  || ((tmp >> (WID*5+1)) & 127) == 127
	  || ((tmp >> (WID*6+1)) & 127) == 127;
	combo_counter++;
      }
    }while(flag);

  }
  
  /// drop event ///
  // 真理値表から考えると
  // 上の行 | 下の行 || 解上  | 解下     
  //     0  |  0     || 0     | 0  
  //     0  |  1     || 0     | 1  
  //     1  |  0     || 0     | 1  
  //     1  |  1     || 1     | 1
  // 解上 = 上の行 & 下の行
  // 解下 = 上の行 | 下の行
  // 上の行をそれぞれの色のtableにして、
  // 下の行を複合した存在するかどうかのtableにする
  // 上の行 color_u = color_u & exist_d
  // 下の行 color_d += (exist_d | color_u) ^ exist_d
  // exist_u = exist_u & exist_d
  // exist_d = exist_u | exist_d

  //print_table(color_table);
  long long exist_table = color_table[0];
  for(num_c = 1;num_c < NUM_COLORS;num_c++){
    exist_table = exist_table | color_table[num_c];
  }

  long long exist_org;
  int upper = 2;
//   do{
//     exist_org = exist_table;
//     for(i = hight;i >= upper;i--){
//       long long exist_u = exist_table & (254L << (WID*(i-1)));
//       long long exist_d = exist_table & (254L << (WID*(i)));
//       //print_table2(exist_u);
//       //print_table2(exist_d);
//       for(num_c = 0;num_c < NUM_COLORS;num_c++){
// 	long long color = color_table[num_c];
// 	long long color_u = color & (254L << (WID*(i-1)));
// 	color = (color & ~(254L << (WID*(i-1)))) | (color_u & (exist_d >> WID));
// 	color_table[num_c] = color | ((exist_d | (color_u << WID)) ^ exist_d) ;
//       }
//       exist_table = (exist_table & ~(254L << (WID*(i-1)))) | (exist_u & (exist_d >> WID));
//       exist_table = (exist_table & ~(254L << (WID*(i)))) | ((exist_u << WID) | exist_d);
//       //print_table2(exist_table);
//     }
//     upper++;
//   }while(exist_org != exist_table);

  long long up, dp;
  do{
    exist_org = exist_table;
    up = (254L << (WID*(hight-1)));
    dp = (254L << (WID*(hight  )));
    for(i = hight;i >= upper;i--,up = up >> WID, dp = dp >> WID){
      long long exist_u = exist_table & up;
      long long exist_d = exist_table & dp;
      //print_table2(exist_u);
      //print_table2(exist_d);
      for(num_c = 0;num_c < NUM_COLORS;num_c++){
	long long color = color_table[num_c];
	long long color_u = color & up;
	color = (color & (~up)) | (color_u & (exist_d >> WID));
	color_table[num_c] = color | ((exist_d | (color_u << WID)) ^ exist_d) ;
      }
      exist_table = (exist_table & (~up)) | (exist_u & (exist_d >> WID));
      exist_table = (exist_table & (~dp)) | ((exist_u << WID) | exist_d);
      //print_table2(exist_table);
    }
    upper++;
  }while(exist_org != exist_table);


  // if(threadIdx.x == 0 && blockIdx.x == 0)
  //   print_table(color_table);
  //print_table2(exist_table);

  /* isLine_combo[combo_counter] = (isErase_table[num_c] >> (WID  +1)) & 63 == 63 */
  /*   || (isErase_table[num_c] >> (WID*2+1)) & 63 == 63 */
  /*   || (isErase_table[num_c] >> (WID*3+1)) & 63 == 63 */
  /*   || (isErase_table[num_c] >> (WID*4+1)) & 63 == 63 */
  /*   || (isErase_table[num_c] >> (WID*5+1)) & 63 == 63;  */
  //printf("combo_counter %d\n", combo_counter);

  return combo_counter;
}


__device__ inline int one_step_opt2(long long color_table[NUM_COLORS], int color_combo[COMBO_LENGTH], int num_drops_combo[COMBO_LENGTH], int isLine_combo[COMBO_LENGTH], int finish){
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
  
  long long isErase_tables[NUM_COLORS];
  int combo_counter = finish;
  /* int line_flag[hight]; */
  int i,j,ii,jj;
  int num_c;
  long long tmp, tmp2;

  for(num_c = 0;num_c < NUM_COLORS;num_c++){
    
    long long color = color_table[num_c];

    //自身の上下左右シフトと比較
    long long n, w, s, e;
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

#if NUM_COLORS==2
  if(isErase_tables[0] == isErase_tables[1])
    return combo_counter;

#endif
  for(num_c = 0;num_c < NUM_COLORS;num_c++){
    long long isErase_table = isErase_tables[num_c];
    color_table[num_c] = color_table[num_c] & (~isErase_table);

    // if(threadIdx.x == 0 && blockIdx.x == 0)
    //   print_table2(isErase_table);
    
    /// count drops ///
    // filter 527874L
    // 2進数 (左上が1桁目)
    // 010000000
    // 111000000
    // 010000000
    // 000000000
    // 000000000
    // 000000000
    // 000000000
    // 000000000
    // 上の左シフトと比較
    // tmpとisEraseと上で&してtmpに|すれば良い? ← ちょっと違う
    int flag;
    do {
      flag = 0;
//       for(i = 1;i <= hight;i++){
// 	for(j = 1;j <= width;j++){
// 	  tmp = (1L << (WID*i+j));
// 	  if(isErase_table & tmp){
// 	    flag = 1;
// 	    break;
// 	  }
// 	}
// 	if(flag) break;
//       }
      tmp = 1L << (WID+1);
      for(i = 1;i <= hight;i++, tmp = tmp << 2){
	for(j = 1;j <= width;j++, tmp = tmp << 1){
	  if(isErase_table & tmp){
	    flag = 1;
	    break;
	  }
	}
	if(flag) break;
      }
      if(flag){
	color_combo[combo_counter] = num_c;
	long long tmp_old;
	do{
	  tmp_old = tmp;
	  long long p = (1L << (WID*i+1));
	  long long filter = 527874L << (WID*(i-1));
	  //#pragma unroll
	  for(ii = i;ii <= hight;ii++,filter = filter << 2,p = p << 2){
	    //for(ii = i-1;ii < hight;ii++,p = p << 2){
	    //#pragma unroll
	    for(jj = 1;jj <= width;jj++,filter = filter << 1,p = p << 1){
	      //for(jj = 0;jj < width;jj++,p = p << 1){
	      if(isErase_table & p){
		//filter = 527874L << (WID*(ii)+(jj));
		if(tmp & filter){
		  tmp = tmp | p;
		  num_drops_combo[combo_counter]++;
		  //color = color & (~p);
		  isErase_table = isErase_table & (~p);
		}
	      }
	    }
	  }

	}while(tmp_old != tmp);
	isLine_combo[combo_counter] = ((tmp >> (WID  +1)) & 127) == 127
	  || ((tmp >> (WID*2+1)) & 127) == 127
	  || ((tmp >> (WID*3+1)) & 127) == 127
	  || ((tmp >> (WID*4+1)) & 127) == 127
	  || ((tmp >> (WID*5+1)) & 127) == 127
	  || ((tmp >> (WID*6+1)) & 127) == 127;
	combo_counter++;
      }
    }while(flag);

  }
  
  /// drop event ///
  // 真理値表から考えると
  // 上の行 | 下の行 || 解上  | 解下     
  //     0  |  0     || 0     | 0  
  //     0  |  1     || 0     | 1  
  //     1  |  0     || 0     | 1  
  //     1  |  1     || 1     | 1
  // 解上 = 上の行 & 下の行
  // 解下 = 上の行 | 下の行
  // 上の行をそれぞれの色のtableにして、
  // 下の行を複合した存在するかどうかのtableにする
  // 上の行 color_u = color_u & exist_d
  // 下の行 color_d += (exist_d | color_u) ^ exist_d
  // exist_u = exist_u & exist_d
  // exist_d = exist_u | exist_d

  //print_table(color_table);
  long long exist_table = color_table[0];
  for(num_c = 1;num_c < NUM_COLORS;num_c++){
    exist_table = exist_table | color_table[num_c];
  }

  long long exist_org;
  //int upper = 2;

//   do{
//     exist_org = exist_table;
//     for(i = hight;i >= upper;i--){
//       long long exist_u = exist_table & (254L << (WID*(i-1)));
//       long long exist_d = exist_table & (254L << (WID*(i)));
//       //print_table2(exist_u);
//       //print_table2(exist_d);
//       for(num_c = 0;num_c < NUM_COLORS;num_c++){
// 	long long color = color_table[num_c];
// 	long long color_u = color & (254L << (WID*(i-1)));
// 	color = (color & ~(254L << (WID*(i-1)))) | (color_u & (exist_d >> WID));
// 	color_table[num_c] = color | ((exist_d | (color_u << WID)) ^ exist_d) ;
//       }
//       exist_table = (exist_table & ~(254L << (WID*(i-1)))) | (exist_u & (exist_d >> WID));
//       exist_table = (exist_table & ~(254L << (WID*(i)))) | ((exist_u << WID) | exist_d);
//       //print_table2(exist_table);
//     }
//     upper++;
//   }while(exist_org != exist_table);

  do{
    exist_org = exist_table;
//     up = (254L << (WID*(hight-1)));
//     dp = (254L << (WID*(hight  )));
//     for(i = hight;i >= upper;i--,up = up >> WID, dp = dp >> WID){
//       long long exist_u = exist_table & up;
//       long long exist_d = exist_table & dp;
//       //print_table2(exist_u);
//       //print_table2(exist_d);
//       for(num_c = 0;num_c < NUM_COLORS;num_c++){
// 	long long color = color_table[num_c];
// 	long long color_u = color & up;
// 	color = (color & (~up)) | (color_u & (exist_d >> WID));
// 	color_table[num_c] = color | ((exist_d | (color_u << WID)) ^ exist_d) ;
//       }
//       exist_table = (exist_table & (~up)) | (exist_u & (exist_d >> WID));
//       exist_table = (exist_table & (~dp)) | ((exist_u << WID) | exist_d);
//       //print_table2(exist_table);
//     }
//     upper++;

    long long exist_u = (exist_table >> WID) | 4575657221408423936L;
    //long long exist_u = ~((~exist_table) >> WID);

    for(num_c = 0;num_c < NUM_COLORS;num_c++){
      long long color = color_table[num_c];
      long long color_u = color & exist_u;
      long long color_d = (color << WID) & (~exist_table);
      color_table[num_c] = color_u | color_d;
    }
    exist_table = color_table[0];
    for(num_c = 1;num_c < NUM_COLORS;num_c++){
      exist_table = exist_table | color_table[num_c];
    }
//     if(threadIdx.x == 0 && blockIdx.x == 0){
//       print_table(color_table);
//       print_table2(exist_table);
//     }
    
  }while(exist_org != exist_table);



  /* isLine_combo[combo_counter] = (isErase_table[num_c] >> (WID  +1)) & 63 == 63 */
  /*   || (isErase_table[num_c] >> (WID*2+1)) & 63 == 63 */
  /*   || (isErase_table[num_c] >> (WID*3+1)) & 63 == 63 */
  /*   || (isErase_table[num_c] >> (WID*4+1)) & 63 == 63 */
  /*   || (isErase_table[num_c] >> (WID*5+1)) & 63 == 63;  */
  //printf("combo_counter %d\n", combo_counter);

  return combo_counter;
}


__device__ inline int one_step_opt3(long long color_table[NUM_COLORS], int color_combo[COMBO_LENGTH], int num_drops_combo[COMBO_LENGTH], int isLine_combo[COMBO_LENGTH], const int *bit_hash_table, int finish){
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
  
  long long isErase_tables[NUM_COLORS];
  int combo_counter = finish;
  /* int line_flag[hight]; */
  int i,j;
  int num_c;
  long long tmp, tmp2;

  for(num_c = 0;num_c < NUM_COLORS;num_c++){
    
    long long color = color_table[num_c];

    //自身の上下左右シフトと比較
    long long n, w, s, e;
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

#if NUM_COLORS==2
  if(isErase_tables[0] == isErase_tables[1])
    return combo_counter;

#endif
  for(num_c = 0;num_c < NUM_COLORS;num_c++){
    long long isErase_table = isErase_tables[num_c];
    color_table[num_c] = color_table[num_c] & (~isErase_table);

    // if(threadIdx.x == 0 && blockIdx.x == 0)
    //   print_table2(isErase_table);
    
    /// count drops ///
    // filter 527874L
    // 2進数 (左上が1桁目)
    // 010000000
    // 111000000
    // 010000000
    // 000000000
    // 000000000
    // 000000000
    // 000000000
    // 000000000
    // 上の左シフトと比較
    // tmpとisEraseと上で&してtmpに|すれば良い? ← ちょっと違う
    int flag;
    do {
      flag = 0;
      tmp = 1L << (WID+1);
      for(i = 1;i <= hight;i++, tmp = tmp << 2){
	for(j = 1;j <= width;j++, tmp = tmp << 1){
	  if(isErase_table & tmp){
	    flag = 1;
	    break;
	  }
	}
	if(flag) break;
      }
      if(flag){
	color_combo[combo_counter] = num_c;
	long long tmp_old;
	do{
	  tmp_old = tmp;
// 	  long long p = (1L << (WID*i+1));
// 	  long long filter = 527874L << (WID*(i-1));
// 	  for(ii = i;ii <= hight;ii++,filter = filter << 2,p = p << 2){
// 	    for(jj = 1;jj <= width;jj++,filter = filter << 1,p = p << 1){
// 	      if(isErase_table & p){
// 		if(tmp & filter){
// 		  tmp = tmp | p;
// 		  num_drops_combo[combo_counter]++;
// 		  isErase_table = isErase_table & (~p);
// 		}
// 	      }
// 	    }
// 	  }
	  tmp = (tmp | (tmp << 1) | (tmp >> 1) | (tmp << WID) | (tmp >> WID)) & isErase_table;
	}while(tmp_old != tmp);
	isErase_table = isErase_table & (~tmp);
	int b1, b2, b3, b4, b5, b6;
	b1 = tmp >> (WID*1) & 254;
	b2 = tmp >> (WID*2) & 254;
	b3 = tmp >> (WID*3) & 254;
	b4 = tmp >> (WID*4) & 254;
	b5 = tmp >> (WID*5) & 254;
	b6 = tmp >> (WID*6) & 254;
	num_drops_combo[combo_counter] = bit_hash_table[b1] + bit_hash_table[b2] 
	  + bit_hash_table[b3] + bit_hash_table[b4] + bit_hash_table[b5] + bit_hash_table[b6];
	isLine_combo[combo_counter] = ((tmp >> (WID  +1)) & 127) == 127
	  || ((tmp >> (WID*2+1)) & 127) == 127
	  || ((tmp >> (WID*3+1)) & 127) == 127
	  || ((tmp >> (WID*4+1)) & 127) == 127
	  || ((tmp >> (WID*5+1)) & 127) == 127
	  || ((tmp >> (WID*6+1)) & 127) == 127;
	combo_counter++;
      }
    }while(flag);

  }
  
  /// drop event ///
  // 真理値表から考えると
  // 上の行 | 下の行 || 解上  | 解下     
  //     0  |  0     || 0     | 0  
  //     0  |  1     || 0     | 1  
  //     1  |  0     || 0     | 1  
  //     1  |  1     || 1     | 1
  // 解上 = 上の行 & 下の行
  // 解下 = 上の行 | 下の行
  // 上の行をそれぞれの色のtableにして、
  // 下の行を複合した存在するかどうかのtableにする
  // 上の行 color_u = color_u & exist_d
  // 下の行 color_d += (exist_d | color_u) ^ exist_d
  // exist_u = exist_u & exist_d
  // exist_d = exist_u | exist_d

  //print_table(color_table);
  long long exist_table = color_table[0];
  for(num_c = 1;num_c < NUM_COLORS;num_c++){
    exist_table = exist_table | color_table[num_c];
  }

  long long exist_org;
  do{
    exist_org = exist_table;

    long long exist_u = (exist_table >> WID) | 4575657221408423936L;

    for(num_c = 0;num_c < NUM_COLORS;num_c++){
      long long color = color_table[num_c];
      long long color_u = color & exist_u;
      long long color_d = (color << WID) & (~exist_table);
      color_table[num_c] = color_u | color_d;
    }
    exist_table = color_table[0];
    for(num_c = 1;num_c < NUM_COLORS;num_c++){
      exist_table = exist_table | color_table[num_c];
    }
//     if(threadIdx.x == 0 && blockIdx.x == 0){
//       print_table(color_table);
//       print_table2(exist_table);
//     }
    
  }while(exist_org != exist_table);

  return combo_counter;
}



__device__ inline int one_step_opt4(long long color_table[NUM_COLORS], int color_combo[COMBO_LENGTH], int num_drops_combo[COMBO_LENGTH], int isLine_combo[COMBO_LENGTH], const int *bit_hash_table, int finish){
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
  
  long long isErase_tables[NUM_COLORS];
  //long long isErase_table;
  int combo_counter = finish;
  int num_c;
  long long tmp, tmp2;

  for(num_c = 0;num_c < NUM_COLORS;num_c++){
    
    long long color = color_table[num_c];

    //自身の上下シフト・左右シフトとビット積をとる。その上下・左右が消すべきビット
    long long n, w, s, e;
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

// #if NUM_COLORS==2
//   if(isErase_tables[0] == isErase_tables[1]) 
//     return combo_counter;
//   // isErase_table[0~N] == 0, つまりは消えるドロップがないなら以降の処理は必要ない。
//   // が、しかしおそらくWarp divergenceの関係で、ない方が速い。(少なくともGPUでは)
//   // とすれば、isEraseをtableにしてループ分割する必要はないが、おそらく最適化の関係で分割した方が速い。
// #endif

  for(num_c = 0;num_c < NUM_COLORS;num_c++){
    long long isErase_table = isErase_tables[num_c];
    color_table[num_c] = color_table[num_c] & (~isErase_table);

//     int flag;
//     do {
//       flag = 0;
//       tmp = 1L << (WID+1);
//       for(i = 1;i <= hight;i++, tmp = tmp << 2){
// 	for(j = 1;j <= width;j++, tmp = tmp << 1){
// 	  if(isErase_table & tmp){
// 	    flag = 1;
// 	    break;
// 	  }
// 	}
// 	if(flag) break;
//       }
//       if(flag){
// 	color_combo[combo_counter] = num_c;
// 	long long tmp_old;
// 	do{
// 	  tmp_old = tmp;
// 	  tmp = (tmp | (tmp << 1) | (tmp >> 1) | (tmp << WID) | (tmp >> WID)) & isErase_table;
// 	}while(tmp_old != tmp);
// 	isErase_table = isErase_table & (~tmp);
// 	int b1, b2, b3, b4, b5, b6;
// 	b1 = tmp >> (WID*1+1) & 127;
// 	b2 = tmp >> (WID*2+1) & 127;
// 	b3 = tmp >> (WID*3+1) & 127;
// 	b4 = tmp >> (WID*4+1) & 127;
// 	b5 = tmp >> (WID*5+1) & 127;
// 	b6 = tmp >> (WID*6+1) & 127;
// 	num_drops_combo[combo_counter] = bit_hash_table[b1] + bit_hash_table[b2] 
// 	  + bit_hash_table[b3] + bit_hash_table[b4] + bit_hash_table[b5] + bit_hash_table[b6];
// 	isLine_combo[combo_counter] = ((tmp >> (WID  +1)) & 127) == 127
// 	  || ((tmp >> (WID*2+1)) & 127) == 127
// 	  || ((tmp >> (WID*3+1)) & 127) == 127
// 	  || ((tmp >> (WID*4+1)) & 127) == 127
// 	  || ((tmp >> (WID*5+1)) & 127) == 127
// 	  || ((tmp >> (WID*6+1)) & 127) == 127;
// 	combo_counter++;
//       }
//     }while(flag);
    long long p = 1L << (WID+1);
    while(isErase_table) {
      while(!(isErase_table & p)){
	p = p << 1;
      }
      
      tmp = p;
      color_combo[combo_counter] = num_c;
      long long tmp_old;
      do{
	tmp_old = tmp;
	tmp = (tmp | (tmp << 1) | (tmp >> 1) | (tmp << WID) | (tmp >> WID)) & isErase_table;
      }while(tmp_old != tmp);
      isErase_table = isErase_table & (~tmp);
      int b1, b2, b3, b4, b5, b6;
      b1 = tmp >> (WID*1+1) & 127;
      b2 = tmp >> (WID*2+1) & 127;
      b3 = tmp >> (WID*3+1) & 127;
      b4 = tmp >> (WID*4+1) & 127;
      b5 = tmp >> (WID*5+1) & 127;
      b6 = tmp >> (WID*6+1) & 127;
      num_drops_combo[combo_counter] = bit_hash_table[b1] + bit_hash_table[b2] 
	+ bit_hash_table[b3] + bit_hash_table[b4] + bit_hash_table[b5] + bit_hash_table[b6];
      isLine_combo[combo_counter] = ((tmp >> (WID  +1)) & 127) == 127
	|| ((tmp >> (WID*2+1)) & 127) == 127
	|| ((tmp >> (WID*3+1)) & 127) == 127
	|| ((tmp >> (WID*4+1)) & 127) == 127
	|| ((tmp >> (WID*5+1)) & 127) == 127
	|| ((tmp >> (WID*6+1)) & 127) == 127;
      combo_counter++;
    }
  }
  
  long long exist_table = color_table[0];
  for(num_c = 1;num_c < NUM_COLORS;num_c++){
    exist_table = exist_table | color_table[num_c];
  }

  long long exist_org;
  do{
    exist_org = exist_table;

    long long exist_u = (exist_table >> WID) | 4575657221408423936L;

    for(num_c = 0;num_c < NUM_COLORS;num_c++){
      long long color = color_table[num_c];
      long long color_u = color & exist_u;
      long long color_d = (color << WID) & (~exist_table);
      color_table[num_c] = color_u | color_d;
    }
    exist_table = color_table[0];
    for(num_c = 1;num_c < NUM_COLORS;num_c++){
      exist_table = exist_table | color_table[num_c];
    }
  }while(exist_org != exist_table);

  return combo_counter;
}


__device__ inline int one_step_opt5(long long color_table[NUM_COLORS], int color_combo[COMBO_LENGTH], int num_drops_combo[COMBO_LENGTH], int isLine_combo[COMBO_LENGTH], int finish){
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
  
  long long isErase_tables[NUM_COLORS];
  //long long isErase_table;
  int combo_counter = finish;
  int num_c;
  long long tmp, tmp2;

  for(num_c = 0;num_c < NUM_COLORS;num_c++){
    
    long long color = color_table[num_c];

    //自身の上下シフト・左右シフトとビット積をとる。その上下・左右が消すべきビット
    long long n, w, s, e;
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

// #if NUM_COLORS==2
//   if(isErase_tables[0] == isErase_tables[1]) 
//     return combo_counter;
//   // isErase_table[0~N] == 0, つまりは消えるドロップがないなら以降の処理は必要ない。
//   // が、しかしおそらくWarp divergenceの関係で、ない方が速い。(少なくともGPUでは)
//   // とすれば、isEraseをtableにしてループ分割する必要はないが、おそらく最適化の関係で分割した方が速い。
// #endif

  for(num_c = 0;num_c < NUM_COLORS;num_c++){
    long long isErase_table = isErase_tables[num_c];
    color_table[num_c] = color_table[num_c] & (~isErase_table);

    long long p = 1L << (WID+1);
    while(isErase_table) {
      while(!(isErase_table & p)){
	p = p << 1;
      }
      
      tmp = p;
      color_combo[combo_counter] = num_c;
      long long tmp_old;
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
//       num_drops_combo[combo_counter] = bit_hash_table[b1] + bit_hash_table[b2] 
// 	+ bit_hash_table[b3] + bit_hash_table[b4] + bit_hash_table[b5] + bit_hash_table[b6];
      long long bits = tmp;
//       bits = (bits & 0x5555555555555555) + (bits >> 1 & 0x5555555555555555);
//       bits = (bits & 0x3333333333333333) + (bits >> 2 & 0x3333333333333333);
//       bits = (bits & 0x0f0f0f0f0f0f0f0f) + (bits >> 4 & 0x0f0f0f0f0f0f0f0f);
//       bits = (bits & 0x00ff00ff00ff00ff) + (bits >> 8 & 0x00ff00ff00ff00ff);
//       bits = (bits & 0x0000ffff0000ffff) + (bits >>16 & 0x0000ffff0000ffff);
//       num_drops_combo[combo_counter] = (bits & 0x00000000ffffffff) + (bits >>32 & 0x00000000ffffffff);

      bits = (bits & 0x5555555555555555LU) + (bits >> 1 & 0x5555555555555555LU);
      bits = (bits & 0x3333333333333333LU) + (bits >> 2 & 0x3333333333333333LU);
      bits = bits + (bits >> 4) & 0x0F0F0F0F0F0F0F0FLU;
      bits = bits + (bits >> 8);
      bits = bits + (bits >> 16);
      bits = bits + (bits >> 32) & 0x0000007F;
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
    long long exist_table = color_table[0];
    for(num_c = 1;num_c < NUM_COLORS;num_c++){
      exist_table = exist_table | color_table[num_c];
    }
    
    long long exist_org;
    do{
      exist_org = exist_table;
      
      long long exist_u = (exist_table >> WID) | 4575657221408423936L;
      
      for(num_c = 0;num_c < NUM_COLORS;num_c++){
	long long color = color_table[num_c];
	long long color_u = color & exist_u;
	long long color_d = (color << WID) & (~exist_table);
	color_table[num_c] = color_u | color_d;
      }
      exist_table = color_table[0];
      for(num_c = 1;num_c < NUM_COLORS;num_c++){
	exist_table = exist_table | color_table[num_c];
      }
    }while(exist_org != exist_table);
  }

  return combo_counter;
}



#endif

#endif



__device__ inline int one_step_opt_c2(long long *color_table, int color_combo[COMBO_LENGTH], int num_drops_combo[COMBO_LENGTH], int isLine_combo[COMBO_LENGTH], int finish){
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
  
  long long color_table0 = color_table[0];
  long long color_table1 = color_table[1];
  long long isErase_table0, isErase_table1;
  int combo_counter = finish;
  /* int line_flag[hight]; */
  int i,j,ii,jj;
  int num_c;
  long long tmp, tmp2;

  //自身の上下左右シフトと比較
  long long n, w, s, e;
  n = color_table0 >> WID;
  w = color_table0 >> 1;
  s = color_table0 << WID;
  e = color_table0 << 1;
  tmp  = (color_table0 & n & s);
  tmp  = tmp  | (tmp  >> WID) | (tmp  << WID);
  tmp2 = (color_table0 & w & e);
  tmp2 = tmp2 | (tmp2 >> 1  ) | (tmp2 << 1  );
  isErase_table0 = (color_table0 & tmp) | (color_table0 & tmp2);
  
  //自身の上下左右シフトと比較
  n = color_table1 >> WID;
  w = color_table1 >> 1;
  s = color_table1 << WID;
  e = color_table1 << 1;
  tmp  = (color_table1 & n & s);
  tmp  = tmp  | (tmp  >> WID) | (tmp  << WID);
  tmp2 = (color_table1 & w & e);
  tmp2 = tmp2 | (tmp2 >> 1  ) | (tmp2 << 1  );
  isErase_table1 = (color_table1 & tmp) | (color_table1 & tmp2);
  
  if(isErase_table0 == isErase_table1)
    return combo_counter;

  color_table0 = color_table0 & (~isErase_table0);
  color_table1 = color_table1 & (~isErase_table1);

  /// count drops ///
  // filter 527874L
  // 2進数 (左上が1桁目)
  // 010000000
  // 111000000
  // 010000000
  // 000000000
  // 000000000
  // 000000000
  // 000000000
  // 000000000
  // 上の左シフトと比較
  // tmpとisEraseと上で&してtmpに|すれば良い? ← ちょっと違う
  int flag;
  do {
    flag = 0;
    tmp = 1L << (WID+1);
    for(i = 1;i <= hight;i++, tmp = tmp << 2){
      for(j = 1;j <= width;j++, tmp = tmp << 1){
	if(isErase_table0 & tmp){
	  flag = 1;
	  break;
	}
      }
      if(flag) break;
    }
    if(flag){
      color_combo[combo_counter] = num_c;
      long long tmp_old;
      do{
	tmp_old = tmp;
	long long p = (1L << (WID*i+1));
	long long filter = 527874L << (WID*(i-1));
	//#pragma unroll
	for(ii = i;ii <= hight;ii++,filter = filter << 2,p = p << 2){
	  //for(ii = i-1;ii < hight;ii++,p = p << 2){
	  //#pragma unroll
	  for(jj = 1;jj <= width;jj++,filter = filter << 1,p = p << 1){
	    //for(jj = 0;jj < width;jj++,p = p << 1){
	    if(isErase_table0 & p){
	      //filter = 527874L << (WID*(ii)+(jj));
	      if(tmp & filter){
		tmp = tmp | p;
		num_drops_combo[combo_counter]++;
		//color = color & (~p);
		isErase_table0 = isErase_table0 & (~p);
	      }
	    }
	  }
	}

      }while(tmp_old != tmp);
      isLine_combo[combo_counter] = ((tmp >> (WID  +1)) & 127) == 127
	|| ((tmp >> (WID*2+1)) & 127) == 127
	|| ((tmp >> (WID*3+1)) & 127) == 127
	|| ((tmp >> (WID*4+1)) & 127) == 127
	|| ((tmp >> (WID*5+1)) & 127) == 127
	|| ((tmp >> (WID*6+1)) & 127) == 127;
      combo_counter++;
    }
  }while(flag);

  do {
    flag = 0;
    tmp = 1L << (WID+1);
    for(i = 1;i <= hight;i++, tmp = tmp << 2){
      for(j = 1;j <= width;j++, tmp = tmp << 1){
	if(isErase_table1 & tmp){
	  flag = 1;
	  break;
	}
      }
      if(flag) break;
    }
    if(flag){
      color_combo[combo_counter] = num_c;
      long long tmp_old;
      do{
	tmp_old = tmp;
	long long p = (1L << (WID*i+1));
	long long filter = 527874L << (WID*(i-1));
	//#pragma unroll
	for(ii = i;ii <= hight;ii++,filter = filter << 2,p = p << 2){
	  //for(ii = i-1;ii < hight;ii++,p = p << 2){
	  //#pragma unroll
	  for(jj = 1;jj <= width;jj++,filter = filter << 1,p = p << 1){
	    //for(jj = 0;jj < width;jj++,p = p << 1){
	    if(isErase_table1 & p){
	      //filter = 527874L << (WID*(ii)+(jj));
	      if(tmp & filter){
		tmp = tmp | p;
		num_drops_combo[combo_counter]++;
		//color = color & (~p);
		isErase_table1 = isErase_table1 & (~p);
	      }
	    }
	  }
	}

      }while(tmp_old != tmp);
      isLine_combo[combo_counter] = ((tmp >> (WID  +1)) & 127) == 127
	|| ((tmp >> (WID*2+1)) & 127) == 127
	|| ((tmp >> (WID*3+1)) & 127) == 127
	|| ((tmp >> (WID*4+1)) & 127) == 127
	|| ((tmp >> (WID*5+1)) & 127) == 127
	|| ((tmp >> (WID*6+1)) & 127) == 127;
      combo_counter++;
    }
  }while(flag);

  
  /// drop event ///
  // 真理値表から考えると
  // 上の行 | 下の行 || 解上  | 解下     
  //     0  |  0     || 0     | 0  
  //     0  |  1     || 0     | 1  
  //     1  |  0     || 0     | 1  
  //     1  |  1     || 1     | 1
  // 解上 = 上の行 & 下の行
  // 解下 = 上の行 | 下の行
  // 上の行をそれぞれの色のtableにして、
  // 下の行を複合した存在するかどうかのtableにする
  // 上の行 color_u = color_u & exist_d
  // 下の行 color_d += (exist_d | color_u) ^ exist_d
  // exist_u = exist_u & exist_d
  // exist_d = exist_u | exist_d

  //print_table(color_table);
  long long exist_table = color_table0 | color_table1;

  long long exist_org;
  int upper = 2;

  long long up, dp;
  do{
    exist_org = exist_table;
    up = (254L << (WID*(hight-1)));
    dp = (254L << (WID*(hight  )));
    for(i = hight;i >= upper;i--,up = up >> WID, dp = dp >> WID){
      long long exist_u = exist_table & up;
      long long exist_d = exist_table & dp;
      //print_table2(exist_u);
      //print_table2(exist_d);
      long long color;
      long long color_u;
      color = color_table0;
      color_u = color & up;
      color = (color & (~up)) | (color_u & (exist_d >> WID));
      color_table0 = color | ((exist_d | (color_u << WID)) ^ exist_d) ;
      color = color_table1;
      color_u = color & up;
      color = (color & (~up)) | (color_u & (exist_d >> WID));
      color_table1 = color | ((exist_d | (color_u << WID)) ^ exist_d) ;

      exist_table = (exist_table & (~up)) | (exist_u & (exist_d >> WID));
      exist_table = (exist_table & (~dp)) | ((exist_u << WID) | exist_d);
      //print_table2(exist_table);
    }
    upper++;
  }while(exist_org != exist_table);

  color_table[0] = color_table0;
  color_table[1] = color_table1;

  return combo_counter;
}


__device__ void print_table(long long color_table[NUM_COLORS]){

  /* printf("c1 %ld \n",color_table[0]); */
  /* printf("c2 %ld \n",color_table[1]); */

  int i, j;
  /* for(i = hight;i >= 1;i--){ */
  /*   for(j = width;j >= 1;j--){ */
  for(i = 1;i <= hight;i++){
    for(j = 1;j <= width;j++){
      long long p = (1L << (WID*i+j));
      if((color_table[0] & p) == p)
	printf("G ");
      else if((color_table[1] & p) == p)
	printf("Y ");
      else
	printf("? ");
    }
    printf("\n");
  }
  printf("\n");
}


__device__ void print_table2(long long color_table){
  int i, j;
  /* for(i = hight;i >= 1;i--){ */
  /*   for(j = width;j >= 1;j--){ */
  for(i = 1;i <= hight;i++){
    for(j = 1;j <= width;j++){
      long long p = (1L << (WID*i+j));
      printf("%d ", (color_table & p) == p);
    }
    printf("\n");
  }
  printf("\n");
}


__device__ inline float return_attack(int combo_counter, int color_combo[COMBO_LENGTH], int num_drops_combo[COMBO_LENGTH], int isLine_combo[COMBO_LENGTH], float line, float way ){
  // used for simulation mode
  // [FIXME] check only Green attack
  int num_line_G = 0;
  float attack_G = 0;
  float l = LS;
  int i;
  float drop_pwr;
  for(i = 0;i < combo_counter;i++){
    int color = color_combo[i];

    //drop_pwr = drop_pwr * (1+0.06*(combo[i].num_strong_drops));
    switch(color){
    case G:
      drop_pwr = num_drops_combo[i]==4 ? (1+0.25*(num_drops_combo[i]-3))*way : 1+0.25*(num_drops_combo[i]-3);
      if(STRONG_DROP)
	drop_pwr = drop_pwr * (1+0.06*num_drops_combo[i]);
      attack_G += drop_pwr; 
      if(isLine_combo[i]) num_line_G++;
      break;
    default:
      break;
    }
  }
  int heroLS6 = 0;
  int heroLS7 = 0;
  int heroLS8 = 0;
  if(HERO){
    for(i = 0;i < combo_counter;i++){
      if(MYCOLOR == color_combo[i]){
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
  }else if(SONIA){
    if(combo_counter < 6)
      l = 6.25;
    else
      l = 2.75*2.75;
  }else if(KRISHNA){
    int count = 0;
    for(i = 0;i < combo_counter;i++){
      if(MYCOLOR == color_combo[i]){
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
  }else if(BASTET){
    if(combo_counter == 5)
      l = 3.0*3.0;
    else if(combo_counter == 6)
      l = 3.5*3.5;
    else if(combo_counter >= 7)
      l = 4.0*4.0;
    else
      l = 1.0;
  }
  attack_G = attack_G * (1+0.25*(combo_counter-1)) * ATT * l * (1+0.1*line*num_line_G) ;
  return attack_G;
}
