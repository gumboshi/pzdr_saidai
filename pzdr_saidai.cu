/* author gumboshi <gumboshi@gmail.com> */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "pzdr_def.h"
#include "pzdr_saidai.h"

#define LOCALRANKINGLENGTH 5

#ifndef NUM_BLOCK
#define NUM_BLOCK 52
#endif
#define NUM_THREAD 256

#define CUDA_SAFE_CALL(func)			\
  do {						\
    cudaError_t err = (func);						\
    if (err != cudaSuccess) {						\
      fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
      exit(err);							\
    }									\
  } while(0)


// __device__ inline void init_combo_info(int *color_combo, int *num_drops_combo, int *isLine_combo, int combo_length){
//   int i;
//   for(i = 0;i < combo_length;i++){
//     color_combo[i] = 0;
//     num_drops_combo[i] = 0;
//     isLine_combo[i] = 0;
//   }
// }

__device__ void print_table_dev(unsigned long long *color_table, int width, int hight){

  int i, j;
  for(i = 1;i <= hight;i++){
    for(j = 1;j <= width;j++){
      unsigned long long p = (1L << ((width*2)*i+j));
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


__device__ void print_table2_dev(unsigned long long color_table, int width, int hight){
  int i, j;
  for(i = 1;i <= hight;i++){
    for(j = 1;j <= width;j++){
      unsigned long long p = (1L << ((width*2)*i+j));
      printf("%d ", (color_table & p) == p);
    }
    printf("\n");
  }
  printf("\n");
}


#if NUM_COLORS==2
#define WID 7
__device__ inline void generate_table_small_dev(unsigned long long tableID, unsigned long long *color_table){

  unsigned long long b0, b1, b2, b3;
  unsigned long long ID = tableID;
  b0 = ID & 31;
  b1 = (ID >> 5 ) & 31;
  b2 = (ID >> 10) & 31;
  b3 = (ID >> 15) & 31;
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
__device__ inline void generate_table_normal_dev(unsigned long long tableID, unsigned long long *color_table){

  unsigned long long b0, b1, b2, b3, b4;
  unsigned long long ID = tableID;
  b0 = ID & 63;
  b1 = (ID >> 6 ) & 63;
  b2 = (ID >> 12) & 63;
  b3 = (ID >> 18) & 63;
  b4 = (ID >> 24) & 63;
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
__device__ inline void generate_table_big_dev(unsigned long long tableID, unsigned long long *color_table){

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
#endif


#if NUM_COLORS==2

#define WID 7
__device__ inline int one_step_small_dev(unsigned long long *color_table, int *color_combo, int *num_drops_combo, int *isLine_combo, int finish){
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
  
  unsigned long long isErase_tables[NUM_COLORS];
  int combo_counter = finish;
  int num_c;
  unsigned long long tmp, tmp2;

  for(num_c = 0;num_c < NUM_COLORS;num_c++){
    
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

  }
  for(num_c = 0;num_c < NUM_COLORS;num_c++){
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

      bits = (bits & 0x5555555555555555LU) + (bits >> 1 & 0x5555555555555555LU);
      bits = (bits & 0x3333333333333333LU) + (bits >> 2 & 0x3333333333333333LU);
      bits = bits + (bits >> 4) & 0x0F0F0F0F0F0F0F0FLU;
      bits = bits + (bits >> 8);
      bits = bits + (bits >> 16);
      bits = bits + (bits >> 32) & 0x0000007F;
      num_drops_combo[combo_counter] = bits;

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
    for(num_c = 1;num_c < NUM_COLORS;num_c++){
      exist_table = exist_table | color_table[num_c];
    }
    
    unsigned long long exist_org;
    do{
      exist_org = exist_table;
      
      unsigned long long exist_u = (exist_table >> WID) | 16642998272L;
      for(num_c = 0;num_c < NUM_COLORS;num_c++){
	unsigned long long color = color_table[num_c];
	unsigned long long color_u = color & exist_u;
	unsigned long long color_d = (color << WID) & (~exist_table) & (~2130303778816L); 
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
#undef WID

#define WID 8
__device__ inline int one_step_normal_dev(unsigned long long *color_table, int *color_combo, int *num_drops_combo, int *isLine_combo, int finish){
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
  
  unsigned long long isErase_tables[NUM_COLORS];
  int combo_counter = finish;
  int num_c;
  unsigned long long tmp, tmp2;

  for(num_c = 0;num_c < NUM_COLORS;num_c++){
    
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

  }

  for(num_c = 0;num_c < NUM_COLORS;num_c++){
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
      unsigned long long bits = tmp;
      bits = (bits & 0x5555555555555555LU) + (bits >> 1 & 0x5555555555555555LU);
      bits = (bits & 0x3333333333333333LU) + (bits >> 2 & 0x3333333333333333LU);
      bits = bits + (bits >> 4) & 0x0F0F0F0F0F0F0F0FLU;
      bits = bits + (bits >> 8);
      bits = bits + (bits >> 16);
      bits = bits + (bits >> 32) & 0x0000007F;
      num_drops_combo[combo_counter] = bits;

      isLine_combo[combo_counter] = ((tmp >> (WID  +1)) & 63) == 63
	|| ((tmp >> (WID*2+1)) & 63) == 63
	|| ((tmp >> (WID*3+1)) & 63) == 63
	|| ((tmp >> (WID*4+1)) & 63) == 63
	|| ((tmp >> (WID*5+1)) & 63) == 63;
      combo_counter++;
    }
  }

  if(finish != combo_counter){
    unsigned long long exist_table = color_table[0];
    for(num_c = 1;num_c < NUM_COLORS;num_c++){
      exist_table = exist_table | color_table[num_c];
    }
    
    unsigned long long exist_org;
    do{
      exist_org = exist_table;
      
      unsigned long long exist_u = (exist_table >> WID) | 138538465099776L;
      for(num_c = 0;num_c < NUM_COLORS;num_c++){
	unsigned long long color = color_table[num_c];
	unsigned long long color_u = color & exist_u;
	unsigned long long color_d = (color << WID) & (~exist_table) & (~35465847065542656L);
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
#undef WID

#define WID 9

__device__ inline int one_step_big_dev(unsigned long long *color_table, int *color_combo, int *num_drops_combo, int *isLine_combo, int finish){
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
  
  unsigned long long isErase_tables[NUM_COLORS];
  int combo_counter = finish;
  int num_c;
  unsigned long long tmp, tmp2;

  for(num_c = 0;num_c < NUM_COLORS;num_c++){
    
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
//       unsigned long long bits = tmp;
//       bits = (bits & 0x5555555555555555) + (bits >> 1 & 0x5555555555555555);
//       bits = (bits & 0x3333333333333333) + (bits >> 2 & 0x3333333333333333);
//       bits = (bits & 0x0f0f0f0f0f0f0f0f) + (bits >> 4 & 0x0f0f0f0f0f0f0f0f);
//       bits = (bits & 0x00ff00ff00ff00ff) + (bits >> 8 & 0x00ff00ff00ff00ff);
//       bits = (bits & 0x0000ffff0000ffff) + (bits >>16 & 0x0000ffff0000ffff);
//       num_drops_combo[combo_counter] = (bits & 0x00000000ffffffff) + (bits >>32 & 0x00000000ffffffff);

//       bits = (bits & 0x5555555555555555LU) + (bits >> 1 & 0x5555555555555555LU);
//       bits = (bits & 0x3333333333333333LU) + (bits >> 2 & 0x3333333333333333LU);
//       bits = bits + (bits >> 4) & 0x0F0F0F0F0F0F0F0FLU;
//       bits = bits + (bits >> 8);
//       bits = bits + (bits >> 16);
//       bits = bits + (bits >> 32) & 0x0000007F;
//       num_drops_combo[combo_counter] = bits;

      unsigned int u = tmp >> 32;
      unsigned int l = tmp;
      u = (u & 0x55555555) + (u >> 1 & 0x55555555);
      u = (u & 0x33333333) + (u >> 2 & 0x33333333);
      u = u + (u >> 4) & 0x0F0F0F0F;
      u = u + (u >> 8);
      u = u + (u >> 16) & 0x0000007F;

      l = (l & 0x55555555) + (l >> 1 & 0x55555555);
      l = (l & 0x33333333) + (l >> 2 & 0x33333333);
      l = l + (l >> 4) & 0x0F0F0F0F;
      l = l + (l >> 8);
      l = l + (l >> 16) & 0x0000007F;
      num_drops_combo[combo_counter] = u + l;

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
    for(num_c = 1;num_c < NUM_COLORS;num_c++){
      exist_table = exist_table | color_table[num_c];
    }
    
    unsigned long long exist_org;
    do{
      exist_org = exist_table;
      
      unsigned long long exist_u = (exist_table >> WID) | 4575657221408423936L;
      
      for(num_c = 0;num_c < NUM_COLORS;num_c++){
	unsigned long long color = color_table[num_c];
	unsigned long long color_u = color & exist_u;
	unsigned long long color_d = (color << WID) & (~exist_table);
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
#undef WID

#endif

__device__ inline float return_attack_dev(int combo_counter, int *color_combo, int *num_drops_combo, int *isLine_combo, int LS, int strong, float line, float way){
  // used for simulation mode
  // [FIXME] check only Green attack
  int num_line = 0;
  float AT = 1.0;
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

  int count;
  switch(LS){
  case HERO: 
    for(i = 0;i < combo_counter;i++){
      if(MAINCOLOR == color_combo[i]){
	int num_drops = num_drops_combo[i];
	if(num_drops >= 8){
	  l = 16;
	}else if(num_drops == 7 && l < 12.25){
	  l = 12.25;
	}else if(num_drops == 6 && l < 9){
	  l = 9;
	}
      }
    }
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


__device__ inline void return_attack_double_dev(float *power, const int combo_counter, int *const color_combo, int * const num_drops_combo, int * const isLine_combo, const int LS, const int strong, const float line, const float way){
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
    if(color == MAINCOLOR){
      drop_pwr = num_drops_combo[i]==4 ? (1+0.25*(num_drops_combo[i]-3))*way : 1+0.25*(num_drops_combo[i]-3); 
      if(strong)
	drop_pwr = drop_pwr * (1+0.06*num_drops_combo[i]);
      attack_m += drop_pwr; 
      if(isLine_combo[i]) num_line_m++;
    }else{
      drop_pwr = num_drops_combo[i]==4 ? (1+0.25*(num_drops_combo[i]-3))*way : 1+0.25*(num_drops_combo[i]-3); 
      if(strong)
	drop_pwr = drop_pwr * (1+0.06*num_drops_combo[i]);
      attack_s += drop_pwr; 
      if(isLine_combo[i]) num_line_s++;
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
  power[1] = attack_s * (1+0.25*(combo_counter-1)) * AT * l_s * (1+0.1*line*num_line_s);
}


#define COMBO_LENGTH 7
#define REVERSE_LENGTH 32
// __global__ void simulate_all_kernel_small(int num_attacks, const int * __restrict__ num_patterns10, unsigned long long *maxID, float *maxPower, float line, float way, const int * __restrict__ tableID_prefix10, int *tableID_table10, const int * __restrict__ reversed_bit_table, /*const int * __restrict__ bit_count_table,*/ int LS, int strong){

//   int tid = threadIdx.x;
//   int bid = blockIdx.x;
//   int gdim = gridDim.x;
//   int bdim = blockDim.x;
//   int color_combo[COMBO_LENGTH];
//   int num_drops_combo[COMBO_LENGTH];
//   int isLine_combo[COMBO_LENGTH];
//   int i,j,k;
//   int rank = LOCALRANKINGLENGTH;
//   float MP[LOCALRANKINGLENGTH];
//   unsigned long long MID[LOCALRANKINGLENGTH];
//   unsigned long long tableID = 0;

//   unsigned long long color_table[NUM_COLORS];
//   int num_c;
//   for(num_c = 0;num_c < NUM_COLORS;num_c++){
//     color_table[num_c] = 0;
//   }

//   for(i = 0;i < rank;i++){
//     MID[i] = 0;
//     MP[i] = 0.0;
//   }

//   int u, l, uu, ll;
//   int bit_num[4];
//   for(u = 0;u <= num_attacks;u++){
//     l = num_attacks - u;
//     if(u <= 10 && l <= 10){
//       int uoffset = tableID_prefix10[u];
//       int loffset = tableID_prefix10[l];
//       for(uu = bid;uu < num_patterns10[u];uu+=gdim){
// 	unsigned long long upperID = (unsigned long long)tableID_table10[uu+uoffset];
//         for(ll = tid;ll < num_patterns10[l];ll+=bdim){
//           unsigned long long lowerID = (unsigned long long)tableID_table10[ll+loffset];
//           tableID = (upperID << 10) | lowerID;

// 	  unsigned long long reversed = 0;
// 	  for(i = 0;i < 4; i++){
// 	    bit_num[i] = (int)((tableID >> (5*i) ) & (REVERSE_LENGTH-1));
// 	    reversed += ((unsigned long long)reversed_bit_table[bit_num[i]]) << (5*i);
// 	  }
// 	  if(tableID <= reversed){
// 	    //init_combo_info(color_combo, num_drops_combo, isLine_combo, COMBO_LENGTH);
// 	    int combo_counter = 0;
// 	    //tableID = 1103874885640L;
// 	    //tableID = 42656280L;
// 	    generate_table_small_dev(tableID, color_table);
// 	    int returned_combo_counter = 0;
// 	    do{
// // 	      if(blockDim.x * blockIdx.x + threadIdx.x == 0){
// // 	        printf("ID %lld\n",tableID);
// // 	        print_table(color_table);
// // 	        print_table2(color_table[0]);
// // 	        print_table2(color_table[1]);
// // 	      }
// 	      combo_counter = returned_combo_counter;
// 	      returned_combo_counter = one_step_small_dev(color_table, color_combo, num_drops_combo, isLine_combo, combo_counter);
// 	      //printf("combo = %d\n", returned_combo_counter);
// 	    }while(returned_combo_counter != combo_counter);
// 	    float power = return_attack_dev(combo_counter, color_combo, num_drops_combo, isLine_combo, LS, strong, line, way);
// 	    if(MP[rank-1] < power){
// 	      for(j = 0;j < rank;j++){
// 		if(MP[j] < power){
// 		  for(k = rank-2;k >= j;k--){
// 		    MID[k+1] = MID[k];
// 		    MP[k+1] = MP[k];
// 		  }
// 		  MID[j] = tableID;
// 		  MP[j] = power;
// 		  break;
// 		}
// 	      }
// 	    }
// 	  }
//         }
//       }
//     }
//   }

//   int id = blockDim.x * blockIdx.x + threadIdx.x;
//   int step = blockDim.x * gridDim.x;

//   for(i = 0;i < rank;i++){
//     maxPower[id + step*i] = MP[i];
//     maxID[id + step*i] = MID[i];
//   }
// }
__global__ void simulate_all_kernel_small(const int num_attacks, const int * __restrict__ num_patterns10, unsigned long long *maxID, float *maxPower, const float line, const float way, const int * __restrict__ tableID_prefix10, const int * __restrict__ tableID_table10, const int * __restrict__ reversed_bit_table, /*const int * __restrict__ bit_count_table,*/ const int LS, const int strong){

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int gdim = gridDim.x;
  const int bdim = blockDim.x;
  int color_combo[COMBO_LENGTH];
  int num_drops_combo[COMBO_LENGTH];
  int isLine_combo[COMBO_LENGTH];
  int i,j,k;
  int rank = LOCALRANKINGLENGTH;
  float MP[2][LOCALRANKINGLENGTH];
  unsigned long long MID[2][LOCALRANKINGLENGTH];
  unsigned long long tableID = 0;
  int ms;

  for(ms = 0;ms < 2;ms++){
    for(i = 0;i < rank;i++){
      MID[ms][i] = 0;
      MP[ms][i] = 0.0;
    }
  }

  int u, l, uu, ll;
  for(u = 0;u <= num_attacks;u++){
    l = num_attacks - u;
    if(u <= 10 && l <= 10){
      int uoffset = tableID_prefix10[u];
      int loffset = tableID_prefix10[l];
      for(uu = bid;uu < num_patterns10[u];uu+=gdim){
	unsigned long long upperID = (unsigned long long)tableID_table10[uu+uoffset];
	for(ll = tid;ll < num_patterns10[l];ll+=bdim){
	  unsigned long long lowerID = (unsigned long long)tableID_table10[ll+loffset];
	  tableID = (upperID << 10) | lowerID;

	  unsigned long long reversed = 0;
	  int reversed_bit[4];
	  for(i = 0;i < 4; i++){
	    reversed_bit[i] = ((tableID >> (5*i) ) & (REVERSE_LENGTH-1));
	    reversed = reversed | ((unsigned long long)reversed_bit_table[reversed_bit[i]]) << (5*i);
	  }
	  if(tableID <= reversed){
	    //init_combo_info(color_combo, num_drops_combo, isLine_combo, COMBO_LENGTH);
	    unsigned long long color_table[NUM_COLORS];
	    generate_table_small_dev(tableID, color_table);
	    int combo_counter;
	    int returned_combo_counter = 0;
	    do{
	      combo_counter = returned_combo_counter;
	      returned_combo_counter = one_step_small_dev(color_table, color_combo, num_drops_combo, isLine_combo, combo_counter);
	    }while(returned_combo_counter != combo_counter);
	    //float power = return_attack_dev(combo_counter, color_combo, num_drops_combo, isLine_combo, LS, strong, line, way);
	    float power[2];
	    return_attack_double_dev(power, combo_counter, color_combo, num_drops_combo, isLine_combo, LS, strong, line, way);
	    if(MP[0][rank-1] < power[0]){
	      for(j = 0;j < rank;j++){
		if(MP[0][j] < power[0]){
		  for(k = rank-2;k >= j;k--){
		    MID[0][k+1] = MID[0][k];
		    MP[0][k+1] = MP[0][k];
		  }
		  MID[0][j] = tableID;
		  MP[0][j] = power[0];
		  break;
		}
	      }
	    }
	    if(MP[1][rank-1] < power[1]){
	      for(j = 0;j < rank;j++){
		if(MP[1][j] < power[1]){
		  for(k = rank-2;k >= j;k--){
		    MID[1][k+1] = MID[1][k];
		    MP[1][k+1] = MP[1][k];
		  }
		  MID[1][j] = (~tableID) & 0x000FFFFF;
		  MP[1][j] = power[1];
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
  for(ms = 0;ms < 2;ms++){
    for(i = 0;i < rank;i++){
      maxPower[id + step*i + ms*step*rank] = MP [ms][i];
      maxID   [id + step*i + ms*step*rank] = MID[ms][i];
    }
  }
}
#undef COMBO_LENGTH
#undef REVERSE_LENGTH

#define COMBO_LENGTH 10
#define REVERSE_LENGTH 64
// __global__ void simulate_all_kernel_normal(int num_attacks, const int * __restrict__ num_patterns15, unsigned long long *maxID, float *maxPower, float line, float way, const int * __restrict__ tableID_prefix15, int *tableID_table15, const int * __restrict__ reversed_bit_table, /*const int * __restrict__ bit_count_table,*/ int LS, int strong){

//   int tid = threadIdx.x;
//   int bid = blockIdx.x;
//   int gdim = gridDim.x;
//   int bdim = blockDim.x;
//   int color_combo[COMBO_LENGTH];
//   int num_drops_combo[COMBO_LENGTH];
//   int isLine_combo[COMBO_LENGTH];
//   int i,j,k;
//   int rank = LOCALRANKINGLENGTH;
//   float MP[LOCALRANKINGLENGTH];
//   unsigned long long MID[LOCALRANKINGLENGTH];
//   unsigned long long tableID = 0;

//   unsigned long long color_table[NUM_COLORS];
//   int num_c;
//   for(num_c = 0;num_c < NUM_COLORS;num_c++){
//     color_table[num_c] = 0;
//   }

//   for(i = 0;i < rank;i++){
//     MID[i] = 0;
//     MP[i] = 0.0;
//   }

//   int u, l, uu, ll;
//   int bit_num[5];
//   for(u = 0;u <= num_attacks;u++){
//     l = num_attacks - u;
//     if(u <= 15 && l <= 15){
//       int uoffset = tableID_prefix15[u];
//       int loffset = tableID_prefix15[l];
//       for(uu = bid;uu < num_patterns15[u];uu+=gdim){
// 	unsigned long long upperID = (unsigned long long)tableID_table15[uu+uoffset];
//         for(ll = tid;ll < num_patterns15[l];ll+=bdim){
//           unsigned long long lowerID = (unsigned long long)tableID_table15[ll+loffset];
//           tableID = (upperID << 15) | lowerID;

// 	  unsigned long long reversed = 0;
// 	  for(i = 0;i < 5; i++){
// 	    bit_num[i] = (int)((tableID >> (6*i) ) & (REVERSE_LENGTH-1));
// 	    reversed += ((unsigned long long)reversed_bit_table[bit_num[i]]) << (6*i);
// 	  }
// 	  if(tableID <= reversed){
// 	    //init_combo_info(color_combo, num_drops_combo, isLine_combo, COMBO_LENGTH);
// 	    int combo_counter = 0;
// 	    //tableID = 1103874885640L;
// 	    //tableID = 42656280L;
// 	    generate_table_normal_dev(tableID, color_table);
// 	    int returned_combo_counter = 0;
// 	    do{
// // 	      if(blockDim.x * blockIdx.x + threadIdx.x == 0){
// // 	        printf("ID %lld\n",tableID);
// // 	        print_table(color_table);
// // 	        print_table2(color_table[0]);
// // 	        print_table2(color_table[1]);
// // 	      }
// 	      combo_counter = returned_combo_counter;
// 	      returned_combo_counter = one_step_normal_dev(color_table, color_combo, num_drops_combo, isLine_combo, combo_counter);
// 	      //printf("combo = %d\n", returned_combo_counter);
// 	    }while(returned_combo_counter != combo_counter);
// 	    float power = return_attack_dev(combo_counter, color_combo, num_drops_combo, isLine_combo, LS, strong, line, way);
// 	    if(MP[rank-1] < power){
// 	      for(j = 0;j < rank;j++){
// 		if(MP[j] < power){
// 		  for(k = rank-2;k >= j;k--){
// 		    MID[k+1] = MID[k];
// 		    MP[k+1] = MP[k];
// 		  }
// 		  MID[j] = tableID;
// 		  MP[j] = power;
// 		  break;
// 		}
// 	      }
// 	    }
// 	  }
//         }
//       }
//     }
//   }

//   int id = blockDim.x * blockIdx.x + threadIdx.x;
//   int step = blockDim.x * gridDim.x;

//   for(i = 0;i < rank;i++){
//     maxPower[id + step*i] = MP[i];
//     maxID[id + step*i] = MID[i];
//   }
// }
__global__ void simulate_all_kernel_normal(const int num_attacks, const int * __restrict__ num_patterns15, unsigned long long *maxID, float *maxPower, const float line, const float way, const int * __restrict__ tableID_prefix15, const int * __restrict__ tableID_table15, const int * __restrict__ reversed_bit_table, /*const int * __restrict__ bit_count_table,*/ const int LS, const int strong){

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int gdim = gridDim.x;
  const int bdim = blockDim.x;
  int color_combo[COMBO_LENGTH];
  int num_drops_combo[COMBO_LENGTH];
  int isLine_combo[COMBO_LENGTH];
  int i,j,k;
  int rank = LOCALRANKINGLENGTH;
  float MP[2][LOCALRANKINGLENGTH];
  unsigned long long MID[2][LOCALRANKINGLENGTH];
  unsigned long long tableID = 0;
  int ms;

  for(ms = 0;ms < 2;ms++){
    for(i = 0;i < rank;i++){
      MID[ms][i] = 0;
      MP[ms][i] = 0.0;
    }
  }

  int u, l, uu, ll;
  for(u = 0;u <= num_attacks;u++){
    l = num_attacks - u;
    if(u <= 15 && l <= 15){
      int uoffset = tableID_prefix15[u];
      int loffset = tableID_prefix15[l];
      for(uu = bid;uu < num_patterns15[u];uu+=gdim){
	unsigned long long upperID = (unsigned long long)tableID_table15[uu+uoffset];
	for(ll = tid;ll < num_patterns15[l];ll+=bdim){
	  unsigned long long lowerID = (unsigned long long)tableID_table15[ll+loffset];
	  tableID = (upperID << 15) | lowerID;

	  unsigned long long reversed = 0;
	  int reversed_bit[5];
	  for(i = 0;i < 5; i++){
	    reversed_bit[i] = ((tableID >> (6*i) ) & (REVERSE_LENGTH-1));
	    reversed = reversed | ((unsigned long long)reversed_bit_table[reversed_bit[i]]) << (6*i);
	  }
	  if(tableID <= reversed){
	    //init_combo_info(color_combo, num_drops_combo, isLine_combo, COMBO_LENGTH);
	    unsigned long long color_table[NUM_COLORS];
	    generate_table_normal_dev(tableID, color_table);
	    int combo_counter;
	    int returned_combo_counter = 0;
	    do{
	      combo_counter = returned_combo_counter;
	      returned_combo_counter = one_step_normal_dev(color_table, color_combo, num_drops_combo, isLine_combo, combo_counter);
	    }while(returned_combo_counter != combo_counter);
	    //float power = return_attack_dev(combo_counter, color_combo, num_drops_combo, isLine_combo, LS, strong, line, way);
	    float power[2];
	    return_attack_double_dev(power, combo_counter, color_combo, num_drops_combo, isLine_combo, LS, strong, line, way);
	    if(MP[0][rank-1] < power[0]){
	      for(j = 0;j < rank;j++){
		if(MP[0][j] < power[0]){
		  for(k = rank-2;k >= j;k--){
		    MID[0][k+1] = MID[0][k];
		    MP[0][k+1] = MP[0][k];
		  }
		  MID[0][j] = tableID;
		  MP[0][j] = power[0];
		  break;
		}
	      }
	    }
	    if(MP[1][rank-1] < power[1]){
	      for(j = 0;j < rank;j++){
		if(MP[1][j] < power[1]){
		  for(k = rank-2;k >= j;k--){
		    MID[1][k+1] = MID[1][k];
		    MP[1][k+1] = MP[1][k];
		  }
		  MID[1][j] = (~tableID) & 0x3FFFFFFF;
		  MP[1][j] = power[1];
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
  for(ms = 0;ms < 2;ms++){
    for(i = 0;i < rank;i++){
      maxPower[id + step*i + ms*step*rank] = MP [ms][i];
      maxID   [id + step*i + ms*step*rank] = MID[ms][i];
    }
  }
}

#undef COMBO_LENGTH
#undef REVERSE_LENGTH

#define COMBO_LENGTH 14
#define REVERSE_LENGTH 128
__global__ void simulate_all_kernel_big(const int num_attacks, const int * __restrict__ num_patterns21, unsigned long long *maxID, float *maxPower, const float line, const float way, const int * __restrict__ tableID_prefix21, const int * __restrict__ tableID_table21, const int * __restrict__ reversed_bit_table, /*const int * __restrict__ bit_count_table,*/ const int LS, const int strong){

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int gdim = gridDim.x;
  const int bdim = blockDim.x;
  int color_combo[COMBO_LENGTH];
  int num_drops_combo[COMBO_LENGTH];
  int isLine_combo[COMBO_LENGTH];
  int i,j,k;
  int rank = LOCALRANKINGLENGTH;
  float MP[2][LOCALRANKINGLENGTH];
  unsigned long long MID[2][LOCALRANKINGLENGTH];
  unsigned long long tableID = 0;
  int ms;
  //   unsigned long long color_table[NUM_COLORS];
  //   int num_c;
  //   for(num_c = 0;num_c < NUM_COLORS;num_c++){
  //     color_table[num_c] = 0;
  //   }
  for(ms = 0;ms < 2;ms++){
    for(i = 0;i < rank;i++){
      MID[ms][i] = 0;
      MP[ms][i] = 0.0;
    }
  }

  int u, l, uu, ll;
  for(u = 0;u <= num_attacks;u++){
    l = num_attacks - u;
    if(u <= 21 && l <= 21){
      int uoffset = tableID_prefix21[u];
      int loffset = tableID_prefix21[l];
      for(uu = bid;uu < num_patterns21[u];uu+=gdim){
	unsigned long long upperID = (unsigned long long)tableID_table21[uu+uoffset];
	for(ll = tid;ll < num_patterns21[l];ll+=bdim){
	  unsigned long long lowerID = (unsigned long long)tableID_table21[ll+loffset];
	  tableID = (upperID << 21) | lowerID;

	  unsigned long long reversed = 0;
	  int reversed_bit[6];
	  for(i = 0;i < 6; i++){
	    reversed_bit[i] = ((tableID >> (7*i) ) & (REVERSE_LENGTH-1));
	    reversed = reversed | ((unsigned long long)reversed_bit_table[reversed_bit[i]]) << (7*i);
	  }
	  if(tableID <= reversed){
	    //init_combo_info(color_combo, num_drops_combo, isLine_combo, COMBO_LENGTH);
	    unsigned long long color_table[NUM_COLORS];
	    generate_table_big_dev(tableID, color_table);
	    int combo_counter;
	    int returned_combo_counter = 0;
	    do{
	      combo_counter = returned_combo_counter;
	      returned_combo_counter = one_step_big_dev(color_table, color_combo, num_drops_combo, isLine_combo, combo_counter);
	    }while(returned_combo_counter != combo_counter);
	    //float power = return_attack_dev(combo_counter, color_combo, num_drops_combo, isLine_combo, LS, strong, line, way);
	    float power[2];
	    return_attack_double_dev(power, combo_counter, color_combo, num_drops_combo, isLine_combo, LS, strong, line, way);
	    if(MP[0][rank-1] < power[0]){
	      for(j = 0;j < rank;j++){
		if(MP[0][j] < power[0]){
		  for(k = rank-2;k >= j;k--){
		    MID[0][k+1] = MID[0][k];
		    MP[0][k+1] = MP[0][k];
		  }
		  MID[0][j] = tableID;
		  MP[0][j] = power[0];
		  break;
		}
	      }
	    }
	    if(MP[1][rank-1] < power[1]){
	      for(j = 0;j < rank;j++){
		if(MP[1][j] < power[1]){
		  for(k = rank-2;k >= j;k--){
		    MID[1][k+1] = MID[1][k];
		    MP[1][k+1] = MP[1][k];
		  }
		  MID[1][j] = (~tableID) & 0x000003FFFFFFFFFFLU;
		  MP[1][j] = power[1];
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
  for(ms = 0;ms < 2;ms++){
    for(i = 0;i < rank;i++){
      maxPower[id + step*i + ms*step*rank] = MP [ms][i];
      maxID   [id + step*i + ms*step*rank] = MID[ms][i];
    }
  }
}

__global__ void simulate_all_kernel_big_21(const int * __restrict__ num_patterns21, unsigned long long *maxID, float *maxPower, const float line, const float way, const int * __restrict__ tableID_prefix21, const int * __restrict__ tableID_table21, const int * __restrict__ reversed_bit_table, /*const int * __restrict__ bit_count_table,*/ const int LS, const int strong){

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int gdim = gridDim.x;
  const int bdim = blockDim.x;
  int color_combo[COMBO_LENGTH];
  int num_drops_combo[COMBO_LENGTH];
  int isLine_combo[COMBO_LENGTH];
  int i,j,k;
  int rank = LOCALRANKINGLENGTH;
  float MP[2][LOCALRANKINGLENGTH];
  unsigned long long MID[2][LOCALRANKINGLENGTH];
  unsigned long long tableID = 0;
  int ms;
  for(ms = 0;ms < 2;ms++){
    for(i = 0;i < rank;i++){
      MID[ms][i] = 0;
      MP[ms][i] = 0.0;
    }
  }

  int u, l, uu, ll;
  for(u = 0;u <= 21;u++){
    l = 21 - u;
    int uoffset = tableID_prefix21[u];
    int loffset = tableID_prefix21[l];
    for(uu = bid;uu < num_patterns21[u];uu+=gdim){
      unsigned long long upperID = (unsigned long long)tableID_table21[uu+uoffset];
      for(ll = tid;ll < num_patterns21[l];ll+=bdim){
	unsigned long long lowerID = (unsigned long long)tableID_table21[ll+loffset];
	tableID = (upperID << 21) | lowerID;

	unsigned long long reversed = 0;
	int reversed_bit;
	for(i = 0;i < 6; i++){
	  reversed_bit = ((tableID >> (7*i) ) & (REVERSE_LENGTH-1));
	  reversed = reversed | ((unsigned long long)reversed_bit_table[reversed_bit]) << (7*i);
	}
	unsigned long long inversed = (~tableID) & 0x000003FFFFFFFFFFLU;
	if(tableID <= reversed && tableID <= inversed){
	  unsigned long long color_table[NUM_COLORS];
	  generate_table_big_dev(tableID, color_table);
	  int combo_counter;
	  int returned_combo_counter = 0;
	  do{
	    combo_counter = returned_combo_counter;
	    returned_combo_counter = one_step_big_dev(color_table, color_combo, num_drops_combo, isLine_combo, combo_counter);
	  }while(returned_combo_counter != combo_counter);
	  float power[2];
	  return_attack_double_dev(power, combo_counter, color_combo, num_drops_combo, isLine_combo, LS, strong, line, way);
	  if(MP[0][rank-1] < power[0]){
	    for(j = 0;j < rank;j++){
	      if(MP[0][j] < power[0]){
		for(k = rank-2;k >= j;k--){
		  MID[0][k+1] = MID[0][k];
		  MP[0][k+1] = MP[0][k];
		}
		MID[0][j] = tableID;
		MP[0][j] = power[0];
		break;
	      }
	    }
	  }
	  if(MP[1][rank-1] < power[1]){
	    for(j = 0;j < rank;j++){
	      if(MP[1][j] < power[1]){
		for(k = rank-2;k >= j;k--){
		  MID[1][k+1] = MID[1][k];
		  MP[1][k+1] = MP[1][k];
		}
		MID[1][j] = (~tableID) & 0x000003FFFFFFFFFFLU;
		MP[1][j] = power[1];
		break;
	      }
	    }
	  }
	}
      }
    }
  }

  int id = blockDim.x * blockIdx.x + threadIdx.x;
  int step = blockDim.x * gridDim.x;
  for(ms = 0;ms < 2;ms++){
    for(i = 0;i < rank;i++){
      maxPower[id + step*i + ms*step*rank] = MP [ms][i];
      maxID   [id + step*i + ms*step*rank] = MID[ms][i];
    }
  }

}

#define WID 9
__global__ void simulate_all_kernel_big_inlined(const int num_attacks, const int * __restrict__ num_patterns21, unsigned long long *maxID, float *maxPower, const float line, const float way, const int * __restrict__ tableID_prefix21, const int * __restrict__ tableID_table21, const int * __restrict__ reversed_bit_table, const int LS, const int strong){

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int gdim = gridDim.x;
  const int bdim = blockDim.x;
  int color_combo[COMBO_LENGTH];
  int num_drops_combo[COMBO_LENGTH];
  int isLine_combo[COMBO_LENGTH];
  int i,j,k;
  const int rank = LOCALRANKINGLENGTH;
  float MP[2][LOCALRANKINGLENGTH];
  unsigned long long MID[2][LOCALRANKINGLENGTH];
  unsigned long long tableID = 0;
  int ms;
  for(ms = 0;ms < 2;ms++){
    for(i = 0;i < rank;i++){
      MID[ms][i] = 0;
      MP[ms][i] = 0.0;
    }
  }

  int u, l, uu, ll;
  for(u = 0;u <= num_attacks;u++){
    l = num_attacks - u;
    if(u <= 21 && l <= 21){
      int uoffset = tableID_prefix21[u];
      int loffset = tableID_prefix21[l];
      for(uu = bid;uu < num_patterns21[u];uu+=gdim){
	unsigned long long upperID = (unsigned long long)tableID_table21[uu+uoffset];
	for(ll = tid;ll < num_patterns21[l];ll+=bdim){
	  unsigned long long lowerID = (unsigned long long)tableID_table21[ll+loffset];
	  tableID = (upperID << 21) | lowerID;

	  unsigned long long reversed = 0;
	  int reversed_bit[6];
	  for(i = 0;i < 6; i++){
	    reversed_bit[i] = ((tableID >> (7*i) ) & (REVERSE_LENGTH-1));
	    reversed = reversed | ((unsigned long long)reversed_bit_table[reversed_bit[i]]) << (7*i);
	  }
	  if(tableID <= reversed){
	    unsigned long long color_table[NUM_COLORS];

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

	    int combo_counter = 0;
	    int combo_counter_org;
	    //returned_combo_counter = one_step_big_dev(color_table, color_combo, num_drops_combo, isLine_combo, combo_counter);
	    do{
	      combo_counter_org = combo_counter;
	      
	      unsigned long long isErase_tables[NUM_COLORS];
	      int num_c;
	      unsigned long long tmp, tmp2;
	      
	      for(num_c = 0;num_c < NUM_COLORS;num_c++){
		
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
		
	      }

	      for(num_c = 0;num_c < NUM_COLORS;num_c++){
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
		  
		  unsigned int ubits = tmp >> 32;
		  unsigned int lbits = tmp;
		  ubits = (ubits & 0x55555555) + (ubits >> 1 & 0x55555555);
		  ubits = (ubits & 0x33333333) + (ubits >> 2 & 0x33333333);
		  ubits = ubits + (ubits >> 4) & 0x0F0F0F0F;
		  ubits = ubits + (ubits >> 8);
		  ubits = ubits + (ubits >> 16) & 0x0000007F;
		  
		  lbits = (lbits & 0x55555555) + (lbits >> 1 & 0x55555555);
		  lbits = (lbits & 0x33333333) + (lbits >> 2 & 0x33333333);
		  lbits = lbits + (lbits >> 4) & 0x0F0F0F0F;
		  lbits = lbits + (lbits >> 8);
		  lbits = lbits + (lbits >> 16) & 0x0000007F;
		  num_drops_combo[combo_counter] = ubits + lbits;
		  
		  isLine_combo[combo_counter] = ((tmp >> (WID  +1)) & 127) == 127
		    || ((tmp >> (WID*2+1)) & 127) == 127
		    || ((tmp >> (WID*3+1)) & 127) == 127
		    || ((tmp >> (WID*4+1)) & 127) == 127
		    || ((tmp >> (WID*5+1)) & 127) == 127
		    || ((tmp >> (WID*6+1)) & 127) == 127;
      
		  combo_counter++;
		}
	      }
	      
	      if(combo_counter_org != combo_counter){
		unsigned long long exist_table = color_table[0];
		for(num_c = 1;num_c < NUM_COLORS;num_c++){
		  exist_table = exist_table | color_table[num_c];
		}
		unsigned long long exist_org;
		do{
		  exist_org = exist_table;
		  unsigned long long exist_u = (exist_table >> WID) | 4575657221408423936L;
		  for(num_c = 0;num_c < NUM_COLORS;num_c++){
		    unsigned long long color = color_table[num_c];
		    unsigned long long color_u = color & exist_u;
		    unsigned long long color_d = (color << WID) & (~exist_table);
		    color_table[num_c] = color_u | color_d;
		  }
		  exist_table = color_table[0];
		  for(num_c = 1;num_c < NUM_COLORS;num_c++){
		    exist_table = exist_table | color_table[num_c];
		  }
		}while(exist_org != exist_table);
	      }
	    }while(combo_counter_org != combo_counter);

	    float power[2];
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
	      if(color == MAINCOLOR){
		drop_pwr = num_drops_combo[i]==4 ? (1+0.25*(num_drops_combo[i]-3))*way : 1+0.25*(num_drops_combo[i]-3); 
		if(strong)
		  drop_pwr = drop_pwr * (1+0.06*num_drops_combo[i]);
		attack_m += drop_pwr; 
		if(isLine_combo[i]) num_line_m++;
	      }else{
		drop_pwr = num_drops_combo[i]==4 ? (1+0.25*(num_drops_combo[i]-3))*way : 1+0.25*(num_drops_combo[i]-3); 
		if(strong)
		  drop_pwr = drop_pwr * (1+0.06*num_drops_combo[i]);
		attack_s += drop_pwr; 
		if(isLine_combo[i]) num_line_s++;
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

	    if(MP[0][rank-1] < power[0]){
	      for(j = 0;j < rank;j++){
		if(MP[0][j] < power[0]){
		  for(k = rank-2;k >= j;k--){
		    MID[0][k+1] = MID[0][k];
		    MP[0][k+1] = MP[0][k];
		  }
		  MID[0][j] = tableID;
		  MP[0][j] = power[0];
		  break;
		}
	      }
	    }
	    if(MP[1][rank-1] < power[1]){
	      for(j = 0;j < rank;j++){
		if(MP[1][j] < power[1]){
		  for(k = rank-2;k >= j;k--){
		    MID[1][k+1] = MID[1][k];
		    MP[1][k+1] = MP[1][k];
		  }
		  MID[1][j] = (~tableID) & 0x000003FFFFFFFFFFLU;
		  MP[1][j] = power[1];
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
  for(ms = 0;ms < 2;ms++){
    for(i = 0;i < rank;i++){
      maxPower[id + step*i + ms*step*rank] = MP [ms][i];
      maxID   [id + step*i + ms*step*rank] = MID[ms][i];
    }
  }
}
#undef WID

#undef COMBO_LENGTH
#undef REVERSE_LENGTH

extern "C"
{

  void simulate_all_cuda(const int table_size, const int start, const int end, /*int *bit_count_table,*/ int *const reversed_bit_table, int *const tableID_half_table, int *const tableID_half_prefix, /*unsigned long long *const num_patterns,*/ int *const num_patterns_half, const int width, const int hight, const int combo_length, const int LS, const int isStrong, const int line, const int way, const int simuave){

    int rank = RANKINGLENGTH;
    int i, j, k;
    unsigned long long *max_powerID_dev;
    float *max_power_dev;
    int tsize = NUM_THREAD;
    //int gsize = ((num_patterns_omitted[num_attacks]-1)/128+1);
    int gsize = NUM_BLOCK;
    const int length = gsize*tsize*LOCALRANKINGLENGTH;
    unsigned long long max_powerID[2*length];
    float max_power[2*length];
    unsigned long long final_MID[43][rank];
    float final_MP[43][rank];
    int reverse_length = 1 << width;
    int *tableID_half_table_dev, *tableID_half_prefix_dev, *num_patterns_half_dev;
    //int *bit_count_table_dev, *reversed_bit_table_dev;
    int *reversed_bit_table_dev;
    const float pline = (float)line;
    const float pway = pow(1.5,way);
    const int half_table_size = width*hight/2;

    for(i = 0;i < 43;i++){
      final_MID[i][0] = 0xFFFFFFFFFFFFFFFFLU;
    }

    //CUDA_SAFE_CALL(cudaMalloc((void**)&bit_count_table_dev,     sizeof(int) * 256));
    CUDA_SAFE_CALL(cudaMalloc((void**)&reversed_bit_table_dev, sizeof(int) * reverse_length));
    CUDA_SAFE_CALL(cudaMalloc((void**)&tableID_half_table_dev, sizeof(int) * (1 << (width*hight/2))));
    CUDA_SAFE_CALL(cudaMalloc((void**)&num_patterns_half_dev,  sizeof(int) * (width*hight/2+1)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&tableID_half_prefix_dev,sizeof(int) * (width*hight/2+1)));
    //CUDA_SAFE_CALL(cudaMemcpy(bit_count_table_dev,    bit_count_table,     sizeof(int) * 256, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(reversed_bit_table_dev,reversed_bit_table, sizeof(int) * reverse_length, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(tableID_half_table_dev,tableID_half_table, sizeof(int) * (1 << (width*hight/2)), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(num_patterns_half_dev, num_patterns_half,  sizeof(int) * (width*hight/2+1), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(tableID_half_prefix_dev,tableID_half_prefix,sizeof(int) * (width*hight/2+1), cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMalloc((void**)&max_powerID_dev, sizeof(unsigned long long) * 2 * length));;
    CUDA_SAFE_CALL(cudaMalloc((void**)&max_power_dev,   sizeof(float) * 2 * length));;

    //fprintf(stdout,"%d\n",__LINE__);
    int num_attacks;
    for(num_attacks = start;num_attacks <= end;num_attacks++){
      if(half_table_size < num_attacks && num_attacks <= width*hight-start) break;
      printf("calculating %2d-%2d & %2d-%2d ...\n", num_attacks, width*hight-num_attacks, width*hight-num_attacks, num_attacks);
      //printf("%2d-%2d, line %d, way %d\n", num_attacks, width*hight-num_attacks, line, way);

      dim3 grid(gsize,1,1);
      dim3 block(tsize,1,1);
#ifdef TIME
      cudaDeviceSynchronize();
      double t1 = gettimeofday_sec();
#endif
      switch(table_size){
      case SMALL_TABLE:
	simulate_all_kernel_small<<<grid, block>>>(num_attacks, num_patterns_half_dev, max_powerID_dev, max_power_dev, pline, pway, tableID_half_prefix_dev, tableID_half_table_dev, reversed_bit_table_dev, /*bit_count_table_dev,*/ LS, isStrong);
	break;
      case NORMAL_TABLE:
	simulate_all_kernel_normal<<<grid, block>>>(num_attacks, num_patterns_half_dev, max_powerID_dev, max_power_dev, pline, pway, tableID_half_prefix_dev, tableID_half_table_dev, reversed_bit_table_dev, /*bit_count_table_dev,*/ LS, isStrong);
	break;
      case BIG_TABLE:
	if(num_attacks == 21){
	  simulate_all_kernel_big_21<<<grid, block>>>(num_patterns_half_dev, max_powerID_dev, max_power_dev, pline, pway, tableID_half_prefix_dev, tableID_half_table_dev, reversed_bit_table_dev, /*bit_count_table_dev,*/ LS, isStrong);
	}else{
	  simulate_all_kernel_big<<<grid, block>>>(num_attacks, num_patterns_half_dev, max_powerID_dev, max_power_dev, pline, pway, tableID_half_prefix_dev, tableID_half_table_dev, reversed_bit_table_dev, /*bit_count_table_dev,*/ LS, isStrong);
	}
	break;
      }
#ifdef TIME
      cudaDeviceSynchronize();
      double t2 = gettimeofday_sec();
      printf("num %d,time,%f\n",num_attacks,t2-t1);
#endif
      //fprintf(stdout,"%d\n",__LINE__);
      cudaMemcpy(max_powerID, max_powerID_dev, sizeof(unsigned long long) * 2 * length, cudaMemcpyDeviceToHost);
      cudaMemcpy(max_power  , max_power_dev  , sizeof(float) * 2 * length, cudaMemcpyDeviceToHost);

      //fprintf(stdout,"%d\n",__LINE__);
      float MP[2][rank];
      unsigned long long MID[2][rank];
      int ms;
      for(ms = 0; ms < 2; ms++){
	for(i = 0;i < rank;i++){
	  MP[ms][i] = 0.0;
	  MID[ms][i]= 0;
	}
	for(i = 0;i < length;i++){
	  float power = max_power[i + length*ms];
	  unsigned long long tableID = max_powerID[i + length*ms];
	  if(MP[ms][rank-1] < power){
	    for(k = 0;k < rank;k++){
	      if(MP[ms][k] < power){
		for(j = rank-2;j >= k;j--){
		  MID[ms][j+1] = MID[ms][j];
		  MP[ms][j+1] = MP[ms][j];
		}
		MID[ms][k] = tableID;
		MP[ms][k] = power;
		break;
	      }
	    }
	  }
	}
	//fprintf(stdout,"%d\n",__LINE__);
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
      }
      //fprintf(stdout,"%d\n",__LINE__);
      if(num_attacks == half_table_size){
	int mc = 0;
	int sc = 0;
	for(i = 0;i < rank;i++){
	  if(MP[0][mc] < MP[1][sc]){
	    final_MID[num_attacks][i] = MID[1][sc];
	    final_MP [num_attacks][i] = MP [1][sc];
	    sc++;
	  }else if(MP[0][mc] == MP[1][sc]){
	    if(MID[0][mc] < MID[1][sc]){
	      final_MID[num_attacks][i] = MID[0][mc];
	      final_MP [num_attacks][i] = MP [0][mc];
	      mc++;
	    }else{
	      final_MID[num_attacks][i] = MID[1][sc];
	      final_MP [num_attacks][i] = MP [1][sc];
	      sc++;
	    }
	  }else{
	    final_MID[num_attacks][i] = MID[0][mc];
	    final_MP [num_attacks][i] = MP [0][mc];
	    mc++;
	  }
	}
      }else{
	for(i = 0;i < rank;i++){
	  final_MID[num_attacks][i] = MID[0][i];
	  final_MP [num_attacks][i] = MP [0][i];
	  final_MID[width*hight-num_attacks][i] = MID[1][i];
	  final_MP [width*hight-num_attacks][i] = MP [1][i];
	}
      }
    }
    //fprintf(stdout,"%d\n",__LINE__);
    for(num_attacks = 0;num_attacks <= width*hight;num_attacks++){
      if(final_MID[num_attacks][0] != 0xFFFFFFFFFFFFFFFFLU){
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
    CUDA_SAFE_CALL(cudaFree(max_powerID_dev));
    CUDA_SAFE_CALL(cudaFree(max_power_dev));
    CUDA_SAFE_CALL(cudaFree(tableID_half_table_dev));
    //CUDA_SAFE_CALL(cudaFree(num_patterns_half_dev));
    CUDA_SAFE_CALL(cudaFree(tableID_half_prefix_dev));
    //CUDA_SAFE_CALL(cudaFree(bit_count_table_dev));
    CUDA_SAFE_CALL(cudaFree(reversed_bit_table_dev));

  }

}
