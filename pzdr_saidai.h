#ifdef __cplusplus
extern "C"{
#endif
  //void init_combo_info(int *color_combo, int *num_drops_combo, int *isLine_combo, int combo_length);
  void init_bit_count_table(int *bit_count_table);
  void init_reversed_bit_table(int *reversed_bit_table, const int width);
  inline void generate_table_small(const unsigned long long tableID, unsigned long long *color_table);
  inline void generate_table_normal(const unsigned long long tableID, unsigned long long *color_table);
  inline void generate_table_big(const unsigned long long tableID, unsigned long long *color_table);
  //void simulate_all(void (*const gt)(long long, long long*), int (*const os)(long long*, int*, int*, int*, int, int), const int start, const int end, int * const bit_count_table, int * const reversed_bit_table, int *const tableID_half_table, int *const tableID_half_prefix, unsigned long long *const num_patterns, int *const num_patterns_half, const int width, const int hight, const int combo_length, const int LS, const int isStrong, const int line, const int way, const int simulate_ave);
  void simulate_all(const int table_size, const int start, const int end, /*int * const bit_count_table, */int * const reversed_bit_table, int *const tableID_half_table, int *const tableID_half_prefix, unsigned long long *const num_patterns, int *const num_patterns_half, const int width, const int hight, const int combo_length, const int LS, const int isStrong, const int line, const int way, const int simulate_ave);
  inline int one_step_small(unsigned long long *color_table, int *color_combo, int *num_drops_combo, int *isLine_combo, const int finish, const int num_colors);
  inline int one_step_normal(unsigned long long *color_table, int *color_combo, int *num_drops_combo, int *isLine_combo, const int finish, const int num_colors);
  inline int one_step_big(unsigned long long *color_table, int *color_combo, int *num_drops_combo, int *isLine_combo, const int finish, const int num_colors);
  float return_attack(const int combo_counter, int *const color_combo, int *const num_drops_combo, int *const isLine_combo, const int LS, const int isStrong, const float line, const float way);
  void return_attack_double(float *power, const int combo_counter, int *const color_combo, int *const num_drops_combo, int *const isLine_combo, const int LS, const int strong, const float line, const float way);
  void create_half_tableID(int *tableID_half_table, int *tableID_half_prefix, int *const bit_count_table, int *const num_patterns, const int half_table_size);
  void print_table(const unsigned long long *color_table, const int width, const int hight);
  void print_table2(const unsigned long long color_table, const int width, const int hight);
  void ID2table(const unsigned long long color_table, const int width, const int hight);
  void fill_random(unsigned long long *color_table, const int width, const int hight, /*struct drand48_data drand_buf*/unsigned int *seed);
  /*   void simulate_average(void (*const gt)(long long, long long*), int (*const os)(long long*, int*, int*, int*, int, int), long long *MID, float *MP, const int num_attacks, const int width, const int hight, const int LS, const int isStrong, const float line, const float way); */
  void simulate_average(const int table_size, unsigned long long * const MID, float * const MP, const int num_attacks, const int width, const int hight, const int LS, const int isStrong, const float line, const float way);

#ifdef CUDA
  void simulate_all_cuda(const int table_size, const int start, const int end, /*int *bit_count_table,*/ int *const reversed_bit_table, int *const tableID_half_table, int *const tableID_half_prefix, /*long long *const num_patterns,*/ int *const num_patterns_half, const int width, const int hight, const int combo_length, const int LS, const int isStrong, const int line, const int way, const int simuave);
#endif
#ifdef __cplusplus
}
#endif
