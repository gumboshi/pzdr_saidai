#ifdef __cplusplus
extern "C"{
#endif
#ifdef CUDA
  void simulate_all_cuda(void (*gt)(long long, long long*), int (*os)(long long*, int*, int*, int*, int, int), int table_size, int start, int end, int *bit_hash_table, int *reversed_bit_table, int *tableID_half_table, int *tableID_half_prefix, long long *num_patterns, int *num_patterns_half, int width, int hight, int combo_length, int LS, int isStrong, int line, int way, int simuave);
#endif
#ifdef __cplusplus
}
#endif
