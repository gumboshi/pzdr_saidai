#ifdef __cplusplus
extern "C"{
#endif
  void simulate_average(void (*gt)(long long, long long*), int (*os)(long long*, int*, int*, int*, int, int), long long *MID, float *MP, int num_attacks, int width, int hight, int LS, int isStrong, float line, float way);
#ifdef __cplusplus
}
#endif
