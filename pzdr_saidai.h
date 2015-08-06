#ifdef __cplusplus
extern "C"{
#endif
/*   void simulate_average(void (*const gt)(long long, long long*), int (*const os)(long long*, int*, int*, int*, int, int), long long *MID, float *MP, const int num_attacks, const int width, const int hight, const int LS, const int isStrong, const float line, const float way); */
  void simulate_average(const int table_size, const int num_attacks, const int width, const int hight, const int LS, const int isStrong, const float line, const float way);
#ifdef __cplusplus
}
#endif
