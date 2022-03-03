#include "utils.h"

float layer1_conv_cuda(unsigned char * const x, float * layer);

float layer2_maxpool_cuda(float * layer1, float * layer2);

float layer4_conv_cuda(unsigned long long * layer1, signed short * layer2);

float layer5_maxpool_cuda(signed short * layer1, signed short * layer2);

float layer8_gemm_cuda(unsigned long long * layer1, signed short * layer2);

float layer10_gemm_cuda(unsigned long long * layer1, signed short * layer2);
