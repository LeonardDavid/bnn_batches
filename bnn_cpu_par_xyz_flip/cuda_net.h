#include <stdio.h>
#include "cuda_kernel.h"

float layer1_conv(unsigned char * const x, float * layer){
  return layer1_conv_cuda(x, layer);
}

float layer2_maxpool(float * layer1, float * layer2){
  return layer2_maxpool_cuda(layer1, layer2);
}

float layer4_conv(unsigned long long * layer1, signed short * layer2){
  return layer4_conv_cuda(layer1, layer2);
}

float layer5_maxpool(signed short * layer1, signed short * layer2){
  return layer5_maxpool_cuda(layer1, layer2);
}

float layer8_gemm(unsigned long long * layer1, signed short * layer2){
  return layer8_gemm_cuda(layer1, layer2);
}

float layer10_gemm(unsigned long long * layer1, signed short * layer2){
  return layer10_gemm_cuda(layer1, layer2);
}