#include <stdio.h>
#include "cuda_kernel.h"

float layer1_conv(unsigned char x[][32][32][3], float * layer){ // unsigned char * const x / unsigned char x[][32][32][3]
  return layer1_conv_cuda(x, layer);
}

float layer3_conv(unsigned long long * layer1, float * layer2){
  return layer3_conv_cuda(layer1, layer2);
}

float layer4_maxpool(float * layer1, float * layer2){
  return layer4_maxpool_cuda(layer1, layer2);
}

float layer6_conv(unsigned long long * layer1, float * layer2){
  return layer6_conv_cuda(layer1, layer2);
}

float layer8_conv(unsigned long long * layer1, float * layer2){
  return layer8_conv_cuda(layer1, layer2);
}

float layer9_maxpool(float * layer1, float * layer2){
  return layer9_maxpool_cuda(layer1, layer2);
}

float layer11_conv(unsigned long long * layer1, float * layer2){
  return layer11_conv_cuda(layer1, layer2);
}

float layer13_conv(unsigned long long * layer1, float * layer2){
  return layer13_conv_cuda(layer1, layer2);
}

float layer14_maxpool(float * layer1, float * layer2){
  return layer14_maxpool_cuda(layer1, layer2);
}

float layer17_gemm(unsigned long long * layer1, float * layer2){
  return layer17_gemm_cuda(layer1, layer2);
}

float layer19_gemm(unsigned long long * layer1, float * layer2){
  return layer19_gemm_cuda(layer1, layer2);
}
