#include "utils.h"

float layer1_conv_cuda(unsigned char x[][32][32][3], float * layer); // unsigned char * const x / unsigned char x[][32][32][3]

float layer3_conv_cuda(unsigned long long * layer1, float * layer2);

float layer4_maxpool_cuda(float * layer1, float * layer2);

float layer6_conv_cuda(unsigned long long * layer1, float * layer2);

float layer8_conv_cuda(unsigned long long * layer1, float * layer2);

float layer9_maxpool_cuda(float * layer1, float * layer2);

float layer11_conv_cuda(unsigned long long * layer1, float * layer2);

float layer13_conv_cuda(unsigned long long * layer1, float * layer2);

float layer14_maxpool_cuda(float * layer1, float * layer2);

float layer17_gemm_cuda(unsigned long long * layer1, float * layer2);

float layer19_gemm_cuda(unsigned long long * layer1, float * layer2);