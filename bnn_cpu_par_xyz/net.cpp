#include <iostream>
#include <chrono>
#include <tuple>

#include "cuda_net.h"
#include "netW.hpp"

using namespace std;

float predict_NeuralNet(unsigned char * const x, float * output) { 
  // possibly not valid c++ code:
  // unsigned char (*layer_0_output)[28][1] = (unsigned char (*)[28][1]) x;

  /* 
   * add all kernel_times from each GPU executed layer 
   * kernel only! Memory allocation and copy are executed on the CPU before GPU begins kernel execution
   */
  float kernel_time = 0;

  kernel_time += layer1_conv(x, cuda_layer_1_output);
  kernel_time += layer2_maxpool(cuda_layer_1_output, cuda_layer_2_output);

  // Layer 3 not parallelizable
  for(int b=0;b<BATCH_SIZE;b++){
    for (int h = 0; h < 14; h++) {
      for (int w = 0; w < 14; w++) {
          for (int c = 0; c < 64; c++) {
          if (cuda_layer_2_output[index4D(b,h,w,c,14,14,64)] > layer_3_threshold[c]) {
            layer_3_output[b][h][w][c / 64] |= (1ULL << (63 - c % 64));
          } else {
            layer_3_output[b][h][w][c / 64] &= ~(1ULL << (63 - c % 64));
          }
        }
      // cout<<endl;
      }
    }
    // cout<<endl;
  }

  // flatten layer_3_output into cuda_layer_3_output for further usage
  for(int i=0;i<14;i++){
    for(int j=0;j<14;j++){
      for(int b=0;b<BATCH_SIZE;b++){
        for(int k=0;k<64;k++){
          cuda_layer_3_output[index4D(b,i,j,k,14,14,64)] = layer_3_output[b][i][j][k];
        }
      }
    }
  }

  /* the method below for flattening does not lead to the correct result */
  // unsigned long long *cuda_layer_3_output = (unsigned long long *) layer_3_output;

  kernel_time += layer4_conv(cuda_layer_3_output, cuda_layer_4_output);
  kernel_time += layer5_maxpool(cuda_layer_4_output, cuda_layer_5_output);

  // Layer 6 not parallelizable
  for(int b=0;b<BATCH_SIZE;b++){
    for (int h = 0; h < 7; h++) {
      for (int w = 0; w < 7; w++) {
        for (int c = 0; c < 64; c++) {
          if (cuda_layer_5_output[index4D(b,h,w,c,7,7,64)] > layer_6_threshold[c]) {
            layer_6_output[b][h][w][c / 64] |= (1ULL << (63 - c % 64));
          } else {
            layer_6_output[b][h][w][c / 64] &= ~(1ULL << (63 - c % 64));
          }
        }
      }
    }
  }

    /* 
     * flatten layer_6_output into cuda_layer_6_output for further usage 
     * in the last step-layer (before the actual flattening), 
     * the direct pointer flattening works as intended, so the for-loops are no necessary
     */
    // for(int i=0;i<7;i++){
    //   for(int j=0;j<7;j++){
    //     for(int k=0;k<64;k++){
    //       cuda_layer_6_output[index3D(i,j,k,7,64)] = layer_6_output[i][j][k];
    //     }
    //   }
    // }

  // Layer 7 is flattening layer -> cuda_layer_6_output skipped
  unsigned long long *layer_7_output = (unsigned long long *) layer_6_output; // size = 49
  
  kernel_time += layer8_gemm(layer_7_output, cuda_layer_8_output);

  // Layer 9 not parallelizable
  for(int b=0;b<BATCH_SIZE;b++){
    for (int d = 0; d < 2048; d++) {
      if (cuda_layer_8_output[b*2048 + d] > layer_9_threshold[d]) {
        layer_9_output[b][d / 64] |= (1ULL << (63 - d % 64));
      } else {
        layer_9_output[b][d / 64] &= ~(1ULL << (63 - d % 64));
      }
    }
  }

  unsigned long long *cuda_layer_9_output = (unsigned long long *) layer_9_output;
  
  // worth it for 10 iterations? not really -> see profiling
  kernel_time += layer10_gemm(cuda_layer_9_output, cuda_layer_10_output);

  for(int b=0;b<BATCH_SIZE;b++){
    for (int i = 0; i < 10; i++) {
      output[b*10 + i] += cuda_layer_10_output[b*10 + i];
    }
  }

  return kernel_time;

}
