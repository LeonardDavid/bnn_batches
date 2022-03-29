#include <iostream>
#include <chrono>
#include <tuple>

#include "cuda_net.h"
#include "netW.hpp"

using namespace std;

std::tuple<float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float>  
predict_NeuralNet(unsigned char * const x, float * output) {
  // possibly not valid c++ code:
  // unsigned char (*layer_0_output)[28][1] = (unsigned char (*)[28][1]) x;

  // add all kernel_time s
  float kernel_time = 0, malloc_time = 0, cpy_time = 0;
  /* Layer 1 GPU */
  auto start = std::chrono::high_resolution_clock::now();

    // float a,b,c;
    // std::tie(a,b,c) = layer1_conv(x, cuda_layer_1_output);
    // kernel_time += a; malloc_time += b, cpy_time += c;

  kernel_time += layer1_conv(x, cuda_layer_1_output);
  auto end = std::chrono::high_resolution_clock::now();
  auto l1_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());
  float l1_kernel_time = kernel_time;
  l1_time -= l1_kernel_time*1000000.0f; // ms->ns

  /* Layer 1 CPU */
  // // initialize layer_0_output where x is the input image
  // unsigned char (*layer_0_output)[BATCH_SIZE][28][1] = (unsigned char (*)[BATCH_SIZE][28][1]) x;

  // // flatten layer_0_output
  // unsigned char *cuda_layer_0_output = (unsigned char *) layer_0_output;

  // auto start = std::chrono::high_resolution_clock::now();
  // // Layer 1: Conv @ cpp.NHWC {% else %} /{% if pads == [0, 0, 0, 0] %}
  // for(int b = 0; b < BATCH_SIZE; b++){
  //   for (int h = 0; h < 28; h++) {
  //     for (int w = 0; w < 28; w++) {
  //       for (int m = 0; m < 64; m++) {
  //         cuda_layer_1_output[index4D(b,h,w,m,28,28,64)] = layer_1_bias[m];
  //       }
  //       for (int kH = 0; kH < 3; kH++) {
  //         int iH = h * 1 + kH - 1;
  //         if (iH >= 0 && iH < 28) {
  //           for (int kW = 0; kW < 3; kW++) {
  //             int iW = w * 1 + kW - 1;
  //             if (iW >= 0 && iW < 28) {
  //               for (int c = 0; c < 1; c++) {
  //                 for (int m = 0; m < 64; m++) {
  //                   cuda_layer_1_output[index4D(b,h,w,m,28,28,64)] += layer_1_weight[kH][kW][c][m] * cuda_layer_0_output[index4D(b,iH,iW,c,28,28,1)];
  //                 }
  //               }
  //             }
  //           }
  //         }
  //       }
  //     }
  //   }
  // }
  // auto end = std::chrono::high_resolution_clock::now();
  // auto l1_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());
  // float l1_kernel_time = 0;

  /* Layer 2 GPU */
  start = std::chrono::high_resolution_clock::now();
  kernel_time += layer2_maxpool(cuda_layer_1_output, cuda_layer_2_output);
  end = std::chrono::high_resolution_clock::now();  
  auto l2_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());
  float l2_kernel_time = kernel_time-l1_kernel_time;
  l2_time -= l2_kernel_time*1000000.0f; // ms->ns

  /* Layer 2 CPU */
  // start = std::chrono::high_resolution_clock::now();
  // // Layer 2: MaxPool @ cpp.NHWC {% if pads == [0, 0, 0, 0] %}
  // for(int b = 0; b < BATCH_SIZE; b++){
  //   for (int h = 0; h < 14; h++) {
  //     for (int w = 0; w < 14; w++) {
  //       for (int c = 0; c < 64; c++) {
  //         cuda_layer_2_output[index4D(b,h,w,c,14,14,64)] = std::numeric_limits<float>::lowest();
  //       }
  //       for (int kH = 0; kH < 2; kH++) {
  //         for (int kW = 0; kW < 2; kW++) {
  //           for (int c = 0; c < 64; c++) {
  //             cuda_layer_2_output[index4D(b,h,w,c,14,14,64)] = std::max(cuda_layer_1_output[index4D(b,(h * 2 + kH),(w * 2 + kW),c,28,28,64)], cuda_layer_2_output[index4D(b,h,w,c,14,14,64)]);
  //           }
  //         }
  //       }
  //     }
  //   }
  // }
  // end = std::chrono::high_resolution_clock::now();  
  // auto l2_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());  
  // float l2_kernel_time = 0;


  // layer3_step(cuda_layer_2_output, cuda_layer_3_output);
  /* Layer 3 CPU */
  start = std::chrono::high_resolution_clock::now();
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
  end = std::chrono::high_resolution_clock::now();
  auto l3_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());

  /* Layer 4 GPU */
  start = std::chrono::high_resolution_clock::now();
  kernel_time += layer4_conv(cuda_layer_3_output, cuda_layer_4_output);
  end = std::chrono::high_resolution_clock::now();    
  auto l4_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());  
  float l4_kernel_time = kernel_time-(l1_kernel_time+l2_kernel_time);
  l4_time -= l4_kernel_time*1000000.0f; // ms->ns

  /* Layer 4 CPU */
  // start = std::chrono::high_resolution_clock::now();
  // // Layer 4: Conv @ cpp.binary {% else %} /{% if layer.pads == [0, 0, 0, 0] %}
  // for(int b = 0; b < BATCH_SIZE; b++){
  //   for (int h = 0; h < 14; h++) {
  //     for (int w = 0; w < 14; w++) {
  //       for (int m = 0; m < 64; m++) {
  //         cuda_layer_4_output[index4D(b,h,w,m,14,14,64)] = layer_4_bias[m];
  //       }
  //       for (int kH = 0; kH < 3; kH++) {
  //         int iH = h * 1 + kH - 1;
  //         if (iH >= 0 && iH < 14) {
  //           for (int kW = 0; kW < 3; kW++) {
  //             int iW = w * 1 + kW - 1;
  //             if (iW >= 0 && iW < 14) {
  //               for (int m = 0; m < 64; m++) {
  //                 for (int c = 0; c < 1; c++) {
  //                   cuda_layer_4_output[index4D(b,h,w,m,14,14,64)] += 2 * __builtin_popcountll((unsigned long long)~(unsigned long long)(layer_4_weight[kH][kW][m][c] ^ cuda_layer_3_output[index4D(b,iH,iW,c,14,14,64)])) - 64;
  //                 }
  //               }
  //             }
  //           }
  //         }
  //       }
  //     }
  //   }
  // }
  // end = std::chrono::high_resolution_clock::now();    
  // auto l4_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());  
  // float l4_kernel_time = 0;

  /* Layer 5 GPU */
  start = std::chrono::high_resolution_clock::now();
  kernel_time += layer5_maxpool(cuda_layer_4_output, cuda_layer_5_output);
  end = std::chrono::high_resolution_clock::now();    
  auto l5_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());
  float l5_kernel_time = kernel_time-(l1_kernel_time+l2_kernel_time+l4_kernel_time);
  l5_time -= l5_kernel_time*1000000.0f; // ms->ns

  /* Layer 5 CPU */
  // start = std::chrono::high_resolution_clock::now();
  // // Layer 5: MaxPool @ cpp.NHWC {% if pads == [0, 0, 0, 0] %}
  // for(int b = 0; b < BATCH_SIZE; b++){
  //   for (int h = 0; h < 7; h++) {
  //     for (int w = 0; w < 7; w++) {
  //       for (int c = 0; c < 64; c++) {
  //         cuda_layer_5_output[index4D(b,h,w,c,7,7,64)] = std::numeric_limits<signed short>::lowest();
  //       }
  //       for (int kH = 0; kH < 2; kH++) {
  //         for (int kW = 0; kW < 2; kW++) {
  //           for (int c = 0; c < 64; c++) {
  //             cuda_layer_5_output[index4D(b,h,w,c,7,7,64)] = std::max(cuda_layer_4_output[index4D(b,(h * 2 + kH),(w * 2 + kW),c,14,14,64)], cuda_layer_5_output[index4D(b,h,w,c,7,7,64)]);
  //           }
  //         }
  //       }
  //     }
  //   }
  // }
  // end = std::chrono::high_resolution_clock::now();    
  // auto l5_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());  
  // float l5_kernel_time = 0;

  // layer6_step(cuda_layer_5_output, cuda_layer_6_output);
  /* Layer 6 CPU */
  start = std::chrono::high_resolution_clock::now();
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
  end = std::chrono::high_resolution_clock::now();
  auto l6_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());

    // flatten layer_6_output into cuda_layer_6_output for further usage
    // for(int i=0;i<7;i++){
    //   for(int j=0;j<7;j++){
    //     for(int k=0;k<64;k++){
    //       cuda_layer_6_output[index3D(i,j,k,7,64)] = layer_6_output[i][j][k];
    //     }
    //   }
    // }

  // Layer 7 is flattening layer -> cuda_layer_6_output skipped
  unsigned long long *layer_7_output = (unsigned long long *) layer_6_output; // size = 49
  
  /* Layer 8 GPU */
  start = std::chrono::high_resolution_clock::now();
  kernel_time += layer8_gemm(layer_7_output, cuda_layer_8_output);
  end = std::chrono::high_resolution_clock::now();  
  auto l8_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());
  float l8_kernel_time = kernel_time-(l1_kernel_time+l2_kernel_time+l4_kernel_time+l5_kernel_time);
  l8_time -= l8_kernel_time*1000000.0f; // ms->ns

  /* Layer 8 CPU */
  start = std::chrono::high_resolution_clock::now();
  // Layer 8: Gemm @ cpp.binary 
  // for(int b = 0; b < BATCH_SIZE; b++){
  //   for (int d = 0; d < 2048; d++) {
  //     cuda_layer_8_output[b*2048 + d] = layer_8_bias[d];
  //   }
  //   for (int d = 0; d < 2048; d++) {
  //     for (int i = 0; i < 49; i++) {
  //       cuda_layer_8_output[b*2048 + d] += 2 * __builtin_popcountll((unsigned long long)~(unsigned long long)(layer_8_weight[d][i] ^ layer_7_output[b*49+i])) - 64;
  //     }
  //   }
  // }
  // end = std::chrono::high_resolution_clock::now();  
  // auto l8_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());
  // float l8_kernel_time = 0;

  // layer9_step(cuda_layer_8_output, cuda_layer_9_output);
  /* Layer 9 CPU */
  start = std::chrono::high_resolution_clock::now();
  for(int b=0;b<BATCH_SIZE;b++){
    for (int d = 0; d < 2048; d++) {
      if (cuda_layer_8_output[b*2048 + d] > layer_9_threshold[d]) {
        layer_9_output[b][d / 64] |= (1ULL << (63 - d % 64));
      } else {
        layer_9_output[b][d / 64] &= ~(1ULL << (63 - d % 64));
      }
    }
  }
  end = std::chrono::high_resolution_clock::now();
  auto l9_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());

  unsigned long long *cuda_layer_9_output = (unsigned long long *) layer_9_output;
  
  /* Layer 10 GPU */
  start = std::chrono::high_resolution_clock::now();
  kernel_time += layer10_gemm(cuda_layer_9_output, cuda_layer_10_output);
  end = std::chrono::high_resolution_clock::now();    
  auto l10_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());  
  float l10_kernel_time = kernel_time-(l1_kernel_time+l2_kernel_time+l4_kernel_time+l5_kernel_time+l8_kernel_time);
  l10_time -= l10_kernel_time*1000000.0f; // ms->ns

  /* Layer 10 CPU */
  // start = std::chrono::high_resolution_clock::now();
  // // Layer 10: Gemm @ cpp.binary
  // for(int b = 0; b < BATCH_SIZE; b++){
  //   for (int d = 0; d < 10; d++) {
  //     cuda_layer_10_output[b*10 + d] = layer_10_bias[d];
  //   }
  //   for (int d = 0; d < 10; d++) {
  //     for (int i = 0; i < 32; i++) {
  //       cuda_layer_10_output[b*10 + d] += 2 * __builtin_popcountll((unsigned long long)~(unsigned long long)(layer_10_weight[d][i] ^ cuda_layer_9_output[b*32+i])) - 64;
  //     }
  //   }
  // }
  // end = std::chrono::high_resolution_clock::now();    
  // auto l10_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());  
  // float l10_kernel_time = 0;

  for(int b=0;b<BATCH_SIZE;b++){
    for (int i = 0; i < 10; i++) {
      output[b*10 + i] += cuda_layer_10_output[b*10 + i];
    }
  }

  return make_tuple(kernel_time, l1_time, l2_time, l3_time, l4_time, l5_time, l6_time, l8_time, l9_time, l10_time, 
    l1_kernel_time, l2_kernel_time, l4_kernel_time, l5_kernel_time, l8_kernel_time, l10_kernel_time, malloc_time, cpy_time);
  // return kernel_time;

}

// ==============================================================================================================================================

// no flat arrays

// std::tuple<float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float>  
// predict_NeuralNet(unsigned char * const x, float * output) {
//   // possibly not valid c++ code:
//   // unsigned char (*layer_0_output)[28][1] = (unsigned char (*)[28][1]) x;

//   // add all kernel_time s
//   float kernel_time = 0;
//   // auto start = std::chrono::high_resolution_clock::now();
//   // kernel_time += layer1_conv(x, cuda_layer_1_output);
//   // auto end = std::chrono::high_resolution_clock::now();
//   // auto l1_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());
//   // float l1_kernel_time = kernel_time;
//   // l1_time -= l1_kernel_time*1000000.0f; // ms->ns

//   // initialize layer_0_output where x is the input image
//   unsigned char (*layer_0_output)[BATCH_SIZE][28][1] = (unsigned char (*)[BATCH_SIZE][28][1]) x;

//   // flatten layer_0_output
//   // unsigned char *cuda_layer_0_output = (unsigned char *) layer_0_output;

//   auto start = std::chrono::high_resolution_clock::now();
//   // Layer 1: Conv @ cpp.NHWC {% else %} /{% if pads == [0, 0, 0, 0] %}
//   for(int b = 0; b < BATCH_SIZE; b++){
//     for (int h = 0; h < 28; h++) {
//       for (int w = 0; w < 28; w++) {
//         for (int m = 0; m < 64; m++) {
//           layer_1_output[b][h][w][m] = layer_1_bias[m];
//         }
//         for (int kH = 0; kH < 3; kH++) {
//           int iH = h * 1 + kH - 1;
//           if (iH >= 0 && iH < 28) {
//             for (int kW = 0; kW < 3; kW++) {
//               int iW = w * 1 + kW - 1;
//               if (iW >= 0 && iW < 28) {
//                 for (int c = 0; c < 1; c++) {
//                   for (int m = 0; m < 64; m++) {
//                     layer_1_output[b][h][w][m] += layer_1_weight[kH][kW][c][m] * layer_0_output[b][iH][iW][c];
//                   }
//                 }
//               }
//             }
//           }
//         }
//       }
//     }
//   }
//   auto end = std::chrono::high_resolution_clock::now();
//   auto l1_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());
//   float l1_kernel_time = 0;

//   // start = std::chrono::high_resolution_clock::now();
//   // kernel_time += layer2_maxpool(cuda_layer_1_output, cuda_layer_2_output);
//   // end = std::chrono::high_resolution_clock::now();  
//   // auto l2_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());
//   // float l2_kernel_time = kernel_time-l1_kernel_time;
//   // l2_time -= l2_kernel_time*1000000.0f; // ms->ns

//   start = std::chrono::high_resolution_clock::now();
//   // Layer 2: MaxPool @ cpp.NHWC {% if pads == [0, 0, 0, 0] %}
//   for(int b = 0; b < BATCH_SIZE; b++){
//     for (int h = 0; h < 14; h++) {
//       for (int w = 0; w < 14; w++) {
//         for (int c = 0; c < 64; c++) {
//           layer_2_output[b][h][w][c] = std::numeric_limits<float>::lowest();
//         }
//         for (int kH = 0; kH < 2; kH++) {
//           for (int kW = 0; kW < 2; kW++) {
//             for (int c = 0; c < 64; c++) {
//               layer_2_output[b][h][w][c] = std::max(layer_1_output[b][h * 2 + kH][w * 2 + kW][c], layer_2_output[b][h][w][c]);
//             }
//           }
//         }
//       }
//     }
//   }
//   end = std::chrono::high_resolution_clock::now();  
//   auto l2_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());  
//   float l2_kernel_time = 0;


//   // layer3_step(cuda_layer_2_output, cuda_layer_3_output);

//   start = std::chrono::high_resolution_clock::now();
//   for(int b=0;b<BATCH_SIZE;b++){
//     for (int h = 0; h < 14; h++) {
//       for (int w = 0; w < 14; w++) {
//           for (int c = 0; c < 64; c++) {
//           if (layer_2_output[b][h][w][c] > layer_3_threshold[c]) {
//             layer_3_output[b][h][w][c / 64] |= (1ULL << (63 - c % 64));
//           } else {
//             layer_3_output[b][h][w][c / 64] &= ~(1ULL << (63 - c % 64));
//           }
//         }
//       // cout<<endl;
//       }
//     }
//     // cout<<endl;
//   }

//   // flatten layer_3_output into cuda_layer_3_output for further usage
//   // for(int i=0;i<14;i++){
//   //   for(int j=0;j<14;j++){
//   //     for(int b=0;b<BATCH_SIZE;b++){
//   //       for(int k=0;k<64;k++){
//   //         cuda_layer_3_output[index4D(b,i,j,k,14,14,64)] = layer_3_output[b][i][j][k];
//   //       }
//   //     }
//   //   }
//   // }
//   end = std::chrono::high_resolution_clock::now();
//   auto l3_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());

//   // start = std::chrono::high_resolution_clock::now();
//   // kernel_time += layer4_conv(cuda_layer_3_output, cuda_layer_4_output);
//   // end = std::chrono::high_resolution_clock::now();    
//   // auto l4_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());  
//   // float l4_kernel_time = kernel_time-(l1_kernel_time+l2_kernel_time);
//   // l4_time -= l4_kernel_time*1000000.0f; // ms->ns

//   start = std::chrono::high_resolution_clock::now();
//   // Layer 4: Conv @ cpp.binary {% else %} /{% if layer.pads == [0, 0, 0, 0] %}
//   for(int b = 0; b < BATCH_SIZE; b++){
//     for (int h = 0; h < 14; h++) {
//       for (int w = 0; w < 14; w++) {
//         for (int m = 0; m < 64; m++) {
//           layer_4_output[b][h][w][m] = layer_4_bias[m];
//         }
//         for (int kH = 0; kH < 3; kH++) {
//           int iH = h * 1 + kH - 1;
//           if (iH >= 0 && iH < 14) {
//             for (int kW = 0; kW < 3; kW++) {
//               int iW = w * 1 + kW - 1;
//               if (iW >= 0 && iW < 14) {
//                 for (int m = 0; m < 64; m++) {
//                   for (int c = 0; c < 1; c++) {
//                     layer_4_output[b][h][w][m] += 2 * __builtin_popcountll((unsigned long long)~(unsigned long long)(layer_4_weight[kH][kW][m][c] ^ layer_3_output[b][iH][iW][c])) - 64;
//                   }
//                 }
//               }
//             }
//           }
//         }
//       }
//     }
//   }
//   end = std::chrono::high_resolution_clock::now();    
//   auto l4_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());  
//   float l4_kernel_time = 0;

//   // start = std::chrono::high_resolution_clock::now();
//   // kernel_time += layer5_maxpool(cuda_layer_4_output, cuda_layer_5_output);
//   // end = std::chrono::high_resolution_clock::now();    
//   // auto l5_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());
//   // float l5_kernel_time = kernel_time-(l1_kernel_time+l2_kernel_time+l4_kernel_time);
//   // l5_time -= l5_kernel_time*1000000.0f; // ms->ns

//   start = std::chrono::high_resolution_clock::now();
//   // Layer 5: MaxPool @ cpp.NHWC {% if pads == [0, 0, 0, 0] %}
//   for(int b = 0; b < BATCH_SIZE; b++){
//     for (int h = 0; h < 7; h++) {
//       for (int w = 0; w < 7; w++) {
//         for (int c = 0; c < 64; c++) {
//           layer_5_output[b][h][w][c] = std::numeric_limits<signed short>::lowest();
//         }
//         for (int kH = 0; kH < 2; kH++) {
//           for (int kW = 0; kW < 2; kW++) {
//             for (int c = 0; c < 64; c++) {
//               layer_5_output[b][h][w][c] = std::max(layer_4_output[b][h * 2 + kH][w * 2 + kW][c], layer_5_output[b][h][w][c]);
//             }
//           }
//         }
//       }
//     }
//   }
//   end = std::chrono::high_resolution_clock::now();    
//   auto l5_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());  
//   float l5_kernel_time = 0;

//   // layer6_step(cuda_layer_5_output, cuda_layer_6_output);
//   /*
//     Same as layer3_step
//   */
//   start = std::chrono::high_resolution_clock::now();
//   for(int b=0;b<BATCH_SIZE;b++){
//     for (int h = 0; h < 7; h++) {
//       for (int w = 0; w < 7; w++) {
//         for (int c = 0; c < 64; c++) {
//           if (layer_5_output[b][h][w][c] > layer_6_threshold[c]) {
//             layer_6_output[b][h][w][c / 64] |= (1ULL << (63 - c % 64));
//           } else {
//             layer_6_output[b][h][w][c / 64] &= ~(1ULL << (63 - c % 64));
//           }
//         }
//       }
//     }
//   }
//   end = std::chrono::high_resolution_clock::now();
//   auto l6_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());

//     // flatten layer_6_output into cuda_layer_6_output for further usage
//     // for(int i=0;i<7;i++){
//     //   for(int j=0;j<7;j++){
//     //     for(int k=0;k<64;k++){
//     //       cuda_layer_6_output[index3D(i,j,k,7,64)] = layer_6_output[i][j][k];
//     //     }
//     //   }
//     // }

//   // Layer 7 is flattening layer -> cuda_layer_6_output skipped
//   unsigned long long *layer_7_output = (unsigned long long *) layer_6_output; // size = 49
  
//   // start = std::chrono::high_resolution_clock::now();
//   // kernel_time += layer8_gemm(layer_7_output, cuda_layer_8_output);
//   // end = std::chrono::high_resolution_clock::now();  
//   // auto l8_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());
//   // float l8_kernel_time = kernel_time-(l1_kernel_time+l2_kernel_time+l4_kernel_time+l5_kernel_time);
//   // l8_time -= l8_kernel_time*1000000.0f; // ms->ns

//   start = std::chrono::high_resolution_clock::now();
//   // Layer 8: Gemm @ cpp.binary 
//   for(int b = 0; b < BATCH_SIZE; b++){
//     for (int d = 0; d < 2048; d++) {
//       layer_8_output[b*2048 + d] = layer_8_bias[d];
//     }
//     for (int d = 0; d < 2048; d++) {
//       for (int i = 0; i < 49; i++) {
//         layer_8_output[b*2048 + d] += 2 * __builtin_popcountll((unsigned long long)~(unsigned long long)(layer_8_weight[d][i] ^ layer_7_output[b*49+i])) - 64;
//       }
//     }
//   }
//   end = std::chrono::high_resolution_clock::now();  
//   auto l8_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());
//   float l8_kernel_time = 0;

//   // layer9_step(cuda_layer_8_output, cuda_layer_9_output);
//   /*
//     Same as layer9_step
//   */
//   start = std::chrono::high_resolution_clock::now();
//   for(int b=0;b<BATCH_SIZE;b++){
//     for (int d = 0; d < 2048; d++) {
//       if (layer_8_output[b*2048 + d] > layer_9_threshold[d]) {
//         layer_9_output[b][d / 64] |= (1ULL << (63 - d % 64));
//       } else {
//         layer_9_output[b][d / 64] &= ~(1ULL << (63 - d % 64));
//       }
//     }
//   }
//   end = std::chrono::high_resolution_clock::now();
//   auto l9_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());

//   // unsigned long long *cuda_layer_9_output = (unsigned long long *) layer_9_output;
  
//   // // worth it for 10 iterations? not really
//   // start = std::chrono::high_resolution_clock::now();
//   // kernel_time += layer10_gemm(cuda_layer_9_output, cuda_layer_10_output);
//   // end = std::chrono::high_resolution_clock::now();    
//   // auto l10_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());  
//   // float l10_kernel_time = kernel_time-(l1_kernel_time+l2_kernel_time+l4_kernel_time+l5_kernel_time+l8_kernel_time);
//   // l10_time -= l10_kernel_time*1000000.0f; // ms->ns

//   // Layer 10: Gemm @ cpp.binary
//   start = std::chrono::high_resolution_clock::now();
//   for(int b = 0; b < BATCH_SIZE; b++){
//     for (int d = 0; d < 10; d++) {
//       layer_10_output[b][d] = layer_10_bias[d];
//     }
//     for (int d = 0; d < 10; d++) {
//       for (int i = 0; i < 32; i++) {
//         layer_10_output[b][d] += 2 * __builtin_popcountll((unsigned long long)~(unsigned long long)(layer_10_weight[d][i] ^ layer_9_output[b][i])) - 64;
//       }
//     }
//   }
//   end = std::chrono::high_resolution_clock::now();    
//   auto l10_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());  
//   float l10_kernel_time = 0;

//   for(int b=0;b<BATCH_SIZE;b++){
//     for (int i = 0; i < 10; i++) {
//       output[b*10 + i] += layer_10_output[b][i];
//     }
//   }

//   return make_tuple(kernel_time, l1_time, l2_time, l3_time, l4_time, l5_time, l6_time, l8_time, l9_time, l10_time, 
//     l1_kernel_time, l2_kernel_time, l4_kernel_time, l5_kernel_time, l8_kernel_time, l10_kernel_time);
//   // return kernel_time;

// }
