#include <stdio.h>
#include <iostream>
#include <fstream>
#include <tuple>
#include <chrono>

#include "cuda_kernel.h"
#include "netW.hpp"
#include "utils.cuh"

using namespace std;

__global__ void layer1_conv_kernel(unsigned char *d_cuda_layer_0_output, float *d_layer_1_bias, signed char *d_cuda_layer_1_weight, float *d_cuda_layer_1_output){

    // https://github.com/ULHPC/tutorials/blob/devel/cuda/exercises/convolution/LoG_gpu_solution.cu

    int N = (32+1); // +1 to cover all edges (fixes bug #ky2)
    int kernel_size = 3;

    int tid = threadIdx.x; // = h
    int bid = blockIdx.y;  // = w
    int h = tid, w = bid;

    int m = blockIdx.z; // neurons in z-dir

    // batches in x-dir
    int b = blockIdx.x;
    //each block is assigned to a row of an image, iy index of y value                  
    int iy = blockIdx.y + (kernel_size - 1)/2;  
    //each thread is assigned to a pixel of a row, ix index of x value
    int ix = threadIdx.x + (kernel_size - 1)/2; 
    
    //idx global index (all blocks) of the image pixel 
    int idx = iy*N +ix;

    // bias is applied to every pixel
    if(tid < N){
        if(b < BATCH_SIZE){
            if(m < 128) {
                d_cuda_layer_1_output[index4D_cuda(b,h,w,m,32,32,128)] = d_layer_1_bias[m];
            }
        }
    }

    __syncthreads();

    // edge pixels are skipped here because they cannot fit entire convolution window
    if(idx < N*N){
        for (int kH = 0; kH < kernel_size; kH++){
            int iH = h * 1 + kH - 1;
            if (iH >= 0 && iH < 32) {
                for (int kW = 0; kW < kernel_size; kW++){
                    int iW = w * 1 + kW - 1;
                    if (iW >= 0 && iW < 32) {
                        if(b < BATCH_SIZE){
                            for (int c = 0; c < 3; c++) {
                                if(m < 128) {
                                    d_cuda_layer_1_output[index4D_cuda(b,h,w,m,32,32,128)] += d_cuda_layer_1_weight[index4D_cuda(kH,kW,c,m,3,3,128)] * d_cuda_layer_0_output[index4D_cuda(b,iH,iW,c,32,32,3)];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

float layer1_conv_cuda(unsigned char x[][32][32][3], float * cuda_layer_1_output){ // unsigned char * const x / unsigned char x[][32][32][3]
    
    setUniGPU();// use the second GPU on Uni-server because the first is used most of the time

    // initialize layer_0_output where x is the input image
    unsigned char (*layer_0_output)[BATCH_SIZE][32][32][3] = (unsigned char (*)[BATCH_SIZE][32][32][3]) x;

    // flatten 3D -> 1D arrays
    // flatten layer_1_weight
    signed char *cuda_layer_1_weight = (signed char *) layer_1_weight;

    // flatten layer_0_output
    unsigned char *cuda_layer_0_output = (unsigned char *) layer_0_output;

    // prepare for kernel call
    // declare storage on device
    unsigned char *d_cuda_layer_0_output; // storage on device for cuda_layer_0_output
    float *d_layer_1_bias; // storage on device for layer_1_bias
    signed char *d_cuda_layer_1_weight; // storage on device for cuda_layer_1_weight
    float *d_cuda_layer_1_output; // RESULT storage on device for cuda_layer_1_output

    // allocate GPU device buffers
    cudaMalloc((void **) &d_cuda_layer_0_output, BATCH_SIZE*32*32*3*sizeof(unsigned char)); // 3072 = 32x32x3 dim of cuda_layer_0_output
    cudaMalloc((void **) &d_layer_1_bias, 128*sizeof(float)); // 128 = dim of layer_1_bias
    cudaMalloc((void **) &d_cuda_layer_1_weight, 3*3*3*128*sizeof(signed char)); // 3456 = 3x3x3x128 dim of layer_1_weight
    cudaMalloc((void **) &d_cuda_layer_1_output, BATCH_SIZE*32*32*128*sizeof(float)); // 131072 = 32x32x128 dim of layer_1_output
    cudaCheckErrors("Failed to allocate device buffer");

    // copy input data from host on device
    cudaMemcpy(d_cuda_layer_0_output, cuda_layer_0_output, (BATCH_SIZE*32*32*3*sizeof(unsigned char)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer_1_bias, layer_1_bias, (128*sizeof(float)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cuda_layer_1_weight, cuda_layer_1_weight, (3*3*3*128*sizeof(signed char)), cudaMemcpyHostToDevice);
    cudaCheckErrors("CUDA memcpy failure");

    // define thread and block sizes
    const int BLKXSIZE = 32;
    const int BLKYSIZE = 1;
    const int GRIDXSIZE = BATCH_SIZE;
    const int GRIDYSIZE = 32;
    const int GRIDZSIZE = 128;

    const dim3 threadsPerBlock(BLKXSIZE, BLKYSIZE);
    const dim3 numBlocks(GRIDXSIZE, GRIDYSIZE, GRIDZSIZE);
    
    // timing of the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // compute result - kernel call
    cudaEventRecord(start);
    layer1_conv_kernel<<<numBlocks,threadsPerBlock>>>(d_cuda_layer_0_output, d_layer_1_bias, d_cuda_layer_1_weight, d_cuda_layer_1_output);
    cudaCheckErrors("Kernel launch failure");
    cudaEventRecord(stop);

    // synchronize threads
    cudaDeviceSynchronize();
    cudaCheckErrors("CUDA synchronize failure");
    cudaEventElapsedTime(&milliseconds, start, stop);

    // copy result from device to host
    cudaMemcpy(cuda_layer_1_output, d_cuda_layer_1_output, (BATCH_SIZE*32*32*128*sizeof(float)), cudaMemcpyDeviceToHost);
    cudaCheckErrors("CUDA memcpy failure");

    // free the memory
    cudaFree(d_cuda_layer_0_output);
    cudaFree(d_layer_1_bias);
    cudaFree(d_cuda_layer_1_weight);
    cudaFree(d_cuda_layer_1_output);
    cudaCheckErrors("cudaFree fail");

    return milliseconds;
}

__global__ void layer3_conv_kernel(unsigned long long *d_cuda_layer_2_output, float *d_layer_3_bias, unsigned long long *d_cuda_layer_3_weight, float *d_cuda_layer_3_output){

    int N = (32+1); // +1 to cover all edges (fixes bug #ky2)
    int kernel_size = 3;

    int tid = threadIdx.x; // = h
    int bid = blockIdx.y;  // = w
    int h = tid, w = bid;

    int m = blockIdx.z; // neurons in z-dir

    // batches in x-dir
    int b = blockIdx.x;
    //each block is assigned to a row of an image, iy index of y value                  
    int iy = blockIdx.y + (kernel_size - 1)/2;  
    //each thread is assigned to a pixel of a row, ix index of x value
    int ix = threadIdx.x + (kernel_size - 1)/2; 
    
    //idx global index (all blocks) of the image pixel 
    int idx = iy*N +ix;

    // bias is applied to every pixel
    if(tid < N){
        if(b < BATCH_SIZE){
            if(m < 128) {
                d_cuda_layer_3_output[index4D_cuda(b,h,w,m,32,32,128)] = d_layer_3_bias[m];
            }
        }
    }

    __syncthreads();

    // edge pixels are skipped here because they cannot fit entire convolution window
    if(idx < N*N){
        for (int kH = 0; kH < kernel_size; kH++){
            int iH = h * 1 + kH - 1;
            if (iH >= 0 && iH < 32) {
                for (int kW = 0; kW < kernel_size; kW++){
                    int iW = w * 1 + kW - 1;
                    if (iW >= 0 && iW < 32) {
                        if(b < BATCH_SIZE){
                            for (int c = 0; c < 2; c++) {
                                if(m < 128) {
                                    d_cuda_layer_3_output[index4D_cuda(b,h,w,m,32,32,128)] += 2 * __popcll((unsigned long long)~(unsigned long long)(d_cuda_layer_3_weight[index4D_cuda(kH,kW,m,c,3,128,2)] ^ d_cuda_layer_2_output[index4D_cuda(b,iH,iW,c,32,32,128)])) - 64; // ,128?
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

float layer3_conv_cuda(unsigned long long * cuda_layer_2_output, float * cuda_layer_3_output){
    
    setUniGPU();// use the second GPU on Uni-server because the first is used most of the time

    // flatten 3D -> 1D arrays
    // flatten layer_3_weight
    unsigned long long *cuda_layer_3_weight = (unsigned long long *) layer_3_weight;

    // prepare for kernel call
    // declare storage on device
    unsigned long long *d_cuda_layer_2_output; // storage on device for cuda_layer_2_output
    float *d_layer_3_bias; // storage on device for layer_3_bias
    unsigned long long *d_cuda_layer_3_weight; // storage on device for cuda_layer_3_weight
    float *d_cuda_layer_3_output; // RESULT storage on device for cuda_layer_3_output

    // allocate GPU device buffers
    cudaMalloc((void **) &d_cuda_layer_2_output, BATCH_SIZE*32*32*2*64*sizeof(unsigned long long)); // 131072 = 32x32x2x64 dim of cuda_layer_2_output
    cudaMalloc((void **) &d_layer_3_bias, 128*sizeof(float)); // 128 = dim of layer_3_bias
    cudaMalloc((void **) &d_cuda_layer_3_weight, 3*3*128*2*sizeof(unsigned long long)); // 2304 = 3x3x128x2 dim of layer_3_weight (without x64 as it would lead to segmentation fault at memcpy (even though there are bigger array mallocs))
    cudaMalloc((void **) &d_cuda_layer_3_output, BATCH_SIZE*32*32*128*sizeof(float)); // 131072 = 32x32x128 dim of layer_3_output
    cudaCheckErrors("Failed to allocate device buffer");

    // copy input data from host on device
    cudaMemcpy(d_cuda_layer_2_output, cuda_layer_2_output, (BATCH_SIZE*32*32*2*64*sizeof(unsigned long long)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer_3_bias, layer_3_bias, (128*sizeof(float)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cuda_layer_3_weight, cuda_layer_3_weight, (3*3*128*2*sizeof(unsigned long long)), cudaMemcpyHostToDevice);
    cudaCheckErrors("CUDA memcpy failure");

    // define thread and block sizes
    const int BLKXSIZE = 32;
    const int BLKYSIZE = 1;
    const int GRIDXSIZE = BATCH_SIZE;
    const int GRIDYSIZE = 32;
    const int GRIDZSIZE = 128;

    const dim3 threadsPerBlock(BLKXSIZE, BLKYSIZE);
    const dim3 numBlocks(GRIDXSIZE, GRIDYSIZE, GRIDZSIZE);
    
    // timing of the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // compute result - kernel call
    cudaEventRecord(start);
    layer3_conv_kernel<<<numBlocks,threadsPerBlock>>>(d_cuda_layer_2_output, d_layer_3_bias, d_cuda_layer_3_weight, d_cuda_layer_3_output);
    cudaCheckErrors("Kernel launch failure");
    cudaEventRecord(stop);

    // synchronize threads
    cudaDeviceSynchronize();
    cudaCheckErrors("CUDA synchronize failure");
    cudaEventElapsedTime(&milliseconds, start, stop);

    // copy result from device to host
    cudaMemcpy(cuda_layer_3_output, d_cuda_layer_3_output, (BATCH_SIZE*32*32*128*sizeof(float)), cudaMemcpyDeviceToHost);
    cudaCheckErrors("CUDA memcpy failure");

    // free the memory
    cudaFree(d_cuda_layer_2_output);
    cudaFree(d_layer_3_bias);
    cudaFree(d_cuda_layer_3_weight);
    cudaFree(d_cuda_layer_3_output);
    cudaCheckErrors("cudaFree fail");

    return milliseconds;
}

__global__ void layer4_maxpool_kernel(float *d_cuda_layer_3_output, float *d_cuda_layer_4_output, float lowest){

    int N = (16+1); // +1 to cover all edges (fixes bug #ky2)
    int kernel_size = 2;

    int tid = threadIdx.x; // = h
    int bid = blockIdx.y;  // = w
    int h = tid, w = bid;

    int c = blockIdx.z; // neurons in z-dir

    int b = blockIdx.x; // Batches index in grid x dir
    //each block is assigned to a row of an image, iy index of y value                  
    int iy = blockIdx.y + (kernel_size - 1)/2;  
    //each thread is assigned to a pixel of a row, ix index of x value
    int ix = threadIdx.x + (kernel_size - 1)/2; 
    
    //idx global index (all blocks) of the image pixel 
    int idx = iy*N +ix;

    // bias is applied to every pixel
    if(tid < N){
        if(b < BATCH_SIZE){
            if(c < 128) {
                d_cuda_layer_4_output[index4D_cuda(b,h,w,c,16,16,128)] = lowest;
            }
        }
    }

    __syncthreads();

    // edge pixels are skipped here because they cannot fit entire convolution window
    if(idx < N*N){
        for (int kH = 0; kH < kernel_size; kH++){
            for (int kW = 0; kW < kernel_size; kW++){
                if(b < BATCH_SIZE){
                    if(c < 128) {
                        d_cuda_layer_4_output[index4D_cuda(b,h,w,c,16,16,128)] = fmax(d_cuda_layer_3_output[index4D_cuda(b,(h * 2 + kH),(w * 2 + kW),c,32,32,128)], d_cuda_layer_4_output[index4D_cuda(b,h,w,c,16,16,128)]);
                    }
                }
            }
        }
    }
}

float layer4_maxpool_cuda(float * cuda_layer_3_output, float * cuda_layer_4_output){

    setUniGPU(); // use the second GPU on Uni-server because the first is used most of the time

    // flatten 3D -> 1D arrays
    // no 3D arrays to be flattened

    // prepare for kernel call
    // declare storage on device
    float *d_cuda_layer_3_output; // storage on device for cuda_layer_3_output
    float *d_cuda_layer_4_output; // RESULT storage on device for cuda_layer_4_output

    // allocate GPU device buffers
    cudaMalloc((void **) &d_cuda_layer_3_output, BATCH_SIZE*32*32*128*sizeof(float)); // 131072 = 32x32x128 dim of layer_3_output
    cudaMalloc((void **) &d_cuda_layer_4_output, BATCH_SIZE*16*16*128*sizeof(float)); // 32768 = 16x16x128 dim of layer_4_output
    cudaCheckErrors("Failed to allocate device buffer");

    // copy input data from host on device
    cudaMemcpy(d_cuda_layer_3_output, cuda_layer_3_output, (BATCH_SIZE*32*32*128*sizeof(float)), cudaMemcpyHostToDevice);
    cudaCheckErrors("CUDA memcpy failure");

    // define thread and block sizes
    const int BLKXSIZE = 16;
    const int BLKYSIZE = 1;
    const int GRIDXSIZE = BATCH_SIZE;
    const int GRIDYSIZE = 16;
    const int GRIDZSIZE = 128;

    const dim3 threadsPerBlock(BLKXSIZE, BLKYSIZE);
    const dim3 numBlocks(GRIDXSIZE, GRIDYSIZE, GRIDZSIZE);

    // std library not allowed on device
    const float LOWEST = std::numeric_limits<float>::lowest();

    // timing of the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // compute result - kernel call
    cudaEventRecord(start);
    layer4_maxpool_kernel<<<numBlocks, threadsPerBlock>>>(d_cuda_layer_3_output, d_cuda_layer_4_output, LOWEST);
    cudaCheckErrors("Kernel launch failure");
    cudaEventRecord(stop);

    // synchronize threads
    cudaDeviceSynchronize();
    cudaCheckErrors("CUDA synchronize failure");
    cudaEventElapsedTime(&milliseconds, start, stop);

    // copy result from device to host
    cudaMemcpy(cuda_layer_4_output, d_cuda_layer_4_output, (BATCH_SIZE*16*16*128*sizeof(float)), cudaMemcpyDeviceToHost);
    cudaCheckErrors("CUDA memcpy failure");

    // free the memory
    cudaFree(d_cuda_layer_3_output);
    cudaFree(d_cuda_layer_4_output);
    cudaCheckErrors("cudaFree fail");

    return milliseconds;
}

__global__ void layer6_conv_kernel(unsigned long long *d_cuda_layer_5_output, float *d_layer_6_bias, unsigned long long *d_cuda_layer_6_weight, float *d_cuda_layer_6_output){
    
    int N = (16+1); // +1 to cover all edges (fixes bug #ky2)
    int kernel_size = 3;

    int tid = threadIdx.x; // = h
    int bid = blockIdx.y;  // = w
    int h = tid, w = bid;

    int m = blockIdx.z; // neurons in z-dir

    // batches in x-dir
    int b = blockIdx.x;
    //each block is assigned to a row of an image, iy index of y value                  
    int iy = blockIdx.y + (kernel_size - 1)/2;  
    //each thread is assigned to a pixel of a row, ix index of x value
    int ix = threadIdx.x + (kernel_size - 1)/2; 
    
    //idx global index (all blocks) of the image pixel 
    int idx = iy*N +ix;

    // bias is applied to every pixel
    if(tid < N){
        if(b < BATCH_SIZE){
            if(m < 256) {
                d_cuda_layer_6_output[index4D_cuda(b,h,w,m,16,16,256)] = d_layer_6_bias[m];
            }
        }
    }

    __syncthreads();

    // edge pixels are skipped here because they cannot fit entire convolution window
    if(idx < N*N){
        for (int kH = 0; kH < kernel_size; kH++){
            int iH = h * 1 + kH - 1;
            if (iH >= 0 && iH < 16) {
                for (int kW = 0; kW < kernel_size; kW++){
                    int iW = w * 1 + kW - 1;
                    if (iW >= 0 && iW < 16) {
                        if(b < BATCH_SIZE){
                            for (int c = 0; c < 2; c++) {
                                if(m < 256) {
                                    d_cuda_layer_6_output[index4D_cuda(b,h,w,m,16,16,256)] += 2 * __popcll((unsigned long long)~(unsigned long long)(d_cuda_layer_6_weight[index4D_cuda(kH,kW,m,c,3,256,2)] ^ d_cuda_layer_5_output[index4D_cuda(b,iH,iW,c,16,16,128)])) - 64; // ,128?
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

float layer6_conv_cuda(unsigned long long * cuda_layer_5_output, float * cuda_layer_6_output){
    
    setUniGPU(); // use the second GPU on Uni-server because the first is used most of the time
    
    // flatten 3D -> 1D arrays
    // flatten layer_6_weight
    unsigned long long *cuda_layer_6_weight = (unsigned long long *) layer_6_weight;

    // prepare for kernel call
    // declare storage on device
    unsigned long long *d_cuda_layer_5_output; // storage on device for cuda_layer_5_output
    float *d_layer_6_bias; // storage on device for layer_6_bias
    unsigned long long *d_cuda_layer_6_weight; // storage on device for cuda_layer_6_weight
    float *d_cuda_layer_6_output; // RESULT storage on device for cuda_layer_6_output

    // allocate GPU device buffers
    cudaMalloc((void **) &d_cuda_layer_5_output, BATCH_SIZE*16*16*2*64*sizeof(unsigned long long)); // 32768 = 16x16x2x64 dim of cuda_layer_5_output
    cudaMalloc((void **) &d_layer_6_bias, 256*sizeof(float)); // 256 = dim of layer_6_bias
    cudaMalloc((void **) &d_cuda_layer_6_weight, 3*3*256*2*sizeof(unsigned long long)); // 4608 = 3x3x256x2 dim of layer_6_weight (without x64 as it would lead to segmentation fault at memcpy (even though there are bigger array mallocs))
    cudaMalloc((void **) &d_cuda_layer_6_output, BATCH_SIZE*16*16*256*sizeof(float)); // 65536 = 16x16x256 dim of layer_6_output
    cudaCheckErrors("Failed to allocate device buffer");

    // copy input data from host on device
    cudaMemcpy(d_cuda_layer_5_output, cuda_layer_5_output, (BATCH_SIZE*16*16*2*64*sizeof(unsigned long long)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer_6_bias, layer_6_bias, (256*sizeof(float)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cuda_layer_6_weight, cuda_layer_6_weight, (3*3*256*2*sizeof(unsigned long long)), cudaMemcpyHostToDevice);
    cudaCheckErrors("CUDA memcpy failure");

    // define thread and block sizes
    const int BLKXSIZE = 16;
    const int BLKYSIZE = 1;
    const int GRIDXSIZE = BATCH_SIZE;
    const int GRIDYSIZE = 16;
    const int GRIDZSIZE = 256;

    const dim3 threadsPerBlock(BLKXSIZE, BLKYSIZE);
    const dim3 numBlocks(GRIDXSIZE, GRIDYSIZE, GRIDZSIZE);
    
    // timing of the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // compute result - kernel call
    cudaEventRecord(start);
    layer6_conv_kernel<<<numBlocks,threadsPerBlock>>>(d_cuda_layer_5_output, d_layer_6_bias, d_cuda_layer_6_weight, d_cuda_layer_6_output);
    cudaCheckErrors("Kernel launch failure");
    cudaEventRecord(stop);

    // synchronize threads
    cudaDeviceSynchronize();
    cudaCheckErrors("CUDA synchronize failure");
    cudaEventElapsedTime(&milliseconds, start, stop);

    // copy result from device to host
    cudaMemcpy(cuda_layer_6_output, d_cuda_layer_6_output, (BATCH_SIZE*16*16*256*sizeof(float)), cudaMemcpyDeviceToHost);
    cudaCheckErrors("CUDA memcpy failure");

    // free the memory
    cudaFree(d_cuda_layer_5_output);
    cudaFree(d_layer_6_bias);
    cudaFree(d_cuda_layer_6_weight);
    cudaFree(d_cuda_layer_6_output);
    cudaCheckErrors("cudaFree fail");

    return milliseconds;
}

__global__ void layer8_conv_kernel(unsigned long long *d_cuda_layer_7_output, float *d_layer_8_bias, unsigned long long *d_cuda_layer_8_weight, float *d_cuda_layer_8_output){

    int N = (16+1); // +1 to cover all edges (fixes bug #ky2)
    int kernel_size = 3;

    int tid = threadIdx.x; // = h
    int bid = blockIdx.y;  // = w
    int h = tid, w = bid;

    int m = blockIdx.z; // neurons in z-dir

    // batches in x-dir
    int b = blockIdx.x;
    //each block is assigned to a row of an image, iy index of y value                  
    int iy = blockIdx.y + (kernel_size - 1)/2;  
    //each thread is assigned to a pixel of a row, ix index of x value
    int ix = threadIdx.x + (kernel_size - 1)/2; 
    
    //idx global index (all blocks) of the image pixel 
    int idx = iy*N +ix;

    // bias is applied to every pixel
    if(tid < N){
        if(b < BATCH_SIZE){
            if(m < 256) {
                d_cuda_layer_8_output[index4D_cuda(b,h,w,m,16,16,256)] = d_layer_8_bias[m];
            }
        }
    }

    __syncthreads();

    // edge pixels are skipped here because they cannot fit entire convolution window
    if(idx < N*N){
        for (int kH = 0; kH < kernel_size; kH++){
            int iH = h * 1 + kH - 1;
            if (iH >= 0 && iH < 16) {
                for (int kW = 0; kW < kernel_size; kW++){
                    int iW = w * 1 + kW - 1;
                    if (iW >= 0 && iW < 16) {
                        if(b < BATCH_SIZE){
                            for (int c = 0; c < 4; c++) {
                                if(m < 256) {
                                    d_cuda_layer_8_output[index4D_cuda(b,h,w,m,16,16,256)] += 2 * __popcll((unsigned long long)~(unsigned long long)(d_cuda_layer_8_weight[index4D_cuda(kH,kW,m,c,3,256,4)] ^ d_cuda_layer_7_output[index4D_cuda(b,iH,iW,c,16,16,256)])) - 64;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

float layer8_conv_cuda(unsigned long long * cuda_layer_7_output, float * cuda_layer_8_output){
    
    setUniGPU(); // use the second GPU on Uni-server because the first is used most of the time
    
    // flatten 3D -> 1D arrays
    // flatten layer_8_weight
    unsigned long long *cuda_layer_8_weight = (unsigned long long *) layer_8_weight;

    // prepare for kernel call
    // declare storage on device    
    unsigned long long *d_cuda_layer_7_output; // storage on device for cuda_layer_7_output
    float *d_layer_8_bias; // storage on device for layer_8_bias
    unsigned long long *d_cuda_layer_8_weight; // storage on device for cuda_layer_8_weight
    float *d_cuda_layer_8_output; // RESULT storage on device for cuda_layer_8_output

    // allocate GPU device buffers
    cudaMalloc((void **) &d_cuda_layer_7_output, BATCH_SIZE*16*16*4*64*sizeof(unsigned long long)); // 65536 = 16x16x4x64 dim of cuda_layer_7_output
    cudaMalloc((void **) &d_layer_8_bias, 256*sizeof(float)); // 256 = dim of layer_8_bias
    cudaMalloc((void **) &d_cuda_layer_8_weight, 3*3*256*4*sizeof(unsigned long long)); // 9216 = 3x3x256x4 dim of layer_8_weight (without x64 as it would lead to segmentation fault at memcpy (even though there are bigger array mallocs))
    cudaMalloc((void **) &d_cuda_layer_8_output, BATCH_SIZE*16*16*256*sizeof(float)); // 65536 = 16x16x256 dim of layer_8_output
    cudaCheckErrors("Failed to allocate device buffer");

    // copy input data from host on device
    cudaMemcpy(d_cuda_layer_7_output, cuda_layer_7_output, (BATCH_SIZE*16*16*4*64*sizeof(unsigned long long)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer_8_bias, layer_8_bias, (256*sizeof(float)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cuda_layer_8_weight, cuda_layer_8_weight, (3*3*256*4*sizeof(unsigned long long)), cudaMemcpyHostToDevice);
    cudaCheckErrors("CUDA memcpy failure");

    // define thread and block sizes
    const int BLKXSIZE = 16;
    const int BLKYSIZE = 1;
    const int GRIDXSIZE = BATCH_SIZE;
    const int GRIDYSIZE = 16;
    const int GRIDZSIZE = 256;

    const dim3 threadsPerBlock(BLKXSIZE, BLKYSIZE);
    const dim3 numBlocks(GRIDXSIZE, GRIDYSIZE, GRIDZSIZE);
    
    // timing of the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // compute result - kernel call
    cudaEventRecord(start);
    layer8_conv_kernel<<<numBlocks,threadsPerBlock>>>(d_cuda_layer_7_output, d_layer_8_bias, d_cuda_layer_8_weight, d_cuda_layer_8_output);
    cudaCheckErrors("Kernel launch failure");
    cudaEventRecord(stop);

    // synchronize threads
    cudaDeviceSynchronize();
    cudaCheckErrors("CUDA synchronize failure");
    cudaEventElapsedTime(&milliseconds, start, stop);

    // copy result from device to host
    cudaMemcpy(cuda_layer_8_output, d_cuda_layer_8_output, (BATCH_SIZE*16*16*256*sizeof(float)), cudaMemcpyDeviceToHost);
    cudaCheckErrors("CUDA memcpy failure");

    // free the memory
    cudaFree(d_cuda_layer_7_output);
    cudaFree(d_layer_8_bias);
    cudaFree(d_cuda_layer_8_weight);
    cudaFree(d_cuda_layer_8_output);
    cudaCheckErrors("cudaFree fail");

    return milliseconds;
}

__global__ void layer9_maxpool_kernel(float *d_cuda_layer_8_output, float *d_cuda_layer_9_output, float lowest){

    int N = (8+1); // +1 to cover all edges (fixes bug #ky2)
    int kernel_size = 2;

    int tid = threadIdx.x; // = h
    int bid = blockIdx.y;  // = w
    int h = tid, w = bid;

    int c = blockIdx.z; // neurons in z-dir

    int b = blockIdx.x; // Batches index in grid x dir
    //each block is assigned to a row of an image, iy index of y value                  
    int iy = blockIdx.y + (kernel_size - 1)/2;  
    //each thread is assigned to a pixel of a row, ix index of x value
    int ix = threadIdx.x + (kernel_size - 1)/2; 
    
    //idx global index (all blocks) of the image pixel 
    int idx = iy*N +ix;

    // bias is applied to every pixel
    if(tid < N){
        if(b < BATCH_SIZE){
            if(c < 256) {
                d_cuda_layer_9_output[index4D_cuda(b,h,w,c,8,8,256)] = lowest;
            }
        }
    }

    __syncthreads();

    // edge pixels are skipped here because they cannot fit entire convolution window
    if(idx < N*N){
        for (int kH = 0; kH < kernel_size; kH++){
            for (int kW = 0; kW < kernel_size; kW++){
                if(b < BATCH_SIZE){
                    if(c < 256) {
                        d_cuda_layer_9_output[index4D_cuda(b,h,w,c,8,8,256)] = fmax(d_cuda_layer_8_output[index4D_cuda(b,(h * 2 + kH),(w * 2 + kW),c,16,16,256)], d_cuda_layer_9_output[index4D_cuda(b,h,w,c,8,8,256)]);
                    }
                }
            }
        }
    }
}

float layer9_maxpool_cuda(float * cuda_layer_8_output, float * cuda_layer_9_output){
    
    setUniGPU(); // use the second GPU on Uni-server because the first is used most of the time
    
    // flatten 3D -> 1D arrays
    // no 3D arrays to be flattened

    // prepare for kernel call
    // declare storage on device
    float *d_cuda_layer_8_output; // storage on device for cuda_layer_8_output
    float *d_cuda_layer_9_output; // RESULT storage on device for cuda_layer_9_output

    // allocate GPU device buffers
    cudaMalloc((void **) &d_cuda_layer_8_output, BATCH_SIZE*16*16*256*sizeof(float)); // 65536 = 16x16x256 dim of layer_8_output
    cudaMalloc((void **) &d_cuda_layer_9_output, BATCH_SIZE*8*8*256*sizeof(float)); // 16384 = 8x8x256 dim of layer_9_output
    cudaCheckErrors("Failed to allocate device buffer");

    // copy input data from host on device
    cudaMemcpy(d_cuda_layer_8_output, cuda_layer_8_output, (BATCH_SIZE*16*16*256*sizeof(float)), cudaMemcpyHostToDevice);
    cudaCheckErrors("CUDA memcpy failure");

    // define thread and block sizes
    const int BLKXSIZE = 8;
    const int BLKYSIZE = 1;
    const int GRIDXSIZE = BATCH_SIZE;
    const int GRIDYSIZE = 8;
    const int GRIDZSIZE = 256;

    const dim3 threadsPerBlock(BLKXSIZE, BLKYSIZE);
    const dim3 numBlocks(GRIDXSIZE, GRIDYSIZE, GRIDZSIZE);

    // std library not allowed on device
    const float LOWEST = std::numeric_limits<float>::lowest();

    // timing of the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // compute result - kernel call
    cudaEventRecord(start);
    layer9_maxpool_kernel<<<numBlocks, threadsPerBlock>>>(d_cuda_layer_8_output, d_cuda_layer_9_output, LOWEST);
    cudaCheckErrors("Kernel launch failure");
    cudaEventRecord(stop);

    // synchronize threads
    cudaDeviceSynchronize();
    cudaCheckErrors("CUDA synchronize failure");
    cudaEventElapsedTime(&milliseconds, start, stop);

    // copy result from device to host
    cudaMemcpy(cuda_layer_9_output, d_cuda_layer_9_output, (BATCH_SIZE*8*8*256*sizeof(float)), cudaMemcpyDeviceToHost);
    cudaCheckErrors("CUDA memcpy failure");

    // free the memory
    cudaFree(d_cuda_layer_8_output);
    cudaFree(d_cuda_layer_9_output);
    cudaCheckErrors("cudaFree fail");

    return milliseconds;
}

__global__ void layer11_conv_kernel(unsigned long long *d_cuda_layer_10_output, float *d_layer_11_bias, unsigned long long *d_cuda_layer_11_weight, float *d_cuda_layer_11_output){

    int N = (8+1); // +1 to cover all edges (fixes bug #ky2) 
    int kernel_size = 3;

    int tid = threadIdx.x; // = h
    int bid = blockIdx.y;  // = w
    int h = tid, w = bid;

    int m = blockIdx.z; // neurons in z-dir

    // batches in x-dir
    int b = blockIdx.x;
    //each block is assigned to a row of an image, iy index of y value                  
    int iy = blockIdx.y + (kernel_size - 1)/2;  
    //each thread is assigned to a pixel of a row, ix index of x value
    int ix = threadIdx.x + (kernel_size - 1)/2; 
    
    //idx global index (all blocks) of the image pixel 
    int idx = iy*N +ix;

    // bias is applied to every pixel
    if(tid < N){
        if(b < BATCH_SIZE){
            if(m < 512) {
                d_cuda_layer_11_output[index4D_cuda(b,h,w,m,8,8,512)] = d_layer_11_bias[m];
            }
        }
    }

    __syncthreads();

    // edge pixels are skipped here because they cannot fit entire convolution window
    if(idx < N*N){
        for (int kH = 0; kH < kernel_size; kH++){
            int iH = h * 1 + kH - 1;
            if (iH >= 0 && iH < 8) {
                for (int kW = 0; kW < kernel_size; kW++){
                    int iW = w * 1 + kW - 1;
                    if (iW >= 0 && iW < 8) {
                        if(b < BATCH_SIZE){
                            for (int c = 0; c < 4; c++) {
                                if(m < 512) {
                                    d_cuda_layer_11_output[index4D_cuda(b,h,w,m,8,8,512)] += 2 * __popcll((unsigned long long)~(unsigned long long)(d_cuda_layer_11_weight[index4D_cuda(kH,kW,m,c,3,512,4)] ^ d_cuda_layer_10_output[index4D_cuda(b,iH,iW,c,8,8,256)])) - 64;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

float layer11_conv_cuda(unsigned long long * cuda_layer_10_output, float * cuda_layer_11_output){
    
    setUniGPU(); // use the second GPU on Uni-server because the first is used most of the time
    
    // flatten 3D -> 1D arrays
    // flatten layer_11_weight
    unsigned long long *cuda_layer_11_weight = (unsigned long long *) layer_11_weight;

    // prepare for kernel call
    // declare storage on device
    unsigned long long *d_cuda_layer_10_output; // storage on device for cuda_layer_10_output
    float *d_layer_11_bias; // storage on device for layer_11_bias
    unsigned long long *d_cuda_layer_11_weight; // storage on device for cuda_layer_11_weight
    float *d_cuda_layer_11_output; // RESULT storage on device for cuda_layer_11_output

    // allocate GPU device buffers
    cudaMalloc((void **) &d_cuda_layer_10_output, BATCH_SIZE*8*8*4*64*sizeof(unsigned long long)); // 16384 = 8x8x4x64 dim of cuda_layer_10_output
    cudaMalloc((void **) &d_layer_11_bias, 512*sizeof(float)); // 512 = dim of layer_11_bias
    cudaMalloc((void **) &d_cuda_layer_11_weight, 3*3*512*4*sizeof(unsigned long long)); // 18432 = 3x3x512x4 dim of layer_11_weight (without x64 as it would lead to segmentation fault at memcpy (even though there are bigger array mallocs))
    cudaMalloc((void **) &d_cuda_layer_11_output, BATCH_SIZE*8*8*512*sizeof(float)); // 32768 = 8x8x512 dim of layer_11_output
    cudaCheckErrors("Failed to allocate device buffer");

    // copy input data from host on device
    cudaMemcpy(d_cuda_layer_10_output, cuda_layer_10_output, (BATCH_SIZE*8*8*4*64*sizeof(unsigned long long)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer_11_bias, layer_11_bias, (512*sizeof(float)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cuda_layer_11_weight, cuda_layer_11_weight, (3*3*512*4*sizeof(unsigned long long)), cudaMemcpyHostToDevice);
    cudaCheckErrors("CUDA memcpy failure");

    // define thread and block sizes
    const int BLKXSIZE = 8;
    const int BLKYSIZE = 1;
    const int GRIDXSIZE = BATCH_SIZE;
    const int GRIDYSIZE = 8;
    const int GRIDZSIZE = 512;

    const dim3 threadsPerBlock(BLKXSIZE, BLKYSIZE);
    const dim3 numBlocks(GRIDXSIZE, GRIDYSIZE, GRIDZSIZE);
    
    // timing of the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // compute result - kernel call
    cudaEventRecord(start);
    layer11_conv_kernel<<<numBlocks,threadsPerBlock>>>(d_cuda_layer_10_output, d_layer_11_bias, d_cuda_layer_11_weight, d_cuda_layer_11_output);
    cudaCheckErrors("Kernel launch failure");
    cudaEventRecord(stop);

    // synchronize threads
    cudaDeviceSynchronize();
    cudaCheckErrors("CUDA synchronize failure");
    cudaEventElapsedTime(&milliseconds, start, stop);

    // copy result from device to host
    cudaMemcpy(cuda_layer_11_output, d_cuda_layer_11_output, (BATCH_SIZE*8*8*512*sizeof(float)), cudaMemcpyDeviceToHost);
    cudaCheckErrors("CUDA memcpy failure");

    // free the memory
    cudaFree(d_cuda_layer_10_output);
    cudaFree(d_layer_11_bias);
    cudaFree(d_cuda_layer_11_weight);
    cudaFree(d_cuda_layer_11_output);
    cudaCheckErrors("cudaFree fail");

    return milliseconds;
}

__global__ void layer13_conv_kernel(unsigned long long *d_cuda_layer_12_output, float *d_layer_13_bias, unsigned long long *d_cuda_layer_13_weight, float *d_cuda_layer_13_output){

    int N = (8+1); // +1 to cover all edges (fixes bug #ky2)
    int kernel_size = 3;

    int tid = threadIdx.x; // = h
    int bid = blockIdx.y;  // = w
    int h = tid, w = bid;

    int m = blockIdx.z; // neurons in z-dir

    // batches in x-dir
    int b = blockIdx.x;
    //each block is assigned to a row of an image, iy index of y value                  
    int iy = blockIdx.y + (kernel_size - 1)/2;  
    //each thread is assigned to a pixel of a row, ix index of x value
    int ix = threadIdx.x + (kernel_size - 1)/2; 
    
    //idx global index (all blocks) of the image pixel 
    int idx = iy*N +ix;

    // bias is applied to every pixel
    if(tid < N){
        if(b < BATCH_SIZE){
            if(m < 512) {
                d_cuda_layer_13_output[index4D_cuda(b,h,w,m,8,8,512)] = d_layer_13_bias[m];
            }
        }
    }

    __syncthreads();

    // edge pixels are skipped here because they cannot fit entire convolution window
    if(idx < N*N){
        for (int kH = 0; kH < kernel_size; kH++){
            int iH = h * 1 + kH - 1;
            if (iH >= 0 && iH < 8) {
                for (int kW = 0; kW < kernel_size; kW++){
                    int iW = w * 1 + kW - 1;
                    if (iW >= 0 && iW < 8) {
                        if(b < BATCH_SIZE){
                            for (int c = 0; c < 8; c++) {
                                if(m < 512) {
                                    d_cuda_layer_13_output[index4D_cuda(b,h,w,m,8,8,512)] += 2 * __popcll((unsigned long long)~(unsigned long long)(d_cuda_layer_13_weight[index4D_cuda(kH,kW,m,c,3,512,8)] ^ d_cuda_layer_12_output[index4D_cuda(b,iH,iW,c,8,8,512)])) - 64;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

float layer13_conv_cuda(unsigned long long * cuda_layer_12_output, float * cuda_layer_13_output){
    
    setUniGPU(); // use the second GPU on Uni-server because the first is used most of the time
    
    // flatten 3D -> 1D arrays
    // flatten layer_13_weight
    unsigned long long *cuda_layer_13_weight = (unsigned long long *) layer_13_weight;

    // prepare for kernel call
    // declare storage on device
    unsigned long long *d_cuda_layer_12_output; // storage on device for cuda_layer_12_output
    float *d_layer_13_bias; // storage on device for layer_13_bias
    unsigned long long *d_cuda_layer_13_weight; // storage on device for cuda_layer_13_weight
    float *d_cuda_layer_13_output; // RESULT storage on device for cuda_layer_13_output

    // allocate GPU device buffers
    cudaMalloc((void **) &d_cuda_layer_12_output, BATCH_SIZE*8*8*8*64*sizeof(unsigned long long)); // 32768 = 8x8x8x64 dim of cuda_layer_12_output
    cudaMalloc((void **) &d_layer_13_bias, 512*sizeof(float)); // 512 = dim of layer_13_bias
    cudaMalloc((void **) &d_cuda_layer_13_weight, 3*3*512*8*sizeof(unsigned long long)); // 36864 = 3x3x512x8 dim of layer_13_weight (without x64 as it would lead to segmentation fault at memcpy (even though there are bigger array mallocs))
    cudaMalloc((void **) &d_cuda_layer_13_output, BATCH_SIZE*8*8*512*sizeof(float)); // 32768 = 8x8x512 dim of layer_13_output
    cudaCheckErrors("Failed to allocate device buffer");

    // copy input data from host on device
    cudaMemcpy(d_cuda_layer_12_output, cuda_layer_12_output, (BATCH_SIZE*8*8*8*64*sizeof(unsigned long long)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer_13_bias, layer_13_bias, (512*sizeof(float)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cuda_layer_13_weight, cuda_layer_13_weight, (3*3*512*8*sizeof(unsigned long long)), cudaMemcpyHostToDevice);
    cudaCheckErrors("CUDA memcpy failure");

    // define thread and block sizes
    const int BLKXSIZE = 8;
    const int BLKYSIZE = 1;
    const int GRIDXSIZE = BATCH_SIZE;
    const int GRIDYSIZE = 8;
    const int GRIDZSIZE = 512;

    const dim3 threadsPerBlock(BLKXSIZE, BLKYSIZE);
    const dim3 numBlocks(GRIDXSIZE, GRIDYSIZE, GRIDZSIZE);
    
    // timing of the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // compute result - kernel call
    cudaEventRecord(start);
    layer13_conv_kernel<<<numBlocks,threadsPerBlock>>>(d_cuda_layer_12_output, d_layer_13_bias, d_cuda_layer_13_weight, d_cuda_layer_13_output);
    cudaCheckErrors("Kernel launch failure");
    cudaEventRecord(stop);

    // synchronize threads
    cudaDeviceSynchronize();
    cudaCheckErrors("CUDA synchronize failure");
    cudaEventElapsedTime(&milliseconds, start, stop);

    // copy result from device to host
    cudaMemcpy(cuda_layer_13_output, d_cuda_layer_13_output, (BATCH_SIZE*8*8*512*sizeof(float)), cudaMemcpyDeviceToHost);
    cudaCheckErrors("CUDA memcpy failure");

    // free the memory
    cudaFree(d_cuda_layer_12_output);
    cudaFree(d_layer_13_bias);
    cudaFree(d_cuda_layer_13_weight);
    cudaFree(d_cuda_layer_13_output);
    cudaCheckErrors("cudaFree fail");

    return milliseconds;
}

__global__ void layer14_maxpool_kernel(float *d_cuda_layer_13_output, float *d_cuda_layer_14_output, float lowest){

    int N = (4+1); // +1 to cover all edges (fixes bug #ky2)
    int kernel_size = 2;

    int tid = threadIdx.x; // = h
    int bid = blockIdx.y;  // = w
    int h = tid, w = bid;

    int c = blockIdx.z; // neurons in z-dir

    int b = blockIdx.x; // Batches index in grid x dir
    //each block is assigned to a row of an image, iy index of y value                  
    int iy = blockIdx.y + (kernel_size - 1)/2;  
    //each thread is assigned to a pixel of a row, ix index of x value
    int ix = threadIdx.x + (kernel_size - 1)/2; 
    
    //idx global index (all blocks) of the image pixel 
    int idx = iy*N +ix;

    // bias is applied to every pixel
    if(tid < N){
        if(b < BATCH_SIZE){
            if(c < 512) {
                d_cuda_layer_14_output[index4D_cuda(b,h,w,c,4,4,512)] = lowest;
            }
        }
    }

    __syncthreads();

    // edge pixels are skipped here because they cannot fit entire convolution window
    if(idx < N*N){
        for (int kH = 0; kH < kernel_size; kH++){
            for (int kW = 0; kW < kernel_size; kW++){
                if(b < BATCH_SIZE){
                    if(c < 512) {
                        d_cuda_layer_14_output[index4D_cuda(b,h,w,c,4,4,512)] = fmax(d_cuda_layer_13_output[index4D_cuda(b,(h * 2 + kH),(w * 2 + kW),c,8,8,512)], d_cuda_layer_14_output[index4D_cuda(b,h,w,c,4,4,512)]);
                    }
                }
            }
        }
    }
}

float layer14_maxpool_cuda(float * cuda_layer_13_output, float * cuda_layer_14_output){
    
    setUniGPU(); // use the second GPU on Uni-server because the first is used most of the time
    
    // flatten 3D -> 1D arrays
    // no 3D arrays to be flattened

    // prepare for kernel call
    // declare storage on device
    float *d_cuda_layer_13_output; // storage on device for cuda_layer_13_output
    float *d_cuda_layer_14_output; // RESULT storage on device for cuda_layer_14_output

    // allocate GPU device buffers
    cudaMalloc((void **) &d_cuda_layer_13_output, BATCH_SIZE*8*8*512*sizeof(float)); // 32768 = 8x8x512 dim of layer_13_output
    cudaMalloc((void **) &d_cuda_layer_14_output, BATCH_SIZE*4*4*512*sizeof(float)); // 8192 = 4x4x512 dim of layer_14_output
    cudaCheckErrors("Failed to allocate device buffer");

    // copy input data from host on device
    cudaMemcpy(d_cuda_layer_13_output, cuda_layer_13_output, (BATCH_SIZE*8*8*512*sizeof(float)), cudaMemcpyHostToDevice);
    cudaCheckErrors("CUDA memcpy failure");

    // define thread and block sizes
    const int BLKXSIZE = 4;
    const int BLKYSIZE = 1;
    const int GRIDXSIZE = BATCH_SIZE;
    const int GRIDYSIZE = 4;
    const int GRIDZSIZE = 512;

    const dim3 threadsPerBlock(BLKXSIZE, BLKYSIZE);
    const dim3 numBlocks(GRIDXSIZE, GRIDYSIZE, GRIDZSIZE);

    // std library not allowed on device
    const float LOWEST = std::numeric_limits<float>::lowest();

    // timing of the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // compute result - kernel call
    cudaEventRecord(start);
    layer14_maxpool_kernel<<<numBlocks, threadsPerBlock>>>(d_cuda_layer_13_output, d_cuda_layer_14_output, LOWEST);
    cudaCheckErrors("Kernel launch failure");
    cudaEventRecord(stop);

    // synchronize threads
    cudaDeviceSynchronize();
    cudaCheckErrors("CUDA synchronize failure");
    cudaEventElapsedTime(&milliseconds, start, stop);

    // copy result from device to host
    cudaMemcpy(cuda_layer_14_output, d_cuda_layer_14_output, (BATCH_SIZE*4*4*512*sizeof(float)), cudaMemcpyDeviceToHost);
    cudaCheckErrors("CUDA memcpy failure");

    // free the memory
    cudaFree(d_cuda_layer_13_output);
    cudaFree(d_cuda_layer_14_output);
    cudaCheckErrors("cudaFree fail");

    return milliseconds;
}

__global__ void layer17_gemm_kernel(unsigned long long *d_cuda_layer_16_output, float *d_layer_17_bias, unsigned long long *d_cuda_layer_17_weight, float *d_cuda_layer_17_output){

    int z = blockDim.x * blockIdx.z + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    int d = z*blockDim.x+y;

    int b = blockIdx.x;

    if(d < 1024){
        if(b < BATCH_SIZE){
            d_cuda_layer_17_output[b*1024 + d] = d_layer_17_bias[d];
            for (int i = 0; i < 128; i++) {
                d_cuda_layer_17_output[b*1024 + d] += 2 * __popcll((unsigned long long)~(unsigned long long)(d_cuda_layer_17_weight[d*128+i] ^ d_cuda_layer_16_output[b*128+i])) - 64; // try also: d_cuda_layer_16_output[b*128+i]
            }
        }
    }
}

float layer17_gemm_cuda(unsigned long long * cuda_layer_16_output, float * cuda_layer_17_output){
    
    setUniGPU(); // use the second GPU on Uni-server because the first is used most of the time
    
    // flatten 3D -> 1D arrays
    // flatten layer_17_weight
    unsigned long long *cuda_layer_17_weight = (unsigned long long *) layer_17_weight;

    // prepare for kernel call
    // declare storage on device
    unsigned long long *d_cuda_layer_16_output; // storage on device for cuda_layer_16_output
    float *d_layer_17_bias;  // storage on device for layer_17_bias
    unsigned long long *d_cuda_layer_17_weight; // storage on device for cuda_layer_17_weight
    float *d_cuda_layer_17_output; // RESULT storage on device for cuda_layer_17_output

    // allocate GPU device buffers
    cudaMalloc((void **) &d_cuda_layer_16_output, BATCH_SIZE*4*4*8*sizeof(unsigned long long)); // 128=4x4x8 dim of cuda_layer_16_output
    cudaMalloc((void **) &d_layer_17_bias, 1024*sizeof(float)); // 1024 = dim of layer_17_bias
    cudaMalloc((void **) &d_cuda_layer_17_weight, 1024*128*sizeof(unsigned long long)); // 131072 = 1024x128 dim of layer_17_weight [ULL]
    cudaMalloc((void **) &d_cuda_layer_17_output, BATCH_SIZE*1024*sizeof(float)); // 1024 = dim of layer_17_output
    cudaCheckErrors("Failed to allocate device buffer");

    // copy input data from host on device
    cudaMemcpy(d_cuda_layer_16_output, cuda_layer_16_output, (BATCH_SIZE*4*4*8*sizeof(unsigned long long)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer_17_bias, layer_17_bias, (1024*sizeof(float)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cuda_layer_17_weight, cuda_layer_17_weight, (1024*128*sizeof(unsigned long long)), cudaMemcpyHostToDevice);
    cudaCheckErrors("CUDA memcpy failure");

    // define thread and block sizes
    /*
        Maximum threads in a block: 1024 => Maximum block size 32x32
        if more than 1024 threads are needed, then set block size to maximum (32x32) and put multiple blocks in z-dir
        else if less than 1024 are needed, then only create 1 (square) block in z-dir, of size ceil(sqrt(THREADS_NEEDED))
    */
    const int BLKXSIZE = 32;
    const int BLKYSIZE = 32;
    const int GRIDXSIZE = BATCH_SIZE;
    const int GRIDYSIZE = 1;
    const int GRIDZSIZE = 1;

    const dim3 threadsPerBlock(BLKXSIZE, BLKYSIZE);
    const dim3 numBlocks(GRIDXSIZE, GRIDYSIZE, GRIDZSIZE);

    // timing of the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // compute result - kernel call
    cudaEventRecord(start);
    layer17_gemm_kernel<<<numBlocks,threadsPerBlock>>>(d_cuda_layer_16_output, d_layer_17_bias, d_cuda_layer_17_weight, d_cuda_layer_17_output);
    cudaEventRecord(stop);
    cudaCheckErrors("Kernel launch failure");

    // synchronize threads
    cudaDeviceSynchronize();
    cudaCheckErrors("CUDA synchronize failure");
    cudaEventElapsedTime(&milliseconds, start, stop);

    // copy result from device to host
    cudaMemcpy(cuda_layer_17_output, d_cuda_layer_17_output, (BATCH_SIZE*1024*sizeof(float)), cudaMemcpyDeviceToHost);
    cudaCheckErrors("CUDA memcpy failure");

    // free the memory
    cudaFree(d_cuda_layer_16_output);
    cudaFree(d_layer_17_bias);
    cudaFree(d_cuda_layer_17_weight);
    cudaFree(d_cuda_layer_17_output);
    cudaCheckErrors("cudaFree fail");

    return milliseconds;
}

__global__ void layer19_gemm_kernel(unsigned long long *d_cuda_layer_18_output, float *d_layer_19_bias, unsigned long long *d_cuda_layer_19_weight, float *d_cuda_layer_19_output){

    int z = blockDim.x * blockIdx.z + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    int d = z*blockDim.x+y;

    int b = blockIdx.x;

    if(d < 10){
        if(b < BATCH_SIZE){
            d_cuda_layer_19_output[b*10 + d] = d_layer_19_bias[d];
            for (int i = 0; i < 16; i++) {
                d_cuda_layer_19_output[b*10 + d] += 2 * __popcll((unsigned long long)~(unsigned long long)(d_cuda_layer_19_weight[d*16+i] ^ d_cuda_layer_18_output[b*16+i])) - 64; // try also: d_cuda_layer_18_output[b*10+i]
            }
        }
    }
}

float layer19_gemm_cuda(unsigned long long * cuda_layer_18_output, float * cuda_layer_19_output){
    
    setUniGPU(); // use the second GPU on Uni-server because the first is used most of the time
    
    // flatten 3D -> 1D arrays
    // flatten layer_19_weight
    unsigned long long *cuda_layer_19_weight = (unsigned long long *) layer_19_weight;

    // prepare for kernel call
    // declare storage on device
    unsigned long long *d_cuda_layer_18_output; // storage on device for cuda_layer_18_output
    float *d_layer_19_bias;  // storage on device for layer_19_bias
    unsigned long long *d_cuda_layer_19_weight; // storage on device for cuda_layer_19_weight
    float *d_cuda_layer_19_output; // RESULT storage on device for cuda_layer_19_output

    // allocate GPU device buffers
    cudaMalloc((void **) &d_cuda_layer_18_output, BATCH_SIZE*16*sizeof(unsigned long long)); // 16 = dim of cuda_layer_18_output
    cudaMalloc((void **) &d_layer_19_bias, 10*sizeof(float)); // 10 = dim of layer_19_bias
    cudaMalloc((void **) &d_cuda_layer_19_weight, 10*16*sizeof(unsigned long long)); // 160 = 10x16 dim of layer_19_weight [ULL]
    cudaMalloc((void **) &d_cuda_layer_19_output, BATCH_SIZE*10*sizeof(float)); // 10 = dim of layer_19_output
    cudaCheckErrors("Failed to allocate device buffer");

    // copy input data from host on device
    cudaMemcpy(d_cuda_layer_18_output, cuda_layer_18_output, (BATCH_SIZE*16*sizeof(unsigned long long)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer_19_bias, layer_19_bias, (10*sizeof(float)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cuda_layer_19_weight, cuda_layer_19_weight, (10*16*sizeof(unsigned long long)), cudaMemcpyHostToDevice);
    cudaCheckErrors("CUDA memcpy failure");

    // define thread and block sizes
    /*
        Maximum threads in a block: 1024 => Maximum block size 32x32
        if more than 1024 threads are needed, then set block size to maximum (32x32) and put multiple blocks in z-dir
        else if less than 1024 are needed, then only create 1 (square) block in z-dir, of size ceil(sqrt(THREADS_NEEDED))
    */
    const int BLKXSIZE = 10;
    const int BLKYSIZE = 10;
    const int GRIDXSIZE = BATCH_SIZE;
    const int GRIDYSIZE = 1;
    const int GRIDZSIZE = 1;

    const dim3 threadsPerBlock(BLKXSIZE, BLKYSIZE);
    const dim3 numBlocks(GRIDXSIZE, GRIDYSIZE, GRIDZSIZE);

    // timing of the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // compute result - kernel call
    cudaEventRecord(start);
    layer19_gemm_kernel<<<numBlocks,threadsPerBlock>>>(d_cuda_layer_18_output, d_layer_19_bias, d_cuda_layer_19_weight, d_cuda_layer_19_output);
    cudaEventRecord(stop);
    cudaCheckErrors("Kernel launch failure");

    // synchronize threads
    cudaDeviceSynchronize();
    cudaCheckErrors("CUDA synchronize failure");
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // copy result from device to host
    cudaMemcpy(cuda_layer_19_output, d_cuda_layer_19_output, (BATCH_SIZE*10*sizeof(float)), cudaMemcpyDeviceToHost);
    cudaCheckErrors("CUDA memcpy failure");

    // free the memory
    cudaFree(d_cuda_layer_18_output);
    cudaFree(d_layer_19_bias);
    cudaFree(d_cuda_layer_19_weight);
    cudaFree(d_cuda_layer_19_output);
    cudaCheckErrors("cudaFree fail");

    return milliseconds;
}
