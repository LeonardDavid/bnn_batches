/*
    For profiling all layers
    
    Run with: 
    $ make
    $ ./parprof.o
*/

#include <iostream>
#include <chrono>
#include <algorithm>
#include <tuple>
#include <cmath>

#include "MNISTLoader.h"
#include "utils.h"

#ifdef BINARY
#define INPUT_FEATURE char
#include "net.hpp"
#elif INT16
#define INPUT_FEATURE int
#include "net.hpp"
#else
#define INPUT_FEATURE float
#include "net.hpp"
#endif

using namespace std;

/* bnn_cpu normal */

// auto benchmark(vector<MNISTLoader> &loaderx, bool verbose = false) {
// #if defined BINARY || defined INT16
//     int output[BATCH_SIZE*OUT_SIZE] = {0};
// #else
//     float output[BATCH_SIZE*OUT_SIZE] = {0};
// #endif

//     int factor = 1;
//     int matches[BATCH_SIZE] = {0};
//     int const imgsize = IMG_HEIGHT*IMG_WIDTH;
//     int lsize = loaderx[0].size();

//     float total_kernel_time = 0; 
//     float l1_time = 0, l2_time = 0, l3_time = 0, l4_time = 0, l5_time = 0, l6_time = 0, l8_time = 0, l9_time = 0, l10_time = 0;
//     float l1_ktime = 0, l2_ktime = 0, l4_ktime = 0, l5_ktime = 0, l8_ktime = 0, l10_ktime = 0;
//     float malloc_time = 0, cpy_time = 0;

//     auto start = std::chrono::high_resolution_clock::now();
//     for (unsigned int i = 0; i < lsize; i+=factor) { // i := # image
//         std::fill(output, output+OUT_SIZE*BATCH_SIZE, 0);
       
//         unsigned char * img;
//         img = (unsigned char*) malloc (BATCH_SIZE*imgsize);

//         // load label i of corresponding image from every batch in an array
//         int label[BATCH_SIZE];

//         for(int b=0; b<BATCH_SIZE; b++){    // b := # batch
//             for(int p=0; p<imgsize; p++){   // p := # pixel
//                 img[b*imgsize+p] = loaderx[b].images(i)[p]; 
//             }
//             label[b] = loaderx[b].labels(i); 
//         }
        
//         // // display img array (remove for before)
//         // for(int b=0;b<BATCH_SIZE;b++){
//         //     printf("batch: %d, label %d:\n",b,label[b]);
//         //     for (int i = 0; i < 28; i++)
//         //     {
//         //         for (int j = 0; j < 28; j++)
//         //         {
//         //             // img[i*28 + j] < 128 ? img[i*28 + j] = 0 : img[i*28 + j] = 255;
//         //             printf("%d ", img[index3D(b,i,j,28,28)]);
//         //             // printf("%d ", img[i*28+j]);
//         //         }
//         //         printf("\n");
//         //     }
//         //     printf("\n\n");
//         // }

//         // for profiling Layers
//         float a,b,c,d,e,f,g,h,ii,j,k,l,m,n,o,p,q,r;
//         std::tie(a,b,c,d,e,f,g,h,ii,j,k,l,m,n,o,p,q,r) = predict_NeuralNet(img, output);
//         total_kernel_time += a; 
//         l1_time += b; l2_time += c; l3_time += d; l4_time += e; l5_time += f; l6_time += g; l8_time += h; l9_time += ii; l10_time += j; 
//         l1_ktime += k; l2_ktime += l; l4_ktime += m; l5_ktime += n; l8_ktime += o; l10_ktime += p;
//         malloc_time += q; cpy_time += r;

//         // total_kernel_time += predict_NeuralNet(img, output);
        
//         for(int b = 0; b < BATCH_SIZE; b++){
//             float max = output[b*OUT_SIZE];
//             int argmax = 0;
//             for (int j = 1; j < OUT_SIZE; j++) {
//                 if (output[b*OUT_SIZE + j] > max) {
//                     max = output[b*OUT_SIZE + j];
//                     argmax = j;
//                 }
//             }

//             if (argmax == label[b]) {
//                 matches[b]++;
//             }
//         }
        
//     }
//     auto end = std::chrono::high_resolution_clock::now();

//     float accuracy[BATCH_SIZE];
//     for(int b = 0; b < BATCH_SIZE; b++){
//         accuracy[b] = static_cast<float>(matches[b]) / (lsize/factor) * 100.f;
//         printf("Accuracy batch %d: %.1f%, Matches: %d/10000\n", b, accuracy[b],matches[b]);
//     }

//     auto total_cpu_time = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());
//     total_cpu_time -= total_kernel_time;
//     auto cpu_time = static_cast<float>(total_cpu_time) / (lsize/factor) / BATCH_SIZE;
//     auto kernel_time = static_cast<float>(total_kernel_time) / (lsize/factor) / BATCH_SIZE;

//     return std::make_tuple(accuracy, total_cpu_time, cpu_time, total_kernel_time, kernel_time, 
//         l1_time, l2_time, l3_time, l4_time, l5_time, l6_time, l8_time, l9_time, l10_time, l1_ktime, l2_ktime, l4_ktime, l5_ktime, l8_ktime, l10_ktime,
//         malloc_time, cpy_time);
// }

// int main() {

//     auto start = std::chrono::high_resolution_clock::now();
//     // load batches in a vector
//     std::vector<MNISTLoader> loaderx(BATCH_SIZE);
//     for(int i = 0; i < BATCH_SIZE; i++){
//         printf("Loading batch %d...",i);
//         loaderx[i] = MNISTLoader("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");
//         printf("loaded\n");
//     }
//     printf("\n");
//     auto end = std::chrono::high_resolution_clock::now();
//     auto batch_loading_time = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());
//     printf("Batch loading time: %.2f [s] => Latency: %.4f [s/batch]\n", batch_loading_time/1000.0f, batch_loading_time/BATCH_SIZE/1000.0f);
//     printf("\n");

//     auto results = benchmark(loaderx);

//     /*
//         For some reason, printing the accuracy here always leads to "0.0%"
//         Therefore it is printed in benchmark()
//         (if it is printed both in benchmark and here, both print the correct accuracy)
//     */
//     // for(int b = 0; b < BATCH_SIZE; b++){
//     //     printf("Accuracy batch %d: %.1f%\n", b, std::get<0>(results)[b]);
//     // }

//     printf("\n");
//     printf("Total CPU time: %.2f [s] => Latency: %.4f [ms/elem]\n", std::get<1>(results)/1000.0f, std::get<2>(results));
//     printf("Total GPU time: %.2f [s] => Latency: %.4f [ms/elem]\n", std::get<3>(results)/1000.0f, std::get<4>(results));
//     printf("\n");

//     // for profiling Layers
//     float l1_time = std::get<5>(results)/1000000000.0f; // ns / 1e9 -> s
//     float l2_time = std::get<6>(results)/1000000000.0f; // ns / 1e9 -> s
//     float l3_time = std::get<7>(results)/1000000000.0f; // ns / 1e9 -> s
//     float l4_time = std::get<8>(results)/1000000000.0f; // ns / 1e9 -> s
//     float l5_time = std::get<9>(results)/1000000000.0f; // ns / 1e9 -> s
//     float l6_time = std::get<10>(results)/1000000000.0f; // ns / 1e9 -> s
//     float l8_time = std::get<11>(results)/1000000000.0f; // ns / 1e9 -> s
//     float l9_time = std::get<12>(results)/1000000000.0f; // ns / 1e9 -> s
//     float l10_time = std::get<13>(results)/1000000000.0f; // ns / 1e9 -> s

//     float l1_ktime = std::get<14>(results)/1000.0f; // ms / 1e3 -> s
//     float l2_ktime = std::get<15>(results)/1000.0f; // ms / 1e3 -> s
//     float l4_ktime = std::get<16>(results)/1000.0f; // ms / 1e3 -> s
//     float l5_ktime = std::get<17>(results)/1000.0f; // ms / 1e3 -> s
//     float l8_ktime = std::get<18>(results)/1000.0f; // ms / 1e3 -> s
//     float l10_ktime = std::get<19>(results)/1000.0f; // ms / 1e3 -> s

//     float malloc_time = std::get<20>(results)/1000000000.0f; // ns / 1e9 -> s
//     float cpy_time = std::get<21>(results)/1000000000.0f; // ns / 1e9 -> s

//     float sum_l = l1_time + l2_time + l3_time + l4_time + l5_time + l6_time + l8_time + l9_time + l10_time;
//     float sum_kl = l1_ktime + l2_ktime + l4_ktime + l5_ktime + l8_ktime + l10_ktime;

//     // printf("%-15s %-10.2f [s], %-10s %-5.2f% => %-5s %-5.2f [s] %-10s %-5.2f%\n %-50s%-10s %-5.2f [s]\n %-50s%-10s %-5.2f [s]\n", "Layer 1 time:", l1_time, "Ratio:", (l1_time/sum_l)*100, "kernel:", l1_ktime, "kRatio:", (l1_ktime/sum_kl)*100,
//     //                                                                             "","=> malloc:", malloc_time, "","=> copy: ", cpy_time);
//     printf("%-15s %-10.2f [s], %-10s %-5.2f% => %-5s %-5.2f [s] %-10s %-5.2f%\n", "Layer 1 time:", l1_time, "Ratio:", (l1_time/sum_l)*100, "kernel:", l1_ktime, "kRatio:", (l1_ktime/sum_kl)*100);
//     printf("%-15s %-10.2f [s], %-10s %-5.2f% => %-5s %-5.2f [s] %-10s %-5.2f%\n", "Layer 2 time:", l2_time, "Ratio:", (l2_time/sum_l)*100, "kernel:", l2_ktime, "kRatio:", (l2_ktime/sum_kl)*100);
//     printf("%-15s %-10.2f [s], %-10s %-5.2f%\n", "Layer 3 time:", l3_time, "Ratio:", (l3_time/sum_l)*100);
//     printf("%-15s %-10.2f [s], %-10s %-5.2f% => %-5s %-5.2f [s] %-10s %-5.2f%\n", "Layer 4 time:", l4_time, "Ratio:", (l4_time/sum_l)*100, "kernel:", l4_ktime, "kRatio:", (l4_ktime/sum_kl)*100);
//     printf("%-15s %-10.2f [s], %-10s %-5.2f% => %-5s %-5.2f [s] %-10s %-5.2f%\n", "Layer 5 time:", l5_time, "Ratio:", (l5_time/sum_l)*100, "kernel:", l5_ktime, "kRatio:", (l5_ktime/sum_kl)*100);
//     printf("%-15s %-10.2f [s], %-10s %-5.2f%\n", "Layer 6 time:", l6_time, "Ratio:", (l6_time/sum_l)*100);
//     printf("%-15s %-10.2f [s], %-10s %-5.2f% => %-5s %-5.2f [s] %-10s %-5.2f%\n", "Layer 8 time:", l8_time, "Ratio:", (l8_time/sum_l)*100, "kernel:", l8_ktime, "kRatio:", (l8_ktime/sum_kl)*100);
//     printf("%-15s %-10.2f [s], %-10s %-5.2f%\n", "Layer 9 time:", l9_time, "Ratio:", (l9_time/sum_l)*100);
//     printf("%-15s %-10.2f [s], %-10s %-5.2f% => %-5s %-5.2f [s] %-10s %-5.2f%\n", "Layer 10 time:", l10_time, "Ratio:", (l10_time/sum_l)*100, "kernel:", l10_ktime, "kRatio:", (l10_ktime/sum_kl)*100);
//     printf("\n");
//     printf("%-15s %.2f [s]\n%-15s %.2f [s]\n", "Total time:", sum_l, "Total ktime:", sum_kl);

//     return 0;
// }

/* bnn_cpu flip */

auto benchmark(vector<MNISTLoader> &loaderx, bool verbose = false) {
#if defined BINARY || defined INT16
    int output[BATCH_SIZE*OUT_SIZE] = {0};
#else
    float output[BATCH_SIZE*OUT_SIZE] = {0};
#endif

    int factor = 1;
    int matches[1] = {0};
    int const imgsize = IMG_HEIGHT*IMG_WIDTH;

    ofstream g("original_img_2.out");

    size_t lsize = loaderx[0].size();
    // size_t lsize = 2; // for testing!

    float total_kernel_time = 0;
    float l1_time = 0, l2_time = 0, l3_time = 0, l4_time = 0, l5_time = 0, l6_time = 0, l8_time = 0, l9_time = 0, l10_time = 0;
    float l1_ktime = 0, l2_ktime = 0, l4_ktime = 0, l5_ktime = 0, l8_ktime = 0, l10_ktime = 0;
    float malloc_time = 0, cpy_time = 0;

    cout<<"Executing "<<lsize<<" images in "<<ceil(float(lsize)/BATCH_SIZE)<<" batches of "<<BATCH_SIZE<<"..."<<endl<<endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    /* using ceil() makes sure to execute even when division is not uniform: */
    for (unsigned int b = 0; b < ceil(float(lsize)/BATCH_SIZE); b+=factor) { // i := # image
        std::fill(output, output+OUT_SIZE*BATCH_SIZE, 0);
       
        unsigned char * img;
        img = (unsigned char*) malloc (BATCH_SIZE*imgsize);

        // load label i of corresponding image from every batch in an array
        int label[BATCH_SIZE];

        size_t bsize = (b == lsize/BATCH_SIZE) ? (lsize % BATCH_SIZE) : BATCH_SIZE; // tsize

        for(int i=0; i<bsize; i++){    // b := # batch
            for(int p=0; p<imgsize; p++){   // p := # pixel
                img[i*imgsize+p] = loaderx[0].images(b*BATCH_SIZE + i)[p]; 
            }
            label[i] = loaderx[0].labels(b*BATCH_SIZE + i); 
        }
        
        // // display img array
        // float sum = 0;
        // for(int i=0;i<bsize;i++){
        //     sum = 0;
        //     g<<"batch: "<<b<<", img: "<<b*BATCH_SIZE+i<<", label: "<<label[i]<<endl;
        //     cout<<"batch: "<<b<<", img: "<<b*BATCH_SIZE+i<<", label: "<<label[i]<<endl;
        //     for (int h = 0; h < 28; h++)
        //     {
        //         for (int w = 0; w < 28; w++)
        //         {
        //             g<<int(img[index3D(i,h,w,28,28)])<<" ";
        //             // cout<<int(img[index3D(i,h,w,28,28)])<<" ";
        //             sum += img[index3D(i,h,w,28,28)];
        //         }
        //         g<<endl;
        //         // cout<<endl;
        //     }
        //     g<<endl<<endl;
        //     // cout<<endl<<endl;
                
        //     g<<"batch: "<<b<<", img: "<<b*BATCH_SIZE+i<<", sum: "<<sum<<endl;
        //     cout<<"batch: "<<b<<", img: "<<b*BATCH_SIZE+i<<", sum: "<<sum<<endl;
        //     g<<endl<<endl<<endl;
        //     // cout<<endl<<endl<<endl;
        // }

        // for profiling Layers
        float a,bb,c,d,e,f,g,h,ii,j,k,l,m,n,o,p,q,r;
        std::tie(a,bb,c,d,e,f,g,h,ii,j,k,l,m,n,o,p,q,r) = predict_NeuralNet(img, output);
        total_kernel_time += a; 
        l1_time += bb; l2_time += c; l3_time += d; l4_time += e; l5_time += f; l6_time += g; l8_time += h; l9_time += ii; l10_time += j; 
        l1_ktime += k; l2_ktime += l; l4_ktime += m; l5_ktime += n; l8_ktime += o; l10_ktime += p;
        malloc_time += q; cpy_time += r;

        // total_kernel_time += predict_NeuralNet(img, output);
        
        for(int i = 0; i < bsize; i++){
            float max = output[i*OUT_SIZE];
            int argmax = 0;
            for (int j = 1; j < OUT_SIZE; j++) {
                if (output[i*OUT_SIZE + j] > max) {
                    max = output[i*OUT_SIZE + j];
                    argmax = j;
                }
            }

            if (argmax == label[i]) {
                matches[0]++;
            }
        }
        
    }
    auto end = std::chrono::high_resolution_clock::now();

    float accuracy[1];
    for(int b = 0; b < 1; b++){
        accuracy[b] = static_cast<float>(matches[b]) / (lsize/factor) * 100.f;
        printf("Accuracy batch %d: %.1f%, Matches: %d/10000\n", b, accuracy[b],matches[b]);
    }

    auto total_cpu_time = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());
    total_cpu_time -= total_kernel_time;
    auto cpu_time = static_cast<float>(total_cpu_time) / (lsize/factor) / 1;
    auto kernel_time = static_cast<float>(total_kernel_time) / (lsize/factor) / 1;

    return std::make_tuple(accuracy, total_cpu_time, cpu_time, total_kernel_time, kernel_time, 
        l1_time, l2_time, l3_time, l4_time, l5_time, l6_time, l8_time, l9_time, l10_time, l1_ktime, l2_ktime, l4_ktime, l5_ktime, l8_ktime, l10_ktime,
        malloc_time, cpy_time);
}

int main() {

    auto start = std::chrono::high_resolution_clock::now();
    // load batches in a vector
    std::vector<MNISTLoader> loaderx(1);
    for(int i = 0; i < 1; i++){
        printf("Loading dataset %d...",i);
        loaderx[i] = MNISTLoader("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");
        printf("loaded\n");
    }
    printf("\n");
    auto end = std::chrono::high_resolution_clock::now();
    auto dataset_loading_time = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());
    printf("Dataset loading time: %.2f [s] => Latency: %.4f [s/dataset]\n", dataset_loading_time/1000.0f, dataset_loading_time/1/1000.0f);
    printf("\n");

    auto results = benchmark(loaderx);

    /*
        For some reason, printing the accuracy here always leads to "0.0%"
        Therefore it is printed in benchmark()
        (if it is printed both in benchmark and here, both print the correct accuracy)
    */
    // for(int b = 0; b < BATCH_SIZE; b++){
    //     printf("Accuracy batch %d: %.1f%\n", b, std::get<0>(results)[b]);
    // }

    printf("\n");
    printf("Total CPU time: %.2f [s] => Latency: %.4f [ms/elem]\n", std::get<1>(results)/1000.0f, std::get<2>(results));
    printf("Total GPU time: %.2f [s] => Latency: %.4f [ms/elem]\n", std::get<3>(results)/1000.0f, std::get<4>(results));
    printf("\n");

    // for profiling Layers
    float l1_time = std::get<5>(results)/1000000000.0f; // ns / 1e9 -> s
    float l2_time = std::get<6>(results)/1000000000.0f; // ns / 1e9 -> s
    float l3_time = std::get<7>(results)/1000000000.0f; // ns / 1e9 -> s
    float l4_time = std::get<8>(results)/1000000000.0f; // ns / 1e9 -> s
    float l5_time = std::get<9>(results)/1000000000.0f; // ns / 1e9 -> s
    float l6_time = std::get<10>(results)/1000000000.0f; // ns / 1e9 -> s
    float l8_time = std::get<11>(results)/1000000000.0f; // ns / 1e9 -> s
    float l9_time = std::get<12>(results)/1000000000.0f; // ns / 1e9 -> s
    float l10_time = std::get<13>(results)/1000000000.0f; // ns / 1e9 -> s

    float l1_ktime = std::get<14>(results)/1000.0f; // ms / 1e3 -> s
    float l2_ktime = std::get<15>(results)/1000.0f; // ms / 1e3 -> s
    float l4_ktime = std::get<16>(results)/1000.0f; // ms / 1e3 -> s
    float l5_ktime = std::get<17>(results)/1000.0f; // ms / 1e3 -> s
    float l8_ktime = std::get<18>(results)/1000.0f; // ms / 1e3 -> s
    float l10_ktime = std::get<19>(results)/1000.0f; // ms / 1e3 -> s

    float malloc_time = std::get<20>(results)/1000000000.0f; // ns / 1e9 -> s
    float cpy_time = std::get<21>(results)/1000000000.0f; // ns / 1e9 -> s

    float sum_l = l1_time + l2_time + l3_time + l4_time + l5_time + l6_time + l8_time + l9_time + l10_time;
    float sum_kl = l1_ktime + l2_ktime + l4_ktime + l5_ktime + l8_ktime + l10_ktime;

    // printf("%-15s %-10.2f [s], %-10s %-5.2f% => %-5s %-5.2f [s] %-10s %-5.2f%\n %-50s%-10s %-5.2f [s]\n %-50s%-10s %-5.2f [s]\n", "Layer 1 time:", l1_time, "Ratio:", (l1_time/sum_l)*100, "kernel:", l1_ktime, "kRatio:", (l1_ktime/sum_kl)*100,
    //                                                                             "","=> malloc:", malloc_time, "","=> copy: ", cpy_time);
    printf("%-15s %-10.2f [s], %-10s %-5.2f% => %-5s %-5.2f [s] %-10s %-5.2f%\n", "Layer 1 time:", l1_time, "Ratio:", (l1_time/sum_l)*100, "kernel:", l1_ktime, "kRatio:", (l1_ktime/sum_kl)*100);
    printf("%-15s %-10.2f [s], %-10s %-5.2f% => %-5s %-5.2f [s] %-10s %-5.2f%\n", "Layer 2 time:", l2_time, "Ratio:", (l2_time/sum_l)*100, "kernel:", l2_ktime, "kRatio:", (l2_ktime/sum_kl)*100);
    printf("%-15s %-10.2f [s], %-10s %-5.2f%\n", "Layer 3 time:", l3_time, "Ratio:", (l3_time/sum_l)*100);
    printf("%-15s %-10.2f [s], %-10s %-5.2f% => %-5s %-5.2f [s] %-10s %-5.2f%\n", "Layer 4 time:", l4_time, "Ratio:", (l4_time/sum_l)*100, "kernel:", l4_ktime, "kRatio:", (l4_ktime/sum_kl)*100);
    printf("%-15s %-10.2f [s], %-10s %-5.2f% => %-5s %-5.2f [s] %-10s %-5.2f%\n", "Layer 5 time:", l5_time, "Ratio:", (l5_time/sum_l)*100, "kernel:", l5_ktime, "kRatio:", (l5_ktime/sum_kl)*100);
    printf("%-15s %-10.2f [s], %-10s %-5.2f%\n", "Layer 6 time:", l6_time, "Ratio:", (l6_time/sum_l)*100);
    printf("%-15s %-10.2f [s], %-10s %-5.2f% => %-5s %-5.2f [s] %-10s %-5.2f%\n", "Layer 8 time:", l8_time, "Ratio:", (l8_time/sum_l)*100, "kernel:", l8_ktime, "kRatio:", (l8_ktime/sum_kl)*100);
    printf("%-15s %-10.2f [s], %-10s %-5.2f%\n", "Layer 9 time:", l9_time, "Ratio:", (l9_time/sum_l)*100);
    printf("%-15s %-10.2f [s], %-10s %-5.2f% => %-5s %-5.2f [s] %-10s %-5.2f%\n", "Layer 10 time:", l10_time, "Ratio:", (l10_time/sum_l)*100, "kernel:", l10_ktime, "kRatio:", (l10_ktime/sum_kl)*100);
    printf("\n");
    printf("%-15s %.2f [s]\n%-15s %.2f [s]\n", "Total time:", sum_l, "Total ktime:", sum_kl);

    return 0;
}
