// /*
//     I believe these args were used in generating the code:
//     --base-implementation: cpp.NHWC
//     --implementation: cpp.binary
// */

#include <iostream>
#include <chrono>
#include <algorithm>
#include <tuple>
#include <cmath>

#include "utils.h"
#include "cifar_reader/cifar10_reader.hpp"

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

/* cifar10 normal */

// auto benchmark(bool verbose = false) {
// #if defined BINARY || defined INT16
//     int output[OUT_SIZE*BATCH_SIZE] = {0};
// #else
//     float output[OUT_SIZE*BATCH_SIZE] = {0};
// #endif

//     // load batches in a vector
//     auto start = std::chrono::high_resolution_clock::now();
//     // std::vector<cifar::CIFAR10_dataset<std::vector, std::vector<uint8_t>, uint8_t>> dataset(BATCH_SIZE);
//     std::vector<std::vector<std::vector<uint8_t>>> test_images(BATCH_SIZE);
//     std::vector<std::vector<uint8_t>> test_labels(BATCH_SIZE);
//     for(int b = 0; b < BATCH_SIZE; b++){
//         printf("Loading batch %d...",b);
//         // dataset[b] = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
//         test_images[b] = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>().test_images;
//         test_labels[b] = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>().test_labels;
//         printf("loaded\n");
//     }
//     printf("\n");
//     auto end = std::chrono::high_resolution_clock::now();
//     auto batch_loading_time = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());
//     printf("Batch loading time: %.2f [s] => Latency: %.4f [s/batch]\n", batch_loading_time/1000.0f, batch_loading_time/BATCH_SIZE/1000.0f);
//     printf("\n");

//     int factor = 1;
//     int matches[BATCH_SIZE] = {0};
//     int const imgsize = IMG_HEIGHT*IMG_WIDTH;
//     size_t tsize = test_images[0].size();

//     float total_kernel_time = 0;
//     float l1_time = 0, l2_time = 0, l3_time = 0, l4_time = 0, l5_time = 0, l6_time = 0, l7_time = 0, l8_time = 0, l9_time = 0, l10_time = 0, l11_time = 0, l12_time = 0, l13_time = 0, l14_time = 0, l15_time = 0, l17_time = 0, l18_time = 0, l19_time = 0;
//     float l1_kernel_time = 0, l3_kernel_time = 0, l4_kernel_time = 0, l6_kernel_time = 0, l8_kernel_time = 0, l9_kernel_time = 0, l11_kernel_time = 0, l13_kernel_time = 0, l14_kernel_time = 0, l17_kernel_time = 0, l19_kernel_time = 0;

//     start = std::chrono::high_resolution_clock::now();
//     for (int i = 0; i < tsize; i+=factor) { // tsize

//         int label[BATCH_SIZE];
//         unsigned char img[BATCH_SIZE][32][32][3];

//         /* leads to stack smashing */
//         // unsigned char * img;
//         // img = (unsigned char*) malloc (BATCH_SIZE*imgsize*NR_CHANNELS);

//         for(int b = 0; b < BATCH_SIZE; b++){
//             for (int j = 0; j < test_images[b][i].size(); j++) { // DO NOT USE tsize HERE! CRUICAL!
//                 int d3 = j / 1024;
//                 int minus = j % 1024;
//                 int d2 = minus % 32;
//                 int d1 = minus / 32;
//                 img[b][d1][d2][d3] = static_cast<unsigned char>(test_images[b][i][j]); // img[index4D(b,d1,d2,d3,32,32,3)] / img[b][d1][d2][d3]
//             }
            
//             std::fill(output, output+OUT_SIZE*BATCH_SIZE, 0);
//             label[b] = static_cast<int>(test_labels[b][i]);
//         }

//         // display img array
//         // ofstream g("original_img_1.out");
//         // for(int b=0;b<BATCH_SIZE;b++){
//         //     for(int c=0;c<NR_CHANNELS;c++){
//         //         // g<<"batch: "<<b<<", label: "<<label[b]<<", channel: "<<c<<endl;
//         //         cout<<"batch: "<<b<<", label: "<<label[b]<<", channel: "<<c<<endl;
//         //         for (int i = 0; i < 32; i++)
//         //         {
//         //             for (int j = 0; j < 32; j++)
//         //             {
//         //                 // g<<int(img[index4D(b,i,j,c,32,32,3)])<<" ";
//         //                 // g<<int(img[b][i][j][c])<<" ";
//         //                 // cout<<int(img[index4D(b,i,j,c,32,32,3)])<<" ";
//         //                 cout<<int(img[b][i][j][c])<<" ";
//         //             }
//         //             // g<<endl;
//         //             cout<<endl;
//         //         }
//         //         // g<<endl<<endl;
//         //         cout<<endl<<endl;
//         //     }
//         //     // g<<endl<<endl<<endl;
//         //     cout<<endl<<endl<<endl;
//         // }

//         // // for profiling layers

//         float a,b,c,d,e,f,g,h,ii,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,aa,bb,cc,dd;
//         std::tie(a,b,c,d,e,f,g,h,ii,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,aa,bb,cc,dd) = predict_NeuralNet(img, output);
//         total_kernel_time += a;
//         l1_time += b; l2_time += c; l3_time += d; l4_time += e; l5_time += f; l6_time += g; l7_time += h; l8_time += ii; l9_time += j;
//         l10_time += k; l11_time += l; l12_time += m; l13_time += n; l14_time += o; l15_time += p; l17_time += q; l18_time += r; l19_time += s;
//         l1_kernel_time += t; l3_kernel_time += u; l4_kernel_time += v; l6_kernel_time += w; l8_kernel_time += x; l9_kernel_time += y;
//         l11_kernel_time += z; l13_kernel_time += aa; l14_kernel_time += bb; l17_kernel_time += cc; l19_kernel_time += dd;

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
//             // cout<<i<<": "<<argmax<<"=="<<label[b]<<endl;
//             if (argmax == label[b]) {
//                 // cout<<matches[b]<<": "<<argmax<<"=="<<label[b]<<endl;
//                 matches[b]++;
//             }
//         }
//         // printf("\n\n");
//     }
//     end = std::chrono::high_resolution_clock::now();
    
//     float accuracy[BATCH_SIZE];
//     for(int b = 0; b < BATCH_SIZE; b++){
//         accuracy[b] = static_cast<float>(matches[b]) / (tsize/factor) * 100.f;
//         printf("Accuracy batch %d: %.1f%, Matches: %d/10000\n", b, accuracy[b],matches[b]);
//     }

//     auto total_cpu_time = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());
//     total_cpu_time -= total_kernel_time;
//     auto cpu_time = static_cast<float>(total_cpu_time) / (tsize/factor) / BATCH_SIZE;
//     auto kernel_time = static_cast<float>(total_kernel_time) / (tsize/factor) / BATCH_SIZE;

//     return std::make_tuple(accuracy, total_cpu_time, cpu_time, total_kernel_time, kernel_time,
//         l1_time, l2_time, l3_time, l4_time, l5_time, l6_time, l7_time, l8_time, l9_time, 
//         l10_time, l11_time, l12_time, l13_time, l14_time, l15_time, l17_time, l18_time, l19_time,
//         l1_kernel_time, l3_kernel_time, l4_kernel_time, l6_kernel_time, l8_kernel_time, l9_kernel_time, 
//         l11_kernel_time, l13_kernel_time, l14_kernel_time, l17_kernel_time, l19_kernel_time);
// }

/* cifar10_flip */

auto benchmark(bool verbose = false) {
#if defined BINARY || defined INT16
    int output[OUT_SIZE*BATCH_SIZE] = {0};
#else
    float output[OUT_SIZE*BATCH_SIZE] = {0};
#endif

    // load batches in a vector
    auto start = std::chrono::high_resolution_clock::now();
    // std::vector<cifar::CIFAR10_dataset<std::vector, std::vector<uint8_t>, uint8_t>> dataset(1);
    std::vector<std::vector<std::vector<uint8_t>>> test_images(1);
    std::vector<std::vector<uint8_t>> test_labels(1);
    for(int b = 0; b < 1; b++){
        printf("Loading dataset %d...",b);
        // dataset[b] = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
        test_images[b] = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>().test_images;
        test_labels[b] = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>().test_labels;
        printf("loaded\n");
    }
    printf("\n");
    auto end = std::chrono::high_resolution_clock::now();
    auto dataset_loading_time = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());
    printf("Dataset loading time: %.2f [s] => Latency: %.4f [s/dataset]\n", dataset_loading_time/1000.0f, dataset_loading_time/1/1000.0f);
    printf("\n");

    int factor = 1;
    int matches[1] = {0};
    int const imgsize = IMG_HEIGHT*IMG_WIDTH;

    size_t tsize = test_images[0].size();
    // size_t tsize = 2; // for testing!

    float total_kernel_time = 0;
    float l1_time = 0, l2_time = 0, l3_time = 0, l4_time = 0, l5_time = 0, l6_time = 0, l7_time = 0, l8_time = 0, l9_time = 0, l10_time = 0, l11_time = 0, l12_time = 0, l13_time = 0, l14_time = 0, l15_time = 0, l17_time = 0, l18_time = 0, l19_time = 0;
    float l1_kernel_time = 0, l3_kernel_time = 0, l4_kernel_time = 0, l6_kernel_time = 0, l8_kernel_time = 0, l9_kernel_time = 0, l11_kernel_time = 0, l13_kernel_time = 0, l14_kernel_time = 0, l17_kernel_time = 0, l19_kernel_time = 0;

    ofstream g("original_img_check.out");

    cout<<"Executing "<<tsize<<" images in "<<ceil(float(tsize)/BATCH_SIZE)<<" batches of "<<BATCH_SIZE<<"..."<<endl<<endl;

    start = std::chrono::high_resolution_clock::now();
    /* using ceil() makes sure to execute even when division is not uniform: */
    for (int b = 0; b < ceil(float(tsize)/BATCH_SIZE); b+=factor) { // tsize/BATCH_SIZE

        int label[BATCH_SIZE];
        unsigned char img[BATCH_SIZE][32][32][3];

        /* leads to stack smashing */
        // unsigned char * img;
        // img = (unsigned char*) malloc (BATCH_SIZE*imgsize*NR_CHANNELS);

        /* 
         * in case the division of tsize to BATCH_SIZE is not uniform:
         * -> the last batch only has to execute a number of (tsize % BATCH_SIZE) images
         * else -> it executes a number of BATCH_SIZE images as usual
         */
        size_t bsize = (b == tsize/BATCH_SIZE) ? (tsize % BATCH_SIZE) : BATCH_SIZE; // tsize

        for(int i = 0; i < bsize; i++){
            for (int j = 0; j < test_images[0][b*BATCH_SIZE+i].size(); j++) { // DO NOT USE tsize HERE! CRUICAL!
                int d3 = j / 1024;
                int minus = j % 1024;
                int d2 = minus % 32;
                int d1 = minus / 32;
                img[i][d1][d2][d3] = static_cast<unsigned char>(test_images[0][b*BATCH_SIZE+i][j]); // img[index4D(b,d1,d2,d3,32,32,3)] / img[b][d1][d2][d3]
            }
            
            std::fill(output, output+OUT_SIZE*BATCH_SIZE, 0);
            label[i] = static_cast<int>(test_labels[0][b*BATCH_SIZE+i]);
        }

        // // display img array
        // float sum = 0;
        // for(int i=0;i<bsize;i++){
        //     sum = 0;
        //     for(int c=0;c<NR_CHANNELS;c++){
        //         g<<"batch: "<<b<<", img: "<<b*BATCH_SIZE+i<<", channel: "<<c<<endl;
        //         // cout<<"batch: "<<b<<", img: "<<b*BATCH_SIZE+i<<", channel: "<<c<<endl;
        //         for (int h = 0; h < 32; h++)
        //         {
        //             for (int w = 0; w < 32; w++)
        //             {
        //                 // g<<int(img[index4D(i,h,w,c,32,32,3)])<<" ";
        //                 g<<int(img[i][h][w][c])<<" ";
        //                 // cout<<int(img[index4D(i,h,w,c,32,32,3)])<<" ";
        //                 // cout<<int(img[i][h][w][c])<<" ";
        //                 sum += img[i][h][w][c];
        //             }
        //             g<<endl;
        //             // cout<<endl;
        //         }
        //         g<<endl<<endl;
        //         // cout<<endl<<endl;
        //     }
        //     g<<"batch: "<<b<<", img: "<<b*BATCH_SIZE+i<<", sum: "<<sum<<endl;
        //     cout<<"batch: "<<b<<", img: "<<b*BATCH_SIZE+i<<", sum: "<<sum<<endl;
        //     g<<endl<<endl<<endl;
        //     // cout<<endl<<endl<<endl;
        // }
        // cout<<endl;

        // cout<<"batch: "<<b<<":"<<endl;
        // total_kernel_time += predict_NeuralNet(img, output);

        float a,bb,c,d,e,f,g,h,ii,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,aa,bbb,cc,dd;
        std::tie(a,bb,c,d,e,f,g,h,ii,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,aa,bbb,cc,dd) = predict_NeuralNet(img, output);
        total_kernel_time += a;
        l1_time += bb; l2_time += c; l3_time += d; l4_time += e; l5_time += f; l6_time += g; l7_time += h; l8_time += ii; l9_time += j;
        l10_time += k; l11_time += l; l12_time += m; l13_time += n; l14_time += o; l15_time += p; l17_time += q; l18_time += r; l19_time += s;
        l1_kernel_time += t; l3_kernel_time += u; l4_kernel_time += v; l6_kernel_time += w; l8_kernel_time += x; l9_kernel_time += y;
        l11_kernel_time += z; l13_kernel_time += aa; l14_kernel_time += bbb; l17_kernel_time += cc; l19_kernel_time += dd;

        
        // for(int i = 0; i < bsize; i++){ 
        //     cout<<"img: "<<b*BATCH_SIZE+i<<": ";
        //     for(int j=0;j<10;j++){
        //         cout<<output[i*10 + j]<<", ";
        //     }
        //     printf("\n");
        // }
        // cout<<endl;

        for(int i = 0; i < bsize; i++){ 
            float max = output[i*OUT_SIZE];
            int argmax = 0;
            for (int j = 1; j < OUT_SIZE; j++) {
                if (output[i*OUT_SIZE + j] > max) {
                    max = output[i*OUT_SIZE + j];
                    argmax = j;
                }
            }
            // cout<<i<<": "<<argmax<<"=="<<label[b]<<endl;
            if (argmax == label[i]) {
                // cout<<matches[b]<<": "<<argmax<<"=="<<label[b]<<endl;
                matches[0]++;
            }
        }
        // printf("\n\n");
    }
    end = std::chrono::high_resolution_clock::now();
    
    float accuracy[1];
    for(int b = 0; b < 1; b++){
        accuracy[b] = static_cast<float>(matches[b]) / (tsize/factor) * 100.f;
        printf("Accuracy dataset %d: %.1f%, Matches: %d/10000\n", b, accuracy[b],matches[b]);
    }

    auto total_cpu_time = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());
    total_cpu_time -= total_kernel_time;
    auto cpu_time = static_cast<float>(total_cpu_time) / (tsize/factor) / 1;
    auto kernel_time = static_cast<float>(total_kernel_time) / (tsize/factor) / 1;

    // return std::make_tuple(accuracy, total_cpu_time, cpu_time, total_kernel_time, kernel_time);
    return std::make_tuple(accuracy, total_cpu_time, cpu_time, total_kernel_time, kernel_time,
        l1_time, l2_time, l3_time, l4_time, l5_time, l6_time, l7_time, l8_time, l9_time, 
        l10_time, l11_time, l12_time, l13_time, l14_time, l15_time, l17_time, l18_time, l19_time,
        l1_kernel_time, l3_kernel_time, l4_kernel_time, l6_kernel_time, l8_kernel_time, l9_kernel_time, 
        l11_kernel_time, l13_kernel_time, l14_kernel_time, l17_kernel_time, l19_kernel_time);
}

int main() {
    
    auto results = benchmark();
    printf("\n");
    printf("Total CPU time: %.2f [s] => Latency: %.4f [ms/elem]\n", std::get<1>(results)/1000.0f, std::get<2>(results));
    printf("Total GPU time: %.2f [s] => Latency: %.4f [ms/elem]\n", std::get<3>(results)/1000.0f, std::get<4>(results));
    printf("\n");

    // for profiling layers
    float l1_time = std::get<5>(results)/1000000000.0f; // ns / 1e9 -> s
    float l2_time = std::get<6>(results)/1000000000.0f; // ns / 1e9 -> s
    float l3_time = std::get<7>(results)/1000000000.0f; // ns / 1e9 -> s
    float l4_time = std::get<8>(results)/1000000000.0f; // ns / 1e9 -> s
    float l5_time = std::get<9>(results)/1000000000.0f; // ns / 1e9 -> s
    float l6_time = std::get<10>(results)/1000000000.0f; // ns / 1e9 -> s
    float l7_time = std::get<11>(results)/1000000000.0f; // ns / 1e9 -> s
    float l8_time = std::get<12>(results)/1000000000.0f; // ns / 1e9 -> s
    float l9_time = std::get<13>(results)/1000000000.0f; // ns / 1e9 -> s
    float l10_time = std::get<14>(results)/1000000000.0f; // ns / 1e9 -> s
    float l11_time = std::get<15>(results)/1000000000.0f; // ns / 1e9 -> s
    float l12_time = std::get<16>(results)/1000000000.0f; // ns / 1e9 -> s
    float l13_time = std::get<17>(results)/1000000000.0f; // ns / 1e9 -> s
    float l14_time = std::get<18>(results)/1000000000.0f; // ns / 1e9 -> s
    float l15_time = std::get<19>(results)/1000000000.0f; // ns / 1e9 -> s
    float l17_time = std::get<20>(results)/1000000000.0f; // ns / 1e9 -> s
    float l18_time = std::get<21>(results)/1000000000.0f; // ns / 1e9 -> s
    float l19_time = std::get<22>(results)/1000000000.0f; // ns / 1e9 -> s

    float l1_ktime = std::get<23>(results)/1000.0f; // ms / 1e3 -> s
    float l3_ktime = std::get<24>(results)/1000.0f; // ms / 1e3 -> s
    float l4_ktime = std::get<25>(results)/1000.0f; // ms / 1e3 -> s
    float l6_ktime = std::get<26>(results)/1000.0f; // ms / 1e3 -> s
    float l8_ktime = std::get<27>(results)/1000.0f; // ms / 1e3 -> s
    float l9_ktime = std::get<28>(results)/1000.0f; // ms / 1e3 -> s
    float l11_ktime = std::get<29>(results)/1000.0f; // ms / 1e3 -> s
    float l13_ktime = std::get<30>(results)/1000.0f; // ms / 1e3 -> s
    float l14_ktime = std::get<31>(results)/1000.0f; // ms / 1e3 -> s
    float l17_ktime = std::get<32>(results)/1000.0f; // ms / 1e3 -> s
    float l19_ktime = std::get<33>(results)/1000.0f; // ms / 1e3 -> s

    float sum_l = l1_time + l2_time + l3_time + l4_time + l5_time + l6_time + l7_time + l8_time + l9_time + l10_time + l11_time + l12_time + l13_time + l14_time + l15_time + l17_time + l18_time + l19_time;
    float sum_kl = l1_ktime + l3_ktime + l4_ktime + l6_ktime + l8_ktime + l9_ktime + l11_ktime + l13_ktime + l14_ktime + l17_ktime + l19_ktime;

    printf("%-15s %-10.2f [s], %-10s %-5.2f% => %-5s %-5.2f [s] %-10s %-5.2f%\n", "Layer 1 time:", l1_time, "Ratio:", (l1_time/sum_l)*100, "kernel:", l1_ktime, "kRatio:", (l1_ktime/sum_kl)*100);
    printf("%-15s %-10.2f [s], %-10s %-5.2f%\n", "Layer 2 time:", l2_time, "Ratio:", (l2_time/sum_l)*100);
    printf("%-15s %-10.2f [s], %-10s %-5.2f% => %-5s %-5.2f [s] %-10s %-5.2f%\n", "Layer 3 time:", l3_time, "Ratio:", (l3_time/sum_l)*100, "kernel:", l3_ktime, "kRatio:", (l3_ktime/sum_kl)*100);
    printf("%-15s %-10.2f [s], %-10s %-5.2f% => %-5s %-5.2f [s] %-10s %-5.2f%\n", "Layer 4 time:", l4_time, "Ratio:", (l4_time/sum_l)*100, "kernel:", l4_ktime, "kRatio:", (l4_ktime/sum_kl)*100);
    printf("%-15s %-10.2f [s], %-10s %-5.2f%\n", "Layer 5 time:", l5_time, "Ratio:", (l5_time/sum_l)*100);
    printf("%-15s %-10.2f [s], %-10s %-5.2f% => %-5s %-5.2f [s] %-10s %-5.2f%\n", "Layer 6 time:", l6_time, "Ratio:", (l6_time/sum_l)*100, "kernel:", l6_ktime, "kRatio:", (l6_ktime/sum_kl)*100);
    printf("%-15s %-10.2f [s], %-10s %-5.2f%\n", "Layer 7 time:", l7_time, "Ratio:", (l7_time/sum_l)*100);
    printf("%-15s %-10.2f [s], %-10s %-5.2f% => %-5s %-5.2f [s] %-10s %-5.2f%\n", "Layer 8 time:", l8_time, "Ratio:", (l8_time/sum_l)*100, "kernel:", l8_ktime, "kRatio:", (l8_ktime/sum_kl)*100);
    printf("%-15s %-10.2f [s], %-10s %-5.2f% => %-5s %-5.2f [s] %-10s %-5.2f%\n", "Layer 9 time:", l9_time, "Ratio:", (l9_time/sum_l)*100, "kernel:", l9_ktime, "kRatio:", (l9_ktime/sum_kl)*100);
    printf("%-15s %-10.2f [s], %-10s %-5.2f%\n", "Layer 10 time:", l10_time, "Ratio:", (l10_time/sum_l)*100); 
    printf("%-15s %-10.2f [s], %-10s %-5.2f% => %-5s %-5.2f [s] %-10s %-5.2f%\n", "Layer 11 time:", l11_time, "Ratio:", (l11_time/sum_l)*100, "kernel:", l11_ktime, "kRatio:", (l11_ktime/sum_kl)*100);
    printf("%-15s %-10.2f [s], %-10s %-5.2f%\n", "Layer 12 time:", l12_time, "Ratio:", (l12_time/sum_l)*100);
    printf("%-15s %-10.2f [s], %-10s %-5.2f% => %-5s %-5.2f [s] %-10s %-5.2f%\n", "Layer 13 time:", l13_time, "Ratio:", (l13_time/sum_l)*100, "kernel:", l13_ktime, "kRatio:", (l13_ktime/sum_kl)*100);
    printf("%-15s %-10.2f [s], %-10s %-5.2f% => %-5s %-5.2f [s] %-10s %-5.2f%\n", "Layer 14 time:", l14_time, "Ratio:", (l14_time/sum_l)*100, "kernel:", l14_ktime, "kRatio:", (l14_ktime/sum_kl)*100);
    printf("%-15s %-10.2f [s], %-10s %-5.2f%\n", "Layer 15 time:", l15_time, "Ratio:", (l15_time/sum_l)*100);
    printf("%-15s %-10.2f [s], %-10s %-5.2f% => %-5s %-5.2f [s] %-10s %-5.2f%\n", "Layer 17 time:", l17_time, "Ratio:", (l17_time/sum_l)*100, "kernel:", l17_ktime, "kRatio:", (l17_ktime/sum_kl)*100);
    printf("%-15s %-10.2f [s], %-10s %-5.2f%\n", "Layer 18 time:", l18_time, "Ratio:", (l18_time/sum_l)*100);
    printf("%-15s %-10.2f [s], %-10s %-5.2f% => %-5s %-5.2f [s] %-10s %-5.2f%\n", "Layer 19 time:", l19_time, "Ratio:", (l19_time/sum_l)*100, "kernel:", l19_ktime, "kRatio:", (l19_ktime/sum_kl)*100);
    printf("\n");
    printf("%-15s %.2f [s]\n%-15s %.2f [s]\n", "Total time:", sum_l, "Total ktime:", sum_kl);
    

    return 0;
}
