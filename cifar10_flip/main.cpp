/*
    I believe these args were used in generating the code:
    --base-implementation: cpp.NHWC
    --implementation: cpp.binary
*/

/*
    Run with: 
    $ make
    $ ./cifar.o
*/

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

auto benchmark(bool verbose = false) {
#if defined BINARY || defined INT16
    int output[OUT_SIZE*BATCH_SIZE] = {0};
#else
    float output[OUT_SIZE*BATCH_SIZE] = {0};
#endif

    /* load dataset in a vector ->
     * stuff like [1] or for-loop with only 1 run-through are remains from code-porting from previous batch-type implementation 
     */
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
    // ofstream g("original_img_check.out");

    float total_kernel_time = 0;

    cout<<"Executing "<<tsize<<" images in "<<ceil(float(tsize)/BATCH_SIZE)<<" batches of "<<BATCH_SIZE<<"..."<<endl<<endl;

    start = std::chrono::high_resolution_clock::now();
    /* using ceil() makes sure to execute even when division is not uniform: */
    for (int b = 0; b < ceil(float(tsize)/BATCH_SIZE); b+=factor) { // b := execute BATCH_SIZE number of images at a time

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
        size_t bsize = (b == tsize/BATCH_SIZE) ? (tsize % BATCH_SIZE) : BATCH_SIZE;

        for(int i = 0; i < bsize; i++){
            for (int j = 0; j < test_images[0][b*BATCH_SIZE+i].size(); j++) {
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

        total_kernel_time += predict_NeuralNet(img, output);

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
    end = std::chrono::high_resolution_clock::now();
    
    /* stuff like [1] or for-loop with only 1 run-through are remains from code-porting from previous batch-type implementation */
    float accuracy[1];
    for(int b = 0; b < 1; b++){
        accuracy[b] = static_cast<float>(matches[b]) / (tsize/factor) * 100.f;
        printf("Accuracy dataset %d: %.1f%, Matches: %d/10000\n", b, accuracy[b],matches[b]);
    }

    auto total_cpu_time = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());
    total_cpu_time -= total_kernel_time;
    auto cpu_time = static_cast<float>(total_cpu_time) / (tsize/factor) / 1;
    auto kernel_time = static_cast<float>(total_kernel_time) / (tsize/factor) / 1;

    return std::make_tuple(accuracy, total_cpu_time, cpu_time, total_kernel_time, kernel_time);
}

int main() {
    
    auto results = benchmark();

    printf("\n");
    printf("Total CPU time: %.2f [s] => Latency: %.4f [ms/elem]\n", std::get<1>(results)/1000.0f, std::get<2>(results));
    printf("Total GPU time: %.2f [s] => Latency: %.4f [ms/elem]\n", std::get<3>(results)/1000.0f, std::get<4>(results));
    printf("\n");

    return 0;
}
