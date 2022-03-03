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

    // load batches in a vector
    auto start = std::chrono::high_resolution_clock::now();
    // std::vector<cifar::CIFAR10_dataset<std::vector, std::vector<uint8_t>, uint8_t>> dataset(BATCH_SIZE);
    std::vector<std::vector<std::vector<uint8_t>>> test_images(BATCH_SIZE);
    std::vector<std::vector<uint8_t>> test_labels(BATCH_SIZE);
    for(int b = 0; b < BATCH_SIZE; b++){
        printf("Loading batch %d...",b);
        // dataset[b] = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
        test_images[b] = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>().test_images;
        test_labels[b] = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>().test_labels;
        printf("loaded\n");
    }
    printf("\n");
    auto end = std::chrono::high_resolution_clock::now();
    auto batch_loading_time = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());
    printf("Batch loading time: %.2f [s] => Latency: %.4f [s/batch]\n", batch_loading_time/1000.0f, batch_loading_time/BATCH_SIZE/1000.0f);
    printf("\n");

    int factor = 1;
    int matches[BATCH_SIZE] = {0};
    int const imgsize = IMG_HEIGHT*IMG_WIDTH;

    size_t tsize = test_images[0].size();
    // size_t tsize = 1; // for testing!

    float total_kernel_time = 0;

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < tsize; i+=factor) {

        int label[BATCH_SIZE];
        unsigned char img[BATCH_SIZE][32][32][3];

        /* leads to stack smashing */
        // unsigned char * img;
        // img = (unsigned char*) malloc (BATCH_SIZE*imgsize*NR_CHANNELS);

        for(int b = 0; b < BATCH_SIZE; b++){
            for (int j = 0; j < test_images[b][i].size(); j++) {
                int d3 = j / 1024;
                int minus = j % 1024;
                int d2 = minus % 32;
                int d1 = minus / 32;
                img[b][d1][d2][d3] = static_cast<unsigned char>(test_images[b][i][j]); // img[index4D(b,d1,d2,d3,32,32,3)] / img[b][d1][d2][d3]
            }
            
            std::fill(output, output+OUT_SIZE*BATCH_SIZE, 0);
            label[b] = static_cast<int>(test_labels[b][i]);
        }

        // display img array
        // ofstream g("original_img_1.out");
        // for(int b=0;b<BATCH_SIZE;b++){
        //     for(int c=0;c<NR_CHANNELS;c++){
        //         // g<<"batch: "<<b<<", label: "<<label[b]<<", channel: "<<c<<endl;
        //         cout<<"batch: "<<b<<", label: "<<label[b]<<", channel: "<<c<<endl;
        //         for (int i = 0; i < 32; i++)
        //         {
        //             for (int j = 0; j < 32; j++)
        //             {
        //                 // g<<int(img[index4D(b,i,j,c,32,32,3)])<<" ";
        //                 // g<<int(img[b][i][j][c])<<" ";
        //                 // cout<<int(img[index4D(b,i,j,c,32,32,3)])<<" ";
        //                 cout<<int(img[b][i][j][c])<<" ";
        //             }
        //             // g<<endl;
        //             cout<<endl;
        //         }
        //         // g<<endl<<endl;
        //         cout<<endl<<endl;
        //     }
        //     // g<<endl<<endl<<endl;
        //     cout<<endl<<endl<<endl;
        // }
        total_kernel_time += predict_NeuralNet(img, output);

        for(int b = 0; b < BATCH_SIZE; b++){ 
            float max = output[b*OUT_SIZE];
            int argmax = 0;
            for (int j = 1; j < OUT_SIZE; j++) {
                if (output[b*OUT_SIZE + j] > max) {
                    max = output[b*OUT_SIZE + j];
                    argmax = j;
                }
            }
            if (argmax == label[b]) {
                matches[b]++;
            }
        }
    }
    end = std::chrono::high_resolution_clock::now();
    
    float accuracy[BATCH_SIZE];
    for(int b = 0; b < BATCH_SIZE; b++){
        accuracy[b] = static_cast<float>(matches[b]) / (tsize/factor) * 100.f;
        printf("Accuracy batch %d: %.1f%, Matches: %d/10000\n", b, accuracy[b],matches[b]);
    }

    auto total_cpu_time = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());
    total_cpu_time -= total_kernel_time;
    auto cpu_time = static_cast<float>(total_cpu_time) / (tsize/factor) / BATCH_SIZE;
    auto kernel_time = static_cast<float>(total_kernel_time) / (tsize/factor) / BATCH_SIZE;

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
