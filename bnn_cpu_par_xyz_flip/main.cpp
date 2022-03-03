/*
    Run with: 
    $ make
    $ ./parxyz.o
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

auto benchmark(vector<MNISTLoader> &loaderx, bool verbose = false) {
#if defined BINARY || defined INT16
    int output[BATCH_SIZE*OUT_SIZE] = {0};
#else
    float output[BATCH_SIZE*OUT_SIZE] = {0};
#endif

    int factor = 1;
    int matches[1] = {0};
    int const imgsize = IMG_HEIGHT*IMG_WIDTH;

    size_t lsize = loaderx[0].size();
    // size_t lsize = 2;                // for testing!
    // ofstream g("original_img_2.out");

    float total_kernel_time = 0;

    cout<<"Executing "<<lsize<<" images in "<<ceil(float(lsize)/BATCH_SIZE)<<" batches of "<<BATCH_SIZE<<"..."<<endl<<endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    /* using ceil() makes sure to execute even when division is not uniform: */
    for (unsigned int b = 0; b < ceil(float(lsize)/BATCH_SIZE); b+=factor) { // b := execute BATCH_SIZE number of images at a time
        std::fill(output, output+OUT_SIZE*BATCH_SIZE, 0);
       
        unsigned char * img;
        img = (unsigned char*) malloc (BATCH_SIZE*imgsize);

        // load label i of corresponding image from every batch in an array
        int label[BATCH_SIZE];

        size_t bsize = (b == lsize/BATCH_SIZE) ? (lsize % BATCH_SIZE) : BATCH_SIZE; // tsize

        for(int i=0; i<bsize; i++){         // i := # batch
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
    auto end = std::chrono::high_resolution_clock::now();

    /* stuff like [1] or for-loop with only 1 run-through are remains from code-porting from previous batch-type implementation */
    float accuracy[1];
    for(int b = 0; b < 1; b++){
        accuracy[b] = static_cast<float>(matches[b]) / (lsize/factor) * 100.f;
        printf("Accuracy batch %d: %.1f%, Matches: %d/10000\n", b, accuracy[b],matches[b]);
    }

    auto total_cpu_time = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());
    total_cpu_time -= total_kernel_time;
    auto cpu_time = static_cast<float>(total_cpu_time) / (lsize/factor) / 1;
    auto kernel_time = static_cast<float>(total_kernel_time) / (lsize/factor) / 1;

    return std::make_tuple(accuracy, total_cpu_time, cpu_time, total_kernel_time, kernel_time);
}

int main() {

    auto start = std::chrono::high_resolution_clock::now();
    /* load dataset in a vector ->
     * stuff like [1] or for-loop with only 1 run-through are remains from code-porting from previous batch-type implementation 
     */
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
    // for(int b = 0; b < 1; b++){
    //     printf("Accuracy batch %d: %.1f%\n", b, std::get<0>(results)[b]);
    // }

    printf("\n");
    printf("Total CPU time: %.2f [s] => Latency: %.4f [ms/elem]\n", std::get<1>(results)/1000.0f, std::get<2>(results));
    printf("Total GPU time: %.2f [s] => Latency: %.4f [ms/elem]\n", std::get<3>(results)/1000.0f, std::get<4>(results));

    return 0;
}
