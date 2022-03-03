#define BATCH_SIZE 1
#define NR_NEURONS 64   // for the fashion net, this value is constant, but for others (i.e. cifar10) it depends on the layer
#define IMG_HEIGHT 28   // original input image
#define IMG_WIDTH 28    // original input image
#define OUT_SIZE 10     // number of classes

#pragma once 
inline int index3D(const int x, const int y, const int z, const int sizey, const int sizez) {
    return x*sizey*sizez + y*sizez + z;
}

#pragma once 
inline int index4D(const int x, const int y, const int z, const int t, const int sizey, const int sizez, const int sizet){
    return x*sizey*sizez*sizet + y*sizez*sizet + z*sizet + t;
}