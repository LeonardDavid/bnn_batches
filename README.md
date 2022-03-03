
# BNNs on GPU

This repository contains 2 different batch-notion implementations of BNNs on GPU on 2 different networks (FashionMNIST and CIFAR10) containing 10.000 images/dataset each.
1. First (original) idea of batches:
    - this implementation can be found in the folders ```bnn_cpu_par_xyz``` and ```cifar10```
    - simulate a large amount of data by copying the same dataset of images multiple times, but each 'batch' of 10.000 images (1 dataset) is in other memory location
    - therefore the algorithm is applied on ```BATCH_SIZE*10000``` images
2. Second (widely used) notion of batches ( **_flip** ):
		- this implementation can be found in the folders ```bnn_cpu_par_xyz_flip``` and ```cifar10_flip```
    - execute multiple consecutive images from the (constant) dataset at once 
    - therefore the algorithm is applied on ```10000``` images, regardless of ```BATCH_SIZE```
    - there is however a limit to the BATCH_SIZE in this notion, for both network types: 2656 for FashionMNIST and 199 for CIFAR10. It leads to errors at compilation which I have not gotten to look into yet (possibly because of the size of the img map)

The code is made to work on either my home system or on the Uni system (depending on which path there is to CUDA runtime library) so it is recommended to be compiled and executed on the Uni-server

The BATCH_SIZE can be changed in ```utils.h``` on the first line. 

For every folder there is also an Excel sheet with benchmarks. At this time, I did not get to execute all of the experiments, will update them as soon as I have the results.

# FashionMNIST 

  - compilation and execution: (first navigate to desired folder: for 1.  ```$ cd bnn_cpu_par_xyz``` or for 2.  ```$ cd bnn_cpu_par_xyz_flip```)
    ```
    $ make
    $ ./parxyz.o
    ```
  - Accuracy should __always__ be 85.9% (8590/10000 Matches)

# CIFAR10 

- compilation and execution: (first navigate to desired folder: for 1.  ```$ cd cifar10``` or for 2.  ```$ cd cifar10_flip```)
    ```
    $ make
    $ ./cifar.o
    ```
- Accuracy should __always__ be 75.1% (7511/10000 Matches)

# Questions

I hope I covered most of what is needed in the README and in comments throughout the code. If I missed something I will add it down the line or you can contact me via Element chat or my email: <leonard.bereholschi@tu-dortmund.de>
