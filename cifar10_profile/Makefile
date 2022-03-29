intel:
	nvcc -c cuda_kernel.cu
	g++ -c -w -std=c++14 main.cpp net.cpp -I net.hpp netW.hpp -O3
ifeq ($(shell hostname), LDB-XPS)
	g++ -o cifar.o cuda_kernel.o net.o main.o -L/usr/local/cuda/lib64 -lcudart # for LDB-XPS system at home
else
	g++ -o cifar.o cuda_kernel.o net.o main.o -L/usr/local/cuda-10.0/lib64 -lcudart # for system on Uni-server
endif

clean:
	rm -f $(binaries) *.o