# PCA-GPU-based-vector-summation.-Explore-the-differences.
i) Using the program sumArraysOnGPU-timer.cu, set the block.x = 1023. Recompile and run it. Compare the result with the execution confi guration of block.x = 1024. Try to explain the difference and the reason.

ii) Refer to sumArraysOnGPU-timer.cu, and let block.x = 256. Make a new kernel to let each thread handle two elements. Compare the results with other execution confi gurations.
## Aim:
(i) To modify or set the execution configuration of block.x as 1023 & 1024 and compare the elapsed time obtained on Host and GPU.

(ii) To set the number of threads as 256 and obtain the elapsed time on Host and GPU.


## Procedure:
1.Initialize the device and set the device properties.

2.Allocate memory on the host for input and output arrays.

3.Initialize input arrays with random values on the host.

4.Allocate memory on the device for input and output arrays, and copy input data from host to device.

5.Launch a CUDA kernel to perform vector addition on the device.

6.Copy output data from the device to the host and verify the results against the host's sequential vector addition. Free memory on the host and the device.

## Program:
Developed by: Sabari Akash A
Register number: 212222230124
# 1.Block.x=1023
```
#include "../common/common.h" 
#include <cuda_runtime.h> 
#include <stdio.h>
void checkResult(float *hostRef, float *gpuRef, const int N) 
{ 
double epsilon = 1.0E-8; bool match = 1;
for (int i = 0; i < N; i++)
{
    if (abs(hostRef[i] - gpuRef[i]) > epsilon)
    {
        match = 0;
        printf("Arrays do not match!\n");
        printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i],
               gpuRef[i], i);
        break;
    }
}

if (match) printf("Arrays match.\n\n");

return;
void initialData(float *ip, int size) { // generate different seed for random number time_t t; srand((unsigned) time(&t));

for (int i = 0; i < size; i++)
{
    ip[i] = (float)( rand() & 0xFF ) / 10.0f;
}

return;
}
void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
for (int idx = 0; idx < N; idx++)
{ 
C[idx] = A[idx] + B[idx]; 
} } 
global void sumArraysOnGPU(float *A, float *B, float *C, const int N)
{ 
int i = blockIdx.x * blockDim.x + threadIdx.x;

if (i < N) C[i] = A[i] + B[i];
}
int main(int argc, char **argv)
{ 
printf("%s Starting...\n", argv[0]);

// set up device
int dev = 0;
cudaDeviceProp deviceProp;
CHECK(cudaGetDeviceProperties(&deviceProp, dev));
printf("Using Device %d: %s\n", dev, deviceProp.name);
CHECK(cudaSetDevice(dev));

// set up data size of vectors
int nElem = 1 << 27;
printf("Vector size %d\n", nElem);

// malloc host memory
size_t nBytes = nElem * sizeof(float);

float *h_A, *h_B, *hostRef, *gpuRef;
h_A     = (float *)malloc(nBytes);
h_B     = (float *)malloc(nBytes);
hostRef = (float *)malloc(nBytes);
gpuRef  = (float *)malloc(nBytes);

double iStart, iElaps;

// initialize data at host side
iStart = seconds();
initialData(h_A, nElem);
initialData(h_B, nElem);
iElaps = seconds() - iStart;
printf("initialData Time elapsed %f sec\n", iElaps);
memset(hostRef, 0, nBytes);
memset(gpuRef,  0, nBytes);

// add vector at host side for result checks
iStart = seconds();
sumArraysOnHost(h_A, h_B, hostRef, nElem);
iElaps = seconds() - iStart;
printf("sumArraysOnHost Time elapsed %f sec\n", iElaps);

// malloc device global memory
float *d_A, *d_B, *d_C;
CHECK(cudaMalloc((float**)&d_A, nBytes));
CHECK(cudaMalloc((float**)&d_B, nBytes));
CHECK(cudaMalloc((float**)&d_C, nBytes));

// transfer data from host to device
CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
CHECK(cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice));

// invoke kernel at host side
int iLen = 1023;
dim3 block (iLen);
dim3 grid  ((nElem + block.x - 1) / block.x);

iStart = seconds();
sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
CHECK(cudaDeviceSynchronize());
iElaps = seconds() - iStart;
printf("sumArraysOnGPU <<<  %d, %d  >>>  Time elapsed %f sec\n", grid.x,
       block.x, iElaps);

// check kernel error
CHECK(cudaGetLastError()) ;

// copy kernel result back to host side
CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

// check device results
checkResult(hostRef, gpuRef, nElem);

// free device global memory
CHECK(cudaFree(d_A));
CHECK(cudaFree(d_B));
CHECK(cudaFree(d_C));

// free host memory
free(h_A);
free(h_B);
free(hostRef);
free(gpuRef);

return(0);
```
# 2.Block.x=1024
```
#include "../common/common.h" 
#include <cuda_runtime.h> 
#include <stdio.h>
void checkResult(float *hostRef, float *gpuRef, const int N) 
{ 
double epsilon = 1.0E-8;
bool match = 1;

for (int i = 0; i < N; i++)
{
    if (abs(hostRef[i] - gpuRef[i]) > epsilon)
    {
        match = 0;
        printf("Arrays do not match!\n");
        printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i],
               gpuRef[i], i);
        break;
    }
}

if (match) printf("Arrays match.\n\n");

return;
}
void initialData(float *ip, int size) 
{ 
time_t t; 
srand((unsigned) time(&t));

for (int i = 0; i < size; i++)
{
    ip[i] = (float)( rand() & 0xFF ) / 10.0f;
}

return;
}
void sumArraysOnHost(float *A, float *B, float *C, const int N) 
{ 
for (int idx = 0; idx < N; idx++)
{ 
C[idx] = A[idx] + B[idx]; 
} }
global void sumArraysOnGPU(float *A, float *B, float *C, const int N) 
{ 
int i = blockIdx.x * blockDim.x + threadIdx.x;

if (i < N) C[i] = A[i] + B[i];
}
int main(int argc, char **argv)
{ 
printf("%s Starting...\n", argv[0]);

// set up device
int dev = 0;
cudaDeviceProp deviceProp;
CHECK(cudaGetDeviceProperties(&deviceProp, dev));
printf("Using Device %d: %s\n", dev, deviceProp.name);
CHECK(cudaSetDevice(dev));

// set up data size of vectors
int nElem = 1 << 27;
printf("Vector size %d\n", nElem);

// malloc host memory
size_t nBytes = nElem * sizeof(float);

float *h_A, *h_B, *hostRef, *gpuRef;
h_A     = (float *)malloc(nBytes);
h_B     = (float *)malloc(nBytes);
hostRef = (float *)malloc(nBytes);
gpuRef  = (float *)malloc(nBytes);

double iStart, iElaps;

// initialize data at host side
iStart = seconds();
initialData(h_A, nElem);
initialData(h_B, nElem);
iElaps = seconds() - iStart;
printf("initialData Time elapsed %f sec\n", iElaps);
memset(hostRef, 0, nBytes);
memset(gpuRef,  0, nBytes);

// add vector at host side for result checks
iStart = seconds();
sumArraysOnHost(h_A, h_B, hostRef, nElem);
iElaps = seconds() - iStart;
printf("sumArraysOnHost Time elapsed %f sec\n", iElaps);

// malloc device global memory
float *d_A, *d_B, *d_C;
CHECK(cudaMalloc((float**)&d_A, nBytes));
CHECK(cudaMalloc((float**)&d_B, nBytes));
CHECK(cudaMalloc((float**)&d_C, nBytes));

// transfer data from host to device
CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
CHECK(cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice));

// invoke kernel at host side
int iLen = 1024;
dim3 block (iLen);
dim3 grid  ((nElem + block.x - 1) / block.x);

iStart = seconds();
sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
CHECK(cudaDeviceSynchronize());
iElaps = seconds() - iStart;
printf("sumArraysOnGPU <<<  %d, %d  >>>  Time elapsed %f sec\n", grid.x,
       block.x, iElaps);

// check kernel error
CHECK(cudaGetLastError()) ;

// copy kernel result back to host side
CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

// check device results
checkResult(hostRef, gpuRef, nElem);

// free device global memory
CHECK(cudaFree(d_A));
CHECK(cudaFree(d_B));
CHECK(cudaFree(d_C));

// free host memory
free(h_A);
free(h_B);
free(hostRef);
free(gpuRef);

return(0);
}
```
# 3.Block.x=256
```
#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

void checkResult(float *hostRef, float *gpuRef, const int N) 
{ 
double epsilon = 1.0E-8;
bool match = 1;

for (int i = 0; i < N; i++)
{
    if (abs(hostRef[i] - gpuRef[i]) > epsilon)
    {
        match = 0;
        printf("Arrays do not match!\n");
        printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i],
               gpuRef[i], i);
        break;
    }
}

if (match) printf("Arrays match.\n\n");

return;
}

void initialData(float *ip, int size) { // generate different seed for random number time_t t; srand((unsigned) time(&t));

for (int i = 0; i < size; i++)
{
    ip[i] = (float)( rand() & 0xFF ) / 10.0f;
}

return;
}

void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
for (int idx = 0; idx < N; idx++)
{ 
C[idx] = A[idx] + B[idx]; 
} } 
global void sumArraysOnGPU(float *A, float *B, float *C, const int N) 
{ 
int i = blockIdx.x * blockDim.x + threadIdx.x;

if (i < N) C[i] = A[i] + B[i];
}

int main(int argc, char **argv) { printf("%s Starting...\n", argv[0]);

// set up device
int dev = 0;
cudaDeviceProp deviceProp;
CHECK(cudaGetDeviceProperties(&deviceProp, dev));
printf("Using Device %d: %s\n", dev, deviceProp.name);
CHECK(cudaSetDevice(dev));

// set up data size of vectors
int nElem = 1 << 24;
printf("Vector size %d\n", nElem);

// malloc host memory
size_t nBytes = nElem * sizeof(float);

float *h_A, *h_B, *hostRef, *gpuRef;
h_A     = (float *)malloc(nBytes);
h_B     = (float *)malloc(nBytes);
hostRef = (float *)malloc(nBytes);
gpuRef  = (float *)malloc(nBytes);

double iStart, iElaps;

// initialize data at host side
iStart = seconds();
initialData(h_A, nElem);
initialData(h_B, nElem);
iElaps = seconds() - iStart;
printf("initialData Time elapsed %f sec\n", iElaps);
memset(hostRef, 0, nBytes);
memset(gpuRef,  0, nBytes);

// add vector at host side for result checks
iStart = seconds();
sumArraysOnHost(h_A, h_B, hostRef, nElem);
iElaps = seconds() - iStart;
printf("sumArraysOnHost Time elapsed %f sec\n", iElaps);

// malloc device global memory
float *d_A, *d_B, *d_C;
CHECK(cudaMalloc((float**)&d_A, nBytes));
CHECK(cudaMalloc((float**)&d_B, nBytes));
CHECK(cudaMalloc((float**)&d_C, nBytes));

// transfer data from host to device
CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
CHECK(cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice));

// invoke kernel at host side
int iLen = 256;
dim3 block (iLen);
dim3 grid  ((nElem + block.x - 1) / block.x);

iStart = seconds();
sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
CHECK(cudaDeviceSynchronize());
iElaps = seconds() - iStart;
printf("sumArraysOnGPU <<<  %d, %d  >>>  Time elapsed %f sec\n", grid.x,
       block.x, iElaps);

// check kernel error
CHECK(cudaGetLastError()) ;

// copy kernel result back to host side
CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

// check device results
checkResult(hostRef, gpuRef, nElem);

// free device global memory
CHECK(cudaFree(d_A));
CHECK(cudaFree(d_B));
CHECK(cudaFree(d_C));

// free host memory
free(h_A);
free(h_B);
free(hostRef);
free(gpuRef);

return(0);
```

## Output:
1.Block.x=1023
![output1](https://github.com/Sabariakash22009103/PCA-GPU-based-vector-summation.-Explore-the-differences./assets/119390227/0ee669f3-acf6-41aa-b014-11e2c15d1b59)
2.Block.x=1024
![output2](https://github.com/Sabariakash22009103/PCA-GPU-based-vector-summation.-Explore-the-differences./assets/119390227/20e248d1-98c9-4f7d-bff1-11292ff50624)
3.Block.x=256
![output3](https://github.com/Sabariakash22009103/PCA-GPU-based-vector-summation.-Explore-the-differences./assets/119390227/5d208fb7-a066-4759-9376-94dba4692b42)

## Result:
i) Thus the program to compare the execution time and result of summing two arrays using CUDA programming by setting block.x = 1023 and block.x = 1024 has been successfully implemented and verified.

ii) Thus the program to To analyze the impact of changing execution configuration on the execution time and result of summing two arrays using CUDA programming by letting each thread handle two elements and setting block.x = 256 has been successfully implemented and verified.

