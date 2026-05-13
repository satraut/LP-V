%%writefile vector_add.cu

#include <iostream>
#include <cuda_runtime.h>

using namespace std;

// Kernel Function
__global__ void addVectors(int *A, int *B, int *C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n)
    {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    int n = 10;
    int size = n * sizeof(int);

    int *A, *B, *C;
    int *dev_A, *dev_B, *dev_C;

    // Allocate Host Memory
    cudaMallocHost(&A, size);
    cudaMallocHost(&B, size);
    cudaMallocHost(&C, size);

    // Initialize Vectors
    for(int i = 0; i < n; i++)
    {
        A[i] = i;
        B[i] = i * 2;
    }

    // Allocate Device Memory
    cudaMalloc(&dev_A, size);
    cudaMalloc(&dev_B, size);
    cudaMalloc(&dev_C, size);

    // Copy Data Host -> Device
    cudaMemcpy(dev_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, size, cudaMemcpyHostToDevice);

    // Launch Kernel
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    addVectors<<<numBlocks, blockSize>>>(dev_A, dev_B, dev_C, n);

    // Copy Result Device -> Host
    cudaMemcpy(C, dev_C, size, cudaMemcpyDeviceToHost);

    // Print Result
    cout << "Vector Addition Result:\n";

    for(int i = 0; i < n; i++)
    {
        cout << A[i] << " + " << B[i] << " = " << C[i] << endl;
    }

    // Free Memory
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);

    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);

    return 0;
}

collab
!nvcc vector_add.cu -o vector_add
!./vector_add


PC
nvcc vector_add.cu -o vector_add
vector_add
