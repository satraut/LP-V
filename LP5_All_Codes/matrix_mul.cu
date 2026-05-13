#include <iostream>
#include <cuda_runtime.h>

using namespace std;

// CUDA Kernel Function
__global__ void matmul(int *A, int *B, int *C, int N)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if (Row < N && Col < N)
    {
        int Pvalue = 0;

        for (int k = 0; k < N; k++)
        {
            Pvalue += A[Row * N + k] * B[k * N + Col];
        }

        C[Row * N + Col] = Pvalue;
    }
}

int main()
{
    int N = 3;

    int size = N * N * sizeof(int);

    int *A, *B, *C;
    int *dev_A, *dev_B, *dev_C;

    // Allocate Host Memory
    cudaMallocHost(&A, size);
    cudaMallocHost(&B, size);
    cudaMallocHost(&C, size);

    // Initialize Matrix A
    A[0] = 1; A[1] = 2; A[2] = 3;
    A[3] = 4; A[4] = 5; A[5] = 6;
    A[6] = 7; A[7] = 8; A[8] = 9;

    // Initialize Matrix B
    B[0] = 1; B[1] = 0; B[2] = 1;
    B[3] = 0; B[4] = 1; B[5] = 0;
    B[6] = 1; B[7] = 0; B[8] = 1;

    // Allocate Device Memory
    cudaMalloc(&dev_A, size);
    cudaMalloc(&dev_B, size);
    cudaMalloc(&dev_C, size);

    // Copy Data Host -> Device
    cudaMemcpy(dev_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, size, cudaMemcpyHostToDevice);

    // Define Block and Grid Size
    dim3 dimBlock(16, 16);
    dim3 dimGrid(1, 1);

    // Launch Kernel
    matmul<<<dimGrid, dimBlock>>>(dev_A, dev_B, dev_C, N);

    // Copy Result Device -> Host
    cudaMemcpy(C, dev_C, size, cudaMemcpyDeviceToHost);

    // Print Matrix A
    cout << "Matrix A:\n";

    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            cout << A[i * N + j] << " ";
        }
        cout << endl;
    }

    // Print Matrix B
    cout << "\nMatrix B:\n";

    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            cout << B[i * N + j] << " ";
        }
        cout << endl;
    }

    // Print Result Matrix
    cout << "\nResult Matrix:\n";

    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            cout << C[i * N + j] << " ";
        }
        cout << endl;
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


nvcc matrix_mul.cu -o matrix_mul
matrix_mul

Collab
!nvcc matrix_mul.cu -o matrix_mul
!./matrix_mul
