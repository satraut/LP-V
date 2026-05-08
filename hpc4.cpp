// OpenMP version of Vector Addition and Matrix Multiplication

#include <iostream>
#include <omp.h>

#define N 1024
#define WIDTH 16

using namespace std;

// ---------------- VECTOR ADDITION ----------------
void vectorAdd(int *A, int *B, int *C, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
}

// ---------------- MATRIX MULTIPLICATION ----------------
void matrixMul(int *A, int *B, int *C, int width) {
    #pragma omp parallel for collapse(2)
    for (int row = 0; row < width; row++) {
        for (int col = 0; col < width; col++) {
            int sum = 0;
            for (int k = 0; k < width; k++) {
                sum += A[row * width + k] * B[k * width + col];
            }
            C[row * width + col] = sum;
        }
    }
}

// ---------------- MAIN ----------------
int main() {

    // -------- VECTOR ADDITION --------
    int A[N], B[N], C[N];

    for (int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = i * 2;
    }

    vectorAdd(A, B, C, N);

    cout << "Vector Addition Result (first 10 elements):\n";
    for (int i = 0; i < 10; i++) {
        cout << C[i] << " ";
    }
    cout << endl;

    // -------- MATRIX MULTIPLICATION --------
    int MA[WIDTH * WIDTH], MB[WIDTH * WIDTH], MC[WIDTH * WIDTH];

    for (int i = 0; i < WIDTH * WIDTH; i++) {
        MA[i] = 1;
        MB[i] = 2;
    }

    matrixMul(MA, MB, MC, WIDTH);

    cout << "\nMatrix Multiplication Result (first row):\n";
    for (int i = 0; i < WIDTH; i++) {
        cout << MC[i] << " ";
    }
    cout << endl;

    return 0;
}