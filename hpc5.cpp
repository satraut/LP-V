#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>

using namespace std;

// Generate synthetic data: y = 3x + noise
void generateData(vector<double>& X, vector<double>& Y, int n) {
    for (int i = 0; i < n; i++) {
        X[i] = i * 0.01;
        Y[i] = 3 * X[i] + (rand() % 10) * 0.01;
    }
}

// -------- Sequential Gradient Descent --------
void sequentialLR(vector<double>& X, vector<double>& Y, double& w, double& b, int n) {
    double lr = 0.01;

    for (int epoch = 0; epoch < 1000; epoch++) {
        double dw = 0.0, db = 0.0;

        for (int i = 0; i < n; i++) {
            double y_pred = w * X[i] + b;
            double error = y_pred - Y[i];

            dw += error * X[i];
            db += error;
        }

        w -= lr * dw / n;
        b -= lr * db / n;
    }
}

// -------- Parallel Gradient Descent --------
void parallelLR(vector<double>& X, vector<double>& Y, double& w, double& b, int n) {
    double lr = 0.01;

    for (int epoch = 0; epoch < 1000; epoch++) {
        double dw = 0.0, db = 0.0;

        #pragma omp parallel for reduction(+:dw, db)
        for (int i = 0; i < n; i++) {
            double y_pred = w * X[i] + b;
            double error = y_pred - Y[i];

            dw += error * X[i];
            db += error;
        }

        w -= lr * dw / n;
        b -= lr * db / n;
    }
}

// -------- Main --------
int main() {
    int n = 100000; // dataset size

    vector<double> X(n), Y(n);

    generateData(X, Y, n);

    double w1 = 0, b1 = 0;
    double w2 = 0, b2 = 0;

    double start, end;

    // Sequential
    start = omp_get_wtime();
    sequentialLR(X, Y, w1, b1, n);
    end = omp_get_wtime();

    cout << "Sequential Result:\n";
    cout << "w = " << w1 << ", b = " << b1 << endl;
    cout << "Time: " << end - start << " sec\n\n";

    // Parallel
    start = omp_get_wtime();
    parallelLR(X, Y, w2, b2, n);
    end = omp_get_wtime();

    cout << "Parallel Result:\n";
    cout << "w = " << w2 << ", b = " << b2 << endl;
    cout << "Time: " << end - start << " sec\n";

    return 0;
}