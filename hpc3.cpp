#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>

using namespace std;

// ----------- Sequential Operations -----------
void sequentialOps(const vector<int>& arr, int &minVal, int &maxVal, long long &sum) {
    minVal = arr[0];
    maxVal = arr[0];
    sum = 0;

    for (int x : arr) {
        if (x < minVal) minVal = x;
        if (x > maxVal) maxVal = x;
        sum += x;
    }
}

// ----------- Parallel Reduction -----------
void parallelOps(const vector<int>& arr, int &minVal, int &maxVal, long long &sum) {
    int n = arr.size();

    minVal = arr[0];
    maxVal = arr[0];
    sum = 0;

    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) reduction(+:sum)
    for (int i = 0; i < n; i++) {
        if (arr[i] < minVal) minVal = arr[i];
        if (arr[i] > maxVal) maxVal = arr[i];
        sum += arr[i];
    }
}

// ----------- Main Function -----------
int main() {
    int n = 1000000; // size of array
    vector<int> arr(n);

    // Generate random data
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 10000;
    }

    int minVal, maxVal;
    long long sum;
    double avg;

    double start, end;

    // Sequential
    start = omp_get_wtime();
    sequentialOps(arr, minVal, maxVal, sum);
    avg = (double)sum / n;
    end = omp_get_wtime();

    cout << "----- Sequential Results -----" << endl;
    cout << "Min: " << minVal << endl;
    cout << "Max: " << maxVal << endl;
    cout << "Sum: " << sum << endl;
    cout << "Average: " << avg << endl;
    cout << "Time: " << end - start << " sec\n" << endl;

    // Parallel
    start = omp_get_wtime();
    parallelOps(arr, minVal, maxVal, sum);
    avg = (double)sum / n;
    end = omp_get_wtime();

    cout << "----- Parallel Results -----" << endl;
    cout << "Min: " << minVal << endl;
    cout << "Max: " << maxVal << endl;
    cout << "Sum: " << sum << endl;
    cout << "Average: " << avg << endl;
    cout << "Time: " << end - start << " sec" << endl;

    return 0;
}