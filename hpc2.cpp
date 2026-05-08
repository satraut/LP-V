#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>

using namespace std;

// ====================== UTILITY ======================
void printArray(const vector<int>& arr) {
    for (int x : arr) cout << x << " ";
    cout << endl;
}

// ====================== SEQUENTIAL BUBBLE ======================
void bubbleSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1])
                swap(arr[j], arr[j + 1]);
        }
    }
}

// ====================== PARALLEL BUBBLE (Odd-Even Sort) ======================
void parallelBubbleSort(vector<int>& arr) {
    int n = arr.size();

    for (int i = 0; i < n; i++) {

        // Even phase
        #pragma omp parallel for
        for (int j = 0; j < n - 1; j += 2) {
            if (arr[j] > arr[j + 1])
                swap(arr[j], arr[j + 1]);
        }

        // Odd phase
        #pragma omp parallel for
        for (int j = 1; j < n - 1; j += 2) {
            if (arr[j] > arr[j + 1])
                swap(arr[j], arr[j + 1]);
        }
    }
}

// ====================== MERGE FUNCTION ======================
void merge(vector<int>& arr, int l, int m, int r) {
    vector<int> temp(r - l + 1);

    int i = l, j = m + 1, k = 0;

    while (i <= m && j <= r) {
        if (arr[i] <= arr[j])
            temp[k++] = arr[i++];
        else
            temp[k++] = arr[j++];
    }

    while (i <= m) temp[k++] = arr[i++];
    while (j <= r) temp[k++] = arr[j++];

    for (int i = l, k = 0; i <= r; i++, k++)
        arr[i] = temp[k];
}

// ====================== SEQUENTIAL MERGE SORT ======================
void mergeSort(vector<int>& arr, int l, int r) {
    if (l >= r) return;

    int m = (l + r) / 2;
    mergeSort(arr, l, m);
    mergeSort(arr, m + 1, r);
    merge(arr, l, m, r);
}

// ====================== PARALLEL MERGE SORT ======================
void parallelMergeSort(vector<int>& arr, int l, int r) {
    if (l >= r) return;

    int m = (l + r) / 2;

    #pragma omp parallel sections
    {
        #pragma omp section
        parallelMergeSort(arr, l, m);

        #pragma omp section
        parallelMergeSort(arr, m + 1, r);
    }

    merge(arr, l, m, r);
}

// ====================== MAIN ======================
int main() {
    int n = 10000;  // size of array
    vector<int> arr(n);

    // Generate random data
    for (int i = 0; i < n; i++)
        arr[i] = rand() % 10000;

    vector<int> arr1 = arr;
    vector<int> arr2 = arr;
    vector<int> arr3 = arr;
    vector<int> arr4 = arr;

    double start, end;

    // Sequential Bubble Sort
    start = omp_get_wtime();
    bubbleSort(arr1);
    end = omp_get_wtime();
    cout << "Sequential Bubble Sort Time: " << end - start << endl;

    // Parallel Bubble Sort
    start = omp_get_wtime();
    parallelBubbleSort(arr2);
    end = omp_get_wtime();
    cout << "Parallel Bubble Sort Time: " << end - start << endl;

    // Sequential Merge Sort
    start = omp_get_wtime();
    mergeSort(arr3, 0, n - 1);
    end = omp_get_wtime();
    cout << "Sequential Merge Sort Time: " << end - start << endl;

    // Parallel Merge Sort
    start = omp_get_wtime();
    parallelMergeSort(arr4, 0, n - 1);
    end = omp_get_wtime();
    cout << "Parallel Merge Sort Time: " << end - start << endl;

    return 0;
}