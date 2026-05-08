/**
 * ============================================================
 *  Parallel Quicksort using MPI
 *  Mini Project: Performance Evaluation
 *  
 *  Author  : Mini Project Group
 *  Course  : Parallel & Distributed Computing
 *  Compiler: mpicc (MPICH / OpenMPI)
 *  Run     : mpirun -np <P> ./parallel_quicksort <N>
 * ============================================================
 *
 *  Algorithm Overview:
 *  -------------------
 *  1. Process 0 generates N random integers and broadcasts
 *     a pivot chosen as the median-of-three.
 *  2. Each process partitions its local data into two halves
 *     (≤ pivot  and  > pivot).
 *  3. Processes pair up: the "lower" process keeps the lower
 *     half and the "upper" process keeps the upper half.
 *     Data is exchanged via MPI_Sendrecv.
 *  4. Steps 2-3 repeat recursively (hypercube decomposition)
 *     until each process has its own independent sub-array.
 *  5. Every process sorts its sub-array with sequential
 *     Quicksort (stdlib qsort).
 *  6. Process 0 gathers all sub-arrays (MPI_Gatherv) and
 *     measures total wall-clock time.
 *
 *  Metrics Captured:
 *  -----------------
 *  - Wall-clock time (MPI_Wtime)
 *  - Speedup  : T_serial / T_parallel
 *  - Efficiency: Speedup / P
 * ============================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

/* ---- Utility: sequential qsort comparator ---- */
int cmp_int(const void *a, const void *b) {
    return (*(int*)a - *(int*)b);
}

/* ---- Generate N random integers [0, 1 000 000) ---- */
void generate_array(int *arr, int n, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < n; i++)
        arr[i] = rand() % 1000000;
}

/* ---- Verify sorted order ---- */
int is_sorted(int *arr, int n) {
    for (int i = 1; i < n; i++)
        if (arr[i] < arr[i-1]) return 0;
    return 1;
}

/* ---- Partition local array around pivot ---- */
int local_partition(int *arr, int n, int pivot) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        while (lo <= hi && arr[lo] <= pivot) lo++;
        while (lo <= hi && arr[hi]  > pivot) hi--;
        if (lo < hi) { int t = arr[lo]; arr[lo] = arr[hi]; arr[hi] = t; }
    }
    return lo;   /* first index > pivot */
}

/* ================================================
   PARALLEL QUICKSORT  (hypercube / recursive halving)
   ================================================ */
void parallel_quicksort(int **local, int *local_n,
                        int rank, int size, MPI_Comm comm)
{
    int active = size;          /* active processes this round */
    int mask   = active - 1;   /* bitmask for pairing          */

    while (active > 1) {
        int pivot;

        /* ---- Rank 0 in the current active group picks pivot ---- */
        int group_rank = rank & mask;
        int partner;

        /* Leader of current group broadcasts pivot */
        int leader = rank & ~mask;           /* base rank of group  */
        if (group_rank == 0) {
            /* median-of-three: first, middle, last */
            int a = (*local)[0];
            int b = (*local)[(*local_n)/2];
            int c = (*local)[(*local_n)-1];
            if (a > b) { int t=a; a=b; b=t; }
            if (b > c) { int t=b; b=c; c=t; }
            if (a > b) { int t=a; a=b; b=t; }
            pivot = b;
        }
        MPI_Bcast(&pivot, 1, MPI_INT, leader, comm);

        /* ---- Partition local data ---- */
        int split = local_partition(*local, *local_n, pivot);

        /* ---- Pair processes: upper half <-> lower half ---- */
        int half = active / 2;
        if (group_rank < half)
            partner = rank + half;
        else
            partner = rank - half;

        int send_count, recv_count;
        int *send_buf;
        int *recv_buf;

        if (group_rank < half) {
            /* keep lower portion, send upper portion */
            send_buf   = *local + split;
            send_count = *local_n - split;
        } else {
            /* keep upper portion, send lower portion */
            send_buf   = *local;
            send_count = split;
        }

        /* Exchange sizes */
        MPI_Sendrecv(&send_count, 1, MPI_INT, partner, 0,
                     &recv_count, 1, MPI_INT, partner, 0,
                     comm, MPI_STATUS_IGNORE);

        recv_buf = (int*)malloc(recv_count * sizeof(int));

        /* Exchange data */
        MPI_Sendrecv(send_buf,   send_count, MPI_INT, partner, 1,
                     recv_buf,   recv_count, MPI_INT, partner, 1,
                     comm, MPI_STATUS_IGNORE);

        /* Merge received data with kept data */
        if (group_rank < half) {
            /* kept [0..split-1], received more lower data */
            int new_n = split + recv_count;
            int *merged = (int*)malloc(new_n * sizeof(int));
            memcpy(merged,       *local,   split      * sizeof(int));
            memcpy(merged+split, recv_buf, recv_count * sizeof(int));
            free(*local);
            free(recv_buf);
            *local   = merged;
            *local_n = new_n;
        } else {
            /* kept [split..*local_n-1], received more upper data */
            int kept_n = *local_n - split;
            int new_n  = kept_n + recv_count;
            int *merged = (int*)malloc(new_n * sizeof(int));
            memcpy(merged,         *local + split, kept_n     * sizeof(int));
            memcpy(merged + kept_n, recv_buf,      recv_count * sizeof(int));
            free(*local);
            free(recv_buf);
            *local   = merged;
            *local_n = new_n;
        }

        /* Shrink active group */
        active /= 2;
        mask   /= 2;
    }

    /* Each process sorts its final sub-array sequentially */
    qsort(*local, *local_n, sizeof(int), cmp_int);
}

/* ================================================
   MAIN
   ================================================ */
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Default array size */
    int N = (argc > 1) ? atoi(argv[1]) : 1000000;

    int *global_arr  = NULL;
    int *sorted_arr  = NULL;

    /* ---- Process 0: generate data ---- */
    if (rank == 0) {
        global_arr = (int*)malloc(N * sizeof(int));
        generate_array(global_arr, N, 42);
        if (N <= 20) {
            printf("Input array: ");
            for (int i = 0; i < N; i++) printf("%d ", global_arr[i]);
            printf("\n");
        }
    }

    /* ---- Scatter data to all processes ---- */
    int base   = N / size;
    int rem    = N % size;
    int *scounts = (int*)malloc(size * sizeof(int));
    int *displs  = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        scounts[i] = base + (i < rem ? 1 : 0);
        displs[i]  = (i == 0) ? 0 : displs[i-1] + scounts[i-1];
    }

    int local_n = scounts[rank];
    int *local  = (int*)malloc(local_n * sizeof(int));

    MPI_Scatterv(global_arr, scounts, displs, MPI_INT,
                 local,      local_n,         MPI_INT,
                 0, MPI_COMM_WORLD);

    /* ---- Time the parallel sort ---- */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    parallel_quicksort(&local, &local_n, rank, size, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();
    double t_parallel = t_end - t_start;

    /* ---- Gather sorted sub-arrays to rank 0 ---- */
    int *rcounts = NULL, *rdispls = NULL;
    if (rank == 0) {
        rcounts = (int*)malloc(size * sizeof(int));
        rdispls = (int*)malloc(size * sizeof(int));
        sorted_arr = (int*)malloc(N * sizeof(int));
    }
    MPI_Gather(&local_n, 1, MPI_INT, rcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        rdispls[0] = 0;
        for (int i = 1; i < size; i++)
            rdispls[i] = rdispls[i-1] + rcounts[i-1];
    }
    MPI_Gatherv(local, local_n, MPI_INT,
                sorted_arr, rcounts, rdispls, MPI_INT,
                0, MPI_COMM_WORLD);

    /* ---- Process 0: report results ---- */
    if (rank == 0) {
        /* Serial baseline */
        int *serial_arr = (int*)malloc(N * sizeof(int));
        memcpy(serial_arr, global_arr, N * sizeof(int));
        double ts = MPI_Wtime();
        qsort(serial_arr, N, sizeof(int), cmp_int);
        double t_serial = MPI_Wtime() - ts;

        double speedup    = t_serial / t_parallel;
        double efficiency = speedup  / size;

        printf("\n========================================\n");
        printf("  Parallel Quicksort via MPI — Results  \n");
        printf("========================================\n");
        printf("  Array size     : %d elements\n", N);
        printf("  Processes      : %d\n", size);
        printf("  Serial time    : %.6f s\n", t_serial);
        printf("  Parallel time  : %.6f s\n", t_parallel);
        printf("  Speedup        : %.4f\n", speedup);
        printf("  Efficiency     : %.4f (%.2f%%)\n", efficiency, efficiency*100);
        printf("  Sorted correctly: %s\n", is_sorted(sorted_arr, N) ? "YES" : "NO");
        printf("========================================\n\n");

        if (N <= 20) {
            printf("Sorted array: ");
            for (int i = 0; i < N; i++) printf("%d ", sorted_arr[i]);
            printf("\n");
        }

        free(serial_arr);
        free(sorted_arr);
        free(rcounts);
        free(rdispls);
        free(global_arr);
    }

    free(local);
    free(scounts);
    free(displs);
    MPI_Finalize();
    return 0;
}
