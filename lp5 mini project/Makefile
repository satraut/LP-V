












# ===========================================================
#  Makefile — Parallel Quicksort (MPI)
#  Usage:
#    make           -> builds parallel_quicksort
#    make run_all   -> runs benchmarks for P=1,2,4,8
#    make clean     -> removes binaries
# ============================================================

CC       = mpicc
CFLAGS   = -O2 -Wall -lm
TARGET   = parallel_quicksort
SRC      = parallel_quicksort.c

# Default array sizes for benchmarking
SIZES    = 100000 500000 1000000 5000000

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^
	@echo "Build successful: $(TARGET)"

run_small:
	mpirun -np 1 ./$(TARGET) 20
	mpirun -np 2 ./$(TARGET) 20
	mpirun -np 4 ./$(TARGET) 20

run_bench:
	@echo "====== Benchmarking N=1000000 ======"
	@for p in 1 2 4 8; do \
		echo "--- P=$$p ---"; \
		mpirun -np $$p ./$(TARGET) 1000000; \
	done

run_all:
	@for n in $(SIZES); do \
		echo ""; \
		echo "====== N = $$n ======"; \
		for p in 1 2 4 8; do \
			echo "  P=$$p:"; \
			mpirun -np $$p ./$(TARGET) $$n; \
		done; \
	done

clean:
	rm -f $(TARGET) *.o

.PHONY: all run_small run_bench run_all clean
