#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char** argv) {

	MPI_Init(&argc, &argv);

	int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		MPI_Finalize();
		return 1;
	}
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);

	// Rank i takes x = i, i + size, i + 2*size, ... Counts differ by at most
    // one iteration and there is no remainder to hand out. The loop body is
    // untouched: the u64 r*r - x*x and the long-double sqrtl have to stay
    // exactly as they are at the largest input.
	unsigned long long pixels = 0;
	for (unsigned long long x = rank; x < r; x += size) {
		unsigned long long y = ceil(sqrtl(r*r - x*x));
		pixels += y;
		pixels %= k;
	}

	// Modular addition is associative, so the sum of the partials mod k is
    // the sequential accumulator, exactly.
    unsigned long long total = 0;
    MPI_Reduce(&pixels, &total, 1, MPI_UNSIGNED_LONG_LONG,
               MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) printf("%llu\n", (4 * (total % k)) % k);

    MPI_Finalize();
}
