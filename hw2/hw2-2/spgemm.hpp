#ifndef SPGEMM_HPP
#define SPGEMM_HPP
#include <mpi.h>
#include <cstdint>
#include <vector>

// All inputs have strictly increasing column ids per row and values in 1..8.
// Dimensions are in 1..2^24, so every output value is <= 64*2^24 and fits uint32.
struct Entry { uint32_t col, val; };
static_assert(sizeof(Entry) == 8);
struct Matrix {
    uint32_t rows = 0, cols = 0, first = 0;
    std::vector<uint64_t> ptr{0};  // local row offsets, relative to data
    std::vector<Entry> data;
    uint32_t local_rows() const { return static_cast<uint32_t>(ptr.size() - 1); }
};
inline uint32_t block_start(uint32_t n, int r, int p) { return uint64_t(n) * r / p; }
inline int row_owner(uint32_t row, uint32_t n, int p) {
    return static_cast<int>((uint64_t(row + 1) * p - 1) / n);
}
inline MPI_Datatype entry_type() {
    MPI_Datatype type;
    MPI_Type_contiguous(2, MPI_UINT32_T, &type);
    MPI_Type_commit(&type);
    return type;
}
// Called once on every rank. A/B initially own contiguous blocks of their rows.
// Return C with the same row ownership as A, sorted and duplicate-free, no explicit zeroes.
// Redistribution and all other preprocessing must happen inside this function.
void multiply(const Matrix &a, const Matrix &b, Matrix &c);
#endif
