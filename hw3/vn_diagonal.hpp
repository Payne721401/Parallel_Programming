#pragma once
#include <cstdint>
#ifdef __CUDACC__
#define PP_VN_HD __host__ __device__ __forceinline__
#else
#define PP_VN_HD inline
#endif
namespace vn_diagonal {
// A accumulates row-prefix endpoints down-right; B accumulates down-left.
// Split the diamond at its central row. Four diagonal intervals, eight reads.
PP_VN_HD int count(const uint16_t *a, const uint16_t *b, int pitch,
                   int x, int y, int r) {
    const int mid = y + r + 1, bottom = y + 2*r + 1;
    const int upper_right = int(a[mid*pitch+x+2*r+1]) - int(a[y*pitch+x+r]);
    const int upper_left = int(b[mid*pitch+x]) - int(b[y*pitch+x+r+1]);
    const int lower_right = int(b[bottom*pitch+x+r+1]) - int(b[mid*pitch+x+2*r+1]);
    const int lower_left = int(a[bottom*pitch+x+r]) - int(a[mid*pitch+x]);
    return upper_right - upper_left + lower_right - lower_left;
}
}
#undef PP_VN_HD
