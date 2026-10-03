#pragma once
#include "spgemm.hpp"
#include <algorithm>

// Reusable local kernel. Inputs are positive and exact output values fit uint32_t.
// Storage: 8*N bytes plus touched-column indices and the returned sparse row.
// Call finish() after each output row (or partial row) to reset the accumulator.
class LocalAccumulator {
    std::vector<uint64_t> values;
    std::vector<uint32_t> touched;
public:
    explicit LocalAccumulator(uint32_t columns): values(columns,0) {}
    void add(uint32_t col,uint64_t value) {
        if(values[col]==0) touched.push_back(col);
        values[col]+=value;
    }
    void add_scaled_row(uint32_t scale,const Entry *entries,uint64_t length) {
        for(uint64_t j=0;j<length;++j) add(entries[j].col,uint64_t(scale)*entries[j].val);
    }
    std::vector<Entry> finish() {
        std::sort(touched.begin(),touched.end());
        std::vector<Entry> out; out.reserve(touched.size());
        for(uint32_t col:touched) { out.push_back({col,uint32_t(values[col])}); values[col]=0; }
        touched.clear(); return out;
    }
};
