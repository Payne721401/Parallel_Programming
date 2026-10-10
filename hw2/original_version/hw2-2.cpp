// MPI starter: replicate B once, then compute the original local A rows.
// Uses the supplied local kernel. Redistribution and selective communication
// are left for students. Full replication can exceed the memory budget.
#include "spgemm.hpp"
#include "local_kernel.hpp"
#include <climits>
#include <stdexcept>

namespace {
int count(uint64_t n) {
    if(n>INT_MAX) throw std::runtime_error("starter MPI count overflow");
    return int(n);
}
}
void multiply(const Matrix &a,const Matrix &b,Matrix &c) {
    int p; MPI_Comm_size(MPI_COMM_WORLD,&p);
    std::vector<int> counts(p),offsets(p);
    for(int r=0;r<p;++r) {
        offsets[r]=block_start(b.rows,r,p);
        counts[r]=block_start(b.rows,r+1,p)-offsets[r];
    }
    std::vector<uint64_t> local_lengths(b.local_rows()),lengths(b.rows);
    for(uint32_t i=0;i<b.local_rows();++i) local_lengths[i]=b.ptr[i+1]-b.ptr[i];
    MPI_Allgatherv(local_lengths.data(),count(local_lengths.size()),MPI_UINT64_T,
                  lengths.data(),counts.data(),offsets.data(),MPI_UINT64_T,MPI_COMM_WORLD);
    std::vector<uint64_t> ptr(uint64_t(b.rows)+1,0);
    for(uint32_t i=0;i<b.rows;++i) ptr[i+1]=ptr[i]+lengths[i];
    for(int r=0;r<p;++r) {
        offsets[r]=count(ptr[block_start(b.rows,r,p)]);
        counts[r]=count(ptr[block_start(b.rows,r+1,p)]-offsets[r]);
    }
    std::vector<Entry> entries(ptr.back());
    MPI_Datatype et=entry_type();
    MPI_Allgatherv(b.data.data(),count(b.data.size()),et,entries.data(),counts.data(),offsets.data(),et,MPI_COMM_WORLD);
    MPI_Type_free(&et);
    c.rows=a.rows; c.cols=b.cols; c.first=a.first;
    c.ptr.assign(uint64_t(a.local_rows())+1,0); c.data.clear();
    LocalAccumulator accum(b.cols);
    for(uint32_t i=0;i<a.local_rows();++i) {
        for(uint64_t q=a.ptr[i];q<a.ptr[i+1];++q) {
            Entry av=a.data[q]; uint64_t size=ptr[av.col+1]-ptr[av.col];
            if(size) accum.add_scaled_row(av.val,entries.data()+ptr[av.col],size);
        }
        auto row=accum.finish();
        c.data.insert(c.data.end(),row.begin(),row.end()); c.ptr[i+1]=c.data.size();
    }
}
