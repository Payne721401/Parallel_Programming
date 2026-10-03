#include "spgemm.hpp"
#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>

static void require(bool ok, const char *why) {
    if (!ok) throw std::runtime_error(why);
}
static void read_bytes(std::ifstream &f, void *p, uint64_t n) {
    if (n) f.read(static_cast<char *>(p), static_cast<std::streamsize>(n));
    require(bool(f), "short matrix file");
}
static Matrix read_shard(const char *path, int rank, int ranks) {
    std::ifstream f(path, std::ios::binary);
    require(bool(f), "cannot open matrix input");
    char magic[8]; uint32_t shape[2]; uint64_t nnz;
    read_bytes(f, magic, 8); read_bytes(f, shape, 8); read_bytes(f, &nnz, 8);
    require(std::memcmp(magic, "SPGCSR01", 8) == 0, "bad matrix magic");
    require(shape[0] && shape[0] <= (1u<<24) && shape[1] && shape[1] <= (1u<<24), "bad dimensions");
    require(nnz <= (1ull<<31), "input nnz exceeds prototype bound");
    f.seekg(0, std::ios::end);
    require(uint64_t(f.tellg()) == 24 + 8ull*(shape[0]+1) + 8*nnz, "bad matrix file size");
    Matrix x; x.rows = shape[0]; x.cols = shape[1];
    x.first = block_start(x.rows, rank, ranks);
    uint32_t count = block_start(x.rows, rank+1, ranks) - x.first;
    x.ptr.resize(uint64_t(count)+1);
    f.seekg(24 + 8ull*x.first);
    read_bytes(f, x.ptr.data(), x.ptr.size()*8);
    uint64_t first_entry = x.ptr[0];
    require(first_entry <= nnz && x.ptr.back() <= nnz && std::is_sorted(x.ptr.begin(), x.ptr.end()), "invalid row offsets");
    if (rank == 0) require(first_entry == 0, "first offset is not zero");
    if (rank == ranks-1) require(x.ptr.back() == nnz, "last offset differs from nnz");
    for (auto &v : x.ptr) v -= first_entry;
    x.data.resize(x.ptr.back());
    f.seekg(24 + 8ull*(x.rows+1) + 8*first_entry);
    read_bytes(f, x.data.data(), x.data.size()*sizeof(Entry));
    for (uint32_t r=0; r<count; ++r) {
        for (uint64_t e=x.ptr[r]; e<x.ptr[r+1]; ++e) {
            require(x.data[e].col < x.cols && x.data[e].val >= 1 && x.data[e].val <= 8, "bad input entry");
            if (e != x.ptr[r]) require(x.data[e-1].col < x.data[e].col, "unsorted/duplicate column");
        }
    }
    return x;
}
static void write_output(const char *path, const Matrix &c, int rank, int ranks) {
    uint64_t local = c.data.size(), base=0, total=0;
    MPI_Exscan(&local, &base, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    if (rank==0) base=0;
    MPI_Allreduce(&local, &total, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    require(c.data.size()*8 <= INT_MAX, "output shard too large for MPI count");
    std::vector<uint64_t> ptr=c.ptr;
    for (auto &v:ptr) v+=base;
    MPI_File f;
    require(MPI_File_open(MPI_COMM_WORLD, path, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &f)==MPI_SUCCESS, "cannot open output");
    MPI_File_set_size(f, 24+8ull*(c.rows+1)+8*total);
    if(rank==0) {
        char header[24]; std::memcpy(header,"SPGCSR01",8);
        std::memcpy(header+8,&c.rows,4); std::memcpy(header+12,&c.cols,4); std::memcpy(header+16,&total,8);
        MPI_File_write_at(f,0,header,24,MPI_BYTE,MPI_STATUS_IGNORE);
    }
    // Independent writes: each rank's slice goes out as one write of exactly its own bytes.
    // A collective write lets an aggregator rewrite a span holding other ranks' bytes,
    // and on BeeGFS it can write back zeros where another node's write has not landed yet.
    int n = static_cast<int>(c.local_rows()) + (rank==ranks-1);
    MPI_File_write_at(f,24+8ull*c.first,ptr.data(),n,MPI_UINT64_T,MPI_STATUS_IGNORE);
    MPI_File_write_at(f,24+8ull*(c.rows+1)+8*base,c.data.data(),static_cast<int>(c.data.size()*8),MPI_BYTE,MPI_STATUS_IGNORE);
    MPI_File_close(&f);
}
int main(int argc,char **argv) {
    MPI_Init(&argc,&argv); int rank,ranks;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    try {
        require(argc==4,"usage: hw2-2 A.csr B.csr C.csr");
        Matrix a=read_shard(argv[1],rank,ranks), b=read_shard(argv[2],rank,ranks), c;
        require(a.cols==b.rows,"incompatible shapes");
        MPI_Barrier(MPI_COMM_WORLD);
        double begin=MPI_Wtime();
        multiply(a,b,c);
        MPI_Barrier(MPI_COMM_WORLD);
        double elapsed=MPI_Wtime()-begin, max_elapsed;
        MPI_Reduce(&elapsed,&max_elapsed,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
        require(c.rows==a.rows && c.cols==b.cols && c.first==a.first && c.local_rows()==a.local_rows(),"wrong result shape/ownership");
        require(c.ptr.front()==0 && c.ptr.back()==c.data.size() && std::is_sorted(c.ptr.begin(),c.ptr.end()),"bad result offsets");
        for(uint32_t r=0;r<c.local_rows();++r) for(uint64_t e=c.ptr[r];e<c.ptr[r+1];++e) {
            require(c.data[e].col<c.cols && c.data[e].val && c.data[e].val <= (1u<<30),"bad result entry");
            if(e>c.ptr[r]) require(c.data[e-1].col<c.data[e].col,"result is not canonical CSR");
        }
        struct rusage usage{}; getrusage(RUSAGE_SELF,&usage);
        uint64_t rss=usage.ru_maxrss, max_rss;
        MPI_Reduce(&rss,&max_rss,1,MPI_UINT64_T,MPI_MAX,0,MPI_COMM_WORLD);
        write_output(argv[3],c,rank,ranks);
        if(rank==0) std::printf("Compute time: %.9f s\nPeak RSS: %llu KiB\n",max_elapsed,(unsigned long long)max_rss);
    } catch(const std::exception &e) {
        std::fprintf(stderr,"rank %d: %s\n",rank,e.what());
        MPI_Abort(MPI_COMM_WORLD,1);
    }
    MPI_Finalize(); return 0;
}
