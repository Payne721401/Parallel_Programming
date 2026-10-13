// Memory and MPI costs on the course's nodes, measured rather than assumed. Run it through srun
// inside an allocation, in one of two modes:
//
//   mem  (srun -N1 -n8 ...)  first touch and release of 256 MiB per rank, for 1, 2, 4, 8 ranks
//        touching at once, in three kinds of memory: private (new[]), one shared window that rank
//        0 maps whole (how hw2-1 allocates), and one window per rank; then copy bandwidth with
//        one rank alone and with every rank at once.
//   net  (srun -N2 -n8 --ntasks-per-node=4 ...)  ping-pong latency and bandwidth between two
//        ranks on one node and on two nodes, every node-0 rank streaming to a node-1 partner at
//        once (one way, then both ways), and MPI_Alltoall throughput.
//
// Times are the slowest rank's; rank 0 prints.
#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

using u64 = uint64_t;
constexpr u64 MB = u64(1) << 20;
constexpr u64 PAGE = 4096;

int rank_, size_;

double slowest(double t)
{
    double m = 0;
    MPI_Allreduce(&t, &m, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return m;
}

// ---- mem ----------------------------------------------------------------------------------

void touch(char *p, u64 n)
{
    for (u64 i = 0; i < n; i += PAGE) p[i] = char(i);
}

void mem_mode()
{
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank_, MPI_INFO_NULL, &node);
    int nr, ns;
    MPI_Comm_rank(node, &nr);
    MPI_Comm_size(node, &ns);
    const u64 per = 256 * MB, pages = per / PAGE;
    if (rank_ == 0)
        std::printf("== first touch and release, %llu MiB per rank (%llu pages); us per page, slowest rank\n",
                    (unsigned long long)(per / MB), (unsigned long long)pages);
    if (rank_ == 0) std::printf("%-26s %8s %8s %8s %8s\n", "kind / ranks touching", "1", "2", "4", "8");
    const char *names[3] = {"private (new[])", "one window, rank 0 maps", "one window per rank"};
    for (int kind = 0; kind < 3; kind++) {
        std::string touch_row, free_row;
        for (int k = 1; k <= 8; k *= 2) {
            if (k > ns) {
                touch_row += "        -";
                free_row += "        -";
                continue;
            }
            const bool me = nr < k;
            double tt = 0, tf = 0;
            if (kind == 0) {
                char *p = me ? new char[per] : nullptr;
                MPI_Barrier(node);
                double t = MPI_Wtime();
                if (me) touch(p, per);
                tt = MPI_Wtime() - t;
                MPI_Barrier(node);
                t = MPI_Wtime();
                delete[] p;
                tf = MPI_Wtime() - t;
            } else if (kind == 1) {
                MPI_Win win;
                void *mine, *base;
                MPI_Aint sz;
                int du;
                MPI_Win_allocate_shared(nr == 0 ? MPI_Aint(per * k) : 0, 1, MPI_INFO_NULL, node, &mine, &win);
                MPI_Win_shared_query(win, 0, &sz, &du, &base);
                MPI_Barrier(node);
                double t = MPI_Wtime();
                if (me) touch(static_cast<char *>(base) + per * nr, per);
                tt = MPI_Wtime() - t;
                MPI_Barrier(node);
                t = MPI_Wtime();
                MPI_Win_free(&win);
                tf = MPI_Wtime() - t;
            } else {
                std::vector<MPI_Win> wins(k);
                char *own = nullptr;
                for (int j = 0; j < k; j++) {
                    void *mine;
                    MPI_Win_allocate_shared(nr == j ? MPI_Aint(per) : 0, 1, MPI_INFO_NULL, node, &mine, &wins[j]);
                    if (nr == j) own = static_cast<char *>(mine);
                }
                MPI_Barrier(node);
                double t = MPI_Wtime();
                if (me) touch(own, per);
                tt = MPI_Wtime() - t;
                MPI_Barrier(node);
                t = MPI_Wtime();
                for (int j = 0; j < k; j++) MPI_Win_free(&wins[j]);
                tf = MPI_Wtime() - t;
            }
            tt = slowest(tt);
            tf = slowest(tf);
            char buf[32];
            std::snprintf(buf, sizeof buf, " %8.3f", tt * 1e6 / pages);
            touch_row += buf;
            std::snprintf(buf, sizeof buf, " %8.3f", tf * 1e6 / (pages * k));
            free_row += buf;
        }
        if (rank_ == 0) {
            std::printf("%-26s%s   touch\n", names[kind], touch_row.c_str());
            std::printf("%-26s%s   release (per page released)\n", "", free_row.c_str());
        }
    }

    // Copy bandwidth over private buffers: one rank alone, then every rank of the node at once.
    const u64 len = 256 * MB;
    std::vector<char> src(len, 1), dst(len, 0);
    for (int all = 0; all < 2; all++) {
        const bool me = all || nr == 0;
        MPI_Barrier(node);
        const double t = MPI_Wtime();
        if (me)
            for (int rep = 0; rep < 4; rep++) std::memcpy(dst.data(), src.data(), len);
        const double el = slowest(me ? MPI_Wtime() - t : 0);
        const double per_rank = 4.0 * len / el / 1e9;
        if (rank_ == 0)
            std::printf("== memcpy %s: %.2f GB/s per rank, %.2f GB/s together\n",
                        all ? "on every rank at once" : "on one rank", per_rank, per_rank * (all ? ns : 1));
    }
    MPI_Comm_free(&node);
}

// ---- net ----------------------------------------------------------------------------------

std::vector<int> node_of;   // world rank -> node index

void pingpong(int a, int b, const char *what)
{
    const u64 sizes[] = {8, 4096, 64 * 1024, MB, 16 * MB, 64 * MB};
    std::vector<char> buf(64 * MB, 1);
    if (rank_ == 0) std::printf("== ping-pong %s (ranks %d and %d)\n", what, a, b);
    for (const u64 s : sizes) {
        const int reps = s <= 64 * 1024 ? 1000 : s <= MB ? 100 : 10;
        MPI_Barrier(MPI_COMM_WORLD);
        double t = MPI_Wtime();
        for (int i = 0; i < reps; i++) {
            if (rank_ == a) {
                MPI_Send(buf.data(), int(s), MPI_BYTE, b, 0, MPI_COMM_WORLD);
                MPI_Recv(buf.data(), int(s), MPI_BYTE, b, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else if (rank_ == b) {
                MPI_Recv(buf.data(), int(s), MPI_BYTE, a, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(buf.data(), int(s), MPI_BYTE, a, 0, MPI_COMM_WORLD);
            }
        }
        const double one_way = slowest(rank_ == a ? (MPI_Wtime() - t) / reps / 2 : 0);
        if (rank_ == 0)
            std::printf("  %9llu B: %9.2f us one way, %7.2f GB/s\n", (unsigned long long)s, one_way * 1e6,
                        double(s) / one_way / 1e9);
    }
}

void net_mode()
{
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank_, MPI_INFO_NULL, &node);
    int lead, nr;
    MPI_Comm_rank(node, &nr);
    MPI_Allreduce(&rank_, &lead, 1, MPI_INT, MPI_MIN, node);
    std::vector<int> leads(size_), locals(size_);
    MPI_Allgather(&lead, 1, MPI_INT, leads.data(), 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(&nr, 1, MPI_INT, locals.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::vector<int> distinct(leads);
    std::sort(distinct.begin(), distinct.end());
    distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
    node_of.resize(size_);
    for (int r = 0; r < size_; r++)
        node_of[r] = int(std::lower_bound(distinct.begin(), distinct.end(), leads[r]) - distinct.begin());
    if (rank_ == 0) {
        std::printf("== %d ranks on %zu node(s):", size_, distinct.size());
        for (int r = 0; r < size_; r++) std::printf(" %d", node_of[r]);
        std::printf("\n");
    }
    int same = -1, other = -1;
    for (int r = 1; r < size_; r++) {
        if (same < 0 && node_of[r] == node_of[0]) same = r;
        if (other < 0 && node_of[r] != node_of[0]) other = r;
    }
    if (same > 0) pingpong(0, same, "within a node");
    if (other > 0) pingpong(0, other, "across nodes");

    // Every node-0 rank streams to the node-1 rank with the same local index, at once.
    if (distinct.size() == 2) {
        int partner = -1;
        for (int r = 0; r < size_; r++)
            if (node_of[r] != node_of[rank_] && locals[r] == nr) partner = r;
        const u64 s = 64 * MB;
        std::vector<char> out(s, 1), in(s, 0);
        for (int both = 0; both < 2; both++) {
            MPI_Barrier(MPI_COMM_WORLD);
            const double t = MPI_Wtime();
            std::vector<MPI_Request> req;
            for (int rep = 0; rep < 4 && partner >= 0; rep++) {
                MPI_Request q;
                if (node_of[rank_] == 0 || both) {
                    MPI_Isend(out.data(), int(s), MPI_BYTE, partner, rep, MPI_COMM_WORLD, &q);
                    req.push_back(q);
                }
                if (node_of[rank_] == 1 || both) {
                    MPI_Irecv(in.data(), int(s), MPI_BYTE, partner, rep, MPI_COMM_WORLD, &q);
                    req.push_back(q);
                }
                MPI_Waitall(int(req.size()), req.data(), MPI_STATUSES_IGNORE);
                req.clear();
            }
            const double el = slowest(MPI_Wtime() - t);
            int pairs = 0;
            for (int r = 0; r < size_; r++)
                if (node_of[r] == 0)
                    for (int x = 0; x < size_; x++) pairs += node_of[x] == 1 && locals[x] == locals[r];
            const double bytes = 4.0 * s * pairs * (both ? 2 : 1);
            if (rank_ == 0)
                std::printf("== %d pairs streaming %s across nodes: %.2f GB/s in total\n", pairs,
                            both ? "both ways" : "one way", bytes / el / 1e9);
        }
    }

    // MPI_Alltoall: every rank sends a block to every rank.
    for (const u64 blk : {u64(64 * 1024), MB, 8 * MB}) {
        std::vector<char> sb(blk * size_, 1), rb(blk * size_, 0);
        MPI_Barrier(MPI_COMM_WORLD);
        const double t = MPI_Wtime();
        for (int rep = 0; rep < 4; rep++)
            MPI_Alltoall(sb.data(), int(blk), MPI_BYTE, rb.data(), int(blk), MPI_BYTE, MPI_COMM_WORLD);
        const double el = slowest(MPI_Wtime() - t) / 4;
        if (rank_ == 0)
            std::printf("== MPI_Alltoall %7llu KiB per pair: %.3f s, %.2f GB/s moved in total\n",
                        (unsigned long long)(blk / 1024), el, double(blk) * size_ * (size_ - 1) / el / 1e9);
    }
    MPI_Comm_free(&node);
}

}  // namespace

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "mem")
        mem_mode();
    else if (mode == "net")
        net_mode();
    else if (rank_ == 0)
        std::fprintf(stderr, "usage: %s mem|net\n", argv[0]);
    MPI_Finalize();
    return 0;
}
