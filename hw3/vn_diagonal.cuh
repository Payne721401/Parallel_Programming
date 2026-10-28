// Read row-prefix table Q. Build A separately, then overwrite Q with B only
// after every A read has finished. Each diagonal belongs to exactly one warp.
__device__ void build_vn_diagonals(uint16_t *q, int rw, int rh) {
    const int pitch = rw + 1;
    uint16_t *a = q + (rh + 1)*pitch;
    const int lane = int(threadIdx.x), warp = int(threadIdx.y);
    const int tid = warp*32 + lane;
    for (int x = tid; x < pitch; x += 256) a[x] = 0;
    for (int pass = 0; pass < 2; ++pass) {
        uint16_t *out = pass == 0 ? a : q;
        for (int diagonal = warp; diagonal < pitch + rh - 1; diagonal += 8) {
            const int row0 = diagonal < pitch ? 1 : diagonal - pitch + 2;
            const int col0 = diagonal < pitch ? diagonal : (pass == 0 ? 0 : rw);
            const int horizontal = pass == 0 ? pitch-col0 : col0+1;
            const int vertical = rh-row0+1;
            const int length = horizontal < vertical ? horizontal : vertical;
            unsigned carry = 0;
            for (int chunk = 0; chunk < length; chunk += 32) {
                const int i = chunk + lane;
                const int row = row0+i, col = pass == 0 ? col0+i : col0-i;
                const unsigned v = i < length ? q[row*pitch+col] : 0;
                const unsigned prefix = warp_prefix(v, lane);
                if (i < length) out[row*pitch+col] = uint16_t(carry+prefix);
                carry += __shfl_sync(0xffffffffu, prefix, 31);
            }
        }
        __syncthreads(); // A completed before Q is overwritten; B ready before queries
    }
}
