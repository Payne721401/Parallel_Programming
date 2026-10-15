// Exercise the submitted accumulator directly without changing its public interface.
#include "../hw2-2/hw2-2.cpp"

// Every check runs twice: with the scalar take() and with take_simd() (PP_EMIT_SIMD=1).
static bool simd_take = false;

static void check_row(Accumulator &acc, bool scan, const std::vector<u32> &expected)
{
    const Entry prefix{UINT32_MAX, 123};
    std::vector<Entry> out{prefix};
    if (simd_take) {
        if (scan) acc.take_simd<true>(out);
        else acc.take_simd(out);
    } else if (scan) {
        acc.take<true>(out);
    } else {
        acc.take(out);
    }
    size_t at = 1;
    for (u32 j = 0; j < expected.size(); j++) {
        if (!expected[j]) continue;
        if (at == out.size() || out[at].col != j || out[at].val != expected[j])
            throw std::runtime_error("accumulator output mismatch");
        at++;
    }
    if (at != out.size() || out[0].col != prefix.col || out[0].val != prefix.val)
        throw std::runtime_error("accumulator append/reset mismatch");
}

static void run_checks()
{
    for (u32 n : {1u, 65u, 6143u, 8192u}) {
        // One bitmap row mixing words take_simd() handles with compresses (64 and exactly 8
        // bits) and with the scalar loop (7 bits and 1 bit).
        {
            Accumulator acc(n);
            std::vector<u32> expected(n, 0);
            std::vector<Entry> entries;
            for (u32 j = 0; j < n; j++) {
                const u32 w = j / 64, b = j % 64;
                const bool on = w % 4 == 0 ? true : w % 4 == 1 ? b < 7 : w % 4 == 2 ? b % 8 == 3 : b == 63;
                if (!on) continue;
                entries.push_back({j, 1 + j % 8});
                expected[j] = 3 * (1 + j % 8);
            }
            acc.add(3, entries.data(), entries.size());
            check_row(acc, false, expected);
            check_row(acc, false, std::vector<u32>(n, 0));
        }
        if (dense_hot_row(0, n) || !dense_hot_row((n + 3) / 4, n) ||
            dense_hot_row((n - 1) / 4, n))
            throw std::runtime_error("dense hot density threshold mismatch");
        const u64 threshold = SCAN_WORK_PER_COLUMN * n;
        if (scan_for_row(AccumulatorMode::Bitmap, threshold, n) ||
            !scan_for_row(AccumulatorMode::Scan, 0, n) ||
            scan_for_row(AccumulatorMode::Auto, threshold - 1, n) ||
            !scan_for_row(AccumulatorMode::Auto, threshold, n))
            throw std::runtime_error("accumulator selection mismatch");
        for (int pattern = 0; pattern < 3; pattern++) {
            Accumulator acc(n);
            for (u32 row = 0; row < 32; row++) {
                const bool scan = pattern == 1 || (pattern == 2 && row % 2);
                std::vector<u32> expected(n, 0);
                for (u32 pass = 0; pass < 4; pass++) {
                    std::vector<Entry> entries;
                    if (row % 8) {
                        const u32 stride = row % 3 ? 97 : 1;
                        for (u32 j = pass % 3; j < n; j += stride) {
                            const u32 value = 1 + (j + row + pass) % 8;
                            entries.push_back({j, value});
                            expected[j] += (pass + 1) * value;
                        }
                    }
                    if (scan) acc.add<true>(pass + 1, entries.data(), entries.size());
                    else acc.add(pass + 1, entries.data(), entries.size());
                }
                check_row(acc, scan, expected);
                check_row(acc, !scan, std::vector<u32>(n, 0));
            }
            // Partial-C values used by the merge can be much larger than input B values.
            const Entry large{n - 1, 1u << 29};
            for (bool scan : {false, true}) {
                if (scan) acc.add<true>(2, &large, 1);
                else acc.add(2, &large, 1);
                std::vector<u32> expected(n, 0);
                expected.back() = 1u << 30;
                check_row(acc, scan, expected);
                check_row(acc, !scan, std::vector<u32>(n, 0));
            }
        }
        Accumulator mixed(n);
        for (u32 row = 0; row < 12; row++) {
            std::vector<u32> expected(n, 0);
            for (u32 pass = 0; pass < 4; pass++) {
                std::vector<uint8_t> dense(n, 0);
                std::vector<Entry> sparse;
                for (u32 j = 0; row % 4 && j < n; j++) {
                    if ((j + pass) % 3) continue;
                    dense[j] = uint8_t(1 + (j + pass) % 8);
                    sparse.push_back({j, dense[j]});
                    expected[j] += 8 * u32(dense[j]);
                }
                if (pass % 2) mixed.add_dense_scan(8, dense.data());
                else mixed.add<true>(8, sparse.data(), sparse.size());
            }
            check_row(mixed, true, expected);
            const Entry last{n - 1, 8};
            mixed.add(8, &last, 1);
            expected.assign(n, 0);
            expected.back() = 64;
            check_row(mixed, false, expected);
            check_row(mixed, true, std::vector<u32>(n, 0));
        }
    }
}

int main()
{
    run_checks();
    simd_take = true;
    run_checks();
    std::printf("PASS accumulator bitmap/scan/dense-hot mixing and reset, density/auto thresholds and large sums; "
                "scalar and SIMD take (avx512=%d)\n", EMIT_AVX512 ? 1 : 0);
}
