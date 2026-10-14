// Exercise the submitted accumulator directly without changing its public interface.
#include "../hw2-2/hw2-2.cpp"

static void check_row(Accumulator &acc, bool scan, const std::vector<u32> &expected)
{
    const Entry prefix{UINT32_MAX, 123};
    std::vector<Entry> out{prefix};
    if (scan) acc.take<true>(out);
    else acc.take(out);
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

int main()
{
    for (u32 n : {1u, 65u, 8192u}) {
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
    }
    std::puts("PASS accumulator bitmap/scan/reset, empty/dense/sparse rows, auto threshold and large sums");
}
