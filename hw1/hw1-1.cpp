#include <iostream>
#include <vector>
#include <algorithm>
#include <png.h>
#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include <omp.h>
#include <sched.h>
// #include <pthread.h>

// PNG output tuning, sweepable with -DOUT_ZLEVEL / -DOUT_ROWFILTER. Deflate is
// lossless at every level, so neither can change the decoded pixels.
#ifndef OUT_ZLEVEL
#define OUT_ZLEVEL 0
#endif
#ifndef OUT_ROWFILTER
#define OUT_ROWFILTER PNG_FILTER_NONE
#endif

// Spec 1.4: size the pool from the affinity mask. The cluster overwrites
// OMP_NUM_THREADS to 1, so the environment cannot be trusted.
static int usableCpus() {
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        int n = CPU_COUNT(&set);
        if (n > 0) return n;
    }
    return omp_get_max_threads();
}

// ---------- adaptive filtering ----------

struct RGB {
    int r, g, b;
};

double calculateLuminance(const RGB& pixel) {
    return 0.299 * pixel.r + 0.587 * pixel.g + 0.114 * pixel.b;
}


int determineKernelSize(double brightness) {
    return brightness > 128 ? 11 : 5;
}

// The largest radius determineKernelSize can produce: 11 / 2 == 5.
static const int RMAX = 5;

// The RMAX columns at each end of a row, where y + j has to be clamped.
static inline int clampedBox(const unsigned char* const* rows, int rad, int y, int width) {
    int s = 0;
    for (int i = -rad; i <= rad; i++) {
        const unsigned char* r = rows[i + RMAX];
        for (int j = -rad; j <= rad; j++)
            s += r[std::min(std::max(y + j, 0), width - 1)];
    }
    const int n = 2 * rad + 1;
    return s / (n * n);
}

// Flat unsigned char planes rather than vector<vector<int>>: a byte per sample
// instead of an int, and consecutive rows actually consecutive. The row lookups
// are clamped once per output row, and y is split so the interior needs no
// clamping -- which makes the inner trip counts compile-time constants, 11x11
// or 5x5. Integer addition is associative, so none of this changes the sum.
void applyFilterToChannel(
    const unsigned char* input,
    unsigned char* output,
    const unsigned char* radii,
    int height,
    int width
) {
    // Each output row is written by one thread and `input` is only read, so
    // rows are independent. schedule(runtime) lets OMP_SCHEDULE sweep it;
    // measurement says it barely matters, because the noisy input leaves every
    // row much the same mix of large and small kernels.
    #pragma omp parallel for schedule(runtime)
    for (int x = 0; x < height; x++) {
        const unsigned char* rows[2 * RMAX + 1];
        for (int i = -RMAX; i <= RMAX; i++)
            rows[i + RMAX] =
                input + (size_t)std::min(std::max(x + i, 0), height - 1) * width;

        const unsigned char* rad = radii + (size_t)x * width;
        unsigned char* out = output + (size_t)x * width;

        const int yLo = std::min(RMAX, width);
        const int yHi = std::max(yLo, width - RMAX);

        for (int y = 0; y < yLo; y++) out[y] = (unsigned char)clampedBox(rows, rad[y], y, width);
        for (int y = yHi; y < width; y++) out[y] = (unsigned char)clampedBox(rows, rad[y], y, width);

        for (int y = yLo; y < yHi; y++) {
            if (rad[y] == RMAX) {
                int s = 0;
                for (int i = 0; i < 2 * RMAX + 1; i++) {
                    const unsigned char* r = rows[i] + y - RMAX;
                    for (int j = 0; j < 2 * RMAX + 1; j++) s += r[j];
                }
                out[y] = (unsigned char)(s / ((2 * RMAX + 1) * (2 * RMAX + 1)));
            } else {
                int s = 0;
                for (int i = RMAX - 2; i <= RMAX + 2; i++) {
                    const unsigned char* r = rows[i] + y - 2;
                    for (int j = 0; j < 5; j++) s += r[j];
                }
                out[y] = (unsigned char)(s / 25);
            }
        }
    }
}

// Writes straight into the planes the encoder will read, so there is no
// interleaved vector<vector<RGB>> in between -- 219 MB at 18 Mpixel.
void adaptiveFilterRGB(
    const unsigned char* redChannel,
    const unsigned char* greenChannel,
    const unsigned char* blueChannel,
    unsigned char* outRed,
    unsigned char* outGreen,
    unsigned char* outBlue,
    int height,
    int width
) {
    // The radius, not the kernel size, and one byte instead of an int.
    std::vector<unsigned char> radii((size_t)height * width);

    #pragma omp parallel for schedule(static)
    for (int x = 0; x < height; x++)
        for (int y = 0; y < width; y++) {
            const size_t p = (size_t)x * width + y;
            // Reassembled into an RGB so calculateLuminance evaluates the same
            // double expression. The integer form 299r + 587g + 114b > 128000
            // is not equivalent -- the weights are inexact in binary, so a flat
            // mid-grey region would pick a different kernel per pixel.
            RGB px;
            px.r = redChannel[p];
            px.g = greenChannel[p];
            px.b = blueChannel[p];
            radii[p] = (unsigned char)(determineKernelSize(calculateLuminance(px)) / 2);
        }

    applyFilterToChannel(redChannel, outRed, radii.data(), height, width);
    applyFilterToChannel(greenChannel, outGreen, radii.data(), height, width);
    applyFilterToChannel(blueChannel, outBlue, radii.data(), height, width);
}

// ---------- shared PNG I/O ----------

// Decodes straight into the three channel planes. png_set_strip_alpha rather
// than png_set_filler, which expanded RGB to RGBA only to discard the alpha;
// and one contiguous buffer instead of a malloc per row.
void read_png_planes(char* file_name,
                     std::vector<unsigned char>& redChannel,
                     std::vector<unsigned char>& greenChannel,
                     std::vector<unsigned char>& blueChannel,
                     int* out_w, int* out_h) {
    FILE *fp = fopen(file_name, "rb");
    if (!fp) {
        std::cerr << "Error: Cannot open file " << file_name << std::endl;
        exit(EXIT_FAILURE);
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) {
        std::cerr << "Error: Cannot create PNG read structure" << std::endl;
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        std::cerr << "Error: Cannot create PNG info structure" << std::endl;
        png_destroy_read_struct(&png, nullptr, nullptr);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    if (setjmp(png_jmpbuf(png))) {
        std::cerr << "Error during PNG creation" << std::endl;
        png_destroy_read_struct(&png, &info, nullptr);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    png_init_io(png, fp);
    png_read_info(png, info);

    int width = png_get_image_width(png, info);
    int height = png_get_image_height(png, info);
    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth = png_get_bit_depth(png, info);

    if(bit_depth == 16)
        png_set_strip_16(png);

    if(color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png);

    if(color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png);

    if(png_get_valid(png, info, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png);

    png_set_strip_alpha(png);   // every source type reduces to plain RGB

    if(color_type == PNG_COLOR_TYPE_GRAY ||
       color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);

    png_read_update_info(png, info);

    const size_t rowbytes = png_get_rowbytes(png, info);
    png_byte* raw = (png_byte*)malloc(rowbytes * (size_t)height);
    png_bytep* row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    if (!raw || !row_pointers) {
        std::cerr << "Error: out of memory for " << width << "x" << height << std::endl;
        exit(EXIT_FAILURE);
    }
    for (int y = 0; y < height; y++)
        row_pointers[y] = raw + (size_t)y * rowbytes;

    png_read_image(png, row_pointers);

    fclose(fp);

    redChannel.resize((size_t)height * width);
    greenChannel.resize((size_t)height * width);
    blueChannel.resize((size_t)height * width);
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; y++) {
        const png_byte* row = raw + (size_t)y * rowbytes;
        const size_t base = (size_t)y * width;
        for (int x = 0; x < width; x++) {
            redChannel[base + x] = row[x * 3];
            greenChannel[base + x] = row[x * 3 + 1];
            blueChannel[base + x] = row[x * 3 + 2];
        }
    }
    free(row_pointers);
    free(raw);
    *out_w = width;
    *out_h = height;

    png_destroy_read_struct(&png, &info, nullptr);
}

void write_png_file(char* file_name, const unsigned char* R,
                    const unsigned char* G, const unsigned char* B,
                    int width, int height) {

    FILE *fp = fopen(file_name, "wb");
    if (!fp) {
        std::cerr << "Error: Cannot open file " << file_name << std::endl;
        exit(EXIT_FAILURE);
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) {
        std::cerr << "Error: Cannot create PNG write structure" << std::endl;
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        std::cerr << "Error: Cannot create PNG info structure" << std::endl;
        png_destroy_write_struct(&png, nullptr);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    if (setjmp(png_jmpbuf(png))) {
        std::cerr << "Error during PNG creation" << std::endl;
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    png_init_io(png, fp);

    // libpng's defaults spend this function's time hunting for longer LZ77
    // matches and filtering every row five ways to keep the best. Both are
    // lossless, so the decoded pixels are unchanged either way.
    png_set_compression_level(png, OUT_ZLEVEL);
    png_set_filter(png, PNG_FILTER_TYPE_BASE, OUT_ROWFILTER);

    png_set_IHDR(
        png,
        info,
        width, height,
        8,
        PNG_COLOR_TYPE_RGB,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT
    );
    png_write_info(png, info);

    const size_t rowbytes = png_get_rowbytes(png, info);
    png_byte* raw = (png_byte*)malloc(rowbytes * (size_t)height);
    png_bytep* row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    if (!raw || !row_pointers) {
        std::cerr << "Error: out of memory for " << width << "x" << height << std::endl;
        exit(EXIT_FAILURE);
    }
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; y++) {
        png_byte* row = raw + (size_t)y * rowbytes;
        row_pointers[y] = row;
        const size_t base = (size_t)y * width;
        for (int x = 0; x < width; x++) {
            row[x * 3] = R[base + x];
            row[x * 3 + 1] = G[base + x];
            row[x * 3 + 2] = B[base + x];
        }
    }

    png_write_image(png, row_pointers);
    png_write_end(png, nullptr);

    free(row_pointers);
    free(raw);

    png_destroy_write_struct(&png, &info);
    fclose(fp);
}

// ---------- driver ----------

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <inputfile.png> <outputfile.png>" << std::endl;
        return -1;
    }

    char* input_file = argv[1];
    char* output_file = argv[2];

    int nthreads = usableCpus();
    if (const char* e = getenv("PP_THREADS")) {   // our own knob, for sweeps
        int v = atoi(e);
        if (v > 0) nthreads = v;
    }
    omp_set_num_threads(nthreads);

    // Backstop for schedule(runtime): with OMP_SCHEDULE unset GCC falls back
    // to static.
    if (!getenv("OMP_SCHEDULE")) omp_set_schedule(omp_sched_guided, 4);

    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<unsigned char> redChannel, greenChannel, blueChannel;
    int width = 0, height = 0;
    read_png_planes(input_file, redChannel, greenChannel, blueChannel, &width, &height);

    auto t1 = std::chrono::high_resolution_clock::now();

    std::vector<unsigned char> outRed((size_t)height * width),
                               outGreen((size_t)height * width),
                               outBlue((size_t)height * width);

    auto t2 = std::chrono::high_resolution_clock::now();

    adaptiveFilterRGB(redChannel.data(), greenChannel.data(), blueChannel.data(),
                      outRed.data(), outGreen.data(), outBlue.data(), height, width);

    auto t3 = std::chrono::high_resolution_clock::now();

    write_png_file(output_file, outRed.data(), outGreen.data(),
                   outBlue.data(), width, height);

    auto t4 = std::chrono::high_resolution_clock::now();

    // PP_TIMING=1 srun -n 1 -c 8 ./hw1-1 in.png out.png
    if (getenv("PP_TIMING")) {
        auto ms = [](std::chrono::high_resolution_clock::time_point a,
                     std::chrono::high_resolution_clock::time_point b) {
            return std::chrono::duration<double>(b - a).count() * 1000.0;
        };
        fprintf(stderr,
                "threads %2d | read %7.1f | alloc %7.1f | filter %8.1f | write %7.1f | total %8.1f  (ms)\n",
                omp_get_max_threads(), ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t3, t4), ms(t0, t4));
    }

    return 0;
}