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

// The filter loop below uses schedule(runtime) so its schedule can be swept
// with OMP_SCHEDULE="guided,4" etc. without recompiling. main() picks a
// default when OMP_SCHEDULE is unset, which is how the judge runs us.

// PNG output tuning. The judge compares the decoded image pixel by pixel, not
// the file's bytes, so both of these only trade CPU time against file size and
// can never change the answer. Sweep without editing the source:
//   make CXXFLAGS="-std=c++11 -O3 -pthread -fopenmp -DOUT_ROWFILTER=PNG_FILTER_SUB"
#ifndef OUT_ZLEVEL
#define OUT_ZLEVEL 1
#endif
#ifndef OUT_ROWFILTER
#define OUT_ROWFILTER PNG_FILTER_NONE
#endif

// The judge runs us as `srun -c N ./hw1-1 ...` and does not set
// OMP_NUM_THREADS, and this cluster overwrites it to 1 anyway. The CPU
// affinity mask is what srun actually handed us, so size the pool from that
// rather than trusting the environment.
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
    return brightness > 128 ? 10 : 5;
}

void applyFilterToChannel(
    const std::vector<std::vector<int>>& input, 
    std::vector<std::vector<int>>& output, 
    const std::vector<std::vector<int>>& kernelSizes, 
    int height,
    int width
) {
    // Each output row is written by exactly one thread and `input` is only
    // read, so rows are independent. The schedule is left to OMP_SCHEDULE /
    // omp_set_schedule() so it can be swept without recompiling; measurement
    // says it barely matters here, because the input is a noisy image and that
    // noise leaves every row with much the same mix of large and small kernels.
    #pragma omp parallel for schedule(runtime)
    for (int x = 0; x < height; x++) {
        for (int y = 0; y < width; y++) {
            int kernelRadius = kernelSizes[x][y] / 2;
            // The border clamps rather than skipping, so every (i, j) pair runs
            // and the divisor is a constant -- no point counting it tap by tap.
            int taps = (2 * kernelRadius + 1) * (2 * kernelRadius + 1);
            // int, not double: a window sums at most 121 * 255 = 30855, exact
            // either way, but integer add is 1 cycle instead of 4 and is
            // associative, which is what lets the reduction vectorize.
            int filteredPixel = 0;

            for (int i = -kernelRadius; i <= kernelRadius; i++) {
                // The row index does not depend on j. Hoisting the lookup here
                // turns two dependent loads per tap into one.
                const std::vector<int>& row =
                    input[std::min(std::max(x + i, 0), height - 1)];
                for (int j = -kernelRadius; j <= kernelRadius; j++) {
                    filteredPixel += row[std::min(std::max(y + j, 0), width - 1)];
                }
            }

            // Both operands are non-negative, so integer division truncates the
            // same way static_cast<int>(double / double) did: bit-identical.
            output[x][y] = filteredPixel / taps;
        }
    }
}

void adaptiveFilterRGB(
    const std::vector<std::vector<RGB>>& inputImage,
    std::vector<std::vector<RGB>>& outputImage,
    int height, 
    int width
) {
    std::vector<std::vector<int>> redChannel(height, std::vector<int>(width));
    std::vector<std::vector<int>> greenChannel(height, std::vector<int>(width));
    std::vector<std::vector<int>> blueChannel(height, std::vector<int>(width));

    #pragma omp parallel for schedule(static)
    for (int x = 0; x < height; x++) {
        for (int y = 0; y < width; y++) {
            redChannel[x][y] = inputImage[x][y].r;
            greenChannel[x][y] = inputImage[x][y].g;
            blueChannel[x][y] = inputImage[x][y].b;
        }
    }

    std::vector<std::vector<int>> kernelSizes(height, std::vector<int>(width));

    #pragma omp parallel for schedule(static)
    for (int x = 0; x < height; x++) {
        for (int y = 0; y < width; y++) {
            double brightness = calculateLuminance(inputImage[x][y]);
            kernelSizes[x][y] = determineKernelSize(brightness);
        }
    }

    std::vector<std::vector<int>> tempRed(height, std::vector<int>(width));
    std::vector<std::vector<int>> tempGreen(height, std::vector<int>(width));
    std::vector<std::vector<int>> tempBlue(height, std::vector<int>(width));

    applyFilterToChannel(redChannel, tempRed, kernelSizes, height, width);
    applyFilterToChannel(greenChannel, tempGreen, kernelSizes, height, width);
    applyFilterToChannel(blueChannel, tempBlue, kernelSizes, height, width);

    #pragma omp parallel for schedule(static)
    for (int x = 0; x < height; x++) {
        for (int y = 0; y < width; y++) {
            outputImage[x][y].r = tempRed[x][y];
            outputImage[x][y].g = tempGreen[x][y];
            outputImage[x][y].b = tempBlue[x][y];
        }
    }
}

// ---------- shared PNG I/O ----------

void read_png_file(char* file_name, std::vector<std::vector<RGB>>& image) {
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

    if(color_type == PNG_COLOR_TYPE_RGB ||
       color_type == PNG_COLOR_TYPE_GRAY ||
       color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_filler(png, 0xFF, PNG_FILLER_AFTER);

    if(color_type == PNG_COLOR_TYPE_GRAY ||
       color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);

    png_read_update_info(png, info);

    png_bytep* row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    for(int y = 0; y < height; y++) {
        row_pointers[y] = (png_byte*)malloc(png_get_rowbytes(png,info));
    }

    png_read_image(png, row_pointers);

    fclose(fp);

    image.resize(height, std::vector<RGB>(width));
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; y++) {
        png_bytep row = row_pointers[y];
        for (int x = 0; x < width; x++) {
            png_bytep px = &(row[x * 4]);
            image[y][x].r = px[0];
            image[y][x].g = px[1];
            image[y][x].b = px[2];
        }
        free(row_pointers[y]);
    }
    free(row_pointers);

    png_destroy_read_struct(&png, &info, nullptr);
}

void write_png_file(char* file_name, std::vector<std::vector<RGB>>& image) {
    int width = image[0].size();
    int height = image.size();

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

    // zlib level 6 (libpng's default) spends most of this function's time
    // hunting for longer LZ77 matches, and PNG_ALL_FILTERS (also the default)
    // filters every row five ways and keeps whichever scores best. Deflate is
    // lossless at every level, so the decoded pixels are unchanged either way.
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

    png_bytep* row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    size_t rowbytes = png_get_rowbytes(png, info);
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; y++) {
        row_pointers[y] = (png_byte*)malloc(rowbytes);
        for (int x = 0; x < width; x++) {
            row_pointers[y][x * 3] = image[y][x].r;
            row_pointers[y][x * 3 + 1] = image[y][x].g;
            row_pointers[y][x * 3 + 2] = image[y][x].b;
        }
    }

    png_write_image(png, row_pointers);
    png_write_end(png, nullptr);

    for (int y = 0; y < height; y++) {
        free(row_pointers[y]);
    }
    free(row_pointers);

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

    // Backstop for schedule(runtime): with OMP_SCHEDULE unset GCC falls back to
    // static, which would silently throw away the load balancing the filter
    // needs. Sweep with OMP_SCHEDULE, then hardcode the winner here.
    if (!getenv("OMP_SCHEDULE")) omp_set_schedule(omp_sched_guided, 4);

    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<RGB>> inputImage;
    read_png_file(input_file, inputImage);

    int height = inputImage.size();
    int width = inputImage[0].size();

    auto t1 = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<RGB>> outputImage(height, std::vector<RGB>(width));

    auto t2 = std::chrono::high_resolution_clock::now();

    adaptiveFilterRGB(inputImage, outputImage, height, width);

    auto t3 = std::chrono::high_resolution_clock::now();

    write_png_file(output_file, outputImage);

    auto t4 = std::chrono::high_resolution_clock::now();

    // Gated on an env var so the judge run stays clean and these lines never
    // have to be commented back out: PP_TIMING=1 srun ... ./hw1-1 in.png out.png
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