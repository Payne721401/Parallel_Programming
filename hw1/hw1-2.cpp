#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <png.h>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <chrono>
#include <omp.h>
#include <sched.h>
// #include <pthread.h>

// ---------- staged timing ----------
// PP_TIMING=1 srun -n 1 -c 8 ./hw1-2 a.png b.png out.txt
// extractFeatures() runs twice, so its inner stages accumulate into globals.
static double g_tPyramid = 0.0, g_tDetect = 0.0, g_tDescribe = 0.0;

static inline double nowMs() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

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

// ---------- shared PNG I/O ----------

struct RGB {
    int r, g, b;
};

using Mat = std::vector<std::vector<double>>;

// The two images decode independently -- separate files, separate png_structp,
// and libpng is reentrant per struct -- so main runs these as two omp sections.
// Nested parallelism is off, so the decode stops at a contiguous RGB buffer
// (dropping the old vector<vector<RGB>>, 63 MB per image at 2640x1980) and the
// grayscale conversion runs afterwards with the whole team. png_set_strip_alpha
// replaces png_set_filler, which expanded RGB to RGBA only to discard the alpha.
static png_byte* read_png_raw(const char* file_name, int* out_w, int* out_h,
                              size_t* out_rowbytes) {
    FILE *fp = fopen(file_name, "rb");
    if (!fp) {
        std::cerr << "Error: Cannot open file " << file_name << std::endl;
        exit(EXIT_FAILURE);
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    png_infop info = png_create_info_struct(png);
    if (setjmp(png_jmpbuf(png))) {
        std::cerr << "Error during PNG creation" << std::endl;
        exit(EXIT_FAILURE);
    }

    png_init_io(png, fp);
    png_read_info(png, info);

    int width = png_get_image_width(png, info);
    int height = png_get_image_height(png, info);
    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth = png_get_bit_depth(png, info);

    if (bit_depth == 16) png_set_strip_16(png);
    if (color_type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);
    png_set_strip_alpha(png);   // every source type reduces to plain RGB
    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
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
    free(row_pointers);
    png_destroy_read_struct(&png, &info, nullptr);

    *out_w = width;
    *out_h = height;
    *out_rowbytes = rowbytes;
    return raw;
}

// ---------- grayscale + Gaussian scale space ----------

// The same expression in the same order as before: the luminance weights are
// inexact in binary, and every later extremum decision rests on these values.
Mat toGrayscale(const png_byte* raw, size_t rowbytes, int height, int width) {
    Mat gray(height, std::vector<double>(width));
    // Row y reads and writes only row y, and the work per row is uniform.
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; y++) {
        const png_byte* row = raw + (size_t)y * rowbytes;
        for (int x = 0; x < width; x++)
            gray[y][x] = (0.299 * row[x * 3] + 0.587 * row[x * 3 + 1] +
                          0.114 * row[x * 3 + 2]) / 255.0;
    }
    return gray;
}

// Separable Gaussian blur. Two independent passes (row-wise, then column-wise)
// -- each row/column is independent, this is the main parallelization target
// in the detection stage.
Mat gaussianBlur(const Mat& in, int height, int width, double sigma) {
    int radius = std::max(1, (int)std::ceil(3 * sigma));
    std::vector<double> kernel(2 * radius + 1);
    double sum = 0.0;
    for (int i = -radius; i <= radius; i++) {
        double v = std::exp(-(i * i) / (2.0 * sigma * sigma));
        kernel[i + radius] = v;
        sum += v;
    }
    for (double& v : kernel) v /= sum;

    // Both passes run x innermost instead of the kernel offset. Each out[x]
    // still accumulates i from -radius to +radius in that order, so the values
    // stay bit-identical; what changes is that different x become independent
    // (no shared accumulator chain) and contiguous, so the loads vectorize.

    // Interior columns need no clamping: x + i stays inside [0, width).
    const int xLo = std::min(radius, width);
    const int xHi = std::max(xLo, width - radius);

    Mat tmp(height, std::vector<double>(width));
    // Rows are independent, and each element is still summed by one thread in
    // the original order.
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; y++) {
        const double* src = in[y].data();
        double* dst = tmp[y].data();
        std::fill(dst, dst + width, 0.0);
        for (int i = -radius; i <= radius; i++) {
            const double kv = kernel[i + radius];
            for (int x = 0; x < xLo; x++)
                dst[x] += src[std::min(std::max(x + i, 0), width - 1)] * kv;
            for (int x = xLo; x < xHi; x++)
                dst[x] += src[x + i] * kv;
            for (int x = xHi; x < width; x++)
                dst[x] += src[std::min(std::max(x + i, 0), width - 1)] * kv;
        }
    }

    Mat out(height, std::vector<double>(width));
    // Reads rows y-radius..y+radius of `tmp` but only ever reads them, and
    // writes only out[y]. Overlapping reads are not a race.
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; y++) {
        double* dst = out[y].data();
        std::fill(dst, dst + width, 0.0);
        for (int i = -radius; i <= radius; i++) {
            // The row index is invariant in x, so clamp once per i.
            const double* src = tmp[std::min(std::max(y + i, 0), height - 1)].data();
            const double kv = kernel[i + radius];
            for (int x = 0; x < width; x++)
                dst[x] += src[x] * kv;
        }
    }
    return out;
}

Mat downsample2x(const Mat& in, int height, int width) {
    int nh = height / 2, nw = width / 2;
    Mat out(nh, std::vector<double>(nw));
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < nh; y++)
        for (int x = 0; x < nw; x++)
            out[y][x] = in[2 * y][2 * x];
    return out;
}

// ponytail: fixed pyramid parameters instead of adapting to image size --
// keeps the pipeline (and its correctness check) deterministic across test
// cases. Revisit if test images vary wildly in resolution.
const int NUM_OCTAVES = 4;
const int S = 3;              // Lowe's scales-per-octave
const int NUM_SCALES = S + 3; // Gaussian images per octave
const double SIGMA0 = 1.6;
const double CONTRAST_THRESH = 0.03;
const double EDGE_THRESH_R = 10.0;

struct Octave {
    int height, width;
    std::vector<Mat> gaussian; // NUM_SCALES images
    std::vector<Mat> dog;      // NUM_SCALES - 1 images
};

std::vector<Octave> buildPyramid(const Mat& gray, int height, int width) {
    std::vector<Octave> octaves(NUM_OCTAVES);
    double k = std::pow(2.0, 1.0 / S);

    Mat base = gray;
    int h = height, w = width;
    for (int o = 0; o < NUM_OCTAVES; o++) {
        Octave& oct = octaves[o];
        oct.height = h;
        oct.width = w;
        oct.gaussian.resize(NUM_SCALES);
        oct.gaussian[0] = base;
        for (int s = 1; s < NUM_SCALES; s++) {
            double sigma = SIGMA0 * std::pow(k, s);
            oct.gaussian[s] = gaussianBlur(base, h, w, sigma);
        }
        // Allocate first, then subtract: collapse(2) needs a perfect nest.
        oct.dog.resize(NUM_SCALES - 1);
        for (int s = 0; s < NUM_SCALES - 1; s++)
            oct.dog[s] = Mat(h, std::vector<double>(w));

        // collapse(2) rather than s alone: the deepest octave is only ~250
        // rows and there are 5 layers, so flattening (s, y) keeps 8 threads fed.
        #pragma omp parallel for collapse(2) schedule(static)
        for (int s = 0; s < NUM_SCALES - 1; s++)
            for (int y = 0; y < h; y++) {
                double* d = oct.dog[s][y].data();
                const double* hi = oct.gaussian[s + 1][y].data();
                const double* lo = oct.gaussian[s][y].data();
                for (int x = 0; x < w; x++) d[x] = hi[x] - lo[x];
            }

        if (o + 1 < NUM_OCTAVES) {
            base = downsample2x(oct.gaussian[S], h, w); // carry over scale = 2*sigma0
            h /= 2;
            w /= 2;
        }
    }
    return octaves;
}

// ---------- keypoint detection, orientation, descriptor ----------

struct Keypoint {
    int octave, layer;   // DoG layer index (1 .. NUM_SCALES-3, inclusive both sides)
    int x, y;             // pixel coords within that octave's resolution
    double scale;         // sigma at this layer, in octave-local units
    double orientation;   // radians
    std::vector<double> descriptor;
};

bool isExtremum(const std::vector<Mat>& dog, int s, int x, int y) {
    double v = dog[s][y][x];
    bool isMax = true, isMin = true;
    for (int ds = -1; ds <= 1 && (isMax || isMin); ds++)
        for (int dy = -1; dy <= 1 && (isMax || isMin); dy++)
            for (int dx = -1; dx <= 1 && (isMax || isMin); dx++) {
                if (ds == 0 && dy == 0 && dx == 0) continue;
                double n = dog[s + ds][y + dy][x + dx];
                if (n >= v) isMax = false;
                if (n <= v) isMin = false;
            }
    return isMax || isMin;
}

bool passesEdgeTest(const Mat& d, int x, int y) {
    double dxx = d[y][x + 1] + d[y][x - 1] - 2 * d[y][x];
    double dyy = d[y + 1][x] + d[y - 1][x] - 2 * d[y][x];
    double dxy = (d[y + 1][x + 1] - d[y + 1][x - 1] - d[y - 1][x + 1] + d[y - 1][x]) / 4.0;
    double trace = dxx + dyy;
    double det = dxx * dyy - dxy * dxy;
    if (det <= 0) return false;
    double ratio = (trace * trace) / det;
    return ratio < (EDGE_THRESH_R + 1) * (EDGE_THRESH_R + 1) / EDGE_THRESH_R;
}

// Independent per (octave, layer, pixel) -- another natural parallelization
// target. Extrema in different octaves/layers never interact.
std::vector<Keypoint> detectKeypoints(const std::vector<Octave>& octaves) {
    std::vector<Keypoint> keypoints;
    for (int o = 0; o < NUM_OCTAVES; o++) {
        const Octave& oct = octaves[o];
        for (int s = 1; s < (int)oct.dog.size() - 1; s++) {
            for (int y = 1; y < oct.height - 1; y++) {
                for (int x = 1; x < oct.width - 1; x++) {
                    if (std::fabs(oct.dog[s][y][x]) < CONTRAST_THRESH) continue;
                    if (!isExtremum(oct.dog, s, x, y)) continue;
                    if (!passesEdgeTest(oct.dog[s], x, y)) continue;

                    Keypoint kp;
                    kp.octave = o;
                    kp.layer = s;
                    kp.x = x;
                    kp.y = y;
                    kp.scale = SIGMA0 * std::pow(2.0, (double)s / S);
                    keypoints.push_back(kp);
                }
            }
        }
    }
    return keypoints;
}

// ponytail: single dominant orientation per keypoint (no histogram-peak
// splitting into multiple keypoints) -- simpler & still demonstrates the
// weighted-histogram parallelization pattern without duplicating keypoints.
void assignOrientation(Keypoint& kp, const Octave& oct) {
    const Mat& img = oct.gaussian[kp.layer];
    double sigma = 1.5 * kp.scale;
    int radius = (int)std::round(3 * sigma);

    const int NBINS = 36;
    std::vector<double> hist(NBINS, 0.0);

    for (int dy = -radius; dy <= radius; dy++) {
        int y = kp.y + dy;
        if (y <= 0 || y >= oct.height - 1) continue;
        for (int dx = -radius; dx <= radius; dx++) {
            int x = kp.x + dx;
            if (x <= 0 || x >= oct.width - 1) continue;

            double gx = img[y][x + 1] - img[y][x - 1];
            double gy = img[y + 1][x] - img[y - 1][x];
            double mag = std::sqrt(gx * gx + gy * gy);
            double angle = std::atan2(gy, gx); // (-pi, pi]

            double weight = std::exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
            int bin = (int)std::round((angle + M_PI) / (2 * M_PI) * NBINS) % NBINS;
            hist[bin] += mag * weight;
        }
    }

    int best = 0;
    for (int b = 1; b < NBINS; b++)
        if (hist[b] > hist[best]) best = b;
    kp.orientation = best * (2 * M_PI / NBINS) - M_PI;
}

// Standard 4x4 cell x 8 orientation bin descriptor, sampled in a 16x16
// window rotated to the keypoint orientation. Each keypoint's descriptor is
// independent of every other's -- embarrassingly parallel across keypoints.
void computeDescriptor(Keypoint& kp, const Octave& oct) {
    const Mat& img = oct.gaussian[kp.layer];
    const int WINDOW = 16, CELLS = 4, BINS = 8;
    double cosA = std::cos(kp.orientation), sinA = std::sin(kp.orientation);

    std::vector<double> desc(CELLS * CELLS * BINS, 0.0);

    for (int i = -WINDOW / 2; i < WINDOW / 2; i++) {
        for (int j = -WINDOW / 2; j < WINDOW / 2; j++) {
            // rotate sample offset into the keypoint's dominant orientation
            double rx = j * cosA - i * sinA;
            double ry = j * sinA + i * cosA;
            int x = kp.x + (int)std::round(rx);
            int y = kp.y + (int)std::round(ry);
            if (x <= 0 || x >= oct.width - 1 || y <= 0 || y >= oct.height - 1) continue;

            double gx = img[y][x + 1] - img[y][x - 1];
            double gy = img[y + 1][x] - img[y - 1][x];
            double mag = std::sqrt(gx * gx + gy * gy);
            double angle = std::atan2(gy, gx) - kp.orientation;
            while (angle < 0) angle += 2 * M_PI;
            while (angle >= 2 * M_PI) angle -= 2 * M_PI;

            double weight = std::exp(-(rx * rx + ry * ry) / (2 * (WINDOW / 2.0) * (WINDOW / 2.0)));

            int cellX = std::min(CELLS - 1, (i + WINDOW / 2) * CELLS / WINDOW);
            int cellY = std::min(CELLS - 1, (j + WINDOW / 2) * CELLS / WINDOW);
            int bin = std::min(BINS - 1, (int)(angle / (2 * M_PI) * BINS));

            desc[(cellY * CELLS + cellX) * BINS + bin] += mag * weight;
        }
    }

    double norm = 0.0;
    for (double v : desc) norm += v * v;
    norm = std::sqrt(norm) + 1e-12;
    for (double& v : desc) v = std::min(v / norm, 0.2); // clip large gradients (illumination robustness)

    norm = 0.0;
    for (double v : desc) norm += v * v;
    norm = std::sqrt(norm) + 1e-12;
    for (double& v : desc) v /= norm;

    kp.descriptor = desc;
}

struct FeatureSet {
    std::vector<Keypoint> keypoints;
};

FeatureSet extractFeatures(const Mat& gray, int height, int width) {
    double t0 = nowMs();
    auto octaves = buildPyramid(gray, height, width);

    double t1 = nowMs();
    auto keypoints = detectKeypoints(octaves);

    double t2 = nowMs();
    // Independent per keypoint. dynamic, unlike the pyramid loops:
    // assignOrientation samples a window of radius round(4.5 * scale), so a
    // layer-3 keypoint covers 29x29 points where a layer-1 one covers 19x19.
    // The explicit index is because OpenMP cannot take a range-for.
    const long nkp = (long)keypoints.size();
    #pragma omp parallel for schedule(dynamic, 16)
    for (long i = 0; i < nkp; i++) {
        Keypoint& kp = keypoints[i];
        assignOrientation(kp, octaves[kp.octave]);
        computeDescriptor(kp, octaves[kp.octave]);
    }
    double t3 = nowMs();

    g_tPyramid  += t1 - t0;
    g_tDetect   += t2 - t1;
    g_tDescribe += t3 - t2;

    FeatureSet fs;
    fs.keypoints = std::move(keypoints);
    return fs;
}

// image-space coordinates, for reporting (octave 0 = full resolution)
void imageCoords(const Keypoint& kp, double& ix, double& iy) {
    double scaleFactor = std::pow(2.0, kp.octave);
    ix = kp.x * scaleFactor;
    iy = kp.y * scaleFactor;
}

// ---------- feature matching ----------

struct Match {
    int idxA, idxB;
    double distance;
};

double descriptorDist(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return std::sqrt(sum);
}

// Brute-force nearest neighbor + Lowe's ratio test. Each query keypoint in A
// is matched independently against all of B -- parallelize over A.
const double RATIO_THRESH = 0.75;

// computeDescriptor() fills CELLS * CELLS * BINS = 4 * 4 * 8 entries.
const int DESC_DIM = 128;

// Every descriptor is its own ~1KB heap block, so the match loop would chase
// nB unpredictable pointers per query. One contiguous nB x 128 array (2.6 MB at
// b08) makes it a stream the prefetcher can follow. Values are untouched.
static std::vector<double> packDescriptors(const std::vector<Keypoint>& kps) {
    std::vector<double> flat(kps.size() * (size_t)DESC_DIM);
    for (size_t i = 0; i < kps.size(); i++)
        std::copy(kps[i].descriptor.begin(), kps[i].descriptor.end(),
                  flat.begin() + (long)i * DESC_DIM);
    return flat;
}

std::vector<Match> matchFeatures(const FeatureSet& a, const FeatureSet& b) {
    const long nA = (long)a.keypoints.size(), nB = (long)b.keypoints.size();
    const std::vector<double> fa = packDescriptors(a.keypoints);
    const std::vector<double> fb = packDescriptors(b.keypoints);

    // One slot per query keypoint, so no critical section is needed, and the
    // accepted matches can still be emitted in ascending i -- the serial order.
    std::vector<int> bestIdx(nA, -1);
    std::vector<double> bestD(nA, 1e18), secondD(nA, 1e18);

    // static: every i scans the whole of B, so the iterations cost the same.
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < nA; i++) {
        const double* da = &fa[i * DESC_DIM];
        double best = 1e18, second = 1e18;
        int bi = -1;
        for (long j = 0; j < nB; j++) {
            const double* db = &fb[j * DESC_DIM];
            // Same summation order as descriptorDist, and the sqrt is kept:
            // sqrt(x) < 0.75 * sqrt(y) and x < 0.5625 * y do not round the same
            // way at the boundary, so dropping it would not be bit-exact.
            double sum = 0.0;
            for (int k = 0; k < DESC_DIM; k++) {
                const double d = da[k] - db[k];
                sum += d * d;
            }
            const double dist = std::sqrt(sum);
            if (dist < best) {
                second = best;
                best = dist;
                bi = (int)j;
            } else if (dist < second) {
                second = dist;
            }
        }
        bestIdx[i] = bi;
        bestD[i] = best;
        secondD[i] = second;
    }

    std::vector<Match> matches;
    for (long i = 0; i < nA; i++)
        if (bestIdx[i] >= 0 && bestD[i] < RATIO_THRESH * secondD[i])
            matches.push_back({(int)i, bestIdx[i], bestD[i]});
    return matches;
}

// ---------- output ----------

void writeOutput(const char* path, const FeatureSet& a, const FeatureSet& b,
                  const std::vector<Match>& matches) {
    std::ofstream out(path);

    out << "KEYPOINTS_A " << a.keypoints.size() << "\n";
    for (const auto& kp : a.keypoints) {
        double ix, iy;
        imageCoords(kp, ix, iy);
        out << ix << " " << iy << " " << kp.scale << " " << kp.orientation << "\n";
    }

    out << "KEYPOINTS_B " << b.keypoints.size() << "\n";
    for (const auto& kp : b.keypoints) {
        double ix, iy;
        imageCoords(kp, ix, iy);
        out << ix << " " << iy << " " << kp.scale << " " << kp.orientation << "\n";
    }

    out << "MATCHES " << matches.size() << "\n";
    for (const auto& m : matches)
        out << m.idxA << " " << m.idxB << " " << m.distance << "\n";
}

// ---------- driver ----------

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <imageA.png> <imageB.png> <output.txt>" << std::endl;
        return -1;
    }

    int nthreads = usableCpus();
    if (const char* e = getenv("PP_THREADS")) {   // our own knob, for sweeps
        int v = atoi(e);
        if (v > 0) nthreads = v;
    }
    omp_set_num_threads(nthreads);

    double t0 = nowMs();

    png_byte *rawA = nullptr, *rawB = nullptr;
    int widthA = 0, heightA = 0, widthB = 0, heightB = 0;
    size_t rowbytesA = 0, rowbytesB = 0;

    // Two independent DEFLATE streams, so this is a genuine 2x -- unlike
    // hw1-1, whose single stream cannot be split at all.
    #pragma omp parallel sections
    {
        #pragma omp section
        rawA = read_png_raw(argv[1], &widthA, &heightA, &rowbytesA);
        #pragma omp section
        rawB = read_png_raw(argv[2], &widthB, &heightB, &rowbytesB);
    }

    double t1 = nowMs();

    Mat grayA = toGrayscale(rawA, rowbytesA, heightA, widthA);
    Mat grayB = toGrayscale(rawB, rowbytesB, heightB, widthB);
    free(rawA);
    free(rawB);

    double t2 = nowMs();

    FeatureSet featuresA = extractFeatures(grayA, heightA, widthA);
    FeatureSet featuresB = extractFeatures(grayB, heightB, widthB);

    double t3 = nowMs();

    std::vector<Match> matches = matchFeatures(featuresA, featuresB);

    double t4 = nowMs();

    writeOutput(argv[3], featuresA, featuresB, matches);

    double t5 = nowMs();

    if (getenv("PP_TIMING")) {
        size_t nA = featuresA.keypoints.size(), nB = featuresB.keypoints.size();
        fprintf(stderr,
                "threads %2d | %dx%d + %dx%d | kpA %zu kpB %zu match %zu\n"
                "  read %7.1f | gray %6.1f | pyramid %8.1f | detect %7.1f | "
                "orient+desc %8.1f | match %8.1f | write %6.1f | total %8.1f  (ms)\n",
                omp_get_max_threads(), widthA, heightA, widthB, heightB,
                nA, nB, matches.size(),
                t1 - t0, t2 - t1, g_tPyramid, g_tDetect, g_tDescribe,
                t4 - t3, t5 - t4, t5 - t0);
    }

    return 0;
}
