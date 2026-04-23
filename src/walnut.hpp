#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace walnut {

namespace fs = std::filesystem;

struct SliceArtifacts {
    cv::Mat gray;
    cv::Mat filtered;
    cv::Mat nut;
    cv::Mat shell;
    cv::Mat kernel;
    cv::Mat septa;
    cv::Mat labels;
};

struct PairInfo {
    std::string id;
    fs::path image;
    fs::path gt;
};

inline std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

inline bool is_tiff(const fs::path& path) {
    const auto ext = lower(path.extension().string());
    return ext == ".tif" || ext == ".tiff";
}

inline std::string strip_suffix(std::string stem) {
    for (const std::string suffix : {"_labels_gt", "_labels_ids", "_labels_pred", "_labels_init", "_gt", "_pred"}) {
        if (stem.size() >= suffix.size() && stem.compare(stem.size() - suffix.size(), suffix.size(), suffix) == 0) {
            stem.resize(stem.size() - suffix.size());
            return stem;
        }
    }
    return stem;
}

inline fs::path arg_value(int argc, char** argv, const std::string& key) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (key == argv[i]) {
            return argv[i + 1];
        }
    }
    return {};
}

inline bool has_arg(int argc, char** argv, const std::string& key) {
    for (int i = 1; i < argc; ++i) {
        if (key == argv[i]) {
            return true;
        }
    }
    return false;
}

inline cv::Mat read_gray8(const fs::path& path) {
    cv::Mat input = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
    if (input.empty()) {
        throw std::runtime_error("Cannot read image: " + path.string());
    }
    if (input.channels() == 3) {
        cv::cvtColor(input, input, cv::COLOR_BGR2GRAY);
    } else if (input.channels() == 4) {
        cv::cvtColor(input, input, cv::COLOR_BGRA2GRAY);
    }
    cv::Mat float_img;
    input.convertTo(float_img, CV_32F);
    double lo = 0.0, hi = 0.0;
    cv::minMaxLoc(float_img, &lo, &hi);
    if (hi <= lo) {
        return cv::Mat::zeros(input.size(), CV_8U);
    }
    float_img = (float_img - static_cast<float>(lo)) * static_cast<float>(255.0 / (hi - lo));
    cv::Mat gray8;
    float_img.convertTo(gray8, CV_8U);
    return gray8;
}

inline cv::Mat read_labels8(const fs::path& path) {
    cv::Mat labels = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
    if (labels.empty()) {
        throw std::runtime_error("Cannot read labels: " + path.string());
    }
    if (labels.channels() == 3) {
        cv::cvtColor(labels, labels, cv::COLOR_BGR2GRAY);
    } else if (labels.channels() == 4) {
        cv::cvtColor(labels, labels, cv::COLOR_BGRA2GRAY);
    }
    if (labels.depth() != CV_8U) {
        cv::Mat converted;
        labels.convertTo(converted, CV_8U);
        labels = converted;
    }
    return labels;
}

inline cv::Mat fill_holes(const cv::Mat& mask) {
    cv::Mat bordered;
    cv::copyMakeBorder(mask, bordered, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
    cv::Mat flooded = bordered.clone();
    cv::floodFill(flooded, {0, 0}, 255);
    cv::Mat holes;
    cv::bitwise_not(flooded(cv::Rect(1, 1, mask.cols, mask.rows)), holes);
    return mask | holes;
}

inline cv::Mat largest_cc(const cv::Mat& mask) {
    cv::Mat labels, stats, centroids;
    const int count = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);
    int best = 0;
    for (int i = 1; i < count; ++i) {
        if (!best || stats.at<int>(i, cv::CC_STAT_AREA) > stats.at<int>(best, cv::CC_STAT_AREA)) {
            best = i;
        }
    }
    cv::Mat out = cv::Mat::zeros(mask.size(), CV_8U);
    if (best) {
        out.setTo(255, labels == best);
    }
    return out;
}

inline cv::Mat remove_small(const cv::Mat& mask, int min_area) {
    cv::Mat labels, stats, centroids;
    const int count = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);
    cv::Mat out = cv::Mat::zeros(mask.size(), CV_8U);
    for (int i = 1; i < count; ++i) {
        if (stats.at<int>(i, cv::CC_STAT_AREA) >= min_area) {
            out.setTo(255, labels == i);
        }
    }
    return out;
}

inline double eq_diameter(const cv::Mat& mask) {
    const double area = static_cast<double>(cv::countNonZero(mask));
    return area > 0.0 ? 2.0 * std::sqrt(area / CV_PI) : 0.0;
}

inline int odd(int value, int lo = 3) {
    value = std::max(value, lo);
    return value % 2 ? value : value + 1;
}

inline int scaled(double diameter, double factor, int lo) {
    return odd(static_cast<int>(std::lround(diameter * factor)), lo);
}

inline int masked_otsu(const cv::Mat& img, const cv::Mat& mask) {
    std::array<int, 256> hist{};
    long long pixels = 0;
    double total = 0.0;
    for (int y = 0; y < img.rows; ++y) {
        const auto* ir = img.ptr<uchar>(y);
        const auto* mr = mask.ptr<uchar>(y);
        for (int x = 0; x < img.cols; ++x) {
            if (mr[x]) {
                ++hist[ir[x]];
                ++pixels;
                total += ir[x];
            }
        }
    }
    long long w0 = 0;
    double s0 = 0.0;
    double best = -1.0;
    int best_t = 0;
    for (int t = 0; t < 256; ++t) {
        w0 += hist[t];
        if (!w0 || w0 == pixels) {
            continue;
        }
        const long long w1 = pixels - w0;
        s0 += t * hist[t];
        const double m0 = s0 / w0;
        const double m1 = (total - s0) / w1;
        const double score = static_cast<double>(w0) * static_cast<double>(w1) * (m0 - m1) * (m0 - m1);
        if (score > best) {
            best = score;
            best_t = t;
        }
    }
    return best_t;
}

inline double masked_mean_kstd(const cv::Mat& img, const cv::Mat& mask, double k) {
    cv::Scalar mu, sigma;
    cv::meanStdDev(img, mu, sigma, mask);
    return mu[0] + k * sigma[0];
}

inline cv::Mat band_mask(const cv::Mat& dist, float lo, float hi, const cv::Mat& support) {
    cv::Mat out = cv::Mat::zeros(dist.size(), CV_8U);
    for (int y = 0; y < dist.rows; ++y) {
        const auto* dr = dist.ptr<float>(y);
        const auto* sr = support.ptr<uchar>(y);
        auto* orow = out.ptr<uchar>(y);
        for (int x = 0; x < dist.cols; ++x) {
            if (sr[x] && dr[x] >= lo && dr[x] <= hi) {
                orow[x] = 255;
            }
        }
    }
    return out;
}

inline cv::Mat upper_band(const cv::Mat& dist, float lo, const cv::Mat& support) {
    return band_mask(dist, lo, std::numeric_limits<float>::max(), support);
}

inline cv::Mat line_kernel(int n, int angle) {
    n = odd(n);
    cv::Mat kernel = cv::Mat::zeros(n, n, CV_8U);
    const int c = n / 2;
    cv::Point p1;
    cv::Point p2;
    if (angle == 0) {
        p1 = {0, c};
        p2 = {n - 1, c};
    } else if (angle == 45) {
        p1 = {0, n - 1};
        p2 = {n - 1, 0};
    } else if (angle == 90) {
        p1 = {c, 0};
        p2 = {c, n - 1};
    } else {
        p1 = {0, 0};
        p2 = {n - 1, n - 1};
    }
    cv::line(kernel, p1, p2, 1, 1);
    return kernel;
}

inline cv::Mat directional_tophat(const cv::Mat& img, int n) {
    cv::Mat out = cv::Mat::zeros(img.size(), CV_8U);
    cv::Mat current;
    for (const int angle : {0, 45, 90, 135}) {
        cv::morphologyEx(img, current, cv::MORPH_TOPHAT, line_kernel(n, angle));
        cv::max(out, current, out);
    }
    cv::morphologyEx(img, current, cv::MORPH_TOPHAT,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, {n, n}));
    cv::max(out, current, out);
    return out;
}

inline cv::Mat gradient_mag(const cv::Mat& img) {
    cv::Mat gx, gy, mag;
    cv::Sobel(img, gx, CV_32F, 1, 0, 3);
    cv::Sobel(img, gy, CV_32F, 0, 1, 3);
    cv::magnitude(gx, gy, mag);
    double lo = 0.0, hi = 0.0;
    cv::minMaxLoc(mag, &lo, &hi);
    cv::Mat out = cv::Mat::zeros(img.size(), CV_8U);
    if (hi > lo) {
        mag = (mag - static_cast<float>(lo)) * static_cast<float>(255.0 / (hi - lo));
        mag.convertTo(out, CV_8U);
    }
    return out;
}

inline cv::Mat color_labels(const cv::Mat& labels) {
    cv::Mat vis(labels.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    vis.setTo(cv::Scalar(0, 0, 255), labels == 1);
    vis.setTo(cv::Scalar(0, 255, 0), labels == 2);
    vis.setTo(cv::Scalar(255, 0, 0), labels == 3);
    return vis;
}

inline cv::Mat overlay_labels(const cv::Mat& gray, const cv::Mat& labels) {
    cv::Mat out;
    cv::cvtColor(gray, out, cv::COLOR_GRAY2BGR);
    auto blend = [&](int id, const cv::Vec3b& color, double alpha) {
        for (int y = 0; y < out.rows; ++y) {
            auto* row = out.ptr<cv::Vec3b>(y);
            const auto* lrow = labels.ptr<uchar>(y);
            for (int x = 0; x < out.cols; ++x) {
                if (lrow[x] == id) {
                    for (int k = 0; k < 3; ++k) {
                        row[x][k] = cv::saturate_cast<uchar>((1.0 - alpha) * row[x][k] + alpha * color[k]);
                    }
                }
            }
        }
    };
    blend(1, {0, 0, 255}, 0.55);
    blend(2, {0, 255, 0}, 0.45);
    blend(3, {255, 0, 0}, 0.60);
    return out;
}

inline void save_artifacts(const fs::path& out_dir, const SliceArtifacts& art) {
    fs::create_directories(out_dir);
    cv::imwrite((out_dir / "01_gray.tiff").string(), art.gray);
    cv::imwrite((out_dir / "02_filtered.tiff").string(), art.filtered);
    cv::imwrite((out_dir / "03_nut_mask.tiff").string(), art.nut);
    cv::imwrite((out_dir / "04_shell_mask.tiff").string(), art.shell);
    cv::imwrite((out_dir / "05_kernel_mask.tiff").string(), art.kernel);
    cv::imwrite((out_dir / "06_septa_mask.tiff").string(), art.septa);
    cv::imwrite((out_dir / "07_labels_vis.tiff").string(), color_labels(art.labels));
    cv::imwrite((out_dir / "08_overlay.tiff").string(), overlay_labels(art.gray, art.labels));
    cv::imwrite((out_dir / "09_labels_ids.tiff").string(), art.labels);
}

inline std::vector<PairInfo> collect_pairs(const fs::path& dataset_root, const fs::path& gt_root) {
    std::vector<PairInfo> pairs;
    for (const auto& entry : fs::directory_iterator(gt_root)) {
        if (!entry.is_regular_file() || !is_tiff(entry.path())) {
            continue;
        }
        const std::string id = strip_suffix(entry.path().stem().string());
        const fs::path image = dataset_root / (id + ".tiff");
        if (!fs::exists(image)) {
            throw std::runtime_error("Missing slice for GT: " + image.string());
        }
        pairs.push_back({id, image, entry.path()});
    }
    std::sort(pairs.begin(), pairs.end(), [](const PairInfo& a, const PairInfo& b) { return a.id < b.id; });
    return pairs;
}

inline std::string format4(double value) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(4) << value;
    return out.str();
}

inline cv::Mat mask_of(const cv::Mat& labels, int cls) {
    cv::Mat mask;
    cv::compare(labels, cls, mask, cv::CMP_EQ);
    return mask;
}

}  // namespace walnut
