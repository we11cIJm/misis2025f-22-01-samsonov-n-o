#include "walnut.hpp"

#include <iostream>

namespace {

walnut::SliceArtifacts segment_v2(const walnut::fs::path& path) {
    using namespace walnut;

    SliceArtifacts out;
    out.gray = read_gray8(path);
    cv::bilateralFilter(out.gray, out.filtered, 7, 40, 9);

    double otsu = cv::threshold(out.filtered, out.nut, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::threshold(out.filtered, out.nut, 0.72 * otsu, 255, cv::THRESH_BINARY);
    out.nut = fill_holes(largest_cc(out.nut));

    const double d = eq_diameter(out.nut);
    const int nut_close = scaled(d, 0.060, 11);
    const int nut_open = scaled(d, 0.015, 3);
    const int shell_lo = std::max(2, static_cast<int>(std::lround(d * 0.006)));
    const int shell_hi = std::max(4, static_cast<int>(std::lround(d * 0.048)));
    const int shell_close = scaled(d, 0.016, 5);
    const int kernel_block = scaled(d, 0.120, 21);
    const int kernel_close = scaled(d, 0.020, 5);
    const int ridge_len = scaled(d, 0.030, 7);
    const int nut_area = cv::countNonZero(out.nut);
    const int min_shell = std::max(40, static_cast<int>(nut_area * 0.0012));
    const int min_kernel = std::max(120, static_cast<int>(nut_area * 0.0035));
    const int min_septa = std::max(12, static_cast<int>(nut_area * 0.00030));

    cv::morphologyEx(out.nut, out.nut, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, {nut_close, nut_close}));
    cv::morphologyEx(out.nut, out.nut, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, {nut_open, nut_open}));
    out.nut = fill_holes(largest_cc(out.nut));

    cv::Mat dist;
    cv::distanceTransform(out.nut, dist, cv::DIST_L2, 3);

    const cv::Mat shell_band = band_mask(dist, static_cast<float>(shell_lo), static_cast<float>(shell_hi), out.nut);
    const int shell_thr = std::max(masked_otsu(out.filtered, shell_band),
                                   static_cast<int>(std::lround(masked_mean_kstd(out.filtered, shell_band, 0.0))));
    cv::threshold(out.filtered, out.shell, shell_thr, 255, cv::THRESH_BINARY);
    cv::bitwise_and(out.shell, shell_band, out.shell);
    cv::morphologyEx(out.shell, out.shell, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, {shell_close, shell_close}));
    cv::morphologyEx(out.shell, out.shell, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3}));
    out.shell = remove_small(out.shell, min_shell);

    cv::Mat inner;
    cv::Mat shell_dilated;
    cv::dilate(out.shell, shell_dilated, cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3}));
    cv::bitwise_and(out.nut, ~shell_dilated, inner);

    cv::adaptiveThreshold(out.filtered, out.kernel, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, kernel_block, -2);
    cv::bitwise_and(out.kernel, inner, out.kernel);
    cv::morphologyEx(out.kernel, out.kernel, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3}));
    cv::morphologyEx(out.kernel, out.kernel, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, {kernel_close, kernel_close}));
    out.kernel = remove_small(out.kernel, min_kernel);

    const cv::Mat deep = upper_band(dist, static_cast<float>(shell_hi + 1), inner);
    const cv::Mat ridge = directional_tophat(out.filtered, ridge_len);
    const int ridge_thr = std::max(masked_otsu(ridge, deep),
                                   static_cast<int>(std::lround(masked_mean_kstd(ridge, deep, 0.25))));
    cv::threshold(ridge, out.septa, ridge_thr, 255, cv::THRESH_BINARY);
    cv::bitwise_and(out.septa, deep, out.septa);
    cv::morphologyEx(out.septa, out.septa, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3}));
    cv::dilate(out.septa, out.septa, cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3}));
    out.septa = remove_small(out.septa, min_septa);

    cv::Mat septa_dilated;
    cv::dilate(out.septa, septa_dilated, cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3}));
    cv::bitwise_and(out.kernel, ~septa_dilated, out.kernel);
    cv::bitwise_and(out.kernel, inner, out.kernel);
    out.kernel = remove_small(out.kernel, min_kernel);

    out.labels = cv::Mat::zeros(out.gray.size(), CV_8U);
    out.labels.setTo(1, out.shell);
    out.labels.setTo(2, out.kernel);
    out.labels.setTo(3, out.septa);
    return out;
}

int run_batch(const walnut::fs::path& dataset_root, const walnut::fs::path& gt_root, const walnut::fs::path& results_root) {
    for (const auto& pair : walnut::collect_pairs(dataset_root, gt_root)) {
        walnut::save_artifacts(results_root / pair.id, segment_v2(pair.image));
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const walnut::fs::path input = walnut::arg_value(argc, argv, "--input");
        const walnut::fs::path output = walnut::arg_value(argc, argv, "--output");
        const walnut::fs::path dataset = walnut::arg_value(argc, argv, "--dataset");
        const walnut::fs::path gt_dir = walnut::arg_value(argc, argv, "--gt-dir");
        const walnut::fs::path results = walnut::arg_value(argc, argv, "--results");

        if (!input.empty() && !output.empty()) {
            walnut::save_artifacts(output, segment_v2(input));
            return 0;
        }
        if (!dataset.empty() && !gt_dir.empty() && !results.empty()) {
            return run_batch(dataset, gt_dir, results);
        }

        std::cerr << "Usage:\n"
                  << "  walnut_v2 --input <slice.tiff> --output <dir>\n"
                  << "  walnut_v2 --dataset <Reconstructions> --gt-dir <gt_final> --results <results/v2>\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
