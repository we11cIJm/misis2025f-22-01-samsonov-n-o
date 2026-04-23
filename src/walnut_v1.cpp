#include "walnut.hpp"

#include <iostream>

namespace {

cv::Mat filter_septa(const cv::Mat& candidates,
                     const cv::Mat& kernel_band,
                     const cv::Mat& dist,
                     const cv::Mat& grad,
                     const cv::Mat& bright,
                     double shell_limit,
                     double grad_limit,
                     double bright_limit,
                     int min_area,
                     int max_area) {
    cv::Mat labels, stats, centroids;
    const int count = cv::connectedComponentsWithStats(candidates, labels, stats, centroids, 8, CV_32S);
    cv::Mat out = cv::Mat::zeros(candidates.size(), CV_8U);
    for (int i = 1; i < count; ++i) {
        const int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area < min_area || area > max_area) {
            continue;
        }
        cv::Mat component = labels == i;
        component.convertTo(component, CV_8U);
        std::vector<cv::Point> points;
        cv::findNonZero(component, points);
        if (points.size() < 3) {
            continue;
        }
        const auto box = cv::minAreaRect(points);
        const double major = std::max(box.size.width, box.size.height);
        const double minor = std::max(1.0f, std::min(box.size.width, box.size.height));
        // if (major / minor < 2.0) {
        if (major / minor < 1.6) {
            continue;
        }
        if (cv::mean(dist, component)[0] <= shell_limit) {
            continue;
        }
        if (cv::mean(grad, component)[0] < grad_limit && cv::mean(bright, component)[0] < bright_limit) {
            continue;
        }
        cv::Mat overlap;
        cv::bitwise_and(component, kernel_band, overlap);
        // if (static_cast<double>(cv::countNonZero(overlap)) / area > 0.95) {
        if (static_cast<double>(cv::countNonZero(overlap)) / area > 0.98) {
            continue;
        }
        out.setTo(255, component);
    }
    return out;
}

walnut::SliceArtifacts segment_v1(const walnut::fs::path& path) {
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
    const int ridge_len = scaled(d, 0.033, 7);
    const int nut_area = cv::countNonZero(out.nut);
    const int min_shell = std::max(40, static_cast<int>(nut_area * 0.0012));
    const int min_kernel = std::max(120, static_cast<int>(nut_area * 0.0035));
    const int min_septa = std::max(8, static_cast<int>(nut_area * 0.00025));
    const int max_septa = std::max(160, static_cast<int>(nut_area * 0.0200));

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

    // const cv::Mat deep = upper_band(dist, static_cast<float>(shell_hi + std::max(1, static_cast<int>(std::lround(d * 0.004)))), inner);
    const cv::Mat deep = upper_band(dist, static_cast<float>(shell_hi + 1), inner);
    cv::Mat local_mean;
    cv::blur(out.filtered, local_mean, {kernel_block, kernel_block});
    cv::Mat bright;
    cv::subtract(out.filtered, local_mean, bright);
    const cv::Mat ridge = directional_tophat(out.filtered, ridge_len);
    const cv::Mat grad = gradient_mag(out.filtered);

    // const int ridge_thr = std::max(masked_otsu(ridge, deep),
    //                            static_cast<int>(std::lround(masked_mean_kstd(ridge, deep, 0.10))));
    const int ridge_thr = std::max(masked_otsu(ridge, deep),
                               static_cast<int>(std::lround(masked_mean_kstd(ridge, deep, 0.00))));
    const int grad_thr = std::max(masked_otsu(grad, deep),
                                  static_cast<int>(std::lround(masked_mean_kstd(grad, deep, -0.10))));
    const int bright_thr = std::max(1, static_cast<int>(std::lround(masked_mean_kstd(bright, deep, -0.10))));

    cv::Mat ridge_mask;
    cv::Mat grad_mask;
    cv::Mat bright_mask;
    cv::threshold(ridge, ridge_mask, ridge_thr, 255, cv::THRESH_BINARY);
    cv::threshold(grad, grad_mask, grad_thr, 255, cv::THRESH_BINARY);
    cv::threshold(bright, bright_mask, bright_thr, 255, cv::THRESH_BINARY);
    cv::Mat candidates;
    candidates = ridge_mask.clone();
    cv::bitwise_and(candidates, deep, candidates);
    cv::morphologyEx(candidates, candidates, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3}));
    cv::morphologyEx(candidates, candidates, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3}));

    cv::Mat kernel_dilated;
    cv::Mat kernel_eroded;
    cv::dilate(out.kernel, kernel_dilated, cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3}));
    cv::erode(out.kernel, kernel_eroded, cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3}));
    cv::Mat kernel_band;
    cv::bitwise_and(kernel_dilated, ~kernel_eroded, kernel_band);

    // out.septa = filter_septa(candidates,
    //                          kernel_band,
    //                          dist,
    //                          grad,
    //                          bright,
    //                          shell_hi + 0.5,
    //                          grad_thr,
    //                          bright_thr,
    //                          min_septa,
    //                          max_septa);
    // cv::bitwise_and(out.septa, inner, out.septa);
    // out.septa = remove_small(out.septa, min_septa);
    out.septa = filter_septa(candidates,
                         kernel_band,
                         dist,
                         grad,
                         bright,
                         shell_hi + 0.5,
                         grad_thr,
                         bright_thr,
                         min_septa,
                         max_septa);
    cv::morphologyEx(out.septa, out.septa, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3}));
    cv::dilate(out.septa, out.septa,
               cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3}));
    cv::bitwise_and(out.septa, deep, out.septa);
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
        walnut::save_artifacts(results_root / pair.id, segment_v1(pair.image));
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
            walnut::save_artifacts(output, segment_v1(input));
            return 0;
        }
        if (!dataset.empty() && !gt_dir.empty() && !results.empty()) {
            return run_batch(dataset, gt_dir, results);
        }

        std::cerr << "Usage:\n"
                  << "  walnut_v1 --input <slice.tiff> --output <dir>\n"
                  << "  walnut_v1 --dataset <Reconstructions> --gt-dir <gt_final> --results <results/v1>\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
