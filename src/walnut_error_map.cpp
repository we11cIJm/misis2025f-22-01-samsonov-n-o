#include "walnut.hpp"

#include <iostream>

namespace {

struct ErrorStats {
    std::string pair;
    std::string cls;
    int tp = 0;
    int fp = 0;
    int fn = 0;
};

cv::Mat overlay(const cv::Mat& gray, const cv::Mat& error_rgb, double alpha = 0.55) {
    cv::Mat base;
    cv::cvtColor(gray, base, cv::COLOR_GRAY2BGR);
    cv::Mat out = base.clone();
    for (int y = 0; y < out.rows; ++y) {
        auto* orow = out.ptr<cv::Vec3b>(y);
        const auto* erow = error_rgb.ptr<cv::Vec3b>(y);
        for (int x = 0; x < out.cols; ++x) {
            if (erow[x] != cv::Vec3b{0, 0, 0}) {
                for (int k = 0; k < 3; ++k) {
                    orow[x][k] = cv::saturate_cast<uchar>((1.0 - alpha) * orow[x][k] + alpha * erow[x][k]);
                }
            }
        }
    }
    return out;
}

std::pair<cv::Mat, ErrorStats> class_error(const std::string& pair_name,
                                           const cv::Mat& pred,
                                           const cv::Mat& gt,
                                           int cls,
                                           const std::string& name) {
    const cv::Mat pred_mask = walnut::mask_of(pred, cls);
    const cv::Mat gt_mask = walnut::mask_of(gt, cls);

    cv::Mat tp_mask;
    cv::Mat fp_mask;
    cv::Mat fn_mask;
    cv::bitwise_and(pred_mask, gt_mask, tp_mask);
    cv::bitwise_and(pred_mask, ~gt_mask, fp_mask);
    cv::bitwise_and(~pred_mask, gt_mask, fn_mask);

    cv::Mat error(pred.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    error.setTo(cv::Scalar(0, 255, 0), tp_mask);
    error.setTo(cv::Scalar(0, 0, 255), fp_mask);
    error.setTo(cv::Scalar(255, 0, 0), fn_mask);

    ErrorStats stats;
    stats.pair = pair_name;
    stats.cls = name;
    stats.tp = cv::countNonZero(tp_mask);
    stats.fp = cv::countNonZero(fp_mask);
    stats.fn = cv::countNonZero(fn_mask);
    return {error, stats};
}

void save_pair_outputs(const walnut::fs::path& pred_file,
                       const walnut::fs::path& gt_file,
                       const walnut::fs::path& gray_file,
                       const walnut::fs::path& out_dir,
                       std::vector<ErrorStats>& rows) {
    const cv::Mat pred = walnut::read_labels8(pred_file);
    const cv::Mat gt = walnut::read_labels8(gt_file);
    const cv::Mat gray = walnut::read_gray8(gray_file);
    const std::string pair_name = walnut::strip_suffix(gt_file.stem().string());

    walnut::fs::create_directories(out_dir);
    cv::Mat all_error(pred.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    for (const auto& meta : std::vector<std::pair<int, std::string>>{{1, "shell"}, {2, "kernel"}, {3, "septa"}}) {
        auto [err, stats] = class_error(pair_name, pred, gt, meta.first, meta.second);
        cv::max(all_error, err, all_error);
        cv::imwrite((out_dir / ("error_" + meta.second + ".png")).string(), err);
        cv::imwrite((out_dir / ("error_" + meta.second + "_overlay.png")).string(), overlay(gray, err));
        rows.push_back(stats);
    }
    cv::imwrite((out_dir / "error_all.png").string(), all_error);
    cv::imwrite((out_dir / "error_all_overlay.png").string(), overlay(gray, all_error));

    std::ofstream summary(out_dir / "error_summary.csv");
    summary << "pair,class,tp,fp,fn\n";
    for (const auto& row : rows) {
        if (row.pair == pair_name) {
            summary << row.pair << ',' << row.cls << ',' << row.tp << ',' << row.fp << ',' << row.fn << '\n';
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const walnut::fs::path pred = walnut::arg_value(argc, argv, "--pred");
        const walnut::fs::path gt = walnut::arg_value(argc, argv, "--gt");
        const walnut::fs::path gray = walnut::arg_value(argc, argv, "--gray");
        const walnut::fs::path out = walnut::arg_value(argc, argv, "--out");

        const walnut::fs::path pred_root = walnut::arg_value(argc, argv, "--pred-root");
        const walnut::fs::path gt_root = walnut::arg_value(argc, argv, "--gt-root");
        const walnut::fs::path out_root = walnut::arg_value(argc, argv, "--out-root");

        std::vector<ErrorStats> rows;
        if (!pred.empty() && !gt.empty() && !gray.empty() && !out.empty()) {
            save_pair_outputs(pred, gt, gray, out, rows);
        } else if (!pred_root.empty() && !gt_root.empty() && !out_root.empty()) {
            for (const auto& gt_entry : std::filesystem::directory_iterator(gt_root)) {
                if (!gt_entry.is_regular_file() || !walnut::is_tiff(gt_entry.path())) {
                    continue;
                }
                const std::string id = walnut::strip_suffix(gt_entry.path().stem().string());
                save_pair_outputs(pred_root / id / "09_labels_ids.tiff",
                                  gt_entry.path(),
                                  pred_root / id / "01_gray.tiff",
                                  out_root / id,
                                  rows);
            }
            std::ofstream summary(out_root / "error_summary.csv");
            summary << "pair,class,tp,fp,fn\n";
            for (const auto& row : rows) {
                summary << row.pair << ',' << row.cls << ',' << row.tp << ',' << row.fp << ',' << row.fn << '\n';
            }
        } else {
            std::cerr << "Usage:\n"
                      << "  walnut_error_map --pred <pred.tiff> --gt <gt.tiff> --gray <gray.tiff> --out <dir>\n"
                      << "  walnut_error_map --pred-root <results/v1> --gt-root <gt_final> --out-root <results/v1/error_maps>\n";
            return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
