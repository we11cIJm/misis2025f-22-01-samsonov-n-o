#include "walnut.hpp"

#include <iostream>
#include <map>

namespace {

struct MetricRow {
    std::string kind;
    std::string pair;
    int cls = 0;
    std::string name;
    long long pred_area = 0;
    long long gt_area = 0;
    long long inter = 0;
    long long uni = 0;
    double iou = 0.0;
    double dice = 0.0;
    double precision = 0.0;
    double recall = 0.0;
    double boundary_f1 = 0.0;
    bool present = false;
};

cv::Mat boundary_of(const cv::Mat& mask) {
    cv::Mat eroded;
    cv::Mat boundary;
    cv::erode(mask, eroded, cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3}));
    cv::subtract(mask, eroded, boundary);
    return boundary;
}

MetricRow eval_class(const cv::Mat& pred, const cv::Mat& gt, int cls, const std::string& pair_name, int tol) {
    MetricRow row;
    row.kind = "pair";
    row.pair = pair_name;
    row.cls = cls;
    row.name = cls == 1 ? "shell" : cls == 2 ? "kernel" : "septa";

    const cv::Mat pred_mask = walnut::mask_of(pred, cls);
    const cv::Mat gt_mask = walnut::mask_of(gt, cls);
    row.pred_area = cv::countNonZero(pred_mask);
    row.gt_area = cv::countNonZero(gt_mask);
    row.present = row.pred_area || row.gt_area;

    cv::Mat inter;
    cv::Mat uni;
    cv::bitwise_and(pred_mask, gt_mask, inter);
    cv::bitwise_or(pred_mask, gt_mask, uni);
    row.inter = cv::countNonZero(inter);
    row.uni = cv::countNonZero(uni);
    if (row.uni) {
        row.iou = static_cast<double>(row.inter) / row.uni;
    }
    if (row.pred_area + row.gt_area) {
        row.dice = 2.0 * row.inter / (row.pred_area + row.gt_area);
    }
    if (row.pred_area) {
        row.precision = static_cast<double>(row.inter) / row.pred_area;
    }
    if (row.gt_area) {
        row.recall = static_cast<double>(row.inter) / row.gt_area;
    }

    const cv::Mat pred_boundary = boundary_of(pred_mask);
    const cv::Mat gt_boundary = boundary_of(gt_mask);
    const int pred_count = cv::countNonZero(pred_boundary);
    const int gt_count = cv::countNonZero(gt_boundary);
    if (!pred_count && !gt_count) {
        row.boundary_f1 = 1.0;
        return row;
    }
    if (!pred_count || !gt_count) {
        return row;
    }

    const cv::Mat disk = cv::getStructuringElement(cv::MORPH_ELLIPSE, {2 * tol + 1, 2 * tol + 1});
    cv::Mat pred_dilated;
    cv::Mat gt_dilated;
    cv::dilate(pred_boundary, pred_dilated, disk);
    cv::dilate(gt_boundary, gt_dilated, disk);

    cv::Mat pred_hits;
    cv::Mat gt_hits;
    cv::bitwise_and(pred_boundary, gt_dilated, pred_hits);
    cv::bitwise_and(gt_boundary, pred_dilated, gt_hits);

    const double bp = static_cast<double>(cv::countNonZero(pred_hits)) / pred_count;
    const double br = static_cast<double>(cv::countNonZero(gt_hits)) / gt_count;
    if (bp + br) {
        row.boundary_f1 = 2.0 * bp * br / (bp + br);
    }
    return row;
}

std::vector<MetricRow> eval_pair(const walnut::fs::path& pred_file, const walnut::fs::path& gt_file, int tol) {
    const cv::Mat pred = walnut::read_labels8(pred_file);
    const cv::Mat gt = walnut::read_labels8(gt_file);
    if (pred.size() != gt.size()) {
        throw std::runtime_error("Prediction and GT size mismatch: " + pred_file.string());
    }
    const std::string pair_name = walnut::strip_suffix(gt_file.stem().string());
    return {
        eval_class(pred, gt, 1, pair_name, tol),
        eval_class(pred, gt, 2, pair_name, tol),
        eval_class(pred, gt, 3, pair_name, tol),
    };
}

double mean_metric(const std::vector<MetricRow>& rows, int cls, double MetricRow::*field) {
    double sum = 0.0;
    int count = 0;
    for (const auto& row : rows) {
        if (row.kind == "pair" && row.cls == cls && row.present) {
            sum += row.*field;
            ++count;
        }
    }
    return count ? sum / count : 0.0;
}

double macro_metric(const std::vector<MetricRow>& rows, double MetricRow::*field) {
    double sum = 0.0;
    int count = 0;
    for (const auto& row : rows) {
        if (row.kind == "pair" && row.present) {
            sum += row.*field;
            ++count;
        }
    }
    return count ? sum / count : 0.0;
}

std::vector<MetricRow> collect_rows(const walnut::fs::path& pred_root,
                                    const walnut::fs::path& gt_root,
                                    int tol) {
    std::vector<MetricRow> rows;
    for (const auto& gt_entry : std::filesystem::directory_iterator(gt_root)) {
        if (!gt_entry.is_regular_file() || !walnut::is_tiff(gt_entry.path())) {
            continue;
        }
        const std::string id = walnut::strip_suffix(gt_entry.path().stem().string());
        const walnut::fs::path pred_file = pred_root / id / "09_labels_ids.tiff";
        if (!walnut::fs::exists(pred_file)) {
            throw std::runtime_error("Missing prediction for " + id + ": " + pred_file.string());
        }
        auto pair_rows = eval_pair(pred_file, gt_entry.path(), tol);
        rows.insert(rows.end(), pair_rows.begin(), pair_rows.end());
    }

    for (const auto& meta : std::vector<std::pair<int, std::string>>{{1, "shell"}, {2, "kernel"}, {3, "septa"}}) {
        MetricRow row;
        row.kind = "mean";
        row.pair = "__mean__";
        row.cls = meta.first;
        row.name = meta.second;
        row.iou = mean_metric(rows, meta.first, &MetricRow::iou);
        row.dice = mean_metric(rows, meta.first, &MetricRow::dice);
        row.precision = mean_metric(rows, meta.first, &MetricRow::precision);
        row.recall = mean_metric(rows, meta.first, &MetricRow::recall);
        row.boundary_f1 = mean_metric(rows, meta.first, &MetricRow::boundary_f1);
        rows.push_back(row);
    }

    MetricRow macro;
    macro.kind = "macro";
    macro.pair = "__macro__";
    macro.cls = 0;
    macro.name = "all";
    macro.iou = macro_metric(rows, &MetricRow::iou);
    macro.dice = macro_metric(rows, &MetricRow::dice);
    macro.precision = macro_metric(rows, &MetricRow::precision);
    macro.recall = macro_metric(rows, &MetricRow::recall);
    macro.boundary_f1 = macro_metric(rows, &MetricRow::boundary_f1);
    rows.push_back(macro);
    return rows;
}

void write_csv(const walnut::fs::path& csv_path, const std::vector<MetricRow>& rows) {
    std::ofstream out(csv_path);
    out << "kind,pair,class,pred_area,gt_area,iou,dice,precision,recall,boundary_f1\n";
    for (const auto& row : rows) {
        out << row.kind << ','
            << row.pair << ','
            << row.name << ','
            << row.pred_area << ','
            << row.gt_area << ','
            << row.iou << ','
            << row.dice << ','
            << row.precision << ','
            << row.recall << ','
            << row.boundary_f1 << '\n';
    }
}

void print_summary(const std::vector<MetricRow>& rows) {
    for (const auto& row : rows) {
        if (row.kind == "mean") {
            std::cout << row.name
                      << ": IoU=" << walnut::format4(row.iou)
                      << ", Dice=" << walnut::format4(row.dice)
                      << ", BoundaryF1=" << walnut::format4(row.boundary_f1) << '\n';
        } else if (row.kind == "macro") {
            std::cout << "macro"
                      << ": IoU=" << walnut::format4(row.iou)
                      << ", Dice=" << walnut::format4(row.dice)
                      << ", BoundaryF1=" << walnut::format4(row.boundary_f1) << '\n';
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const walnut::fs::path pred_root = walnut::arg_value(argc, argv, "--pred-root");
        const walnut::fs::path gt_root = walnut::arg_value(argc, argv, "--gt-root");
        const walnut::fs::path csv_path = walnut::arg_value(argc, argv, "--csv");
        const int tol = walnut::has_arg(argc, argv, "--tol")
                            ? std::stoi(walnut::arg_value(argc, argv, "--tol").string())
                            : 2;

        if (pred_root.empty() || gt_root.empty() || csv_path.empty()) {
            std::cerr << "Usage: walnut_metrics --pred-root <results/v1> --gt-root <gt_final> --csv <metrics.csv> [--tol 2]\n";
            return 1;
        }

        const auto rows = collect_rows(pred_root, gt_root, tol);
        walnut::fs::create_directories(csv_path.parent_path());
        write_csv(csv_path, rows);
        print_summary(rows);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
