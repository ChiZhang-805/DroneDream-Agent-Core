#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <sstream>
#include <vector>

namespace dronedream::replica {
// Diagnostic only. Callers provide clocks and synchronize concurrent access.
// Store a bounded tail, while counts/means/maxima cover the complete run.
class TimingDistribution {
 public:
  // 功能：
  //   更新真实时长的累计均值和最大值，只保留最近 1024 条分位样本；无效值只记诊断次数。
  // 输入：
  //   value：非负有限时长，与调用方约定相同单位。
  // 输出：
  //   void：更新统计，不改变任何帧的有效期或控制权限。
  void Add(double value) {
    if (!std::isfinite(value) || value < 0 || count == std::numeric_limits<std::uint64_t>::max()) {
      ++invalid;
      return;
    }
    ++count;
    // 在线均值不累计总和，避免大量大值把仅用于诊断的 JSON 污染成无穷。
    mean += (value - mean) / static_cast<double>(count);
    maximum = std::max(maximum, value);
    tail.push_back(value);
    if (tail.size() > kTailCapacity) tail.pop_front();
  }
  // 功能：
  //   返回实际保留的近期样本数，不冒充全程总样本数。
  // 输入：
  //   无。
  // 输出：
  //   size：当前尾部窗口长度。
  std::size_t TailSize() const { return tail.size(); }
  // 功能：
  //   导出全程计数、均值、最大值与近期窗口分位，分位不代表全程或端到端时延之和。
  // 输入：
  //   无；调用方负责与写入统计互斥。
  // 输出：
  //   text：有限数字组成的 JSON 统计对象。
  std::string Json() const {
    std::vector<double> sorted(tail.begin(), tail.end());
    std::sort(sorted.begin(), sorted.end());
    // 功能：
    //   从已排序近期窗口取下取整经验分位，空窗口返回零并由 tail_count 明确其无样本。
    // 输入：
    //   p：本函数使用的 50、95 或 99 百分位。
    // 输出：
    //   value：对应近期样本的时长。
    const auto percentile = [&](std::size_t p) {
      return sorted.empty() ? 0.0 : sorted[(sorted.size()-1)*p/100];
    };
    std::ostringstream out;
    out << "{\"count\":" << count << ",\"maximum\":" << maximum
        << ",\"mean\":" << mean
        << ",\"invalid_count\":" << invalid
        << ",\"tail_count\":" << sorted.size()
        << ",\"tail_p50\":" << percentile(50)
        << ",\"tail_p95\":" << percentile(95)
        << ",\"tail_p99\":" << percentile(99) << "}";
    const auto text = out.str();
    return text;
  }
  static constexpr std::size_t kTailCapacity = 1024;
 private:
  std::deque<double> tail;
  std::uint64_t count = 0;
  std::uint64_t invalid = 0;
  double mean = 0, maximum = 0;
};

// Every row is for ONE real image and its exact source scene identity. No
// independent per-stage percentiles are summed into an end-to-end percentile.
class ImageTransitTimings {
 public:
  using Row = std::array<double, 6>;
  // 功能：
  //   只接纳同一真实图像的六项有限时长，任一列异常则整行拒绝，避免错拼不同图像阶段。
  // 输入：
  //   row：源到接收、验证、等待、场景应用、成像及端到端六项毫秒时长。
  // 输出：
  //   accepted：整行进入统计时为真，异常行返回假。
  bool Add(const Row &row) {
    if (!std::all_of(row.begin(), row.end(), [](double x) {
          return std::isfinite(x) && x >= 0;
        })) {
      ++invalid;
      return false; // Clock diagnostics never authorize or re-date a frame.
    }
    for (std::size_t i = 0; i < row.size(); ++i) columns[i].Add(row[i]);
    return true;
  }
  // 功能：
  //   分别输出同图像各阶段统计，端到端分位来自实际端到端测量而非独立阶段分位相加。
  // 输入：
  //   无；调用方同步读写。
  // 输出：
  //   text：带六类时长及异常行数的 JSON。
  std::string Json() const {
    static constexpr std::array<const char *, 6> names{
      "source_to_scene_receive_ms", "scene_validation_ms", "pending_scene_wait_ms",
      "scene_apply_to_sensor_submit_ms", "sensor_submit_to_image_ms",
      "source_to_image_ms"};
    std::ostringstream out;
    out << "{\"invalid_clock_rows\":" << invalid;
    for (std::size_t i = 0; i < columns.size(); ++i)
      out << ",\"" << names[i] << "\":" << columns[i].Json();
    out << "}";
    const auto text = out.str();
    return text;
  }
 private:
  std::array<TimingDistribution, 6> columns;
  std::uint64_t invalid = 0;
};
}  // namespace dronedream::replica
