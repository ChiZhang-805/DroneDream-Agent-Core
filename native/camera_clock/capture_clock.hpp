#pragma once
#include <cstdint>
#include <deque>
#include <unordered_map>
#include <optional>
#include <cstdlib>

namespace dronedream {
// 有界仿真采集时钟。只映射精确物理 tick，不插值、不读取位姿、不使用接收时间补票。
class CaptureClock {
 public:
  struct Stamp { std::int64_t unixNs, steadyNs; };
  // 功能：记录渲染前的物理 tick；暂停清空、重复不续期、墙钟跳变隔离旧帧后自动恢复。
  //       仿真回退或原始双钟倒退仍封闭本次运行；可恢复的偏移变化不毒化后续新帧。
  // 输入：仿真纳秒、墙钟纳秒、单调纳秒及暂停状态。输出：是否仍可接纳图像。
  bool Add(std::int64_t sim, std::int64_t unixNs, std::int64_t steadyNs, bool paused) {
    if (failed) return false;
    if (paused) { ticks.clear(); order.clear(); return true; }
    if (sim < 0 || unixNs <= 0 || steadyNs <= 0 ||
        (lastSim >= 0 && (sim < lastSim || steadyNs < last.steadyNs))) {
      failed = true; ticks.clear(); order.clear(); return false;
    }
    // 重复 tick 不重新映射曝光时间；暂停恢复可能先报告同一个 simTime。
    if (sim == lastSim) return true;
    if (lastSim >= 0 && (unixNs <= last.unixNs || steadyNs == last.steadyNs ||
        std::llabs((unixNs-last.unixNs)-(steadyNs-last.steadyNs)) > 5000000)) {
      // Windows/WSL 时间同步可能把墙钟向前或向后校正。墙钟不参与控制
      // 顺序；单调钟和仿真钟仍向前时，只隔离跨越跳变的旧映射。跳变
      // 所在 tick 不发布，下一次连续 tick 自动恢复，不能永久关闭视觉流。
      ticks.clear(); order.clear(); lastSim = sim; last = {unixNs, steadyNs}; return true;
    }
    lastSim = sim; last = {unixNs, steadyNs};
    ticks.emplace(sim, last);
    order.push_back(sim);
    // Rendering may complete after many 1 ms physics ticks in a complex 3D
    // world even though less than the 250 ms wall-clock freshness limit has
    // elapsed. Keep a bounded 65.536 s simulation-time window so the exact
    // exposure tick survives that render pipeline; Find still applies both
    // wall and steady-clock freshness checks and never renews an old frame.
    while (order.size() > 65536) {
      ticks.erase(order.front());
      order.pop_front();
    }
    return true;
  }
  // 功能：只查找同一 tick 且仍在原始 250 毫秒期限内的来源，双钟同时检查。
  // 输入：图像原生仿真时间及当前双钟。输出：原始时间或空；从不重新计时。
  int FindIssue(std::int64_t sim, std::int64_t unixNs, std::int64_t steadyNs) const {
    const auto it = ticks.find(sim);
    if (failed) return 1;
    if (it == ticks.end()) return 2;
    const auto s = it->second;
    if (unixNs < s.unixNs) return 3;
    if (steadyNs < s.steadyNs) return 4;
    if (unixNs-s.unixNs > 250000000) return 5;
    if (steadyNs-s.steadyNs > 250000000) return 6;
    if (std::llabs((unixNs-s.unixNs)-(steadyNs-s.steadyNs)) > 5000000) return 7;
    return 0;
  }
  std::optional<Stamp> Find(std::int64_t sim, std::int64_t unixNs, std::int64_t steadyNs) const {
    if (FindIssue(sim, unixNs, steadyNs) != 0) return {};
    const auto s = ticks.find(sim)->second;
    return s;
  }
 private:
  // Physics may run much faster than wall time. Hash lookup plus FIFO eviction
  // keeps every PreUpdate operation O(1), preventing render callbacks from
  // starving behind a growing ordered-tree mutation.
  std::unordered_map<std::int64_t, Stamp> ticks;
  std::deque<std::int64_t> order;
  std::int64_t lastSim = -1;
  Stamp last{0, 0};
  bool failed = false;
};
}
