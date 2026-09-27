#pragma once
#include <cstdint>
#include <map>
#include <optional>
#include <cstdlib>

namespace dronedream {
// 有界仿真采集时钟。只映射精确物理 tick，不插值、不读取位姿、不使用接收时间补票。
class CaptureClock {
 public:
  struct Stamp { std::int64_t unixNs, steadyNs; };
  // 功能：记录渲染前的物理 tick；暂停清空、重复不续期、时间跳变隔离旧帧。
  //       仿真回退或原始双钟倒退仍封闭本次运行；可恢复的偏移变化不毒化后续新帧。
  // 输入：仿真纳秒、墙钟纳秒、单调纳秒及暂停状态。输出：是否仍可接纳图像。
  bool Add(std::int64_t sim, std::int64_t unixNs, std::int64_t steadyNs, bool paused) {
    if (failed) return false;
    if (paused) { ticks.clear(); return true; }
    if (sim < 0 || unixNs <= 0 || steadyNs <= 0 ||
        (lastSim >= 0 && (sim < lastSim || unixNs < last.unixNs || steadyNs < last.steadyNs))) {
      failed = true; ticks.clear(); return false;
    }
    // 重复 tick 不重新映射曝光时间；暂停恢复可能先报告同一个 simTime。
    if (sim == lastSim) return true;
    if (lastSim >= 0 && (unixNs == last.unixNs || steadyNs == last.steadyNs ||
        std::llabs((unixNs-last.unixNs)-(steadyNs-last.steadyNs)) > 5000000)) {
      // 清空跨越跳变的映射，跳变所在 tick 也不可发布；下一次连续 tick 才能恢复。
      ticks.clear(); lastSim = sim; last = {unixNs, steadyNs}; return true;
    }
    lastSim = sim; last = {unixNs, steadyNs};
    ticks.emplace(sim, last);
    while (ticks.size() > 4096) ticks.erase(ticks.begin());
    return true;
  }
  // 功能：只查找同一 tick 且仍在原始 250 毫秒期限内的来源，双钟同时检查。
  // 输入：图像原生仿真时间及当前双钟。输出：原始时间或空；从不重新计时。
  std::optional<Stamp> Find(std::int64_t sim, std::int64_t unixNs, std::int64_t steadyNs) const {
    const auto it = ticks.find(sim);
    if (failed || it == ticks.end()) return {};
    const auto s = it->second;
    if (unixNs < s.unixNs || steadyNs < s.steadyNs ||
        unixNs-s.unixNs > 250000000 || steadyNs-s.steadyNs > 250000000 ||
        std::llabs((unixNs-s.unixNs)-(steadyNs-s.steadyNs)) > 5000000) return {};
    return s;
  }
 private:
  std::map<std::int64_t, Stamp> ticks;
  std::int64_t lastSim = -1;
  Stamp last{0, 0};
  bool failed = false;
};
}
