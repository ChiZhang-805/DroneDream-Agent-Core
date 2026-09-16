#include "dronedream_agent_plugin_api/capability_plugin.hpp"

#include <chrono>
#include <cmath>
#include <pluginlib/class_list_macros.hpp>
#include <string_view>

namespace dronedream_agent_plugin_api
{
class SafeHoldCapability final : public CapabilityPlugin
{
public:
  // 功能：
  //   返回该插件的固定能力身份，供主机与冻结插件声明核对。
  // 输入：
  //   无。
  // 输出：
  //   capability_id：安全悬停策略的标识。
  std::string id() const override {return "runtime.safe-hold";}
  // 功能：
  //   声明策略建议权限；不是电机或飞控写权限。
  // 输入：
  //   无。
  // 输出：
  //   authority_name：control-policy 权限标签。
  std::string authority() const override {return "control-policy";}
  // 功能：
  //   重置旧运行状态并核对任务、权限及有限的超时范围，失败后不得继续使用旧配置。
  // 输入：
  //   configuration：主机提供的类型化任务和健康检测时限。
  // 输出：
  //   configured_：配置完整且匹配本插件时为真。
  bool configure(const CapabilityConfiguration & configuration) override
  {
    active_ = holding_ = flight_active_ = false;
    observation_ = {};
    configuration_ = configuration;
    configured_ = configuration.capability_id == id() && configuration.authority == authority() &&
      !configuration.contract_id.empty() && configuration.deadline_ms > 0 &&
      configuration.deadline_ms <= 60000 &&
      configuration.startup_deadline_ms >= configuration.deadline_ms &&
      configuration.startup_deadline_ms <= 300000;
    return configured_;
  }
  // 功能：
  //   启动一次新的健康检测周期，重复激活不能刷新已运行任务的计时或清除故障。
  // 输入：
  //   无；使用已验证配置。
  // 输出：
  //   active_：本次成功激活时为真，已有活动周期时返回假。
  bool activate() override
  {
    if (active_) return false;
    holding_ = flight_active_ = false;
    observation_ = {};
    active_ = configured_;
    last_observation_ = std::chrono::steady_clock::now();
    return active_;
  }
  // 功能：
  //   接受本次任务的递增观测，拒绝重放、链路失效或回退起飞前状态，避免放宽飞行时限。
  // 输入：
  //   observation：含任务身份、序号、主机单调时间及有效性标识的观测。
  // 输出：
  //   accepted：接收有效新观测时为真，否则为假且不刷新上次有效时间。
  bool observe(const CapabilityObservation & observation) override
  {
    if (!active_ || observation.contract_id != configuration_.contract_id) {
      return false;
    }
    if (!valid_runtime_phase(observation.runtime_phase)) {
      return false;
    }
    if (observation.sequence == 0 || observation.sequence <= observation_.sequence ||
        observation.monotonic_time_ns <= 0 ||
        observation.monotonic_time_ns <= observation_.monotonic_time_ns ||
        !observation.localization_ok || !observation.link_ok ||
        !std::isfinite(observation.battery_percent) || observation.battery_percent < 0 ||
        observation.battery_percent > 100 ||
        (flight_active_ && observation.runtime_phase == "PREFLIGHT")) {
      return false;
    }
    observation_ = observation;
    last_observation_ = std::chrono::steady_clock::now();
    flight_active_ = flight_active_ || observation.runtime_phase != "PREFLIGHT";
    return true;
  }
  // 功能：
  //   提出与当前观测序号绑定的安全悬停建议，不向执行器发送实际控制量。
  // 输入：
  //   无；读取最近有效观测。
  // 输出：
  //   proposal：必须经固定内核授权的策略建议。
  CapabilityProposal propose() override
  {
    const CapabilityProposal proposal{
      "safe-hold-" + std::to_string(observation_.sequence),
      "safe_hold",
      "Runtime policy requests zero-velocity hold through core authorization.",
      true};
    return proposal;
  }
  // 功能：
  //   只接纳当前健康观测对应的悬停提案；记录策略状态不等同于实际飞机已经悬停。
  // 输入：
  //   proposal：固定内核交回的类型化提案。
  // 输出：
  //   receipt：策略接纳或拒绝回执，物理执行需主机安全事件与执行器完成。
  CapabilityExecutionReceipt execute(const CapabilityProposal & proposal) override
  {
    if (!active_ || !health().healthy || observation_.sequence == 0 ||
        proposal.proposal_id != "safe-hold-" + std::to_string(observation_.sequence) ||
        proposal.command != "safe_hold" || !proposal.requires_core_authorization) {
      return CapabilityExecutionReceipt{
        proposal.proposal_id, false, "rejected", {"CORE_AUTHORIZATION_REQUIRED"}};
    }
    holding_ = true;
    return CapabilityExecutionReceipt{proposal.proposal_id, true, "holding", {}};
  }
  // 功能：
  //   标记安全策略进入保持请求状态，把原因交回主机，不直接伪造物理悬停证据。
  // 输入：
  //   reason：主机确认的安全原因。
  // 输出：
  //   receipt：策略状态和原因的回执。
  CapabilityExecutionReceipt hold(const std::string & reason) noexcept override
  {
    holding_ = active_;
    return CapabilityExecutionReceipt{
      "watchdog-hold", active_, active_ ? "holding" : "inactive", {reason}};
  }
  // 功能：
  //   用单调时钟检查最后有效观测年龄；起飞后只能使用严格飞行时限。
  // 输入：
  //   无；使用当前配置及观测接收时间。
  // 输出：
  //   report：健康、超时、策略状态、观测年龄和时限。
  CapabilityHealth health() const noexcept override
  {
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - last_observation_).count();
    const auto limit = flight_active_ ? configuration_.deadline_ms :
      configuration_.startup_deadline_ms;
    const bool missed = active_ && elapsed > limit;
    const CapabilityHealth report{
      active_ && !missed, missed,
      holding_ ? "holding" : (missed ? "deadline-missed" : (active_ ? "active" : "inactive")),
      static_cast<std::uint64_t>(elapsed < 0 ? 0 : elapsed),
      static_cast<std::uint64_t>(limit)};
    return report;
  }
  // 功能：
  //   导出本插件的任务和观测身份；这里的 holding 是策略请求状态，不证明物理运动结果。
  // 输入：
  //   无。
  // 输出：
  //   evidence_record：策略生命周期证据。
  CapabilityEvidence evidence() const noexcept override
  {
    const CapabilityEvidence evidence_record{
      id(), configuration_.contract_id, observation_.sequence,
      holding_ ? "holding" : (active_ ? "active" : "inactive")};
    return evidence_record;
  }
  // 功能：
  //   撤销当前插件周期，不改变飞控状态或删除任务证据。
  // 输入：
  //   无。
  // 输出：
  //   void：插件转为不活动。
  void deactivate() noexcept override {active_ = false;}

private:
  // 功能：
  //   仅接纳与 ROS 桥及执行器共享的显式阶段，未知阶段不能默认为起飞前。
  // 输入：
  //   phase：观测中的运行阶段。
  // 输出：
  //   valid：阶段属于当前契约时为真。
  static bool valid_runtime_phase(const std::string_view phase) noexcept
  {
    return phase == "PREFLIGHT" || phase == "TAKEOFF" || phase == "ACTION" ||
           phase == "TRACK" || phase == "HOLDING" || phase == "LOCAL_SLOW" ||
           phase == "LOCAL_REPLAN" || phase == "MODEL_AUTHORITY_HOLD" ||
           phase == "PERCEPTION_REFRESH_HOLD" ||
           phase == "PERCEPTION_STARTUP_HOLD" || phase == "TRACKING_RECOVERY" ||
           phase == "WAYPOINT_SETTLE" || phase == "CHECKPOINT" ||
           phase == "PAUSED" || phase == "LANDING" || phase == "LANDED" ||
           phase == "COMPLETE" || phase == "FAILED";
  }

  CapabilityConfiguration configuration_{};
  CapabilityObservation observation_{};
  bool configured_{false};
  bool active_{false};
  bool holding_{false};
  bool flight_active_{false};
  std::chrono::steady_clock::time_point last_observation_{};
};
}  // namespace dronedream_agent_plugin_api

PLUGINLIB_EXPORT_CLASS(
  dronedream_agent_plugin_api::SafeHoldCapability,
  dronedream_agent_plugin_api::CapabilityPlugin)
