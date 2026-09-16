#include "dronedream_agent_plugin_api/capability_plugin.hpp"

#include <chrono>
#include <iostream>
#include <memory>
#include <pluginlib/class_loader.hpp>
#include <thread>

// 功能：
//   动态加载实际策略插件，测试生命周期、重复观测、旧提案及飞行阶段回退的拒绝行为。
// 输入：
//   无；使用当前安装前缀注册的 pluginlib 插件。
// 输出：
//   exit_code：所有契约检查通过为 0，非零指明失败检查；不会驱动飞控。
int main()
{
  pluginlib::ClassLoader<dronedream_agent_plugin_api::CapabilityPlugin> loader(
    "dronedream_agent_plugin_api",
    "dronedream_agent_plugin_api::CapabilityPlugin");
  auto plugin = loader.createSharedInstance(
    "dronedream_agent_plugin_api/SafeHoldCapability");
  dronedream_agent_plugin_api::CapabilityConfiguration configuration{
    plugin->id(), "probe-contract", plugin->authority(), 100, 1000};
  if (!plugin->configure(configuration) || !plugin->activate() || !plugin->health().healthy) {
    return 2;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(125));
  if (!plugin->health().healthy) {
    return 3;
  }
  dronedream_agent_plugin_api::CapabilityObservation observation{
    "probe-contract", "PREFLIGHT", 1, 1, true, true, true, 100.0};
  if (!plugin->observe(observation)) {
    return 4;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(125));
  if (!plugin->health().healthy) {
    return 6;
  }
  observation.runtime_phase = "TAKEOFF";
  observation.sequence = 2;
  observation.monotonic_time_ns = 2;
  if (!plugin->observe(observation) || !plugin->health().healthy) {
    return 7;
  }
  auto invalid_observation = observation;
  invalid_observation.runtime_phase = "UNRECOGNIZED_PHASE";
  if (plugin->observe(invalid_observation)) {
    return 8;
  }
  const auto proposal = plugin->propose();
  if (plugin->observe(observation) || plugin->activate()) return 9;
  auto preflight_replay = observation;
  preflight_replay.runtime_phase = "PREFLIGHT";
  preflight_replay.sequence = 3;
  preflight_replay.monotonic_time_ns = 3;
  if (plugin->observe(preflight_replay)) return 10;
  auto stale_proposal = proposal;
  stale_proposal.proposal_id = "safe-hold-1";
  if (plugin->execute(stale_proposal).accepted) return 11;
  const auto execution = plugin->execute(proposal);
  const auto evidence = plugin->evidence();
  if (!execution.accepted || evidence.observation_sequence != 2) {
    return 5;
  }
  std::cout << "PLUGIN_PROBE_READY id=" << plugin->id()
            << " authority=" << plugin->authority()
            << " abi=configure,observe,propose,execute,hold,health,evidence" << std::endl;
  plugin->deactivate();
  const int exit_code = plugin->health().healthy ? 3 : 0;
  return exit_code;
}
