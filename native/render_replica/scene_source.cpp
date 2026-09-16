// Read-only physics-side snapshots. No rendering or flight command code belongs here.
#include "replica_contract.hpp"
#include "scene_capture.hpp"
#include "scene_worker.hpp"
#include "replica_timing.hpp"

#include <sstream>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <gz/msgs/bytes.pb.h>
#include <gz/msgs/serialized_map.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/Conversions.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/System.hh>
#include <gz/sim/components/Factory.hh>
#include <gz/transport/Node.hh>

namespace dronedream {
class RenderSceneSource final : public gz::sim::System,
    public gz::sim::ISystemConfigure, public gz::sim::ISystemPostUpdate {
 public:
  // 功能：
  //   绑定本次仿真的唯一源身份和发布主题，创建只读采集与有界发布线程；禁止重配置混源。
  // 输入：
  //   sdf：明确提供 epoch、snapshot_topic 和独立回执路径的插件配置。
  //   其余 SDK 参数：接口必需，本方法不修改实体或物理状态。
  // 输出：
  //   void：配置成功后启用采集；参数或传输初始化失败抛出异常。
  void Configure(const gz::sim::Entity &, const std::shared_ptr<const sdf::Element> &sdf,
      gz::sim::EntityComponentManager &, gz::sim::EventManager &) override {
    if (!sdf || configured || worker || capture)
      throw std::runtime_error("SCENE_SOURCE_CONFIGURATION_LIFETIME");
    epoch = sdf->Get<std::string>("epoch");
    const auto topic = sdf->Get<std::string>("snapshot_topic");
    receipt = sdf->Get<std::string>("receipt");
    if (!replica::IsDigest(epoch) || topic.empty() || topic.front() != '/' || topic.size() > 512
        || !receipt.is_absolute() || std::filesystem::exists(receipt))
      throw std::runtime_error("SCENE_SOURCE_CONFIG_INVALID");
    publisher = node.Advertise<gz::msgs::Bytes>(topic);
    if (!publisher) throw std::runtime_error("SCENE_SOURCE_ADVERTISE_FAILED");
    capture = std::make_unique<replica::CompleteSceneCapture>();
    worker = std::make_unique<replica::LatestSceneWorker>(
        [this](const replica::CapturedScene &scene) { Publish(scene); });
    configured = true;
  }

  // 功能：
  //   限频采集当前完整物理场景，保留采集前的源时间；暂停、时间倒退或后台失败不伪造新图。
  // 输入：
  //   info：本次物理更新的仿真时间和暂停状态。
  //   ecm：当前只读实体组件状态。
  // 输出：
  //   void：最新完整快照提交后台发布，或记录终止原因。
  void PostUpdate(const gz::sim::UpdateInfo &info,
                  const gz::sim::EntityComponentManager &ecm) override {
    if (!configured || failed || info.paused) return;
    if (worker->Failed()) {
      failed = true;
      failure = "SCENE_BACKGROUND_PUBLICATION_FAILED";
      return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (now < next) return;
    next = now + std::chrono::milliseconds(50);  // No catch-up publication bursts.
    const auto started = now;
    try {
      const auto stamp = replica::UnixNs();  // Before copying, not after serialization.
      const auto sim = std::chrono::duration_cast<std::chrono::nanoseconds>(info.simTime).count();
      if (sim <= lastSim || stamp <= lastUnix || ecm.EntityCount() > replica::kMaximumEntities
          || sequence == std::numeric_limits<std::int64_t>::max())
        throw std::runtime_error("SCENE_SOURCE_RESET_OR_CAPACITY");
      auto scene = std::make_unique<replica::CapturedScene>();
      auto &state = scene->state;
      gz::sim::set(state.mutable_stats(), info);
      *state.mutable_state() = capture->Capture(ecm);
      const auto captured = std::chrono::steady_clock::now();
      if (state.ByteSizeLong() > replica::kMaximumSceneBytes)
        throw std::runtime_error("SCENE_SOURCE_CAPACITY_EXCEEDED");
      scene->identity = {epoch, "", ++sequence, stamp, sim};
      maximumBytes = std::max(maximumBytes, state.ByteSizeLong());
      worker->Submit(std::move(scene));
      captureTimes.Add(std::chrono::duration<double, std::milli>(captured-started).count());
      const auto stages = capture->LastTiming();
      membershipTimes.Add(stages.membershipMs);
      readJoinTimes.Add(stages.readJoinMs);
      mergeTimes.Add(stages.mergeMs);
      for (std::size_t i = 0; i < capture->WorkerCount(); ++i)
        shardTimes[i].Add(stages.shardReadMs[i]);
      largestPayloads = stages.largestPayloads;
      if (lastUnix > 0) maximumIntervalMs = std::max(maximumIntervalMs, (stamp-lastUnix)/1e6);
      lastSim = sim;
      lastUnix = stamp;
    } catch (const std::exception &error) {
      failed = true;
      failure = error.what(); // Local fixed diagnostic codes only.
    }
    maximumWorkMs = std::max(maximumWorkMs,
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-started).count());
  }

  // 功能：
  //   先回收发布与采集线程，再写诊断；该回执不代表飞行成功或模型资格。
  // 输入：
  //   无；本实例累计计数、耗时及失败状态。
  // 输出：
  //   void：释放线程并尽力保存本次源端诊断，写入失败输出错误。
  ~RenderSceneSource() override {
    if (!configured) return;
    try {
    worker->Stop(); // No publication may outlive the owning transport Node.
    capture->Stop(); // Joined full reads; no ECM is retained between callbacks.
    if (worker->Failed()) { failed = true; failure = "SCENE_BACKGROUND_PUBLICATION_FAILED"; }
    // Diagnostics only: this does not prove that any vehicle landed.
    std::ostringstream stream;
    const auto modelStats = capture->ModelStatistics();
    stream << "{\"qualification_granted\":false,\"snapshots\":" << sequence
      << ",\"maximum_work_ms\":" << maximumWorkMs
      << ",\"maximum_source_interval_ms\":" << maximumIntervalMs
      << ",\"maximum_scene_bytes\":" << maximumBytes
      << ",\"capture_workers\":" << capture->WorkerCount()
      << ",\"model_xml_content_reuse\":{\"hits\":" << modelStats.hits
      << ",\"misses\":" << modelStats.misses
      << ",\"removed_component_bypasses\":" << modelStats.removedBypasses
      << ",\"untraversable_tree_bypasses\":" << modelStats.treeBypasses << '}'
      << ",\"publication_worker\":" << worker->Summary()
      << ",\"expired_before_publish\":" << expired
      << ",\"published\":" << published
      << ",\"capture_ms\":" << captureTimes.Json()
      << ",\"capture_membership_ms\":" << membershipTimes.Json()
      << ",\"capture_read_join_ms\":" << readJoinTimes.Json()
      << ",\"capture_merge_ms\":" << mergeTimes.Json()
      << ",\"capture_shard_read_ms\":[";
    for (std::size_t i = 0; i < capture->WorkerCount(); ++i) {
      if (i) stream << ',';
      stream << shardTimes[i].Json();
    }
    stream << "],\"last_largest_component_payloads\":[";
    for (std::size_t i = 0; i < capture->WorkerCount(); ++i) {
      if (i) stream << ',';
      const auto &payload = largestPayloads[i];
      stream << "{\"entity\":" << payload.entity << ",\"type\":" << payload.type
        << ",\"bytes\":" << payload.bytes << ",\"name\":"
        << replica::JsonText(gz::sim::components::Factory::Instance()->Name(payload.type)) << '}';
    }
    stream << ']'
      << ",\"encode_ms\":" << encodeTimes.Json()
      << ",\"publish_ms\":" << publishTimes.Json()
      << ",\"failed\":" << (failed ? "true" : "false")
      << ",\"failure\":" << replica::JsonText(failure) << "}";
    replica::WriteNew(receipt.string(), stream.str());
    }
    catch (const std::exception &error) { std::cerr << error.what() << '\n'; }
  }
 private:
  // 功能：
  //   在发布线程编码并散列完整快照，编码前后均检查原始帧龄，不能给慢图重新打新时间。
  // 输入：
  //   scene：采集线程移交的独立场景及原始身份。
  // 输出：
  //   void：有效快照发布，超龄计入丢弃，失败由后台线程封闭后续发布。
  void Publish(const replica::CapturedScene &scene) {
    const auto begin = std::chrono::steady_clock::now();
    if (!replica::Fresh(scene.identity, replica::UnixNs())) { ++expired; return; }
    gz::msgs::Bytes packet;
    if (!scene.state.SerializeToString(packet.mutable_data()))
      throw std::runtime_error("SCENE_SOURCE_SERIALIZATION_FAILED");
    auto identity = scene.identity;
    identity.sha256 = replica::Sha256(packet.data());
    replica::WriteIdentity(packet.mutable_header(), identity);
    const auto encoded = std::chrono::steady_clock::now();
    if (!replica::Fresh(identity, replica::UnixNs())) { ++expired; return; }
    if (!publisher.Publish(packet)) throw std::runtime_error("SCENE_SOURCE_PUBLICATION_FAILED");
    const auto completed = std::chrono::steady_clock::now();
    encodeTimes.Add(std::chrono::duration<double, std::milli>(encoded-begin).count());
    publishTimes.Add(std::chrono::duration<double, std::milli>(completed-encoded).count());
    ++published;
  }
  // 与副本共用有界、拒绝非有限值的统计实现，避免两套相同指标逐渐分叉。
  replica::TimingDistribution captureTimes, encodeTimes, publishTimes;
  replica::TimingDistribution membershipTimes, readJoinTimes, mergeTimes;
  std::array<replica::TimingDistribution, 4> shardTimes;
  std::array<replica::CompleteSceneCapture::Timing::Payload, 4> largestPayloads{};
  gz::transport::Node node;
  gz::transport::Node::Publisher publisher;
  std::unique_ptr<replica::LatestSceneWorker> worker;
  std::unique_ptr<replica::CompleteSceneCapture> capture;
  std::uint64_t expired = 0, published = 0; // Worker-owned; read only after join.
  std::string epoch, failure;
  std::filesystem::path receipt;
  bool configured = false, failed = false;
  std::chrono::steady_clock::time_point next{};
  std::int64_t sequence = 0, lastSim = -1, lastUnix = 0;
  std::size_t maximumBytes = 0;
  double maximumWorkMs = 0, maximumIntervalMs = 0;
};
}
GZ_ADD_PLUGIN(dronedream::RenderSceneSource, gz::sim::System,
  gz::sim::ISystemConfigure, gz::sim::ISystemPostUpdate)
GZ_ADD_PLUGIN_ALIAS(dronedream::RenderSceneSource, "dronedream::RenderSceneSource")
