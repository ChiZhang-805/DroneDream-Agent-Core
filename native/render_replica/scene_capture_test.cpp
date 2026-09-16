#include "scene_capture.hpp"
#include <atomic>
#include <cmath>
#include <future>
#include <google/protobuf/util/message_differencer.h>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/Pose.hh>
#include <gz/sim/components/ParentEntity.hh>

namespace dronedream::replica {
namespace {
// 功能：
//   断言场景采集条件并报告失败行。
// 输入：
//   value：待验证条件。
//   line：调用位置。
// 输出：
//   void：条件为假时抛出测试异常。
void Check(bool value, int line = __builtin_LINE()) {
  if (!value) throw std::runtime_error("SCENE_CAPTURE_TEST_FAILED_LINE_" +
                                      std::to_string(line));
}
// 功能：
//   确认非法采集操作被拒绝，而不是产生不完整的成功快照。
// 输入：
//   action：预期失败的操作。
// 输出：
//   void：操作未抛异常时测试失败。
template<class F> void Fails(F action) {
  bool rejected = false;
  try { action(); } catch (const std::exception &) { rejected = true; }
  Check(rejected);
}
}

// 功能：
//   对照 Gazebo 全量序列化验证分片完整性、变化、删除及线程退出边界。
// 输入：
//   无；创建独立 ECM 与可控阻塞/故障读取器。
// 输出：
//   void：所有断言通过或抛出测试异常。
void TestSceneCapture() {
  extern void TestModelSdfState();
  TestModelSdfState();
  Fails([] { CompleteSceneCapture invalid(0); });
  Fails([] { CompleteSceneCapture invalid(5); });
  gz::sim::EntityComponentManager source;
  CompleteSceneCapture serial(1), parallel(4);
  Check(parallel.WorkerCount() == 4);
  // 功能：
  //   对比单线程和多线程结果与 SDK 完整输出，不能只比较实体数量。
  // 输入：
  //   无；捕获当前 source 与两种采集器。
  // 输出：
  //   void：逐组件字节及删除标志一致时通过。
  auto equivalent = [&] {
    // Compare directly against the original full LIST API. Gazebo's map API
    // omits an entity with no components, unlike its full list serialization.
    const auto reference = source.State();
    for (auto *capture : {&serial, &parallel}) {
      const auto actual = capture->Capture(source);
      Check(reference.entities_size() == actual.entities_size());
      for (const auto &entity : reference.entities()) {
        Check(actual.entities().count(entity.id()) == 1);
        const auto &value = actual.entities().at(entity.id());
        Check(value.id() == entity.id() && value.remove() == entity.remove());
        Check(value.components_size() == entity.components_size());
        for (const auto &component : entity.components()) {
          const auto key = static_cast<std::int64_t>(component.type());
          Check(value.components().count(key) == 1);
          Check(google::protobuf::util::MessageDifferencer::Equivalent(
              component, value.components().at(key)));
        }
      }
    }
  };
  equivalent(); // Empty subsets must not accidentally read the whole world.
  const auto bare = source.CreateEntity();
  equivalent(); // Full list API preserves entities without components.
  source.CreateComponent(bare, gz::sim::components::Name(""));
  equivalent(); // Empty component payload is valid.
  std::vector<gz::sim::Entity> entities;
  for (std::size_t i = 0; i < 129; ++i) {
    const auto id = source.CreateEntity();
    entities.push_back(id);
    source.CreateComponent(id, gz::sim::components::Name(std::to_string(i)));
    source.CreateComponent(id, gz::sim::components::Pose(
        gz::math::Pose3d(i, -1, 3, 0, 0, 0)));
    source.CreateComponent(id, gz::sim::components::ParentEntity(bare));
    source.SetParentEntity(id, bare);
    if (i < 8) equivalent(); // Fewer, equal and more entities than workers.
  }
  for (std::size_t tick = 0; tick < 12; ++tick) {
    for (const auto id : entities)
      source.Component<gz::sim::components::Pose>(id)->Data().Pos().X(tick);
    equivalent(); // No SetChanged() call: full capture must still observe data.
  }
  const auto child = entities.back();
  source.Component<gz::sim::components::ParentEntity>(child)->Data() = entities.front();
  source.SetParentEntity(child, entities.front());
  source.RemoveComponent<gz::sim::components::Pose>(entities[2]);
  source.RequestRemoveEntity(entities[3], false);
  equivalent(); // Reparenting and BOTH component and entity tombstones.
  source.ClearRemovedComponents();
  source.ProcessRemoveEntityRequests();
  source.ClearNewlyCreatedEntities();
  equivalent(); // Omission after tombstones were cleared must stay complete.

  // Every worker has a disjoint nonempty filter, with bounded live threads.
  std::atomic<unsigned> reads{0};
  CompleteSceneCapture checked(4, [&](const auto &ecm, const auto &filter) {
    Check(!filter.empty());
    ++reads;
    return ecm.State(filter);
  });
  gz::sim::EntityComponentManager empty;
  checked.Capture(empty);
  Check(reads == 0);
  const auto only = empty.CreateEntity();
  empty.CreateComponent(only, gz::sim::components::Name("only"));
  checked.Capture(empty);
  Check(reads == 1); // Three empty shards must not duplicate the entire world.
  checked.Capture(source);
  Check(reads == 5);
  const auto measured = checked.LastTiming();
  Check(std::isfinite(measured.membershipMs) && measured.membershipMs >= 0);
  Check(std::isfinite(measured.readJoinMs) && measured.readJoinMs >= 0);
  Check(std::isfinite(measured.mergeMs) && measured.mergeMs >= 0);
  for (const auto elapsed : measured.shardReadMs)
    Check(std::isfinite(elapsed) && elapsed >= 0 && elapsed <= measured.readJoinMs);

  // One failed shard must not let Capture return while another reads the ECM.
  std::promise<void> entered, release;
  auto releaseFuture = release.get_future().share();
  auto enteredFuture = entered.get_future();
  std::atomic<unsigned> started{0}, active{0};
  std::atomic<bool> inject{true};
  CompleteSceneCapture faults(2, [&](const auto &ecm, const auto &filter) {
    const auto order = started++;
    ++active;
    if (inject && order == 0) {
      entered.set_value();
      releaseFuture.wait();
    }
    --active;
    if (inject && order == 1) throw std::runtime_error("INJECTED_READ_FAILURE");
    return ecm.State(filter);
  });
  auto pending = std::async(std::launch::async, [&] {
    Fails([&] { faults.Capture(source); });
  });
  const bool blocked = enteredFuture.wait_for(std::chrono::seconds(2)) ==
      std::future_status::ready;
  const bool premature = pending.wait_for(std::chrono::milliseconds(20)) ==
      std::future_status::ready;
  release.set_value(); // Release even if a test assertion failed.
  pending.get();
  Check(blocked && !premature && active == 0);
  inject = false;
  Check(google::protobuf::util::MessageDifferencer::Equivalent(
      serial.Capture(source), faults.Capture(source))); // No stale partial reuse.
  faults.Stop();
  faults.Stop();
  Fails([&] { faults.Capture(source); });

  // Stop cannot detach outstanding reads or destroy their input lifetime.
  std::promise<void> stopEntered, stopRelease;
  auto stopEnteredFuture = stopEntered.get_future();
  auto stopReleaseFuture = stopRelease.get_future().share();
  CompleteSceneCapture stopping(1, [&](const auto &ecm, const auto &filter) {
    stopEntered.set_value();
    stopReleaseFuture.wait();
    return ecm.State(filter);
  });
  auto capturing = std::async(std::launch::async, [&] { return stopping.Capture(source); });
  const bool stopBlocked = stopEnteredFuture.wait_for(std::chrono::seconds(2)) ==
      std::future_status::ready;
  auto shutdown = std::async(std::launch::async, [&] { stopping.Stop(); });
  const bool stoppedEarly = shutdown.wait_for(std::chrono::milliseconds(20)) ==
      std::future_status::ready;
  stopRelease.set_value();
  const auto stoppedCapture = capturing.get();
  shutdown.get();
  Check(stopBlocked && !stoppedEarly);
  Check(google::protobuf::util::MessageDifferencer::Equivalent(
      serial.Capture(source), stoppedCapture));
  Fails([&] { stopping.Capture(source); });

  // Corrupt capture output fails as a whole, never publishes a partial scene.
  for (const int fault : {0, 1, 2, 3, 4}) {
    CompleteSceneCapture malformed(1, [fault](const auto &ecm, const auto &filter) {
      auto state = ecm.State(filter);
      auto *entity = state.mutable_entities(0);
      if (fault == 0) {
        entity->set_id(gz::sim::kNullEntity);
      } else if (fault == 1) {
        state.add_entities()->CopyFrom(*entity);
      } else if (fault == 2) {
        entity->add_components()->CopyFrom(entity->components(0));
      } else if (fault == 3) {
        entity->mutable_components(0)->set_component(
            std::string(kMaximumSceneBytes + 1, 'x'));
      } else {
        state.mutable_entities()->RemoveLast(); // Missing is not a deletion.
      }
      return state;
    });
    Fails([&] { malformed.Capture(source); });
  }
}
}  // namespace dronedream::replica
