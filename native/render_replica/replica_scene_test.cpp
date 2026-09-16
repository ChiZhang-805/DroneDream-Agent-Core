#include "replica_scene.hpp"
#include "replica_camera.hpp"
#include "scene_capture.hpp"
#include "scene_worker.hpp"
#include <future>
#include <google/protobuf/util/message_differencer.h>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/Pose.hh>
#include <gz/sim/components/World.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Link.hh>
#include <gz/sim/components/Sensor.hh>
#include <gz/sim/components/ParentEntity.hh>
#include <gz/sim/components/GpuLidar.hh>
#include <iostream>

using namespace dronedream::replica;
namespace dronedream::replica { void TestSceneCapture(); }
// 功能：
//   验证场景复制和后台线程的可观察行为。
// 输入：
//   value：测试条件。
// 输出：
//   void：条件不成立时抛出测试异常。
void Require(bool value) { if (!value) throw std::runtime_error("TEST_FAILED"); }
// 功能：
//   验证异常输入没有被当作成功应用。
// 输入：
//   action：预期抛出异常的操作。
// 输出：
//   void：仅发生异常时通过。
template<class F> void Reject(F action) {
  bool rejected = false;
  try { action(); } catch (const std::exception &) { rejected = true; }
  Require(rejected);
}

// 功能：
//   回归真实 SDK 场景增删、组件变化、传感器主题及有界后台队列生命周期。
// 输入：
//   无；使用独立 ECM 和受控线程夹具。
// 输出：
//   result：全部通过返回零，异常导致测试失败。
int main() {
  TestSceneCapture();
  CompleteSceneCapture sceneCapture;
  gz::sim::EntityComponentManager source, replica;
  const auto first = source.CreateEntity();
  source.CreateComponent(first, gz::sim::components::Name("first"));
  source.CreateComponent(first, gz::sim::components::Pose(gz::math::Pose3d(1, 2, 3, 0, 0, 0)));
  // 功能：
  //   以 SDK 序列化对照自定义采集器，防止优化后丢失实际组件。
  // 输入：
  //   无；捕获当前 source。
  // 输出：
  //   captured：与 SDK 一致的完整场景。
  auto snapshot = [&] {
    gz::msgs::SerializedStateMap state;
    source.State(state, {}, {}, true);
    auto captured = sceneCapture.Capture(source);
    Require(google::protobuf::util::MessageDifferencer::Equivalent(state, captured));
    return captured;
  };
  CompleteScene scene;
  scene.Apply(snapshot(), replica);
  Require(replica.Component<gz::sim::components::Name>(first)->Data() == "first");
  Require(replica.Component<gz::sim::components::Pose>(first)->Data().Pos().X() == 1);
  Require(scene.LastComponentsApplied() == 2);
  scene.Apply(snapshot(), replica);
  Require(scene.LastComponentsApplied() == 0);
  source.Component<gz::sim::components::Pose>(first)->Data().Pos().X(4);
  scene.Apply(snapshot(), replica);
  Require(scene.LastComponentsApplied() == 1);
  Require(replica.Component<gz::sim::components::Pose>(first)->Data().Pos().X() == 4);
  // A local render-only override is restored from the authoritative source
  // even when its wire bytes did not change (camera private-topic redirection).
  replica.Component<gz::sim::components::Name>(first)->Data() = "private-override";
  scene.Apply(snapshot(), replica, {gz::sim::components::Name::typeId});
  Require(scene.LastComponentsApplied() == 1);
  Require(replica.Component<gz::sim::components::Name>(first)->Data() == "first");
  replica.ClearNewlyCreatedEntities();

  // Changes in intermediate snapshots may be skipped without losing deletion.
  source.RemoveComponent<gz::sim::components::Pose>(first);
  // Tombstones are preserved before Gazebo clears them, and omission still
  // removes components when intermediate snapshots were coalesced.
  const auto marked = snapshot();
  Require(marked.entities().at(first).components().at(
      static_cast<std::int64_t>(gz::sim::components::Pose::typeId)).remove());
  source.ClearRemovedComponents();
  auto removedComponent = snapshot();
  Require(!removedComponent.entities().at(first).components().count(
      gz::sim::components::Pose::typeId));
  scene.Apply(removedComponent, replica);
  Require(replica.Component<gz::sim::components::Pose>(first) == nullptr);

  source.RequestRemoveEntity(first);
  Require(snapshot().entities().at(first).remove());
  source.ProcessRemoveEntityRequests();
  source.ClearNewlyCreatedEntities();
  const auto second = source.CreateEntity();
  source.CreateComponent(second, gz::sim::components::Name("second"));
  scene.Apply(snapshot(), replica);
  Require(!scene.Contains(first) && scene.Contains(second));
  bool removalObserved = false;
  replica.EachRemoved<gz::sim::components::Name>([&](const gz::sim::Entity &id,
      const gz::sim::components::Name *) { removalObserved |= id == first; return true; });
  Require(removalObserved);
  replica.ProcessRemoveEntityRequests();
  Require(!replica.HasEntity(first));
  Require(replica.Component<gz::sim::components::Name>(second)->Data() == "second");

  auto bad = snapshot();
  (*bad.mutable_entities())[second].set_id(second + 100);
  Reject([&] { scene.Apply(bad, replica); });
  Require(scene.Contains(second));
  bad = snapshot();
  auto &unknown = (*(*bad.mutable_entities())[second].mutable_components())[0];
  unknown.set_type(0);
  Reject([&] { scene.Apply(bad, replica); });
  Require(replica.HasEntity(second));
  // Explicit tombstones and omitted entries have identical removal semantics.
  const auto third = source.CreateEntity();
  source.CreateComponent(third, gz::sim::components::Name("third"));
  scene.Apply(snapshot(), replica);
  source.RemoveComponent<gz::sim::components::Name>(third);
  scene.Apply(snapshot(), replica);
  Require(scene.Contains(third)); // An empty entity must still exist.
  Require(replica.HasEntity(third));
  Require(replica.Component<gz::sim::components::Name>(third) == nullptr);
  source.ClearRemovedComponents();
  source.CreateComponent(third, gz::sim::components::Name("returned"));
  scene.Apply(snapshot(), replica);
  Require(replica.Component<gz::sim::components::Name>(third)->Data() == "returned");
  source.RequestRemoveEntity(third, false);
  scene.Apply(snapshot(), replica); // Explicit entity tombstone, not omission.
  Require(!scene.Contains(third));
  replica.ProcessRemoveEntityRequests();
  Require(!replica.HasEntity(third));
  source.ProcessRemoveEntityRequests();
  scene.Apply(snapshot(), replica);
  Require(scene.Contains(second));
  // Whole-input rejection must precede any removal or membership replacement.
  bad = snapshot();
  bad.mutable_entities()->erase(second);
  (*bad.mutable_entities())[999].set_id(1000);
  Reject([&] { scene.Apply(bad, replica); });
  Require(scene.Contains(second) && replica.HasEntity(second));
  gz::sim::EntityComponentManager cameraEcm;
  const auto world = cameraEcm.CreateEntity();
  cameraEcm.CreateComponent(world, gz::sim::components::World());
  cameraEcm.CreateComponent(world, gz::sim::components::Name("campus"));
  const auto model = cameraEcm.CreateEntity();
  cameraEcm.CreateComponent(model, gz::sim::components::Model());
  cameraEcm.CreateComponent(model, gz::sim::components::Name("drone"));
  cameraEcm.CreateComponent(model, gz::sim::components::ParentEntity(world));
  const auto link = cameraEcm.CreateEntity();
  cameraEcm.CreateComponent(link, gz::sim::components::Link());
  cameraEcm.CreateComponent(link, gz::sim::components::Name("camera_link"));
  cameraEcm.CreateComponent(link, gz::sim::components::ParentEntity(model));
  const auto camera = cameraEcm.CreateEntity();
  cameraEcm.CreateComponent(camera, gz::sim::components::Sensor());
  cameraEcm.CreateComponent(camera, gz::sim::components::Name("IMX214"));
  cameraEcm.CreateComponent(camera, gz::sim::components::ParentEntity(link));
  sdf::Sensor sensor;
  Require(CameraTopic(sensor, camera, cameraEcm, "/image") ==
      "/world/campus/model/drone/link/camera_link/sensor/IMX214/image");
  sensor.SetTopic("depth_camera");
  Require(CameraTopic(sensor, camera, cameraEcm, "/depth_image") == "/depth_camera");
  sensor.SetTopic("/explicit/depth");
  Require(CameraTopic(sensor, camera, cameraEcm, "/depth_image") == "/explicit/depth");
  Require(!HasComponent<gz::sim::components::GpuLidar>(cameraEcm));
  sdf::Sensor lidar;
  lidar.SetName("real-nondefault-lidar");
  lidar.SetType(sdf::SensorType::GPU_LIDAR);
  cameraEcm.CreateComponent(camera, gz::sim::components::GpuLidar(lidar));
  Require(HasComponent<gz::sim::components::GpuLidar>(cameraEcm));
  gz::msgs::SerializedStateMap cameraMap;
  cameraEcm.State(cameraMap, {}, {}, true);
  Require(google::protobuf::util::MessageDifferencer::Equivalent(
      cameraMap, sceneCapture.Capture(cameraEcm)));
  // A blocked transport callback must not hold the submission lock. Only the
  // newest pending COMPLETE frame remains; source timestamps stay unchanged.
  std::promise<void> entered, release, lastConsumed;
  auto enteredFuture = entered.get_future();
  auto releaseFuture = release.get_future().share();
  auto lastFuture = lastConsumed.get_future();
  std::vector<std::int64_t> seen;
  LatestSceneWorker worker([&](const CapturedScene &frame) {
    if (frame.identity.sequence == 1) {
      entered.set_value();
      if (releaseFuture.wait_for(std::chrono::seconds(2)) != std::future_status::ready)
        throw std::runtime_error("TEST_RELEASE_TIMEOUT");
    }
    Require(frame.identity.sourceUnixNs == 1000 + frame.identity.sequence);
    seen.push_back(frame.identity.sequence);
    if (frame.identity.sequence == 3) lastConsumed.set_value();
  });
  // 功能：
  //   生成只用于队列测试的帧，不宣称该空场景来自实际飞行。
  // 输入：
  //   sequence：期望观察到的帧顺序。
  // 输出：
  //   frame：带固定关联时间的测试帧。
  auto capture = [](std::int64_t sequence) {
    auto frame = std::make_unique<CapturedScene>();
    frame->identity = {std::string(64, 'a'), "", sequence, 1000+sequence, sequence};
    return frame;
  };
  worker.Submit(capture(1));
  Require(enteredFuture.wait_for(std::chrono::seconds(1)) == std::future_status::ready);
  worker.Submit(capture(2));
  worker.Submit(capture(3));
  release.set_value();
  Require(lastFuture.wait_for(std::chrono::seconds(1)) == std::future_status::ready);
  worker.Stop();
  Require((seen == std::vector<std::int64_t>{1, 3}));
  Require(!worker.Failed());
  Require(worker.Summary().find("\"superseded\":1") != std::string::npos);
  Require(worker.Summary().find("\"consumed\":2") != std::string::npos);
  Reject([&] { worker.Submit(capture(4)); });
  worker.Stop(); // Idempotent stop, no detached callback.
  LatestSceneWorker broken([](const CapturedScene &) { throw std::runtime_error("fixture"); });
  broken.Submit(capture(1));
  const auto failureDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
  while (!broken.Failed() && std::chrono::steady_clock::now() < failureDeadline)
    std::this_thread::yield();
  Require(broken.Failed());
  Reject([&] { broken.Submit(capture(2)); });
  broken.Stop();
  Reject([] { LatestSceneWorker invalid({}); });
  // 两个关闭者必须共同等待同一在途回调，不能对 std::thread 并发 join。
  std::promise<void> concurrentEntered, concurrentRelease;
  auto concurrentEnteredFuture = concurrentEntered.get_future();
  auto concurrentReleaseFuture = concurrentRelease.get_future().share();
  LatestSceneWorker concurrent([&](const CapturedScene &) {
    concurrentEntered.set_value();
    if (concurrentReleaseFuture.wait_for(std::chrono::seconds(2)) != std::future_status::ready)
      throw std::runtime_error("TEST_RELEASE_TIMEOUT");
  });
  concurrent.Submit(capture(1));
  Require(concurrentEnteredFuture.wait_for(std::chrono::seconds(1)) == std::future_status::ready);
  auto stopOne = std::async(std::launch::async, [&] { concurrent.Stop(); });
  auto stopTwo = std::async(std::launch::async, [&] { concurrent.Stop(); });
  const bool waitingOne = stopOne.wait_for(std::chrono::milliseconds(20)) != std::future_status::ready;
  const bool waitingTwo = stopTwo.wait_for(std::chrono::milliseconds(20)) != std::future_status::ready;
  concurrentRelease.set_value();
  stopOne.get();
  stopTwo.get();
  Require(waitingOne && waitingTwo && !concurrent.Failed());
  LatestSceneWorker *selfPointer = nullptr;
  LatestSceneWorker selfStopping([&](const CapturedScene &) { selfPointer->Stop(); });
  selfPointer = &selfStopping;
  selfStopping.Submit(capture(1));
  const auto selfDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
  while (!selfStopping.Failed() && std::chrono::steady_clock::now() < selfDeadline)
    std::this_thread::yield();
  Require(selfStopping.Failed());
  selfStopping.Stop();
  std::cout << "full ECM creation, omitted-component/entity removal and validation passed\n";
  const int result = 0;
  return result;
}
