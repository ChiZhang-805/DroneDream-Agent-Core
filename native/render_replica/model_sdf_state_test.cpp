#include "scene_capture.hpp"
#include <cmath>
#include <google/protobuf/util/message_differencer.h>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/Pose.hh>
#include <sdf/Root.hh>

namespace dronedream::replica {
namespace {
// 功能：
//   校验模型缓存测试条件并定位失败行。
// 输入：
//   value：待断言条件。
//   line：调用位置。
// 输出：
//   void：断言失败时抛出异常。
void Check(bool value, int line = __builtin_LINE()) {
  if (!value) throw std::runtime_error("MODEL_SDF_STATE_TEST_FAILED_LINE_" + std::to_string(line));
}
// 功能：
//   逐实体、逐组件对照原始 SDK 列表和优化后的映射，确认内容与删除标记都相同。
// 输入：
//   expected：SDK 全量序列化结果。
//   actual：优化采集结果。
// 输出：
//   void：存在任何差异时抛出异常。
void Equivalent(const gz::msgs::SerializedState &expected, const gz::msgs::SerializedStateMap &actual) {
  Check(expected.entities_size() == actual.entities_size());
  for (const auto &entity : expected.entities()) {
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
}

// 功能：
//   验证 DOM 直接修改、极小变化、默认值、删除和断裂父子关系均不能误命中旧缓存。
// 输入：
//   无；建立独立 SDK 模型及 ECM。
// 输出：
//   void：全部与 SDK 完整输出一致时通过。
void TestModelSdfState() {
  sdf::Root root;
  const auto errors = root.LoadSdfString(R"(<sdf version='1.9'><model name='world_geometry'>
    <static>true</static><pose>1 2 3 0.1 0.2 0.3</pose><link name='link'>
      <inertial><mass>1.2</mass></inertial>
      <collision name='wall'><geometry><box><size>1 2 3</size></box></geometry></collision>
      <visual name='box'><geometry><box><size>1 2 3</size></box></geometry>
        <material><diffuse>1 0.25 0.5 1</diffuse></material></visual>
    </link></model></sdf>)");
  Check(errors.empty() && root.Model());
  gz::sim::EntityComponentManager source;
  const auto entity = source.CreateEntity();
  // Only ModelSdf: the filtered native read must NOT interpret empty as all.
  source.CreateComponent(entity, gz::sim::components::ModelSdf(*root.Model()));
  CompleteSceneCapture capture;
  // 功能：
  //   在每次 DOM 改动后直接对照当前完整 SDK 状态。
  // 输入：
  //   无；捕获 source 与 capture。
  // 输出：
  //   void：优化采集未丢失改动时通过。
  auto compare = [&] { Equivalent(source.State(), capture.Capture(source)); };
  compare();
  const auto first = capture.ModelStatistics();
  compare();
  Check(first.misses == 1 && capture.ModelStatistics().hits == 1);
  auto dom = source.Component<gz::sim::components::ModelSdf>(entity)->Data().Element();
  auto size = dom->GetElement("link")->GetElement("collision")->GetElement("geometry")
      ->GetElement("box")->GetElement("size")->GetValue();
  Check(size->Set(gz::math::Vector3d(1., 2., 3.125)));
  compare(); // Direct DOM mutation WITHOUT SetChanged must invalidate bytes.
  Check(capture.ModelStatistics().misses == 2);
  Check(size->Set(gz::math::Vector3d(1., 2., std::nextafter(3.125, 4.))));
  compare();
  Check(capture.ModelStatistics().misses == 3); // No approximate vector equality.
  Check(dom->GetAttribute("name")->Set(std::string("changed-model")));
  compare();
  auto mass = dom->GetElement("link")->GetElement("inertial")->GetElement("mass")->GetValue();
  Check(mass->Set(-0.)); compare();
  const auto negativeZero = capture.ModelStatistics().misses;
  Check(mass->Set(0.)); compare();
  Check(capture.ModelStatistics().misses == negativeZero + 1);
  mass->Reset(); compare(); // Unset parameters use DEFAULT, not current value.
  *mass = sdf::Param("mass", "double", "2.5", false);
  compare(); // Replacing a default is a content change even when unset.

  auto pose = dom->GetElement("pose");
  Check(pose->GetAttribute("degrees")->Set(true));
  compare(); // Pose output can depend on parent interpretation attributes.
  Check(pose->GetValue()->SetFromString("1 2 3 0.1 0.2 0.3", true));
  compare(); // Same visible attributes, changed hidden ignore-parent setting.
  auto link = dom->GetElement("link");
  auto clone = link->GetElement("collision")->Clone();
  Check(clone->GetAttribute("name")->Set(std::string("new-wall")));
  link->InsertElement(clone);
  compare(); // Child insertion, deletion and order are part of the live key.
  link->GetElement("collision")->RemoveFromParent();
  compare();
  clone->RemoveFromParent();
  compare();

  const auto other = source.CreateEntity();
  source.CreateComponent(other, gz::sim::components::Name("ordinary"));
  source.CreateComponent(entity, gz::sim::components::Pose(gz::math::Pose3d::Zero));
  compare();
  source.RemoveComponent<gz::sim::components::Pose>(entity);
  compare(); // Non-ModelSdf tombstones must not be lost in a component filter.
  Check(capture.ModelStatistics().removedBypasses > 0);
  source.ClearRemovedComponents();
  compare();
  source.RemoveComponent<gz::sim::components::ModelSdf>(entity);
  compare();
  source.ClearRemovedComponents();
  compare();
  source.CreateComponent(entity, gz::sim::components::ModelSdf(*root.Model()));
  compare();
  source.RequestRemoveEntity(entity, false);
  compare(); // Entity tombstone retains the complete payload until SDK removal.
  source.ProcessRemoveEntityRequests();
  compare();
  capture.Stop();

  // Public InsertElement() does NOT set parent by default. The SDK serializer
  // still emits later siblings, which GetNextElement() would otherwise miss.
  sdf::Root disconnected;
  Check(disconnected.LoadSdfString(R"(<sdf version='1.9'><model name='disconnected'>
    <link name='first'/><link name='last'/></model></sdf>)").empty());
  gz::sim::EntityComponentManager disconnectedSource;
  const auto brokenEntity = disconnectedSource.CreateEntity();
  disconnectedSource.CreateComponent(brokenEntity,
      gz::sim::components::ModelSdf(*disconnected.Model()));
  CompleteSceneCapture disconnectedCapture;
  // 功能：
  //   验证父引用断裂时仍能保留 SDK 输出的后续兄弟节点。
  // 输入：
  //   无；捕获断裂树测试状态与采集器。
  // 输出：
  //   void：旁路缓存后的全量结果一致时通过。
  auto disconnectedCompare = [&] {
    Equivalent(disconnectedSource.State(), disconnectedCapture.Capture(disconnectedSource));
  };
  disconnectedCompare();
  const auto brokenDom = disconnectedSource.Component<gz::sim::components::ModelSdf>(
      brokenEntity)->Data().Element();
  auto firstChild = brokenDom->GetFirstElement();
  auto laterChild = firstChild->GetNextElement();
  Check(firstChild && laterChild);
  firstChild->SetParent(nullptr);
  disconnectedCompare();
  Check(laterChild->GetAttribute("name")->Set(std::string("changed-hidden-sibling")));
  disconnectedCompare();
  Check(disconnectedCapture.ModelStatistics().treeBypasses == 2);
  firstChild->SetParent(brokenDom);
  disconnectedCompare();
  const auto restoredMisses = disconnectedCapture.ModelStatistics().misses;
  disconnectedCompare();
  Check(disconnectedCapture.ModelStatistics().misses == restoredMisses);
  Check(disconnectedCapture.ModelStatistics().hits == 1);
  disconnectedCapture.Stop();
}
}  // namespace dronedream::replica
