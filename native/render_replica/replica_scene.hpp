#pragma once

#include "replica_contract.hpp"
#include <set>
#include <gz/msgs/serialized_map.pb.h>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/components/Factory.hh>

namespace dronedream::replica {
// Single replica-thread owner. SetState is a merge API, not full replacement:
// omitted components/entities MUST be removed explicitly after coalescing frames.
class CompleteScene {
 public:
  // 功能：
  //   将完整权威快照同步到只渲染的副本；显式删除被省略内容，只跳过字节相同的组件。
  // 输入：
  //   input：已校验来源的 SDK 全量场景序列化，不能传入增量快照。
  //   ecm：当前渲染线程独占的副本，不是物理仿真的 ECM。
  //   restoreReplicaOverrides：必须恢复源值的本地渲染覆盖组件类型。
  // 输出：
  //   void：副本和已应用快照更新；异常时调用方必须停止发布感知图像。
  void Apply(const gz::msgs::SerializedStateMap &input, gz::sim::EntityComponentManager &ecm,
             const std::set<gz::sim::ComponentTypeId> &restoreReplicaOverrides = {}) {
    if (input.ByteSizeLong() > kMaximumSceneBytes
        || input.entities_size() > static_cast<int>(kMaximumEntities))
      throw std::runtime_error("REPLICA_SCENE_CAPACITY");
    gz::msgs::SerializedStateMap changes;
    if (input.has_one_time_component_changes())
      changes.set_has_one_time_component_changes(input.has_one_time_component_changes());
    std::size_t componentsApplied = 0;
    for (const auto &[id, entity] : input.entities()) {
      if (id == gz::sim::kNullEntity || id != entity.id())
        throw std::runtime_error("REPLICA_ENTITY_ID_MISMATCH");
      if (entity.remove()) continue;
      const auto old = previousInput.entities().find(id);
      const bool newEntity = old == previousInput.entities().end() || old->second.remove();
      if (newEntity) (*changes.mutable_entities())[id].set_id(id);
      if (entity.components_size() > 256) throw std::runtime_error("REPLICA_COMPONENT_CAPACITY");
      for (const auto &[type, component] : entity.components()) {
        // Gazebo encodes the unsigned type hash as a signed protobuf map key.
        const auto typeId = static_cast<gz::sim::ComponentTypeId>(type);
        if (typeId != component.type() || !gz::sim::components::Factory::Instance()->HasType(typeId))
          throw std::runtime_error("REPLICA_COMPONENT_IDENTITY_UNSUPPORTED");
        if (component.remove()) continue;
        const auto oldComponent = newEntity ? nullptr : [&]() {
          const auto found = old->second.components().find(type);
          return found == old->second.components().end() ? nullptr : &found->second;
        }();
        if (!oldComponent || oldComponent->remove()
            || oldComponent->component() != component.component()
            || restoreReplicaOverrides.count(typeId)) {
          auto &changed = (*changes.mutable_entities())[id];
          changed.set_id(id);
          (*changed.mutable_components())[type] = component;
          ++componentsApplied;
        }
      }
    }
    // Validate the whole input before mutating this replica. This never touches
    // the authoritative source ECM, physics state or any model policy input.
    for (const auto &[id, oldEntity] : previousInput.entities()) {
      if (oldEntity.remove()) continue;
      const auto current = input.entities().find(id);
      if (current == input.entities().end() || current->second.remove()) {
        // Remove exactly the entities omitted by the complete authoritative
        // snapshot. Recursive removal would lose a child reparented this frame.
        ecm.RequestRemoveEntity(id, false);
      } else {
        for (const auto &[type, oldComponent] : oldEntity.components()) {
          if (oldComponent.remove()) continue;
          const auto component = current->second.components().find(type);
          if (component == current->second.components().end() || component->second.remove())
            ecm.RemoveComponent(id, static_cast<gz::sim::ComponentTypeId>(type));
        }
      }
    }
    // Full snapshots remain the wire contract. Only byte-identical source
    // components skip expensive Gazebo XML deserialization on the replica.
    // No intermediate update is assumed delivered; compare to LAST APPLIED.
    ecm.SetState(changes);
    previousInput = input;
    lastComponentsApplied = componentsApplied;
  }

  // 功能：
  //   查询最近成功应用的完整快照是否保留实体，不把待删除实体视为仍在场。
  // 输入：
  //   id：实体标识。
  // 输出：
  //   present：实体存在且没有删除标记。
  bool Contains(gz::sim::Entity id) const {
    const auto found = previousInput.entities().find(id);
    const bool present = found != previousInput.entities().end() && !found->second.remove();
    return present;
  }
  // 功能：
  //   提供最近成功应用的组件写入数量，不把无效输入的部分预检当作实际应用。
  // 输入：
  //   无；当前副本统计。
  // 输出：
  //   count：最近成功应用的组件数量。
  std::size_t LastComponentsApplied() const { const auto count = lastComponentsApplied; return count; }
 private:
  // The complete protobuf already owns the entity/component membership.
  // Do not rebuild a second tree of heap-allocated sets on every sensor frame.
  gz::msgs::SerializedStateMap previousInput;
  std::size_t lastComponentsApplied = 0;
};
}  // namespace dronedream::replica
