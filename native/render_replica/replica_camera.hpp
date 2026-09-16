#pragma once
#include <string>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/Util.hh>
#include <gz/transport/TopicUtils.hh>
#include <sdf/Sensor.hh>

namespace dronedream::replica {
// 功能：
//   判断当前实体快照中是否包含指定组件类型，找到首个后停止遍历。
// 输入：
//   Component：待查询的 Gazebo 组件类型。
//   ecm：当前场景实体管理器。
// 输出：
//   found：至少一个实体具有该组件时为真。
template<class Component>
inline bool HasComponent(gz::sim::EntityComponentManager &ecm) {
  bool found = false;
  ecm.Each<Component>([&](const gz::sim::Entity &, const Component *) {
    found = true;
    return false;
  });
  return found;
}
// 功能：
//   按实际渲染器的作用域规则解析相机话题，不依赖无作用域的旧传感器名称。
// 输入：
//   sensor：当前相机 SDF 声明。
//   entity：相机实体。
//   ecm：当前作用域和父实体结构。
//   suffix：默认传感器话题后缀。
// 输出：
//   resolved：规范绝对话题；无法规范化时为空。
inline std::string CameraTopic(const sdf::Sensor &sensor, gz::sim::Entity entity,
    const gz::sim::EntityComponentManager &ecm, const std::string &suffix) {
  const auto topic = sensor.Topic().empty()
      ? gz::sim::scopedName(entity, ecm) + suffix : sensor.Topic();
  const auto valid = gz::transport::TopicUtils::AsValidTopic(topic);
  const auto resolved = valid.empty() || valid.front() == '/' ? valid : "/" + valid;
  return resolved;
}
}  // namespace dronedream::replica
