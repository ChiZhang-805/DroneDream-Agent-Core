#pragma once

#include "replica_contract.hpp"
#include <any>
#include <map>
#include <sstream>
#include <type_traits>
#include <unordered_set>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/components/Model.hh>
#include <sdf/Element.hh>
#include <sdf/Param.hh>

namespace dronedream::replica {
// Memoize only expensive ModelSdf XML, after a complete LIVE DOM content read.
// The key is compared byte-for-byte, not a pointer, change flag or lossy hash.
// All output is produced by the installed SDK serializer; no XML is rewritten.
// This class has one capture-worker owner. Only owned bytes survive Read().
class ModelSdfState {
 public:
  using Entities = std::unordered_set<gz::sim::Entity>;
  struct Statistics {
    std::uint64_t hits = 0, misses = 0, removedBypasses = 0, treeBypasses = 0;
  };

  // 功能：
  //   完整读取实体，仅复用内容逐字节相等的 ModelSdf 序列化结果；删除期间走 SDK 全量路径。
  // 输入：
  //   ecm：本次 PostUpdate 内稳定的源状态，只读且不能并发修改。
  //   entities：本线程分片的全部实体，空集合代表不读任何实体。
  // 输出：
  //   state：拥有独立字节的完整分片，不能把缓存命中当作新的物理样本。
  gz::msgs::SerializedState Read(const gz::sim::EntityComponentManager &ecm, const Entities &entities) {
    if (entities.empty()) return {};
    // The SDK's type filter also filters tombstones. During removal, preserve
    // its FULL output instead of guessing removed types from live membership.
    if (ecm.HasRemovedComponents()) {
      Clear();
      ++statistics.removedBypasses;
      return ecm.State(entities);
    }
    Entities ordinary = entities;
    std::map<gz::sim::Entity, const gz::sim::components::ModelSdf *> models;
    for (auto entity : entities) {
      if (const auto *model = ecm.Component<gz::sim::components::ModelSdf>(entity)) {
        ordinary.erase(entity);
        models.emplace(entity, model);
      }
    }
    for (auto it = entries.begin(); it != entries.end();) {
      if (models.count(it->first)) { ++it; continue; }
      retainedBytes -= it->second.key.size() + it->second.xml.size();
      it = entries.erase(it);
    }
    auto state = ordinary.empty() ? gz::msgs::SerializedState{} : ecm.State(ordinary);
    for (const auto &[entity, model] : models) {
      auto types = ecm.ComponentTypes(entity);
      if (types.erase(gz::sim::components::ModelSdf::typeId) != 1)
        throw std::runtime_error("SCENE_MODEL_MEMBERSHIP_CHANGED");
      if (types.empty()) types.insert(0); // An empty SDK filter means ALL types.
      auto part = ecm.State({entity}, types);
      if (part.entities_size() != 1 || part.entities(0).id() != entity)
        throw std::runtime_error("SCENE_MODEL_ENTITY_INVALID");
      auto *message = part.mutable_entities(0)->add_components();
      message->set_type(gz::sim::components::ModelSdf::typeId);
      message->set_component(Xml(entity, *model));
      state.add_entities()->Swap(part.mutable_entities(0));
    }
    return state;
  }

  // 功能：
  //   读取当前分片缓存诊断，调用者必须保证没有并发 Read。
  // 输入：
  //   无；当前对象统计。
  // 输出：
  //   result：命中、未命中及旁路计数副本。
  Statistics Stats() const { const auto result = statistics; return result; }

 private:
  struct Entry { std::string key, xml; };
  struct UncacheableTree {};
  // Per worker: at most 16 MiB of keys plus SDK-generated XML together.
  static constexpr std::size_t kRetainedBytes = kMaximumSceneBytes;
  std::map<gz::sim::Entity, Entry> entries;
  std::size_t retainedBytes = 0;
  Statistics statistics;

  // 功能：
  //   释放缓存内容并同步容量账本，不重置累计诊断。
  // 输入：
  //   无；当前分片缓存。
  // 输出：
  //   void：缓存为空且保留字节数为零。
  void Clear() { entries.clear(); retainedBytes = 0; }
  // 功能：
  //   对内容指纹施加硬上限，防止大型模型使缓存无限膨胀。
  // 输入：
  //   key：当前累计指纹字节。
  // 输出：
  //   void：超限抛出异常。
  static void Capacity(const std::string &key) {
    if (key.size() > kMaximumSceneBytes)
      throw std::runtime_error("SCENE_MODEL_CONTENT_KEY_CAPACITY");
  }
  // 功能：
  //   保存算术值的精确位模式，保留负零和极小变化；指纹只在同一进程内比较。
  // 输入：
  //   key：累积指纹。
  //   value：不含指针或结构体填充的算术值。
  // 输出：
  //   key：追加位模式后的指纹。
  template<class T> static void Number(std::string &key, T value) {
    static_assert(std::is_arithmetic_v<T>);
    // Arithmetic values only, never object padding or pointers. Exact float
    // bits retain signed zero and sub-epsilon changes (no fuzzy math equality).
    key.append(reinterpret_cast<const char *>(&value), sizeof(value));
    Capacity(key);
  }
  // 功能：
  //   以长度加内容编码字符串，避免相邻字段拼接形成相同指纹。
  // 输入：
  //   key：累积指纹。
  //   value：待编码原文。
  // 输出：
  //   key：追加长度与原文字节后的指纹。
  static void Text(std::string &key, const std::string &value) {
    if (value.size() > kMaximumSceneBytes)
      throw std::runtime_error("SCENE_MODEL_CONTENT_KEY_CAPACITY");
    Number(key, static_cast<std::uint64_t>(value.size()));
    key.append(value);
    Capacity(key);
  }
  // 功能：
  //   仅在参数类型精确匹配时写入数值，不进行有损隐式类型转换。
  // 输入：
  //   key：累积指纹。
  //   value：SDK 返回的类型化值。
  // 输出：
  //   matched：类型是否匹配；匹配时 key 追加对应数值。
  template<class T> static bool Scalar(std::string &key, const std::any &value) {
    if (const auto *number = std::any_cast<T>(&value)) {
      Number(key, *number);
      return true;
    }
    return false;
  }
  // 功能：
  //   编码参数值和影响 SDK 输出的设置状态，覆盖默认值与姿态父元素解释规则。
  // 输入：
  //   key：累积指纹。
  //   param：当前实时 DOM 参数，可为空。
  // 输出：
  //   key：追加参数语义与值后的指纹。
  static void Parameter(std::string &key, const sdf::ParamPtr &param) {
    Number(key, param != nullptr);
    if (!param) return;
    Text(key, param->GetKey());
    Text(key, param->GetTypeName());
    Number(key, param->GetSet());
    Number(key, param->GetRequired());
    // Unset values print the default. Pose formatting also depends on parent
    // interpretation state; include the SDK's explicit ignore-parent flag and
    // the actual parameter parent attributes, not an assumed tree parent.
    const auto &type = param->GetTypeName();
    const bool pose = type == "pose" || type == "Pose" || type == "gz::math::Pose3d";
    if (pose) {
      const bool ignoreParent = param->IgnoresParentElementAttribute();
      Number(key, ignoreParent);
      const auto parent = param->GetParentElement();
      if (!ignoreParent && parent) {
        Number(key, static_cast<std::uint64_t>(parent->GetAttributeCount()));
        for (const auto &attribute : parent->GetAttributes()) {
          Text(key, attribute->GetKey());
          Number(key, attribute->GetSet());
          Text(key, attribute->GetAsString());
        }
      }
    }
    std::any value;
    sdf::Errors errors;
    if (param->GetSet() && param->GetAny(value, errors) && errors.empty()) {
      key.push_back('T');
      Text(key, value.type().name());
      if (Scalar<bool>(key, value) || Scalar<char>(key, value)
          || Scalar<int>(key, value) || Scalar<unsigned int>(key, value)
          || Scalar<std::uint64_t>(key, value) || Scalar<float>(key, value)
          || Scalar<double>(key, value)) return;
      if (const auto *text = std::any_cast<std::string>(&value)) {
        Text(key, *text);
        return;
      }
      if (const auto *vector = std::any_cast<gz::math::Vector3d>(&value)) {
        Number(key, vector->X()); Number(key, vector->Y()); Number(key, vector->Z());
        return;
      }
      if (const auto *color = std::any_cast<gz::math::Color>(&value)) {
        Number(key, color->R()); Number(key, color->G());
        Number(key, color->B()); Number(key, color->A());
        return;
      }
      if (const auto *position = std::any_cast<gz::math::Pose3d>(&value)) {
        Number(key, position->Pos().X()); Number(key, position->Pos().Y());
        Number(key, position->Pos().Z()); Number(key, position->Rot().W());
        Number(key, position->Rot().X()); Number(key, position->Rot().Y());
        Number(key, position->Rot().Z());
        return;
      }
    } else key.push_back('S');
    Text(key, param->GetAsString());
  }
  // 功能：
  //   有界遍历实际 DOM 内容与子节点顺序；无法完整遍历的树拒绝缓存，不能猜测其内容。
  // 输入：
  //   key：累积指纹。
  //   element：当前 DOM 元素。
  //   depth：递归深度。
  //   count：本次累计访问节点数。
  // 输出：
  //   key：追加完整元素内容后的指纹；count 同步更新。
  static void Element(std::string &key, const sdf::ElementPtr &element, std::size_t depth, std::size_t &count) {
    if (depth > 128 || ++count > 131072)
      throw std::runtime_error("SCENE_MODEL_CONTENT_TREE_CAPACITY");
    Number(key, element != nullptr);
    if (!element) return;
    const bool useInclude = sdf::PrintConfig{}.PreserveIncludes() && element->GetIncludeElement();
    Number(key, useInclude);
    if (useInclude) { Element(key, element->GetIncludeElement(), depth + 1, count); return; }
    Text(key, element->GetName());
    Number(key, element->GetExplicitlySetInFile());
    Number(key, static_cast<std::uint64_t>(element->GetAttributeCount()));
    for (const auto &attribute : element->GetAttributes()) Parameter(key, attribute);
    Parameter(key, element->GetValue());
    for (auto child = element->GetFirstElement(); child; child = child->GetNextElement()) {
      // SDK PrintValuesImpl walks the owner's child vector, but GetNextElement
      // follows the child's parent. InsertElement(child) need not set that
      // parent! Such a tree cannot be completely inspected by this public
      // iterator: bypass memoization rather than omit later siblings.
      if (child->GetParent() != element) throw UncacheableTree{};
      key.push_back('C');
      Element(key, child, depth + 1, count);
    }
    key.push_back('E');
    Capacity(key);
  }
  // 功能：
  //   以实时内容判定是否复用 XML，未命中仍由 SDK 唯一序列化；总缓存容量有界。
  // 输入：
  //   entity：当前实体标识，不单独作为缓存有效性依据。
  //   model：该实体当前 ModelSdf 组件。
  // 输出：
  //   xml：与当前完整模型内容匹配的 SDK 序列化字节。
  std::string Xml(gz::sim::Entity entity, const gz::sim::components::ModelSdf &model) {
    std::string key;
    std::size_t count = 0;
    bool cacheable = true;
    try {
      Element(key, model.Data().Element(), 0, count);
    } catch (const UncacheableTree &) {
      cacheable = false;
      ++statistics.treeBypasses;
    }
    const auto previous = entries.find(entity);
    if (cacheable && previous != entries.end() && previous->second.key == key) {
      ++statistics.hits;
      return previous->second.xml;
    }
    ++statistics.misses;
    std::ostringstream stream;
    model.Serialize(stream); // Sole authority for serialized bytes.
    auto xml = stream.str();
    if (xml.size() > kMaximumSceneBytes)
      throw std::runtime_error("SCENE_MODEL_SERIALIZATION_CAPACITY");
    if (previous != entries.end()) {
      retainedBytes -= previous->second.key.size() + previous->second.xml.size();
      entries.erase(previous);
    }
    const auto size = key.size() + xml.size();
    if (cacheable && size <= kRetainedBytes) {
      if (size > kRetainedBytes - retainedBytes) Clear();
      entries.emplace(entity, Entry{std::move(key), xml});
      retainedBytes += size;
    }
    return xml;
  }
};
}  // namespace dronedream::replica
