// Native simulated sensor with explicit world ENU tesla and PX4 wire contracts.
// Gazebo world pose is used ONLY to simulate the physical sensor here, never
// published as a learned-policy observation or fitted into estimator coordinates.
#include "magnetic_wire.hpp"
#include "magnetic_field.hpp"
#include <cmath>
#include <memory>
#include <unordered_map>
#include <gz/msgs/Utility.hh>
#include <gz/msgs/magnetometer.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sensors/MagnetometerSensor.hh>
#include <gz/sensors/SensorFactory.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/MagneticField.hh>
#include <gz/sim/components/Magnetometer.hh>
#include <gz/sim/components/World.hh>
#include <gz/transport/Node.hh>
#include <sdf/Magnetometer.hh>

namespace dronedream {
class Magnetometer final : public gz::sim::System,
                          public gz::sim::ISystemConfigure,
                          public gz::sim::ISystemPostUpdate {
  struct Sensor {
    std::unique_ptr<gz::sensors::MagnetometerSensor> native;
    gz::transport::Node::Publisher output;
    std::uint64_t sequence = 0;
  };
  gz::transport::Node node;
  gz::sensors::SensorFactory factory;
  std::unordered_map<gz::sim::Entity, Sensor> sensors;
  gz::math::Vector3d worldField;
  bool configured = false;

 public:
  // 功能：
  //   核对世界磁场、噪声单位与 PX4 接线契约；重配置先撤销旧传感器，失败不沿用旧状态。
  // 输入：
  //   entity：插件挂载的世界实体。
  //   config：当前插件 SDF 配置。
  //   ecm：读取世界和磁场组件的实体管理器。
  //   eventManager：Gazebo 生命周期接口参数，本实现不注册事件。
  // 输出：
  //   void：只在契约及磁场有效时设置 configured。
  void Configure(const gz::sim::Entity &entity,
                 const std::shared_ptr<const sdf::Element> &config,
                 gz::sim::EntityComponentManager &ecm,
                 gz::sim::EventManager &) override {
    configured = false;
    sensors.clear();
    const auto field = ecm.Component<gz::sim::components::MagneticField>(entity);
    if (!config || !ecm.Component<gz::sim::components::World>(entity) || !field ||
        !config->HasElement("wire_contract") ||
        config->Get<std::string>("wire_contract") != "px4-gz-fimex-gauss" ||
        !config->HasElement("source_noise_units") ||
        config->Get<std::string>("source_noise_units") != "gauss" ||
        !config->HasElement("field_provider") ||
        config->Get<std::string>("field_provider") != "px4-world-magnetic-model") {
      gzerr << "DRONEDREAM_MAGNETIC_CONTRACT_INVALID\n";
      return;
    }
    worldField = field->Data();
    const double strength = worldField.Length();
    if (!worldField.IsFinite() || strength < 1e-6 || strength > 1e-3 ||
        std::hypot(worldField.X(), worldField.Y()) < 1e-6) {
      gzerr << "DRONEDREAM_MAGNETIC_FIELD_INVALID\n";
      return;
    }
    configured = true;
  }

  // 功能：
  //   1. 按实际传感器采样率和含噪姿态计算磁场，单次转换为 PX4 接线格式并附原仿真时间。
  //   2. 世界真值仅在模拟物理传感器内部使用；暂停不制造新观测，时间倒退清除旧采样调度。
  // 输入：
  //   info：Gazebo 本次更新的时间和暂停状态。
  //   ecm：当前实体、安装姿态及地理位置。
  // 输出：
  //   void：在声明话题发布新磁场样本，删除已移除传感器的缓存。
  void PostUpdate(const gz::sim::UpdateInfo &info,
                  const gz::sim::EntityComponentManager &ecm) override {
    if (info.dt < std::chrono::steady_clock::duration::zero()) {
      sensors.clear();
      return;
    }
    if (!configured || info.paused)
      return;
    ecm.Each<gz::sim::components::Magnetometer>(
      [&](const gz::sim::Entity &entity, const gz::sim::components::Magnetometer *component) {
        auto found = sensors.find(entity);
        if (found == sensors.end()) {
          sdf::Sensor description = component->Data();
          auto *magnetic = description.MagnetometerSensor();
          if (!magnetic) return true;
          // The bound PX4 x500 package declares gauss amplitudes for the
          // Harmonic wire. Convert all dimensional noise terms exactly once.
          magnetic->SetXNoise(GaussNoiseToTesla(magnetic->XNoise()));
          magnetic->SetYNoise(GaussNoiseToTesla(magnetic->YNoise()));
          magnetic->SetZNoise(GaussNoiseToTesla(magnetic->ZNoise()));
          const std::string topic = description.Topic().empty()
            ? gz::sim::scopedName(entity, ecm) + "/magnetometer" : description.Topic();
          // The standard sensor applies its declared sampling rate and noise in
          // physical tesla. Its native diagnostic topic cannot collide with PX4.
          description.SetTopic(topic + "/native_flu_tesla");
          auto native = factory.CreateSensor<gz::sensors::MagnetometerSensor>(description);
          auto output = node.Advertise<gz::msgs::Magnetometer>(topic);
          if (!native || !output) {
            gzerr << "DRONEDREAM_MAGNETIC_SENSOR_START_FAILED\n";
            return true;
          }
          native->SetWorldMagneticField(worldField);
          found = sensors.emplace(entity, Sensor{std::move(native), std::move(output), 0}).first;
        }
        auto &sensor = found->second;
        if (sensor.native->NextDataUpdateTime() > info.simTime)
          return true;
        // Use exactly the same geographic field model as PX4's GNSS-assisted
        // estimator. Do not combine Harmonic's older lookup with PX4's WMM.
        const auto geographic = gz::sim::sphericalCoordinates(entity, ecm);
        if (!geographic) return true;
        if (!geographic->IsFinite() || geographic->X() < -90 || geographic->X() > 90 ||
            geographic->Y() < -180 || geographic->Y() > 180) return true;
        sensor.native->SetWorldMagneticField(MagneticWorldEnuTesla(
          geographic->X(), geographic->Y()));
        // The package's magnetometer mount is part of this physical transform.
        sensor.native->SetWorldPose(gz::sim::worldPose(entity, ecm));
        if (!sensor.native->Update(info.simTime, false))
          return true;
        if (!sensor.native->MagneticField().IsFinite()) return true;
        gz::msgs::Magnetometer message;
        *message.mutable_header()->mutable_stamp() = gz::msgs::Convert(info.simTime);
        auto *frame = message.mutable_header()->add_data();
        frame->set_key("frame_id");
        frame->add_value("px4-gz-fimex-gauss");
        auto *sequence = message.mutable_header()->add_data();
        sequence->set_key("seq");
        sequence->add_value(std::to_string(++sensor.sequence));
        gz::msgs::Set(message.mutable_field_tesla(),
                     Px4GzMagneticWire(sensor.native->MagneticField()));
        sensor.output.Publish(message);
        return true;
      });
    ecm.EachRemoved<gz::sim::components::Magnetometer>(
      [&](const gz::sim::Entity &entity, const gz::sim::components::Magnetometer *) {
        sensors.erase(entity);
        return true;
      });
  }
};
}  // namespace dronedream
GZ_ADD_PLUGIN(dronedream::Magnetometer, gz::sim::System,
              gz::sim::ISystemConfigure, gz::sim::ISystemPostUpdate)
