// Read-only relay: unchanged native camera pixels plus a conservative PRE-update clock.
#include "capture_clock.hpp"
#include <chrono>
#include <mutex>
#include <string>
#include <array>
#include <stdexcept>
#include <algorithm>
#include <gz/msgs/image.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/System.hh>
#include <gz/transport/Node.hh>

namespace dronedream {
class NativeCameraClock final : public gz::sim::System,
    public gz::sim::ISystemConfigure, public gz::sim::ISystemPreUpdate {
 public:
  // 功能：固定本次运行身份及输入输出主题；不修改传感器、地图或飞控。
  // 输入：SDF 配置和 Gazebo SDK 参数。输出：完成两个只读原生相机订阅。
  void Configure(const gz::sim::Entity &, const std::shared_ptr<const sdf::Element> &sdf,
      gz::sim::EntityComponentManager &, gz::sim::EventManager &) override {
    if (!sdf || configured) throw std::runtime_error("NATIVE_CAMERA_CLOCK_CONFIG_INVALID");
    epoch = sdf->Get<std::string>("epoch");
    if (epoch.size() != 64 || !std::all_of(epoch.begin(), epoch.end(), [](char c) {
      return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); }))
      throw std::runtime_error("NATIVE_CAMERA_CLOCK_EPOCH_INVALID");
    for (std::size_t i = 0; i < 2; ++i) {
      const auto kind = i == 0 ? "rgb" : "depth";
      inputs[i] = sdf->Get<std::string>(std::string(kind)+"_input");
      outputs[i] = sdf->Get<std::string>(std::string(kind)+"_output");
      if (inputs[i].empty() || outputs[i].empty() || inputs[i].front() != '/' ||
          outputs[i].front() != '/' || inputs[i].size() > 512 || outputs[i].size() > 512)
        throw std::runtime_error("NATIVE_CAMERA_CLOCK_TOPIC_INVALID");
    }
    if (inputs[0] == inputs[1] || outputs[0] == outputs[1] ||
        std::find(inputs.begin(), inputs.end(), outputs[0]) != inputs.end() ||
        std::find(inputs.begin(), inputs.end(), outputs[1]) != inputs.end())
      throw std::runtime_error("NATIVE_CAMERA_CLOCK_TOPIC_ALIAS");
    for (std::size_t i = 0; i < 2; ++i) {
      publishers[i] = node.Advertise<gz::msgs::Image>(outputs[i]);
      if (!publishers[i] || !node.Subscribe<gz::msgs::Image>(inputs[i],
          [this, i](const gz::msgs::Image &image) { Relay(i, image); }))
        throw std::runtime_error("NATIVE_CAMERA_CLOCK_TRANSPORT_FAILED");
    }
    configured = true;
  }
  // 功能：在 Sensors 的 Update/PostUpdate 渲染前保存同一 simTime 的双钟下界。
  // 输入：本次物理 tick；ECM 不读取也不修改。输出：有界时钟记录，无位姿或控制输出。
  void PreUpdate(const gz::sim::UpdateInfo &info, gz::sim::EntityComponentManager &) override {
    std::lock_guard<std::mutex> guard(mutex);
    clock.Add(std::chrono::duration_cast<std::chrono::nanoseconds>(info.simTime).count(),
              Unix(), Steady(), info.paused);
  }
 private:
  // 功能：读取实际墙钟。输入：无。输出：Unix 纳秒，不代表接收图像的曝光时间。
  static std::int64_t Unix() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
  }
  // 功能：读取单调钟以检查墙钟跳变。输入：无。输出：单调纳秒。
  static std::int64_t Steady() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
  }
  // 功能：添加独立来源元数据，不改原生 stamp 或像素。输入：报头与键值。输出：void。
  static void Field(gz::msgs::Header *header, const std::string &key, const std::string &value) {
    auto *field = header->add_data(); field->set_key("dronedream_scene_"+key); field->add_value(value);
  }
  // 功能：精确匹配原生帧的仿真 tick，传输前后复核期限，拒绝混源、重复和未知来源。
  // 输入：RGB/深度流序号及原始消息。输出：同像素新主题；超时或缺失时不发布。
  void Relay(std::size_t i, const gz::msgs::Image &image) {
    if (!image.has_header() || !image.header().has_stamp() ||
        image.data().empty() || image.data().size() > 64*1024*1024) return;
    const auto &stamp = image.header().stamp();
    if (stamp.sec() < 0 || stamp.sec() > 9000000000LL || stamp.nsec() < 0 || stamp.nsec() >= 1000000000) return;
    const std::int64_t sim = stamp.sec()*1000000000LL+stamp.nsec();
    for (const auto &field : image.header().data())
      if (field.key().find("dronedream_scene_") == 0) return;
    std::lock_guard<std::mutex> guard(mutex);
    const auto source = clock.Find(sim, Unix(), Steady());
    if (!source || sim <= lastImages[i]) return;
    gz::msgs::Image output(image);
    auto *header = output.mutable_header();
    Field(header, "basis", "native-preupdate");
    Field(header, "epoch", epoch); Field(header, "sha256", epoch);
    Field(header, "sequence", std::to_string(sim+1));
    Field(header, "simulation_ns", std::to_string(sim));
    Field(header, "source_unix_ns", std::to_string(source->unixNs));
    if (!clock.Find(sim, Unix(), Steady())) return;
    if (publishers[i].Publish(output)) lastImages[i] = sim;
  }
  // Node 在时钟/锁销毁前析构，停止订阅；其回调不会持有 ECM 或执行器。
  std::mutex mutex;
  CaptureClock clock;
  std::string epoch;
  bool configured = false;
  std::array<std::int64_t, 2> lastImages{{-1, -1}};
  std::array<std::string, 2> inputs, outputs;
  std::array<gz::transport::Node::Publisher, 2> publishers;
  gz::transport::Node node;
};
}
GZ_ADD_PLUGIN(dronedream::NativeCameraClock, gz::sim::System,
  gz::sim::ISystemConfigure, gz::sim::ISystemPreUpdate)
GZ_ADD_PLUGIN_ALIAS(dronedream::NativeCameraClock, "dronedream::NativeCameraClock")
