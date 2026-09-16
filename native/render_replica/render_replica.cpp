// An opt-in sensor-only process. It never creates a physics system or sends controls.
#include "replica_contract.hpp"
#include "replica_scene.hpp"
#include "replica_camera.hpp"
#include "replica_timing.hpp"

#include <atomic>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <dlfcn.h>
#include <link.h>
#include <gz/common/Console.hh>
#include <gz/msgs/bytes.pb.h>
#include <gz/msgs/image.pb.h>
#include <gz/msgs/serialized_map.pb.h>
#include "component_registry.hpp"
#include <gz/sim/components/RenderEngineServerHeadless.hh>
#include <gz/sim/Conversions.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/EventManager.hh>
#include <gz/sim/Events.hh>
#include <gz/sim/rendering/Events.hh>
#include <gz/sim/SystemLoader.hh>
#include <gz/transport/Node.hh>

namespace {
using namespace dronedream::replica;
volatile std::sig_atomic_t interrupted = 0;
// 功能：
//   以信号安全方式请求主循环停止，不在信号处理器中释放线程或调用渲染库。
// 输入：
//   信号编号：由操作系统提供，本函数不依赖其具体值。
// 输出：
//   interrupted：置一，主循环随后执行正常清理。
void OnSignal(int) { interrupted = 1; }

struct Frame {
  FrameIdentity identity;
  gz::msgs::SerializedStepMap message;
  std::int64_t receivedUnixNs = 0;
  std::chrono::steady_clock::time_point received, validated;
};

struct AppliedFrame {
  FrameIdentity identity;
  std::int64_t receivedUnixNs = 0;
  std::chrono::steady_clock::time_point received, validated, applyBegin, sensorSubmit;
};

struct Settings {
  std::string epoch, world, topic, rgb, depth, plugin;
  std::filesystem::path receipt;
  int seconds = 0, stallMs = 0, stallEvery = 40;
};

// 功能：
//   解析显式源身份、世界、传感器主题及插件路径，拒绝未知/重复参数与主题冲突。
// 输入：
//   argc、argv：进程命令行参数。
// 输出：
//   s：经过边界校验的本次运行配置。
Settings Arguments(int argc, char **argv) {
  const std::set<std::string> known{"--epoch", "--world", "--snapshot-topic", "--rgb-topic",
      "--depth-topic", "--sensors-plugin", "--receipt", "--duration-seconds",
      "--fault-stall-ms", "--fault-every"};
  std::map<std::string, std::string> args;
  for (int i = 1; i < argc; i += 2) {
    if (i+1 >= argc || !known.count(argv[i]) || !args.emplace(argv[i], argv[i+1]).second)
      throw std::runtime_error("REPLICA_ARGUMENT_INVALID");
  }
  Settings s{args.at("--epoch"), args.at("--world"), args.at("--snapshot-topic"),
      args.at("--rgb-topic"), args.at("--depth-topic"), args.at("--sensors-plugin"),
      args.at("--receipt")};
  // 功能：
  //   解析有界非负整数，禁止尾随文本、符号和溢出。
  // 输入：
  //   key：参数名。
  //   fallback：参数缺省值。
  // 输出：
  //   number：有效整数或缺省值。
  auto number = [&](const std::string &key, int fallback) {
    if (!args.count(key)) return fallback;
    const auto value = Decimal(args.at(key));
    if (value > 1000) throw std::runtime_error("REPLICA_ARGUMENT_OUT_OF_RANGE");
    return static_cast<int>(value);
  };
  s.seconds = number("--duration-seconds", s.seconds);
  s.stallMs = number("--fault-stall-ms", s.stallMs);
  s.stallEvery = number("--fault-every", s.stallEvery);
  if (!IsDigest(s.epoch) || s.world.empty() || s.world.size() > 128
      || !s.receipt.is_absolute() || std::filesystem::exists(s.receipt)
      || s.seconds < 0 || s.seconds > 600 || s.stallMs > 1000 || s.stallMs < 0
      || s.stallEvery < 1 || s.stallEvery > 1000
      || !std::filesystem::path(s.plugin).is_absolute()
      || !std::filesystem::is_regular_file(s.plugin))
    throw std::runtime_error("REPLICA_CONFIG_INVALID");
  for (const auto &topic : {s.topic, s.rgb, s.depth})
    if (topic.empty() || topic.front() != '/' || topic.size() > 512)
      throw std::runtime_error("REPLICA_TOPIC_INVALID");
  if (s.rgb == s.depth || s.topic == s.rgb || s.topic == s.depth)
    throw std::runtime_error("REPLICA_TOPIC_COLLISION");
  const auto privatePrefix = "/dronedream/replica/" + s.epoch + "/";
  for (const auto &topic : {s.topic, s.rgb, s.depth})
    if (topic == privatePrefix + "raw_rgb" || topic == privatePrefix + "raw_depth")
      throw std::runtime_error("REPLICA_PRIVATE_TOPIC_COLLISION");
  return s;
}

class SensorReplica {
 public:
  // 功能：
  //   建立完整场景与原始图像传输；部分订阅失败时先停止回调再释放成员。
  // 输入：
  //   settings：已验证的本次运行配置。
  // 输出：
  //   replica：拥有独立渲染器与传输节点的对象，不含物理控制系统。
  explicit SensorReplica(Settings settings) : settings(std::move(settings)) {
    try {
    node = std::make_unique<gz::transport::Node>();
    rawRgb = "/dronedream/replica/" + this->settings.epoch + "/raw_rgb";
    rawDepth = "/dronedream/replica/" + this->settings.epoch + "/raw_depth";
    rgbPublisher = node->Advertise<gz::msgs::Image>(this->settings.rgb);
    depthPublisher = node->Advertise<gz::msgs::Image>(this->settings.depth);
    if (!rgbPublisher || !depthPublisher
        || !node->Subscribe(this->settings.topic, &SensorReplica::Receive, this)
        || !node->Subscribe(rawRgb, &SensorReplica::Rgb, this)
        || !node->Subscribe(rawDepth, &SensorReplica::Depth, this))
      throw std::runtime_error("REPLICA_TRANSPORT_SETUP_FAILED");
    } catch (...) {
      Stop(); // 构造失败不会调用本类析构函数，必须在 mutex 等成员仍活着时退订。
      throw;
    }
  }

  // 功能：
  //   持续应用最新完整场景并驱动传感器渲染，终止后保存真实计数和逐帧耗时诊断。
  // 输入：
  //   无；消费当前实例收到的源场景与停止信号。
  // 输出：
  //   result：无失败且 RGB、深度均实际发布过时返回零；不代表完整飞行验收。
  int Run() {
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(settings.seconds);
    while (!interrupted && (settings.seconds == 0 || std::chrono::steady_clock::now() < end)) {
      std::shared_ptr<const Frame> frame;
      {
        std::lock_guard<std::mutex> lock(mutex);
        if (failed) break;
        frame.swap(pending);
      }
      if (!frame) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      }
      try {
        Apply(*frame);
      } catch (const std::exception &error) {
        Fail(error.what());
      }
    }
    // Own-process diagnostic mapping, before renderer libraries are unloaded.
    std::ifstream maps("/proc/self/maps");
    std::ostringstream mapping;
    mapping << maps.rdbuf();
    WriteNew(settings.receipt.string() + ".loaded-maps", mapping.str());
    Stop();
    std::lock_guard<std::mutex> lock(mutex);
    std::ostringstream out;
    out << "{\"qualification_granted\":false,\"sensor_replica_only\":true,"
        << "\"source_snapshots_received\":" << received
        << ",\"source_snapshots_applied\":" << applied
        << ",\"superseded_full_snapshots\":" << superseded
        << ",\"rgb_published\":" << rgbCount << ",\"depth_published\":" << depthCount
        << ",\"late_frames_rejected\":" << late << ",\"unbound_frames_rejected\":" << unbound
        << ",\"maximum_published_source_age_ms\":" << maximumAgeMs
        << ",\"maximum_scene_apply_ms\":" << maximumSceneApplyMs
        << ",\"maximum_sensor_post_update_ms\":" << maximumPostMs
        << ",\"maximum_steady_scene_apply_ms\":" << maximumSteadySceneApplyMs
        << ",\"maximum_steady_sensor_post_update_ms\":" << maximumSteadyPostMs
        << ",\"total_changed_components_applied\":" << totalChangedComponents
        << ",\"rgb_frame_transit_ms\":" << rgbTransit.Json()
        << ",\"depth_frame_transit_ms\":" << depthTransit.Json()
        << ",\"injected_render_stalls\":" << injected.load()
        << ",\"retained_loaded_driver_mappings\":" << retainedDriverMappings
        << ",\"failed\":" << (failed ? "true" : "false")
        << ",\"failure\":" << JsonText(failure) << "}";
    WriteNew(settings.receipt.string(), out.str());
    if (failed) std::cerr << failure << '\n';
    const int result = failed || rgbCount == 0 || depthCount == 0 ? 1 : 0;
    return result;
  }

  // 功能：
  //   在正常和异常退出时都停止传感器并退订回调，避免成员销毁后的后台访问。
  // 输入：
  //   无；由拥有主循环的线程销毁。
  // 输出：
  //   void：传感器和传输资源释放。
  ~SensorReplica() { Stop(); }

 private:
  // 功能：
  //   锁存首个失败原因，后续故障不能覆盖最初诊断。
  // 输入：
  //   message：有界化保存的错误文本。
  // 输出：
  //   failed、failure：失败状态与首个原因。
  void Fail(const std::string &message) {
    std::lock_guard<std::mutex> lock(mutex);
    if (!failed) failure = message.substr(0, 512);
    failed = true;
  }

  // 功能：
  //   校验场景身份、字节摘要、时间与顺序，仅保留最新完整快照，不将到达时间当采样时间。
  // 输入：
  //   packet：物理源端发布的序列化场景。
  // 输出：
  //   pending：最新不可变快照；无效输入锁存失败。
  void Receive(const gz::msgs::Bytes &packet) {
    const auto receiveBegin = std::chrono::steady_clock::now();
    const auto receivedUnixNs = UnixNs();
    if (!accepting.load()) return;
    try {
      if (packet.data().size() > kMaximumSceneBytes || !packet.has_header())
        throw std::runtime_error("REPLICA_SCENE_SIZE_OR_HEADER");
      auto frame = std::make_shared<Frame>();
      frame->received = receiveBegin;
      frame->receivedUnixNs = receivedUnixNs;
      frame->identity = ReadIdentity(packet.header(), settings.epoch);
      if (Sha256(packet.data()) != frame->identity.sha256)
        throw std::runtime_error("REPLICA_SCENE_HASH_MISMATCH");
      if (!frame->message.ParseFromString(packet.data()) || !frame->message.has_stats()
          || !frame->message.has_state()
          || frame->message.state().entities_size() > static_cast<int>(kMaximumEntities))
        throw std::runtime_error("REPLICA_SCENE_INVALID");
      const auto &stats = frame->message.stats();
      if (!stats.has_sim_time() || stats.paused())
        throw std::runtime_error("REPLICA_SCENE_TIME_MISMATCH");
      const auto simNs = SimulationTime(stats.sim_time().sec(), stats.sim_time().nsec());
      // 必须在 SDK duration 转换之前验证，极端时间不能先溢出再来比较。
      for (const auto *time : {&stats.real_time(), &stats.pause_time(), &stats.step_size()})
        SimulationTime(time->sec(), time->nsec());
      if (simNs != frame->identity.simulationNs || frame->identity.sourceUnixNs > UnixNs())
        throw std::runtime_error("REPLICA_SCENE_TIME_MISMATCH");
      std::lock_guard<std::mutex> lock(mutex);
      if (!accepting.load() || failed) return;
      order.Admit(frame->identity);
      frame->validated = std::chrono::steady_clock::now();
      ++received;
      if (pending) ++superseded;
      pending = std::move(frame);  // At most one complete immutable pending snapshot.
    } catch (const std::exception &error) {
      Fail(error.what());
    }
  }

  // 功能：
  //   只在副本重定向明确选中的摄像头，保留实际内参；配置漂移或多摄像头歧义立即拒绝。
  // 输入：
  //   expected：源端明确选中的主题。
  //   raw：副本专用中继输入主题。
  //   suffix：该传感器类型的默认主题后缀。
  // 输出：
  //   void：副本组件主题重定向并登记配置，物理源模型不被改写。
  template<class Component>
  void RedirectCameras(const std::string &expected, const std::string &raw,
                       const std::string &suffix) {
    unsigned matched = 0;
    ecm.Each<Component>([&](const gz::sim::Entity &id, Component *component) {
      auto sensor = component->Data();
      if (CameraTopic(sensor, id, ecm, suffix) != expected)
        throw std::runtime_error("REPLICA_UNSELECTED_CAMERA");
      if (++matched > 1) throw std::runtime_error("REPLICA_DUPLICATE_CAMERA_TOPIC");
      // A sensor reconstructed from Gazebo messages has no original Element().
      // ToElement serializes its actual camera intrinsics and current values.
      sdf::Errors errors;
      const auto element = sensor.ToElement(errors);
      if (!element || !errors.empty()) throw std::runtime_error("REPLICA_CAMERA_SDF_INVALID");
      const auto serialized = element->ToString("");
      const auto previous = cameraContracts.find(id);
      if (previous != cameraContracts.end() && previous->second != serialized)
        throw std::runtime_error("REPLICA_CAMERA_CONFIGURATION_CHANGED");
      cameraContracts[id] = serialized;
      sensor.SetTopic(raw);
      component->Data() = sensor; // Replica only; authoritative camera SDF is untouched.
      return true;
    });
  }

  // 功能：
  //   把完整源场景应用到独立传感器进程；保留实际几何、时间和内参，不制造虚拟飞行进度。
  // 输入：
  //   frame：经身份与顺序校验的完整场景。
  // 输出：
  //   void：更新副本并触发当前场景成像，过期帧直接丢弃。
  void Apply(const Frame &frame) {
    if (!Fresh(frame.identity, UnixNs())) {
      std::lock_guard<std::mutex> lock(mutex);
      ++late;
      return;
    }
    const auto sceneBegin = std::chrono::steady_clock::now();
    sceneState.Apply(frame.message.state(), ecm,
        {gz::sim::components::Camera::typeId, gz::sim::components::DepthCamera::typeId});
    totalChangedComponents += sceneState.LastComponentsApplied();
    const auto sceneMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - sceneBegin).count();
    maximumSceneApplyMs = std::max(maximumSceneApplyMs, sceneMs);
    if (applied >= 5) maximumSteadySceneApplyMs = std::max(maximumSteadySceneApplyMs, sceneMs);
    for (auto it = cameraContracts.begin(); it != cameraContracts.end();)
      if (!sceneState.Contains(it->first)) it = cameraContracts.erase(it); else ++it;
    const auto world = ecm.EntityByComponents(gz::sim::components::World(),
        gz::sim::components::Name(settings.world));
    if (world == gz::sim::kNullEntity) throw std::runtime_error("REPLICA_WORLD_IDENTITY_MISSING");
    auto headless = ecm.Component<gz::sim::components::RenderEngineServerHeadless>(world);
    if (headless) headless->Data() = true;
    else ecm.CreateComponent(world, gz::sim::components::RenderEngineServerHeadless(true));
    RedirectCameras<gz::sim::components::Camera>(settings.rgb, rawRgb, "/image");
    RedirectCameras<gz::sim::components::DepthCamera>(settings.depth, rawDepth, "/depth_image");
    // Additional GPU streams need their own original-time relay and explicit
    // admission contract. Do not silently publish them from the replica.
    if (HasComponent<gz::sim::components::RgbdCamera>(ecm)
        || HasComponent<gz::sim::components::GpuLidar>(ecm)
        || HasComponent<gz::sim::components::SegmentationCamera>(ecm)
        || HasComponent<gz::sim::components::ThermalCamera>(ecm)
        || HasComponent<gz::sim::components::BoundingBoxCamera>(ecm)
        || HasComponent<gz::sim::components::WideAngleCamera>(ecm))
      throw std::runtime_error("REPLICA_UNSUPPORTED_RENDER_SENSOR");

    if (!plugin) {
      sdf::Plugin description;
      description.SetFilename(settings.plugin);
      description.SetName("gz::sim::systems::Sensors");
      if (!description.InsertContent("<render_engine>ogre2</render_engine>"))
        throw std::runtime_error("REPLICA_SENSOR_CONFIG_FAILED");
      plugin = loader.LoadPlugin(description);
      if (!plugin) throw std::runtime_error("REPLICA_SENSOR_PLUGIN_FAILED");
      auto configure = (*plugin)->QueryInterface<gz::sim::ISystemConfigure>();
      update = (*plugin)->QueryInterface<gz::sim::ISystemUpdate>();
      post = (*plugin)->QueryInterface<gz::sim::ISystemPostUpdate>();
      if (!configure || !update || !post) throw std::runtime_error("REPLICA_SENSOR_INTERFACE_MISSING");
      configure->Configure(world, description.ToElement(), ecm, events);
      bootConnection = events.Connect<gz::sim::events::SceneUpdate>([this] {
        firstSceneApplied.store(true);
      });
      // Start the graphics context with an EMPTY ECM. Sensors initializes
      // asynchronously and may skip the initial PostUpdate. Retrying the real
      // EachNew entities can either lose them or instantiate them twice.
      // Empty startup has no cameras and cannot publish an observation.
      gz::sim::EntityComponentManager empty;
      const auto bootstrapInfo = gz::sim::convert<gz::sim::UpdateInfo>(frame.message.stats());
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
      while (!firstSceneApplied.load() && !interrupted) {
        if (std::chrono::steady_clock::now() >= deadline)
          throw std::runtime_error("REPLICA_SCENE_BOOTSTRAP_TIMEOUT");
        events.Emit<gz::sim::events::ForceRender>();
        post->PostUpdate(bootstrapInfo, empty);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      }
      if (!firstSceneApplied.load()) throw std::runtime_error("REPLICA_STARTUP_INTERRUPTED");
      bootConnection.reset();
      if (settings.stallMs) {
        faultConnection = events.Connect<gz::sim::events::PreRender>([this] {
          if (++renderCount % settings.stallEvery == 0) {
            ++injected;
            std::this_thread::sleep_for(std::chrono::milliseconds(settings.stallMs));
          }
        });
      }
    }
    const auto info = gz::sim::convert<gz::sim::UpdateInfo>(frame.message.stats());
    const auto postBegin = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> lock(mutex);
      clocks.emplace(frame.identity.simulationNs, AppliedFrame{frame.identity,
          frame.receivedUnixNs, frame.received, frame.validated, sceneBegin, postBegin});
      while (clocks.size() > 256) clocks.erase(clocks.begin());
    }
    update->Update(info, ecm);
    post->PostUpdate(info, ecm); // May block this replica, never the physics process.
    const auto postMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - postBegin).count();
    maximumPostMs = std::max(maximumPostMs, postMs);
    if (applied >= 5) maximumSteadyPostMs = std::max(maximumSteadyPostMs, postMs);
    ecm.ClearRemovedComponents();
    ecm.ClearNewlyCreatedEntities();
    ecm.ProcessRemoveEntityRequests();
    ++applied;
  }

  // 功能：
  //   将真实 RGB 成像送入共同的来源、时效和顺序校验链。
  // 输入：
  //   image：传感器插件发布的 RGB 图像。
  // 输出：
  //   void：通过统一中继处理。
  void Rgb(const gz::msgs::Image &image) { Relay(image, true); }
  // 功能：
  //   将真实深度成像送入共同的来源、时效和顺序校验链。
  // 输入：
  //   image：传感器插件发布的深度图。
  // 输出：
  //   void：通过统一中继处理。
  void Depth(const gz::msgs::Image &image) { Relay(image, false); }
  // 功能：
  //   将实际图像绑定到产生它的精确场景时间，拒绝过期、未绑定和重复/倒退帧。
  // 输入：
  //   image：原始传感器图像，不能携带伪造的场景身份字段。
  //   rgb：真代表 RGB，假代表深度流。
  // 输出：
  //   void：有效图像携原始身份发布并统计，不以处理完成时间刷新帧龄。
  void Relay(const gz::msgs::Image &image, bool rgb) {
    const auto imageReceived = std::chrono::steady_clock::now();
    const auto imageReceivedUnixNs = UnixNs();
    std::lock_guard<std::mutex> relay(relayMutex);
    if (!accepting.load()) return;
    if (!image.has_header() || !image.header().has_stamp()
        || image.data().size() > kMaximumSceneBytes) { Fail("REPLICA_IMAGE_INVALID"); return; }
    const auto &stamp = image.header().stamp();
    std::int64_t sim;
    try { sim = SimulationTime(stamp.sec(), stamp.nsec()); }
    catch (const std::exception &error) { Fail(error.what()); return; }
    for (const auto &field : image.header().data())
      if (field.key().rfind("dronedream_scene_", 0) == 0) {
        Fail("REPLICA_IMAGE_IDENTITY_PREPOPULATED"); return;
      }
    AppliedFrame source;
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (failed) return;
      const auto found = clocks.find(sim);
      if (found == clocks.end()) { ++unbound; return; }
      source = found->second;
    }
    const auto &identity = source.identity;
    auto &lastSimulation = rgb ? lastRgbSimulation : lastDepthSimulation;
    if (sim <= lastSimulation) {
      std::lock_guard<std::mutex> lock(mutex);
      ++unbound;
      return;
    }
    // Exposure is the scene source time, not arrival at this relay. Slow GPU
    // output cannot rejuvenate old scene geometry, even with current telemetry.
    if (!Fresh(identity, UnixNs())) { std::lock_guard<std::mutex> lock(mutex); ++late; return; }
    auto output = image;
    auto *header = output.mutable_header();
    Field(header, "dronedream_scene_epoch", identity.epoch);
    Field(header, "dronedream_scene_sha256", identity.sha256);
    Field(header, "dronedream_scene_sequence", std::to_string(identity.sequence));
    Field(header, "dronedream_scene_source_unix_ns", std::to_string(identity.sourceUnixNs));
    Field(header, "dronedream_scene_simulation_ns", std::to_string(identity.simulationNs));
    const auto now = UnixNs();
    if (!Fresh(identity, now)) { std::lock_guard<std::mutex> lock(mutex); ++late; return; }
    std::lock_guard<std::mutex> lock(mutex);
    if (failed || !accepting.load()) return;
    if (!(rgb ? rgbPublisher : depthPublisher).Publish(output)) {
      failed = true;
      failure = "REPLICA_IMAGE_PUBLICATION_FAILED";
      return;
    }
    lastSimulation = sim;
    if (rgb) ++rgbCount; else ++depthCount;
    maximumAgeMs = std::max(maximumAgeMs, (now - identity.sourceUnixNs)/1e6);
    // 功能：
    //   换算同一图像各阶段的单调时钟间隔，不能拼接不同图像的独立分位数。
    // 输入：
    //   duration：同一图像两阶段之间的间隔。
    // 输出：
    //   milliseconds：对应毫秒值。
    const auto milliseconds = [](auto duration) {
      return std::chrono::duration<double, std::milli>(duration).count();
    };
    (rgb ? rgbTransit : depthTransit).Add({
      (source.receivedUnixNs - identity.sourceUnixNs)/1e6,
      milliseconds(source.validated - source.received),
      milliseconds(source.applyBegin - source.validated),
      milliseconds(source.sensorSubmit - source.applyBegin),
      milliseconds(imageReceived - source.sensorSubmit),
      (imageReceivedUnixNs - identity.sourceUnixNs)/1e6,
    });
  }

  // 功能：
  //   关闭接收、停止渲染并退订全部回调；异常路径也保留已加载驱动映射直到进程退出。
  // 输入：
  //   无；只能由主循环拥有者调用。
  // 输出：
  //   void：所有成员仍存活时完成资源清理，重复调用无副作用。
  void Stop() {
    if (!accepting.exchange(false)) return;
    // WSL D3D12 的已加载核心可能仍拥有驱动线程。只保留已有映射，不加载新库或跳过析构。
    dl_iterate_phdr([](dl_phdr_info *info, std::size_t, void *context) {
      const std::filesystem::path path(info->dlpi_name);
      if (path.filename() != "libd3d12core.so") return 0;
      auto handle = dlopen(info->dlpi_name, RTLD_NOW | RTLD_NOLOAD | RTLD_NODELETE);
      if (!handle) return 1;
      ++*static_cast<unsigned int *>(context);
      dlclose(handle);
      return 0;
    }, &retainedDriverMappings);
    if (plugin) events.Emit<gz::sim::events::Stop>();
    bootConnection.reset();
    faultConnection.reset();
    plugin.reset();
    update = nullptr;
    post = nullptr;
    if (node) {
      node->Unsubscribe(settings.topic);
      node->Unsubscribe(rawRgb);
      node->Unsubscribe(rawDepth);
      node.reset();
    }
  }

  Settings settings;
  std::unique_ptr<gz::transport::Node> node;
  gz::transport::Node::Publisher rgbPublisher, depthPublisher;
  std::string rawRgb, rawDepth;
  std::mutex mutex, relayMutex;
  std::int64_t lastRgbSimulation = -1, lastDepthSimulation = -1;
  std::atomic<bool> accepting{true};
  bool failed = false;
  std::string failure;
  SourceOrder order;
  std::shared_ptr<const Frame> pending;
  std::map<std::int64_t, AppliedFrame> clocks;
  ImageTransitTimings rgbTransit, depthTransit;
  std::uint64_t received = 0, superseded = 0, applied = 0;
  std::uint64_t rgbCount = 0, depthCount = 0, late = 0, unbound = 0;
  double maximumAgeMs = 0;
  double maximumSceneApplyMs = 0, maximumPostMs = 0;
  double maximumSteadySceneApplyMs = 0, maximumSteadyPostMs = 0;
  std::size_t totalChangedComponents = 0;
  unsigned int retainedDriverMappings = 0;
  gz::sim::EntityComponentManager ecm;
  gz::sim::EventManager events;
  gz::sim::SystemLoader loader;
  std::optional<gz::sim::SystemPluginPtr> plugin;
  gz::sim::ISystemUpdate *update = nullptr;
  gz::sim::ISystemPostUpdate *post = nullptr;
  gz::common::ConnectionPtr faultConnection;
  gz::common::ConnectionPtr bootConnection;
  std::atomic<bool> firstSceneApplied{false};
  std::uint64_t renderCount = 0;
  std::atomic<std::uint64_t> injected{0};
  CompleteScene sceneState;
  std::map<gz::sim::Entity, std::string> cameraContracts;
};
}

// 功能：
//   启动仅传感器渲染的独立进程，沿当前运行资源路径解析模型，失败明确返回非零。
// 输入：
//   argc、argv：源身份、主题及插件等显式配置参数。
// 输出：
//   result：运行状态码，不作为飞行成功证明。
int main(int argc, char **argv) {
  gz::common::Console::SetVerbosity(3);
  std::signal(SIGINT, OnSignal);
  std::signal(SIGTERM, OnSignal);
  try {
    // Unlike gz sim's server runner, a standalone Sensors process has not
    // registered model:// and mesh resource roots with Gazebo Common yet.
    // Use exactly the launcher's run-local resource search order.
    gz::sim::addResourcePaths();
    SensorReplica replica(Arguments(argc, argv));
    const int result = replica.Run();
    return result;
  } catch (const std::exception &error) {
    std::cerr << "Sensor replica stopped: " << error.what() << '\n';
    return 1;
  }
}
