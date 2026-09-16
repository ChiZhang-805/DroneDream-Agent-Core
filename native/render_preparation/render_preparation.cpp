// Render-thread-only preparation. No sensor timestamps or flight commands are
// read or changed. A source-matched cache is an optimization, not qualification.
#include "cache_contract.hpp"
#include <OgreDataStream.h>
#include <OgreGpuProgramManager.h>
#include <OgreHlmsDiskCache.h>
#include <OgreHlmsManager.h>
#include <OgreRoot.h>
#include <GL/gl.h>
#include <gz/common/Console.hh>
#include <gz/plugin/Register.hh>
#include <gz/sim/EventManager.hh>
#include <gz/sim/System.hh>
#include <gz/sim/components/World.hh>
#include <gz/sim/rendering/Events.hh>
#include <chrono>
#include <cstdio>
#include <map>
#include <vector>

#if OGRE_VERSION_MAJOR != 2 || OGRE_VERSION_MINOR != 3
#error Render preparation must be built against the qualified Ogre Next 2.3 headers
#endif

namespace dronedream {
namespace fs = std::filesystem;
class BoundedStream final : public Ogre::FileHandleDataStream {
 public:
  // 功能：
  //   把本次独占创建的文件交给 Ogre 写流管理，不借用其他输出文件。
  // 输入：
  //   file：当前写入者拥有的文件流。
  // 输出：
  //   当前实例：受写入预算约束的 Ogre 流。
  explicit BoundedStream(FILE *file) : FileHandleDataStream(file, Ogre::DataStream::WRITE) {}
  // 功能：
  //   在每次写入前验证文件总字节预算，短写必须失败，不能产生看似完整的缓存。
  // 输入：
  //   bytes：本次序列化内容。
  //   count：待写字节数。
  // 输出：
  //   written：实际完整写入的字节数。
  size_t write(const void *bytes, size_t count) override {
    if (count > cache::MaxFileBytes || tell() > cache::MaxFileBytes - count)
      throw std::runtime_error("cache serialization exceeds budget");
    const auto written = FileHandleDataStream::write(bytes, count);
    if (written != count) throw std::runtime_error("cache serialization incomplete");
    return written;
  }
};
class RenderPreparation final : public gz::sim::System, public gz::sim::ISystemConfigure {
  struct File { std::string name, hash; std::uintmax_t size; int type; };
  fs::path output, input;
  std::string identity, renderer, expectedRenderer;
  std::vector<File> files;
  gz::common::ConnectionPtr prepareConnection, finishConnection;
  bool attempted = false, ready = false, finished = false, microcodeSupported = false;
  double prepareMs = 0;

  // 功能：
  //   编译缓存准备回执，明确缓存加速不授予渲染或飞行验收资格。
  // 输入：
  //   status：当前准备或保存阶段结果。
  //   error：可选错误说明。
  //   rows：实际加载或保存文件的摘要、大小及类型。
  // 输出：
  //   receipt：包含源身份和实际渲染器身份的 XML 回执。
  std::string Receipt(const std::string &status, const std::string &error = "",
                      const std::vector<File> &rows = {}) const {
    std::ostringstream out;
    out << "<receipt status=\"" << status << "\" identity_sha256=\"" << identity
        << "\" renderer_sha256=\"" << renderer << "\" mode=\""
        << (input.empty() ? "capture" : "load") << "\" prepare_ms=\"" << prepareMs
        << "\" microcode_supported=\"" << (microcodeSupported ? "true" : "false")
        << "\" qualification_granted=\"false\" error=\"" << cache::Xml(error) << "\">";
    for (const auto &file : rows)
      out << "<file name=\"" << file.name << "\" sha256=\"" << file.hash
          << "\" bytes=\"" << file.size << "\" type=\"" << file.type << "\"/>";
    const auto receipt = out.str() + "</receipt>\n";
    return receipt;
  }
  // 功能：
  //   取得当前渲染线程已经创建的 Ogre Next 单例，禁止另建引擎或误用重复动态库实例。
  // 输入：
  //   无。
  // 输出：
  //   root：当前渲染根；尚未创建时为空。
  Ogre::Root *Root() const {
    // Gazebo ships a second physical copy of its Ogre2 engine DSO in its
    // plugin directory. Linking that engine and calling its Instance() can
    // return an UNUSED singleton. Obtain the already-created shared Next root;
    // never load/create an engine here. PreRender guarantees render ownership.
    auto *root = Ogre::Root::getSingletonPtr();
    return root;
  }
  // 功能：
  //   1. 在实际渲染线程核对硬件身份，全部输入验证完成后才按 Ogre 规定顺序加载。
  //   2. 引擎读取已验证的内存字节，不在摘要核对后重新打开可能被替换的文件。
  // 输入：
  //   无；使用本次 Configure 固定的输入清单和输出路径。
  // 输出：
  //   void：独占发布 ready.xml；失败回执不代表缓存或渲染资格通过。
  void Prepare() noexcept {
    if (attempted) return;
    auto *root = Root();
    if (!root || !root->getRenderSystem() || !root->getHlmsManager()) return;
    attempted = true;
    // Sensors treats ANY persistent PreRender connection as a forced update at
    // physics rate. Disconnect once initialized; its deferred cleanup is done
    // by the next event emission, still during the preflight camera sweep.
    prepareConnection.reset();
    const auto start = std::chrono::steady_clock::now();
    try {
      std::string graphics;
      for (auto code : {GL_VENDOR, GL_RENDERER, GL_VERSION}) {
        const auto *value = glGetString(code);
        if (!value || !*value) throw std::runtime_error("actual graphics identity unavailable");
        graphics += reinterpret_cast<const char *>(value);
        graphics += '\n';
      }
      renderer = cache::Sha(graphics);
      if (!input.empty() && renderer != expectedRenderer)
        throw std::runtime_error("actual graphics identity mismatch");
      auto *manager = Ogre::GpuProgramManager::getSingletonPtr();
      if (!manager) throw std::runtime_error("microcode manager unavailable");
      microcodeSupported = manager->canGetCompiledShaderBuffer();
      // Ogre requires this order: enable -> microcode -> HLMS. Validate ALL
      // inputs before touching the engine, not one file after a partial load.
      std::uintmax_t total = 0;
      std::vector<std::string> snapshots;
      for (const auto &file : files) {
        if (file.type == 0 && !microcodeSupported)
          throw std::runtime_error("input microcode unsupported by actual renderer");
        auto bytes = cache::ReadFile(input / file.name);
        if (bytes.size() != file.size || cache::Sha(bytes) != file.hash)
          throw std::runtime_error("cache file changed before render-thread load");
        total += file.size;
        if (total > cache::MaxTotalBytes) throw std::runtime_error("cache total size invalid");
        if (file.type != 0 && !root->getHlmsManager()->getHlms(
              static_cast<Ogre::HlmsTypes>(file.type)))
          throw std::runtime_error("cache HLMS unavailable in this engine");
        snapshots.push_back(std::move(bytes));
      }
      manager->setSaveMicrocodesToCache(true);
      for (std::size_t index = 0; index < files.size(); ++index) {
        const auto &file = files[index];
        auto &bytes = snapshots[index];
        Ogre::DataStreamPtr stream(new Ogre::MemoryDataStream(bytes.data(), bytes.size(), false, true));
        if (file.type == 0) manager->loadMicrocodeCache(stream);
        else {
          Ogre::HlmsDiskCache disk(root->getHlmsManager());
          disk.loadFrom(stream);
          disk.applyTo(root->getHlmsManager()->getHlms(static_cast<Ogre::HlmsTypes>(file.type)));
        }
        stream->close();
      }
      prepareMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
      cache::WriteNew(output / "ready.xml", Receipt("ready", "", files));
      ready = true;
    } catch (const std::exception &error) {
      gzerr << "DRONEDREAM_RENDER_PREPARATION_FAILED: " << error.what() << '\n';
      try { cache::WriteNew(output / "ready.xml", Receipt("failed", error.what())); }
      catch (...) { gzerr << "DRONEDREAM_RENDER_RECEIPT_WRITE_FAILED\n"; }
    }
  }
  // 功能：
  //   在渲染器销毁前保存实际缓存，预算、文件摘要和未改写输入均通过才发布完成回执。
  // 输入：
  //   无；读取当前 Ogre 缓存及本次已验证文件清单。
  // 输出：
  //   void：发布 finished.xml，不制造空 microcode 或成功验收标签。
  void Finish() noexcept {
    if (!ready || finished) return;
    finished = true;
    try {
      auto *root = Root();
      if (!root || !Ogre::GpuProgramManager::getSingletonPtr())
        throw std::runtime_error("render teardown preceded cache finalization");
      std::vector<File> saved;
      std::uintmax_t total = 0;
      for (int type = 0; type < Ogre::HLMS_MAX; ++type) {
        if (type == Ogre::HLMS_COMPUTE ||
            (type != 0 && !root->getHlmsManager()->getHlms(static_cast<Ogre::HlmsTypes>(type))))
          continue;
        const std::string name = type == 0 ? "microcode.bin" : "hlms-" + std::to_string(type) + ".bin";
        const auto path = output / "output" / name;
        if (type == 0 && !Ogre::GpuProgramManager::getSingleton().isCacheDirty()) {
          // Ogre saves NOTHING when clean, including a supported but empty
          // cache. Never manufacture a binary header or label an empty file.
          // Preserve a previously loaded, unchanged microcode verbatim instead.
          for (const auto &file : files) if (file.type == 0) {
            const auto bytes = cache::ReadFile(input / file.name);
            if (bytes.size() != file.size || cache::Sha(bytes) != file.hash)
              throw std::runtime_error("unchanged input microcode was modified");
            cache::WriteNew(path, bytes);
            saved.push_back(file);
            total += file.size;
            if (total > cache::MaxTotalBytes) throw std::runtime_error("cache total output exceeds budget");
          }
          continue;
        }
        FILE *handle = std::fopen(path.c_str(), "wbx");
        if (!handle) throw std::runtime_error("cache output exists or cannot be created");
        Ogre::DataStreamPtr stream(new BoundedStream(handle));
        if (type == 0) Ogre::GpuProgramManager::getSingleton().saveMicrocodeCache(stream);
        else {
          Ogre::HlmsDiskCache disk(root->getHlmsManager());
          disk.copyFrom(root->getHlmsManager()->getHlms(static_cast<Ogre::HlmsTypes>(type)));
          disk.saveTo(stream);
        }
        stream->close();
        const auto bytes = cache::Size(path);
        total += bytes;
        if (total > cache::MaxTotalBytes) throw std::runtime_error("cache total output exceeds budget");
        saved.push_back({name, cache::FileSha(path), bytes, type});
      }
      cache::WriteNew(output / "finished.xml", Receipt("complete", "", saved));
    } catch (const std::exception &error) {
      gzerr << "DRONEDREAM_RENDER_CACHE_SAVE_FAILED: " << error.what() << '\n';
      try { cache::WriteNew(output / "finished.xml", Receipt("failed", error.what())); }
      catch (...) { gzerr << "DRONEDREAM_RENDER_RECEIPT_WRITE_FAILED\n"; }
    }
  }
 public:
  // 功能：
  //   核对当前世界、输出目录、输入类型和摘要后绑定渲染事件；失败撤销部分事件连接。
  // 输入：
  //   entity：Gazebo 世界实体。
  //   config：本次缓存准备插件配置。
  //   ecm：实体组件管理器。
  //   events：Gazebo 渲染生命周期事件管理器。
  // 输出：
  //   void：有效配置发布 configured.xml，不能沿用前次配置或旧渲染器身份。
  void Configure(const gz::sim::Entity &entity,
                 const std::shared_ptr<const sdf::Element> &config,
                 gz::sim::EntityComponentManager &ecm,
                 gz::sim::EventManager &events) override {
    prepareConnection.reset();
    finishConnection.reset();
    files.clear();
    input.clear();
    renderer.clear();
    expectedRenderer.clear();
    attempted = ready = finished = microcodeSupported = false;
    prepareMs = 0;
    try {
      if (!config || !ecm.Component<gz::sim::components::World>(entity))
        throw std::runtime_error("render preparation requires a world owner");
      output = config->Get<std::string>("output");
      identity = config->Get<std::string>("identity_sha256");
      if (!output.is_absolute() || fs::is_symlink(output) || fs::is_symlink(output / "output") ||
          !fs::is_directory(output / "output") ||
          !cache::Digest(identity) || fs::exists(output / "ready.xml"))
        throw std::runtime_error("render preparation output or identity invalid");
      if (config->HasElement("input")) {
        input = config->Get<std::string>("input");
        expectedRenderer = config->Get<std::string>("renderer_sha256");
        if (!input.is_absolute() || fs::is_symlink(input) || !cache::Digest(expectedRenderer))
          throw std::runtime_error("render cache input invalid");
        std::map<int, File> ordered;
        auto mutableConfig = config->Clone();
        for (auto item = mutableConfig->GetElement("file"); item; item = item->GetNextElement("file")) {
          const int type = item->Get<int>("type");
          File file{item->Get<std::string>("name"), item->Get<std::string>("sha256"),
                    item->Get<unsigned int>("bytes"), type};
          const std::string name = type == 0 ? "microcode.bin" : "hlms-" + std::to_string(type) + ".bin";
          if (type < 0 || type >= Ogre::HLMS_MAX || type == Ogre::HLMS_COMPUTE ||
              name != file.name || !cache::Digest(file.hash) ||
              !ordered.emplace(type, file).second)
            throw std::runtime_error("invalid or duplicate render cache file");
        }
        if (ordered.empty() || (ordered.size() == 1 && ordered.count(0)))
          throw std::runtime_error("at least one HLMS cache required");
        for (const auto &entry : ordered) files.push_back(entry.second);
      }
      prepareConnection = events.Connect<gz::sim::events::PreRender>([this] { Prepare(); });
      finishConnection = events.Connect<gz::sim::events::RenderTeardown>([this] { Finish(); });
      cache::WriteNew(output / "configured.xml", Receipt("configured"));
    } catch (const std::exception &error) {
      prepareConnection.reset();
      finishConnection.reset();
      gzerr << "DRONEDREAM_RENDER_CONFIGURATION_FAILED: " << error.what() << '\n';
    }
  }
};
}
GZ_ADD_PLUGIN(dronedream::RenderPreparation, gz::sim::System, gz::sim::ISystemConfigure)
