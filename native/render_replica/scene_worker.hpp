#pragma once

#include "replica_contract.hpp"
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <gz/msgs/serialized_map.pb.h>

namespace dronedream::replica {
struct CapturedScene {
  FrameIdentity identity;
  gz::msgs::SerializedStepMap state;
};

// Owns protobuf values only, never an ECM or physics callback. One active and
// one replaceable pending full snapshot; no FIFO backlog or detached work.
class LatestSceneWorker {
 public:
  // 功能：
  //   启动单个发布线程，最多保留一个在途和一个最新待发快照，禁止无限排队。
  // 输入：
  //   consume：处理完整快照的非空回调，只能在回调外关闭或销毁本对象。
  // 输出：
  //   worker：拥有发布线程及有界待发槽的对象。
  explicit LatestSceneWorker(std::function<void(const CapturedScene &)> consume)
      : consume(std::move(consume)) {
    if (!this->consume) throw std::runtime_error("SCENE_WORKER_CONSUMER_REQUIRED");
    thread = std::thread([this] { Run(); });
  }
  LatestSceneWorker(const LatestSceneWorker &) = delete;
  LatestSceneWorker &operator=(const LatestSceneWorker &) = delete;
  // 功能：
  //   等待正在执行的回调退出，确保不存在访问已销毁成员的后台任务。
  // 输入：
  //   无；由工作线程外的所有者销毁。
  // 输出：
  //   void：线程已回收。
  ~LatestSceneWorker() { Stop(); }

  // 功能：
  //   用最新完整场景替换待发场景，过时快照在锁外释放，不阻塞物理采样。
  // 输入：
  //   scene：独占拥有的有界场景快照。
  // 输出：
  //   void：快照进入待发槽，关闭或后台失败时抛出异常。
  void Submit(std::unique_ptr<CapturedScene> scene) {
    if (!scene || scene->state.ByteSizeLong() > kMaximumSceneBytes)
      throw std::runtime_error("SCENE_WORKER_CAPTURE_INVALID");
    // Destroy a superseded large protobuf outside the shared lock. Transport
    // work and its callback never hold this lock while the source submits.
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (closed || !failure.empty()) throw std::runtime_error("SCENE_WORKER_UNAVAILABLE");
      if (pending) ++superseded;
      ++submitted;
      pending.swap(scene);
    }
    ready.notify_one();
  }

  // 功能：
  //   查询后台发布失败，所有者应据此停止继续声称感知流健康。
  // 输入：
  //   无；读取受锁保护的失败状态。
  // 输出：
  //   failed：是否已有后台异常。
  bool Failed() const {
    std::lock_guard<std::mutex> lock(mutex);
    const bool failed = !failure.empty();
    return failed;
  }

  // 功能：
  //   丢弃待发旧图并等待在途回调完成；并发关闭串行 join，拒绝回调自我等待。
  // 输入：
  //   无；由回调外的所有者调用。
  // 输出：
  //   void：后台线程退出，重复关闭安全。
  void Stop() {
    if (activeWorker == this) throw std::runtime_error("SCENE_WORKER_SELF_STOP");
    std::unique_ptr<CapturedScene> discarded;
    {
      std::lock_guard<std::mutex> lock(mutex);
      closed = true;
      if (pending) ++discardedAtStop;
      discarded.swap(pending);
    }
    ready.notify_one();
    std::lock_guard<std::mutex> joining(joinMutex);
    if (thread.joinable()) thread.join();
  }

  // 功能：
  //   生成线程安全诊断，不把提交或丢弃计数当作物理观测成功。
  // 输入：
  //   无；当前计数及失败状态。
  // 输出：
  //   summary：有效 JSON 诊断文本。
  std::string Summary() const {
    std::lock_guard<std::mutex> lock(mutex);
    const auto summary = "{\"submitted\":" + std::to_string(submitted)
        + ",\"superseded\":" + std::to_string(superseded)
        + ",\"consumed\":" + std::to_string(consumed)
        + ",\"discarded_at_stop\":" + std::to_string(discardedAtStop)
        + ",\"closed\":" + (closed ? "true" : "false")
        + ",\"failed\":" + (!failure.empty() ? "true" : "false")
        + ",\"failure\":" + JsonText(failure) + "}";
    return summary;
  }

 private:
  // 功能：
  //   顺序消费最新快照，后台异常立即封闭发布，不能继续维持最后一帧的健康假象。
  // 输入：
  //   无；消费受条件变量保护的待发槽。
  // 输出：
  //   void：成功计数或失败原因写入共享状态。
  void Run() noexcept {
    activeWorker = this;
    for (;;) {
      std::unique_ptr<CapturedScene> scene;
      {
        std::unique_lock<std::mutex> lock(mutex);
        ready.wait(lock, [this] { return closed || pending != nullptr; });
        if (closed) return;
        scene.swap(pending);
      }
      try {
        consume(*scene);
        std::lock_guard<std::mutex> lock(mutex);
        ++consumed;
      } catch (...) {
        // A background exception stops publication; never silently keep the
        // last healthy scene alive. Diagnostics expose failure to the owner.
        std::lock_guard<std::mutex> lock(mutex);
        failure = "SCENE_BACKGROUND_PUBLICATION_FAILED";
        pending.reset();
        return;
      }
    }
  }
  std::function<void(const CapturedScene &)> consume;
  inline static thread_local const LatestSceneWorker *activeWorker = nullptr;
  mutable std::mutex mutex;
  std::mutex joinMutex;
  std::condition_variable ready;
  std::unique_ptr<CapturedScene> pending;
  bool closed = false;
  std::string failure;
  std::uint64_t submitted = 0, consumed = 0, superseded = 0, discardedAtStop = 0;
  std::thread thread;
};
}  // namespace dronedream::replica
