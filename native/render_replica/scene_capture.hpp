#pragma once

#include "replica_contract.hpp"
#include "model_sdf_state.hpp"
#include <array>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <vector>
#include <gz/msgs/serialized.pb.h>
#include <gz/msgs/serialized_map.pb.h>
#include <gz/sim/EntityComponentManager.hh>

namespace dronedream::replica {
// Full, disjoint reads of the SAME PostUpdate ECM. Threads persist across
// captures, but no ECM access survives Capture(), including on an exception.
// Never use ChangedState: component data can change without SetChanged.
class CompleteSceneCapture {
 public:
  struct Timing {
    double membershipMs = 0, readJoinMs = 0, mergeMs = 0;
    std::array<double, 4> shardReadMs{};
    struct Payload {
      std::uint64_t entity = 0, type = 0;
      std::size_t bytes = 0;
    };
    std::array<Payload, 4> largestPayloads{};
  };
  using Entities = std::unordered_set<gz::sim::Entity>;
  using Read = std::function<gz::msgs::SerializedState(
      const gz::sim::EntityComponentManager &, const Entities &)>;
  // 功能：
  //   建立有限个长期读取线程；构造失败也回收已启动的线程，不留下 ECM 访问者。
  // 输入：
  //   count：1 至 4 个分片线程。
  //   read：可选的完整分片读取器，必须保留分片全部实体且不能回调本对象。
  // 输出：
  //   capture：拥有线程和分片状态的采集对象。
  explicit CompleteSceneCapture(std::size_t count = 4, Read read = {})
      : read(std::move(read)) {
    if (count == 0 || count > 4) throw std::runtime_error("SCENE_CAPTURE_WORKER_COUNT");
    shards.resize(count);
    threads.reserve(count);
    try {
      for (std::size_t i = 0; i < count; ++i)
        threads.emplace_back([this, i] { Run(i); });
    } catch (...) {
      Stop(); // Join successfully created workers even during construction.
      throw;
    }
  }
  CompleteSceneCapture(const CompleteSceneCapture &) = delete;
  CompleteSceneCapture &operator=(const CompleteSceneCapture &) = delete;
  // 功能：
  //   等待采集结束并回收工作线程，防止销毁后继续访问仿真状态。
  // 输入：
  //   无；使用当前对象拥有的线程。
  // 输出：
  //   void：对象资源全部释放。
  ~CompleteSceneCapture() { Stop(); }

  // 功能：
  //   同步采集同一次 PostUpdate 的完整场景；所有读取完成后才返回或抛出异常。
  // 输入：
  //   input：调用期间不可被外部修改或销毁的权威 ECM。
  // 输出：
  //   result：独立拥有组件字节的完整场景快照。
  gz::msgs::SerializedStateMap Capture(const gz::sim::EntityComponentManager &input) {
    std::lock_guard<std::mutex> owner(callMutex);
    const auto started = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (closed) throw std::runtime_error("SCENE_CAPTURE_CLOSED");
    }
    if (input.EntityCount() > kMaximumEntities)
      throw std::runtime_error("SCENE_SOURCE_ENTITY_CAPACITY");
    for (auto &shard : shards) {
      shard.entities.clear();
      shard.state.Clear();
      shard.failure = nullptr;
      shard.readMs = 0;
      shard.largestPayload = {};
    }
    std::size_t index = 0;
    for (const auto &vertex : input.Entities().Vertices()) {
      if (vertex.first == gz::sim::kNullEntity || index >= kMaximumEntities)
        throw std::runtime_error("SCENE_SOURCE_ENTITY_CAPACITY");
      shards[index++ % shards.size()].entities.insert(vertex.first);
    }
    const auto partitioned = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> lock(mutex);
      ecm = &input;
      pending = shards.size();
      ++generation;
    }
    ready.notify_all();
    {
      std::unique_lock<std::mutex> lock(mutex);
      completed.wait(lock, [this] { return pending == 0; });
      ecm = nullptr; // All reads finished before any error can escape.
    }
    for (const auto &shard : shards)
      if (shard.failure) std::rethrow_exception(shard.failure);
    const auto joined = std::chrono::steady_clock::now();
    auto result = Merge();
    const auto merged = std::chrono::steady_clock::now();
    lastTiming.membershipMs = Milliseconds(partitioned - started);
    lastTiming.readJoinMs = Milliseconds(joined - partitioned);
    lastTiming.mergeMs = Milliseconds(merged - joined);
    for (std::size_t i = 0; i < shards.size(); ++i) {
      lastTiming.shardReadMs[i] = shards[i].readMs;
      lastTiming.largestPayloads[i] = shards[i].largestPayload;
    }
    return result;
  }

  // 功能：
  //   关闭后禁止新采集；并发所有者等待在途采集结束，再统一回收线程。
  // 输入：
  //   无；只能由所有者调用，分片读取器不能调用。
  // 输出：
  //   void：所有工作线程已退出，重复关闭无额外副作用。
  void Stop() {
    std::lock_guard<std::mutex> owner(callMutex);
    {
      std::lock_guard<std::mutex> lock(mutex);
      closed = true;
    }
    ready.notify_all();
    for (auto &thread : threads) if (thread.joinable()) thread.join();
  }
  // 功能：
  //   提供固定工作线程数，不以线程数代替实际吞吐测量。
  // 输入：
  //   无；读取构造后不再改变的分片集合。
  // 输出：
  //   count：分片线程数量。
  std::size_t WorkerCount() const { const auto count = shards.size(); return count; }
  // 功能：
  //   读取最后一次成功采集的耗时；该诊断不能用作传感器采样时间。
  // 输入：
  //   无；等待在途采集后复制诊断。
  // 输出：
  //   timing：采集、读取和合并耗时快照。
  Timing LastTiming() const {
    std::lock_guard<std::mutex> owner(callMutex);
    const auto timing = lastTiming;
    return timing;
  }
  // 功能：
  //   在没有分片写入时汇总模型缓存统计，避免与采集线程发生数据竞争。
  // 输入：
  //   无；等待当前采集完成。
  // 输出：
  //   result：各分片缓存命中、未命中及旁路次数之和。
  ModelSdfState::Statistics ModelStatistics() const {
    std::lock_guard<std::mutex> owner(callMutex);
    ModelSdfState::Statistics result;
    for (const auto &shard : shards) {
      const auto stats = shard.models.Stats();
      result.hits += stats.hits;
      result.misses += stats.misses;
      result.removedBypasses += stats.removedBypasses;
      result.treeBypasses += stats.treeBypasses;
    }
    return result;
  }

 private:
  struct Shard {
    Entities entities;
    gz::msgs::SerializedState state;
    std::exception_ptr failure;
    double readMs = 0;
    Timing::Payload largestPayload;
    ModelSdfState models;
  };
  // 功能：
  //   将单调时钟间隔换算为诊断用毫秒，不依赖系统时钟校准。
  // 输入：
  //   value：单调时钟测得的间隔。
  // 输出：
  //   milliseconds：以毫秒表示的间隔。
  static double Milliseconds(std::chrono::steady_clock::duration value) {
    const auto milliseconds = std::chrono::duration<double, std::milli>(value).count();
    return milliseconds;
  }
  // 功能：
  //   每代只读取分配的实体；记录异常并通知所有者，异常不能绕过其他线程的回收。
  // 输入：
  //   index：当前线程唯一拥有的分片索引。
  // 输出：
  //   void：结果或异常写入对应分片，关闭时退出。
  void Run(std::size_t index) noexcept {
    std::uint64_t seen = 0;
    for (;;) {
      const gz::sim::EntityComponentManager *input;
      {
        std::unique_lock<std::mutex> lock(mutex);
        ready.wait(lock, [&] { return closed || generation != seen; });
        if (closed) return;
        seen = generation;
        input = ecm;
      }
      auto &shard = shards[index];
      const auto started = std::chrono::steady_clock::now();
      try {
        // Gazebo interprets an empty filter as ALL entities, not none.
        if (!shard.entities.empty()) shard.state = read
            ? read(*input, shard.entities) : shard.models.Read(*input, shard.entities);
      } catch (...) {
        shard.failure = std::current_exception();
      }
      shard.readMs = Milliseconds(std::chrono::steady_clock::now() - started);
      {
        std::lock_guard<std::mutex> lock(mutex);
        --pending;
      }
      completed.notify_one();
    }
  }

  // 功能：
  //   校验完整成员、唯一组件及容量后合并；缺失实体不能被误解释为已从世界删除。
  // 输入：
  //   无；所有工作线程已完成的分片快照。
  // 输出：
  //   result：通过完整性检查的场景映射。
  gz::msgs::SerializedStateMap Merge() {
    gz::msgs::SerializedStateMap result;
    std::size_t bytes = 0;
    for (auto &shard : shards) {
      // 数量相等再结合下方成员/重复检查，才能证明没有漏掉实体。
      if (static_cast<std::size_t>(shard.state.entities_size()) != shard.entities.size())
        throw std::runtime_error("SCENE_SOURCE_INCOMPLETE_CAPTURE");
      const auto size = shard.state.ByteSizeLong();
      if (size > kMaximumSceneBytes - bytes)
        throw std::runtime_error("SCENE_SOURCE_CAPTURE_CAPACITY");
      bytes += size;
      for (auto &entity : *shard.state.mutable_entities()) {
        if (!shard.entities.count(entity.id()) || result.entities().count(entity.id())
            || entity.components_size() > 256 || result.entities_size() >=
                static_cast<int>(kMaximumEntities))
          throw std::runtime_error("SCENE_SOURCE_ENTITY_INVALID");
        auto &target = (*result.mutable_entities())[entity.id()];
        target.set_id(entity.id());
        target.set_remove(entity.remove());
        for (auto &component : *entity.mutable_components()) {
          if (component.component().size() > shard.largestPayload.bytes)
            shard.largestPayload = {entity.id(), component.type(), component.component().size()};
          const auto key = static_cast<std::int64_t>(component.type());
          if (target.components().count(key))
            throw std::runtime_error("SCENE_SOURCE_DUPLICATE_COMPONENT");
          // Transfer owned payload bytes; never reinterpret or truncate them.
          (*target.mutable_components())[key].Swap(&component);
        }
      }
    }
    if (result.ByteSizeLong() > kMaximumSceneBytes)
      throw std::runtime_error("SCENE_SOURCE_MAP_CAPACITY");
    return result;
  }

  Read read;
  mutable std::mutex callMutex;
  std::mutex mutex;
  std::condition_variable ready, completed;
  std::vector<Shard> shards;
  std::vector<std::thread> threads;
  const gz::sim::EntityComponentManager *ecm = nullptr;
  std::uint64_t generation = 0;
  std::size_t pending = 0;
  bool closed = false;
  Timing lastTiming;
};
}  // namespace dronedream::replica
