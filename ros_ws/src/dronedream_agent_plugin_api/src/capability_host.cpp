#include "dronedream_agent_plugin_api/capability_plugin.hpp"
#include "dronedream_agent_msgs/msg/mission_lifecycle.hpp"
#include "dronedream_agent_msgs/msg/mission_observation.hpp"
#include "dronedream_agent_msgs/msg/safety_event.hpp"

#include <lifecycle_msgs/msg/state.hpp>
#include <iostream>
#include <memory>
#include <pluginlib/class_loader.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <chrono>

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

class CapabilityHost final : public rclcpp_lifecycle::LifecycleNode
{
public:
  // 功能：
  //   声明本次任务的策略插件、话题及有界时限；不默认绑定探针或旧任务身份。
  // 输入：
  //   无。
  // 输出：
  //   当前节点：仅完成参数声明，尚未激活插件。
  CapabilityHost()
  : rclcpp_lifecycle::LifecycleNode("dronedream_capability_host")
  {
    declare_parameter<std::vector<std::string>>(
      "plugins", {"dronedream_agent_plugin_api/SafeHoldCapability"});
    declare_parameter<std::string>("contract_id", "");
    declare_parameter<int>("watchdog_deadline_ms", 1000);
    declare_parameter<int>("watchdog_startup_deadline_ms", 10000);
    declare_parameter<int>("watchdog_scheduling_jitter_grace_ms", 250);
    declare_parameter<int>("watchdog_persistent_miss_ms", 250);
    declare_parameter<std::string>("observation_topic", "/dronedream/mission_observation");
    declare_parameter<std::string>("lifecycle_topic", "/dronedream/mission_lifecycle");
    declare_parameter<std::string>("safety_event_topic", "/dronedream/safety_event");
  }

  // 功能：
  //   1. 核对时限后加载当前插件，防止负数转无符号值变成超长的安全宽限期。
  //   2. 建立真实观测及已落地终态订阅；初始化失败释放部分资源，不能混用旧插件。
  // 输入：
  //   state：ROS 生命周期传入的前置状态，本回调不修改该对象。
  // 输出：
  //   callback_result：全部插件成功配置为 SUCCESS，否则为 FAILURE。
  CallbackReturn on_configure(const rclcpp_lifecycle::State &) override
  {
    stopping_ = true;
    watchdog_.reset();
    observation_subscription_.reset();
    lifecycle_subscription_.reset();
    safety_publisher_.reset();
    deactivate_all();
    plugins_.clear();
    loader_.reset();
    try {
      const auto deadline = get_parameter("watchdog_deadline_ms").as_int();
      const auto startup_deadline = get_parameter("watchdog_startup_deadline_ms").as_int();
      if (get_parameter("contract_id").as_string().empty() || deadline <= 0 ||
          deadline > 60000 || startup_deadline < deadline || startup_deadline > 300000) {
        throw std::runtime_error("contract identity or watchdog deadlines invalid");
      }
      const auto scheduling_jitter_grace_ms =
        get_parameter("watchdog_scheduling_jitter_grace_ms").as_int();
      if (scheduling_jitter_grace_ms < 0 || scheduling_jitter_grace_ms > 500) {
        throw std::runtime_error("watchdog scheduling jitter grace is outside [0, 500] ms");
      }
      const auto persistent_miss_ms = get_parameter("watchdog_persistent_miss_ms").as_int();
      if (persistent_miss_ms < 25 || persistent_miss_ms > 1000) {
        throw std::runtime_error("watchdog persistent miss window is outside [25, 1000] ms");
      }
      loader_ = std::make_unique<Loader>(
        "dronedream_agent_plugin_api",
        "dronedream_agent_plugin_api::CapabilityPlugin");
      const auto plugin_names = get_parameter("plugins").as_string_array();
      std::unordered_map<std::string, bool> identities;
      for (const auto & name : plugin_names) {
        auto plugin = loader_->createSharedInstance(name);
        if (!identities.emplace(plugin->id(), true).second) {
          throw std::runtime_error("duplicate runtime capability identity");
        }
        dronedream_agent_plugin_api::CapabilityConfiguration configuration{
          plugin->id(), get_parameter("contract_id").as_string(), plugin->authority(),
          static_cast<std::uint32_t>(deadline), static_cast<std::uint32_t>(startup_deadline)};
        if (!plugin->configure(configuration)) {
          throw std::runtime_error("plugin rejected typed configuration");
        }
        plugins_.push_back(plugin);
      }
      safety_publisher_ = create_publisher<dronedream_agent_msgs::msg::SafetyEvent>(
        get_parameter("safety_event_topic").as_string(), rclcpp::QoS(20).reliable());
      observation_subscription_ =
        create_subscription<dronedream_agent_msgs::msg::MissionObservation>(
        get_parameter("observation_topic").as_string(), rclcpp::SensorDataQoS(),
        [this](const dronedream_agent_msgs::msg::MissionObservation::SharedPtr message) {
          if (failed_closed_ || stopping_ || !rclcpp::ok()) {
            return;
          }
          const auto monotonic_time = std::chrono::steady_clock::now().time_since_epoch();
          dronedream_agent_plugin_api::CapabilityObservation observation{
            message->contract_id,
            message->runtime_phase,
            message->sequence,
            std::chrono::duration_cast<std::chrono::nanoseconds>(monotonic_time).count(),
            message->localization_ok,
            message->link_ok,
            message->geofence_ok,
            static_cast<double>(message->battery_percent)};
          for (const auto & plugin : plugins_) {
            if (!plugin->observe(observation)) {
              fail_closed(plugin, "OBSERVATION_REJECTED");
              return;
            }
            deadline_miss_started_at_.erase(plugin->id());
          }
        });
      lifecycle_subscription_ =
        create_subscription<dronedream_agent_msgs::msg::MissionLifecycle>(
        get_parameter("lifecycle_topic").as_string(), rclcpp::QoS(10).reliable(),
        [this](const dronedream_agent_msgs::msg::MissionLifecycle::SharedPtr message) {
          if (
            message->contract_id != get_parameter("contract_id").as_string() ||
            message->terminal_state != "ON_GROUND" || !message->landing_confirmed ||
            !message->safe_to_stop_watchdog)
          {
            return;
          }
          stopping_ = true;
          watchdog_.reset();
          RCLCPP_INFO(
            get_logger(),
            "accepted core terminal lifecycle event for contract %s with executor return code %d",
            message->contract_id.c_str(), message->executor_return_code);
        });
      return plugins_.empty() ? CallbackReturn::FAILURE : CallbackReturn::SUCCESS;
    } catch (const std::exception & error) {
      RCLCPP_ERROR(get_logger(), "plugin configure failed: %s", error.what());
      observation_subscription_.reset();
      lifecycle_subscription_.reset();
      safety_publisher_.reset();
      deactivate_all();
      plugins_.clear();
      loader_.reset();
      return CallbackReturn::FAILURE;
    }
  }

  // 功能：
  //   激活已配置插件并启动 25 毫秒健康检查，抖动宽限及持续失败窗口都有明确上限。
  // 输入：
  //   state：ROS 生命周期前置状态。
  // 输出：
  //   callback_result：插件健康且定时器启动时为 SUCCESS，否则为 FAILURE。
  CallbackReturn on_activate(const rclcpp_lifecycle::State &) override
  {
    stopping_ = false;
    failed_closed_ = false;
    deadline_miss_started_at_.clear();
    for (const auto & plugin : plugins_) {
      if (!plugin->activate() || !plugin->health().healthy) {
        deactivate_all();
        return CallbackReturn::FAILURE;
      }
      RCLCPP_INFO(get_logger(), "activated plugin %s", plugin->id().c_str());
    }
    watchdog_ = create_wall_timer(std::chrono::milliseconds(25), [this]() {
      // SIGINT/SIGTERM makes rclcpp::ok() false before spin() returns.  Ignore
      // that orderly teardown window: observations have intentionally stopped,
      // so treating the resulting stale health report as an in-flight deadline
      // miss would publish a false emergency after a successful landing.
      if (stopping_ || !rclcpp::ok()) {
        return;
      }
      for (const auto & plugin : plugins_) {
        const auto report = plugin->health();
        if (!report.healthy && !report.deadline_missed) {
          deadline_miss_started_at_.erase(plugin->id());
          fail_closed(
            plugin, "WATCHDOG_HEALTH_OR_DEADLINE_FAILURE",
            report.observation_age_ms, report.deadline_ms, report.deadline_missed);
          return;
        }
        if (report.deadline_missed) {
          const auto scheduling_jitter_grace_ms = static_cast<std::uint64_t>(
            get_parameter("watchdog_scheduling_jitter_grace_ms").as_int());
          const auto effective_deadline_ms = report.deadline_ms + scheduling_jitter_grace_ms;
          // The observation publisher and this wall timer are independently
          // scheduled on a non-real-time Windows/WSL host.  Preserve the
          // contract deadline as the health signal, but cross the fail-closed
          // gate only after its explicitly qualified scheduling grace.  A
          // fresh observation clears the miss without weakening the bound.
          if (report.observation_age_ms <= effective_deadline_ms) {
            deadline_miss_started_at_.erase(plugin->id());
            continue;
          }
          const auto now = std::chrono::steady_clock::now();
          const auto [miss, inserted] = deadline_miss_started_at_.try_emplace(
            plugin->id(), now);
          if (inserted) {
            continue;
          }
          const auto persistent_miss_ms = static_cast<std::uint64_t>(
            get_parameter("watchdog_persistent_miss_ms").as_int());
          const auto continuously_missed_ms = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(now - miss->second).count());
          // Crossing the scheduling envelope first enters a bounded debounce
          // window.  A newly accepted observation erases the window in the
          // subscription callback; only a continuously stale stream reaches
          // the irreversible land gate.  This keeps fail-closed behavior while
          // refusing to turn one delayed Windows/WSL scheduling slice into a
          // false emergency.
          if (continuously_missed_ms < persistent_miss_ms) {
            continue;
          }
          fail_closed(
            plugin, "WATCHDOG_HEALTH_OR_DEADLINE_FAILURE",
            report.observation_age_ms, effective_deadline_ms + persistent_miss_ms,
            report.deadline_missed);
          return;
        }
        deadline_miss_started_at_.erase(plugin->id());
      }
    });
    return CallbackReturn::SUCCESS;
  }

  // 功能：
  //   停止安全检测定时器并撤销插件活动状态，不把软件停用当作飞机已落地。
  // 输入：
  //   state：ROS 生命周期前置状态。
  // 输出：
  //   callback_result：完成停用后为 SUCCESS。
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State &) override
  {
    stopping_ = true;
    watchdog_.reset();
    deadline_miss_started_at_.clear();
    deactivate_all();
    return CallbackReturn::SUCCESS;
  }

  // 功能：
  //   按先停用、后销毁插件、最后释放动态加载器的顺序回收本轮 ROS 资源。
  // 输入：
  //   state：ROS 生命周期前置状态。
  // 输出：
  //   callback_result：资源清理完成后为 SUCCESS。
  CallbackReturn on_cleanup(const rclcpp_lifecycle::State &) override
  {
    stopping_ = true;
    watchdog_.reset();
    deadline_miss_started_at_.clear();
    deactivate_all();
    plugins_.clear();
    observation_subscription_.reset();
    lifecycle_subscription_.reset();
    safety_publisher_.reset();
    loader_.reset();
    return CallbackReturn::SUCCESS;
  }

  // 功能：
  //   析构前撤销全部插件活动状态；插件实例成员在加载器成员之前销毁。
  // 输入：
  //   无。
  // 输出：
  //   无返回值：结束节点生命周期。
  ~CapabilityHost() override {deactivate_all();}

private:
  using Plugin = dronedream_agent_plugin_api::CapabilityPlugin;
  using Loader = pluginlib::ClassLoader<Plugin>;

  // 功能：
  //   撤销所有已加载插件的策略活动状态，不产生额外飞控命令。
  // 输入：
  //   无；访问当前 plugins_ 集合。
  // 输出：
  //   void：所有插件均完成停用请求。
  void deactivate_all() noexcept
  {
    for (const auto & plugin : plugins_) {
      plugin->deactivate();
    }
  }

  // 功能：
  //   一次性发出固定内核识别的中止事件并停用策略插件；飞机悬停及降落由执行器完成。
  // 输入：
  //   plugin：触发失败的策略插件。
  //   issue：拒绝原因。
  //   observation_age_ms：最近有效观测年龄。
  //   deadline_ms：包含经确认宽限的最大等待时限。
  //   deadline_missed：是否属于观测超时。
  // 输出：
  //   void：发布安全事件并锁定 failed_closed_，不虚构物理悬停成功。
  void fail_closed(
    const std::shared_ptr<Plugin> & plugin, const std::string & issue,
    const std::uint64_t observation_age_ms = 0, const std::uint64_t deadline_ms = 0,
    const bool deadline_missed = false)
  {
    if (failed_closed_) {
      return;
    }
    failed_closed_ = true;
    const auto receipt = plugin->hold(issue);
    if (safety_publisher_) {
      dronedream_agent_msgs::msg::SafetyEvent event;
      event.header.stamp = now();
      event.contract_id = get_parameter("contract_id").as_string();
      event.observation_sequence = plugin->evidence().observation_sequence;
      event.observation_age_ms = observation_age_ms;
      event.deadline_ms = deadline_ms;
      event.severity = 3;
      event.action = "safe_hold_then_land";
      event.issue_codes = receipt.issue_codes;
      if (deadline_missed) {
        event.issue_codes.push_back("MISSION_OBSERVATION_DEADLINE_MISSED");
      }
      safety_publisher_->publish(event);
    }
    deactivate_all();
    watchdog_.reset();
    RCLCPP_FATAL(
      get_logger(),
      "runtime plugin watchdog forced fail-closed hold: %s observation_age_ms=%llu "
      "deadline_ms=%llu observation_sequence=%llu",
      issue.c_str(), static_cast<unsigned long long>(observation_age_ms),
      static_cast<unsigned long long>(deadline_ms),
      static_cast<unsigned long long>(plugin->evidence().observation_sequence));
  }

  std::unique_ptr<Loader> loader_;
  std::vector<std::shared_ptr<Plugin>> plugins_;
  rclcpp::TimerBase::SharedPtr watchdog_;
  rclcpp::Subscription<dronedream_agent_msgs::msg::MissionObservation>::SharedPtr
    observation_subscription_;
  rclcpp::Subscription<dronedream_agent_msgs::msg::MissionLifecycle>::SharedPtr
    lifecycle_subscription_;
  rclcpp::Publisher<dronedream_agent_msgs::msg::SafetyEvent>::SharedPtr safety_publisher_;
  std::unordered_map<std::string, std::chrono::steady_clock::time_point>
    deadline_miss_started_at_;
  bool failed_closed_{false};
  bool stopping_{false};
};

// 功能：
//   配置并运行生命周期主机；显式自检可使用探针任务，正常执行必须提供真实任务身份。
// 输入：
//   argc、argv：程序参数及可选 --self-test。
// 输出：
//   exit_code：正常结束为 0，配置失败为 2，激活失败为 3。
int main(int argc, char ** argv)
{
  const bool self_test = argc == 2 && std::string(argv[1]) == "--self-test";
  const int ros_argc = self_test ? 1 : argc;
  rclcpp::init(ros_argc, argv);
  auto node = std::make_shared<CapabilityHost>();
  if (self_test) {
    node->set_parameter(rclcpp::Parameter("contract_id", "runtime-probe-contract"));
  }
  node->configure();
  if (node->get_current_state().id() != lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE) {
    rclcpp::shutdown();
    return 2;
  }
  node->activate();
  if (node->get_current_state().id() != lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE) {
    rclcpp::shutdown();
    return 3;
  }
  if (self_test) {
    node->deactivate();
    node->cleanup();
    std::cout << "PLUGIN_LIFECYCLE_READY configure=ok activate=ok deactivate=ok cleanup=ok"
              << std::endl;
    rclcpp::shutdown();
    return 0;
  }
  rclcpp::spin(node->get_node_base_interface());
  node->deactivate();
  node->cleanup();
  rclcpp::shutdown();
  return 0;
}
