#pragma once

#include <string>
#include <cstdint>
#include <vector>

namespace dronedream_agent_plugin_api
{
struct CapabilityConfiguration
{
  // 固定内核签发任务身份和检测时限；插件名称、权限及消息时间不由用户自然语言直接授权。
  std::string capability_id;
  std::string contract_id;
  std::string authority;
  std::uint32_t deadline_ms{1000};
  std::uint32_t startup_deadline_ms{10000};
};

struct CapabilityObservation
{
  // 只包含已桥接的观测，不携带执行电机或覆盖安全边界的权限。
  std::string contract_id;
  std::string runtime_phase;
  std::uint64_t sequence{0};
  std::int64_t monotonic_time_ns{0};
  bool localization_ok{false};
  bool link_ok{false};
  bool geofence_ok{false};
  double battery_percent{0.0};
};

struct CapabilityProposal
{
  std::string proposal_id;
  std::string command;
  std::string reason;
  bool requires_core_authorization{true};
};

struct CapabilityExecutionReceipt
{
  std::string proposal_id;
  bool accepted{false};
  std::string terminal_status;
  std::vector<std::string> issue_codes;
};

struct CapabilityHealth
{
  bool healthy{false};
  bool deadline_missed{false};
  std::string state;
  std::uint64_t observation_age_ms{0};
  std::uint64_t deadline_ms{0};
};

struct CapabilityEvidence
{
  std::string capability_id;
  std::string contract_id;
  std::uint64_t observation_sequence{0};
  std::string terminal_status;
};

class CapabilityPlugin
{
public:
  // 功能：
  //   允许主机通过抽象接口安全销毁具体插件对象。
  // 输入：
  //   无。
  // 输出：
  //   无返回值：释放具体实现资源。
  virtual ~CapabilityPlugin() = default;
  // 功能：
  //   声明可核对的稳定能力身份。
  // 输入：
  //   无。
  // 输出：
  //   capability_id：当前插件标识。
  virtual std::string id() const = 0;
  // 功能：
  //   声明插件权限类别，固定内核仍负责最终授权。
  // 输入：
  //   无。
  // 输出：
  //   authority_name：权限标签。
  virtual std::string authority() const = 0;
  // 功能：
  //   验证并绑定本次任务配置。
  // 输入：
  //   configuration：主机提供的身份、权限及时间预算。
  // 输出：
  //   configured：配置可使用时为真。
  virtual bool configure(const CapabilityConfiguration & configuration) = 0;
  // 功能：
  //   激活已配置插件，不应自行恢复已失效的旧任务。
  // 输入：
  //   无。
  // 输出：
  //   activated：激活成功标识。
  virtual bool activate() = 0;
  // 功能：
  //   接收并校验新的任务观测，拒绝跨任务或失效输入。
  // 输入：
  //   observation：类型化实时观测。
  // 输出：
  //   accepted：观测接收结果。
  virtual bool observe(const CapabilityObservation & observation) = 0;
  // 功能：
  //   生成交给固定内核审批的策略建议。
  // 输入：
  //   无；使用已接收观测。
  // 输出：
  //   proposal：含标识、命令类别及授权需求的建议。
  virtual CapabilityProposal propose() = 0;
  // 功能：
  //   处理固定内核交回的提案；回执语义须区分策略接纳与物理执行。
  // 输入：
  //   proposal：待处理提案。
  // 输出：
  //   receipt：接纳状态和原因。
  virtual CapabilityExecutionReceipt execute(const CapabilityProposal & proposal) = 0;
  // 功能：
  //   撤销当前活动建议并响应安全保持请求，具体飞控动作仍须由已授权执行器完成。
  // 输入：
  //   reason：固定内核给出的安全原因。
  // 输出：
  //   receipt：保持请求的策略回执。
  virtual CapabilityExecutionReceipt hold(const std::string & reason) noexcept = 0;
  // 功能：
  //   报告健康和观测时效，不把诊断状态作为动作完成证据。
  // 输入：
  //   无。
  // 输出：
  //   report：状态、观测年龄及截止时间。
  virtual CapabilityHealth health() const noexcept = 0;
  // 功能：
  //   导出本次策略运行的任务和观测绑定。
  // 输入：
  //   无。
  // 输出：
  //   evidence_record：插件生命周期证据。
  virtual CapabilityEvidence evidence() const noexcept = 0;
  // 功能：
  //   撤销插件活动状态，允许主机安全结束生命周期。
  // 输入：
  //   无。
  // 输出：
  //   void：不返回业务数据。
  virtual void deactivate() noexcept = 0;
};
}  // namespace dronedream_agent_plugin_api
