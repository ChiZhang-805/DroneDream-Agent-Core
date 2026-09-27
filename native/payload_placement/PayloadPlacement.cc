// Native, one-step placement of a detached parcel by a simulated human.
// This system never controls the aircraft, changes gravity, or moves an attached parcel.
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>

#include <gz/math/Pose3.hh>
#include <gz/msgs/boolean.pb.h>
#include <gz/msgs/pose.pb.h>
#include <gz/msgs/stringmsg.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/AngularVelocityCmd.hh>
#include <gz/sim/components/DetachableJoint.hh>
#include <gz/sim/components/LinearVelocityCmd.hh>
#include <gz/sim/components/Name.hh>
#include <gz/transport/Node.hh>

namespace dronedream
{
class PayloadPlacement final : public gz::sim::System,
    public gz::sim::ISystemConfigure, public gz::sim::ISystemPreUpdate
{
  // 功能：
  //   将服务严格绑定到包含本插件的载荷模型，不接受任意世界实体名称。
  // 输入：
  //   entity：载荷模型实体；ecm：仿真实体管理器。
  // 输出：
  //   无：注册专用服务，非法模型不提供放置入口。
  public: void Configure(const gz::sim::Entity &entity,
      const std::shared_ptr<const sdf::Element> &, gz::sim::EntityComponentManager &ecm,
      gz::sim::EventManager &) override
  {
    model = gz::sim::Model(entity);
    if (!model.Valid(ecm))
    {
      std::cerr << "Payload placement requires a model entity" << std::endl;
      return;
    }
    // Configure 期间 SDF 父子组件尚可能未全部建立，首个 PreUpdate 再绑定世界和唯一链接。
  }

  // 功能：
  //   1. 接纳一个有限、单位四元数的放置请求，并等待仿真线程处理。
  //   2. 请求超时即撤销，拒绝并发覆盖和迟到执行。
  // 输入：
  //   request：仅允许当前载荷的世界位姿。
  // 输出：
  //   response：本次命令是否由仿真线程接纳；返回值为传输成功状态。
  private: bool Request(const gz::msgs::Pose &request, gz::msgs::Boolean &response)
  {
    response.set_data(false);
    const auto &p = request.position();
    const auto &q = request.orientation();
    const double norm = q.w()*q.w() + q.x()*q.x() + q.y()*q.y() + q.z()*q.z();
    if (request.name() != name || request.id() != 0 || !std::isfinite(norm)
        || std::abs(norm - 1.) > 1e-5) return true;
    for (const double value : {p.x(), p.y(), p.z()})
      if (!std::isfinite(value) || std::abs(value) > 1e6) return true;
    std::unique_lock<std::mutex> guard(mutex);
    if (busy) return true;
    busy = true;
    completed = accepted = false;
    pending = gz::math::Pose3d(gz::math::Vector3d(p.x(), p.y(), p.z()),
                             gz::math::Quaterniond(q.w(), q.x(), q.y(), q.z()));
    expires = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    changed.wait_until(guard, expires, [this] { return completed; });
    response.set_data(completed && accepted);
    pending.reset();
    busy = false;
    return true;
  }

  // 功能：按请求回读下一仿真时步的真实关节状态；不发布命令、不搬动载荷。
  // 输入：request：32 位十六进制请求标识；输出：原标识、状态和实际迭代号。
  // 超时、并发、暂停均返回 unknown，绝不把没有事件误判为 detached。
  private: bool ReadState(const gz::msgs::StringMsg &request, gz::msgs::StringMsg &response)
  {
    const auto token = request.data();
    response.set_data(token + "|unknown|0");
    if (token.size() != 32 || token.find_first_not_of("0123456789abcdef") != std::string::npos)
      return true;
    std::unique_lock<std::mutex> guard(mutex);
    if (stateBusy) return true;
    stateBusy = true;
    stateCompleted = false;
    stateToken = token;
    stateExpires = std::chrono::steady_clock::now() + std::chrono::milliseconds(1000);
    changed.wait_until(guard, stateExpires, [this] { return stateCompleted; });
    if (stateCompleted) response.set_data(stateReply);
    stateBusy = false;
    return true;
  }

  // 功能：
  //   1. 只对已分离载荷同时设置位置与零初速度，模拟工作人员拿稳后放置。
  //   2. 下一步移除本插件自己的速度命令，避免 Harmonic 保留零命令而冻结自由落体。
  // 输入：
  //   info：当前仿真时步；ecm：实际实体和关节状态。
  // 输出：
  //   无：接纳或拒绝待处理请求，挂载状态下绝不移动载荷。
  public: void PreUpdate(const gz::sim::UpdateInfo &info,
      gz::sim::EntityComponentManager &ecm) override
  {
    if (info.paused) return;
    if (!registered)
    {
      name = model.Name(ecm);
      const auto world = ecm.Component<gz::sim::components::Name>(
          gz::sim::worldEntity(model.Entity(), ecm));
      if (name.empty() || !world || model.LinkCount(ecm) != 1) return;
      const std::string service = "/world/" + world->Data() + "/model/" + name + "/place_detached";
      registered = node.Advertise(service, &PayloadPlacement::Request, this);
      if (!registered)
        std::cerr << "Payload placement service registration failed: " << service << std::endl;
    }
    if (registered && !stateRegistered)
    {
      const auto world = ecm.Component<gz::sim::components::Name>(
          gz::sim::worldEntity(model.Entity(), ecm));
      if (world)
        stateRegistered = node.Advertise("/world/" + world->Data() + "/model/" + name +
            "/attachment_state", &PayloadPlacement::ReadState, this);
    }
    if (clearVelocity)
    {
      ecm.RemoveComponent<gz::sim::components::LinearVelocityCmd>(model.Entity());
      ecm.RemoveComponent<gz::sim::components::AngularVelocityCmd>(model.Entity());
      clearVelocity = false;
    }
    std::lock_guard<std::mutex> guard(mutex);
    if ((!pending || completed) && (!stateBusy || stateCompleted)) return;
    link = model.CanonicalLink(ecm);
    bool attached = false;
    ecm.Each<gz::sim::components::DetachableJoint>(
      [&](const gz::sim::Entity &, const gz::sim::components::DetachableJoint *joint)
      {
        if (joint->Data().childLink == link || joint->Data().parentLink == link)
          attached = true;
        return !attached;
      });
    if (stateBusy && !stateCompleted && std::chrono::steady_clock::now() < stateExpires)
    {
      const bool valid = model.Valid(ecm) && link != gz::sim::kNullEntity && model.LinkCount(ecm) == 1;
      stateReply = stateToken + "|" + (valid ? (attached ? "attached" : "detached") : "unknown")
          + "|" + std::to_string(info.iterations);
      stateCompleted = true;
      changed.notify_all();
    }
    if (!pending || completed) return;
    accepted = !attached && std::chrono::steady_clock::now() < expires
        && model.Valid(ecm) && link != gz::sim::kNullEntity && model.LinkCount(ecm) == 1
        && !ecm.Component<gz::sim::components::LinearVelocityCmd>(model.Entity())
        && !ecm.Component<gz::sim::components::AngularVelocityCmd>(model.Entity());
    if (accepted)
    {
      model.SetWorldPoseCmd(ecm, *pending);
      ecm.SetComponentData<gz::sim::components::LinearVelocityCmd>(model.Entity(), gz::math::Vector3d::Zero);
      ecm.SetComponentData<gz::sim::components::AngularVelocityCmd>(model.Entity(), gz::math::Vector3d::Zero);
      clearVelocity = true;
    }
    pending.reset();
    completed = true;
    changed.notify_all();
  }

  private: gz::sim::Model model{gz::sim::kNullEntity};
  private: gz::sim::Entity link{gz::sim::kNullEntity};
  private: std::string name;
  private: std::mutex mutex;
  private: std::condition_variable changed;
  private: std::optional<gz::math::Pose3d> pending;
  private: std::chrono::steady_clock::time_point expires;
  private: bool busy{false}, completed{false}, accepted{false}, clearVelocity{false};
  private: bool registered{false};
  private: bool stateRegistered{false}, stateBusy{false}, stateCompleted{false};
  private: std::string stateToken, stateReply;
  private: std::chrono::steady_clock::time_point stateExpires;
  // Node 先析构并排空回调，再销毁回调使用的互斥量和等待状态。
  private: gz::transport::Node node;
};
}

GZ_ADD_PLUGIN(dronedream::PayloadPlacement, gz::sim::System,
    dronedream::PayloadPlacement::ISystemConfigure, dronedream::PayloadPlacement::ISystemPreUpdate)
GZ_ADD_PLUGIN_ALIAS(dronedream::PayloadPlacement, "dronedream::PayloadPlacement")
