#pragma once

#include <chrono>
#include <cerrno>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <fcntl.h>
#include <unistd.h>
#include <gz/msgs/header.pb.h>
#include <openssl/evp.h>

namespace dronedream::replica {
constexpr std::size_t kMaximumSceneBytes = 16 * 1024 * 1024;
constexpr std::size_t kMaximumEntities = 65536;
constexpr std::int64_t kMaximumFrameAgeNs = 250000000;

// 功能：
//   取得跨进程场景时间戳使用的 Unix 纳秒；时钟跳变仍由接收端的新鲜度检查拒绝。
// 输入：
//   无。
// 输出：
//   timestamp：当前系统时钟纳秒值。
inline std::int64_t UnixNs() {
  const auto timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
  return timestamp;
}

// 功能：
//   对完整场景字节计算 SHA-256，失败不返回占位摘要。
// 输入：
//   bytes：本次场景快照的确切序列化字节。
// 输出：
//   text：规范小写十六进制摘要。
inline std::string Sha256(const std::string &bytes) {
  unsigned char result[EVP_MAX_MD_SIZE];
  unsigned int length = 0;
  if (EVP_Digest(bytes.data(), bytes.size(), result, &length, EVP_sha256(), nullptr) != 1
      || length != 32) throw std::runtime_error("SCENE_HASH_FAILED");
  static constexpr char hex[] = "0123456789abcdef";
  std::string text;
  for (unsigned int i = 0; i < length; ++i) {
    text.push_back(hex[result[i] >> 4]);
    text.push_back(hex[result[i] & 15]);
  }
  return text;
}

// 功能：
//   核验源世代和场景摘要的规范格式。
// 输入：
//   value：待核验文本。
// 输出：
//   valid：长度及字符集合均正确时为真。
inline bool IsDigest(const std::string &value) {
  return value.size() == 64 && value.find_first_not_of("0123456789abcdef") == std::string::npos;
}

// 功能：
//   解析无符号文本形式的非负有符号 64 位整数，不允许科学记数、负号、尾随字符或溢出。
// 输入：
//   value：消息头中的数字文本。
// 输出：
//   result：解析后的整数。
inline std::int64_t Decimal(const std::string &value) {
  if (value.empty() || value.size() > 19 || value.find_first_not_of("0123456789") !=
      std::string::npos) throw std::runtime_error("SCENE_INTEGER_INVALID");
  std::size_t used = 0;
  const auto result = std::stoll(value, &used);
  if (used != value.size()) throw std::runtime_error("SCENE_INTEGER_INVALID");
  return result;
}

// 功能：
//   将规范秒/纳秒时间转换为有符号纳秒，先检查总和边界再计算，避免极端消息触发整数溢出。
// 输入：
//   seconds：非负整秒。
//   nanoseconds：零至 999999999 的纳秒余数。
// 输出：
//   timestamp：可安全表示的仿真纳秒时间。
inline std::int64_t SimulationTime(std::int64_t seconds, std::int64_t nanoseconds) {
  constexpr std::int64_t scale = 1000000000;
  constexpr auto maximum = std::numeric_limits<std::int64_t>::max();
  if (seconds < 0 || nanoseconds < 0 || nanoseconds >= scale ||
      seconds > maximum / scale ||
      (seconds == maximum / scale && nanoseconds > maximum % scale))
    throw std::runtime_error("REPLICA_IMAGE_TIME_INVALID");
  const auto timestamp = seconds * scale + nanoseconds;
  return timestamp;
}

// 功能：
//   给有效 protobuf 消息头追加一个单值身份字段，接收端仍须核对完整字段集合。
// 输入：
//   header：当前消息头指针。
//   key、value：字段名与其唯一值。
// 输出：
//   void：消息头新增一项。
inline void Field(gz::msgs::Header *header, const std::string &key, const std::string &value) {
  if (!header) throw std::runtime_error("SCENE_HEADER_MISSING");
  auto *row = header->add_data();
  row->set_key(key);
  row->add_value(value);
}

struct FrameIdentity {
  std::string epoch;
  std::string sha256;
  std::int64_t sequence = 0;
  std::int64_t sourceUnixNs = 0;
  std::int64_t simulationNs = 0;
};

// 功能：
//   有界解析六项完整场景身份，拒绝重复字段、错误来源、非法时钟及空序号。
// 输入：
//   header：接收到的 protobuf 消息头。
//   expectedEpoch：本次执行固定的源身份摘要。
// 输出：
//   result：校验后的场景身份；是否新鲜及按序仍由专门检查完成。
inline FrameIdentity ReadIdentity(const gz::msgs::Header &header,
                                 const std::string &expectedEpoch) {
  if (header.data_size() != 6 || header.ByteSizeLong() > 1024)
    throw std::runtime_error("SCENE_HEADER_CAPACITY");
  std::unordered_map<std::string, std::string> values;
  for (const auto &row : header.data()) {
    if (row.key().size() > 32 || row.value_size() != 1 || row.value(0).size() > 64
        || !values.emplace(row.key(), row.value(0)).second)
      throw std::runtime_error("SCENE_HEADER_AMBIGUOUS");
  }
  if (values.size() != 6 || values.at("contract") != "complete-scene-snapshot"
      || values.at("epoch") != expectedEpoch || !IsDigest(expectedEpoch)
      || !IsDigest(values.at("sha256"))) throw std::runtime_error("SCENE_IDENTITY_INVALID");
  FrameIdentity result{expectedEpoch, values.at("sha256"), Decimal(values.at("sequence")),
      Decimal(values.at("source_unix_ns")), Decimal(values.at("simulation_ns"))};
  if (result.sequence == 0 || result.sourceUnixNs == 0)
    throw std::runtime_error("SCENE_IDENTITY_INVALID");
  return result;
}

// 功能：
//   在空消息头写入完整有效身份，不在已有身份后继续追加冲突字段。
// 输入：
//   header：待写入的空数据消息头。
//   identity：本次完整场景的源、摘要、序号和时钟。
// 输出：
//   void：完成六个身份字段写入。
inline void WriteIdentity(gz::msgs::Header *header, const FrameIdentity &identity) {
  if (!header || header->data_size() != 0 || !IsDigest(identity.epoch) ||
      !IsDigest(identity.sha256) || identity.sequence <= 0 || identity.sourceUnixNs <= 0 ||
      identity.simulationNs < 0) throw std::runtime_error("SCENE_IDENTITY_INVALID");
  Field(header, "contract", "complete-scene-snapshot");
  Field(header, "epoch", identity.epoch);
  Field(header, "sha256", identity.sha256);
  Field(header, "sequence", std::to_string(identity.sequence));
  Field(header, "source_unix_ns", std::to_string(identity.sourceUnixNs));
  Field(header, "simulation_ns", std::to_string(identity.simulationNs));
}

// 功能：
//   按原始源时间判定最大允许帧年龄，不使用接收时间重置旧帧年龄，也拒绝未来时间。
// 输入：
//   identity：待消费场景身份。
//   now：当前 Unix 纳秒时间。
// 输出：
//   fresh：年龄在闭区间零到 kMaximumFrameAgeNs 内时为真。
inline bool Fresh(const FrameIdentity &identity, std::int64_t now) {
  return identity.sourceUnixNs > 0 && now >= identity.sourceUnixNs
      && now - identity.sourceUnixNs <= kMaximumFrameAgeNs;
}

// Full snapshots may skip intermediate states, never time-regress or switch source.
class SourceOrder {
 public:
  // 功能：
  //   接纳同来源严格递增的完整快照；可跳过中间帧，但失败不更新当前序列基线。
  // 输入：
  //   identity：已解码且待按序检查的场景身份。
  // 输出：
  //   void：成功时推进序号和两个时钟，失败抛出拒绝异常。
  void Admit(const FrameIdentity &identity) {
    if (!IsDigest(identity.epoch) || !IsDigest(identity.sha256) || (!epoch.empty() && identity.epoch != epoch)
        || identity.sequence <= sequence || identity.simulationNs <= simulationNs
        || identity.sourceUnixNs <= sourceUnixNs)
      throw std::runtime_error("SCENE_SOURCE_REGRESSED_OR_REPLAYED");
    sequence = identity.sequence;
    simulationNs = identity.simulationNs;
    sourceUnixNs = identity.sourceUnixNs;
    epoch = identity.epoch;
  }
 private:
  std::int64_t sequence = 0;
  std::int64_t simulationNs = -1;
  std::int64_t sourceUnixNs = 0;
  std::string epoch;
};

// 功能：
//   将最多 512 个输入字节转成安全 ASCII JSON 诊断文本；高位字节转义，不截出非法 UTF-8。
// 输入：
//   value：诊断文本，不用作场景身份或控制输入。
// 输出：
//   result：包含引号的标准 JSON 字符串；非 ASCII 内容按原字节转义显示。
inline std::string JsonText(const std::string &value) {
  std::string result = "\"";
  static constexpr char hex[] = "0123456789abcdef";
  for (unsigned char c : value.substr(0, 512)) {
    if (c >= 128) {
      result += "\\u00";
      result += hex[c >> 4];
      result += hex[c & 15];
      continue;
    }
    if (c == '"' || c == '\\') result += '\\';
    result += c < 32 ? '?' : c;
  }
  result += "\"";
  return result;
}

// 功能：
//   独占写入当前运行的有界诊断记录并同步，拒绝覆盖先前证据；不完整文件只能视为失败。
// 输入：
//   path：调用方固定的诊断输出路径。
//   bytes：完整诊断文本。
// 输出：
//   void：写入及同步成功无业务返回值，失败保留错误证据而不授予权限。
inline void WriteNew(const std::string &path, const std::string &bytes) {
  if (bytes.empty() || bytes.size() > kMaximumSceneBytes)
    throw std::runtime_error("REPLICA_RECEIPT_CAPACITY");
  const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
  if (fd < 0) throw std::runtime_error("REPLICA_RECEIPT_CREATE_FAILED");
  std::size_t used = 0;
  while (used < bytes.size()) {
    const auto count = ::write(fd, bytes.data() + used, bytes.size() - used);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) { ::close(fd); throw std::runtime_error("REPLICA_RECEIPT_WRITE_FAILED"); }
    used += count;
  }
  const auto synced = ::fsync(fd);
  const auto closed = ::close(fd);
  if (synced || closed) throw std::runtime_error("REPLICA_RECEIPT_SYNC_FAILED");
}
}  // namespace dronedream::replica
