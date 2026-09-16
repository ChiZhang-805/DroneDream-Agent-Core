#include "replica_contract.hpp"
#include "replica_timing.hpp"
#include <iostream>

using namespace dronedream::replica;
// 功能：
//   断言契约条件，失败立即结束测试，不继续输出成功标记。
// 输入：
//   value：待验证条件。
// 输出：
//   void：条件为真正常返回，否则抛出异常。
void Require(bool value) { if (!value) throw std::runtime_error("TEST_FAILED"); }
// 功能：
//   验证错误输入被明确拒绝，而不是产生可使用的场景或诊断。
// 输入：
//   action：预期抛出标准异常的测试操作。
// 输出：
//   void：确实拒绝时通过，否则测试失败。
template<class F> void Reject(F action) {
  bool rejected = false;
  try { action(); } catch (const std::exception &) { rejected = true; }
  Require(rejected);
}
// 功能：
//   检查场景身份、乱序及超时拒绝，并验证真实图像时长统计的有界性和数值稳定性。
// 输入：
//   无。
// 输出：
//   exit_code：所有契约通过为零，异常使测试失败。
int main() {
  Require(Sha256("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  FrameIdentity first{std::string(64, 'a'), Sha256("scene"), 1, 1000000000, 0};
  gz::msgs::Header header;
  WriteIdentity(&header, first);
  Require(ReadIdentity(header, first.epoch).sourceUnixNs == first.sourceUnixNs);
  Reject([&] { WriteIdentity(&header, first); });
  Reject([&] { WriteIdentity(nullptr, first); });
  Reject([&] { ReadIdentity(header, std::string(64, 'b')); });
  Field(&header, "sequence", "1");
  Reject([&] { ReadIdentity(header, first.epoch); });
  header.clear_data();
  WriteIdentity(&header, first);
  header.mutable_data(0)->set_value(0, std::string(2048, 'a'));
  Reject([&] { ReadIdentity(header, first.epoch); });
  for (const std::string value : {"-1", "1.1", "1e3", "true", "", "9223372036854775808"})
    Reject([&] { Decimal(value); });
  Require(Fresh(first, 1250000000));
  Require(!Fresh(first, 1250000001));
  Require(!Fresh(first, 999999999));
  const auto maxTime = std::numeric_limits<std::int64_t>::max();
  Require(SimulationTime(maxTime / 1000000000, maxTime % 1000000000) == maxTime);
  Require(SimulationTime(1, 2) == 1000000002);
  Reject([&] { SimulationTime(maxTime / 1000000000, maxTime % 1000000000 + 1); });
  Reject([] { SimulationTime(-1, 0); });
  Reject([] { SimulationTime(1, 1000000000); });
  SourceOrder order;
  order.Admit(first);
  Reject([&] { order.Admit(first); });
  auto next = first;
  next.sequence += 10;
  next.sourceUnixNs += 50000000;
  next.simulationNs += 50000000;
  order.Admit(next); // Omitted complete frames do not omit structural changes.
  auto changed = next;
  changed.sequence++;
  changed.sourceUnixNs++;
  changed.simulationNs++;
  changed.epoch = std::string(64, 'b');
  Reject([&] { order.Admit(changed); });
  auto reversed = next;
  reversed.sequence++;
  reversed.simulationNs--;
  reversed.sourceUnixNs++;
  Reject([&] { order.Admit(reversed); });
  TimingDistribution distribution;
  for (int i = 0; i < 1100; ++i) distribution.Add(i);
  Require(distribution.TailSize() == TimingDistribution::kTailCapacity);
  Require(distribution.Json().find("\"count\":1100") != std::string::npos);
  Require(distribution.Json().find("\"maximum\":1099") != std::string::npos);
  distribution.Add(std::numeric_limits<double>::infinity());
  distribution.Add(-1);
  Require(distribution.Json().find("\"invalid_count\":2") != std::string::npos);
  Require(distribution.Json().find("\"count\":1100") != std::string::npos);
  TimingDistribution large;
  large.Add(1e308);
  large.Add(1e308);
  Require(large.Json().find("inf") == std::string::npos);
  Require(JsonText(std::string(1, static_cast<char>(0xff))) == "\"\\u00ff\"");
  ImageTransitTimings transit;
  Require(transit.Add({30, 1, 2, 8, 40, 81}));
  Require(!transit.Add({30, -1, 2, 8, 40, 81}));
  Require(!transit.Add({30, 1, 2, 8, 40, std::numeric_limits<double>::infinity()}));
  Require(transit.Json().find("\"invalid_clock_rows\":2") != std::string::npos);
  Require(transit.Json().find("\"source_to_image_ms\":{\"count\":1,\"maximum\":81")
      != std::string::npos);
  std::cout << "scene identity, source ordering, full-snapshot skips and exact age bounds passed\n";
  const int exit_code = 0;
  return exit_code;
}
