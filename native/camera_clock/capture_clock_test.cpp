#include "capture_clock.hpp"
#include <stdexcept>
// 功能：不依赖 assert/NDEBUG 的真实失败断言。输入：条件。输出：失败抛出异常。
void Check(bool value) { if (!value) throw std::runtime_error("CAPTURE_CLOCK_TEST_FAILED"); }
// 功能：验证精确配对、双钟、暂停、回退和有界缓存。输入：无。输出：成功返回零。
int main() {
  dronedream::CaptureClock c;
  Check(c.Add(1, 1000000000, 2000000000, false));
  Check(c.Find(1, 1100000000, 2100000000).has_value());
  Check(!c.Find(2, 1100000000, 2100000000));
  Check(!c.Find(1, 1300000000, 2300000000));
  Check(!c.Find(1, 1100000000, 2200000000));
  Check(c.Add(1, 1100000000, 2100000000, true));
  Check(!c.Find(1, 1100000000, 2100000000));
  Check(c.Add(2, 1200000000, 2200000000, false));
  Check(!c.Add(1, 1200000001, 2200000001, false));
  Check(!c.Add(3, 1200000002, 2200000002, false));
  dronedream::CaptureClock capacity;
  for (int i=1; i<=4097; ++i) Check(capacity.Add(i, 1000000000+i, 2000000000+i, false));
  Check(!capacity.Find(1, 1000005000, 2000005000));
  Check(capacity.Find(4097, 1000005000, 2000005000).has_value());
  // 功能：一次墙钟偏移不永久停流，但跳变前及跳变当帧均不得获得新租期。
  dronedream::CaptureClock recovery;
  Check(recovery.Add(1, 1000000000, 2000000000, false));
  Check(recovery.Add(2, 1110000000, 2010000000, false));
  Check(!recovery.Find(1, 1110000000, 2010000000));
  Check(!recovery.Find(2, 1110000000, 2010000000));
  Check(recovery.Add(3, 1120000000, 2020000000, false));
  Check(recovery.Find(3, 1120000000, 2020000000).has_value());
  Check(recovery.Add(3, 1220000000, 2120000000, false));
  Check(recovery.Find(3, 1220000000, 2120000000)->unixNs == 1120000000);
  Check(recovery.Add(3, 1420000000, 2320000000, false));
  Check(!recovery.Find(3, 1420000000, 2320000000));
  // 功能：暂停后同 tick 不致永久失败，也不能还原已清空的旧图像映射。
  Check(recovery.Add(3, 1420000000, 2320000000, true));
  Check(recovery.Add(3, 1430000000, 2330000000, false));
  Check(!recovery.Find(3, 1430000000, 2330000000));
  Check(recovery.Add(4, 1440000000, 2340000000, false));
  Check(recovery.Find(4, 1440000000, 2340000000).has_value());
  Check(!recovery.Add(2, 1450000000, 2350000000, false));
  Check(!recovery.Add(5, 1460000000, 2360000000, false));
  return 0;
}
