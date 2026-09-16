#include "magnetic_field.hpp"
#include <iomanip>
#include <iostream>
#include <string>
// 功能：
//   用实际编译绑定的 PX4 磁场表输出探针值；拒绝尾随文本及无效经纬度，不静默截断参数。
// 输入：
//   argc：必须包含程序名、纬度和经度三个参数。
//   argv：两个以度为单位的经纬度文本。
// 输出：
//   exit_code：成功为 0，参数数量错误为 2，解析或物理边界错误为 1；结果 JSON 写标准输出。
int main(int argc, char **argv) {
  if (argc != 3) return 2;
  try {
    std::size_t latitude_end = 0, longitude_end = 0;
    const float lat = std::stof(argv[1], &latitude_end), lon = std::stof(argv[2], &longitude_end);
    if (latitude_end != std::string(argv[1]).size() || longitude_end != std::string(argv[2]).size()) return 1;
    const auto field = dronedream::MagneticWorldEnuTesla(lat, lon);
    std::cout << std::setprecision(12) << "{\"declination_deg\":"
      << dronedream::MagneticDeclination(lat, lon) << ",\"field_enu_tesla\":["
      << field.X() << ',' << field.Y() << ',' << field.Z() << "]}\n";
  } catch (const std::exception &) { return 1; }
  const int exit_code = 0;
  return exit_code;
}
