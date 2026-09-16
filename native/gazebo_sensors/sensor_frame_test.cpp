#include "magnetic_wire.hpp"
#include "magnetic_field.hpp"
#include <gz/math/Quaternion.hh>
#include <cmath>
#include <iostream>

// 功能：
//   验证地磁表边界、噪声量纲与完整姿态下 ENU/FLU/FRD 接线往返，不依赖固定航向修正。
// 输入：
//   无。
// 输出：
//   exit_code：全部几何及噪声检查通过为 0；非零表示对应约束失败。
int main() {
  for (float latitude : {-90.f, -30.f, 0.f, 30.f, 90.f})
    for (float longitude : {-180.f, -120.f, 0.f, 120.f, 180.f}) {
      const auto field = dronedream::MagneticWorldEnuTesla(latitude, longitude);
      if (!field.IsFinite() || field.Length() < 1e-6 || field.Length() > 1e-3) return 4;
      const double declared = dronedream::MagneticDeclination(latitude, longitude);
      const double recovered = std::atan2(field.X(), field.Y()) * 180. / std::acos(-1.);
      if (std::abs(declared - recovered) > 1e-4) return 5;
    }
  sdf::Noise noise;
  noise.SetStdDev(.0001);
  noise.SetBiasMean(.0002);
  noise.SetDynamicBiasCorrelationTime(10.);
  const auto physicalNoise = dronedream::GaussNoiseToTesla(noise);
  if (std::abs(physicalNoise.StdDev() - 1e-8) > 1e-16 ||
      std::abs(physicalNoise.BiasMean() - 2e-8) > 1e-16 ||
      physicalNoise.DynamicBiasCorrelationTime() != 10.) return 3;
  const gz::math::Vector3d world{6e-6, 23e-6, -42e-6};
  for (double roll : {-0.5, 0., 0.5})
    for (double pitch : {-0.5, 0., 0.5})
      for (double yaw : {-3., -1.5, 0., 1.5, 3.}) {
        const gz::math::Quaterniond orientation(roll, pitch, yaw);
        const auto flu = orientation.Inverse().RotateVector(world);
        const auto wire = dronedream::Px4GzMagneticWire(flu);
        const gz::math::Vector3d decodedFrd{-wire.Y(), -wire.X(), wire.Z()};
        const gz::math::Vector3d expectedFrd{flu.X()*1e4, -flu.Y()*1e4, -flu.Z()*1e4};
        if ((decodedFrd - expectedFrd).Length() > 1e-12) return 1;
        const auto recovered = orientation.RotateVector(
          {decodedFrd.X()/1e4, -decodedFrd.Y()/1e4, -decodedFrd.Z()/1e4});
        if ((recovered - world).Length() > 1e-12) return 2;
      }
  std::cout << "45 full-attitude magnetic frame round trips passed\n";
  const int exit_code = 0;
  return exit_code;
}
