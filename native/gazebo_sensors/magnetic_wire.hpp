#pragma once
#include <gz/math/Vector3.hh>
#include <sdf/Noise.hh>

namespace dronedream {
// 功能：
//   反解当前 PX4 GZBridge 的 FRD 高斯解码映射，保持物理磁场及噪声，不添加航向补偿。
// 输入：
//   fluTesla：传感器前、左、上坐标系的含噪磁场，单位为特斯拉。
// 输出：
//   wire：传输兼容分量；接收端按 (-y,-x,z) 解码为 FRD 高斯。
inline gz::math::Vector3d Px4GzMagneticWire(const gz::math::Vector3d &fluTesla) {
  const gz::math::Vector3d wire{fluTesla.Y() * 1e4, -fluTesla.X() * 1e4, -fluTesla.Z() * 1e4};
  return wire;
}
// 功能：
//   将传感器声明中的所有磁场幅度噪声从高斯换为特斯拉，只转换副本并保留时间常数。
// 输入：
//   noise：以高斯声明的 SDF 噪声模型副本。
// 输出：
//   noise：保持原随机模型类型、幅度改为特斯拉的噪声模型。
inline sdf::Noise GaussNoiseToTesla(sdf::Noise noise) {
  noise.SetMean(noise.Mean() * 1e-4);
  noise.SetStdDev(noise.StdDev() * 1e-4);
  noise.SetBiasMean(noise.BiasMean() * 1e-4);
  noise.SetBiasStdDev(noise.BiasStdDev() * 1e-4);
  noise.SetPrecision(noise.Precision() * 1e-4);
  noise.SetDynamicBiasStdDev(noise.DynamicBiasStdDev() * 1e-4);
  // Correlation time is seconds, not a field amplitude.
  return noise;
}
}  // namespace dronedream
