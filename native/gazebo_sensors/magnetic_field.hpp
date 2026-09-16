#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <gz/math/Vector3.hh>
// Copied by CMake from this Runtime's PX4 source, with original license intact.
#include "geo_magnetic_tables.hpp"

namespace dronedream {
// 功能：
//   对当前 PX4 磁场表做双线性插值；拒绝非法经纬度，并在表覆盖纬度边界内取值。
// 输入：
//   lat、lon：地理纬度和经度，单位为度。
//   table：同一 Runtime 的 PX4 原始世界磁场查找表。
// 输出：
//   interpolated：保留表原始量纲的插值值，调用方负责应用对应缩放系数。
inline float MagneticLookup(float lat, float lon, const int16_t table[LAT_DIM][LON_DIM]) {
  if (!std::isfinite(lat) || !std::isfinite(lon) || lat < -90 || lat > 90 ||
      lon < -180 || lon > 180) throw std::invalid_argument("magnetic coordinates invalid");
  lat = std::clamp(lat, SAMPLING_MIN_LAT, SAMPLING_MAX_LAT);
  const float south = std::clamp(std::floor(lat / SAMPLING_RES) * SAMPLING_RES,
                                SAMPLING_MIN_LAT, SAMPLING_MAX_LAT - SAMPLING_RES);
  const float west = std::clamp(std::floor(lon / SAMPLING_RES) * SAMPLING_RES,
                               SAMPLING_MIN_LON, SAMPLING_MAX_LON - SAMPLING_RES);
  const unsigned row = (south - SAMPLING_MIN_LAT) / SAMPLING_RES;
  const unsigned col = (west - SAMPLING_MIN_LON) / SAMPLING_RES;
  const float x = std::clamp((lon - west) / SAMPLING_RES, 0.f, 1.f);
  const float y = std::clamp((lat - south) / SAMPLING_RES, 0.f, 1.f);
  const float low = x * (table[row][col+1] - table[row][col]) + table[row][col];
  const float high = x * (table[row+1][col+1] - table[row+1][col]) + table[row+1][col];
  const float interpolated = y * (high-low) + low;
  return interpolated;
}
// 功能：
//   用已绑定的 PX4 磁偏角表查询地磁北与地理北的夹角，不混入机体姿态偏置。
// 输入：
//   lat、lon：地理纬度和经度，单位为度。
// 输出：
//   declination：磁偏角，单位为度。
inline double MagneticDeclination(float lat, float lon) {
  const double declination = MagneticLookup(lat, lon, declination_table) * WMM_DECLINATION_SCALE_TO_DEGREES;
  return declination;
}
// 功能：
//   把地磁强度、磁倾角和磁偏角变成世界 ENU 磁场，明确北向分量及向下倾角的符号。
// 输入：
//   lat、lon：地理纬度和经度，单位为度。
// 输出：
//   field：世界东、北、上三个分量，单位为特斯拉。
inline gz::math::Vector3d MagneticWorldEnuTesla(float lat, float lon) {
  const double radians = std::acos(-1.) / 180.;
  const double declination = MagneticDeclination(lat, lon) * radians;
  const double inclination = MagneticLookup(lat, lon, inclination_table) *
                             WMM_INCLINATION_SCALE_TO_DEGREES * radians;
  const double strength = MagneticLookup(lat, lon, totalintensity_table) *
                          WMM_TOTALINTENSITY_SCALE_TO_NANOTESLA * 1e-9;
  const gz::math::Vector3d field{strength * std::cos(inclination) * std::sin(declination),
          strength * std::cos(inclination) * std::cos(declination),
          -strength * std::sin(inclination)};
  return field;
}
}  // namespace dronedream
