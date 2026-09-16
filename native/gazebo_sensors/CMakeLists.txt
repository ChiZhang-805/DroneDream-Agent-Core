cmake_minimum_required(VERSION 3.22)
project(dronedream_native_sensors LANGUAGES CXX)
# 当前 PX4 源码提供唯一磁场表；禁止因缺少路径而回退到另一 Runtime 的旧表。
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT EXISTS "${DRONEDREAM_PX4_ROOT}/src/lib/world_magnetic_model/geo_magnetic_tables.hpp")
  message(FATAL_ERROR "An explicit matching PX4 source root is required")
endif()
configure_file("${DRONEDREAM_PX4_ROOT}/src/lib/world_magnetic_model/geo_magnetic_tables.hpp"
  "${CMAKE_CURRENT_BINARY_DIR}/geo_magnetic_tables.hpp" COPYONLY)
include_directories("${CMAKE_CURRENT_BINARY_DIR}")
find_package(gz-sim8 REQUIRED)
find_package(gz-sensors8 REQUIRED)
find_package(gz-plugin2 REQUIRED COMPONENTS register)
add_library(dronedream-magnetometer SHARED magnetometer_system.cpp)
target_link_libraries(dronedream-magnetometer PRIVATE gz-sim8::gz-sim8
  gz-sensors8::gz-sensors8 gz-sensors8::gz-sensors8-magnetometer
  gz-plugin2::register)
target_compile_options(dronedream-magnetometer PRIVATE -Wall -Wextra -Werror)
enable_testing()
# 坐标及量纲测试使用与产品插件相同的头文件和磁场表。
add_executable(sensor-frame-test sensor_frame_test.cpp)
target_link_libraries(sensor-frame-test PRIVATE gz-sim8::gz-sim8)
add_test(NAME sensor-frame-contract COMMAND sensor-frame-test)
add_executable(magnetic-field-probe magnetic_field_probe.cpp)
target_link_libraries(magnetic-field-probe PRIVATE gz-sim8::gz-sim8)
add_test(NAME magnetic-probe-rejects-trailing-text COMMAND magnetic-field-probe "30invalid" "120")
set_tests_properties(magnetic-probe-rejects-trailing-text PROPERTIES WILL_FAIL TRUE)
