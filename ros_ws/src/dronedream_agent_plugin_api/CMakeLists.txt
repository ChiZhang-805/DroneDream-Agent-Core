cmake_minimum_required(VERSION 3.16)
project(dronedream_agent_plugin_api)

find_package(ament_cmake REQUIRED)
find_package(dronedream_agent_msgs REQUIRED)
find_package(lifecycle_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)

# 插件只暴露策略契约；实际安全事件由独立生命周期主机交回固定内核。
add_library(dronedream_runtime_plugins SHARED
  src/safe_hold_capability.cpp
)
target_compile_features(dronedream_runtime_plugins PUBLIC cxx_std_17)
target_include_directories(dronedream_runtime_plugins PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(dronedream_runtime_plugins pluginlib)

add_executable(capability_host src/capability_host.cpp)
target_compile_features(capability_host PUBLIC cxx_std_17)
target_include_directories(capability_host PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(
  capability_host dronedream_agent_msgs lifecycle_msgs pluginlib rclcpp rclcpp_lifecycle
)

add_executable(plugin_probe src/plugin_probe.cpp)
target_compile_features(plugin_probe PUBLIC cxx_std_17)
target_include_directories(plugin_probe PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(plugin_probe pluginlib)

# 同一构建产物通过 pluginlib 真正加载，不以静态源码检查代替动态生命周期自检。
enable_testing()
add_test(NAME native-plugin-lifecycle COMMAND plugin_probe)
add_test(NAME native-host-lifecycle COMMAND capability_host --self-test)
add_test(NAME native-host-rejects-negative-deadline
  COMMAND capability_host --ros-args -p contract_id:=test-contract -p watchdog_deadline_ms:=-1)
set_tests_properties(native-host-rejects-negative-deadline PROPERTIES WILL_FAIL TRUE)

pluginlib_export_plugin_description_file(dronedream_agent_plugin_api plugins.xml)

install(
  TARGETS dronedream_runtime_plugins capability_host plugin_probe
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/ DESTINATION include)
install(FILES plugins.xml DESTINATION share/${PROJECT_NAME})

ament_export_include_directories(include)
ament_export_dependencies(pluginlib)
ament_package()
