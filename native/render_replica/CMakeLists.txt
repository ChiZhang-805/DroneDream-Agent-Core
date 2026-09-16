cmake_minimum_required(VERSION 3.22)
# 实时渲染副本是传感器源的一部分；源端与副本必须使用同一 Gazebo 主版本和协议。
project(dronedream_render_replica LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(gz-sim8 REQUIRED)
find_package(gz-transport13 REQUIRED)
find_package(gz-plugin2 REQUIRED COMPONENTS register)
find_package(OpenSSL REQUIRED)
find_package(Threads REQUIRED)
add_library(dronedream-render-source SHARED scene_source.cpp)
target_link_libraries(dronedream-render-source PRIVATE gz-sim8::gz-sim8
  gz-transport13::gz-transport13 gz-plugin2::register OpenSSL::Crypto Threads::Threads)
target_compile_options(dronedream-render-source PRIVATE -Wall -Wextra -Werror)
add_executable(dronedream-render-replica render_replica.cpp)
# Gazebo's generated components.hh is not installed. Register the installed
# component serializers explicitly so complete ECM snapshots never omit types.
file(GLOB COMPONENT_HEADERS CONFIGURE_DEPENDS
  /usr/include/gz/sim8/gz/sim/components/*.hh)
list(SORT COMPONENT_HEADERS)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/component_registry.hpp" "#pragma once\n")
foreach(COMPONENT_HEADER IN LISTS COMPONENT_HEADERS)
  get_filename_component(COMPONENT_NAME "${COMPONENT_HEADER}" NAME)
  file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/component_registry.hpp"
    "#include <gz/sim/components/${COMPONENT_NAME}>\n")
endforeach()
target_include_directories(dronedream-render-replica PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(dronedream-render-replica PRIVATE gz-sim8::gz-sim8
  gz-transport13::gz-transport13 OpenSSL::Crypto ${CMAKE_DL_LIBS})
target_compile_options(dronedream-render-replica PRIVATE -Wall -Wextra -Werror)
enable_testing()
add_executable(replica-contract-test replica_contract_test.cpp)
target_link_libraries(replica-contract-test PRIVATE gz-transport13::gz-transport13
  OpenSSL::Crypto)
target_compile_options(replica-contract-test PRIVATE -Wall -Wextra -Werror)
add_test(NAME replica-contract COMMAND replica-contract-test)
add_executable(replica-scene-test replica_scene_test.cpp scene_capture_test.cpp model_sdf_state_test.cpp)
target_link_libraries(replica-scene-test PRIVATE gz-sim8::gz-sim8
  gz-transport13::gz-transport13 OpenSSL::Crypto Threads::Threads)
target_compile_options(replica-scene-test PRIVATE -Wall -Wextra -Werror)
add_test(NAME replica-scene COMMAND replica-scene-test)
set_tests_properties(replica-contract replica-scene PROPERTIES TIMEOUT 30)
