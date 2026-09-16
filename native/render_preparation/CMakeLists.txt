cmake_minimum_required(VERSION 3.22)
project(dronedream_render_preparation LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(gz-sim8 REQUIRED)
find_package(gz-plugin2 REQUIRED COMPONENTS register)
find_package(OpenSSL REQUIRED)
find_package(OpenGL REQUIRED COMPONENTS OpenGL)
# 与实际 Runtime 的 Ogre Next 保持一致；原生库不能混用 Ogre 1.x 的同名符号。
find_path(OGRE_INCLUDE OgreHlmsDiskCache.h PATHS /usr/include/OGRE-2.3 NO_DEFAULT_PATH REQUIRED)
# The generic OgreMain name on this runtime belongs to Ogre 1.9, whose C++
# symbols collide with Next. Never search it or inherit its cached CMake value.
find_library(OGRE_NEXT_MAIN OgreNextMain
  PATHS /usr/lib/x86_64-linux-gnu/OGRE-2.3 NO_DEFAULT_PATH REQUIRED)
add_library(dronedream-render-preparation SHARED render_preparation.cpp)
target_include_directories(dronedream-render-preparation SYSTEM PRIVATE ${OGRE_INCLUDE})
target_link_libraries(dronedream-render-preparation PRIVATE gz-sim8::gz-sim8
  gz-plugin2::register
  ${OGRE_NEXT_MAIN} OpenSSL::Crypto OpenGL::OpenGL)
target_compile_options(dronedream-render-preparation PRIVATE -Wall -Wextra -Werror)
enable_testing()
add_executable(cache-contract-test cache_contract_test.cpp)
target_link_libraries(cache-contract-test PRIVATE OpenSSL::Crypto)
target_compile_options(cache-contract-test PRIVATE -Wall -Wextra -Werror)
add_test(NAME cache-contract COMMAND cache-contract-test)
