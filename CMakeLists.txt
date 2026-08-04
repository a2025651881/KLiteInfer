cmake_minimum_required(VERSION 3.20)
project(KLiteInfer LANGUAGES CXX CUDA)

# ------------------------------------------------------------------
# 基础设置
# ------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif ()

# 本机 GPU 架构（H20 = 90）；可用 -DCMAKE_CUDA_ARCHITECTURES=80 覆盖
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif ()

option(QWEN3_SUPPORT "Enable Qwen3 / BPE(tiktoken) tokenizer support" ON)
option(KLITE_BUILD_TESTS "Build gtest unit tests" OFF)

# 本机手工编译安装的依赖根目录（可通过 -D覆盖）
set(SENTENCEPIECE_ROOT "/root/sentencepiece" CACHE PATH "sentencepiece source/build root")

# ------------------------------------------------------------------
# 依赖查找
# ------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)
find_package(glog REQUIRED)
find_package(Armadillo REQUIRED)

# ---- sentencepiece（encode.h 无条件依赖）----
find_path(SENTENCEPIECE_INCLUDE_DIR
        NAMES sentencepiece_processor.h
        HINTS ${SENTENCEPIECE_ROOT}/src
              ${SENTENCEPIECE_ROOT}/build/src/installed_headers
              /usr/local/include /usr/include)
find_library(SENTENCEPIECE_LIBRARY
        NAMES sentencepiece
        HINTS ${SENTENCEPIECE_ROOT}/build/src
              /usr/local/lib /usr/lib)
if (NOT SENTENCEPIECE_INCLUDE_DIR OR NOT SENTENCEPIECE_LIBRARY)
    message(FATAL_ERROR "sentencepiece not found. 请设置 -DSENTENCEPIECE_ROOT=<sentencepiece 源码/构建目录>")
endif ()
message(STATUS "sentencepiece: ${SENTENCEPIECE_LIBRARY}")

# sentencepiece_processor.h 内部会 #include "third_party/absl/..."，需要仓库根目录
set(KLITE_INCLUDE_EXT ${SENTENCEPIECE_INCLUDE_DIR})
if (EXISTS ${SENTENCEPIECE_ROOT}/third_party/absl/strings/string_view.h)
    list(APPEND KLITE_INCLUDE_EXT ${SENTENCEPIECE_ROOT})
endif ()
set(KLITE_LINK_EXT ${SENTENCEPIECE_LIBRARY})

# ---- BPE tokenizer依赖：absl / re2 / nlohmann_json ----
if (QWEN3_SUPPORT)
    find_path(ABSL_INCLUDE_DIR
            NAMES absl/strings/str_split.h
            HINTS ${SENTENCEPIECE_ROOT}/build/_deps/abseil-cpp-src
                  /usr/local/include /usr/include)
    file(GLOB ABSL_LIBRARIES
            ${SENTENCEPIECE_ROOT}/build/_deps/abseil-cpp-build/absl/*/libabsl_*.a)
    if (NOT ABSL_LIBRARIES)
        find_package(absl REQUIRED)
        set(ABSL_LIBRARIES absl::strings)
    endif ()

    find_path(NLOHMANN_INCLUDE_DIR
            NAMES nlohmann/json.hpp
            HINTS /usr/local/include /usr/include /root/miniconda3/include)

    find_path(RE2_INCLUDE_DIR NAMES re2/re2.h HINTS /usr/local/include /usr/include)
    find_library(RE2_LIBRARY NAMES re2 HINTS /usr/local/lib /usr/lib /lib/x86_64-linux-gnu)

    if (NOT ABSL_INCLUDE_DIR OR NOT NLOHMANN_INCLUDE_DIR OR NOT RE2_LIBRARY)
        message(FATAL_ERROR "QWEN3_SUPPORT 需要 absl / nlohmann_json / re2，未全部找到")
    endif ()
    message(STATUS "absl include: ${ABSL_INCLUDE_DIR}")
    message(STATUS "nlohmann include: ${NLOHMANN_INCLUDE_DIR}")
    message(STATUS "re2: ${RE2_LIBRARY}")

    list(APPEND KLITE_INCLUDE_EXT ${ABSL_INCLUDE_DIR} ${NLOHMANN_INCLUDE_DIR} ${RE2_INCLUDE_DIR})
    # absl 为静态库，用 --start-group 规避链接顺序问题
    list(APPEND KLITE_LINK_EXT "-Wl,--start-group" ${ABSL_LIBRARIES} "-Wl,--end-group" ${RE2_LIBRARY})
endif ()

# ------------------------------------------------------------------
# 核心库
# ------------------------------------------------------------------
file(GLOB_RECURSE KLITE_CXX_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE KLITE_CUDA_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu)

add_library(klite STATIC ${KLITE_CXX_SRCS} ${KLITE_CUDA_SRCS})

target_include_directories(klite PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${ARMADILLO_INCLUDE_DIRS}
        ${KLITE_INCLUDE_EXT})

target_link_libraries(klite PUBLIC
        glog::glog
        ${ARMADILLO_LIBRARIES}
        CUDA::cudart
        CUDA::cublas
        ${KLITE_LINK_EXT})

if (QWEN3_SUPPORT)
    target_compile_definitions(klite PUBLIC QWEN3_SUPPORT)
endif ()

set_target_properties(klite PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_compile_options(klite PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

# ------------------------------------------------------------------
# 可执行示例
# ------------------------------------------------------------------
add_executable(demo ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(demo PRIVATE klite)
set_target_properties(demo PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ------------------------------------------------------------------
# 推理服务：server常驻模型，client 只依赖 socket
# ------------------------------------------------------------------
add_executable(klite_server ${CMAKE_CURRENT_SOURCE_DIR}/apps/server.cpp)
target_include_directories(klite_server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/apps)
target_link_libraries(klite_server PRIVATE klite)
set_target_properties(klite_server PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(klite_client ${CMAKE_CURRENT_SOURCE_DIR}/apps/client.cpp)
target_include_directories(klite_client PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/apps)

# ------------------------------------------------------------------
# 单元测试
# ------------------------------------------------------------------
if (KLITE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif ()
