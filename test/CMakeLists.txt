cmake_minimum_required(VERSION 3.10)
project(ForTest)

# 启用 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 启用 CUDA
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# 路径配置（你的本地源码路径）
set(GTEST_SRC_DIR "/pfs/wukeliang/googletest")
set(GLOG_SRC_DIR "/pfs/wukeliang/glog")
set(Armadillo_SRC_DIR "/pfs/wukeliang/armadillo")
set(KL_SRC_DIR "/pfs/wukeliang/KLiteInfer")

# 编译第三方库
add_subdirectory(${GTEST_SRC_DIR} ${CMAKE_BINARY_DIR}/gtest_build)
add_subdirectory(${GLOG_SRC_DIR} ${CMAKE_BINARY_DIR}/glog_build)
add_subdirectory(${Armadillo_SRC_DIR} ${CMAKE_BINARY_DIR}/armadillo_build)

# 包含头文件目录
include_directories(
    ${GTEST_SRC_DIR}/googletest/include
    ${GTEST_SRC_DIR}/googlemock/include
    ${GLOG_SRC_DIR}/include
    ${Armadillo_SRC_DIR}/include 
    ${KL_SRC_DIR}/include
    ${KL_SRC_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
    ./
)

# 收集源文件
file(GLOB_RECURSE BASE_SOURCES
    ${KL_SRC_DIR}/src/base/*.cpp
    ${KL_SRC_DIR}/src/op/*.cpp
    ${KL_SRC_DIR}/src/op/kernels/cpu/*.cpp
    ${KL_SRC_DIR}/src/op/kernels/gpu/*.cu    
    ${KL_SRC_DIR}/src/tensor/*.cpp
    ${KL_SRC_DIR}/src/config/*.cpp
    ${KL_SRC_DIR}/src/sampler/*.cpp
)

file(GLOB TEST_SOURCES
    ${KL_SRC_DIR}/test/main.cpp              
    ${KL_SRC_DIR}/test/op_test/*.cpp         
    ${KL_SRC_DIR}/test/base_test/*.cpp
)

# 生成测试可执行文件
add_executable(operator_test
    ${TEST_SOURCES}
    ${BASE_SOURCES}
)

# 链接库
target_link_libraries(operator_test
    gtest
    gtest_main
    gmock
    glog
    pthread
    CUDA::cudart
    CUDA::cublas
)

# 宏定义
add_definitions(-DGOOGLE_GLOG_NO_GTEST_COMPAT=1)

# 启用测试
enable_testing()
add_test(NAME operator_test COMMAND operator_test)