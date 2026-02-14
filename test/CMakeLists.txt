cmake_minimum_required(VERSION 3.10)
project(ForTest)

# ####################### 核心修正：直接引入本地GTest源码（不用find_package） #######################
# 1. 定义本地GTest和glog的绝对路径（确认路径存在！）
set(GTEST_SRC_DIR "/usr/kl/googletest")
set(GLOG_SRC_DIR "/usr/kl/glog")
set(Armadillo_SRC_DIR "/usr/kl/armadillo")
set(KL_SRC_DIR "/usr/kl")

# 2. 引入GTest源码（显式指定二进制目录，解决跨目录问题）
# 这一步会自动编译GTest，生成gtest/gtest_main/gmock库
add_subdirectory(${GTEST_SRC_DIR} ${CMAKE_BINARY_DIR}/gtest_build)

# 3. 引入glog源码（如果glog是本地源码，同理处理；如果是系统安装，保留find_package(glog)）
# 如果你系统装了glog，注释下面这行，保留find_package(glog REQUIRED)
add_subdirectory(${GLOG_SRC_DIR} ${CMAKE_BINARY_DIR}/glog_build)

# 引入Armadillo源码（如果需要测试中使用Armadillo）
add_subdirectory(${Armadillo_SRC_DIR} ${CMAKE_BINARY_DIR}/armadillo_build)

# 4. 包含GTest和glog的头文件目录（本地路径）
include_directories(
    ${GTEST_SRC_DIR}/googletest/include
    ${GTEST_SRC_DIR}/googlemock/include
    ${Armadillo_SRC_DIR}/include
    ${GLOG_SRC_DIR}/src/glog # 本地glog头文件
    ${KL_SRC_DIR}/include  # 如果KL库有头文件，包含它
    /usr/local/cuda/include  # CUDA头文件路径
)

# ####################### 收集测试源文件 #######################
# 这里我们直接指定测试源文件，或者你也可以使用file(GLOB ...)来自动收集
file(GLOB BASE_SOURCES 
    /usr/kl/src/base/*.cpp  # 所有.cpp文件
)

set(TEST_SOURCES
    main.cpp                # 测试入口
    base/test_alloc.cpp     # 测试用例文件
)

# ####################### 构建测试可执行文件 #######################
add_executable(alloc_test ${TEST_SOURCES} ${BASE_SOURCES})

# 包含当前目录（找到alloc.h等头文件）
target_include_directories(alloc_test PRIVATE 
    /usr/kl/include/base  # 如果base有头文件，加这个
)

# ####################### 链接库（关键：直接链接本地编译的GTest/glog） #######################
target_link_libraries(alloc_test
    # 本地编译的GTest库（add_subdirectory自动生成的目标）
    gtest
    gtest_main
    gmock
    # 本地编译的glog库（如果是add_subdirectory引入，用glog；如果是find_package，用glog::glog）
    glog
    # Linux下必须的线程库
    pthread
    /usr/local/cuda/lib64/libcudart.so
)

# 启用测试（可选）
enable_testing()
add_test(NAME alloc_test COMMAND alloc_test)
add_definitions(-DGOOGLE_GLOG_NO_GTEST_COMPAT=1)