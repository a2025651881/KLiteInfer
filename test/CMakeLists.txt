find_package(GTest REQUIRED)

file(GLOB TEST_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/base_test/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/op_test/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/serve_test/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tensor_test/*.cpp)
file(GLOB TEST_CUDA_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

add_executable(test_llm ${TEST_SRCS} ${TEST_CUDA_SRCS})

# apps/ 提供 http_util.h（服务端与客户端共用的极简 HTTP 工具）
target_include_directories(test_llm PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../apps)
target_link_libraries(test_llm PRIVATE klite GTest::gtest)

set_target_properties(test_llm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_test(NAME test_llm COMMAND test_llm)
