# CMake generated Testfile for 
# Source directory: /pfs/wukeliang/KLiteInfer/test
# Build directory: /pfs/wukeliang/KLiteInfer/test/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(operator_test "/pfs/wukeliang/KLiteInfer/test/build/operator_test")
set_tests_properties(operator_test PROPERTIES  _BACKTRACE_TRIPLES "/pfs/wukeliang/KLiteInfer/test/CMakeLists.txt;74;add_test;/pfs/wukeliang/KLiteInfer/test/CMakeLists.txt;0;")
subdirs("gtest_build")
subdirs("glog_build")
subdirs("armadillo_build")
