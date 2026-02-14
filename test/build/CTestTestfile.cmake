# CMake generated Testfile for 
# Source directory: /usr/kl/test
# Build directory: /usr/kl/test/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(alloc_test "/usr/kl/test/build/alloc_test")
set_tests_properties(alloc_test PROPERTIES  _BACKTRACE_TRIPLES "/usr/kl/test/CMakeLists.txt;66;add_test;/usr/kl/test/CMakeLists.txt;0;")
subdirs("gtest_build")
subdirs("glog_build")
subdirs("armadillo_build")
