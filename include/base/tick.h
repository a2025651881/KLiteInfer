#ifndef KELI_INCLUDE_BASE_TICK_H_
#define KELI_INCLUDE_BASE_TICK_H_
#include <chrono>
#include <iostream>

#define TICK(x) auto bench_##x = std::chrono::steady_clock::now();
#define TOCK(x)     \
  printf("%s: %lfs\n", #x,                                          \
         std::chrono::duration_cast<std::chrono::duration<double>>( \
             std::chrono::steady_clock::now() - bench_##x)          \
             .count());
#endif