// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Ryan Curtin (http://www.ratml.org/)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------



#include <armadillo>

#if defined(ARMA_HAVE_FP16)
  #define TEST_FLOAT_TYPES double, float, fp16
  #define TEST_CX_FLOAT_TYPES cx_double, cx_float, cx_fp16
#else
  #define TEST_FLOAT_TYPES double, float
  #define TEST_CX_FLOAT_TYPES cx_double, cx_float
#endif

