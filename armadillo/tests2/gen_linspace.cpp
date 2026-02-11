// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2015 Conrad Sanderson (http://conradsanderson.id.au)
// Copyright 2015 National ICT Australia (NICTA)
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
#include "catch.hpp"
#include "utils.hpp"

using namespace arma;


TEMPLATE_TEST_CASE("gen_linspace_1", "[linspace]", TEST_FLOAT_TYPES)
  {
  typedef TestType eT;

  Col<eT> a = linspace<Col<eT>>(1,5,5);

  constexpr eT margin = is_blas_real<eT>::value ? eT(0.0001) : eT(0.05);
  
  REQUIRE(a(0) == Approx(1.0).margin(margin));
  REQUIRE(a(1) == Approx(2.0).margin(margin));
  REQUIRE(a(2) == Approx(3.0).margin(margin));
  REQUIRE(a(3) == Approx(4.0).margin(margin));
  REQUIRE(a(4) == Approx(5.0).margin(margin));
  
  Col<eT> b = linspace<Col<eT>>(1,5,6);
  
  REQUIRE(b(0) == Approx(1.0).margin(margin));
  REQUIRE(b(1) == Approx(1.8).margin(margin));
  REQUIRE(b(2) == Approx(2.6).margin(margin));
  REQUIRE(b(3) == Approx(3.4).margin(margin));
  REQUIRE(b(4) == Approx(4.2).margin(margin));
  REQUIRE(b(5) == Approx(5.0).margin(margin));
  
  Row<eT> c = linspace<Row<eT>>(1,5,6);
  
  REQUIRE(c(0) == Approx(1.0).margin(margin));
  REQUIRE(c(1) == Approx(1.8).margin(margin));
  REQUIRE(c(2) == Approx(2.6).margin(margin));
  REQUIRE(c(3) == Approx(3.4).margin(margin));
  REQUIRE(c(4) == Approx(4.2).margin(margin));
  REQUIRE(c(5) == Approx(5.0).margin(margin));
  
  Mat<eT> X = linspace<Mat<eT>>(1,5,6);
  
  REQUIRE(X.n_rows == 6);
  REQUIRE(X.n_cols == 1);
  }
