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


TEMPLATE_TEST_CASE("fn_cross_1", "[cross]", TEST_FLOAT_TYPES)
  {
  typedef TestType eT;

  Col<eT> a = { eT(0.1), eT(2.3), eT( 4.5) };
  Col<eT> b = { eT(6.7), eT(8.9), eT(10.0) };

  Col<eT> c = { eT(-17.050), eT(29.150), eT(-14.520) };

  constexpr eT margin = is_blas_real<eT>::value ? eT(0.001) : eT(0.1);

  REQUIRE( accu(abs(cross(a,b) - c)) == Approx(eT(0)).margin(margin) );
  }



TEST_CASE("fn_cross_invalid", "[cross]")
  {
  vec x;

  REQUIRE_THROWS( x = cross(randu<vec>(4), randu<vec>(4)) );
  }
