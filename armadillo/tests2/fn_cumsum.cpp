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


TEMPLATE_TEST_CASE("fn_cumsum_1", "[cumsum]", TEST_FLOAT_TYPES)
  {
  typedef TestType eT;

  Col<eT> a = linspace<Col<eT>>(1,5,6);
  Row<eT> b = linspace<Row<eT>>(1,5,6);

  Col<eT> c = { eT(1.0000), eT(2.8000), eT(5.4000), eT(8.8000), eT(13.0000), eT(18.0000) };

  constexpr eT margin = is_blas_real<eT>::value ? eT(0.001) : eT(0.2);

  REQUIRE( accu(abs(cumsum(a) - c    )) == Approx(0.0).margin(margin) );
  REQUIRE( accu(abs(cumsum(b) - c.t())) == Approx(0.0).margin(margin) );

  REQUIRE_THROWS( b = cumsum(a) );
  }



TEST_CASE("fn_cumsum_2", "[cumsum]")
  {
  mat A =
    {
    { -0.78838,  0.69298,  0.41084,  0.90142 },
    {  0.49345, -0.12020,  0.78987,  0.53124 },
    {  0.73573,  0.52104, -0.22263,  0.40163 }
    };
  
  mat B =
    {
    { -0.78838,  0.69298,  0.41084,  0.90142 },
    { -0.29493,  0.57278,  1.20071,  1.43266 },
    {  0.44080,  1.09382,  0.97808,  1.83429 }
    };
  
  mat C =
    {
    {-0.78838, -0.09540,  0.31544,  1.21686 },
    { 0.49345,  0.37325,  1.16312,  1.69436 },
    { 0.73573,  1.25677,  1.03414,  1.43577 }
    };
  
  REQUIRE( accu(abs(cumsum(A)   - B)) == Approx(0.0).margin(0.001) );
  REQUIRE( accu(abs(cumsum(A,0) - B)) == Approx(0.0).margin(0.001) );
  REQUIRE( accu(abs(cumsum(A,1) - C)) == Approx(0.0).margin(0.001) );
  }



