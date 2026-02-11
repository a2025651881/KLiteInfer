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


TEST_CASE("fn_cumprod_1", "[cumprod]")
  {
  colvec a = linspace<colvec>(1,5,6);
  rowvec b = linspace<rowvec>(1,5,6);
  
  colvec c = { 1.0000, 1.8000, 4.6800, 15.9120, 66.8304, 334.1520 };

  REQUIRE( accu(abs(cumprod(a) - c    )) == Approx(0.0).margin(0.001) );
  REQUIRE( accu(abs(cumprod(b) - c.t())) == Approx(0.0).margin(0.001) );
  
  REQUIRE_THROWS( b = cumprod(a) );
  }



TEST_CASE("fn_cumprod_2", "[cumprod]")
  {
  mat A =
    {
    { -0.78838,  0.69298,  0.41084,  0.90142 },
    {  0.49345, -0.12020,  0.78987,  0.53124 },
    {  0.73573,  0.52104, -0.22263,  0.40163 }
    };
  
  mat B =
    {
    { -0.788380,  0.692980,  0.410840,  0.901420 },
    { -0.389026, -0.083296,  0.324510,  0.478870 },
    { -0.286218, -0.043401, -0.072246,  0.192329 }

    };
  
  mat C =
    {
    { -0.788380, -0.546332, -0.224455, -0.202328 },
    {  0.493450, -0.059313, -0.046849, -0.024888 },
    {  0.735730,  0.383345, -0.085344, -0.034277 }
    };
  
  REQUIRE( accu(abs(cumprod(A)   - B)) == Approx(0.0).margin(0.001) );
  REQUIRE( accu(abs(cumprod(A,0) - B)) == Approx(0.0).margin(0.001) );
  REQUIRE( accu(abs(cumprod(A,1) - C)) == Approx(0.0).margin(0.001) );
  }



TEMPLATE_TEST_CASE("fn_cumprod_small_fp", "[cumprod]", TEST_FLOAT_TYPES)
  {
  typedef TestType eT;

  Col<eT> a = linspace<Col<eT>>(1,4,5);
  Row<eT> b = linspace<Row<eT>>(1,4,5);
  
  Col<eT> c = { eT(1.0000), eT(1.7500), eT(4.3750), eT(14.2188), eT(56.8750) };

  constexpr eT margin = is_blas_real<eT>::value ? eT(0.001) : eT(0.1);

  REQUIRE( accu(abs(cumprod(a) - c    )) == Approx(eT(0)).margin(margin) );
  REQUIRE( accu(abs(cumprod(b) - c.t())) == Approx(eT(0)).margin(margin) );
  }
