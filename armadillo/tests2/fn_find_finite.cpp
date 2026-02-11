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


TEST_CASE("fn_find_finite_1", "[find]")
  {
  mat A = 
    "\
     0.061198   0.201990   0.019678  -0.493936  -0.126745   0.051408;\
     0.437242   0.058956  -0.149362  -0.045465   0.296153   0.035437;\
    -0.492474  -0.031309   0.314156   0.419733   0.068317  -0.454499;\
     0.336352   0.411541   0.458476  -0.393139  -0.135040   0.373833;\
     0.239585  -0.428913  -0.406953  -0.291020  -0.353768   0.258704;\
    ";
  
  mat B = A;
  
  B( 6) =  datum::nan;
  B( 8) =  datum::inf;
  B(10) = -datum::inf;
  
  REQUIRE( accu(A.elem(find_finite(A))) == Approx(+0.240136) );
  REQUIRE( accu(B.elem(find_finite(B))) == Approx(-0.250039) );
  
  // REQUIRE_THROWS(  );
  }



TEMPLATE_TEST_CASE("fn_find_finite_fp", "[find]", TEST_FLOAT_TYPES)
  {
  typedef TestType eT;

  Mat<eT> X(5, 1, fill::zeros);

  X[1] =  Datum<eT>::nan;
  X[2] =  Datum<eT>::inf;
  X[3] = -Datum<eT>::inf;

  uvec r = find_finite(X);

  REQUIRE( all( r == uvec({ 0, 4 }) ) );
  }
