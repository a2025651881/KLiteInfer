// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2015 Conrad Sanderson (http://conradsanderson.id.au)
// Copyright 2015 National ICT Australia (NICTA)
// Copyright 2025 Ryan Curtin (http://ratml.org)
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


TEST_CASE("fn_find_nonnan_1", "[find]")
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

  uvec indices1 = find_nonnan(A);
  uvec indices2 = find_nonnan(B);

  REQUIRE( indices1.n_elem == A.n_elem );
  REQUIRE( indices2.n_elem == B.n_elem - 1 );

  for (uword i = 0; i < A.n_elem; ++i)
    {
    REQUIRE( indices1[i] == i );
    }

  for (uword i = 0; i < B.n_elem - 1; ++i)
    {
    if (i < 6)
      {
      REQUIRE( indices2[i] == i );
      }
    else
      {
      REQUIRE( indices2[i] == i + 1 );
      }
    }
  }



TEST_CASE("fn_find_nonnan_cube", "[find]")
  {
  cube A(5, 4, 3, fill::randu);

  A(1, 2, 1) = datum::nan;
  A(2, 3, 1) = datum::nan;
  A(3, 1, 1) = datum::inf;

  uvec indices = find_nonnan(A);

  REQUIRE( indices.n_elem == A.n_elem - 2 );

  for (uword i = 0; i < A.n_elem - 2; ++i)
    {
    if (i < (1 + (2 * 5) + (1 * 4 * 5)))
      {
      REQUIRE( indices[i] == i );
      }
    else if (i < ((2 + (3 * 5) + (1 * 4 * 5)) - 1))
      {
      REQUIRE( indices[i] == i + 1 );
      }
    else
      {
      REQUIRE( indices[i] == i + 2 );
      }
    }
  }



TEST_CASE("fn_find_nonnan_spmat", "[find]")
  {
  // sparse matrices will only return nonzero non-nan indices
  sp_mat A(10, 10);

  A(3, 4) = 1.0;
  A(4, 5) = 1.0;
  A(5, 6) = datum::inf;
  A(6, 6) = datum::nan;
  A(6, 7) = datum::inf;
  A(8, 9) = 1.0;

  uvec indices = find_nonnan(A);

  REQUIRE( indices.n_elem == 5 );
  REQUIRE( indices[0] == 43 );
  REQUIRE( indices[1] == 54 );
  REQUIRE( indices[2] == 65 );
  REQUIRE( indices[3] == 76 );
  REQUIRE( indices[4] == 98 );
  }



TEMPLATE_TEST_CASE("fn_find_nonnan_fp", "[find]", TEST_FLOAT_TYPES)
  {
  typedef TestType eT;

  Mat<eT> X(5, 1, fill::zeros);

  X[1] =  Datum<eT>::nan;
  X[2] =  Datum<eT>::inf;
  X[3] = -Datum<eT>::inf;

  uvec r = find_nonnan(X);

  REQUIRE( all( r == uvec({ 0, 2, 3, 4 }) ) );
  }
