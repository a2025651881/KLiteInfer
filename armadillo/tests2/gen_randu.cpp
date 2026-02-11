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


TEMPLATE_TEST_CASE("gen_randu_1", "[randu]", TEST_FLOAT_TYPES)
  {
  typedef TestType eT;

  const uword n_rows = 100;
  const uword n_cols = 101;

  Mat<eT> A(n_rows,n_cols, fill::randu);

  Mat<eT> B(n_rows,n_cols); B.randu();

  Mat<eT> C; C.randu(n_rows,n_cols);

  constexpr const double margin1 = is_blas_real<eT>::value ? 0.02  : 0.2;
  constexpr const double margin2 = is_blas_real<eT>::value ? 0.025 : 0.3;

  // low-precision types could underflow, so convert to doubles before computing the mean

  REQUIRE( (accu(conv_to<mat>::from(A))/A.n_elem) == Approx(0.5).margin(margin1) );
  REQUIRE( (accu(conv_to<mat>::from(B))/A.n_elem) == Approx(0.5).margin(margin1) );
  REQUIRE( (accu(conv_to<mat>::from(C))/A.n_elem) == Approx(0.5).margin(margin1) );

  REQUIRE( (mean(vectorise(conv_to<mat>::from(A)))) == Approx(eT(0.5)).margin(margin2) );
  }



TEMPLATE_TEST_CASE("gen_randu_2", "[randu]", TEST_FLOAT_TYPES)
  {
  typedef TestType eT;

  Mat<eT> A(50,60,fill::zeros);

  A(span(1,48),span(1,58)).randu();

  constexpr const     eT margin1 = is_blas_real<eT>::value ? eT(0.001) : eT(0.01 );
  constexpr const double margin2 = is_blas_real<eT>::value ? eT(0.025) : eT(0.025);

  REQUIRE( accu(A.head_cols(1)) == Approx(eT(0)).margin(margin1) );
  REQUIRE( accu(A.head_rows(1)) == Approx(eT(0)).margin(margin1) );

  REQUIRE( accu(A.tail_cols(1)) == Approx(eT(0)).margin(margin1) );
  REQUIRE( accu(A.tail_rows(1)) == Approx(eT(0)).margin(margin1) );

  // low-precision types could overflow

  REQUIRE( mean(vectorise(conv_to<mat>::from(A)(span(1,48),span(1,58)))) == Approx(0.5).margin(margin2) );
  }
