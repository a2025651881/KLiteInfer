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


TEMPLATE_TEST_CASE("gen_zeros_1", "[zeros]", TEST_FLOAT_TYPES)
  {
  typedef TestType eT;

  constexpr const eT margin = is_blas_real<eT>::value ? eT(0.001) : eT(0.01);

  Mat<eT> A(5,6,fill::zeros);

  REQUIRE( accu(A)  == Approx(eT(0)).margin(margin) );
  REQUIRE( A.n_rows == 5 );
  REQUIRE( A.n_cols == 6 );

  Mat<eT> B(5,6,fill::randu);

  B.zeros();

  REQUIRE( accu(B)  == Approx(eT(0)).margin(margin) );
  REQUIRE( B.n_rows == 5 );
  REQUIRE( B.n_cols == 6 );

  Mat<eT> C = zeros<Mat<eT>>(5,6);

  REQUIRE( accu(C)  == Approx(eT(0)).margin(margin) );
  REQUIRE( C.n_rows == 5 );
  REQUIRE( C.n_cols == 6 );

  Mat<eT> D; D = zeros<Mat<eT>>(5,6);

  REQUIRE( accu(D)  == Approx(eT(0)).margin(margin) );
  REQUIRE( D.n_rows == 5 );
  REQUIRE( D.n_cols == 6 );

  Mat<eT> E; E = 2*zeros<Mat<eT>>(5,6);

  REQUIRE( accu(E)  == Approx(eT(0)).margin(margin) );
  REQUIRE( E.n_rows == 5 );
  REQUIRE( E.n_cols == 6 );
  }



TEMPLATE_TEST_CASE("gen_zeros_2", "[zeros]", TEST_FLOAT_TYPES)
  {
  typedef TestType eT;

  Mat<eT> A(5,6,fill::ones);

  A.col(1).zeros();

  constexpr const eT margin = is_blas_real<eT>::value ? eT(0.001) : eT(0.01);

  REQUIRE( accu(A.col(0)) == Approx(eT(A.n_rows)) );
  REQUIRE( accu(A.col(1)) == Approx(eT(0)).margin(margin)              );
  REQUIRE( accu(A.col(2)) == Approx(eT(A.n_rows)) );

  Mat<eT> B(5,6,fill::ones);

  B.row(1).zeros();

  REQUIRE( accu(B.row(0)) == Approx(eT(B.n_cols)) );
  REQUIRE( accu(B.row(1)) == Approx(eT(0)).margin(margin)              );
  REQUIRE( accu(B.row(2)) == Approx(eT(B.n_cols)) );

  Mat<eT> C(5,6,fill::ones);

  C(span(1,3),span(1,4)).zeros();

  REQUIRE( accu(C.head_cols(1)) == Approx(eT(5)) );
  REQUIRE( accu(C.head_rows(1)) == Approx(eT(6)) );

  REQUIRE( accu(C.tail_cols(1)) == Approx(eT(5)) );
  REQUIRE( accu(C.tail_rows(1)) == Approx(eT(6)) );

  REQUIRE( accu(C(span(1,3),span(1,4))) == Approx(eT(0)).margin(margin) );

  Mat<eT> D(5,6,fill::ones);

  D.diag().zeros();

  REQUIRE( accu(D.diag()) == Approx(eT(0)).margin(margin) );
  }



TEMPLATE_TEST_CASE("gen_zeros_3", "[zeros]", TEST_FLOAT_TYPES)
  {
  typedef TestType eT;

  Mat<eT> A(5,6,fill::ones);

  constexpr const eT margin = is_blas_real<eT>::value ? eT(0.001) : eT(0.01);

  uvec indices = { 2, 4, 6 };

  A(indices).zeros();

  REQUIRE( accu(A) == Approx(eT(5*6-3)) );

  REQUIRE( A(0)          == Approx(eT(1)) );
  REQUIRE( A(A.n_elem-1) == Approx(eT(1)) );

  REQUIRE( A(indices(0)) == Approx(eT(0)).margin(margin) );
  REQUIRE( A(indices(1)) == Approx(eT(0)).margin(margin) );
  REQUIRE( A(indices(2)) == Approx(eT(0)).margin(margin) );
  }



TEST_CASE("gen_zeros_sp_mat", "[zeros]")
  {
  SpMat<unsigned int> e(2, 2);

  e(0, 0) = 3;
  e(1, 1) = 2;

  e *= zeros<SpMat<unsigned int> >(2, 2);

  REQUIRE( e.n_nonzero == 0 );
  REQUIRE( (unsigned int) e(0, 0) == 0 );
  REQUIRE( (unsigned int) e(1, 0) == 0 );
  REQUIRE( (unsigned int) e(0, 1) == 0 );
  REQUIRE( (unsigned int) e(1, 1) == 0 );

  // Just test compilation here.
  e = zeros<SpMat<unsigned int> >(5, 5);
  e *= zeros<SpMat<unsigned int> >(5, 5);
  e %= zeros<SpMat<unsigned int> >(5, 5);
  }
