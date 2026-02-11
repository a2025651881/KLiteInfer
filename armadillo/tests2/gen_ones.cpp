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


TEMPLATE_TEST_CASE("gen_ones_1", "[ones]", TEST_FLOAT_TYPES)
  {
  typedef TestType eT;

  Mat<eT> A(5,6,fill::ones);
  
  REQUIRE( double(accu(A))  == Approx(double(5*6)) );
  REQUIRE( A.n_rows == 5 );
  REQUIRE( A.n_cols == 6 );
  
  Mat<eT> B(5,6,fill::randu);
  
  B.ones();
  
  REQUIRE( double(accu(B))  == Approx(double(5*6)) );
  REQUIRE( B.n_rows == 5 );
  REQUIRE( B.n_cols == 6 );
  
  Mat<eT> C = ones<Mat<eT>>(5,6);
  
  REQUIRE( double(accu(C))  == Approx(double(5*6)) );
  REQUIRE( C.n_rows == 5 );
  REQUIRE( C.n_cols == 6 );
  
  Mat<eT> D; D = ones<Mat<eT>>(5,6);
  
  REQUIRE( double(accu(D))  == Approx(double(5*6)) );
  REQUIRE( D.n_rows == 5 );
  REQUIRE( D.n_cols == 6 );
  
  Mat<eT> E; E = 2*ones<Mat<eT>>(5,6);
  
  REQUIRE( double(accu(E))  == Approx(double(2*5*6)) );
  REQUIRE( E.n_rows == 5 );
  REQUIRE( E.n_cols == 6 );
  }



TEMPLATE_TEST_CASE("gen_ones_2", "[ones]", TEST_FLOAT_TYPES)
  {
  typedef TestType eT;

  Mat<eT> A(5,6,fill::zeros);
  
  A.col(1).ones();
  
  REQUIRE( double(accu(A.col(0))) == Approx(0.0).margin(0.001)              );
  REQUIRE( double(accu(A.col(1))) == Approx(double(A.n_rows)) );
  REQUIRE( double(accu(A.col(2))) == Approx(0.0).margin(0.001)              );
  
  Mat<eT> B(5,6,fill::zeros);
  
  B.row(1).ones();
  
  REQUIRE( double(accu(B.row(0))) == Approx(0.0).margin(0.001)              );
  REQUIRE( double(accu(B.row(1))) == Approx(double(B.n_cols)) );
  REQUIRE( double(accu(B.row(2))) == Approx(0.0).margin(0.001)              );
  
  Mat<eT> C(5,6,fill::zeros);
  
  C(span(1,3),span(1,4)).ones();
  
  REQUIRE( double(accu(C.head_cols(1))) == Approx(0.0).margin(0.001) );
  REQUIRE( double(accu(C.head_rows(1))) == Approx(0.0).margin(0.001) );
  
  REQUIRE( double(accu(C.tail_cols(1))) == Approx(0.0).margin(0.001) );
  REQUIRE( double(accu(C.tail_rows(1))) == Approx(0.0).margin(0.001) );
  
  REQUIRE( double(accu(C(span(1,3),span(1,4)))) == Approx(double(3*4)) );

  Mat<eT> D(5,6,fill::zeros);
  
  D.diag().ones();
  
  REQUIRE( double(accu(D.diag())) == Approx(double(5)) );
  }



TEMPLATE_TEST_CASE("gen_ones_3", "[ones]", TEST_FLOAT_TYPES)
  {
  typedef TestType eT;

  Mat<eT> A(5,6,fill::zeros);
  
  uvec indices = { 2, 4, 6 };
  
  A(indices).ones();
  
  REQUIRE( double(accu(A)) == Approx(double(3)) );
  
  REQUIRE( A(0)          == Approx(eT(0)).margin(0.001) );
  REQUIRE( A(A.n_elem-1) == Approx(eT(0)).margin(0.001) );
  
  REQUIRE( A(indices(0)) == Approx(eT(1)) );
  REQUIRE( A(indices(1)) == Approx(eT(1)) );
  REQUIRE( A(indices(2)) == Approx(eT(1)) );
  }
