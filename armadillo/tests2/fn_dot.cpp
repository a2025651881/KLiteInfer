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


TEST_CASE("fn_dot_1", "[dot]")
  {
  mat A =
    "\
     0.061198   0.201990   0.019678  -0.493936  -0.126745;\
     0.437242   0.058956  -0.149362  -0.045465   0.296153;\
    -0.492474  -0.031309   0.314156   0.419733   0.068317;\
     0.336352   0.411541   0.458476  -0.393139  -0.135040;\
     0.239585  -0.428913  -0.406953  -0.291020  -0.353768;\
    ";

  vec a = A.head_cols(1);
  vec b = A.tail_cols(1);

  rowvec c = A.head_rows(1);
  rowvec d = A.tail_rows(1);

  REQUIRE( dot(  a,  b) == Approx(-0.04208883710200) );
  REQUIRE( dot(2*a,2+b) == Approx( 2.24343432579600) );

  REQUIRE( dot(    c,  d) == Approx( 0.108601544706000) );
  REQUIRE( dot(0.5*c,2-d) == Approx(-0.392115772353000) );

  REQUIRE( dot(a,b) == Approx( dot(A.head_cols(1), A.tail_cols(1)) ) );
  REQUIRE( dot(c,d) == Approx( dot(A.head_rows(1), A.tail_rows(1)) ) );
  }



TEST_CASE("fn_dot_2", "[dot]")
  {
  mat A =
    "\
     0.061198   0.201990   0.019678  -0.493936  -0.126745;\
     0.437242   0.058956  -0.149362  -0.045465   0.296153;\
    -0.492474  -0.031309   0.314156   0.419733   0.068317;\
     0.336352   0.411541   0.458476  -0.393139  -0.135040;\
     0.239585  -0.428913  -0.406953  -0.291020  -0.353768;\
    ";

  cx_vec a = cx_vec(A.col(0), A.col(1));
  cx_vec b = cx_vec(A.col(2), A.col(3));

  cx_rowvec c = cx_rowvec(A.row(0), A.row(1));
  cx_rowvec d = cx_rowvec(A.row(2), A.row(3));

  REQUIRE( abs( dot(a,b) - cx_double(-0.009544718641000, -0.110209641379000)) == Approx(0.0).margin(0.001) );
  REQUIRE( abs( dot(c,d) - cx_double(-0.326993347830000, +0.061084261990000)) == Approx(0.0).margin(0.001) );

  REQUIRE( abs(cdot(a,b) - cx_double(-0.314669805873000, -0.807333974477000)) == Approx(0.0).margin(0.001) );
  REQUIRE( abs(cdot(c,d) - cx_double(-0.165527940664000, +0.586984291846000)) == Approx(0.0).margin(0.001) );
  }



TEST_CASE("fn_dot_sp_mat_mat", "[dot]")
  {
  // Make matrices.
  SpMat<double> a("3.0 0.0 0.0; 1.0 2.0 2.0; 0.0 0.0 1.0");
  Mat<double> b("1.0 2.0 1.0; 1.0 2.0 2.0; 3.0 4.0 5.0");

  REQUIRE( dot(a, b) == Approx(17.0) );
  REQUIRE( dot(b, a) == Approx(17.0) );
  }



TEST_CASE("fn_dot_sp_col_col", "[dot]")
  {
  SpCol<unsigned int> a("3; 4; 0; 0; 0; 2; 0; 0");
  Col<unsigned int> b("1 6 1 2 3 7 1 2");

  REQUIRE( dot(a, b) == 41 );
  REQUIRE( dot(b, a) == 41 );
  }



TEST_CASE("fn_dot_sp_mat_sp_mat", "[dot]")
  {
  SpMat<double> a("3.0 0.0 0.0; 1.0 2.0 2.0; 0.0 0.0 1.0");
  SpMat<double> b("3.0 0.0 0.0; 1.0 2.0 2.0; 0.0 0.0 1.0");

  REQUIRE( dot(a, b) == Approx(19.0) );
  REQUIRE( dot(b, a) == Approx(19.0) );
  }



TEST_CASE("fn_dot_sp_col_sp_col", "[dot]")
  {
  SpCol<unsigned int> a("3; 4; 0; 0; 0; 2; 0; 0");
  SpCol<unsigned int> b("0; 8; 0; 1; 1; 0; 0; 0");

  REQUIRE( dot(a, b) == 32 );
  REQUIRE( dot(b, a) == 32 );
  }



TEMPLATE_TEST_CASE("fn_dot_fp_randu", "[dot]", TEST_FLOAT_TYPES)
  {
  typedef TestType eT;

  Col<eT> x1 = randu<Col<eT>>(100);
  Col<eT> x2 = randu<Col<eT>>(100);

  vec x1_ref = conv_to<vec>::from(x1);
  vec x2_ref = conv_to<vec>::from(x2);

  eT d = dot(x1, x2);
  double d_ref = dot(x1_ref, x2_ref);

  constexpr eT eps = is_blas_real<eT>::value ? eT(0.001) : eT(0.1);

  REQUIRE( double(d) == Approx(d_ref).epsilon(eps) );
  }



TEMPLATE_TEST_CASE("fn_sp_dot_fp_randu", "[dot]", TEST_FLOAT_TYPES)
  {
  typedef TestType eT;

  SpCol<eT> x1, x2;
  x1.sprandu(1000, 1, 0.3);
  x2.sprandu(1000, 1, 0.3);

  sp_mat x1_ref = conv_to<sp_mat>::from(x1);
  sp_mat x2_ref = conv_to<sp_mat>::from(x2);

  eT d = dot(x1, x2);
  double d_ref = dot(x1_ref, x2_ref);

  constexpr eT eps = is_blas_real<eT>::value ? eT(0.001) : eT(0.1);

  REQUIRE( double(d) == Approx(d_ref).epsilon(eps) );
  }



TEMPLATE_TEST_CASE("fn_cdot_fp_randu", "[dot]", TEST_CX_FLOAT_TYPES)
  {
  typedef TestType eT;

  Col<eT> x1 = randu<Col<eT>>(100);
  Col<eT> x2 = randu<Col<eT>>(100);

  cx_vec x1_ref = conv_to<cx_vec>::from(x1);
  cx_vec x2_ref = conv_to<cx_vec>::from(x2);

  eT d = cdot(x1, x2);
  std::complex<double> d_ref = cdot(x1_ref, x2_ref);

  typedef typename get_pod_type<eT>::result epsT;
  constexpr epsT eps = is_blas_real<eT>::value ? epsT(0.001) : epsT(0.1);

  REQUIRE( double(std::real(d)) == Approx(std::real(d_ref)).epsilon(eps) );
  REQUIRE( double(std::imag(d)) == Approx(std::imag(d_ref)).epsilon(eps) );
  }



TEMPLATE_TEST_CASE("fn_norm_dot_fp_randu", "[dot]", TEST_FLOAT_TYPES)
  {
  typedef TestType eT;

  Col<eT> x1 = randu<Col<eT>>(10);
  Col<eT> x2 = randu<Col<eT>>(10);

  const eT d_unnorm = dot(x1, x2);
  const eT norm1 = norm(x1);
  const eT norm2 = norm(x2);

  const eT d_norm = norm_dot(x1, x2);

  constexpr eT eps = is_blas_real<eT>::value ? eT(0.001) : eT(0.1);

  REQUIRE( d_norm == Approx(d_unnorm / (norm1 * norm2)).epsilon(eps) );
  }
