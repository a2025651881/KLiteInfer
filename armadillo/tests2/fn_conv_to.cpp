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

using namespace arma;


TEST_CASE("fn_conv_to_1", "[conv_to]")
  {
  typedef std::vector<double> stdvec;

  stdvec x(3);
  x[0] = 10.0; x[1] = 20.0;  x[2] = 30.0;

  colvec y = conv_to< colvec >::from(x);
  stdvec z = conv_to< stdvec >::from(y);
  
  REQUIRE( z[0] == Approx(10.0) );
  REQUIRE( z[1] == Approx(20.0) );
  REQUIRE( z[2] == Approx(30.0) );
  }



TEST_CASE("fn_conv_to2", "[conv_to]")
  {
  mat A(5,6); A.fill(0.1);
  
  umat uA = conv_to<umat>::from(A);
  imat iA = conv_to<imat>::from(A);
  
  REQUIRE( (uA.n_rows - A.n_rows) == 0 );
  REQUIRE( (iA.n_rows - A.n_rows) == 0 );
  
  REQUIRE( (uA.n_cols - A.n_cols) == 0 );
  REQUIRE( (iA.n_cols - A.n_cols) == 0 );
  
  REQUIRE( any(vectorise(uA)) == false);
  REQUIRE( any(vectorise(iA)) == false);
  }
  

TEST_CASE("fn_conv_to3", "[conv_to]")
  {
  mat A(5,6); A.fill(1.0);
  
  umat uA = conv_to<umat>::from(A);
  imat iA = conv_to<imat>::from(A);
  
  REQUIRE( all(vectorise(uA)) == true);
  REQUIRE( all(vectorise(iA)) == true);
  }


TEST_CASE("fn_conv_to4", "[conv_to]")
  {
  mat A =   linspace<rowvec>(1,5,6);
  mat B = 2*linspace<colvec>(1,5,6);
  mat C = randu<mat>(5,6);
  
  REQUIRE( as_scalar( conv_to<rowvec>::from(A) * conv_to<colvec>::from(B) ) == Approx(130.40) );
  
  REQUIRE( conv_to<double>::from(A * B) == Approx(130.40) );
  
  REQUIRE_THROWS( conv_to<colvec>::from(C) );
  }


TEST_CASE("fn_conv_to_spmat_mat_different_eT", "[conv_to]")
  {
  sp_fmat A;
  A.sprandu(10, 10, 0.3);

  mat B = conv_to<mat>::from(A);

  REQUIRE( B.n_rows == 10 );
  REQUIRE( B.n_cols == 10 );
  for (size_t c = 0; c < 10; ++c)
    {
    for (size_t r = 0; r < 10; ++r)
      {
      REQUIRE( (double) B(r, c) == Approx((double) A(r, c)).margin(1e-5) );
      }
    }

  // And the other way...
  B.randu(8, 8);
  B(3, 3) = 0.0;
  A = conv_to<sp_fmat>::from(B);

  REQUIRE( A.n_rows == 8 );
  REQUIRE( A.n_cols == 8 );
  for (size_t c = 0; c < 8; ++c)
    {
    for (size_t r = 0; r < 8; ++r)
      {
      REQUIRE( (double) A(r, c) == Approx((double) B(r, c)).margin(1e-5) );
      }
    }
  }


TEST_CASE("fn_conv_to_complex_sparse_to_real", "[conv_to]")
  {
  sp_cx_mat A;
  A.sprandu(10, 10, 0.3);

  cx_mat B = conv_to<cx_mat>::from(A);

  REQUIRE( B.n_rows == 10 );
  REQUIRE( B.n_cols == 10 );
  for (size_t c = 0; c < 10; ++c)
    {
    for (size_t r = 0; r < 10; ++r)
      {
      const std::complex<double> a_val = A(r, c);
      const std::complex<double> b_val = B(r, c);
      REQUIRE( a_val.real() == Approx(b_val.real()).margin(1e-5) );
      REQUIRE( a_val.imag() == Approx(b_val.imag()).margin(1e-5) );
      }
    }
  }


TEST_CASE("fn_conv_to_complex_real_to_sparse", "[conv_to]")
  {
  cx_mat A;
  A.randu(10, 10);

  sp_cx_mat B = conv_to<sp_cx_mat>::from(A);

  REQUIRE( B.n_rows == 10 );
  REQUIRE( B.n_cols == 10 );
  for (size_t c = 0; c < 10; ++c)
    {
    for (size_t r = 0; r < 10; ++r)
      {
      const std::complex<double> a_val = A(r, c);
      const std::complex<double> b_val = B(r, c);
      REQUIRE( a_val.real() == Approx(b_val.real()).margin(1e-5) );
      REQUIRE( a_val.imag() == Approx(b_val.imag()).margin(1e-5) );
      }
    }
  }


TEST_CASE("fn_conv_to_complex_sparse_to_different_eT_real", "[conv_to]")
  {
  sp_cx_fmat A;
  A.sprandu(10, 10, 0.3);

  cx_mat B = conv_to<cx_mat>::from(A);

  REQUIRE( B.n_rows == 10 );
  REQUIRE( B.n_cols == 10 );
  for (size_t c = 0; c < 10; ++c)
    {
    for (size_t r = 0; r < 10; ++r)
      {
      const std::complex<float> a_val = A(r, c);
      const std::complex<double> b_val = B(r, c);
      REQUIRE( (double) a_val.real() == Approx(b_val.real()).margin(1e-5) );
      REQUIRE( (double) a_val.imag() == Approx(b_val.imag()).margin(1e-5) );
      }
    }
  }


TEST_CASE("fn_conv_to_complex_real_to_different_eT_sparse", "[conv_to]")
  {
  cx_mat A;
  A.randu(10, 10);

  sp_cx_fmat B = conv_to<sp_cx_fmat>::from(A);
  REQUIRE( B.n_rows == 10 );
  REQUIRE( B.n_cols == 10 );
  for (size_t c = 0; c < 10; ++c)
    {
    for (size_t r = 0; r < 10; ++r)
      {
      const std::complex<double> a_val = A(r, c);
      const std::complex<float> b_val = B(r, c);
      REQUIRE( (float) a_val.real() == Approx(b_val.real()).margin(1e-5) );
      REQUIRE( (float) a_val.imag() == Approx(b_val.imag()).margin(1e-5) );
      }
    }
  }



#if defined(ARMA_HAVE_FP16)
TEST_CASE("fn_conv_to_fp16", "[conv_to]")
  {
  hmat x1 = randu<hmat>(50, 50);
  mat x2 = conv_to<mat>::from(x1);

  REQUIRE( x1.n_rows == x2.n_rows );
  REQUIRE( x1.n_cols == x2.n_cols );
  for (uword i = 0; i < x1.n_elem; ++i)
    {
    REQUIRE( double(x1[i]) == Approx(x2[i]) );
    }

  x2.randu();
  x1 = conv_to<hmat>::from(x2);

  for (uword i = 0; i < x1.n_elem; ++i)
    {
    REQUIRE( x1[i] == Approx(fp16(x2[i])) );
    }
  }
#endif
