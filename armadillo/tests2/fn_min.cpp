// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2011-2017 Ryan Curtin (http://www.ratml.org/)
// Copyright 2017 National ICT Australia (NICTA)
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

TEST_CASE("fn_min_subview_test", "[min]")
  {
  // We will assume subview.at() works and returns points within the bounds of
  // the matrix, so we just have to ensure the results are the same as
  // Mat.min()...
  for (size_t r = 50; r < 150; ++r)
    {
    mat x(r, r, fill::randn);

    uword x_min          = x.index_min();
    uword x_subview_min1 = x.submat(0, 0, r - 1, r - 1).index_min();
    uword x_subview_min2 = x.cols(0, r - 1).index_min();
    uword x_subview_min3 = x.rows(0, r - 1).index_min();

    const double mval  = x.min();
    const double mval1 = x.submat(0, 0, r - 1, r - 1).min();
    const double mval2 = x.cols(0, r - 1).min();
    const double mval3 = x.rows(0, r - 1).min();

    REQUIRE( x_min == x_subview_min1 );
    REQUIRE( x_min == x_subview_min2 );
    REQUIRE( x_min == x_subview_min3 );

    REQUIRE( mval == Approx(mval1) );
    REQUIRE( mval == Approx(mval2) );
    REQUIRE( mval == Approx(mval3) );

    REQUIRE( mval == Approx(x(x_min)) );
    }
  }



TEST_CASE("fn_min_subview_col_test", "[min]")
  {
  for (size_t r = 10; r < 50; ++r)
    {
    vec x(r, fill::randn);

    uword x_min          = x.index_min();
    uword x_subview_min1 = x.submat(0, 0, r - 1, 0).index_min();
    uword x_subview_min2 = x.rows(0, r - 1).index_min();

    const double mval  = x.min();
    const double mval1 = x.submat(0, 0, r - 1, 0).min();
    const double mval2 = x.rows(0, r - 1).min();

    REQUIRE( x_min == x_subview_min1 );
    REQUIRE( x_min == x_subview_min2 );

    REQUIRE( mval == Approx(mval1) );
    REQUIRE( mval == Approx(mval2) );

    REQUIRE( mval == Approx(x(x_min)) );
    }
  }



TEST_CASE("fn_min_subview_row_test", "[min]")
  {
  for (size_t r = 10; r < 50; ++r)
    {
    rowvec x(r, fill::randn);

    uword x_min          = x.index_min();
    uword x_subview_min1 = x.submat(0, 0, 0, r - 1).index_min();
    uword x_subview_min2 = x.cols(0, r - 1).index_min();

    const double mval  = x.min();
    const double mval1 = x.submat(0, 0, 0, r - 1).min();
    const double mval2 = x.cols(0, r - 1).min();

    REQUIRE( x_min == x_subview_min1 );
    REQUIRE( x_min == x_subview_min2 );

    REQUIRE( mval == Approx(mval1) );
    REQUIRE( mval == Approx(mval2) );

    REQUIRE( mval == Approx(x(x_min)) );
    }
  }



// TEST_CASE("fn_min_incomplete_subview_test", "[min]")
//   {
//   for (size_t r = 50; r < 150; ++r)
//     {
//     mat x(r, r, fill::randn);
// 
//     uword x_min;
//     uword x_subview_min1;
//     uword x_subview_min2;
//     uword x_subview_min3;
// 
//     const double mval = x.min(x_min);
//     const double mval1 = x.submat(1, 1, r - 2, r - 2).min(x_subview_min1);
//     const double mval2 = x.cols(1, r - 2).min(x_subview_min2);
//     const double mval3 = x.rows(1, r - 2).min(x_subview_min3);
// 
//     uword row, col;
//     x.min(row, col);
// 
//     if (row != 0 && row != r - 1 && col != 0 && col != r - 1)
//       {
//       uword srow, scol;
// 
//       srow = x_subview_min1 % (r - 2);
//       scol = x_subview_min1 / (r - 2);
//       REQUIRE( x_min == (srow + 1) + r * (scol + 1) );
//       REQUIRE( x_min == x_subview_min2 + r );
// 
//       srow = x_subview_min3 % (r - 2);
//       scol = x_subview_min3 / (r - 2);
//       REQUIRE( x_min == (srow + 1) + r * scol );
// 
//       REQUIRE( mval == Approx(mval1) );
//       REQUIRE( mval == Approx(mval2) );
//       REQUIRE( mval == Approx(mval3) );
//       }
//     }
//   }



TEST_CASE("fn_min_incomplete_subview_col_test", "[min]")
  {
  for (size_t r = 10; r < 50; ++r)
    {
    vec x(r, fill::randn);

    uword x_min          = x.index_min();
    uword x_subview_min1 = x.submat(1, 0, r - 2, 0).index_min();
    uword x_subview_min2 = x.rows(1, r - 2).index_min();

    const double mval  = x.min();
    const double mval1 = x.submat(1, 0, r - 2, 0).min();
    const double mval2 = x.rows(1, r - 2).min();

    if (x_min != 0 && x_min != r - 1)
      {
      REQUIRE( x_min == x_subview_min1 + 1 );
      REQUIRE( x_min == x_subview_min2 + 1 );

      REQUIRE( mval == Approx(mval1) );
      REQUIRE( mval == Approx(mval2) );

      REQUIRE( mval == Approx(x(x_min)) );
      }
    }
  }



TEST_CASE("fn_min_cx_subview_row_test", "[min]")
  {
  for (size_t r = 10; r < 50; ++r)
    {
    cx_rowvec x(r, fill::randn);

    uword x_min          = x.index_min();
    uword x_subview_min1 = x.submat(0, 0, 0, r - 1).index_min();
    uword x_subview_min2 = x.cols(0, r - 1).index_min();

    const std::complex<double> mval  = x.min();
    const std::complex<double> mval1 = x.submat(0, 0, 0, r - 1).min();
    const std::complex<double> mval2 = x.cols(0, r - 1).min();

    REQUIRE( x_min == x_subview_min1 );
    REQUIRE( x_min == x_subview_min2 );

    REQUIRE( mval.real() == Approx(mval1.real()) );
    REQUIRE( mval.imag() == Approx(mval1.imag()) );
    REQUIRE( mval.real() == Approx(mval2.real()) );
    REQUIRE( mval.imag() == Approx(mval2.imag()) );

    REQUIRE( mval.real() == Approx(x(x_min).real()) );
    REQUIRE( mval.imag() == Approx(x(x_min).imag()) );
    }
  }



// TEST_CASE("fn_min_cx_incomplete_subview_test", "[min]")
//   {
//   for (size_t r = 50; r < 150; ++r)
//     {
//     cx_mat x(r, r, fill::randn);
// 
//     uword x_min;
//     uword x_subview_min1;
//     uword x_subview_min2;
//     uword x_subview_min3;
// 
//     const std::complex<double> mval = x.min(x_min);
//     const std::complex<double> mval1 = x.submat(1, 1, r - 2, r - 2).min(x_subview_min1);
//     const std::complex<double> mval2 = x.cols(1, r - 2).min(x_subview_min2);
//     const std::complex<double> mval3 = x.rows(1, r - 2).min(x_subview_min3);
// 
//     uword row, col;
//     x.min(row, col);
// 
//     if (row != 0 && row != r - 1 && col != 0 && col != r - 1)
//       {
//       uword srow, scol;
// 
//       srow = x_subview_min1 % (r - 2);
//       scol = x_subview_min1 / (r - 2);
//       REQUIRE( x_min == (srow + 1) + r * (scol + 1) );
//       REQUIRE( x_min == x_subview_min2 + r );
// 
//       srow = x_subview_min3 % (r - 2);
//       scol = x_subview_min3 / (r - 2);
//       REQUIRE( x_min == (srow + 1) + r * scol );
// 
//       REQUIRE( mval.real() == Approx(mval1.real()) );
//       REQUIRE( mval.imag() == Approx(mval1.imag()) );
//       REQUIRE( mval.real() == Approx(mval2.real()) );
//       REQUIRE( mval.imag() == Approx(mval2.imag()) );
//       REQUIRE( mval.real() == Approx(mval3.real()) );
//       REQUIRE( mval.imag() == Approx(mval3.imag()) );
//       }
//     }
//   }



TEST_CASE("fn_min_cx_incomplete_subview_col_test", "[min]")
  {
  for (size_t r = 10; r < 50; ++r)
    {
    cx_vec x(r, fill::randn);

    uword x_min          = x.index_min();
    uword x_subview_min1 = x.submat(1, 0, r - 2, 0).index_min();
    uword x_subview_min2 = x.rows(1, r - 2).index_min();

    const std::complex<double> mval  = x.min();
    const std::complex<double> mval1 = x.submat(1, 0, r - 2, 0).min();
    const std::complex<double> mval2 = x.rows(1, r - 2).min();

    if (x_min != 0 && x_min != r - 1)
      {
      REQUIRE( x_min == x_subview_min1 + 1 );
      REQUIRE( x_min == x_subview_min2 + 1 );

      REQUIRE( mval.real() == Approx(mval1.real()) );
      REQUIRE( mval.imag() == Approx(mval1.imag()) );
      REQUIRE( mval.real() == Approx(mval2.real()) );
      REQUIRE( mval.imag() == Approx(mval2.imag()) );

      REQUIRE( mval.real() == Approx(x(x_min).real()) );
      REQUIRE( mval.imag() == Approx(x(x_min).imag()) );
      }
    }
  }



TEST_CASE("fn_min_cx_incomplete_subview_row_test", "[min]")
  {
  for (size_t r = 10; r < 50; ++r)
    {
    cx_rowvec x(r, fill::randn);

    uword x_min          = x.index_min();
    uword x_subview_min1 = x.submat(0, 1, 0, r - 2).index_min();
    uword x_subview_min2 = x.cols(1, r - 2).index_min();

    const std::complex<double> mval  = x.min();
    const std::complex<double> mval1 = x.submat(0, 1, 0, r - 2).min();
    const std::complex<double> mval2 = x.cols(1, r - 2).min();

    if (x_min != 0 && x_min != r - 1)
      {
      REQUIRE( x_min == x_subview_min1 + 1 );
      REQUIRE( x_min == x_subview_min2 + 1 );

      REQUIRE( mval.real() == Approx(mval1.real()) );
      REQUIRE( mval.imag() == Approx(mval1.imag()) );
      REQUIRE( mval.real() == Approx(mval2.real()) );
      REQUIRE( mval.imag() == Approx(mval2.imag()) );

      REQUIRE( mval.real() == Approx(x(x_min).real()) );
      REQUIRE( mval.imag() == Approx(x(x_min).imag()) );
      }
    }
  }



TEST_CASE("fn_min_weird_operation", "[min]")
  {
  mat a(10, 10, fill::randn);
  mat b(25, 10, fill::randn);

  mat output = a * b.t();

  uword real_min      = output.index_min();
  uword operation_min = (a * b.t()).index_min();

  const double mval       = output.min();
  const double other_mval = (a * b.t()).min();

  REQUIRE( real_min == operation_min            );
  REQUIRE( mval     == Approx(other_mval)       );
  REQUIRE( mval     == Approx(output(real_min)) );
  }



TEST_CASE("fn_min_weird_sparse_operation", "[min]")
  {
  sp_mat a; a.sprandn(10, 10, 0.3);
  sp_mat b; b.sprandn(25, 10, 0.3);

  sp_mat output = a * b.t();

  uword real_min      =      output.index_min();
  uword operation_min = (a * b.t()).index_min();

  const double mval       =      output.min();
  const double other_mval = (a * b.t()).min();

  REQUIRE( real_min == operation_min            );
  REQUIRE( mval     == Approx(other_mval)       );
  REQUIRE( mval     == Approx(output(real_min)) );
  }



TEST_CASE("fn_min_sp_subview_test", "[min]")
  {
  // We will assume subview.at() works and returns points within the bounds of
  // the matrix, so we just have to ensure the results are the same as
  // Mat.min()...
  for (size_t r = 50; r < 150; ++r)
    {
    sp_mat x;  x.sprandn(r, r, 0.3);

    uword x_min          = x.index_min();
    uword x_subview_min1 = x.submat(0, 0, r - 1, r - 1).index_min();
    uword x_subview_min2 = x.cols(0, r - 1).index_min();
    uword x_subview_min3 = x.rows(0, r - 1).index_min();

    const double mval  = x.min();
    const double mval1 = x.submat(0, 0, r - 1, r - 1).min();
    const double mval2 = x.cols(0, r - 1).min();
    const double mval3 = x.rows(0, r - 1).min();

    if (mval != 0.0)
      {
      REQUIRE( x_min == x_subview_min1 );
      REQUIRE( x_min == x_subview_min2 );
      REQUIRE( x_min == x_subview_min3 );

      REQUIRE( mval == Approx(mval1) );
      REQUIRE( mval == Approx(mval2) );
      REQUIRE( mval == Approx(mval3) );
      
      REQUIRE( mval == Approx(x(x_min)) );
      }
    }
  }



TEST_CASE("fn_min_spsubview_col_test", "[min]")
  {
  for (size_t r = 10; r < 50; ++r)
    {
    sp_vec x;  x.sprandn(r, 1, 0.3);

    uword x_min          = x.index_min();
    uword x_subview_min1 = x.submat(0, 0, r - 1, 0).index_min();
    uword x_subview_min2 = x.rows(0, r - 1).index_min();

    const double mval  = x.min();
    const double mval1 = x.submat(0, 0, r - 1, 0).min();
    const double mval2 = x.rows(0, r - 1).min();

    if (mval != 0.0)
      {
      REQUIRE( x_min == x_subview_min1 );
      REQUIRE( x_min == x_subview_min2 );

      REQUIRE( mval == Approx(mval1) );
      REQUIRE( mval == Approx(mval2) );
      
      REQUIRE( mval == Approx(x(x_min)) );
      }
    }
  }



TEST_CASE("fn_min_spsubview_row_min_test", "[min]")
  {
  for (size_t r = 10; r < 50; ++r)
    {
    sp_rowvec x;  x.sprandn(1, r, 0.3);

    uword x_min          = x.index_min();
    uword x_subview_min1 = x.submat(0, 0, 0, r - 1).index_min();
    uword x_subview_min2 = x.cols(0, r - 1).index_min();

    const double mval  = x.min();
    const double mval1 = x.submat(0, 0, 0, r - 1).min();
    const double mval2 = x.cols(0, r - 1).min();

    if (mval != 0.0)
      {
      REQUIRE( x_min == x_subview_min1 );
      REQUIRE( x_min == x_subview_min2 );

      REQUIRE( mval == Approx(mval1) );
      REQUIRE( mval == Approx(mval2) );
      
      REQUIRE( mval == Approx(x(x_min)) );
      }
    }
  }



// TEST_CASE("fn_min_spincompletesubview_min_test", "[min]")
//   {
//   for (size_t r = 50; r < 150; ++r)
//     {
//     sp_mat x;
//     x.sprandn(r, r, 0.3);
// 
//     uword x_min;
//     uword x_subview_min1;
//     uword x_subview_min2;
//     uword x_subview_min3;
// 
//     const double mval = x.min(x_min);
//     const double mval1 = x.submat(1, 1, r - 2, r - 2).min(x_subview_min1);
//     const double mval2 = x.cols(1, r - 2).min(x_subview_min2);
//     const double mval3 = x.rows(1, r - 2).min(x_subview_min3);
// 
//     uword row, col;
//     x.min(row, col);
// 
//     if (row != 0 && row != r - 1 && col != 0 && col != r - 1 && mval != 0.0)
//       {
//       uword srow, scol;
// 
//       srow = x_subview_min1 % (r - 2);
//       scol = x_subview_min1 / (r - 2);
//       REQUIRE( x_min == (srow + 1) + r * (scol + 1) );
//       REQUIRE( x_min == x_subview_min2 + r );
// 
//       srow = x_subview_min3 % (r - 2);
//       scol = x_subview_min3 / (r - 2);
//       REQUIRE( x_min == (srow + 1) + r * scol );
// 
//       REQUIRE( mval == Approx(mval1) );
//       REQUIRE( mval == Approx(mval2) );
//       REQUIRE( mval == Approx(mval3) );
//       }
//     }
//   }



TEST_CASE("fn_min_spincompletesubview_col_min_test", "[min]")
  {
  for (size_t r = 10; r < 50; ++r)
    {
    sp_vec x;  x.sprandu(r, 1, 0.3);

    uword x_min          = x.index_min();
    uword x_subview_min1 = x.submat(1, 0, r - 2, 0).index_min();
    uword x_subview_min2 = x.rows(1, r - 2).index_min();

    const double mval  = x.min();
    const double mval1 = x.submat(1, 0, r - 2, 0).min();
    const double mval2 = x.rows(1, r - 2).min();

    if (x_min != 0 && x_min != r - 1 && mval != 0.0)
      {
      REQUIRE( x_min == x_subview_min1 + 1 );
      REQUIRE( x_min == x_subview_min2 + 1 );

      REQUIRE( mval == Approx(mval1) );
      REQUIRE( mval == Approx(mval2) );
      
      REQUIRE( mval == Approx(x(x_min)));
      }
    }
  }



TEST_CASE("fn_min_spincompletesubview_row_min_test", "[min]")
  {
  for (size_t r = 10; r < 50; ++r)
    {
    sp_rowvec x;  x.sprandn(1, r, 0.3);

    uword x_min          = x.index_min();
    uword x_subview_min1 = x.submat(0, 1, 0, r - 2).index_min();
    uword x_subview_min2 = x.cols(1, r - 2).index_min();

    const double mval  = x.min();
    const double mval1 = x.submat(0, 1, 0, r - 2).min();
    const double mval2 = x.cols(1, r - 2).min();

    if (mval != 0.0 && x_min != 0 && x_min != r - 1)
      {
      REQUIRE( x_min == x_subview_min1 + 1 );
      REQUIRE( x_min == x_subview_min2 + 1 );

      REQUIRE( mval == Approx(mval1) );
      REQUIRE( mval == Approx(mval2) );

      REQUIRE( mval == Approx( x(x_min) ) );
      }
    }
  }



TEST_CASE("fn_min_sp_cx_subview_min_test", "[min]")
  {
  // We will assume subview.at() works and returns points within the bounds of
  // the matrix, so we just have to ensure the results are the same as
  // Mat.min()...
  for (size_t r = 50; r < 150; ++r)
    {
    sp_cx_mat x;  x.sprandn(r, r, 0.3);

    uword x_min          = x.index_min();
    uword x_subview_min1 = x.submat(0, 0, r - 1, r - 1).index_min();
    uword x_subview_min2 = x.cols(0, r - 1).index_min();
    uword x_subview_min3 = x.rows(0, r - 1).index_min();

    const std::complex<double> mval  = x.min();
    const std::complex<double> mval1 = x.submat(0, 0, r - 1, r - 1).min();
    const std::complex<double> mval2 = x.cols(0, r - 1).min();
    const std::complex<double> mval3 = x.rows(0, r - 1).min();

    if (mval != std::complex<double>(0.0))
      {
      REQUIRE( x_min == x_subview_min1 );
      REQUIRE( x_min == x_subview_min2 );
      REQUIRE( x_min == x_subview_min3 );

      REQUIRE( mval.real() == Approx(mval1.real()) );
      REQUIRE( mval.imag() == Approx(mval1.imag()) );
      REQUIRE( mval.real() == Approx(mval2.real()) );
      REQUIRE( mval.imag() == Approx(mval2.imag()) );
      REQUIRE( mval.real() == Approx(mval3.real()) );
      REQUIRE( mval.imag() == Approx(mval3.imag()) );

      REQUIRE( mval.real() == Approx(x(x_min).real()) );
      REQUIRE( mval.imag() == Approx(x(x_min).imag()) );
      }
    }
  }



TEST_CASE("fn_min_sp_cx_subview_col_min_test", "[min]")
  {
  for (size_t r = 10; r < 50; ++r)
    {
    sp_cx_vec x;  x.sprandn(r, 1, 0.3);

    uword x_min          = x.index_min();
    uword x_subview_min1 = x.submat(0, 0, r - 1, 0).index_min();
    uword x_subview_min2 = x.rows(0, r - 1).index_min();

    const std::complex<double> mval  = x.min();
    const std::complex<double> mval1 = x.submat(0, 0, r - 1, 0).min();
    const std::complex<double> mval2 = x.rows(0, r - 1).min();

    if (mval != std::complex<double>(0.0))
      {
      REQUIRE( x_min == x_subview_min1 );
      REQUIRE( x_min == x_subview_min2 );

      REQUIRE( mval.real() == Approx(mval1.real()) );
      REQUIRE( mval.imag() == Approx(mval1.imag()) );
      REQUIRE( mval.real() == Approx(mval2.real()) );
      REQUIRE( mval.imag() == Approx(mval2.imag()) );

      REQUIRE( mval.real() == Approx(x(x_min).real()) );
      REQUIRE( mval.imag() == Approx(x(x_min).imag()) );
      }
    }
  }



TEST_CASE("fn_min_sp_cx_subview_row_min_test", "[min]")
  {
  for (size_t r = 10; r < 50; ++r)
    {
    sp_cx_rowvec x;  x.sprandn(1, r, 0.3);

    uword x_min          = x.index_min();
    uword x_subview_min1 = x.submat(0, 0, 0, r - 1).index_min();
    uword x_subview_min2 = x.cols(0, r - 1).index_min();

    const std::complex<double> mval  = x.min();
    const std::complex<double> mval1 = x.submat(0, 0, 0, r - 1).min();
    const std::complex<double> mval2 = x.cols(0, r - 1).min();

    if (mval != std::complex<double>(0.0))
      {
      REQUIRE( x_min == x_subview_min1 );
      REQUIRE( x_min == x_subview_min2 );

      REQUIRE( mval.real() == Approx(mval1.real()) );
      REQUIRE( mval.imag() == Approx(mval1.imag()) );
      REQUIRE( mval.real() == Approx(mval2.real()) );
      REQUIRE( mval.imag() == Approx(mval2.imag()) );

      REQUIRE( mval.real() == Approx(x(x_min).real()) );
      REQUIRE( mval.imag() == Approx(x(x_min).imag()) );
      }
    }
  }



// TEST_CASE("fn_min_sp_cx_incomplete_subview_min_test", "[min]")
//   {
//   for (size_t r = 50; r < 150; ++r)
//     {
//     sp_cx_mat x;
//     x.sprandn(r, r, 0.3);
// 
//     uword x_min;
//     uword x_subview_min1;
//     uword x_subview_min2;
//     uword x_subview_min3;
// 
//     const std::complex<double> mval = x.min(x_min);
//     const std::complex<double> mval1 = x.submat(1, 1, r - 2, r - 2).min(x_subview_min1);
//     const std::complex<double> mval2 = x.cols(1, r - 2).min(x_subview_min2);
//     const std::complex<double> mval3 = x.rows(1, r - 2).min(x_subview_min3);
// 
//     uword row, col;
//     x.min(row, col);
// 
//     if (row != 0 && row != r - 1 && col != 0 && col != r - 1 && mval != std::complex<double>(0.0))
//       {
//       uword srow, scol;
// 
//       srow = x_subview_min1 % (r - 2);
//       scol = x_subview_min1 / (r - 2);
//       REQUIRE( x_min == (srow + 1) + r * (scol + 1) );
//       REQUIRE( x_min == x_subview_min2 + r );
// 
//       srow = x_subview_min3 % (r - 2);
//       scol = x_subview_min3 / (r - 2);
//       REQUIRE( x_min == (srow + 1) + r * scol );
// 
//       REQUIRE( mval.real() == Approx(mval1.real()) );
//       REQUIRE( mval.imag() == Approx(mval1.imag()) );
//       REQUIRE( mval.real() == Approx(mval2.real()) );
//       REQUIRE( mval.imag() == Approx(mval2.imag()) );
//       REQUIRE( mval.real() == Approx(mval3.real()) );
//       REQUIRE( mval.imag() == Approx(mval3.imag()) );
//       }
//     }
//   }



TEST_CASE("fn_min_sp_cx_incomplete_subview_col_min_test", "[min]")
  {
  for (size_t r = 10; r < 50; ++r)
    {
    arma::sp_cx_vec x;  x.sprandn(r, 1, 0.3);

    uword x_min          = x.index_min();
    uword x_subview_min1 = x.submat(1, 0, r - 2, 0).index_min();
    uword x_subview_min2 = x.rows(1, r - 2).index_min();

    const std::complex<double> mval  = x.min();
    const std::complex<double> mval1 = x.submat(1, 0, r - 2, 0).min();
    const std::complex<double> mval2 = x.rows(1, r - 2).min();

    if (x_min != 0 && x_min != r - 1 && mval != std::complex<double>(0.0))
      {
      REQUIRE( x_min == x_subview_min1 + 1 );
      REQUIRE( x_min == x_subview_min2 + 1 );

      REQUIRE( mval.real() == Approx(mval1.real()) );
      REQUIRE( mval.imag() == Approx(mval1.imag()) );
      REQUIRE( mval.real() == Approx(mval2.real()) );
      REQUIRE( mval.imag() == Approx(mval2.imag()) );

      REQUIRE( mval.real() == Approx(x(x_min).real()) );
      REQUIRE( mval.imag() == Approx(x(x_min).imag()) );
      }
    }
  }



TEST_CASE("fn_min_sp_cx_incomplete_subview_row_min_test", "[min]")
  {
  for (size_t r = 10; r < 50; ++r)
    {
    sp_cx_rowvec x;  x.sprandn(1, r, 0.3);

    uword x_min          = x.index_min();
    uword x_subview_min1 = x.submat(0, 1, 0, r - 2).index_min();
    uword x_subview_min2 = x.cols(1, r - 2).index_min();

    const std::complex<double> mval  = x.min();
    const std::complex<double> mval1 = x.submat(0, 1, 0, r - 2).min();
    const std::complex<double> mval2 = x.cols(1, r - 2).min();

    if (x_min != 0 && x_min != r - 1 && mval != std::complex<double>(0.0))
      {
      REQUIRE( x_min == x_subview_min1 + 1 );
      REQUIRE( x_min == x_subview_min2 + 1 );

      REQUIRE( mval.real() == Approx(mval1.real()) );
      REQUIRE( mval.imag() == Approx(mval1.imag()) );
      REQUIRE( mval.real() == Approx(mval2.real()) );
      REQUIRE( mval.imag() == Approx(mval2.imag()) );

      REQUIRE( mval.real() == Approx(x(x_min).real()) );
      REQUIRE( mval.imag() == Approx(x(x_min).imag()) );
      }
    }
  }



TEMPLATE_TEST_CASE("fn_min_unary_fp_reference", "[min]", TEST_FLOAT_TYPES)
  {
  typedef TestType eT;

  constexpr eT margin = is_blas_real<eT>::value ? eT(0.001) : eT(0.1);

  Mat<eT> X(10, 10, fill::randn);
  mat X_ref = conv_to<mat>::from(X);

  eT min_val = X.min();
  uword min_val_index = X.index_min();

  double min_val_ref = X_ref.min();
  uword min_val_index_ref = X_ref.index_min();

  REQUIRE( min_val == Approx(eT(min_val_ref)).margin(margin) );
  REQUIRE( min_val_index == min_val_index_ref );

  min_val = min(vectorise(X));
  min_val_index = index_min(vectorise(X));
  min_val_ref = min(vectorise(X_ref));
  min_val_index_ref = index_min(vectorise(X_ref));

  REQUIRE( min_val == Approx(eT(min_val_ref)).margin(margin) );
  REQUIRE( min_val_index == min_val_index_ref );

  min_val = X.submat(1, 1, 6, 6).min();
  min_val_index = X.submat(1, 1, 6, 6).index_min();
  min_val_ref = X_ref.submat(1, 1, 6, 6).min();
  min_val_index_ref = X_ref.submat(1, 1, 6, 6).index_min();

  REQUIRE( min_val == Approx(eT(min_val_ref)).margin(margin) );
  REQUIRE( min_val_index == min_val_index_ref );

  min_val = min(vectorise(X.submat(1, 1, 6, 6)));
  min_val_index = index_min(vectorise(X.submat(1, 1, 6, 6)));
  min_val_ref = min(vectorise(X_ref.submat(1, 1, 6, 6)));
  min_val_index_ref = index_min(vectorise(X_ref.submat(1, 1, 6, 6)));

  REQUIRE( min_val == Approx(eT(min_val_ref)).margin(margin) );
  REQUIRE( min_val_index == min_val_index_ref );
  }



TEMPLATE_TEST_CASE("fn_min_binary_fp_reference", "[min]", TEST_FLOAT_TYPES)
  {
  typedef TestType eT;

  Mat<eT> X(10, 10, fill::randn);
  Mat<eT> Y(10, 10, fill::randn);

  Mat<eT> Z = min(X, Y);

  REQUIRE( Z.n_rows == X.n_rows );
  REQUIRE( Z.n_cols == X.n_cols );

  for (uword i = 0; i < Z.n_elem; ++i)
    {
    REQUIRE( Z[i] == Approx(std::min(X[i], Y[i])) );
    }

  Z = min(X.submat(1, 1, 6, 6), Y.submat(1, 1, 6, 6));

  REQUIRE( Z.n_rows == 6 );
  REQUIRE( Z.n_cols == 6 );

  for (uword c = 0; c < Z.n_cols; ++c)
    {
    for (uword r = 0; r < Z.n_rows; ++r)
      {
      REQUIRE( Z(r, c) == Approx(std::min(X(r + 1, c + 1), Y(r + 1, c + 1))) );
      }
    }
  }
