// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2011-2023 Ryan Curtin (http://www.ratml.org/)
// Copyright 2017-2023 National ICT Australia (NICTA)
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

TEST_CASE("fn_stddev_empty_sparse_test", "[stddev]")
  {
  SpMat<double> m(100, 100);

  SpRow<double> result = stddev(m);

  REQUIRE( result.n_cols == 100 );
  REQUIRE( result.n_rows == 1 );
  for (uword i = 0; i < 100; ++i)
    {
    REQUIRE( (double) result[i] == Approx(0.0).margin(0.001) );
    }

  result = stddev(m, 0, 0);

  REQUIRE( result.n_cols == 100 );
  REQUIRE( result.n_rows == 1 );
  for (uword i = 0; i < 100; ++i)
    {
    REQUIRE( (double) result[i] == Approx(0.0).margin(0.001) );
    }

  result = stddev(m, 1, 0);

  REQUIRE( result.n_cols == 100 );
  REQUIRE( result.n_rows == 1 );
  for (uword i = 0; i < 100; ++i)
    {
    REQUIRE( (double) result[i] == Approx(0.0).margin(0.001) );
    }

  result = stddev(m, 1);

  REQUIRE( result.n_cols == 100 );
  REQUIRE( result.n_rows == 1 );
  for (uword i = 0; i < 100; ++i)
    {
    REQUIRE( (double) result[i] == Approx(0.0).margin(0.001) );
    }

  SpCol<double> colres = stddev(m, 1, 1);

  REQUIRE( colres.n_cols == 1 );
  REQUIRE( colres.n_rows == 100 );
  for (uword i = 0; i < 100; ++i)
    {
    REQUIRE( (double) colres[i] == Approx(0.0).margin(0.001) );
    }

  colres = stddev(m, 0, 1);

  REQUIRE( colres.n_cols == 1 );
  REQUIRE( colres.n_rows == 100 );
  for (uword i = 0; i < 100; ++i)
    {
    REQUIRE( (double) colres[i] == Approx(0.0).margin(0.001) );
    }
  }



TEST_CASE("fn_stddev_empty_cx_sparse_test", "[stddev]")
  {
  SpMat<std::complex<double> > m(100, 100);

  SpRow<double> result = stddev(m);

  REQUIRE( result.n_cols == 100 );
  REQUIRE( result.n_rows == 1 );
  for (uword i = 0; i < 100; ++i)
    {
    REQUIRE( (double) result[i] == Approx(0.0).margin(0.001) );
    }

  result = stddev(m, 0, 0);

  REQUIRE( result.n_cols == 100 );
  REQUIRE( result.n_rows == 1 );
  for (uword i = 0; i < 100; ++i)
    {
    REQUIRE( (double) result[i] == Approx(0.0).margin(0.001) );
    }

  result = stddev(m, 1, 0);

  REQUIRE( result.n_cols == 100 );
  REQUIRE( result.n_rows == 1 );
  for (uword i = 0; i < 100; ++i)
    {
    REQUIRE( (double) result[i] == Approx(0.0).margin(0.001) );
    }

  result = stddev(m, 1);

  REQUIRE( result.n_cols == 100 );
  REQUIRE( result.n_rows == 1 );
  for (uword i = 0; i < 100; ++i)
    {
    REQUIRE( (double) result[i] == Approx(0.0).margin(0.001) );
    }

  SpCol<double> colres = stddev(m, 1, 1);

  REQUIRE( colres.n_cols == 1 );
  REQUIRE( colres.n_rows == 100 );
  for (uword i = 0; i < 100; ++i)
    {
    REQUIRE( (double) colres[i] == Approx(0.0).margin(0.001) );
    }

  colres = stddev(m, 0, 1);

  REQUIRE( colres.n_cols == 1 );
  REQUIRE( colres.n_rows == 100 );
  for (uword i = 0; i < 100; ++i)
    {
    REQUIRE( (double) colres[i] == Approx(0.0).margin(0.001) );
    }
  }



TEST_CASE("fn_stddev_sparse_test", "[stddev]")
  {
  // Create a random matrix and do variance testing on it, with varying levels
  // of nonzero (eventually this becomes a fully dense matrix).
  for (int i = 0; i < 10; ++i)
    {
    SpMat<double> x;
    x.sprandu(50, 75, ((double) (i + 1)) / 10);
    mat d(x);

    SpRow<double> rr = stddev(x);
    rowvec drr = stddev(d);

    REQUIRE( rr.n_rows == 1 );
    REQUIRE( rr.n_cols == 75 );
    for (uword j = 0; j < 75; ++j)
      {
      REQUIRE( drr[j] == Approx((double) rr[j]) );
      }

    rr = stddev(x, 0);

    REQUIRE( rr.n_rows == 1 );
    REQUIRE( rr.n_cols == 75 );
    for (uword j = 0; j < 75; ++j)
      {
      REQUIRE( drr[j] == Approx((double) rr[j]) );
      }

    rr = stddev(x, 1, 0);
    drr = stddev(d, 1, 0);

    REQUIRE( rr.n_rows == 1 );
    REQUIRE( rr.n_cols == 75 );
    for (uword j = 0; j < 75; ++j)
      {
      REQUIRE( drr[j] == Approx((double) rr[j]) );
      }

    SpCol<double> cr = stddev(x, 0, 1);
    vec dcr = stddev(d, 0, 1);

    REQUIRE( cr.n_rows == 50 );
    REQUIRE( cr.n_cols == 1 );
    for (uword j = 0; j < 50; ++j)
      {
      REQUIRE( dcr[j] == Approx((double) cr[j]) );
      }

    cr = stddev(x, 1, 1);
    dcr = stddev(d, 1, 1);

    REQUIRE( cr.n_rows == 50 );
    REQUIRE( cr.n_cols == 1 );
    for (uword j = 0; j < 50; ++j)
      {
      REQUIRE( dcr[j] == Approx((double) cr[j]) );
      }

    // Now on a subview.
    rr = stddev(x.submat(11, 11, 30, 45), 0, 0);
    drr = stddev(d.submat(11, 11, 30, 45), 0, 0);

    REQUIRE( rr.n_rows == 1 );
    REQUIRE( rr.n_cols == 35 );
    for (uword j = 0; j < 35; ++j)
      {
      REQUIRE( drr[j] == Approx((double) rr[j]) );
      }

    rr = stddev(x.submat(11, 11, 30, 45), 1, 0);
    drr = stddev(d.submat(11, 11, 30, 45), 1, 0);

    REQUIRE( rr.n_rows == 1 );
    REQUIRE( rr.n_cols == 35 );
    for (uword j = 0; j < 35; ++j)
      {
      REQUIRE( drr[j] == Approx((double) rr[j]) );
      }

    cr = stddev(x.submat(11, 11, 30, 45), 0, 1);
    dcr = stddev(d.submat(11, 11, 30, 45), 0, 1);

    REQUIRE( cr.n_rows == 20 );
    REQUIRE( cr.n_cols == 1 );
    for (uword j = 0; j < 20; ++j)
      {
      REQUIRE( dcr[j] == Approx((double) cr[j]) );
      }

    cr = stddev(x.submat(11, 11, 30, 45), 1, 1);
    dcr = stddev(d.submat(11, 11, 30, 45), 1, 1);

    REQUIRE( cr.n_rows == 20 );
    REQUIRE( cr.n_cols == 1 );
    for (uword j = 0; j < 20; ++j)
      {
      REQUIRE( dcr[j] == Approx((double) cr[j]) );
      }

    // Now on an SpOp (spop_scalar_times)
    rr = stddev(3.0 * x, 0, 0);
    drr = stddev(3.0 * d, 0, 0);

    REQUIRE( rr.n_rows == 1 );
    REQUIRE( rr.n_cols == 75 );
    for (uword j = 0; j < 75; ++j)
      {
      REQUIRE( drr[j] == Approx((double) rr[j]) );
      }

    rr = stddev(3.0 * x, 1, 0);
    drr = stddev(3.0 * d, 1, 0);

    REQUIRE( rr.n_rows == 1 );
    REQUIRE( rr.n_cols == 75 );
    for (uword j = 0; j < 75; ++j)
      {
      REQUIRE( drr[j] == Approx((double) rr[j]) );
      }

    cr = stddev(4.5 * x, 0, 1);
    dcr = stddev(4.5 * d, 0, 1);

    REQUIRE( cr.n_rows == 50 );
    REQUIRE( cr.n_cols == 1 );
    for (uword j = 0; j < 50; ++j)
      {
      REQUIRE( dcr[j] == Approx((double) cr[j]) );
      }

    cr = stddev(4.5 * x, 1, 1);
    dcr = stddev(4.5 * d, 1, 1);

    REQUIRE( cr.n_rows == 50 );
    REQUIRE( cr.n_cols == 1 );
    for (uword j = 0; j < 50; ++j)
      {
      REQUIRE( dcr[j] == Approx((double) cr[j]) );
      }

    // Now on an SpGlue!
    SpMat<double> y;
    y.sprandu(50, 75, 0.3);
    mat e(y);

    rr = stddev(x + y);
    drr = stddev(d + e);

    REQUIRE( rr.n_rows == 1 );
    REQUIRE( rr.n_cols == 75 );
    for (uword j = 0; j < 75; ++j)
      {
      REQUIRE( drr[j] == Approx((double) rr[j]) );
      }

    rr = stddev(x + y, 1);
    drr = stddev(d + e, 1);

    REQUIRE( rr.n_rows == 1 );
    REQUIRE( rr.n_cols == 75 );
    for (uword j = 0; j < 75; ++j)
      {
      REQUIRE( drr[j] == Approx((double) rr[j]) );
      }

    cr = stddev(x + y, 0, 1);
    dcr = stddev(d + e, 0, 1);

    REQUIRE( cr.n_rows == 50 );
    REQUIRE( cr.n_cols == 1 );
    for (uword j = 0; j < 50; ++j)
      {
      REQUIRE( dcr[j] == Approx((double) cr[j]) );
      }

    cr = stddev(x + y, 1, 1);
    dcr = stddev(d + e, 1, 1);

    REQUIRE( cr.n_rows == 50 );
    REQUIRE( cr.n_cols == 1 );
    for (uword j = 0; j < 50; ++j)
      {
      REQUIRE( dcr[j] == Approx((double) cr[j]) );
      }
    }
  }



TEST_CASE("fn_stddev_sparse_cx_test", "[stddev]")
  {
  // Create a random matrix and do variance testing on it, with varying levels
  // of nonzero (eventually this becomes a fully dense matrix).
  for (int i = 0; i < 10; ++i)
    {
    SpMat<std::complex<double> > x;
    x.sprandu(50, 75, ((double) (i + 1)) / 10);
    cx_mat d(x);

    SpRow<double> rr = stddev(x);
    rowvec drr = stddev(d);

    REQUIRE( rr.n_rows == 1 );
    REQUIRE( rr.n_cols == 75 );
    for (uword j = 0; j < 75; ++j)
      {
      REQUIRE( drr[j] == Approx((double) rr[j]) );
      }

    rr = stddev(x, 0);

    REQUIRE( rr.n_rows == 1 );
    REQUIRE( rr.n_cols == 75 );
    for (uword j = 0; j < 75; ++j)
      {
      REQUIRE( drr[j] == Approx((double) rr[j]) );
      }

    rr = stddev(x, 1, 0);
    drr = stddev(d, 1, 0);

    REQUIRE( rr.n_rows == 1 );
    REQUIRE( rr.n_cols == 75 );
    for (uword j = 0; j < 75; ++j)
      {
      REQUIRE( drr[j] == Approx((double) rr[j]) );
      }

    SpCol<double> cr = stddev(x, 0, 1);
    vec dcr = stddev(d, 0, 1);

    REQUIRE( cr.n_rows == 50 );
    REQUIRE( cr.n_cols == 1 );
    for (uword j = 0; j < 50; ++j)
      {
      REQUIRE( dcr[j] == Approx((double) cr[j]) );
      }

    cr = stddev(x, 1, 1);
    dcr = stddev(d, 1, 1);

    REQUIRE( cr.n_rows == 50 );
    REQUIRE( cr.n_cols == 1 );
    for (uword j = 0; j < 50; ++j)
      {
      REQUIRE( dcr[j] == Approx((double) cr[j]) );
      }

    // Now on a subview.
    rr = stddev(x.submat(11, 11, 30, 45), 0, 0);
    drr = stddev(d.submat(11, 11, 30, 45), 0, 0);

    REQUIRE( rr.n_rows == 1 );
    REQUIRE( rr.n_cols == 35 );
    for (uword j = 0; j < 35; ++j)
      {
      REQUIRE( drr[j] == Approx((double) rr[j]) );
      }

    rr = stddev(x.submat(11, 11, 30, 45), 1, 0);
    drr = stddev(d.submat(11, 11, 30, 45), 1, 0);

    REQUIRE( rr.n_rows == 1 );
    REQUIRE( rr.n_cols == 35 );
    for (uword j = 0; j < 35; ++j)
      {
      REQUIRE( drr[j] == Approx((double) rr[j]) );
      }

    cr = stddev(x.submat(11, 11, 30, 45), 0, 1);
    dcr = stddev(d.submat(11, 11, 30, 45), 0, 1);

    REQUIRE( cr.n_rows == 20 );
    REQUIRE( cr.n_cols == 1 );
    for (uword j = 0; j < 20; ++j)
      {
      REQUIRE( dcr[j] == Approx((double) cr[j]) );
      }

    cr = stddev(x.submat(11, 11, 30, 45), 1, 1);
    dcr = stddev(d.submat(11, 11, 30, 45), 1, 1);

    REQUIRE( cr.n_rows == 20 );
    REQUIRE( cr.n_cols == 1 );
    for (uword j = 0; j < 20; ++j)
      {
      REQUIRE( dcr[j] == Approx((double) cr[j]) );
      }

    // Now on an SpOp (spop_scalar_times)
    rr = stddev(3.0 * x, 0, 0);
    drr = stddev(3.0 * d, 0, 0);

    REQUIRE( rr.n_rows == 1 );
    REQUIRE( rr.n_cols == 75 );
    for (uword j = 0; j < 75; ++j)
      {
      REQUIRE( drr[j] == Approx((double) rr[j]) );
      }

    rr = stddev(3.0 * x, 1, 0);
    drr = stddev(3.0 * d, 1, 0);

    REQUIRE( rr.n_rows == 1 );
    REQUIRE( rr.n_cols == 75 );
    for (uword j = 0; j < 75; ++j)
      {
      REQUIRE( drr[j] == Approx((double) rr[j]) );
      }

    cr = stddev(4.5 * x, 0, 1);
    dcr = stddev(4.5 * d, 0, 1);

    REQUIRE( cr.n_rows == 50 );
    REQUIRE( cr.n_cols == 1 );
    for (uword j = 0; j < 50; ++j)
      {
      REQUIRE( dcr[j] == Approx((double) cr[j]) );
      }

    cr = stddev(4.5 * x, 1, 1);
    dcr = stddev(4.5 * d, 1, 1);

    REQUIRE( cr.n_rows == 50 );
    REQUIRE( cr.n_cols == 1 );
    for (uword j = 0; j < 50; ++j)
      {
      REQUIRE( dcr[j] == Approx((double) cr[j]) );
      }

    // Now on an SpGlue!
    SpMat<std::complex<double> > y;
    y.sprandu(50, 75, 0.3);
    cx_mat e(y);

    rr = stddev(x + y);
    drr = stddev(d + e);

    REQUIRE( rr.n_rows == 1 );
    REQUIRE( rr.n_cols == 75 );
    for (uword j = 0; j < 75; ++j)
      {
      REQUIRE( drr[j] == Approx((double) rr[j]) );
      }

    rr = stddev(x + y, 1);
    drr = stddev(d + e, 1);

    REQUIRE( rr.n_rows == 1 );
    REQUIRE( rr.n_cols == 75 );
    for (uword j = 0; j < 75; ++j)
      {
      REQUIRE( drr[j] == Approx((double) rr[j]) );
      }

    cr = stddev(x + y, 0, 1);
    dcr = stddev(d + e, 0, 1);

    REQUIRE( cr.n_rows == 50 );
    REQUIRE( cr.n_cols == 1 );
    for (uword j = 0; j < 50; ++j)
      {
      REQUIRE( dcr[j] == Approx((double) cr[j]) );
      }

    cr = stddev(x + y, 1, 1);
    dcr = stddev(d + e, 1, 1);

    REQUIRE( cr.n_rows == 50 );
    REQUIRE( cr.n_cols == 1 );
    for (uword j = 0; j < 50; ++j)
      {
      REQUIRE( dcr[j] == Approx((double) cr[j]) );
      }
    }
  }



TEST_CASE("fn_stddev_sparse_alias_test", "[stddev]")
  {
  sp_mat s;
  s.sprandu(70, 70, 0.3);
  mat d(s);

  s = stddev(s);
  d = stddev(d);

  REQUIRE( d.n_rows == s.n_rows );
  REQUIRE( d.n_cols == s.n_cols );
  for (uword i = 0; i < d.n_elem; ++i)
    {
    REQUIRE(d[i] == Approx((double) s[i]) );
    }

  s.sprandu(70, 70, 0.3);
  d = s;

  s = stddev(s, 1);
  d = stddev(d, 1);
  for (uword i = 0; i < d.n_elem; ++i)
    {
    REQUIRE( d[i] == Approx((double) s[i]) );
    }
  }



TEMPLATE_TEST_CASE("fn_stddev_fp_reference", "[stddev]", TEST_FLOAT_TYPES)
  {
  typedef TestType eT;

  Col<eT> X(10, fill::randn);
  vec X_ref = conv_to<vec>::from(X);

  const eT stddev_val = stddev(X);
  const double stddev_ref = stddev(X_ref);

  const eT margin = is_blas_real<eT>::value ? eT(0.001) : eT(0.02);

  REQUIRE( stddev_val == Approx(eT(stddev_ref)).margin(margin) );
  }



TEMPLATE_TEST_CASE("fn_stddev_sp_fp_reference", "[stddev]", TEST_FLOAT_TYPES)
  {
  typedef TestType eT;

  SpCol<eT> X;
  X.sprandu(100, 1, 0.3);
  sp_vec X_ref = conv_to<sp_mat>::from(X);

  const eT stddev_val = stddev(X);
  const double stddev_ref = stddev(X_ref);

  const eT margin = is_blas_real<eT>::value ? eT(0.001) : eT(0.02);

  REQUIRE( stddev_val == Approx(eT(stddev_ref)).margin(margin) );
  }
