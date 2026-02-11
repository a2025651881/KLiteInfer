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


TEST_CASE("fn_symmat_1", "[symmat]")
  {
  mat A = 
    "\
     0.061198   0.201990   0.019678  -0.493936  -0.126745   0.051408;\
     0.437242   0.058956  -0.149362  -0.045465   0.296153   0.035437;\
    -0.492474  -0.031309   0.314156   0.419733   0.068317  -0.454499;\
     0.336352   0.411541   0.458476  -0.393139  -0.135040   0.373833;\
     0.239585  -0.428913  -0.406953  -0.291020  -0.353768   0.258704;\
    ";
  
  mat B = symmatu( A(0,0,size(5,5)) );
  mat C = symmatl( A(0,0,size(5,5)) );
  
  mat BB = 
    "\
     0.061198   0.201990   0.019678  -0.493936  -0.126745;\
     0.201990   0.058956  -0.149362  -0.045465   0.296153;\
     0.019678  -0.149362   0.314156   0.419733   0.068317;\
    -0.493936  -0.045465   0.419733  -0.393139  -0.135040;\
    -0.126745   0.296153   0.068317  -0.135040  -0.353768;\
    ";
  
  mat CC = 
    "\
     0.061198   0.437242  -0.492474   0.336352   0.239585;\
     0.437242   0.058956  -0.031309   0.411541  -0.428913;\
    -0.492474  -0.031309   0.314156   0.458476  -0.406953;\
     0.336352   0.411541   0.458476  -0.393139  -0.291020;\
     0.239585  -0.428913  -0.406953  -0.291020  -0.353768;\
    ";
  
  REQUIRE( accu(abs( B - BB )) == Approx(0.0).margin(0.001) );
  REQUIRE( accu(abs( C - CC )) == Approx(0.0).margin(0.001) );
  
  mat X;
  REQUIRE_THROWS( X = symmatu(A) ); // symmatu() and symmatl() currently handle only square matrices
  }



TEST_CASE("fn_symmat_2", "[symmat]")
  {
  mat A = 
    "\
     0.061198   0.201990   0.019678  -0.493936  -0.126745   0.051408;\
     0.437242   0.058956  -0.149362  -0.045465   0.296153   0.035437;\
    -0.492474  -0.031309   0.314156   0.419733   0.068317  -0.454499;\
     0.336352   0.411541   0.458476  -0.393139  -0.135040   0.373833;\
     0.239585  -0.428913  -0.406953  -0.291020  -0.353768   0.258704;\
    ";
  
  cx_mat B1 = symmatu( cx_mat( A(0,0,size(3,3)), A(0,3,size(3,3)) ) );
  cx_mat C1 = symmatl( cx_mat( A(0,0,size(3,3)), A(0,3,size(3,3)) ) );
  
  cx_mat B2 = symmatu( cx_mat( A(0,0,size(3,3)), A(0,3,size(3,3)) ), true );
  cx_mat C2 = symmatl( cx_mat( A(0,0,size(3,3)), A(0,3,size(3,3)) ), true );
  
  cx_mat D  = symmatu( cx_mat( A(0,0,size(3,3)), A(0,3,size(3,3)) ), false );
  cx_mat E  = symmatl( cx_mat( A(0,0,size(3,3)), A(0,3,size(3,3)) ), false );
  
  cx_mat BB = 
    {
    { cx_double( 0.06120, -0.49394), cx_double( 0.20199, -0.12674), cx_double( 0.01968, +0.05141) },
    { cx_double( 0.20199, +0.12674), cx_double( 0.05896, +0.29615), cx_double(-0.14936, +0.03544) },
    { cx_double( 0.01968, -0.05141), cx_double(-0.14936, -0.03544), cx_double( 0.31416, -0.45450) }
    };
  
  cx_mat CC = 
    {
    { cx_double( 0.06120, -0.49394), cx_double( 0.43724, +0.04546), cx_double(-0.49247, -0.41973) },
    { cx_double( 0.43724, -0.04546), cx_double( 0.05896, +0.29615), cx_double(-0.03131, -0.06832) },
    { cx_double(-0.49247, +0.41973), cx_double(-0.03131, +0.06832), cx_double( 0.31416, -0.45450) }
    };
  
  cx_mat DD = 
    {
    { cx_double( 0.06120, -0.49394), cx_double( 0.20199, -0.12674), cx_double( 0.01968, +0.05141) },
    { cx_double( 0.20199, -0.12674), cx_double( 0.05896, +0.29615), cx_double(-0.14936, +0.03544) },
    { cx_double( 0.01968, +0.05141), cx_double(-0.14936, +0.03544), cx_double( 0.31416, -0.45450) }
    };
  
  cx_mat EE = 
    {
    { cx_double( 0.06120, -0.49394), cx_double( 0.43724, -0.04546), cx_double(-0.49247, +0.41973) },
    { cx_double( 0.43724, -0.04546), cx_double( 0.05896, +0.29615), cx_double(-0.03131, +0.06832) },
    { cx_double(-0.49247, +0.41973), cx_double(-0.03131, +0.06832), cx_double( 0.31416, -0.45450) }
    };
  
  REQUIRE( accu(abs( B1 - BB )) == Approx(0.0).margin(0.0001) );
  REQUIRE( accu(abs( C1 - CC )) == Approx(0.0).margin(0.0001) );
  
  REQUIRE( accu(abs( B2 - BB )) == Approx(0.0).margin(0.0001) );
  REQUIRE( accu(abs( C2 - CC )) == Approx(0.0).margin(0.0001) );
  
  REQUIRE( accu(abs( D  - DD )) == Approx(0.0).margin(0.0001) );
  REQUIRE( accu(abs( E  - EE )) == Approx(0.0).margin(0.0001) );
  
  cx_mat  X;
  REQUIRE_THROWS( X = symmatu( cx_mat(A(0,0,size(2,3)), A(0,3,size(2,3))) ) ); // symmatu() and symmatl() currently handle only square matrices
  }



TEMPLATE_TEST_CASE("fn_symmat_fp", "[symmat]", TEST_FLOAT_TYPES)
  {
  typedef TestType eT;

  Mat<eT> X(10, 10, fill::randn);

  Mat<eT> XU = symmatu(X);
  Mat<eT> XL = symmatl(X);

  REQUIRE( XU.n_rows == X.n_rows );
  REQUIRE( XU.n_cols == X.n_cols );
  REQUIRE( XL.n_rows == X.n_rows );
  REQUIRE( XL.n_cols == X.n_cols );

  constexpr const eT tol = is_blas_real<eT>::value ? eT(0.0001) : eT(0.01);

  for (uword c = 0; c < X.n_cols; ++c)
    {
    for (uword r = 0; r < X.n_rows; ++r)
      {
      if (r > c)
        {
        REQUIRE( XU(r, c) == Approx(X(c, r)).epsilon(tol) );
        }
      else
        {
        REQUIRE( XU(r, c) == Approx(X(r, c)).epsilon(tol) );
        }

      if (c > r)
        {
        REQUIRE( XL(r, c) == Approx(X(c, r)).epsilon(tol) );
        }
      else
        {
        REQUIRE( XL(r, c) == Approx(X(r, c)).epsilon(tol) );
        }
      }
    }
  }



TEMPLATE_TEST_CASE("fn_symmat_sparse_fp", "[symmat]", TEST_FLOAT_TYPES)
  {
  typedef TestType eT;

  SpMat<eT> X;
  X.sprandn(20, 20, 0.3);

  SpMat<eT> XU = symmatu(X);
  SpMat<eT> XL = symmatl(X);

  REQUIRE( XU.n_rows == X.n_rows );
  REQUIRE( XU.n_cols == X.n_cols );
  REQUIRE( XL.n_rows == X.n_rows );
  REQUIRE( XL.n_cols == X.n_cols );

  constexpr const eT tol = is_blas_real<eT>::value ? eT(0.0001) : eT(0.01);

  for (uword c = 0; c < X.n_cols; ++c)
    {
    for (uword r = 0; r < X.n_rows; ++r)
      {
      if (r > c)
        {
        REQUIRE( XU(r, c) == Approx(X(c, r)).epsilon(tol) );
        }
      else
        {
        REQUIRE( XU(r, c) == Approx(X(r, c)).epsilon(tol) );
        }

      if (c > r)
        {
        REQUIRE( XL(r, c) == Approx(X(c, r)).epsilon(tol) );
        }
      else
        {
        REQUIRE( XL(r, c) == Approx(X(r, c)).epsilon(tol) );
        }
      }
    }
  }
