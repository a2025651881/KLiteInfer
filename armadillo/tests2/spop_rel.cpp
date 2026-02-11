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

using namespace arma;

TEMPLATE_TEST_CASE("spmat_rel_val", "[spop_rel]", float, double, int, long)
  {
  typedef TestType eT;

  Mat<eT> X_ref {{ 1, 0, 0 }, { 0, 2, -2 }, { -5, 0, 5 }};
  SpMat<eT> X(X_ref);

  sp_umat Y1 =  (X > 1);
  sp_umat Y2 =  (1 > X);
  sp_umat Y3 =  (X >= 1);
  sp_umat Y4 =  (1 >= X);
  sp_umat Y5 =  (X < 0);
  sp_umat Y6 =  (0 < X);
  sp_umat Y7 =  (X <= -1);
  sp_umat Y8 =  (-1 <= X);
  sp_umat Y9 =  (X == 5);
  sp_umat Y10 = (5 == X);
  sp_umat Y11 = (X == 0);
  sp_umat Y12 = (0 == X);
  sp_umat Y13 = (X != 2);
  sp_umat Y14 = (2 != X);
  sp_umat Y15 = (X != 0);
  sp_umat Y16 = (0 != X);

  umat Y1_ref  = {{ 0, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }}; // X > 1
  umat Y2_ref  = {{ 0, 1, 1 }, { 1, 0, 1 }, { 1, 1, 0 }}; // 1 > X
  umat Y3_ref  = {{ 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }}; // X >= 1
  umat Y4_ref  = {{ 1, 1, 1 }, { 1, 0, 1 }, { 1, 1, 0 }}; // 1 >= X
  umat Y5_ref  = {{ 0, 0, 0 }, { 0, 0, 1 }, { 1, 0, 0 }}; // X < 0
  umat Y6_ref  = {{ 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }}; // 0 < X
  umat Y7_ref  = {{ 0, 0, 0 }, { 0, 0, 1 }, { 1, 0, 0 }}; // X <= -1
  umat Y8_ref  = {{ 1, 1, 1 }, { 1, 1, 0 }, { 0, 1, 1 }}; // -1 <= X
  umat Y9_ref  = {{ 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 1 }}; // X == 5
  umat Y10_ref = {{ 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 1 }}; // 5 == X
  umat Y11_ref = {{ 0, 1, 1 }, { 1, 0, 0 }, { 0, 1, 0 }}; // X == 0
  umat Y12_ref = {{ 0, 1, 1 }, { 1, 0, 0 }, { 0, 1, 0 }}; // 0 == X
  umat Y13_ref = {{ 1, 1, 1 }, { 1, 0, 1 }, { 1, 1, 1 }}; // X != 2
  umat Y14_ref = {{ 1, 1, 1 }, { 1, 0, 1 }, { 1, 1, 1 }}; // 2 != X
  umat Y15_ref = {{ 1, 0, 0 }, { 0, 1, 1 }, { 1, 0, 1 }}; // X != 0
  umat Y16_ref = {{ 1, 0, 0 }, { 0, 1, 1 }, { 1, 0, 1 }}; // 0 != X

  REQUIRE( all(all(Y1.as_dense()  == Y1_ref )) );
  REQUIRE( all(all(Y2.as_dense()  == Y2_ref )) );
  REQUIRE( all(all(Y3.as_dense()  == Y3_ref )) );
  REQUIRE( all(all(Y4.as_dense()  == Y4_ref )) );
  REQUIRE( all(all(Y5.as_dense()  == Y5_ref )) );
  REQUIRE( all(all(Y6.as_dense()  == Y6_ref )) );
  REQUIRE( all(all(Y7.as_dense()  == Y7_ref )) );
  REQUIRE( all(all(Y8.as_dense()  == Y8_ref )) );
  REQUIRE( all(all(Y9.as_dense()  == Y9_ref )) );
  REQUIRE( all(all(Y10.as_dense() == Y10_ref)) );
  REQUIRE( all(all(Y11.as_dense() == Y11_ref)) );
  REQUIRE( all(all(Y12.as_dense() == Y12_ref)) );
  REQUIRE( all(all(Y13.as_dense() == Y13_ref)) );
  REQUIRE( all(all(Y14.as_dense() == Y14_ref)) );
  REQUIRE( all(all(Y15.as_dense() == Y15_ref)) );
  REQUIRE( all(all(Y16.as_dense() == Y16_ref)) );
  }



TEMPLATE_TEST_CASE("spcol_rel_val", "[spop_rel]", float, double, int, long)
  {
  typedef TestType eT;

  Col<eT> X_ref { 1, 0, 0, 0, 2, -2, -5, 0, 5 };
  SpCol<eT> X(X_ref);

  sp_umat Y1 =  (X > 1);
  sp_umat Y2 =  (1 > X);
  sp_umat Y3 =  (X >= 1);
  sp_umat Y4 =  (1 >= X);
  sp_umat Y5 =  (X < 0);
  sp_umat Y6 =  (0 < X);
  sp_umat Y7 =  (X <= -1);
  sp_umat Y8 =  (-1 <= X);
  sp_umat Y9 =  (X == 5);
  sp_umat Y10 = (5 == X);
  sp_umat Y11 = (X == 0);
  sp_umat Y12 = (0 == X);
  sp_umat Y13 = (X != 2);
  sp_umat Y14 = (2 != X);
  sp_umat Y15 = (X != 0);
  sp_umat Y16 = (0 != X);

  uvec Y1_ref  = { 0, 0, 0, 0, 1, 0, 0, 0, 1 }; // X > 1
  uvec Y2_ref  = { 0, 1, 1, 1, 0, 1, 1, 1, 0 }; // 1 > X
  uvec Y3_ref  = { 1, 0, 0, 0, 1, 0, 0, 0, 1 }; // X >= 1
  uvec Y4_ref  = { 1, 1, 1, 1, 0, 1, 1, 1, 0 }; // 1 >= X
  uvec Y5_ref  = { 0, 0, 0, 0, 0, 1, 1, 0, 0 }; // X < 0
  uvec Y6_ref  = { 1, 0, 0, 0, 1, 0, 0, 0, 1 }; // 0 < X
  uvec Y7_ref  = { 0, 0, 0, 0, 0, 1, 1, 0, 0 }; // X <= -1
  uvec Y8_ref  = { 1, 1, 1, 1, 1, 0, 0, 1, 1 }; // -1 <= X
  uvec Y9_ref  = { 0, 0, 0, 0, 0, 0, 0, 0, 1 }; // X == 5
  uvec Y10_ref = { 0, 0, 0, 0, 0, 0, 0, 0, 1 }; // 5 == X
  uvec Y11_ref = { 0, 1, 1, 1, 0, 0, 0, 1, 0 }; // X == 0
  uvec Y12_ref = { 0, 1, 1, 1, 0, 0, 0, 1, 0 }; // 0 == X
  uvec Y13_ref = { 1, 1, 1, 1, 0, 1, 1, 1, 1 }; // X != 2
  uvec Y14_ref = { 1, 1, 1, 1, 0, 1, 1, 1, 1 }; // 2 != X
  uvec Y15_ref = { 1, 0, 0, 0, 1, 1, 1, 0, 1 }; // X != 0
  uvec Y16_ref = { 1, 0, 0, 0, 1, 1, 1, 0, 1 }; // 0 != X

  REQUIRE( all(Y1.as_dense()  == Y1_ref ) );
  REQUIRE( all(Y2.as_dense()  == Y2_ref ) );
  REQUIRE( all(Y3.as_dense()  == Y3_ref ) );
  REQUIRE( all(Y4.as_dense()  == Y4_ref ) );
  REQUIRE( all(Y5.as_dense()  == Y5_ref ) );
  REQUIRE( all(Y6.as_dense()  == Y6_ref ) );
  REQUIRE( all(Y7.as_dense()  == Y7_ref ) );
  REQUIRE( all(Y8.as_dense()  == Y8_ref ) );
  REQUIRE( all(Y9.as_dense()  == Y9_ref ) );
  REQUIRE( all(Y10.as_dense() == Y10_ref) );
  REQUIRE( all(Y11.as_dense() == Y11_ref) );
  REQUIRE( all(Y12.as_dense() == Y12_ref) );
  REQUIRE( all(Y13.as_dense() == Y13_ref) );
  REQUIRE( all(Y14.as_dense() == Y14_ref) );
  REQUIRE( all(Y15.as_dense() == Y15_ref) );
  REQUIRE( all(Y16.as_dense() == Y16_ref) );
  }



TEMPLATE_TEST_CASE("sprow_rel_val", "[spop_rel]", float, double, int, long)
  {
  typedef TestType eT;

  Row<eT> X_ref { 1, 0, 0, 0, 2, -2, -5, 0, 5 };
  SpRow<eT> X(X_ref);

  sp_umat Y1 =  (X > 1);
  sp_umat Y2 =  (1 > X);
  sp_umat Y3 =  (X >= 1);
  sp_umat Y4 =  (1 >= X);
  sp_umat Y5 =  (X < 0);
  sp_umat Y6 =  (0 < X);
  sp_umat Y7 =  (X <= -1);
  sp_umat Y8 =  (-1 <= X);
  sp_umat Y9 =  (X == 5);
  sp_umat Y10 = (5 == X);
  sp_umat Y11 = (X == 0);
  sp_umat Y12 = (0 == X);
  sp_umat Y13 = (X != 2);
  sp_umat Y14 = (2 != X);
  sp_umat Y15 = (X != 0);
  sp_umat Y16 = (0 != X);

  urowvec Y1_ref  = { 0, 0, 0, 0, 1, 0, 0, 0, 1 }; // X > 1
  urowvec Y2_ref  = { 0, 1, 1, 1, 0, 1, 1, 1, 0 }; // 1 > X
  urowvec Y3_ref  = { 1, 0, 0, 0, 1, 0, 0, 0, 1 }; // X >= 1
  urowvec Y4_ref  = { 1, 1, 1, 1, 0, 1, 1, 1, 0 }; // 1 >= X
  urowvec Y5_ref  = { 0, 0, 0, 0, 0, 1, 1, 0, 0 }; // X < 0
  urowvec Y6_ref  = { 1, 0, 0, 0, 1, 0, 0, 0, 1 }; // 0 < X
  urowvec Y7_ref  = { 0, 0, 0, 0, 0, 1, 1, 0, 0 }; // X <= -1
  urowvec Y8_ref  = { 1, 1, 1, 1, 1, 0, 0, 1, 1 }; // -1 <= X
  urowvec Y9_ref  = { 0, 0, 0, 0, 0, 0, 0, 0, 1 }; // X == 5
  urowvec Y10_ref = { 0, 0, 0, 0, 0, 0, 0, 0, 1 }; // 5 == X
  urowvec Y11_ref = { 0, 1, 1, 1, 0, 0, 0, 1, 0 }; // X == 0
  urowvec Y12_ref = { 0, 1, 1, 1, 0, 0, 0, 1, 0 }; // 0 == X
  urowvec Y13_ref = { 1, 1, 1, 1, 0, 1, 1, 1, 1 }; // X != 2
  urowvec Y14_ref = { 1, 1, 1, 1, 0, 1, 1, 1, 1 }; // 2 != X
  urowvec Y15_ref = { 1, 0, 0, 0, 1, 1, 1, 0, 1 }; // X != 0
  urowvec Y16_ref = { 1, 0, 0, 0, 1, 1, 1, 0, 1 }; // 0 != X

  REQUIRE( all(Y1.as_dense()  == Y1_ref ) );
  REQUIRE( all(Y2.as_dense()  == Y2_ref ) );
  REQUIRE( all(Y3.as_dense()  == Y3_ref ) );
  REQUIRE( all(Y4.as_dense()  == Y4_ref ) );
  REQUIRE( all(Y5.as_dense()  == Y5_ref ) );
  REQUIRE( all(Y6.as_dense()  == Y6_ref ) );
  REQUIRE( all(Y7.as_dense()  == Y7_ref ) );
  REQUIRE( all(Y8.as_dense()  == Y8_ref ) );
  REQUIRE( all(Y9.as_dense()  == Y9_ref ) );
  REQUIRE( all(Y10.as_dense() == Y10_ref) );
  REQUIRE( all(Y11.as_dense() == Y11_ref) );
  REQUIRE( all(Y12.as_dense() == Y12_ref) );
  REQUIRE( all(Y13.as_dense() == Y13_ref) );
  REQUIRE( all(Y14.as_dense() == Y14_ref) );
  REQUIRE( all(Y15.as_dense() == Y15_ref) );
  REQUIRE( all(Y16.as_dense() == Y16_ref) );
  }



TEMPLATE_TEST_CASE("spsubview_rel_val_small", "[spop_rel]", float, double, int, long)
  {
  typedef TestType eT;

  Mat<eT> X_ref {{ 3, 1, 0, 0, 0 }, { 0, 1, 0, 0, 3}, { 2, 0, 2, -2, 0 }, { 1, -5, 0, 5, 0 }, { 3, -1, -1, 0, 0 }};
  SpMat<eT> X(X_ref);

  sp_umat Y1 =  (X.submat(1, 1, 3, 3) > 1);
  sp_umat Y2 =  (1 > X.submat(1, 1, 3, 3));
  sp_umat Y3 =  (X.submat(1, 1, 3, 3) >= 1);
  sp_umat Y4 =  (1 >= X.submat(1, 1, 3, 3));
  sp_umat Y5 =  (X.submat(1, 1, 3, 3) < 0);
  sp_umat Y6 =  (0 < X.submat(1, 1, 3, 3));
  sp_umat Y7 =  (X.submat(1, 1, 3, 3) <= -1);
  sp_umat Y8 =  (-1 <= X.submat(1, 1, 3, 3));
  sp_umat Y9 =  (X.submat(1, 1, 3, 3) == 5);
  sp_umat Y10 = (5 == X.submat(1, 1, 3, 3));
  sp_umat Y11 = (X.submat(1, 1, 3, 3) == 0);
  sp_umat Y12 = (0 == X.submat(1, 1, 3, 3));
  sp_umat Y13 = (X.submat(1, 1, 3, 3) != 2);
  sp_umat Y14 = (2 != X.submat(1, 1, 3, 3));
  sp_umat Y15 = (X.submat(1, 1, 3, 3) != 0);
  sp_umat Y16 = (0 != X.submat(1, 1, 3, 3));

  umat Y1_ref  = {{ 0, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }}; // X > 1
  umat Y2_ref  = {{ 0, 1, 1 }, { 1, 0, 1 }, { 1, 1, 0 }}; // 1 > X
  umat Y3_ref  = {{ 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }}; // X >= 1
  umat Y4_ref  = {{ 1, 1, 1 }, { 1, 0, 1 }, { 1, 1, 0 }}; // 1 >= X
  umat Y5_ref  = {{ 0, 0, 0 }, { 0, 0, 1 }, { 1, 0, 0 }}; // X < 0
  umat Y6_ref  = {{ 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }}; // 0 < X
  umat Y7_ref  = {{ 0, 0, 0 }, { 0, 0, 1 }, { 1, 0, 0 }}; // X <= -1
  umat Y8_ref  = {{ 1, 1, 1 }, { 1, 1, 0 }, { 0, 1, 1 }}; // -1 <= X
  umat Y9_ref  = {{ 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 1 }}; // X == 5
  umat Y10_ref = {{ 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 1 }}; // 5 == X
  umat Y11_ref = {{ 0, 1, 1 }, { 1, 0, 0 }, { 0, 1, 0 }}; // X == 0
  umat Y12_ref = {{ 0, 1, 1 }, { 1, 0, 0 }, { 0, 1, 0 }}; // 0 == X
  umat Y13_ref = {{ 1, 1, 1 }, { 1, 0, 1 }, { 1, 1, 1 }}; // X != 2
  umat Y14_ref = {{ 1, 1, 1 }, { 1, 0, 1 }, { 1, 1, 1 }}; // 2 != X
  umat Y15_ref = {{ 1, 0, 0 }, { 0, 1, 1 }, { 1, 0, 1 }}; // X != 0
  umat Y16_ref = {{ 1, 0, 0 }, { 0, 1, 1 }, { 1, 0, 1 }}; // 0 != X

  REQUIRE( all(all(Y1.as_dense()  == Y1_ref )) );
  REQUIRE( all(all(Y2.as_dense()  == Y2_ref )) );
  REQUIRE( all(all(Y3.as_dense()  == Y3_ref )) );
  REQUIRE( all(all(Y4.as_dense()  == Y4_ref )) );
  REQUIRE( all(all(Y5.as_dense()  == Y5_ref )) );
  REQUIRE( all(all(Y6.as_dense()  == Y6_ref )) );
  REQUIRE( all(all(Y7.as_dense()  == Y7_ref )) );
  REQUIRE( all(all(Y8.as_dense()  == Y8_ref )) );
  REQUIRE( all(all(Y9.as_dense()  == Y9_ref )) );
  REQUIRE( all(all(Y10.as_dense() == Y10_ref)) );
  REQUIRE( all(all(Y11.as_dense() == Y11_ref)) );
  REQUIRE( all(all(Y12.as_dense() == Y12_ref)) );
  REQUIRE( all(all(Y13.as_dense() == Y13_ref)) );
  REQUIRE( all(all(Y14.as_dense() == Y14_ref)) );
  REQUIRE( all(all(Y15.as_dense() == Y15_ref)) );
  REQUIRE( all(all(Y16.as_dense() == Y16_ref)) );
  }



TEMPLATE_TEST_CASE("spmat_rel_val_vs_mat", "[spop_rel]", float, double)
  {
  typedef TestType eT;

  for (uword trial = 0; trial < 10; ++trial)
    {
    SpMat<eT> X;
    X.sprandn(256, 256, 0.3);

    sp_umat Y1 =  (X > 1);
    sp_umat Y2 =  (1 > X);
    sp_umat Y3 =  (X >= 1);
    sp_umat Y4 =  (1 >= X);
    sp_umat Y5 =  (X < 0);
    sp_umat Y6 =  (0 < X);
    sp_umat Y7 =  (X <= -1);
    sp_umat Y8 =  (-1 <= X);
    sp_umat Y9 =  (X == 3);
    sp_umat Y10 = (3 == X);
    sp_umat Y11 = (X == 0);
    sp_umat Y12 = (0 == X);
    sp_umat Y13 = (X != 2);
    sp_umat Y14 = (2 != X);
    sp_umat Y15 = (X != 0);
    sp_umat Y16 = (0 != X);

    Mat<eT> X_ref(X);

    umat Y1_ref =  (X_ref > 1);
    umat Y2_ref =  (1 > X_ref);
    umat Y3_ref =  (X_ref >= 1);
    umat Y4_ref =  (1 >= X_ref);
    umat Y5_ref =  (X_ref < 0);
    umat Y6_ref =  (0 < X_ref);
    umat Y7_ref =  (X_ref <= -1);
    umat Y8_ref =  (-1 <= X_ref);
    umat Y9_ref =  (X_ref == 3);
    umat Y10_ref = (3 == X_ref);
    umat Y11_ref = (X_ref == 0);
    umat Y12_ref = (0 == X_ref);
    umat Y13_ref = (X_ref != 2);
    umat Y14_ref = (2 != X_ref);
    umat Y15_ref = (X_ref != 0);
    umat Y16_ref = (0 != X_ref);

    REQUIRE( all(all(Y1.as_dense()  == Y1_ref )) );
    REQUIRE( all(all(Y2.as_dense()  == Y2_ref )) );
    REQUIRE( all(all(Y3.as_dense()  == Y3_ref )) );
    REQUIRE( all(all(Y4.as_dense()  == Y4_ref )) );
    REQUIRE( all(all(Y5.as_dense()  == Y5_ref )) );
    REQUIRE( all(all(Y6.as_dense()  == Y6_ref )) );
    REQUIRE( all(all(Y7.as_dense()  == Y7_ref )) );
    REQUIRE( all(all(Y8.as_dense()  == Y8_ref )) );
    REQUIRE( all(all(Y9.as_dense()  == Y9_ref )) );
    REQUIRE( all(all(Y10.as_dense() == Y10_ref)) );
    REQUIRE( all(all(Y11.as_dense() == Y11_ref)) );
    REQUIRE( all(all(Y12.as_dense() == Y12_ref)) );
    REQUIRE( all(all(Y13.as_dense() == Y13_ref)) );
    REQUIRE( all(all(Y14.as_dense() == Y14_ref)) );
    REQUIRE( all(all(Y15.as_dense() == Y15_ref)) );
    REQUIRE( all(all(Y16.as_dense() == Y16_ref)) );
    }
  }



TEMPLATE_TEST_CASE("spsubview_rel_val_vs_mat", "[spop_rel]", float, double, int, long)
  {
  typedef TestType eT;

  for (uword trial = 0; trial < 10; ++trial)
    {
    SpMat<eT> X;
    X.sprandn(256, 256, 0.3);

    sp_umat Y1 =  (X.submat(10, 15, 173, 211) > 1);
    sp_umat Y2 =  (1 > X.submat(10, 15, 173, 211));
    sp_umat Y3 =  (X.submat(10, 15, 173, 211) >= 1);
    sp_umat Y4 =  (1 >= X.submat(10, 15, 173, 211));
    sp_umat Y5 =  (X.submat(10, 15, 173, 211) < 0);
    sp_umat Y6 =  (0 < X.submat(10, 15, 173, 211));
    sp_umat Y7 =  (X.submat(10, 15, 173, 211) <= -1);
    sp_umat Y8 =  (-1 <= X.submat(10, 15, 173, 211));
    sp_umat Y9 =  (X.submat(10, 15, 173, 211) == 3);
    sp_umat Y10 = (3 == X.submat(10, 15, 173, 211));
    sp_umat Y11 = (X.submat(10, 15, 173, 211) == 0);
    sp_umat Y12 = (0 == X.submat(10, 15, 173, 211));
    sp_umat Y13 = (X.submat(10, 15, 173, 211) != 2);
    sp_umat Y14 = (2 != X.submat(10, 15, 173, 211));
    sp_umat Y15 = (X.submat(10, 15, 173, 211) != 0);
    sp_umat Y16 = (0 != X.submat(10, 15, 173, 211));

    Mat<eT> X_ref(X.submat(10, 15, 173, 211));

    umat Y1_ref =  (X_ref > 1);
    umat Y2_ref =  (1 > X_ref);
    umat Y3_ref =  (X_ref >= 1);
    umat Y4_ref =  (1 >= X_ref);
    umat Y5_ref =  (X_ref < 0);
    umat Y6_ref =  (0 < X_ref);
    umat Y7_ref =  (X_ref <= -1);
    umat Y8_ref =  (-1 <= X_ref);
    umat Y9_ref =  (X_ref == 3);
    umat Y10_ref = (3 == X_ref);
    umat Y11_ref = (X_ref == 0);
    umat Y12_ref = (0 == X_ref);
    umat Y13_ref = (X_ref != 2);
    umat Y14_ref = (2 != X_ref);
    umat Y15_ref = (X_ref != 0);
    umat Y16_ref = (0 != X_ref);

    REQUIRE( all(all(Y1.as_dense()  == Y1_ref )) );
    REQUIRE( all(all(Y2.as_dense()  == Y2_ref )) );
    REQUIRE( all(all(Y3.as_dense()  == Y3_ref )) );
    REQUIRE( all(all(Y4.as_dense()  == Y4_ref )) );
    REQUIRE( all(all(Y5.as_dense()  == Y5_ref )) );
    REQUIRE( all(all(Y6.as_dense()  == Y6_ref )) );
    REQUIRE( all(all(Y7.as_dense()  == Y7_ref )) );
    REQUIRE( all(all(Y8.as_dense()  == Y8_ref )) );
    REQUIRE( all(all(Y9.as_dense()  == Y9_ref )) );
    REQUIRE( all(all(Y10.as_dense() == Y10_ref)) );
    REQUIRE( all(all(Y11.as_dense() == Y11_ref)) );
    REQUIRE( all(all(Y12.as_dense() == Y12_ref)) );
    REQUIRE( all(all(Y13.as_dense() == Y13_ref)) );
    REQUIRE( all(all(Y14.as_dense() == Y14_ref)) );
    REQUIRE( all(all(Y15.as_dense() == Y15_ref)) );
    REQUIRE( all(all(Y16.as_dense() == Y16_ref)) );
    }
  }



TEST_CASE("spmat_rel_empty", "[spop_rel]")
  {
  sp_mat X;
  sp_umat Y = (X > 0);

  REQUIRE( Y.is_empty() );
  }



TEST_CASE("spmat_rel_all_zero", "[spop_rel]")
  {
  sp_mat X(150, 150);
  sp_umat Y = (X != 0);

  REQUIRE( Y.n_rows == 150 );
  REQUIRE( Y.n_cols == 150 );
  REQUIRE( all(all(Y.as_dense() == 0)) );
  }



TEMPLATE_TEST_CASE("spmat_rel_accu_mat_comparison", "[spop_rel]", float, double)
  {
  typedef TestType eT;

  for (uword trial = 0; trial < 10; ++trial)
    {
    SpMat<eT> X;
    X.sprandn(256, 256, 0.3);

    uword Y1 =  accu(X > 1);
    uword Y2 =  accu(1 > X);
    uword Y3 =  accu(X >= 1);
    uword Y4 =  accu(1 >= X);
    uword Y5 =  accu(X < 0);
    uword Y6 =  accu(0 < X);
    uword Y7 =  accu(X <= -1);
    uword Y8 =  accu(-1 <= X);
    uword Y9 =  accu(X == 3);
    uword Y10 = accu(3 == X);
    uword Y11 = accu(X == 0);
    uword Y12 = accu(0 == X);
    uword Y13 = accu(X != 2);
    uword Y14 = accu(2 != X);
    uword Y15 = accu(X != 0);
    uword Y16 = accu(0 != X);

    Mat<eT> X_ref(X);

    uword Y1_ref =  accu(X_ref > 1);
    uword Y2_ref =  accu(1 > X_ref);
    uword Y3_ref =  accu(X_ref >= 1);
    uword Y4_ref =  accu(1 >= X_ref);
    uword Y5_ref =  accu(X_ref < 0);
    uword Y6_ref =  accu(0 < X_ref);
    uword Y7_ref =  accu(X_ref <= -1);
    uword Y8_ref =  accu(-1 <= X_ref);
    uword Y9_ref =  accu(X_ref == 3);
    uword Y10_ref = accu(3 == X_ref);
    uword Y11_ref = accu(X_ref == 0);
    uword Y12_ref = accu(0 == X_ref);
    uword Y13_ref = accu(X_ref != 2);
    uword Y14_ref = accu(2 != X_ref);
    uword Y15_ref = accu(X_ref != 0);
    uword Y16_ref = accu(0 != X_ref);

    REQUIRE( Y1  == Y1_ref  );
    REQUIRE( Y2  == Y2_ref  );
    REQUIRE( Y3  == Y3_ref  );
    REQUIRE( Y4  == Y4_ref  );
    REQUIRE( Y5  == Y5_ref  );
    REQUIRE( Y6  == Y6_ref  );
    REQUIRE( Y7  == Y7_ref  );
    REQUIRE( Y8  == Y8_ref  );
    REQUIRE( Y9  == Y9_ref  );
    REQUIRE( Y10 == Y10_ref );
    REQUIRE( Y11 == Y11_ref );
    REQUIRE( Y12 == Y12_ref );
    REQUIRE( Y13 == Y13_ref );
    REQUIRE( Y14 == Y14_ref );
    REQUIRE( Y15 == Y15_ref );
    REQUIRE( Y16 == Y16_ref );
    }
  }



TEMPLATE_TEST_CASE("spsubview_rel_accu_mat_comparison", "[spop_rel]", float, double)
  {
  typedef TestType eT;

  for (uword trial = 0; trial < 10; ++trial)
    {
    SpMat<eT> X;
    X.sprandn(256, 256, 0.3);

    uword Y1 =  accu(X.submat(10, 15, 173, 211) > 1);
    uword Y2 =  accu(1 > X.submat(10, 15, 173, 211));
    uword Y3 =  accu(X.submat(10, 15, 173, 211) >= 1);
    uword Y4 =  accu(1 >= X.submat(10, 15, 173, 211));
    uword Y5 =  accu(X.submat(10, 15, 173, 211) < 0);
    uword Y6 =  accu(0 < X.submat(10, 15, 173, 211));
    uword Y7 =  accu(X.submat(10, 15, 173, 211) <= -1);
    uword Y8 =  accu(-1 <= X.submat(10, 15, 173, 211));
    uword Y9 =  accu(X.submat(10, 15, 173, 211) == 3);
    uword Y10 = accu(3 == X.submat(10, 15, 173, 211));
    uword Y11 = accu(X.submat(10, 15, 173, 211) == 0);
    uword Y12 = accu(0 == X.submat(10, 15, 173, 211));
    uword Y13 = accu(X.submat(10, 15, 173, 211) != 2);
    uword Y14 = accu(2 != X.submat(10, 15, 173, 211));
    uword Y15 = accu(X.submat(10, 15, 173, 211) != 0);
    uword Y16 = accu(0 != X.submat(10, 15, 173, 211));

    Mat<eT> X_ref(X.submat(10, 15, 173, 211));

    uword Y1_ref =  accu(X_ref > 1);
    uword Y2_ref =  accu(1 > X_ref);
    uword Y3_ref =  accu(X_ref >= 1);
    uword Y4_ref =  accu(1 >= X_ref);
    uword Y5_ref =  accu(X_ref < 0);
    uword Y6_ref =  accu(0 < X_ref);
    uword Y7_ref =  accu(X_ref <= -1);
    uword Y8_ref =  accu(-1 <= X_ref);
    uword Y9_ref =  accu(X_ref == 3);
    uword Y10_ref = accu(3 == X_ref);
    uword Y11_ref = accu(X_ref == 0);
    uword Y12_ref = accu(0 == X_ref);
    uword Y13_ref = accu(X_ref != 2);
    uword Y14_ref = accu(2 != X_ref);
    uword Y15_ref = accu(X_ref != 0);
    uword Y16_ref = accu(0 != X_ref);

    REQUIRE( Y1  == Y1_ref  );
    REQUIRE( Y2  == Y2_ref  );
    REQUIRE( Y3  == Y3_ref  );
    REQUIRE( Y4  == Y4_ref  );
    REQUIRE( Y5  == Y5_ref  );
    REQUIRE( Y6  == Y6_ref  );
    REQUIRE( Y7  == Y7_ref  );
    REQUIRE( Y8  == Y8_ref  );
    REQUIRE( Y9  == Y9_ref  );
    REQUIRE( Y10 == Y10_ref );
    REQUIRE( Y11 == Y11_ref );
    REQUIRE( Y12 == Y12_ref );
    REQUIRE( Y13 == Y13_ref );
    REQUIRE( Y14 == Y14_ref );
    REQUIRE( Y15 == Y15_ref );
    REQUIRE( Y16 == Y16_ref );
    }
  }
