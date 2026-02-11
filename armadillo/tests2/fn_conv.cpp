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


TEST_CASE("fn_conv_1", "[conv]")
  {
  vec a =   linspace<vec>(1,5,6);
  vec b = 2*linspace<vec>(1,6,7);
  
  vec c = conv(a,b);
  vec d =
    {
      2.00000000000000,
      7.26666666666667,
     17.13333333333333,
     32.93333333333334,
     56.00000000000000,
     87.66666666666667,
    117.66666666666666,
    134.00000000000003,
    137.73333333333335,
    127.53333333333336,
    102.06666666666668,
     60.00000000000000
     };
  
  REQUIRE( accu(abs(c - d)) == Approx(0.0).margin(0.001) );
  }



TEMPLATE_TEST_CASE("fn_conv_fp_randu", "[conv]", TEST_FLOAT_TYPES)
  {
  typedef TestType eT;

  // generalized version of the test above with higher tolerances
  Col<eT> a =   linspace<Col<eT>>(1,5,6);
  Col<eT> b = 2*linspace<Col<eT>>(1,6,7);

  Col<eT> c = conv(a,b);
  Col<eT> d =
    {
    eT(  2.00),
    eT(  7.27),
    eT( 17.13),
    eT( 32.93),
    eT( 56.00),
    eT( 87.67),
    eT(117.67),
    eT(134.00),
    eT(137.73),
    eT(127.53),
    eT(102.07),
    eT( 60.00)
    };

  for (uword i = 0; i < c.n_elem; ++i)
    {
    REQUIRE( eT(c[i]) == Approx(eT(d[i])).epsilon(eT(0.02)) );
    }
  }
