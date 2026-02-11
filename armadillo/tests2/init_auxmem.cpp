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


TEMPLATE_TEST_CASE("init_auxmem_1", "[init]", TEST_FLOAT_TYPES)
  {
  typedef TestType eT;

  eT data[] = { 1, 2, 3, 4, 5, 6 };
  
  Mat<eT> A(data, 2, 3);
  Mat<eT> B(data, 2, 3, false);
  Mat<eT> C(data, 2, 3, false, true);
  
  REQUIRE( A(0,0) == eT(1) );
  REQUIRE( A(1,0) == eT(2) );
  
  REQUIRE( A(0,1) == eT(3) );
  REQUIRE( A(1,1) == eT(4) );
  
  REQUIRE( A(0,2) == eT(5) );
  REQUIRE( A(1,2) == eT(6) );
  
  A(0,0) = eT(123.0);  REQUIRE( data[0] == eT(1) );
  
  B(0,0) = eT(123.0);  REQUIRE( data[0] == eT(123.0) );
  
  REQUIRE_THROWS( C.set_size(5,6) );
  }



