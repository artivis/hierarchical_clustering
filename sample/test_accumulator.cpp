/** Copyright (c) 2014, Pal Robotics
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
  * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
  * Neither the name of Pal Robotics nor the
    names of its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL Pal Robotics BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

@author Jeremie Deray

*/

#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <limits.h>

#include "hierarchical_clustering/Accumulator.h"
#include "hierarchical_clustering/test_utils.h"

int main(int /*argc*/, char ** /*argv*/)
{
  int max_iteration = 100000;

  typedef float feature_type;

  Accumulator<feature_type> simple_accu_add;
  KahanAccumulator<feature_type> kahan_accu_add;

  KahanAccumulator<feature_type> kahan_accu_sum;
  feature_type sum, sum_up;

  sum = 0.;

  for (feature_type i=0; i<150; ++i)
  {
    sum += i;
    simple_accu_add += i;
    kahan_accu_add += i;
  }

  kahan_accu_sum = kahan_accu_add + simple_accu_add;
  sum_up = kahan_accu_add + simple_accu_add;

  std::cout.precision(15);
  std::cout << "Simple accumulator summed up : " << simple_accu_add << std::endl;
  std::cout << "Kahan  accumulator summed up : " << kahan_accu_add << std::endl;
  std::cout << "Sum                summed up :  [ " << sum << " ]" << std::endl;
  std::cout << std::endl;

  std::cout << "Kahan accumulator summed up both previous accumulator : " << kahan_accu_sum << std::endl;
  std::cout << "Simple sum : " << sum_up << std::endl;
  std::cout << std::endl;

  Accumulator<feature_type> simple_accu;
  KahanAccumulator<feature_type> kahan_accu;

  utils::Rand<feature_type> random;

  for (feature_type i=0; i<max_iteration; ++i)
  {
    feature_type value = random(-max_iteration, +max_iteration);

    simple_accu += value;
    kahan_accu += value;
  }

  std::cout.precision(15);
  std::cout << "After summing " << max_iteration << " random numbers in range [ "
            << -max_iteration << " / " << +max_iteration << " ] and of size "
            << sizeof(feature_type) * CHAR_BIT << " bits.\n\n";

  std::cout << "Simple accumulator summed up : " << simple_accu << std::endl;
  std::cout << "Kahan  accumulator summed up : " << kahan_accu  << std::endl;

  return 0;
}
