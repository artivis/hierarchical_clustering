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
#include <vector>

#include "hierarchical_clustering/clusteringInitializer.h"
#include "hierarchical_clustering/test_utils.h"

int main(int /*argc*/, char ** /*argv*/)
{
  uint feature_num = 250000;
  uint feature_size = 32;
  uint seed_num = 5;

  typedef int feature_type;
  typedef std::vector<feature_type> feature;
  typedef std::vector<feature> feature_vector;

  feature_vector feature_vec(feature_num, feature(feature_size, 0));

  feature_vector seed_vector_rands(seed_num, feature(feature_size, 0));
  feature_vector seed_vector_randa(seed_num, feature(feature_size, 0));
  feature_vector seed_vector_forgy(seed_num, feature(feature_size, 0));
  feature_vector seed_vector_plplk(seed_num, feature(feature_size, 0));
//  feature_vector seed_vector_fastM(seed_num, feature(feature_size, 0));

  utils::Rand<feature_type> random;

  for (uint i=0; i<feature_num; ++i)
    for (uint j=0; j<feature_size; ++j)
      feature_vec[i][j] = random(0., 255.);

  // Display features
//  std::cout << "features : \n" << feature_vec << std::endl;

  clustering::initializer::RandomSeedsInitialization<feature> rands_init(seed_num);

  clustering::initializer::RandomAssignmentInitialization<feature> randa_init(seed_num);

  clustering::initializer::ForgyInitializer<feature> forgy_init(seed_num);

  clustering::initializer::PlusPlusKmeansInitialization<feature> plplk_init(seed_num);

//  clustering::initializer::FastKMedoids<feature> fastM_init(seed_num);

  rands_init(feature_vec, seed_vector_rands);
  randa_init(feature_vec, seed_vector_randa);
  forgy_init(feature_vec, seed_vector_forgy);
  plplk_init(feature_vec, seed_vector_plplk);
//  fastM_init(feature_vec, seed_vector_fastM);

//  // Equivalent
//  //forgy_init.initialize(feature_vector, seed_vector);

  std::cout << "Random seed initialization : " << std::endl;
  for (unsigned int i=0; i<seed_vector_rands.size(); ++i)
    std::cout << "seed " << i << " : " << seed_vector_rands[i] << std::endl;
  std::cout << std::endl;

  std::cout << "Random assignment initialization : " << std::endl;
  for (unsigned int i=0; i<seed_vector_randa.size(); ++i)
    std::cout << "seed " << i << " : " << seed_vector_randa[i] << std::endl;
  std::cout << std::endl;

  std::cout << "Forgy initialization : " << std::endl;
  for (unsigned int i=0; i<seed_vector_forgy.size(); ++i)
    std::cout << "seed " << i << " : " << seed_vector_forgy[i] << std::endl;
  std::cout << std::endl;

  std::cout << "PlusPlusKmeans initialization : " << std::endl;
  for (unsigned int i=0; i<seed_vector_plplk.size(); ++i)
    std::cout << "seed " << i << " : " << seed_vector_plplk[i] << std::endl;
  std::cout << std::endl;

//  std::cout << "FastKMedoids initialization : " << std::endl;
//  for (unsigned int i=0; i<seed_vector_fastM.size(); ++i)
//    std::cout << "seed " << i << " : " << seed_vector_fastM[i] << std::endl;

  return 0;
}
