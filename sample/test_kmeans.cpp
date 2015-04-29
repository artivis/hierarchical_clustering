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

#include "hierarchical_clustering/kMeansClustering.h"
#include "hierarchical_clustering/feature_allocator.h"
#include "hierarchical_clustering/test_utils.h"

int main(int /*argc*/, char ** /*argv*/)
{
  uint feature_num  = 250000*5;
  uint feature_size = 128;
  uint cluster_num  = 10;

  typedef float feature_type;
  typedef std::vector<feature_type> feature;
  typedef std::vector<feature> feature_vector;

  feature_vector feature_vec(feature_num,   feature(feature_size, 0));
  feature_vector cluster_vec(cluster_num,   feature(feature_size, 0));
  feature_vector cluster_vec_s(cluster_num, feature(feature_size, 0));

  utils::Rand<feature_type> random;

  for (int j=0; j<feature_size; ++j)
    for (int i=0; i<feature_num; ++i)
      feature_vec[i][j] = random(0., 255.);

//  std::cout << "feature : " << "\n" << feature_vec << std::endl;
//  std::cout << std::endl;

  clustering::KMeans<feature> kmeans(cluster_num);
//  clustering::KMeans<feature, Accumulator<feature_type> > kmeansimple(cluster_num);

  kmeans.initializeSeeds(feature_vec, cluster_vec);

  // Seed initialization is done by default, 'false' indicates
  // that seeds are already init.
  kmeans.clusterPoints(feature_vec, cluster_vec, false);

  // Seeds initialization and points clustering
//  kmeansimple(feature_vec, cluster_vec_s);

//  std::cout << "cluster : " << "\n" << cluster_vec << std::endl;
  std::cout << std::endl;

//  std::cout << "cluster simple : " << "\n" << cluster_vec_s << std::endl;
  std::cout << std::endl;

  return 0;
}
