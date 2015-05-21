///** Copyright (c) 2014, Pal Robotics
//All rights reserved.

//Redistribution and use in source and binary forms, with or without
//modification, are permitted provided that the following conditions are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of Pal Robotics nor the
//    names of its contributors may be used to endorse or promote products
//    derived from this software without specific prior written permission.

//THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
//ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//DISCLAIMED. IN NO EVENT SHALL Pal Robotics BE LIABLE FOR ANY
//DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
//LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
//ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

//@author Jeremie Deray

//*/

#include <iostream>
#include <time.h>
#include <stdlib.h>

#include "hierarchical_clustering/hierarchicalClustering.h"
#include "hierarchical_clustering/test_utils.h"

int main(int /*argc*/, char ** /*argv*/)
{
  uint feature_num  = 250000 * 6;
  uint feature_size = 128;
  uint cluster_num  = 10;
  uint levels = 6;

  typedef float feature_type;
  typedef std::vector<feature_type> feature;
  typedef std::vector<feature> feature_vector;

  // Memory must be pre-allocated
  feature_vector feature_vec(feature_num, feature(feature_size, 0.));
  feature_vector cluster_vec;

  utils::Rand<feature_type> random;

  for (int i=0; i<feature_num; ++i)
    for (int j=0; j<feature_size; ++j)
      feature_vec[i][j] = random(0., 255.);

  std::cout << "\n" << feature_vec.size() << " random features initialized.\n\n";
  std::cout << "Expect approximately : " << pow(cluster_num, levels) << " clusters.\n\n";

  // Display features
  //std::cout << "feature : " << "\n" << feature_vec << std::endl << std::endl;

  // Instantiate with default constructor
  // Requiers to set the number of clusters and levels
  clustering::HierarchicalClustering<feature> h_clustering;

  h_clustering.setClustersNumbers(cluster_num);

  h_clustering.setlevel(levels);

  h_clustering.hierarchicalClusterPoints(feature_vec, cluster_vec);

  std::cout << cluster_vec.size() << " cluster centroids computed." << std::endl;

  // Display centroid
  //std::cout << cluster_vec << std::endl;

  /**
    * A sligthly more custom hierarchical kMeans
    *
      typedef distance::ManhattanDistance<feature> dist_func;
      typedef clustering::initializer::RandomAssignmentInitialization<feature> rand_assign_init;

      feature_vector cluster_vec_s;

      clustering::HierarchicalClustering<feature, dist_func, rand_assign_init> custom_h_Kmeans( 10 );

      custom_h_Kmeans( feature_vec, cluster_vec_s );

      std::cout << "\n" << cluster_vec_s.size() << " cluster centroid computed." << std::endl;

    */

  return 0;
}
