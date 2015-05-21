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
#include <omp.h>

#include "hierarchical_clustering/kMeansClustering.h"
#include "hierarchical_clustering/feature_allocator.h"
#include "hierarchical_clustering/test_utils.h"

int main(int /*argc*/, char ** /*argv*/)
{
  uint feature_size = 128;
  uint cluster_num  = 16;

  typedef int feature_type;
  typedef std::vector<feature_type> feature;
  typedef std::vector<feature> feature_vector;

  feature_vector data_set;
  feature_vector ground_truth_centroids;
  std::vector<uint> ground_truth_assignments;

  std::vector<uint> feat_assignments;

  if ( utils::loadDataSet<feature>(data_set, ground_truth_centroids, ground_truth_assignments, feature_size) )
    std::cout << "Dataset loaded.\n\n";
  else
  {
    std::cout << "Error while loading dataset !" << std::endl;
    return -1;
  }

  // Memory must be pre-allocated
  feature_vector cluster_vec(cluster_num, feature(feature_size, 0.));;

  // Display features
  //std::cout << "feature : " << "\n" << data_set << std::endl << std::endl;

  typedef distance::ManhattanDistance<feature> man_dist_func;
  man_dist_func dist_func;

  // Instantiate with default constructor
  // Requiers to set the number of clusters,
  // The cluster initializer & the distance
  clustering::KMeans<feature, man_dist_func> kmeans;

  kmeans.setClustersNumbers( cluster_num );

  if ( !kmeans.initializeSeeds(data_set, cluster_vec) )
  {
    std::cerr << "\nCouldn't initialize seeds !" << std::endl;
    return -1;
  }
  else
    std::cout << cluster_vec.size() << " seeds initialized." << std::endl;

  // Display inital seeds
  //std::cout << cluster_vec << std::endl;

  /**
    * Seed initialization is done by default, 'false' indicates
    * that seeds are already init.
    */
  kmeans.clusterPoints(data_set, cluster_vec, feat_assignments, false);

  std::cout << cluster_vec.size() << " cluster centroid computed." << std::endl;

  // Display centroid
  //std::cout << cluster_vec << std::endl;

  std::vector<uint> centroids_assign;
  utils::clustering::computeAssignments(ground_truth_centroids, cluster_vec, centroids_assign, dist_func);

  // Display centroids pairwise (groundtruth / computed)
//  for (size_t i=0; i<cluster_vec.size(); ++i)
//  {
//    std::cout << "gt " << i << " : " << ground_truth_centroids[ centroids_assign[i] ] << std::endl;
//    std::cout << "cp " << i << " : " << cluster_vec[i] << std::endl << std::endl;
//  }

  std::vector<double> distances;
  double error_sum;

  error_sum = utils::clustering::compareCentroids(ground_truth_centroids, cluster_vec,
                                                  centroids_assign, distances, dist_func);

  // Display centroid pairs distances
  //std::cout << "distances : \n" << distances << std::endl;

  // Display error sum
  std::cout << "\n" << "Sum error distance(groundtruth, computed) : " << error_sum << std::endl;

  std::vector<bool> correct_assign;
  double assign_correctness;

  assign_correctness = utils::compareAssignments(ground_truth_assignments, feat_assignments,
                                                 centroids_assign, correct_assign);

  std::cout << "Assignment correctness : " << assign_correctness << " %" << std::endl;


  /**
    * A sligthly more custom kMeans
    *

      typedef distance::ManhattanDistance<feature> dist_func;
      typedef clustering::initializer::RandomAssignmentInitialization<feature> rand_assign_init;

      feature_vector cluster_vec_s(10, feature(feature_size, 0));

      clustering::KMeans<feature, dist_func, rand_assign_init> custom_kmeans( 10 );

      custom_kmeans( feature_vec, cluster_vec_s );

      std::cout << "\n" << cluster_vec_s.size() << " cluster centroid computed." << std::endl;

    */

  return 0;
}
