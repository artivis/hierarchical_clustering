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

#include "hierarchical_clustering/distances.h"
#include "hierarchical_clustering/test_utils.h"

int main(int /*argc*/, char ** /*argv*/)
{
  unsigned int feature_size = 128;

  typedef float feature_type;
  typedef std::vector<feature_type> feature;

  feature feature_a(feature_size);
  feature feature_b(feature_size);
  feature feature_c(feature_size);

  utils::Rand<feature_type> random;

  for (size_t i=0; i<feature_a.size(); ++i)
  {
    feature_a[i] = (i + 1.4);
    feature_c[i] = random(0+1.4, feature_a.size()+1.4);
  }

  feature_b = feature_a;

  std::cout << "feature a : " << feature_a << std::endl;
  std::cout << "feature b : " << feature_b << std::endl;
  std::cout << "feature c : " << feature_c << std::endl;
  std::cout << std::endl;

  // Declare a distance
  distance::EuclideanDistance<feature> EuclideanDistance;

  // Compute distance calling explicitly distance
  double L2_dist_ab = EuclideanDistance.distance(feature_a, feature_b);

  // Compute distance
  double L2_dist_ac = EuclideanDistance(feature_a, feature_c);

  std::cout << "EuclideanDistance : " << std::endl;
  std::cout << "   feature_a - feature_b : " << L2_dist_ab << std::endl;
  std::cout << "   feature_a - feature_c : " << L2_dist_ac << std::endl;
  std::cout << std::endl;

  distance::ManhattanDistance<feature> ManhattanDistance;

  double L1_dist_ab = ManhattanDistance.distance(feature_a, feature_b);
  double L1_dist_ac = ManhattanDistance(feature_a, feature_c);

  std::cout << "ManhattanDistance : " << std::endl;
  std::cout << "   feature_a - feature_b : " << L1_dist_ab << std::endl;
  std::cout << "   feature_a - feature_c : " << L1_dist_ac << std::endl;
  std::cout << std::endl;

  distance::Chi2Distance<feature> Chi2Distance;

  double chi_dist_ab = Chi2Distance.distance(feature_a, feature_b);
  double chi_dist_ac = Chi2Distance(feature_a, feature_c);

  std::cout << "Chi2Distance : " << std::endl;
  std::cout << "   feature_a - feature_b : " << chi_dist_ab << std::endl;
  std::cout << "   feature_a - feature_c : " << chi_dist_ac << std::endl;
  std::cout << std::endl;

  distance::SymmetricChi2Distance<feature> SymmetricChi2Distance;

  double symchi_dist_ab = SymmetricChi2Distance.distance(feature_a, feature_b);
  double symchi_dist_ac = SymmetricChi2Distance(feature_a, feature_c);

  std::cout << "SymmetricChi2Distance : " << std::endl;
  std::cout << "   feature_a - feature_b : " << symchi_dist_ab << std::endl;
  std::cout << "   feature_a - feature_c : " << symchi_dist_ac << std::endl;
  std::cout << std::endl;

  distance::BhattacharyyaDistance<feature> BhattacharyyaDistance;

  double batt_dist_ab = BhattacharyyaDistance.distance(feature_a, feature_b);
  double batt_dist_ac = BhattacharyyaDistance(feature_a, feature_c);

  std::cout << "BhattacharyyaDistance : " << std::endl;
  std::cout << "   feature_a - feature_b : " << batt_dist_ab << std::endl;
  std::cout << "   feature_a - feature_c : " << batt_dist_ac << std::endl;
  std::cout << std::endl;

  // Using an accumulator of integer
  distance::BhattacharyyaDistance<feature, Accumulator<int> > BhattacharyyaDistanceAccu;

  double batt_acc_dist_ab = BhattacharyyaDistanceAccu.distance(feature_a, feature_b);
  double batt_acc_dist_ac = BhattacharyyaDistanceAccu(feature_a, feature_c);

  std::cout << "BhattacharyyaDistance accumulator of int : " << std::endl;
  std::cout << "   feature_a - feature_b : " << batt_acc_dist_ab << std::endl;
  std::cout << "   feature_a - feature_c : " << batt_acc_dist_ac << std::endl;
  std::cout << std::endl;

  distance::JensenShannonDistance<feature> JensenShannonDistance;

  double js_dist_ab = JensenShannonDistance.distance(feature_a, feature_b);
  double js_dist_ac = JensenShannonDistance(feature_a, feature_c);

  std::cout << "JensenShannonDistance : " << std::endl;
  std::cout << "   feature_a - feature_b : " << js_dist_ab << std::endl;
  std::cout << "   feature_a - feature_c : " << js_dist_ac << std::endl;
  std::cout << std::endl;

  distance::KullbackLeiblerDistance<feature> KullbackLeiblerDistance;

  double kull_dist_ab = KullbackLeiblerDistance.distance(feature_a, feature_b);
  double kull_dist_ac = KullbackLeiblerDistance(feature_a, feature_c);

  std::cout << "KullbackLeiblerDistance : " << std::endl;
  std::cout << "   feature_a - feature_b : " << kull_dist_ab << std::endl;
  std::cout << "   feature_a - feature_c : " << kull_dist_ac << std::endl;

  return 0;
}
