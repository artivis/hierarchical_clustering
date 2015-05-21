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

#ifndef HIERARCHICAL_CLUSTERING_TEST_UTILS_H
#define HIERARCHICAL_CLUSTERING_TEST_UTILS_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>
#include <stdlib.h>
#include <vector>
#include <limits>

#include "hierarchical_clustering/distances.h"

namespace utils
{
  const double MAX_DB = std::numeric_limits<double>::max();

  template <class T>
  class Rand
  {
  public:

    Rand() { srand( time(0) ); }
    ~Rand() {}

    double operator()(T fMin, T fMax)
    {
      return get(fMin, fMax);
    }

    double get(T fMin, T fMax)
    {
      double f = (double)rand() / (double)RAND_MAX;
      return T(fMin + f * (fMax - fMin));
    }
  };

  class Timer
  {
  public:

    Timer(std::string name = std::string("")) :
      _duration(0),
      _name(name) { tic(); }

    void tic() {_start = clock();}

    void toc(bool reset = false)
    {
      _duration = ( clock() - _start ) / (double) CLOCKS_PER_SEC;

      std::cout << _name << " " << "duration : " << _duration << " secs." << std::endl;

      if (reset)
        _start = clock();
    }

  private:

    double  _duration;
    clock_t _start;

    std::string _name;
  };

  template <typename feature>
  bool loadDataSetFeatures(const std::string& file, std::vector<feature>& features, uint size = 32)
  {
    std::ifstream ifs(file.c_str());

    if (ifs.fail())
    {
      std::cerr << "Failed to open scans file " << file << " !";
      return false;
    }

    while (ifs.good())
    {
      std::string line;
      std::getline(ifs, line);

      if(line.empty())
        break;

      std::istringstream inputLine(line);

      feature feat(size);

      for (uint i=0; i<size; ++i)
        inputLine >> feat[i];

      features.push_back(feat);
    }

    std::cout << "Parsed " << features.size() << " features from file!" << std::endl;
    return true;
  }

  bool loadDataSetAssignment(const std::string& file, std::vector<uint>& assign, uint size = 32)
  {
    std::ifstream ifs(file.c_str());

    if (ifs.fail())
    {
      std::cerr << "Failed to open scans file " << file << " !";
      return false;
    }

    std::string line;

    for (int off=0; off<5; ++off)
      std::getline(ifs, line);

    while (ifs.good())
    {
      std::getline(ifs, line);

      if(line.empty())
        break;

      std::istringstream inputLine(line);

      uint assignement;

      inputLine >> assignement;
      assign.push_back(assignement);
    }

    std::cout << "Parsed " << assign.size() << " assignment from file!" << std::endl;
    return true;
  }

  template <typename Feature>
  bool loadDataSet(std::vector<Feature>& features, std::vector<Feature>& centroids,
                   std::vector<uint>& assign, uint size = 32)
  {
    std::string scans_file;
    std::string assign_file;
    std::string gt_file;

    switch (size)
    {
      case 32:
        scans_file  = ("../data/dim032.txt");
        assign_file = ("../data/dim032.pa");
        gt_file     = ("../data/gtdim032.txt");
        break;

      case 64:
        scans_file  = ("../data/dim064.txt");
        assign_file = ("../data/dim064.pa");
        gt_file     = ("../data/gtdim064.txt");
        break;

      case 128:
        scans_file  = ("../data/dim128.txt");
        assign_file = ("../data/dim128.pa");
        gt_file     = ("../data/gtdim128.txt");
        break;

      default:
        std::cerr << "Wrong feature size !\nChoices are 32 / 64 / 128 !" << std::endl;
        return false;
    }

    bool loaded_scan = loadDataSetFeatures(scans_file, features, size);

    bool loaded_gt = loadDataSetFeatures(gt_file, centroids, size);

    bool loaded_assign = loadDataSetAssignment(assign_file, assign, size);

    return loaded_scan && loaded_assign && loaded_gt;
  }



  namespace clustering
  {
    double compareAssignments(const std::vector<uint>& assign_gt, const std::vector<uint>& assign_comp,
                              const std::vector<uint>& mapping, std::vector<bool>& correct)
    {
      if (assign_gt.size() != assign_comp.size() &&
          assign_gt.size() != mapping.size())
      {
        std::cerr << "Assignments size are not equal !" << std::endl;
        return assign_gt.size();
      }

      correct.clear();
      correct.resize(assign_gt.size(), false);

      double correctness = 0.;

      for (size_t i=0; i<assign_gt.size(); ++i)
      {
        uint assign = assign_comp[i];

        if (assign_gt[i] == (mapping[assign]+1))
        {
          correct[i] = true;
          correctness += 1.;
        }
      }

      return (correctness / assign_gt.size()) * 100.;
    }

    template <typename Feature>
    double computeAssignments(const std::vector<Feature>& seeds,
                              const std::vector<Feature>& points,
                              std::vector<uint>& assignment,
                              distance::HistogramDistance<Feature>& distance)
    {
      double error = 0;

      assignment.clear();
      assignment.resize( points.size() );

      // Assign each point to a cluster
      #pragma omp parallel for reduction(+:error)
      for(uint p = 0; p < points.size(); ++p)
      {
        uint bestCluster = 0;
        double maxDist = MAX_DB;

        for(uint c = 0; c < seeds.size(); ++c)
        {
          double dist = distance.distance(seeds[c], points[p]);

          if(dist < maxDist)
          {
            maxDist = dist;
            bestCluster = c;
          }
        }

        error += maxDist;

        assignment[p] = bestCluster;
      }

      return error;
    }

    template <typename Feature>
    double compareCentroids(const std::vector<Feature>& ground_truth,
                            const std::vector<Feature>& computed,
                            std::vector<double>& distances,
                            distance::HistogramDistance<Feature>& distance)
    {
      if (ground_truth.size() != computed.size())
      {
        std::cerr << "Number of centroids unequal !" << std::endl;
        return ground_truth.size();
      }

      std::vector<uint> assign;

      // compute assignment of 'computed' to 'ground_truth'
      computeAssignments(ground_truth, computed, assign, distance);

      distances.clear();
      distances.resize( ground_truth.size() );

      double error = 0;

      for (size_t i=0; i<ground_truth.size(); ++i)
      {
        distances[i] = distance.distance(ground_truth[ assign[i] ], computed[i]);
        error += distances[i];
      }

      return error;
    }

    template <typename Feature>
    double compareCentroids(const std::vector<Feature>& ground_truth,
                            const std::vector<Feature>& computed,
                            const std::vector<uint>& assign,
                            std::vector<double>& distances,
                            distance::HistogramDistance<Feature>& distance)
    {
      if (ground_truth.size() != computed.size())
      {
        std::cerr << "Number of centroids unequal !" << std::endl;
        return ground_truth.size();
      }

      distances.clear();
      distances.resize( ground_truth.size() );

      double error = 0;

      for (size_t i=0; i<ground_truth.size(); ++i)
      {
        distances[i] = distance.distance(ground_truth[ assign[i] ], computed[i]);
        error += distances[i];
      }

      return error;
    }

  } //namespace clustering

} //namespace utils

template <class T>
inline std::ostream& operator<< (std::ostream& os, const std::vector<T>& v)
{
  os << "[";

  for (typename std::vector<T>::const_iterator cit = v.begin(); cit != v.end(); ++cit)
    os << " " << *cit;

  os << " ]";
  return os;
}

template <class T>
inline std::ostream& operator<< (std::ostream& os, const std::vector<std::vector<T> >& v)
{
  for (size_t i = 0; i < v.size(); ++i)
    os << " " << v[i] << std::endl;

  return os;
}

#endif
