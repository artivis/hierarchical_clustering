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

#ifndef HIERARCHICAL_CLUSTERING_INITIALIZER_H
#define HIERARCHICAL_CLUSTERING_INITIALIZER_H

#include "hierarchical_clustering/accumulator.h"
#include "hierarchical_clustering/distances.h"

#include <map>

#include <omp.h>

#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/ptr_container/ptr_vector.hpp>

#include "hierarchical_clustering/test_utils.h"

namespace clustering
{
  namespace initializer
  {

    template <typename Feature>
    class Initializer
    {
    public:

      Initializer() : _k(0), _init(false) {}

      Initializer(uint k) : _k(k), _init(false) { _init = _k > 0 ? true : false; }

      /** Default destructor. */
      virtual ~Initializer() { }

      virtual bool operator()(const std::vector<Feature>& points, std::vector<Feature>& seeds) = 0;

      /** Initializes clustering seeds */
      virtual bool initialize(const std::vector<Feature>& points, std::vector<Feature>& seeds) = 0;

      /** Initializes clustering seeds */
      virtual bool initialize(const std::vector<Feature*>& points, std::vector<Feature>& seeds) = 0;

      void setClusterNumber(uint k)
      {
        _k = k;
        _init = (_k > 0) ? true : false;
      }

      virtual bool isInit() {return this->_init;}

    protected:

      /** Number of seeds */
      uint _k;

      /** init */
      bool _init;
    };

    /**
      * Implement the Forgy initialization for clustering algorithm.
      *
      */

    template <typename Feature>
    class ForgyInitializer : public Initializer<Feature>
    {
    public:

      ForgyInitializer() : Initializer<Feature>() {}

      ForgyInitializer(int k, bool repetition = false) :
        Initializer<Feature>(k),
        _repetition(repetition) {}

      /** Default destructor. */
      virtual ~ForgyInitializer() { }

      bool operator()(const std::vector<Feature>& points, std::vector<Feature>& seeds)
      {
        return initialize(points, seeds);
      }

      bool initialize(const std::vector<Feature>& points, std::vector<Feature>& seeds)
      {
        if (!this->_init)
        {
          std::cerr << "Number of clusters is not defined !" << std::endl;
          return false;
        }

        if (points.size() <= this->_k)
        {
          seeds = points;
          return true;
        }

        boost::mt19937 rng;
        boost::uniform_int<> distribution(0, points.size() - 1);
        boost::variate_generator<boost::mt19937&, boost::uniform_int<> > generator(rng, distribution);

        switch (_repetition)
        {
          case true:

            for (unsigned int i = 0; i < this->_k; ++i)
            {
              int idx = generator();
              seeds[i] = points[idx];
            }

            break;

          case false:

            std::vector<int> rands;
            unsigned int i;

            for (i = 0; i < this->_k; ++i)
            {
              int idx = generator();
              if ( std::find(rands.begin(), rands.end(), idx) != rands.end() )
              {
                --i;
                continue;
              }

              seeds[i] = points[idx];
              rands.push_back(idx);
            }

            break;
          }

          return true;
        }

      bool initialize(const std::vector<Feature*>& points, std::vector<Feature>& seeds)
      {
        if (!this->_init)
        {
          std::cerr << "Number of clusters is not defined !" << std::endl;
          return false;
        }

        if (points.size() <= this->_k)
        {
          for (int i=0; i<this->_k; ++i)
            seeds[i] = *(points[i]);
          return true;
        }

        boost::mt19937 rng;
        boost::uniform_int<> distribution(0, points.size() - 1);
        boost::variate_generator<boost::mt19937&, boost::uniform_int<> > generator(rng, distribution);

        switch (_repetition)
        {
          case true:

            for (unsigned int i = 0; i < this->_k; ++i)
            {
              int idx = generator();
              seeds[i] = *(points[idx]);
            }

            break;

          case false:

            std::vector<int> rands;
            unsigned int i;

            for (i = 0; i < this->_k; ++i)
            {
              int idx = generator();
              if ( std::find(rands.begin(), rands.end(), idx) != rands.end() )
              {
                --i;
                continue;
              }

              seeds[i] = *(points[idx]);
              rands.push_back(idx);
            }

            break;
          }

          return true;
        }

    private:

      bool _repetition;
    };

    /**
      *
      *
      */

    template <typename Feature>
    class RandomSeedsInitialization : public Initializer<Feature>
    {
    public:

      RandomSeedsInitialization() : Initializer<Feature>() {}

      RandomSeedsInitialization(int k) : Initializer<Feature>(k) {}

      /** Default destructor. */
      virtual ~RandomSeedsInitialization() { }

      bool operator()(const std::vector<Feature>& points, std::vector<Feature>& seeds)
      {
        return initialize(points, seeds);
      }

      // TODO So far seeds need to be allocated
      // TODO So far uses std::vector
      bool initialize(const std::vector<Feature>& points, std::vector<Feature>& seeds)
      {
        if (!this->_init)
        {
          std::cerr << "Number of clusters is not defined !" << std::endl;
          return false;
        }

        if (points.size() <= this->_k)
        {
          seeds = points;
          return true;
        }

        // Construct a random permutation of the features using a Fisher-Yates shuffle
        std::vector<const Feature*> features_perm(points.size());

        boost::mt19937 rng;
        boost::uniform_int<> distribution(0, RAND_MAX);
        boost::variate_generator<boost::mt19937&, boost::uniform_int<> > generator(rng, distribution);

        #pragma omp parallel for
        for (size_t i = 0; i < points.size(); ++i)
          features_perm[i] = &points[i];

        for (size_t i = points.size(); i > 1; --i)
          std::swap(features_perm[i-1], features_perm[generator() % i]);

        // Take the first k permuted features as the initial centers
        for (size_t i = 0; i < this->_k; ++i)
          seeds[i] = *features_perm[i];

        return true;
      }

      bool initialize(const std::vector<Feature*>& points, std::vector<Feature>& seeds)
      {
        if (!this->_init)
        {
          std::cerr << "Number of clusters is not defined !" << std::endl;
          return false;
        }

        if (points.size() <= this->_k)
        {
          for (int i=0; i<this->_k; ++i)
            seeds[i] = *(points[i]);
          return true;
        }

        // Construct a random permutation of the features using a Fisher-Yates shuffle
        std::vector<const Feature*> features_perm(points.size());

        boost::mt19937 rng;
        boost::uniform_int<> distribution(0, RAND_MAX);
        boost::variate_generator<boost::mt19937&, boost::uniform_int<> > generator(rng, distribution);

        #pragma omp parallel for
        for (size_t i = 0; i < points.size(); ++i)
          features_perm[i] = points[i];

        for (size_t i = points.size(); i > 1; --i)
          std::swap(features_perm[i-1], features_perm[generator() % i]);

        // Take the first k permuted features as the initial centers
        for (size_t i = 0; i < this->_k; ++i)
          seeds[i] = *features_perm[i];

        return true;
      }
    };

    /**
      *
      *
      */

    template <typename Feature, typename Accu = KahanAccumulator<double> >
    class RandomAssignmentInitialization : public Initializer<Feature>
    {
    private:

      typedef typename Accu::type acc_type;

    public:

      RandomAssignmentInitialization() : Initializer<Feature>() {}

      RandomAssignmentInitialization(int k) : Initializer<Feature>(k) {}

      /** Default destructor. */
      virtual ~RandomAssignmentInitialization() { }

      bool operator()(const std::vector<Feature>& points, std::vector<Feature>& seeds)
      {
        initialize(points, seeds);
      }

      bool initialize(const std::vector<Feature>& points, std::vector<Feature>& seeds)
      {
        if (!this->_init)
        {
          std::cerr << "Number of clusters is not defined !" << std::endl;
          return false;
        }

        if (points.size() <= this->_k)
        {
          seeds = points;
          return true;
        }

        boost::mt19937 rng;
        boost::uniform_int<> distribution(0, this->_k - 1);
        boost::variate_generator<boost::mt19937&, boost::uniform_int<> > generator(rng, distribution);

        std::vector<std::vector<const Feature*> > assignments(this->_k);

        // Randomly assign features to a cluster
        for (size_t i = 0; i < points.size(); ++i)
            assignments[generator()].push_back( &points[i] );

        int f_size = points[0].size();

        // Compute cluters centroids
        #pragma omp parallel for num_threads(this->_k)
        for (size_t i = 0; i < this->_k; ++i)
        {
          std::vector<Accu> accumulator_arr( f_size );

          for (size_t j = 0; j < assignments[i].size(); ++j)
            for (size_t l = 0; l < f_size; ++l)
              accumulator_arr[l] += acc_type( (*(assignments[i][j]))[l] ) / acc_type(assignments[i].size());

          for (size_t t = 0; t < f_size; ++t)
            seeds[i][t] = accumulator_arr[t];
        }

        return true;
      }

      bool initialize(const std::vector<Feature*>& points, std::vector<Feature>& seeds)
      {
        if (!this->_init)
        {
          std::cerr << "Number of clusters is not defined !" << std::endl;
          return false;
        }

        if (points.size() <= this->_k)
        {
          for (int i=0; i<this->_k; ++i)
            seeds[i] = *(points[i]);
          return true;
        }

        boost::mt19937 rng;
        boost::uniform_int<> distribution(0, this->_k - 1);
        boost::variate_generator<boost::mt19937&, boost::uniform_int<> > generator(rng, distribution);

        std::vector<std::vector<const Feature*> > assignments(this->_k);

        // Randomly assign features to a cluster
        for (size_t i = 0; i < points.size(); ++i)
            assignments[generator()].push_back( points[i] );

        int f_size = points[0]->size();

        // Compute cluters centroids
        #pragma omp parallel for num_threads(this->_k)
        for (size_t i = 0; i < this->_k; ++i)
        {
          std::vector<Accu> accumulator_arr( f_size );

          for (size_t j = 0; j < assignments[i].size(); ++j)
            for (size_t l = 0; l < f_size; ++l)
              accumulator_arr[l] += acc_type( (*(assignments[i][j]))[l] ) / acc_type(assignments[i].size());

          for (size_t t = 0; t < f_size; ++t)
            seeds[i][t] = accumulator_arr[t];
        }

        return true;
      }
    };

    /**
      * Implement the KMeans++ initialization for the K-Means clustering algorithm.
      *
      */

    template <typename Feature,
              typename Distance = distance::EuclideanDistance<Feature, KahanAccumulator<double> >,
              typename Accu = KahanAccumulator<double> >
    class PlusPlusKmeansInitialization : public Initializer<Feature>
    {
    private:

      typedef typename Accu::type acc_type;

    public:

      PlusPlusKmeansInitialization() : Initializer<Feature>() {}

      PlusPlusKmeansInitialization(int k) : Initializer<Feature>(k) {}

      /** Default destructor. */
      virtual ~PlusPlusKmeansInitialization() { }

      bool operator()(const std::vector<Feature>& points, std::vector<Feature>& seeds)
      {
        initialize(points, seeds);
      }

      bool initialize(const std::vector<Feature>& points, std::vector<Feature>& seeds)
      {
        if (!this->_init)
        {
          std::cerr << "Number of clusters is not defined !" << std::endl;
          return false;
        }

        if (points.size() <= this->_k)
        {
          seeds = points;
          return true;
        }

        boost::mt19937 rng;
        boost::uniform_real<double> distribution(0, 1);
        boost::uniform_int<> distribution2(0, seeds.size() - 1);

        boost::variate_generator<boost::mt19937&, boost::uniform_real<double> > generator(rng, distribution);
        boost::variate_generator<boost::mt19937&, boost::uniform_int<> > generator2(rng, distribution2);

        Accu maxCumulative;
        std::vector<double> cumulative(points.size());

        // Pick a first at random
        seeds[0] = points[ generator2() ];

        // Initialize the distances
//        #pragma omp parallel for ordered
        for(uint j = 0; j < points.size(); ++j)
        {
          double distance = _distance(seeds[0], points[j]);
//          cumulative[j] = maxCumulative + (distance * distance);
//          maxCumulative = cumulative[j];
          maxCumulative += (distance * distance);
          cumulative[j] = maxCumulative;
        }

        // Loop over the cluster seeds
        for(uint i = 1; i < seeds.size(); ++i)
        {
          // Sample the new seed
          acc_type r_ind = maxCumulative * generator();
          std::vector<double>::iterator it = std::lower_bound(cumulative.begin(), cumulative.end(), r_ind);
          uint index = std::distance(cumulative.begin(), it);
          seeds[i] = points[index];

          // Update the distances
          double distance = _distance(seeds[i], points[0]);
          distance = distance * distance;

          double correction = 0;

          if (distance < cumulative[0])
          {
            correction = cumulative[0] - distance;
            cumulative[0] = fabs(cumulative[0] - correction);
          }

          for(uint j = 1; j < points.size(); ++j)
          {
            cumulative[j] = fabs(cumulative[j] - correction);

            double distance = _distance(seeds[i], points[j]);

            distance = distance * distance;

            double oldDistance = cumulative[j] - cumulative[j-1];

            if (distance < oldDistance)
            {
              correction += oldDistance - distance;
              cumulative[j] = fabs(cumulative[j] - (oldDistance - distance));
            }
          }

          maxCumulative = cumulative.back();
        }

        return true;
      }

      bool initialize(const std::vector<Feature*>& points, std::vector<Feature>& seeds)
      {
        return true;

      }

    private:

      Distance _distance;
    };

    /**
      * Implement the Forgy initialization for clustering algorithm.
      * Warning : This scales very bad with the number of features.
      *
      */

    template <typename Feature,
              typename Distance = distance::EuclideanDistance<Feature, KahanAccumulator<double> >,
              typename Accu = KahanAccumulator<double> >
    class FastKMedoids : public Initializer<Feature>
    {
    private:

      typedef typename Accu::type acc_type;

      typedef typename std::vector<acc_type> distance_vector;
      typedef typename std::vector<distance_vector> distance_table;

      typedef typename std::vector<Accu> accu_vec;

    public:

      FastKMedoids() : Initializer<Feature>() {}

      FastKMedoids(int k) :
        Initializer<Feature>(k) {}

      /** Default destructor. */
      virtual ~FastKMedoids() { }

      bool operator()(const std::vector<Feature>& points, std::vector<Feature>& seeds)
      {
        return initialize(points, seeds);
      }

      bool initialize(const std::vector<Feature>& points, std::vector<Feature>& seeds)
      {
        if (!this->_init)
        {
          std::cerr << "Number of clusters is not defined !" << std::endl;
          return false;
        }

        uint p_size = points.size();

        if (p_size <= this->_k)
        {
          seeds = points;
          return true;
        }

        accu_vec dil_accu(p_size, Accu());
        accu_vec dij_accu(p_size, Accu());

        // double loop over all features (redundancy)
        // because some of the distance functions are
        // not symmetric (e.g Chi2Distance)

//        std::cout << "DERRE1" << std::endl;

//        for (uint i=0; i<p_size; ++i)
//          for (uint j=0; j<p_size; ++j)
//          {
//            double val = _distance(points[i], points[j]);
//            dil_accu[i] += val;
//            _dist_table[i][j] = val;
//          }

//        std::cout << "DERRE2" << std::endl;

//        for (uint j=0; j<p_size; ++j)
//          for (uint i=0; i<p_size; ++i)
//            dij_accu[j] += _dist_table[i][j] / dil_accu[i];

//        #pragma omp parallel for collapse(2)
//        for (uint i=0; i<p_size; ++i)
//          for (uint j=0; j<p_size; ++j)
//            dil_accu[i] += _distance(points[i], points[j]);

//        #pragma omp parallel for collapse(2)
//        for (uint j=0; j<p_size; ++j)
//          for (uint i=0; i<p_size; ++i)
//            dij_accu[j] += _distance(points[i], points[j]) / dil_accu[i];

//        uint i1, j2;
//        #pragma omp parallel shared(p_size, dil_accu, dij_accu)
//        {
//          #pragma omp for collapse(2)
//          for (i1=0; i1<p_size; ++i1)
//            for (uint j1=0; j1<p_size; ++j1)
//              dil_accu[i1] += _distance(points[i1], points[j1]);

//          #pragma omp for collapse(2)
//          for (j2=0; j2<p_size; ++j2)
//            for (uint i2=0; i2<p_size; ++i2)
//              dij_accu[j2] += _distance(points[i2], points[j2]) / dil_accu[i2];
//        }

//        uint j;
//        _dist_table.resize(p_size);

//        #pragma omp parallel shared(p_size, dil_accu, dij_accu, j)
//        {
//          #pragma omp for
//          for ( j=0; j<p_size; ++j)
//          {
//            for (uint i=0; i<p_size; ++i)
//            {
//              for (uint l=0; l<p_size; ++l)
//              {
//                double val = _distance(points[i], points[l]);
//                dil_accu[i] += val;
//                _dist_table[l] = val;
//              }

//              dij_accu[j] += _dist_table[i][j] / dil_accu[i];
//            }
//          }
//        }

        std::vector<uint> sorted_inx = sort_indexes(dij_accu);

        for (uint k=0; k<this->_k; ++k)
          seeds[k] = points[ sorted_inx[k] ];

        return true;
      }

      bool initialize(const std::vector<Feature*>& points, std::vector<Feature>& seeds)
      {
        if (!this->_init)
        {
          std::cerr << "Number of clusters is not defined !" << std::endl;
          return false;
        }

        if (points.size() <= this->_k)
        {
          for (int i=0; i<this->_k; ++i)
            seeds[i] = *(points[i]);
          return true;
        }

        return true;
      }

    private:

      Distance _distance;

      std::vector<double> _dist_table;
    };

  } // namespace initializer
} // namespace clustering

namespace
{
  // Workarround to compare
  // vectors values given their
  // indexes
  template<typename T>
  class Comp
  {
    std::vector<T>* _values;
  public:

    Comp(std::vector<T>* values) :
      _values(values) {}

    bool operator()(const int& a, const int& b) const
    {
      return (*_values)[a] < (*_values)[b];
    }
  };

  // Return a vector of indexes sorted
  // according to 'v' values
  template <typename T>
  std::vector<size_t> sort_indexes(std::vector<T> &v)
  {
    // initialize original index locations
    std::vector<size_t> idx(v.size());
    for (size_t i=0; i!=idx.size(); ++i)
      idx[i] = i;

    Comp<T> comp(&v);

    // sort indexes based on comparing values in v
    std::sort(idx.begin(), idx.end(), comp);

    return idx;
  }
}

#endif //HIERARCHICAL_CLUSTERING_INITIALIZER_H
