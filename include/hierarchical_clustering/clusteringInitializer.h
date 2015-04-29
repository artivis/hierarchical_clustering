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

#include "hierarchical_clustering/Accumulator.h"
#include "hierarchical_clustering/distances.h"

#include <map>

#include <omp.h>

#include <boost/random/uniform_int.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/ptr_container/ptr_vector.hpp>

namespace clustering
{
  namespace initializer
  {

    template <typename Feature>
    class Initializer
    {
    public:

      Initializer() : _k(0), _init(false) {}

      Initializer(uint k) : _k(k), _init(true) {}

      /** Default destructor. */
      virtual ~Initializer() { }

      virtual void operator()(const std::vector<Feature>& points, std::vector<Feature>& seeds) = 0;

      /** Initializes clustering seeds */
      virtual void initialize(const std::vector<Feature>& points, std::vector<Feature>& seeds) = 0;

      void setClusterNumber(uint k) { _k = k; }

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

      void operator()(const std::vector<Feature>& points, std::vector<Feature>& seeds)
      {
        initialize(points, seeds);
      }

      void initialize(const std::vector<Feature>& points, std::vector<Feature>& seeds)
      {
        if (!this->_init)
        {
          std::cerr << "Number of clusters is not defined !" << std::endl;
          return;
        }

        if (points.size() <= this->_k)
        {
          seeds = points;
          return;
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

      void operator()(const std::vector<Feature>& points, std::vector<Feature>& seeds)
      {
        initialize(points, seeds);
      }

      // TODO So far seeds need to be allocated
      // TODO So far uses std::vector
      void initialize(const std::vector<Feature>& points, std::vector<Feature>& seeds)
      {
        if (!this->_init)
        {
          std::cerr << "Number of clusters is not defined !" << std::endl;
          return;
        }

        if (points.size() <= this->_k)
        {
          seeds = points;
          return;
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

      void operator()(const std::vector<Feature>& points, std::vector<Feature>& seeds)
      {
        initialize(points, seeds);
      }

      void initialize(const std::vector<Feature>& points, std::vector<Feature>& seeds)
      {
        if (!this->_init)
        {
          std::cerr << "Number of clusters is not defined !" << std::endl;
          return;
        }

        if (points.size() <= this->_k)
        {
          seeds = points;
          return;
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
      }
    };

    /**
      * Implement the KMeans++ initialization for the K-Means clustering algorithm.
      *
      */

  //  template <typename Feature>
  //  class PlusPlusKmeansInitialization : public Initializer
  //  {
  //  public:

  //    /** Default destructor. */
  //    virtual ~PlusPlusKmeansInitialization() { }

  //    void initialize(const Feature& points, Feature& seeds)
  //    {
  //      if (points.size() <= _k)
  //        return seeds = points;

  //    //	std::mt19937 rng;
  //    //	std::uniform_real_distribution<double> distribution(0, 1);
  //    //	std::uniform_int_distribution<int> distribution2(0, seeds.size() - 1);
  //    //	auto generator2 = std::bind(distribution2, rng);
  //    //// 	std::variate_generator<std::mt19937&, std::uniform_int<int> > generator2(rng, distribution2);
  //    //	auto generator = std::bind(distribution, rng);
  //    //// 	std::variate_generator<std::mt19937&, std::uniform_real<double> > generator(rng, distribution);
  //    //	double maxCumulative = 0;
  //    //	std::vector<double> cumulative(points.size());

  //    //	// Pick a first at random
  //    //	seeds[0] = points[generator2()];
  //    //	// Initialize the distances
  //    //	for(unsigned int j = 0; j < points.size(); j++) {
  //    //		double distance = 1. - seeds[0].sim(&points[j]);
  //    //		cumulative[j] = maxCumulative + distance * distance;
  //    //		maxCumulative = cumulative[j];
  //    //	}

  //    //	// Loop over the cluster seeds
  //    //	for(unsigned int i = 1; i < seeds.size(); i++) {
  //    //		// Sample the new seed
  //    //		std::vector<double>::iterator it = std::lower_bound(cumulative.begin(), cumulative.end(), generator() * maxCumulative);
  //    //		unsigned int index = std::distance(cumulative.begin(), it);
  //    //		seeds[i] = points[index];

  //    //		// Update the distances
  //    //		double distance = 1. - seeds[i].sim(&points[0]);
  //    //		distance = distance * distance;
  //    //		double correction = 0;
  //    //		if (distance < cumulative[0]) {
  //    //			correction = cumulative[0] - distance;
  //    //			cumulative[0] = fabs(cumulative[0] - correction);
  //    //		}
  //    //		for(unsigned int j = 1; j < points.size(); j++) {
  //    //			cumulative[j] = fabs(cumulative[j] - correction);
  //    //			double distance = 1. - seeds[i].sim(&points[j]);
  //    //			distance = distance * distance;
  //    //			double oldDistance = cumulative[j] - cumulative[j-1];
  //    //			if (distance < oldDistance) {
  //    //				correction += oldDistance - distance;
  //    //				cumulative[j] = fabs(cumulative[j] - (oldDistance - distance));
  //    //			}
  //    //		}
  //    //		maxCumulative = cumulative.back();
  //    //	}

  //    // 	std::binary_search<>();
  //    }
  //  };

  } // namespace initializer
} // namespace clustering

#endif //HIERARCHICAL_CLUSTERING_INITIALIZER_H
