/* *
 * GFLIP - Geometrical FLIRT Phrases for Large Scale Place Recognition
 * Copyright (C) 2012-2013 Gian Diego Tipaldi and Luciano Spinello and Wolfram
 * Burgard
 *
 * This file is part of GFLIP.
 *
 * GFLIP is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * GFLIP is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with GFLIP.  If not, see <http://www.gnu.org/licenses/>.
 */

////// TODO LICENCE

/**
 *
 * hierarchical_clustering is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with hierarchical_clustering. If not, see <http://www.gnu.org/licenses/>.
 *
 * @author Jeremie Deray
 *
*/

#ifndef HIERARCHICAL_CLUSTERING_KMEANSCLUSTRING_H
#define HIERARCHICAL_CLUSTERING_KMEANSCLUSTRING_H

#include <vector>

#include <omp.h>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include <hierarchical_clustering/distances.h>
#include <hierarchical_clustering/feature_allocator.h>
#include <hierarchical_clustering/clusteringInitializer.h>

namespace clustering
{
  template <typename Feature, typename Accu = KahanAccumulator<double> /*, typename Alloc = allocator::DefaultAllocator<Feature>*/ >
  class KMeans
  {
    typedef typename Feature::value_type value_type;
    typedef typename Accu::type acc_type;

  public:

    /**
      * Default constructor. It set the maximum iterations for the clustering and the minimum error difference for convergence.
      *
      */
    KMeans(uint clusters, uint maxIterations = 100, double minError = 0.001);

    /**
      * Initializes the centroids @param seeds from the @param points.
      * The @param seeds are modified to hold the clusters.
      *
      */
    void initializeSeeds(const std::vector<Feature>& points, std::vector<Feature>& seeds);

    /**
      * Cluster the @param points into clusters. It initialize the centroids with the @param seeds.
      * The @param seeds are modified to hold the clusters.
      *
      */
    double clusterPoints(const std::vector<Feature>& points, std::vector<Feature>& seeds, bool init_seeds = true);

    /**
      * Cluster the @param points into clusters. It initialize the centroids with the @param seeds.
      * The @param seeds are modified to hold the clusters.
      * This overloaded version returns also the point assignments.
      *
      */
    double clusterPoints(const std::vector<Feature>& points, std::vector<Feature>& seeds,
                         std::vector< std::vector<uint> >& assignment, bool init_seeds = true);



    void operator()(const std::vector<Feature>& points, std::vector<Feature>& seeds, bool init_seeds = true)
    {
      clusterPoints(points, seeds, init_seeds);
    }

    void operator()(const std::vector<Feature>& points, std::vector<Feature>& seeds,
                    std::vector< std::vector<uint> >& assignment, bool init_seeds = true)
    {
      clusterPoints(points, seeds, assignment, init_seeds);
    }


//    void setClusterInitializer( initializer::Initializer<Feature>& cluster_initializer )
//    {
//      _cluster_initializer.reset( &cluster_initializer );
//    }

  protected:

    /**< The number of clusters. */
    uint _k;

    /**< The maximum number of iterations. */
    uint _maxIterations;

    /**< The minimum error difference. */
    double _minError;

    /**< The distance function. */
    boost::shared_ptr<distance::HistogramDistance<Feature> > _distance;

    /**< The cluster initialization function. */
    boost::shared_ptr<initializer::Initializer<Feature> > _cluster_initializer;
  };

  template <typename Feature, typename Accu>
  KMeans<Feature, Accu>::KMeans(uint clusters, uint maxIterations, double minError):
      _k(clusters),
      _maxIterations(maxIterations),
      _minError(minError)
  {
    _cluster_initializer.reset( new initializer::RandomAssignmentInitialization<Feature>(_k) );
    _distance.reset( new distance::EuclideanDistance<Feature> );
  }

  template <typename Feature, typename Accu>
  double KMeans<Feature, Accu>::clusterPoints(const std::vector<Feature>& points,
                                              std::vector<Feature>& seeds, bool init_seeds)
  {
    std::vector< std::vector<unsigned int> > assignment;
    return clusterPoints(points, seeds, assignment, init_seeds);
  }

  template <typename Feature, typename Accu>
  void KMeans<Feature, Accu>::initializeSeeds(const std::vector<Feature>& points,
                                              std::vector<Feature>& seeds)
  {
    _cluster_initializer->initialize(points, seeds);
  }

  template <typename Feature, typename Accu>
  double KMeans<Feature, Accu>::clusterPoints(const std::vector<Feature>& points,
                                              std::vector<Feature>& seeds,
                                              std::vector< std::vector<uint> >& assignment,
                                              bool init_seeds)
  {
    if(points.size() <= this->_k)
    {
      seeds = points;
      return 0.;
    }

    if (init_seeds)
      _cluster_initializer->initialize(points, seeds);

    double error, oldError = 0;
    for (uint i = 0; i < _maxIterations; ++i)
    {
      error = 0;

      assignment.clear();
      assignment.resize(this->_k);

      std::vector<std::vector<const Feature*> > assignment_ptr(this->_k);

      // Assign each point to a cluster
      #pragma omp parallel for reduction(+:error)
      for(uint p = 0; p < points.size(); ++p)
      {
        uint bestCluster = 0;
        double maxSim = 0;

        for(uint c = 0; c < seeds.size(); ++c)
        {
          double sim = exp(- _distance->distance(seeds[c], points[p]) );

          if(sim > maxSim)
          {
            maxSim = sim;
            bestCluster = c;
          }
        }

        error += maxSim;

        #pragma omp critical(datapush)
        {
          assignment[bestCluster].push_back(p);
          assignment_ptr[bestCluster].push_back( &points[p] );
        }
      }

      size_t f_size = points[0].size();

      // Recompute each cluster
      for (uint c = 0; c < this->_k; ++c)
      {

        std::vector<Accu> accumulator_arr( f_size );

        // Number of assigned features
        for (size_t a = 0; a < assignment_ptr[c].size(); ++a)
          for (size_t l = 0; l < f_size; ++l)
            accumulator_arr[l] += acc_type( (*(assignment_ptr[c][a]))[l] ) / acc_type(assignment_ptr[c].size());

        for (size_t t = 0; t < f_size; ++t)
          seeds[c][t] = accumulator_arr[t];
      }

      if(fabs(error - oldError) < _minError)
        break;

      oldError = error;
    }

    return error;
  }

} // namespace clustering

#endif
