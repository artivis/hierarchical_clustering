
#ifndef HIERARCHICAL_CLUSTERING_HIERARCHICALCLUSTERING_H
#define HIERARCHICAL_CLUSTERING_HIERARCHICALCLUSTERING_H

/**
 * Implement a centroid-based hierarchical clustering algorithm.
 *
 * @author TODO
 *
 */

#include <vector>
#include <deque>
#include <set>

#include <omp.h>

#include <boost/foreach.hpp>

#include "hierarchical_clustering/kMeansClustering.h"

namespace
{
  int distict_val(const std::vector<uint>& v)
  {
     std::set<uint> dist_val;

     for(std::vector<uint>::const_iterator curr_int = v.begin(), end = v.end();
         curr_int != end; ++curr_int)
         dist_val.insert(*curr_int);

     return dist_val.size();
  }
}

namespace clustering
{

template <typename Feature,
          typename Distance = distance::EuclideanDistance<Feature, KahanAccumulator<double> >,
          typename SeedInitializer = initializer::ForgyInitializer<Feature>,
          typename Accu = KahanAccumulator<double>
          /*, typename Alloc = allocator::DefaultAllocator<Feature>*/ >
  class HierarchicalClustering :
      public KMeans<Feature, Distance, SeedInitializer, Accu>
  {
    public:

    /**
      * Default constructor. It set the number of clusters and levels to zero,
      * they have to be set later on. It set the maximum iterations for the
      * clustering and the minimum error difference for convergence default values.
      * @param clusters - number of clusters per node
      * @param levels - number of levels
      * @param maxIterations - maximum number of iterations
      * @param minError - minimum error
      *
      */
    HierarchicalClustering(uint clusters = 0, uint levels = 0,
                           uint maxIterations = 100, double minError = 0.001);

    /**
     * Cluster the @param points into clusters. It initialize the centroids with the
     * provided Initialization class.
     * The number of clusters is the size of @param seeds.
     * The @param seeds are modified to hold the clusters.
     *
     */
    virtual void hierarchicalClusterPoints(const std::vector<Feature>& points,
                                           std::vector<Feature>& seeds)
    {
      if (!_assertParams())
      {
        std::cerr << "\nHierarchical clustering parameters not set !" << std::endl;
        return;
      }

      if (points.size() == 0)
        return;

      uint feature_size = points[0].size();

      if (feature_size == 0)
        return;

      // We keep a queue of disjoint feature subsets to cluster.
      // Feature* is used to avoid copying features.
      std::deque< std::vector<Feature*> > subset_queue(1);

      // At first the queue contains one "subset" containing all the features.
      std::vector<Feature*> &feature_ptrs = subset_queue.front();

      feature_ptrs.resize( points.size() );

      seeds.clear();
      seeds.reserve( this->_exp_leafs_count );

      #pragma omp parallel for
      for (size_t i=0; i<points.size(); ++i)
        feature_ptrs[i] = const_cast<Feature*>( &points[i] );

      std::vector<Feature> local_seeds(this->_k, Feature(feature_size, 0.));

      for (uint level = 0; level < _levels; ++level)
      {
        //std::cout << "Level : " << level << std::endl;

        size_t ie = subset_queue.size();

        for (size_t is = 0; is < ie; ++is)
        {
          //std::cout << "is : " << is << std::endl;

          std::vector<Feature*>& subset = subset_queue.front();

          uint seeds_num = std::min(subset.size(), this->_k);

          local_seeds.resize(seeds_num, Feature(feature_size, 0.));
          std::vector<uint> local_assignement;

          this->clusterPoints(subset, local_seeds, local_assignement);

          _comp_centroids_count += local_seeds.size();

          if (subset.size() > this->_k)
          {
            // Partition the current subset into k new subsets based on the cluster assignments.
            std::vector< std::vector<Feature*> > new_subsets( this->_k );

            for (size_t j = 0; j < subset.size(); ++j)
              new_subsets[ local_assignement[j] ].push_back( subset[j] );

            /*for (size_t j = 0; j < new_subsets.size(); ++j)
              if (new_subsets[j].size() == 0)
                std::cout << "this subste is size zero !" << std::endl;
              else
                std::cout << "this subste is size " << new_subsets[j].size() << " !" << std::endl;
            */

            // Update the queue
            subset_queue.pop_front();
            subset_queue.insert(subset_queue.end(), new_subsets.begin(), new_subsets.end());

            // If reached last level then local seeds are seeds.
            if (level == (_levels - 1))
              std::copy(local_seeds.begin(), local_seeds.end(), std::back_inserter(seeds));

            // Keep local copy of intermediate centroids
            //std::copy(local_seeds.begin(), local_seeds.end(), std::back_inserter(_centroids));
          }
          else
          {
            // If subset is less than number of clusters then local seeds are seeds.
            std::copy(local_seeds.begin(), local_seeds.end(), std::back_inserter(seeds));
            subset_queue.pop_front();
          }
        }
      }

      //std::cout << "Expect approximatly : " << _exp_centroids_count << " centroids." << std::endl;
      //std::cout << "Computed : " << _comp_centroids_count << " centroids." << std::endl << std::endl;

      //std::cout << "Expect approximatly : " << _exp_leafs_count << " leafs centroids." << std::endl;
      //std::cout << "Computed : "  << seeds.size() << " leafs centroids." << std::endl << std::endl;

      _comp_leafs_count = seeds.size();

      // trick to release reserved but unused memory.
      // @TODO check how bad it is in term of memory consumption
      // probably does a local copy
      std::vector<Feature>(seeds).swap(seeds);
    }

    /**
     * Cluster the @param points into clusters. It initialize the centroids with the @param seeds.
     * The number of clusters is the size of @param seeds. The @param seeds are modified to hold the clusters.
     *
     */
    virtual void hierarchicalClusterPoints(const std::vector<Feature>& points,
                                           std::vector<Feature>& seeds,
                                           std::vector<uint>& assignment)
    {
      hierarchicalClusterPoints(points, seeds);

      assignment.clear();
      assignment.resize( points.size() );

      // Todo : can't use Accu for error
      // due to omp reduction
      // Assign each point to a cluster
      #pragma omp parallel for
      for(uint p = 0; p < points.size(); ++p)
      {
        uint bestCluster = 0;
        double maxDist = MAX_DOUBLE;

        for(uint c = 0; c < seeds.size(); ++c)
        {
          double dist = this->_distance.distance(seeds[c], points[p] );

          if(dist < maxDist)
          {
            maxDist = dist;
            bestCluster = c;
          }
        }

        assignment[p] = bestCluster;
      }
    }

    virtual void operator()(const std::vector<Feature>& points, std::vector<Feature>& seeds)
    {
      hierarchicalClusterPoints(points, seeds);
    }

    virtual void operator()(const std::vector<Feature>& points, std::vector<Feature>& seeds,
                            std::vector<uint>& assignment)
    {
      hierarchicalClusterPoints(points, seeds, assignment);
    }

    virtual void setClustersNumbers(uint k)
    {
      this->_k = k;
      this->_cluster_initializer.setClusterNumber(this->_k);
      _reserve();
    }

    virtual void setLevels(uint levels)
    {
      _levels = levels;
      _reserve();
    }

    uint getLevels() { return _levels; }

    protected:

    uint _exp_centroids_count;
    uint _exp_leafs_count;

    uint _comp_centroids_count;
    uint _comp_leafs_count;

    /**< The number of level. */
    uint _levels;

    bool virtual _assertParams()
    {
      return (this->_k > 0) &&
             (this->_levels >= 0) &&
             (this->_minError > 0) &&
             (this->_maxIterations > 0);
    }

  private:

    bool _reserve()
    {
      _exp_leafs_count = pow(this->_k, this->_levels);

      _exp_centroids_count = 0;
      for (int i=1; i<=this->_levels; ++i)
        _exp_centroids_count += pow(this->_k, i);
    }

  };

  template <typename Feature, typename Distance, typename SeedInitializer, typename Accu>
  HierarchicalClustering<Feature, Distance, SeedInitializer, Accu>::HierarchicalClustering(uint clusters, uint level,
                                                                                           uint maxIterations, double minError) :
    KMeans<Feature, Distance, SeedInitializer, Accu>(clusters, maxIterations, minError),
    _levels(level),
    _exp_centroids_count(0),
    _exp_leafs_count(0),
    _comp_centroids_count(0),
    _comp_leafs_count(0)
  {
    _reserve();
  }

} // namespace clustering

#endif //HIERARCHICAL_CLUSTERING_HIERARCHICALCLUSTERING_H
