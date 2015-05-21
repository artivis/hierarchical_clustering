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
#include <limits>

#include <omp.h>

#include "hierarchical_clustering/distances.h"
#include "hierarchical_clustering/feature_allocator.h"
#include "hierarchical_clustering/clusteringInitializer.h"

namespace
{
  const double MAX_DOUBLE = std::numeric_limits<double>::max();
}

namespace clustering
{
  template <typename Feature,
            typename Distance = distance::EuclideanDistance<Feature, KahanAccumulator<double> >,
            typename SeedInitializer = initializer::PlusPlusKmeansInitialization<Feature, Distance>,
            typename Accu = KahanAccumulator<double>
            /*, typename Alloc = allocator::DefaultAllocator<Feature>*/ >
  class KMeans
  {
    typedef typename Feature::value_type value_type;
    typedef typename Accu::type acc_type;

  public:

    /**
      * Default constructor. It set the number of clusters to zero, it has to be set later on.
      * It set the maximum iterations for the clustering and the minimum error difference for convergence default values.
      * @param clusters - number of clusters
      * @param maxIterations - maximum number of iterations
      * @param minError - minimum error
      *
      */
    KMeans(uint clusters = 0, uint maxIterations = 100, double minError = 0.001);

    ~KMeans() { }

    /**
      * Initializes the centroids @param seeds from the @param points.
      * The @param seeds are modified to hold the clusters.
      *
      */
    bool initializeSeeds(const std::vector<Feature>& points, std::vector<Feature>& seeds);

    /**
      * Cluster the @param points into clusters. It initialize the centroids with the @param seeds.
      * The @param seeds are modified to hold the clusters.
      *
      */
    double clusterPoints(const std::vector<Feature>& points, std::vector<Feature>& seeds, bool init_seeds = true);

    /**
      * Cluster the @param points into clusters. It initialize the centroids with the @param seeds.
      * The @param seeds are modified to hold the clusters.
      *
      */
    double clusterPoints(const std::vector<Feature*>& points, std::vector<Feature>& seeds, bool init_seeds = true);

    /**
      * Cluster the @param points into clusters. It initialize the centroids with the @param seeds.
      * The @param seeds are modified to hold the clusters.
      * This overloaded version returns also the point assignments.
      *
      */
    double clusterPoints(const std::vector<Feature>& points, std::vector<Feature>& seeds,
                         std::vector<uint>& assignment, bool init_seeds = true);

    /**
      * Cluster the @param points into clusters. It initialize the centroids with the @param seeds.
      * The @param seeds are modified to hold the clusters.
      * This overloaded version returns also the point assignments.
      *
      */
    double clusterPoints(const std::vector<Feature*>& points, std::vector<Feature>& seeds,
                         std::vector<uint>& assignment, bool init_seeds = true);



    virtual void operator()(const std::vector<Feature>& points, std::vector<Feature>& seeds, bool init_seeds = true)
    {
      clusterPoints(points, seeds, init_seeds);
    }

    virtual void operator()(const std::vector<Feature>& points, std::vector<Feature>& seeds,
                            std::vector<uint>& assignment, bool init_seeds = true)
    {
      clusterPoints(points, seeds, assignment, init_seeds);
    }

    virtual void setClustersNumbers(uint k)
    {
      _k = k;
      _cluster_initializer.setClusterNumber(_k);
    }

    void setMaxIteration(uint max_ite) {_maxIterations = max_ite;}

    void setMinError(double min_error) {_minError = min_error;}

    uint getClustersNumbers() {return _k;}

    uint getMaxIteration() {return _maxIterations;}

    double getMinError() {return _minError;}

  protected:

    /**< The number of clusters. */
    uint _k;

    /**< The maximum number of iterations. */
    uint _maxIterations;

    /**< The minimum error difference. */
    double _minError;

    /**< The distance function. */
    Distance _distance;

    /**< The cluster initialization function. */
    SeedInitializer _cluster_initializer;

    bool virtual _assertParams()
      {return (_k > 0) && (_maxIterations > 0) && (_minError > 0);}
  };

  template <typename Feature, typename Distance, typename SeedInitializer, typename Accu>
  KMeans<Feature, Distance, SeedInitializer, Accu>::KMeans(uint clusters, uint maxIterations, double minError) :
      _k(clusters),
      _maxIterations(maxIterations),
      _minError(minError),
      _cluster_initializer(_k)
  {

  }

  template <typename Feature, typename Distance, typename SeedInitializer, typename Accu>
  bool KMeans<Feature, Distance, SeedInitializer, Accu>::initializeSeeds(const std::vector<Feature>& points,
                                                                         std::vector<Feature>& seeds)
  {
    return _cluster_initializer.initialize(points, seeds);
  }

  template <typename Feature, typename Distance, typename SeedInitializer, typename Accu>
  double KMeans<Feature, Distance, SeedInitializer, Accu>::clusterPoints(const std::vector<Feature>& points,
                                                                         std::vector<Feature>& seeds, bool init_seeds)
  {
    std::vector<uint> assignment;
    return clusterPoints(points, seeds, assignment, init_seeds);
  }

  template <typename Feature, typename Distance, typename SeedInitializer, typename Accu>
  double KMeans<Feature, Distance, SeedInitializer, Accu>::clusterPoints(const std::vector<Feature*>& points,
                                                                         std::vector<Feature>& seeds, bool init_seeds)
  {
    std::vector<uint> assignment;
    return clusterPoints(points, seeds, assignment, init_seeds);
  }

  template <typename Feature, typename Distance, typename SeedInitializer, typename Accu>
  double KMeans<Feature, Distance, SeedInitializer, Accu>::clusterPoints(const std::vector<Feature>& points,
                                                                         std::vector<Feature>& seeds,
                                                                         std::vector<uint>& assignment,
                                                                         bool init_seeds)
  {
    if (!_assertParams())
    {
      std::cerr << "Kmeans clustering parameters not set !" << std::endl;
      return 10e16;
    }

    if (!_cluster_initializer.isInit() && init_seeds)
    {
      std::cerr << "Cluster initializer is not set !" << std::endl;
      return 10e16;
    }

    if(points.size() <= this->_k)
    {
      seeds = points;
      return 0.;
    }

    if (init_seeds)
      _cluster_initializer.initialize(points, seeds);

    assignment.resize( points.size() );

    double error, oldError = 0;
    for (uint i = 0; i < _maxIterations; ++i)
    {
      error = 0;

      std::vector<std::vector<const Feature*> > assignment_ptr(this->_k);

      // Assign each point to a cluster
      #pragma omp parallel for reduction(+:error)
      for(uint p = 0; p < points.size(); ++p)
      {
        uint bestCluster = 0;
        double maxDist = MAX_DOUBLE;

        for(uint c = 0; c < seeds.size(); ++c)
        {
          double dist = _distance.distance(seeds[c], points[p]);

          if(dist < maxDist)
          {
            maxDist = dist;
            bestCluster = c;
          }
        }

        error += maxDist;

        assignment[p] = bestCluster;

        #pragma omp critical(datapush)
        {
          assignment_ptr[bestCluster].push_back( &points[p] );
        }
      }

      size_t f_size = points[0].size();

      // Recompute each cluster
      #pragma omp parallel for
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

    // Assign each point to a cluster
    #pragma omp parallel for reduction(+:error)
    for(uint p = 0; p < points.size(); ++p)
    {
      uint bestCluster = 0;
      double maxDist = MAX_DOUBLE;

      for(uint c = 0; c < seeds.size(); ++c)
      {
        double dist = _distance.distance(seeds[c], points[p]);

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

  template <typename Feature, typename Distance, typename SeedInitializer, typename Accu>
  double KMeans<Feature, Distance, SeedInitializer, Accu>::clusterPoints(const std::vector<Feature*>& points,
                                                                         std::vector<Feature>& seeds,
                                                                         std::vector<uint>& assignment,
                                                                         bool init_seeds)
  {
    if (!_assertParams())
    {
      std::cerr << "Kmeans clustering parameters not set !" << std::endl;
      return 10e16;
    }

    if (!_cluster_initializer.isInit() && init_seeds)
    {
      std::cerr << "Cluster initializer is not set !" << std::endl;
      return 10e16;
    }

    if(points.size() <= this->_k)
    {
      for (size_t i=0; i<points.size(); ++i)
        seeds[i] = *points[i];
      return 0.;
    }

    if (init_seeds)
      _cluster_initializer.initialize(points, seeds);

    assignment.resize( points.size() );

    double error, oldError = 0;
    for (uint i = 0; i < _maxIterations; ++i)
    {
      error = 0;

      std::vector<std::vector<const Feature*> > assignment_ptr(this->_k);

      uint p;

      // Todo : can't use Accu for error
      // due to omp reduction
      // Assign each point to a cluster
      #pragma omp parallel for reduction(+:error)
      for(p = 0; p < points.size(); ++p)
      {
        uint bestCluster = 0;
        double maxDist = MAX_DOUBLE;

        for(uint c = 0; c < seeds.size(); ++c)
        {
          double dist = _distance.distance(seeds[c], *points[p] );

          if(dist < maxDist)
          {
            maxDist = dist;
            bestCluster = c;
          }
        }

        error += maxDist;

        assignment[p] = bestCluster;

        #pragma omp critical(datapush)
        {
          assignment_ptr[bestCluster].push_back( points[p] );
        }
      }

      size_t f_size = points[0]->size();

      // Recompute each cluster
      #pragma omp parallel for
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

    // Assign each point to a cluster
    #pragma omp parallel for reduction(+:error)
    for(uint p = 0; p < points.size(); ++p)
    {
      uint bestCluster = 0;
      double maxDist = MAX_DOUBLE;

      for(uint c = 0; c < seeds.size(); ++c)
      {
        double dist = _distance.distance(seeds[c], *points[p]);

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

//   TODO typedef default e.g. EuclideanKmeans

} // namespace clustering

#endif //HIERARCHICAL_CLUSTERING_KMEANSCLUSTRING_H
