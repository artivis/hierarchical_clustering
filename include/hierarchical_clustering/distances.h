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

#ifndef HIERARCHICAL_CLUSTERING_DISTANCES_H
#define HIERARCHICAL_CLUSTERING_DISTANCES_H

//#include <limits>
#include <cmath>

#include <hierarchical_clustering/Accumulator.h>

namespace
{
  template<class T>
  T ZERO()
  {
    //return std::numeric_limits<T>::min();
    return T(10e-45);
  }

  template<class T>
  T ERROR_VAL()
  {
    return T(10e16);
  }
}

namespace distance
{
  template<class Feature, class Accu = KahanAccumulator<double> >
  class HistogramDistance
  {
    typedef typename Feature::value_type value_type;
    typedef typename Accu::type result_type;

  public:

    /** Default destructor. */
    virtual ~HistogramDistance() { }

    /** Computes the distance between the first and last histogram (1D). */
    virtual result_type distance(const Feature& first, const Feature& last) const = 0;

    /** Computes the weighted distance between the first and last histogram (1D). */
    virtual result_type distance(const Feature& first, const Feature& weightFirst,
                                 const Feature& last,  const Feature& weightLast) const = 0;
  };

  template<class Feature, class Accu = KahanAccumulator<double> >
  class EuclideanDistance: public HistogramDistance<Feature, Accu>
  {
    typedef typename Feature::value_type value_type;
    typedef typename Accu::type result_type;

  public:
    /** Default destructor. */
    virtual ~EuclideanDistance() { }

    result_type distance(const Feature& first, const Feature& last) const
    {
      if (first.size() != last.size())
        return ERROR_VAL<result_type>();

      Accu accumulator;

      for (size_t i=0; i<first.size(); ++i)
        accumulator += (first[i] - last[i])*(first[i] - last[i]);

      return std::sqrt(result_type(accumulator));
    }

    result_type distance(const Feature& first, const Feature& weightFirst,
                         const Feature& last,  const Feature& weightLast) const
    {
      if (first.size() != last.size() ||
          first.size() != weightFirst.size() ||
          last.size()  != weightLast.size())
        return ERROR_VAL<result_type>();

      Accu accumulator;
      Accu normalizer;

      for (size_t i=0; i<first.size(); ++i)
      {
        accumulator += (first[i] - last[i])*(first[i] - last[i])*(weightFirst[i] + weightLast[i]);
        normalizer  += (weightFirst[i] + weightLast[i]);
      }

      return std::sqrt(accumulator/normalizer);
    }

    result_type operator()(const Feature& first, const Feature& last)
    {
      return distance(first, last);
    }

    result_type operator()(const Feature& first, const Feature& weightFirst,
                           const Feature& last, const Feature& weightLast)
    {
      return distance(first, weightFirst, last, weightLast);
    }
  };

  template<class Feature, class Accu = KahanAccumulator<double> >
  class ManhattanDistance: public HistogramDistance<Feature, Accu>
  {
    typedef typename Feature::value_type value_type;
    typedef typename Accu::type result_type;

  public:
    /** Default destructor. */
    virtual ~ManhattanDistance() { }

    result_type distance(const Feature& first, const Feature& last) const
    {
      if (first.size() != last.size())
        return ERROR_VAL<result_type>();

      Accu accumulator;

      for (size_t i=0; i<first.size(); ++i)
        accumulator += std::abs(first[i] - last[i]);

      return accumulator;
    }

    result_type distance(const Feature& first, const Feature& weightFirst,
                         const Feature& last,  const Feature& weightLast) const
    {
      if (first.size() != last.size() ||
          first.size() != weightFirst.size() ||
          last.size()  != weightLast.size())
        return ERROR_VAL<result_type>();

      Accu accumulator;
      Accu normalizer;

      for (size_t i=0; i<first.size(); ++i)
      {
        accumulator += std::abs(first[i] - last[i])*(weightFirst[i] + weightLast[i]);
        normalizer  += (weightFirst[i] + weightLast[i]);
      }

      return accumulator/normalizer;
    }

    result_type operator()(const Feature& first, const Feature& last)
    {
      return distance(first, last);
    }

    result_type operator()(const Feature& first, const Feature& weightFirst,
                           const Feature& last, const Feature& weightLast)
    {
      return distance(first, weightFirst, last, weightLast);
    }
  };

  template<class Feature, class Accu = KahanAccumulator<double> >
  class Chi2Distance: public HistogramDistance<Feature>
  {
    typedef typename Feature::value_type value_type;
    typedef typename Accu::type result_type;

  public:
    /** Default destructor. */
    virtual ~Chi2Distance() { }

    result_type distance(const Feature& first, const Feature& last) const
    {
      if (first.size() != last.size())
        return ERROR_VAL<result_type>();

      Accu accumulator;

      for (size_t i=0; i<first.size(); ++i)
      {
        result_type q = (first[i] == 0) ? ZERO<result_type>() : first[i];
        result_type p = (last[i] == 0)  ? ZERO<result_type>() : last[i];
        accumulator += (q - p)*(q - p)/q;
      }

      return accumulator;
    }

    result_type distance(const Feature& first, const Feature& weightFirst,
                         const Feature& last,  const Feature& weightLast) const
    {
      if (first.size() != last.size() ||
          first.size() != weightFirst.size() ||
          last.size() != weightLast.size() )
        return ERROR_VAL<result_type>();

      Accu accumulator;
      Accu normalizer;

      for (size_t i=0; i<first.size(); ++i)
      {
        size_t q = (first[i] == 0) ? ZERO<result_type>() : first[i];
        size_t p = (last[i] == 0)  ? ZERO<result_type>() : last[i];
        accumulator += ((q - p)*(q - p)/q)*(weightFirst[i] + weightLast[i]);
        normalizer += (weightFirst[i] + weightLast[i]);
      }

      return accumulator/normalizer;
    }

    result_type operator()(const Feature& first, const Feature& last)
    {
      return distance(first, last);
    }

    result_type operator()(const Feature& first, const Feature& weightFirst,
                           const Feature& last, const Feature& weightLast)
    {
      return distance(first, weightFirst, last, weightLast);
    }
  };

  template<class Feature, class Accu = KahanAccumulator<double> >
  class SymmetricChi2Distance: public HistogramDistance<Feature>
  {
    typedef typename Feature::value_type value_type;
    typedef typename Accu::type result_type;

  public:
    /** Default destructor. */
    virtual ~SymmetricChi2Distance() { }

    result_type distance(const Feature& first, const Feature& last) const
    {
      if (first.size() != last.size())
        return ERROR_VAL<result_type>();

      Accu accumulator;

      for (size_t i=0; i<first.size(); ++i)
      {
        result_type q = (first[i] == 0) ? ZERO<result_type>() : first[i];
        result_type p = (last[i] == 0)  ? ZERO<result_type>() : last[i];
        accumulator += (q - p)*(q - p)/(q + p);
      }

      return 0.5 * result_type(accumulator); //TOCHECK Some books says to use 2 other 0.5
    }

    result_type distance(const Feature& first, const Feature& weightFirst,
                         const Feature& last,  const Feature& weightLast) const
    {
      if (first.size() != last.size() ||
          first.size() != weightFirst.size() ||
          last.size() != weightLast.size())
        return ERROR_VAL<result_type>();

      Accu accumulator;
      Accu normalizer;

      for (size_t i=0; i<first.size(); ++i)
      {
        result_type q = (first[i] == 0) ? ZERO<result_type>() : first[i];
        result_type p = (last[i] == 0)  ? ZERO<result_type>() : last[i];
        accumulator += ((q - p)*(q - p)/(q + p))*(weightFirst[i] + weightLast[i]);
        normalizer += (weightFirst[i] + weightLast[i]);
      }

      return accumulator/normalizer;
    }

    result_type operator()(const Feature& first, const Feature& last)
    {
      return distance(first, last);
    }

    result_type operator()(const Feature& first, const Feature& weightFirst,
                           const Feature& last, const Feature& weightLast)
    {
      return distance(first, weightFirst, last, weightLast);
    }
  };

  template<class Feature, class Accu = KahanAccumulator<double> >
  class BhattacharyyaDistance: public HistogramDistance<Feature, Accu>
  {
    typedef typename Feature::value_type value_type;
    typedef typename Accu::type result_type;

  public:
    /** Default destructor. */
    virtual ~BhattacharyyaDistance() { }

    result_type distance(const Feature& first, const Feature& last) const
    {
      if (first.size() != last.size())
        return ERROR_VAL<result_type>();

      Accu accumulator;

      for (size_t i=0; i < first.size(); ++i)
      {
        result_type q = (first[i] == 0) ? ZERO<result_type>() : first[i];
        result_type p = (last[i] == 0)  ? ZERO<result_type>() : last[i];
        accumulator += std::sqrt(q * p);
      }

      return -std::log( result_type(accumulator) );
    }

    result_type distance(const Feature& first, const Feature& weightFirst,
                         const Feature& last,  const Feature& weightLast) const
    {
      if (first.size() != last.size() ||
          first.size() != weightFirst.size() ||
          last.size()  != weightLast.size())
        return ERROR_VAL<result_type>();

      Accu accumulator;
      Accu normalizer;

      for (size_t i=0; i<first.size(); ++i)
      {
        result_type q = (first[i] == 0) ? ZERO<result_type>() : first[i];
        result_type p = (last[i] == 0)  ? ZERO<result_type>() : last[i];
        accumulator += std::sqrt(q * p)*(weightFirst[i] + weightLast[i]);
        normalizer += (weightFirst[i] + weightLast[i]);
      }

      return -std::log( result_type(accumulator/normalizer) );
    }

    result_type operator()(const Feature& first, const Feature& last)
    {
      return distance(first, last);
    }

    result_type operator()(const Feature& first, const Feature& weightFirst,
                           const Feature& last, const Feature& weightLast)
    {
      return distance(first, weightFirst, last, weightLast);
    }
  };

  template<class Feature, class Accu = KahanAccumulator<double> >
  class KullbackLeiblerDistance: public HistogramDistance<Feature, Accu>
  {
    typedef typename Feature::value_type value_type;
    typedef typename Accu::type result_type;

  public:
    /** Default destructor. */
    virtual ~KullbackLeiblerDistance() { }

    result_type distance(const Feature& first, const Feature& last) const
    {
      if (first.size() != last.size())
        return ERROR_VAL<result_type>();

      Accu accumulator;

      for (size_t i=0; i<first.size(); ++i)
      {
        result_type q = (first[i] <= 0) ? ZERO<result_type>() : first[i];
        result_type p = (last[i]  <= 0) ? ZERO<result_type>() : last[i];
        accumulator += p * std::log(p/q);
      }

      return accumulator;
    }

    result_type distance(const Feature& first, const Feature& weightFirst,
                            const Feature& last,  const Feature& weightLast) const
    {
      if (first.size() != last.size() ||
          first.size() != weightFirst.size() ||
          last.size()  != weightLast.size())
        return ERROR_VAL<result_type>();

      Accu accumulator;
      Accu normalizer;

      for (size_t i=0; i<first.size(); ++i)
      {
        result_type q = (first[i] <= 0) ? ZERO<result_type>() : first[i];
        result_type p = (last[i]  <= 0) ? ZERO<result_type>() : last[i];
        accumulator += p * std::log(p/q)*(weightFirst[i] + weightLast[i]);
        normalizer  += (weightFirst[i] + weightLast[i]);
      }

      return accumulator/normalizer;
    }

    result_type operator()(const Feature& first, const Feature& last)
    {
      return distance(first, last);
    }

    result_type operator()(const Feature& first, const Feature& weightFirst,
                           const Feature& last, const Feature& weightLast)
    {
      return distance(first, weightFirst, last, weightLast);
    }
  };

  template<class Feature, class Accu = KahanAccumulator<double> >
  class JensenShannonDistance: public HistogramDistance<Feature, Accu>
  {
    typedef typename Feature::value_type value_type;
    typedef typename Accu::type result_type;

  public:
    /** Default destructor. */
    virtual ~JensenShannonDistance() { }

    result_type distance(const Feature& first, const Feature& last) const
    {
      if (first.size() != last.size())
        return ERROR_VAL<result_type>();

      Accu accumulator;

      for (size_t i=0; i<first.size(); ++i)
      {
        result_type q = (first[i] <= 0) ? ZERO<result_type>() : first[i];
        result_type p = (last[i]  <= 0) ? ZERO<result_type>() : last[i];
        accumulator += p * std::log(2*p/(p + q)) + q * std::log(2*q/(p + q));
      }

      return 0.5 * result_type(accumulator) / std::log(2);
    }

    result_type distance(const Feature& first, const Feature& weightFirst,
                         const Feature& last,  const Feature& weightLast) const
    {
      if (first.size() != last.size() ||
          first.size() != weightFirst.size() ||
          last.size()  != weightLast.size())
        return ERROR_VAL<result_type>();

      Accu accumulator;
      Accu normalizer;

      for (size_t i=0; i<first.size(); ++i)
      {
        result_type q = (first[i] <= 0) ? ZERO<result_type>() : first[i];
        result_type p = (last[i]  <= 0) ? ZERO<result_type>() : last[i];
        accumulator += (p * std::log(2*p/(p + q)) + q * std::log(2*q/(p + q)))*(weightFirst[i] + weightLast[i]);
        normalizer += (weightFirst[i] + weightLast[i]);
      }

      return 0.5 * result_type(accumulator/normalizer) / std::log(2);
    }

    result_type operator()(const Feature& first, const Feature& last)
    {
      return distance(first, last);
    }

    result_type operator()(const Feature& first, const Feature& weightFirst,
                           const Feature& last, const Feature& weightLast)
    {
      return distance(first, weightFirst, last, weightLast);
    }
  };

} //namespace distance
#endif
