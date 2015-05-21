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

#ifndef HIERARCHICAL_CLUSTERING_ACCUMULATOR_H
#define HIERARCHICAL_CLUSTERING_ACCUMULATOR_H

#include <vector>

template<class Numeric>
class KahanAccumulator;

template<class Numeric = double>
class Accumulator
{
#pragma omp declare reduction (+ : Accumulator : Accumulator + Accumulator)

public:

  friend class KahanAccumulator<Numeric>;

  typedef Numeric type;

  /** Default destructor. */
  Accumulator() { this->_value = 0.; }
  Accumulator(Numeric val) { this->_value = val; }

  virtual ~Accumulator() {}

  template<class T>
  Accumulator<Numeric>& operator+=(const T& o)
  {
    this->_value += o;
    return *this;
  }

  template<class T>
  Numeric operator+(const T& o)
  {
    return this->_value + o;
  }

  template<class T>
  Numeric operator+(const Accumulator<T>& acc)
  {
    return this->_value + acc._value;
  }

  template<class T>
  Accumulator<T> operator+(const KahanAccumulator<T>& acc)
  {
    return this->_value + acc._value;
  }

  template<class T>
  friend T operator/(const Accumulator<T>& lhs, const Accumulator<T>& rhs);

  template<class T>
  Numeric& operator=(const T& val)
  {
    return this->_value = val;
  }

  template<class T>
  Accumulator<Numeric>& operator=(const Accumulator<T>& acc)
  {
    this->_value = acc._value;
    return *this;
  }

  template<class Type>
  operator Type()
  {
    return static_cast<Type>(this->_value);
  }

protected:

  Numeric _value;
};

template<class Numeric = double>
class KahanAccumulator: public Accumulator<Numeric>
{
public:

//  #pragma omp declare reduction (merge : KahanAccumulator : KahanAccumulator + KahanAccumulator)

  typedef Numeric type;

  KahanAccumulator() :
    _c(0) {}

  virtual ~KahanAccumulator() {}

  /**
    * Operator =
    *
    */
  template<class T>
  KahanAccumulator& operator=(const KahanAccumulator<T>& acc)
  {
    this->_value = acc._value;
    this->_c = acc._C;
    return *this;
  }

  template<class T>
  KahanAccumulator& operator=(const Accumulator<T>& acc)
  {
    this->_value = acc._value;
    return *this;
  }

  template<class T>
  KahanAccumulator& operator=(const T& o)
  {
    this->_value = o;
    _c = 0.;
    return *this;
  }

  /**
    * Operator +=
    *
    */
  template<class T>
  KahanAccumulator<Numeric>& operator+=(const T& o)
  {
    Numeric y( Numeric(o) - _c);
    Numeric t(this->_value + y);

    _c = (t - this->_value) - y;

    this->_value = t;

    return *this;
  }

  /**
    * Operator +
    *
    */
  template<class T>
  KahanAccumulator<Numeric> operator+(const T& o)
  {
    Numeric y( Numeric(o) - _c);
    Numeric t(this->_value + y);

    _c = (t - this->_value) - y;

    this->_value = t;

    return *this;
  }

  template<class T>
  Accumulator<T> operator+(const Accumulator<T>& acc)
  {
    Numeric y( Numeric(acc._value) - _c);
    Numeric t(this->_value + y);

    _c = (t - this->_value) - y;

    this->_value = t;

    return *this;
  }

  /**
    * Operator *
    *
    */
  template<class T>
  KahanAccumulator<Numeric>& operator*(const KahanAccumulator<T>& acc)
  {
    this->_value *= acc._value;

    return *this;
  }

  template<class T>
  KahanAccumulator<Numeric> operator*(const T& o)
  {
    KahanAccumulator<Numeric> kacc;

    kacc = *this;

    kacc._value *= o;

    return kacc;
  }

  /**
    * Operator type()
    *
    */
  template<class Type>
  operator Type()
  {
    return static_cast<Type>(this->_value);
  }

private:

  double _c;
};

template<class T>
T operator/(const Accumulator<T>& lhs, const Accumulator<T>& rhs)
{
  return lhs._value / rhs._value;
}

template <class T>
inline std::ostream& operator<< (std::ostream& os, Accumulator<T>& acc)
{
  os << " [ " << T(acc) << " ]";

  return os;
}

#endif //HIERARCHICAL_CLUSTERING_ACCUMULATOR_H
