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
#include <time.h>
#include <stdlib.h>
#include <vector>

namespace utils
{
  template <class T>
  class Rand
  {
  public:

    Rand() { srand( time(0) ); }
    ~Rand() {}

    double operator()(T fMin, T fMax)
    {
      double f = (double)rand() / (double)RAND_MAX;
      return T(fMin + f * (fMax - fMin));
    }

    static double get(T fMin, T fMax)
    {
      double f = (double)rand() / (double)RAND_MAX;
      return T(fMin + f * (fMax - fMin));
    }
  };
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
