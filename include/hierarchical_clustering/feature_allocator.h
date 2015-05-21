#ifndef HIERARCHICAL_CLUSTERING_FEATURE_ALLOCATOR_H
#define HIERARCHICAL_CLUSTERING_FEATURE_ALLOCATOR_H

//#include <Eigen3/Core>
//#include <Eigen/StdVector>

#ifdef _WIN32
  #include <malloc.h>
#endif

// The following headers are required for all allocators.
#include <stddef.h>  // Required for size_t and ptrdiff_t and NULL
#include <new>       // Required for placement new and std::bad_alloc
#include <stdexcept> // Required for std::length_error

// The following headers contain stuff that Mallocator uses.
#include <stdlib.h>  // For malloc() and free()
#include <iostream>  // For std::cout
#include <ostream>   // For std::endl

namespace allocator
{
  /**
   * \brief Meta-function to get the default allocator for a particular feature type.
   *
   * Defaults to \c std::allocator<Feature>.
   */
  template<class Feature>
  struct DefaultAllocator
  {
    typedef std::allocator<Feature> type;
  };

  template<class Feature>
  struct Feature32Allocator
  {
    typedef std::allocator<Feature> type;
  };

  //// Specialization to use aligned allocator for Eigen::Matrix types.
  //template<typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
  //struct DefaultAllocator< Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols> >
  //{
  //  typedef Eigen::aligned_allocator<Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols> > type;
  //};

  // STL align allocator from Stephan T. Lavavej, Visual C++ Libraries Developer
  // (http://blogs.msdn.com/b/vcblog/archive/2008/08/28/the-mallocator.aspx)
  template <typename T>
  class Mallocator
  {
  public:
    // The following will be the same for virtually all allocators.
    typedef T * pointer;
    typedef const T * const_pointer;
    typedef T& reference;
    typedef const T& const_reference;
    typedef T value_type;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;

    T * address(T& r) const
    {
      return &r;
    }

    const T * address(const T& s) const
    {
      return &s;
    }

    size_t max_size() const
    {
      // The following has been carefully written to be independent of
      // the definition of size_t and to avoid signed/unsigned warnings.
      return (static_cast<size_t>(0) - static_cast<size_t>(1)) / sizeof(T);
    }

    // The following must be the same for all allocators.
    template <typename U>
    struct rebind
    {
      typedef Mallocator<U> other;
    };

    bool operator!=(const Mallocator& other) const
    {
      return !(*this == other);
    }

    void construct(T * const p, const T& t) const
    {
      void * const pv = static_cast<void *>(p);
      new (pv) T(t);
    }

    void destroy(T * const p) const
    {
      p->~T();
    } // Defined below.

    // Returns true if and only if storage allocated from *this
    // can be deallocated from other, and vice versa.
    // Always returns true for stateless allocators.
    bool operator==(const Mallocator& other) const
    {
      return true;
    }

    // Default constructor, copy constructor, rebinding constructor, and destructor.
    // Empty for stateless allocators.
    Mallocator() { }

    Mallocator(const Mallocator&) { }

    template <typename U> Mallocator(const Mallocator<U>&) { }

    ~Mallocator() { }

    // The following will be different for each allocator.
    T * allocate(const size_t n) const
    {
      // The return value of allocate(0) is unspecified.
      // Mallocator returns NULL in order to avoid depending
      // on malloc(0)'s implementation-defined behavior
      // (the implementation can define malloc(0) to return NULL,
      // in which case the bad_alloc check below would fire).
      // All allocators can return NULL in this case.
      if (n == 0)
      {
        return NULL;
      }

      // All allocators should contain an integer overflow check.
      // The Standardization Committee recommends that std::length_error
      // be thrown in the case of integer overflow.
      if (n > max_size())
      {
        throw std::length_error("Mallocator<T>::allocate() - Integer overflow.");
      }

      // Mallocator wraps malloc().
      void * const pv = malloc(n * sizeof(T));

      // Allocators should throw std::bad_alloc in the case of memory allocation failure.
      if (pv == NULL)
      {
        throw std::bad_alloc();
      }
      return static_cast<T *>(pv);
    }

    void deallocate(T * const p, const size_t n) const
    {
      // Mallocator wraps free().
      free(p);
    }

    // The following will be the same for all allocators that ignore hints.
    template <typename U> T * allocate(const size_t n, const U * /* const hint */) const
    {
      return allocate(n);
    }

    // Allocators are not required to be assignable, so
    // all allocators should have a private unimplemented
    // assignment operator. Note that this will trigger the
    // off-by-default (enabled under /Wall) warning C4626
    // "assignment operator could not be generated because a
    // base class assignment operator is inaccessible" within
    // the STL headers, but that warning is useless.

  private:
    Mallocator& operator=(const Mallocator&);
  };

//  // STL align allocator for placing SIMD types in std::vector
//  // inspired from Stephan T. Lavavej, Visual C++ Libraries Developer
//  // (http://blogs.msdn.com/b/vcblog/archive/2008/08/28/the-mallocator.aspx)
//  template <typename T, typename Alignment>
//  class MallocatorSIMD
//  {
//  public:
//    // The following will be the same for virtually all allocators.
//    typedef T * pointer;
//    typedef const T * const_pointer;
//    typedef T& reference;
//    typedef const T& const_reference;
//    typedef T value_type;
//    typedef size_t size_type;
//    typedef ptrdiff_t difference_type;

//    T * address(T& r) const
//    {
//      return &r;
//    }

//    const T * address(const T& s) const
//    {
//      return &s;
//    }

//    size_t max_size() const
//    {
//      // The following has been carefully written to be independent of
//      // the definition of size_t and to avoid signed/unsigned warnings.
//      return (static_cast<size_t>(0) - static_cast<size_t>(1)) / sizeof(T);
//    }

//    // The following must be the same for all allocators.
//    template <typename U>
//    struct rebind
//    {
//      typedef MallocatorSIMD<U, Alignment> other;
//    };

//    bool operator!=(const MallocatorSIMD& other) const
//    {
//      return !(*this == other);
//    }

//    void construct(T * const p, const T& t) const
//    {
//      void * const pv = static_cast<void *>(p);
//      new (pv) T(t);
//    }

//    void destroy(T * const p) const
//    {
//      p->~T();
//    }

//    // Returns true if and only if storage allocated from *this
//    // can be deallocated from other, and vice versa.
//    // Always returns true for stateless allocators.
//    bool operator==(const MallocatorSIMD& other) const
//    {
//      return true;
//    }

//    // Default constructor, copy constructor, rebinding constructor, and destructor.
//    // Empty for stateless allocators.
//    MallocatorSIMD() { }

//    MallocatorSIMD(const MallocatorSIMD&) { }

//    template<typename U>
//    MallocatorSIMD(const MallocatorSIMD<U, Alignment>&) { }

//    ~MallocatorSIMD() { }

//    // The following will be different for each allocator.
//    T * allocate(const size_t n) const
//    {
//      // The return value of allocate(0) is unspecified.
//      // Mallocator returns NULL in order to avoid depending
//      // on malloc(0)'s implementation-defined behavior
//      // (the implementation can define malloc(0) to return NULL,
//      // in which case the bad_alloc check below would fire).
//      // All allocators can return NULL in this case.
//      if (n == 0)
//      {
//        return NULL;
//      }

//      // All allocators should contain an integer overflow check.
//      // The Standardization Committee recommends that std::length_error
//      // be thrown in the case of integer overflow.
//      if (n > max_size())
//      {
//        throw std::length_error("MallocatorSIMD<T>::allocate() - Integer overflow.");
//      }

//      // MallocatorSIMD wraps malloc().
//      void* const pv/* = _mm_malloc(n * sizeof(T), Alignment)*/;

//      // Allocators should throw std::bad_alloc in the case of memory allocation failure.
//      if (pv == NULL)
//      {
//        throw std::bad_alloc();
//      }
//      return static_cast<T *>(pv);
//    }

//    void deallocate(T * const p, const size_t n) const
//    {
//      // MallocatorSIMD wraps free().
//      _mm_free(p);
//    }

//    // The following will be the same for all allocators that ignore hints.
//    template <typename U> T * allocate(const size_t n, const U * /* const hint */) const
//    {
//      return allocate(n);
//    }

//    // Allocators are not required to be assignable, so
//    // all allocators should have a private unimplemented
//    // assignment operator. Note that this will trigger the
//    // off-by-default (enabled under /Wall) warning C4626
//    // "assignment operator could not be generated because a
//    // base class assignment operator is inaccessible" within
//    // the STL headers, but that warning is useless.

//  private:
//    MallocatorSIMD& operator=(const MallocatorSIMD&);
//  };

} //namespace allocator

#endif
