#ifndef OPM_Allocator_H_
#define OPM_Allocator_H_

#include <hbwmalloc.h> // Include the header for the memkind library

#include <cstddef>     // for size_t
#include <stdexcept>   // for std::bad_alloc


template <typename T>
class OPM_Allocator {
public:
    using value_type = T;

    // Constructor
    OPM_Allocator() noexcept = default;

    // Templated copy constructor
    template <typename U>
    OPM_Allocator(const OPM_Allocator<U>&) noexcept {}

    // Core: memory allocation function
    T* allocate(size_t n) {
        // Use hbw_malloc to allocate HBM memory
        void* p = hbw_malloc(n * sizeof(T));
        
        if (!p) {
            // If allocation fails, throw a standard exception
            throw std::bad_alloc();
        }
        return static_cast<T*>(p);
    }

    // Core: memory deallocation function
    void deallocate(T* p, size_t /*n*/) noexcept {
        // Use hbw_free to deallocate HBM memory
        // Note: the argument for hbw_free is void*
        hbw_free(static_cast<void*>(p));
    }
};

// For a stateless allocator, all instances are equal
template <typename T, typename U>
bool operator==(const OPM_Allocator<T>&, const OPM_Allocator<U>&) {
    return true;
}

template <typename T, typename U>
bool operator!=(const OPM_Allocator<T>&, const OPM_Allocator<U>&) {
    return false;
}


// Define a vector type that uses HBM memory
template <typename T>
using HBM_Vector = std::vector<T, OPM_Allocator<T>>;

// Apply to your data structures
// Inner vector: HBM vector for storing amrex::Real or int
using HBM_Inner_Vector_Real = HBM_Vector<amrex::Real>;
using HBM_Inner_Vector_Int = HBM_Vector<int>;

// Outer vector: HBM vector for storing other "HBM vectors"
using HBM_Nested_Vector_Real = HBM_Vector<HBM_Inner_Vector_Real>;
using HBM_Nested_Vector_Int = HBM_Vector<HBM_Inner_Vector_Int>;

#endif // OPM_Allocator_H_
