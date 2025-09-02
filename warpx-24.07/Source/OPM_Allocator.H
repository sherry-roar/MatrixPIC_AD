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
        // Use hbw_malloc to allocate OPM memory
        void* p = hbw_malloc(n * sizeof(T));
        
        if (!p) {
            // If allocation fails, throw a standard exception
            throw std::bad_alloc();
        }
        return static_cast<T*>(p);
    }

    // Core: memory deallocation function
    void deallocate(T* p, size_t /*n*/) noexcept {
        // Use hbw_free to deallocate OPM memory
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


// Define a vector type that uses OPM memory
template <typename T>
using OPM_Vector = std::vector<T, OPM_Allocator<T>>;

// Apply to your data structures
// Inner vector: OPM vector for storing amrex::Real or int
using OPM_Inner_Vector_Real = OPM_Vector<amrex::Real>;
using OPM_Inner_Vector_Int = OPM_Vector<int>;

// Outer vector: OPM vector for storing other "OPM vectors"
using OPM_Nested_Vector_Real = OPM_Vector<OPM_Inner_Vector_Real>;
using OPM_Nested_Vector_Int = OPM_Vector<OPM_Inner_Vector_Int>;

#endif // OPM_Allocator_H_
