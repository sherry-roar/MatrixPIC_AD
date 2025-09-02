#ifndef HBM_ALLOCATOR_H_
#define HBM_ALLOCATOR_H_

#include <hbwmalloc.h> // 引入memkind库的头文件

#include <cstddef>     // for size_t
#include <stdexcept>   // for std::bad_alloc


template <typename T>
class HBM_Allocator {
public:
    using value_type = T;

    // 构造函数
    HBM_Allocator() noexcept = default;

    // 模板化的拷贝构造函数
    template <typename U>
    HBM_Allocator(const HBM_Allocator<U>&) noexcept {}

    // 核心：分配内存的函数
    T* allocate(size_t n) {
        // 使用 hbw_malloc 来分配HBM内存
        void* p = hbw_malloc(n * sizeof(T));
        // void* p = memkind_malloc(MEMKIND_HBW, n * sizeof(T));
        
        if (!p) {
            // 如果分配失败，抛出标准异常
            throw std::bad_alloc();
        }
        return static_cast<T*>(p);
    }

    // 核心：释放内存的函数
    void deallocate(T* p, size_t /*n*/) noexcept {
        // 使用 hbw_free 释放HBM内存
        // 注意：hbw_free的参数是void*
        hbw_free(static_cast<void*>(p));
    }
};

// 对于无状态的分配器，所有实例都是相等的
template <typename T, typename U>
bool operator==(const HBM_Allocator<T>&, const HBM_Allocator<U>&) {
    return true;
}

template <typename T, typename U>
bool operator!=(const HBM_Allocator<T>&, const HBM_Allocator<U>&) {
    return false;
}


// 定义一个使用HBM内存的vector类型
template <typename T>
using HBM_Vector = std::vector<T, HBM_Allocator<T>>;
// 应用到你的数据结构
// 内层 vector：存放 double/int 的 HBM vector
using HBM_Inner_Vector_Real = HBM_Vector<amrex::Real>;
using HBM_Inner_Vector_Int = HBM_Vector<int>;

// 外层 vector：存放 "HBM vector" 的 HBM vector
using HBM_Nested_Vector_Real = HBM_Vector<HBM_Inner_Vector_Real>;
using HBM_Nested_Vector_Int = HBM_Vector<HBM_Inner_Vector_Int>;

// ====================================================================
// 步骤三：新增的、安全的HBM内存收缩函数模板
// ====================================================================

/**
 * @brief 安全地收缩任何std::vector的容量以匹配其大小，释放未使用的内存。
 * @tparam T vector中存储的元素类型。
 * @tparam Alloc vector使用的分配器类型。
 * @param vec 要进行收缩操作的vector的引用。
 */
 template <typename T>
 void shrink_vector_capacity(HBM_Vector<T>& vec) {
     if (vec.capacity() <= 2* vec.size()) {
         return; // 容量已最优，无需操作
     }
 
     // 1. 创建一个临时的vector，关键是使用与原vector完全相同的分配器。
     //    这样可以确保新内存也在正确的设备上（例如HBM）分配。
     HBM_Vector<T> new_vec;
    //  std::vector<T, Alloc> new_vec(vec.get_allocator());
 
     // 2. 在新vector中一次性预留恰好足够的内存。
     if (vec.size() > 0) {
         new_vec.reserve(vec.size());
         
         // 3. 高效地将所有元素从旧vector“移动”到新vector。
         std::memcpy(new_vec.data(),
                        vec.data(),
                        vec.size() * sizeof(T));
        //  new_vec.assign(std::make_move_iterator(vec.begin()),
        //                 std::make_move_iterator(vec.end()));
     }
     
     // 4. 交换vec和new_vec的内部状态。操作后，vec将持有紧凑的内存，
     //    而new_vec将持有旧的、超额的内存。
     vec.swap(new_vec);
 
     // 5. 此函数结束时，new_vec析构，其持有的旧内存被自动、安全地释放。
 }


#endif // HBM_ALLOCATOR_H_