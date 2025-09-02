#ifndef AMREX_PARTICLETILE_H_
#define AMREX_PARTICLETILE_H_
#include <AMReX_Config.H>

#include <AMReX_Extension.H>
#include <AMReX_Particle.H>
#include <AMReX_ArrayOfStructs.H>
#include <AMReX_StructOfArrays.H>
#include <AMReX_Vector.H>
#include <AMReX_REAL.H>
#include <AMReX_RealVect.H>

#include <array>
#include <type_traits>
// #include <../HBM_Allocator.H>
#include "/pacific_ext/SYSU/sysu04/Test_install_warpx_hw/WarpX-development/Source/HBM_Allocator.H"

namespace amrex {

// Forward Declaration
template <int NArrayReal, int NArrayInt>
struct ConstSoAParticle;
template <int NArrayReal, int NArrayInt>
struct SoAParticle;

template <typename T_ParticleType, int NArrayReal, int NArrayInt>
struct ConstParticleTileData;

template <typename T_ParticleType, int NArrayReal, int NArrayInt>
struct ParticleTileData
{
    static constexpr int NAR = NArrayReal;
    static constexpr int NAI = NArrayInt;

    using ParticleType = T_ParticleType;
    using ParticleRefType = T_ParticleType&;
    using Self = ParticleTileData<ParticleType, NAR, NAI>;

    static constexpr int NStructReal = ParticleType::NReal;
    static constexpr int NStructInt = ParticleType::NInt;

    using SuperParticleType = Particle<NStructReal+NAR, NStructInt+NAI>;

    static constexpr bool is_particle_tile_data = true;

    Long m_size;

    using AOS_PTR = std::conditional_t<T_ParticleType::is_soa_particle,
                                       void * AMREX_RESTRICT, ParticleType * AMREX_RESTRICT>;
    AOS_PTR m_aos;

    uint64_t* m_idcpu;
    GpuArray<ParticleReal*, NAR> m_rdata;
    GpuArray<int*, NAI> m_idata;

    int m_num_runtime_real;
    int m_num_runtime_int;
    ParticleReal* AMREX_RESTRICT * AMREX_RESTRICT m_runtime_rdata;
    int* AMREX_RESTRICT * AMREX_RESTRICT m_runtime_idata;

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    ParticleReal& pos (const int dir, const int index) const &
    {
        if constexpr(!ParticleType::is_soa_particle) {
            return this->m_aos[index].pos(dir);
        } else {
            return this->m_rdata[dir][index];
        }
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    decltype(auto) id (const int index) const &
    {
        if constexpr(!ParticleType::is_soa_particle) {
            return this->m_aos[index].id();
        } else {
            return ParticleIDWrapper(this->m_idcpu[index]);
        }
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    decltype(auto) cpu (const int index) const &
    {
        if constexpr(!ParticleType::is_soa_particle) {
            return this->m_aos[index].cpu();
        } else {
            return ParticleCPUWrapper(this->m_idcpu[index]);
        }
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    decltype(auto) idcpu (const int index) const &
    {
        if constexpr(ParticleType::is_soa_particle) {
            return this->m_idcpu[index];
        } else {
            amrex::Abort("not implemented");
        }
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    ParticleReal * rdata (const int attribute_index) const
    {
        return this->m_rdata[attribute_index];
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int * idata (const int attribute_index) const
    {
        return this->m_idata[attribute_index];
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    decltype(auto) operator[] (const int index) const
    {
        if constexpr (!ParticleType::is_soa_particle) {
            return m_aos[index];
        } else {
            return SoAParticle<NAR, NAI>(*this, index);
        }
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void packParticleData (char* buffer, int src_index, std::size_t dst_offset,
                           const int* comm_real, const int * comm_int) const noexcept
    {
        AMREX_ASSERT(src_index < m_size);
        auto* dst = buffer + dst_offset;
        if constexpr (!ParticleType::is_soa_particle) {
            memcpy(dst, m_aos + src_index, sizeof(ParticleType));
            dst += sizeof(ParticleType);
        } else {
            memcpy(dst, m_idcpu + src_index, sizeof(uint64_t));
            dst += sizeof(uint64_t);
        }
        int array_start_index  = AMREX_SPACEDIM + NStructReal;
        for (int i = 0; i < NAR; ++i)
        {
            if (comm_real[array_start_index + i])
            {
                memcpy(dst, m_rdata[i] + src_index, sizeof(ParticleReal));
                dst += sizeof(ParticleReal);
            }
        }
        int runtime_start_index  = AMREX_SPACEDIM + NStructReal + NAR;
        for (int i = 0; i < m_num_runtime_real; ++i)
        {
            if (comm_real[runtime_start_index + i])
            {
                memcpy(dst, m_runtime_rdata[i] + src_index, sizeof(ParticleReal));
                dst += sizeof(ParticleReal);
            }
        }
        array_start_index  = 2 + NStructInt;
        for (int i = 0; i < NAI; ++i)
        {
            if (comm_int[array_start_index + i])
            {
                memcpy(dst, m_idata[i] + src_index, sizeof(int));
                dst += sizeof(int);
            }
        }
        runtime_start_index  = 2 + NStructInt + NAI;
        for (int i = 0; i < m_num_runtime_int; ++i)
        {
            if (comm_int[runtime_start_index + i])
            {
                memcpy(dst, m_runtime_idata[i] + src_index, sizeof(int));
                dst += sizeof(int);
            }
        }
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void unpackParticleData (const char* buffer, Long src_offset, int dst_index,
                             const int* comm_real, const int* comm_int) const noexcept
    {
        AMREX_ASSERT(dst_index < m_size);
        const auto* src = buffer + src_offset;
        if constexpr (!ParticleType::is_soa_particle) {
            memcpy(m_aos + dst_index, src, sizeof(ParticleType));
            src += sizeof(ParticleType);
        } else {
            memcpy(m_idcpu + dst_index, src, sizeof(uint64_t));
            src += sizeof(uint64_t);
        }
        int array_start_index  = AMREX_SPACEDIM + NStructReal;
        for (int i = 0; i < NAR; ++i)
        {
            if (comm_real[array_start_index + i])
            {
                memcpy(m_rdata[i] + dst_index, src, sizeof(ParticleReal));
                src += sizeof(ParticleReal);
            }
        }
        int runtime_start_index  = AMREX_SPACEDIM + NStructReal + NAR;
        for (int i = 0; i < m_num_runtime_real; ++i)
        {
            if (comm_real[runtime_start_index + i])
            {
                memcpy(m_runtime_rdata[i] + dst_index, src, sizeof(ParticleReal));
                src += sizeof(ParticleReal);
            }
        }
        array_start_index  = 2 + NStructInt;
        for (int i = 0; i < NAI; ++i)
        {
            if (comm_int[array_start_index + i])
            {
                memcpy(m_idata[i] + dst_index, src, sizeof(int));
                src += sizeof(int);
            }
        }
        runtime_start_index  = 2 + NStructInt + NAI;
        for (int i = 0; i < m_num_runtime_int; ++i)
        {
            if (comm_int[runtime_start_index + i])
            {
                memcpy(m_runtime_idata[i] + dst_index, src, sizeof(int));
                src += sizeof(int);
            }
        }
    }

    template <typename T = ParticleType, std::enable_if_t<!T::is_soa_particle, int> = 0>
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    SuperParticleType getSuperParticle (int index) const noexcept
    {
        AMREX_ASSERT(index < m_size);
        SuperParticleType sp;
        for (int i = 0; i < AMREX_SPACEDIM; ++i) {
            sp.pos(i) = m_aos[index].pos(i);
        }
        for (int i = 0; i < NStructReal; ++i) {
            sp.rdata(i) = m_aos[index].rdata(i);
        }
        for (int i = 0; i < NAR; ++i) {
            sp.rdata(NStructReal+i) = m_rdata[i][index];
        }
        sp.id() = m_aos[index].id();
        sp.cpu() = m_aos[index].cpu();
        for (int i = 0; i < NStructInt; ++i) {
            sp.idata(i) = m_aos[index].idata(i);
        }
        for (int i = 0; i < NAI; ++i) {
            sp.idata(NStructInt+i) = m_idata[i][index];
        }
        return sp;
    }

    template <typename T = ParticleType, std::enable_if_t<T::is_soa_particle, int> = 0>
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    SuperParticleType getSuperParticle (int index) const noexcept
    {
        AMREX_ASSERT(index < m_size);
        SuperParticleType sp;
        sp.m_idcpu = m_idcpu[index];
        for (int i = 0; i < AMREX_SPACEDIM; ++i) {sp.pos(i) = m_rdata[i][index];}
        for (int i = 0; i < NAR; ++i) {
            sp.rdata(i) = m_rdata[i][index];
        }
        for (int i = 0; i < NAI; ++i) {
            sp.idata(i) = m_idata[i][index];
        }
        return sp;
    }

    template <typename T = ParticleType, std::enable_if_t<!T::is_soa_particle, int> = 0>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void setSuperParticle (const SuperParticleType& sp, int index) const noexcept
    {
        for (int i = 0; i < AMREX_SPACEDIM; ++i) {
            m_aos[index].pos(i) = sp.pos(i);
        }
        for (int i = 0; i < NStructReal; ++i) {
            m_aos[index].rdata(i) = sp.rdata(i);
        }
        for (int i = 0; i < NAR; ++i) {
            m_rdata[i][index] = sp.rdata(NStructReal+i);
        }
        m_aos[index].id() = sp.id();
        m_aos[index].cpu() = sp.cpu();
        for (int i = 0; i < NStructInt; ++i) {
            m_aos[index].idata(i) = sp.idata(i);
        }
        for (int i = 0; i < NAI; ++i) {
            m_idata[i][index] = sp.idata(NStructInt+i);
        }
    }

    template <typename T = ParticleType, std::enable_if_t<T::is_soa_particle, int> = 0>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void setSuperParticle (const SuperParticleType& sp, int index) const noexcept
    {
        m_idcpu[index] = sp.m_idcpu;
        for (int i = 0; i < NAR; ++i) {
            m_rdata[i][index] = sp.rdata(i);
        }
        for (int i = 0; i < NAI; ++i) {
            m_idata[i][index] = sp.idata(i);
        }
    }
};

// SOA Particle Structure
template <int T_NArrayReal, int T_NArrayInt>
struct ConstSoAParticle : SoAParticleBase
{
    static constexpr int NArrayReal = T_NArrayReal;
    static constexpr int NArrayInt = T_NArrayInt;
    using StorageParticleType = SoAParticleBase;
    using ConstPTD = ConstParticleTileData<SoAParticleBase, NArrayReal, NArrayInt>;
    static constexpr bool is_soa_particle = true;
    static constexpr bool is_constsoa_particle = true;

    using RealType = ParticleReal;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    ConstSoAParticle (ConstPTD const& ptd, long i) : // Note: should this be int instead?
        m_constparticle_tile_data(ptd), m_index(int(i))
    {
    }

    //static Long the_next_id;

    //functions to get id and cpu in the SOA data

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    ConstParticleCPUWrapper cpu () const { return this->m_constparticle_tile_data.m_idcpu[m_index]; }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    ConstParticleIDWrapper id () const { return this->m_constparticle_tile_data.m_idcpu[m_index]; }

    //functions to get positions of the particle in the SOA data

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    RealVect pos () const & {return RealVect(AMREX_D_DECL(this->m_constparticle_tile_data.m_rdata[0][m_index], this->m_constparticle_tile_data.m_rdata[1][m_index], this->m_constparticle_tile_data.m_rdata[2][m_index]));}

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    const RealType&  pos (int position_index) const &
    {
        AMREX_ASSERT(position_index < AMREX_SPACEDIM);
        return this->m_constparticle_tile_data.m_rdata[position_index][m_index];
    }

    static Long NextID ();

    /**
    * \brief This version can only be used inside omp critical.
    */
    static Long UnprotectedNextID ();

    /**
    * \brief Reset on restart.
    *
    * \param nextid
    */
    static void NextID (Long nextid);

    private :

    static_assert(std::is_trivially_copyable<ParticleTileData<SoAParticleBase, NArrayReal, NArrayInt>>(), "ParticleTileData is not trivially copyable");

    ConstPTD m_constparticle_tile_data;
    int m_index;
};

template <int T_NArrayReal, int T_NArrayInt>
struct SoAParticle : SoAParticleBase
{
    static constexpr int NArrayReal = T_NArrayReal;
    static constexpr int NArrayInt = T_NArrayInt;
    using StorageParticleType = SoAParticleBase;
    using PTD = ParticleTileData<SoAParticleBase, NArrayReal, NArrayInt>;
    static constexpr bool is_soa_particle = true;
    static constexpr bool is_constsoa_particle = false;

    using ConstType = ConstSoAParticle<T_NArrayReal, T_NArrayInt>;
    using RealType = ParticleReal;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    SoAParticle (PTD const& ptd, long i) : // Note: should this be int instead?
        m_particle_tile_data(ptd), m_index(int(i))
    {
    }

    static Long the_next_id;

    //functions to get id and cpu in the SOA data

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    ParticleCPUWrapper cpu () & { return this->m_particle_tile_data.m_idcpu[m_index]; }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    ParticleIDWrapper id () & { return this->m_particle_tile_data.m_idcpu[m_index]; }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    uint64_t& idcpu () & { return this->m_particle_tile_data.m_idcpu[m_index]; }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    ConstParticleCPUWrapper cpu () const & { return this->m_particle_tile_data.m_idcpu[m_index]; }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    ConstParticleIDWrapper id () const & { return this->m_particle_tile_data.m_idcpu[m_index]; }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    const uint64_t& idcpu () const & { return this->m_particle_tile_data.m_idcpu[m_index]; }

    //functions to get positions of the particle in the SOA data

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    RealVect pos () const & {return RealVect(AMREX_D_DECL(this->m_particle_tile_data.m_rdata[0][m_index], this->m_particle_tile_data.m_rdata[1][m_index], this->m_particle_tile_data.m_rdata[2][m_index]));}

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    RealType& pos (int position_index) &
    {
        AMREX_ASSERT(position_index < AMREX_SPACEDIM);
        return this->m_particle_tile_data.m_rdata[position_index][m_index];
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    RealType pos (int position_index) const &
    {
        AMREX_ASSERT(position_index < AMREX_SPACEDIM);
        return this->m_particle_tile_data.m_rdata[position_index][m_index];
    }

    static Long NextID ();

    /**
    * \brief This version can only be used inside omp critical.
    */
    static Long UnprotectedNextID ();

    /**
    * \brief Reset on restart.
    *
    * \param nextid
    */
    static void NextID (Long nextid);

private :

    static_assert(std::is_trivially_copyable<ParticleTileData<SoAParticleBase, NArrayReal, NArrayInt>>(), "ParticleTileData is not trivially copyable");

    PTD m_particle_tile_data;
    int m_index;
};

//template <int NArrayReal, int NArrayInt> Long ConstSoAParticle<NArrayReal, NArrayInt>::the_next_id = 1;
template <int NArrayReal, int NArrayInt> Long SoAParticle<NArrayReal, NArrayInt>::the_next_id = 1;

template <int NArrayReal, int NArrayInt>
Long
SoAParticle<NArrayReal, NArrayInt>::NextID ()
{
    Long next;
// we should be able to test on _OPENMP < 201107 for capture (version 3.1)
// but we must work around a bug in gcc < 4.9
#if defined(AMREX_USE_OMP) && defined(_OPENMP) && _OPENMP < 201307
#pragma omp critical (amrex_particle_nextid)
#elif defined(AMREX_USE_OMP)
#pragma omp atomic capture
#endif
    next = the_next_id++;

    if (next > LongParticleIds::LastParticleID) {
        amrex::Abort("SoAParticle<NArrayReal, NArrayInt>::NextID() -- too many particles");
    }

    return next;
}

template <int NArrayReal, int NArrayInt>
Long
SoAParticle<NArrayReal, NArrayInt>::UnprotectedNextID ()
{
    Long next = the_next_id++;
    if (next > LongParticleIds::LastParticleID) {
        amrex::Abort("SoAParticle<NArrayReal, NArrayInt>::NextID() -- too many particles");
    }
    return next;
}

template <int NArrayReal, int NArrayInt>
void
SoAParticle<NArrayReal, NArrayInt>::NextID (Long nextid)
{
    the_next_id = nextid;
}

template <typename T_ParticleType, int NArrayReal, int NArrayInt>
struct ConstParticleTileData
{
    static constexpr int NAR = NArrayReal;
    static constexpr int NAI = NArrayInt;
    using ParticleType = T_ParticleType;
    using ParticleRefType = T_ParticleType const&;

    static constexpr int NStructReal = ParticleType::NReal;
    static constexpr int NStructInt = ParticleType::NInt;

    using SuperParticleType = Particle<NStructReal+NArrayReal, NStructInt+NArrayInt>;

    static constexpr bool is_particle_tile_data = true;

    Long m_size;

    using AOS_PTR = std::conditional_t<T_ParticleType::is_soa_particle,
                                       void const * AMREX_RESTRICT, ParticleType const * AMREX_RESTRICT>;
    AOS_PTR m_aos;

    const uint64_t* m_idcpu;
    GpuArray<const ParticleReal*, NArrayReal> m_rdata;
    GpuArray<const int*, NArrayInt > m_idata;

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    const ParticleReal& pos (const int dir, const int index) const &
    {
        if constexpr(!ParticleType::is_soa_particle) {
            return this->m_aos[index].pos(dir);
        } else {
            return this->m_rdata[dir][index];
        }
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    decltype(auto) id (const int index) const &
    {
        if constexpr(!ParticleType::is_soa_particle) {
            return this->m_aos[index].id();
        } else {
            return ConstParticleIDWrapper(this->m_idcpu[index]);
        }
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    decltype(auto) cpu (const int index) const &
    {
        if constexpr(!ParticleType::is_soa_particle) {
            return this->m_aos[index].cpu();
        } else {
            return ConstParticleCPUWrapper(this->m_idcpu[index]);
        }
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    decltype(auto) idcpu (const int index) const &
    {
        if constexpr(ParticleType::is_soa_particle) {
            return this->m_idcpu[index];
        } else {
            amrex::Abort("not implemented");
        }
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    const ParticleReal * rdata (const int attribute_index) const
    {
        return this->m_rdata[attribute_index];
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    const int * idata (const int attribute_index) const
    {
        return this->m_idata[attribute_index];
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    decltype(auto) operator[] (const int index) const
    {
        if constexpr (!ParticleType::is_soa_particle) {
            return m_aos[index];
        } else {
            return ConstSoAParticle<NAR, NAI>(*this, index);
        }
    }

    int m_num_runtime_real;
    int m_num_runtime_int;
    const ParticleReal* AMREX_RESTRICT * AMREX_RESTRICT m_runtime_rdata;
    const int* AMREX_RESTRICT * AMREX_RESTRICT m_runtime_idata;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void packParticleData(char* buffer, int src_index, Long dst_offset,
                          const int* comm_real, const int * comm_int) const noexcept
    {
        AMREX_ASSERT(src_index < m_size);
        auto* dst = buffer + dst_offset;
        if constexpr (!ParticleType::is_soa_particle) {
            memcpy(dst, m_aos + src_index, sizeof(ParticleType));
            dst += sizeof(ParticleType);
        } else {
            memcpy(dst, m_idcpu + src_index, sizeof(uint64_t));
            dst += sizeof(uint64_t);
        }
        int array_start_index  = AMREX_SPACEDIM + NStructReal;
        for (int i = 0; i < NArrayReal; ++i)
        {
            if (comm_real[array_start_index + i])
            {
                memcpy(dst, m_rdata[i] + src_index, sizeof(ParticleReal));
                dst += sizeof(ParticleReal);
            }
        }
        int runtime_start_index  = AMREX_SPACEDIM + NStructReal + NArrayReal;
        for (int i = 0; i < m_num_runtime_real; ++i)
        {
            if (comm_real[runtime_start_index + i])
            {
                memcpy(dst, m_runtime_rdata[i] + src_index, sizeof(ParticleReal));
                dst += sizeof(ParticleReal);
            }
        }
        array_start_index  = 2 + NStructInt;
        for (int i = 0; i < NArrayInt; ++i)
        {
            if (comm_int[array_start_index + i])
            {
                memcpy(dst, m_idata[i] + src_index, sizeof(int));
                dst += sizeof(int);
            }
        }
        runtime_start_index  = 2 + NStructInt + NArrayInt;
        for (int i = 0; i < m_num_runtime_int; ++i)
        {
            if (comm_int[runtime_start_index + i])
            {
                memcpy(dst, m_runtime_idata[i] + src_index, sizeof(int));
                dst += sizeof(int);
            }
        }
    }

    template <typename T = ParticleType, std::enable_if_t<!T::is_soa_particle, int> = 0>
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    SuperParticleType getSuperParticle (int index) const noexcept
    {
        AMREX_ASSERT(index < m_size);
        SuperParticleType sp;
        for (int i = 0; i < AMREX_SPACEDIM; ++i) {
            sp.pos(i) = m_aos[index].pos(i);
        }
        for (int i = 0; i < NStructReal; ++i) {
            sp.rdata(i) = m_aos[index].rdata(i);
        }
        if constexpr(NArrayReal > 0) {
            for (int i = 0; i < NArrayReal; ++i) {
                sp.rdata(NStructReal+i) = m_rdata[i][index];
            }
        }
        sp.id() = m_aos[index].id();
        sp.cpu() = m_aos[index].cpu();
        for (int i = 0; i < NStructInt; ++i) {
            sp.idata(i) = m_aos[index].idata(i);
        }
        if constexpr(NArrayInt > 0) {
            for (int i = 0; i < NArrayInt; ++i) {
                sp.idata(NStructInt+i) = m_idata[i][index];
            }
        }
        return sp;
    }

    template <typename T = ParticleType, std::enable_if_t<T::is_soa_particle, int> = 0>
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    SuperParticleType getSuperParticle (int index) const noexcept
    {
        AMREX_ASSERT(index < m_size);
        SuperParticleType sp;
        for (int i = 0; i < AMREX_SPACEDIM; ++i) {sp.pos(i) = m_rdata[i][index];}
        sp.m_idcpu = m_idcpu[index];
        for (int i = 0; i < NAR; ++i) {
            sp.rdata(i) = m_rdata[i][index];
        }
        for (int i = 0; i < NAI; ++i) {
            sp.idata(i) = m_idata[i][index];
        }
        return sp;
    }
};

struct ThisParticleTileHasNoParticleVector {};

struct ThisParticleTileHasNoAoS {
    using ParticleVector = ThisParticleTileHasNoParticleVector;
};

template <typename T_ParticleType, int NArrayReal, int NArrayInt,
          template<class> class Allocator=DefaultAllocator>
struct ParticleTile
{
    template <typename T>
    using AllocatorType = Allocator<T>;

    using ParticleType = T_ParticleType;
    static constexpr int NAR = NArrayReal;
    static constexpr int NAI = NArrayInt;
    using RealType = typename ParticleType::RealType;

    static constexpr int NStructReal = ParticleType::NReal;
    static constexpr int NStructInt = ParticleType::NInt;

    using SuperParticleType = Particle<NStructReal + NArrayReal, NStructInt + NArrayInt>;

    using AoS = std::conditional_t<
        ParticleType::is_soa_particle,
        ThisParticleTileHasNoAoS,
        ArrayOfStructs<ParticleType, Allocator>>;
    //using ParticleVector = typename AoS::ParticleVector;

    using SoA = std::conditional_t<
        ParticleType::is_soa_particle,
        StructOfArrays<NArrayReal, NArrayInt, Allocator, true>,
        StructOfArrays<NArrayReal, NArrayInt, Allocator, false>>;
    using RealVector = typename SoA::RealVector;
    using IntVector = typename SoA::IntVector;
    using StorageParticleType = typename ParticleType::StorageParticleType;

    using ParticleTileDataType = ParticleTileData<StorageParticleType, NArrayReal, NArrayInt>;
    using ConstParticleTileDataType = ConstParticleTileData<StorageParticleType, NArrayReal, NArrayInt>;

    ParticleTile () = default;

#ifndef _WIN32  // workaround windows compiler bug
    ~ParticleTile () = default;

    ParticleTile (ParticleTile const&) = delete;
    ParticleTile (ParticleTile &&) noexcept = default;

    ParticleTile& operator= (ParticleTile const&) = delete;
    ParticleTile& operator= (ParticleTile &&) noexcept = default;
#endif

    void define (int a_num_runtime_real, int a_num_runtime_int)
    {
        m_defined = true;
        GetStructOfArrays().define(a_num_runtime_real, a_num_runtime_int);
        m_runtime_r_ptrs.resize(a_num_runtime_real);
        m_runtime_i_ptrs.resize(a_num_runtime_int);
        m_runtime_r_cptrs.resize(a_num_runtime_real);
        m_runtime_i_cptrs.resize(a_num_runtime_int);
    }

    // Get id data
    decltype(auto) id (int index) & {
        if constexpr (!ParticleType::is_soa_particle) {
            return m_aos_tile[index].id();
        } else {
            return ParticleIDWrapper(m_soa_tile.GetIdCPUData()[index]);
        }
    }

    // const
    decltype(auto) id (int index) const & {
        if constexpr (!ParticleType::is_soa_particle) {
            return m_aos_tile[index].id();
        } else {
            return ConstParticleIDWrapper(m_soa_tile.GetIdCPUData()[index]);
        }
    }

    // Get cpu data
    decltype(auto) cpu (int index) & {
        if constexpr (!ParticleType::is_soa_particle) {
            return m_aos_tile[index].cpu();
        } else {
            return ParticleCPUWrapper(m_soa_tile.GetIdCPUData()[index]);
        }
    }

    // const
    decltype(auto) cpu (int index) const & {
        if constexpr (!ParticleType::is_soa_particle) {
            return m_aos_tile[index].cpu();
        } else {
            return ConstParticleCPUWrapper(m_soa_tile.GetIdCPUData()[index]);
        }
    }

    // Get positions data
    RealType& pos (int index, int position_index) & {
        if constexpr (!ParticleType::is_soa_particle) {
            return m_aos_tile[index].pos(position_index);
        } else {
            static_assert(NArrayReal == ParticleType::PTD::NAR, "ParticleTile mismatch in R");
            static_assert(NArrayInt == ParticleType::PTD::NAI, "ParticleTile mismatch in I");
            static_assert(0 == ParticleType::StorageParticleType::NReal, "ParticleTile 2 mismatch in R");
            static_assert(0 == ParticleType::StorageParticleType::NInt, "ParticleTile 2 mismatch in I");

            return m_soa_tile.GetRealData(position_index)[index];
        }
    }

    // const
    RealType  pos (int index, int position_index) const &
    {
        if constexpr (!ParticleType::is_soa_particle) {
            return m_aos_tile[index].pos(position_index);
        } else {
            return m_soa_tile.GetRealData(position_index)[index];
        }
    }

    AoS&       GetArrayOfStructs ()       { return m_aos_tile; }
    const AoS& GetArrayOfStructs () const { return m_aos_tile; }

    SoA&       GetStructOfArrays ()       { return m_soa_tile; }
    const SoA& GetStructOfArrays () const { return m_soa_tile; }

    bool empty () const { return size() == 0; }

    /**
    * \brief Returns the total number of particles (real and neighbor)
    *
    */
    std::size_t size () const
    {
        if constexpr (!ParticleType::is_soa_particle) {
            return m_aos_tile.size();
        } else {
            return m_soa_tile.size();
        }
    }

    /**
    * \brief Returns the number of real particles (excluding neighbors)
    *
    */
    int numParticles () const
    {
        if constexpr (!ParticleType::is_soa_particle) {
            return m_aos_tile.numParticles();
        } else {
            return m_soa_tile.numParticles();
        }
    }

    /**
    * \brief Returns the number of real particles (excluding neighbors)
    *
    */
    int numRealParticles () const
    {
        if constexpr (!ParticleType::is_soa_particle) {
            return m_aos_tile.numRealParticles();
        } else {
            return m_soa_tile.numRealParticles();
        }
    }

    /**
    * \brief Returns the number of neighbor particles (excluding reals)
    *
    */
    int numNeighborParticles () const
    {
        if constexpr (!ParticleType::is_soa_particle) {
            return m_aos_tile.numNeighborParticles();
        } else {
            return m_soa_tile.numNeighborParticles();
        }
    }

    /**
    * \brief Returns the total number of particles, real and neighbor
    *
    */
    int numTotalParticles () const
    {
        if constexpr (!ParticleType::is_soa_particle) {
            return m_aos_tile.numTotalParticles();
        } else {
            return m_soa_tile.numTotalParticles();
        }
    }

    void setNumNeighbors (int num_neighbors)
    {
        if constexpr(!ParticleType::is_soa_particle) {
            m_aos_tile.setNumNeighbors(num_neighbors);
        }
        m_soa_tile.setNumNeighbors(num_neighbors);
    }

    int getNumNeighbors () const
    {
        if constexpr (!ParticleType::is_soa_particle) {
            AMREX_ASSERT( m_soa_tile.getNumNeighbors() == m_aos_tile.getNumNeighbors() );
            return m_aos_tile.getNumNeighbors();
        } else {
            return m_soa_tile.getNumNeighbors();
        }
    }

    void resize (std::size_t count)
    {
        if constexpr (!ParticleType::is_soa_particle) {
            m_aos_tile.resize(count);
        }
        m_soa_tile.resize(count);
    }

    ///
    /// Add one particle to this tile.
    ///
    template <typename T = ParticleType, std::enable_if_t<!T::is_soa_particle, int> = 0>
    void push_back (const ParticleType& p) { m_aos_tile().push_back(p); }

    ///
    /// Add one particle to this tile.
    ///
    template < int NR = NArrayReal, int NI = NArrayInt,
               std::enable_if_t<NR != 0 || NI != 0, int> foo = 0>
    void push_back (const SuperParticleType& sp)
    {
        auto np = numParticles();

        if constexpr (!ParticleType::is_soa_particle) {
            m_aos_tile.resize(np+1);
            for (int i = 0; i < AMREX_SPACEDIM; ++i) {
                m_aos_tile[np].pos(i) = sp.pos(i);
            }
            for (int i = 0; i < NStructReal; ++i) {
                m_aos_tile[np].rdata(i) = sp.rdata(i);
            }
            m_aos_tile[np].id() = sp.id();
            m_aos_tile[np].cpu() = sp.cpu();
            for (int i = 0; i < NStructInt; ++i) {
                m_aos_tile[np].idata(i) = sp.idata(i);
            }
        }

        m_soa_tile.resize(np+1);
        if constexpr (ParticleType::is_soa_particle) {
            m_soa_tile.GetIdCPUData()[np] = sp.m_idcpu;
        }
        auto& arr_rdata = m_soa_tile.GetRealData();
        auto& arr_idata = m_soa_tile.GetIntData();
        for (int i = 0; i < NArrayReal; ++i) {
            arr_rdata[i][np] = sp.rdata(NStructReal+i);
        }
        for (int i = 0; i < NArrayInt; ++i) {
            arr_idata[i][np] = sp.idata(NStructInt+i);
        }
    }

    ///
    /// Add a Real value to the struct-of-arrays at index comp.
    /// This sets the data for one particle.
    ///
    void push_back_real (int comp, ParticleReal v) {
        m_soa_tile.GetRealData(comp).push_back(v);
    }

    ///
    /// Add Real values to the struct-of-arrays, for all comps at once.
    /// This sets the data for one particle.
    ///
    void push_back_real (const std::array<ParticleReal, NArrayReal>& v) {
        for (int i = 0; i < NArrayReal; ++i) {
            m_soa_tile.GetRealData(i).push_back(v[i]);
        }
    }

    ///
    /// Add a range of Real values to the struct-of-arrays for the given comp.
    /// This sets the data for several particles at once.
    ///
    void push_back_real (int comp, const ParticleReal* beg, const ParticleReal* end) {
        auto it = m_soa_tile.GetRealData(comp).end();
        m_soa_tile.GetRealData(comp).insert(it, beg, end);
    }

    ///
    /// Add a range of Real values to the struct-of-arrays for the given comp.
    /// This sets the data for several particles at once.
    ///
    void push_back_real (int comp, amrex::Vector<amrex::ParticleReal>::const_iterator beg, amrex::Vector<amrex::ParticleReal>::const_iterator end) {
        push_back_real(comp, &(*beg), &(*end));
    }

    ///
    /// Add a range of Real values to the struct-of-arrays for the given comp.
    /// This sets the data for several particles at once.
    ///
    void push_back_real (int comp, amrex::Vector<amrex::ParticleReal> const & vec) {
        push_back_real(comp, vec.cbegin(), vec.cend());
    }

    ///
    /// Add npar copies of the Real value v to the struct-of-arrays for the given comp.
    /// This sets the data for several particles at once.
    ///
    void push_back_real (int comp, std::size_t npar, ParticleReal v) {
        auto new_size = m_soa_tile.GetRealData(comp).size() + npar;
        m_soa_tile.GetRealData(comp).resize(new_size, v);
    }

    ///
    /// Add an int value to the struct-of-arrays at index comp.
    /// This sets the data for one particle.
    ///
    void push_back_int (int comp, int v) {
        m_soa_tile.GetIntData(comp).push_back(v);
    }

    ///
    /// Add int values to the struct-of-arrays, for all comps at once.
    /// This sets the data for one particle.
    ///
    void push_back_int (const std::array<int, NArrayInt>& v) {
        for (int i = 0; i < NArrayInt; ++i) {
            m_soa_tile.GetIntData(i).push_back(v[i]);
        }
    }

    ///
    /// Add a range of int values to the struct-of-arrays for the given comp.
    /// This sets the data for several particles at once.
    ///
    void push_back_int (int comp, const int* beg, const int* end) {
        auto it = m_soa_tile.GetIntData(comp).end();
        m_soa_tile.GetIntData(comp).insert(it, beg, end);
    }

    ///
    /// Add a range of int values to the struct-of-arrays for the given comp.
    /// This sets the data for several particles at once.
    ///
    void push_back_int (int comp, amrex::Vector<int>::const_iterator beg, amrex::Vector<int>::const_iterator end) {
        push_back_int(comp, &(*beg), &(*end));
    }

    ///
    /// Add a range of int values to the struct-of-arrays for the given comp.
    /// This sets the data for several particles at once.
    ///
    void push_back_int (int comp, amrex::Vector<int> const & vec) {
        push_back_int(comp, vec.cbegin(), vec.cend());
    }

    ///
    /// Add npar copies of the int value v to the struct-of-arrays for the given comp.
    /// This sets the data for several particles at once.
    ///
    void push_back_int (int comp, std::size_t npar, int v) {
        auto new_size = m_soa_tile.GetIntData(comp).size() + npar;
        m_soa_tile.GetIntData(comp).resize(new_size, v);
    }

    int NumRealComps () const noexcept { return m_soa_tile.NumRealComps(); }

    int NumIntComps () const noexcept { return m_soa_tile.NumIntComps(); }

    int NumRuntimeRealComps () const noexcept { return m_runtime_r_ptrs.size(); }

    int NumRuntimeIntComps () const noexcept { return m_runtime_i_ptrs.size(); }

    void shrink_to_fit ()
    {
        if constexpr (ParticleType::is_soa_particle) {
            GetStructOfArrays().GetIdCPUData().shrink_to_fit();
        } else {
            m_aos_tile().shrink_to_fit();
        }
        for (int j = 0; j < NumRealComps(); ++j)
        {
            auto& rdata = GetStructOfArrays().GetRealData(j);
            rdata.shrink_to_fit();
        }

        for (int j = 0; j < NumIntComps(); ++j)
        {
            auto& idata = GetStructOfArrays().GetIntData(j);
            idata.shrink_to_fit();
        }
    }

    Long capacity () const
    {
        Long nbytes = 0;
        if constexpr (ParticleType::is_soa_particle) {
            nbytes += GetStructOfArrays().GetIdCPUData().capacity() * sizeof(uint64_t);
        } else {
            nbytes += m_aos_tile().capacity() * sizeof(ParticleType);
        }
        for (int j = 0; j < NumRealComps(); ++j)
        {
            auto& rdata = GetStructOfArrays().GetRealData(j);
            nbytes += rdata.capacity() * sizeof(ParticleReal);
        }

        for (int j = 0; j < NumIntComps(); ++j)
        {
            auto& idata = GetStructOfArrays().GetIntData(j);
            nbytes += idata.capacity()*sizeof(int);
        }
        return nbytes;
    }

    void swap (ParticleTile<ParticleType, NArrayReal, NArrayInt, Allocator>& other) noexcept
    {
        if constexpr (ParticleType::is_soa_particle) {
            GetStructOfArrays().GetIdCPUData().swap(other.GetStructOfArrays().GetIdCPUData());
        } else {
            m_aos_tile().swap(other.GetArrayOfStructs()());
        }
        for (int j = 0; j < NumRealComps(); ++j)
        {
            auto& rdata = GetStructOfArrays().GetRealData(j);
            rdata.swap(other.GetStructOfArrays().GetRealData(j));
        }

        for (int j = 0; j < NumIntComps(); ++j)
        {
            auto& idata = GetStructOfArrays().GetIntData(j);
            idata.swap(other.GetStructOfArrays().GetIntData(j));
        }
    }

    ParticleTileDataType getParticleTileData ()
    {
        m_runtime_r_ptrs.resize(m_soa_tile.NumRealComps() - NArrayReal);
        m_runtime_i_ptrs.resize(m_soa_tile.NumIntComps() - NArrayInt);
#ifdef AMREX_USE_GPU
        bool copy_real = false;
        m_h_runtime_r_ptrs.resize(m_soa_tile.NumRealComps() - NArrayReal, nullptr);
        for (std::size_t i = 0; i < m_h_runtime_r_ptrs.size(); ++i) {
            if (m_h_runtime_r_ptrs[i] != m_soa_tile.GetRealData(i + NArrayReal).dataPtr()) {
                m_h_runtime_r_ptrs[i] = m_soa_tile.GetRealData(i + NArrayReal).dataPtr();
                copy_real = true;
            }
        }
        if (copy_real) {
            Gpu::htod_memcpy_async(m_runtime_r_ptrs.data(), m_h_runtime_r_ptrs.data(),
                                   m_h_runtime_r_ptrs.size()*sizeof(ParticleReal*));
        }

        bool copy_int = false;
        m_h_runtime_i_ptrs.resize(m_soa_tile.NumIntComps() - NArrayInt, nullptr);
        for (std::size_t i = 0; i < m_h_runtime_i_ptrs.size(); ++i) {
            if (m_h_runtime_i_ptrs[i] != m_soa_tile.GetIntData(i + NArrayInt).dataPtr()) {
                m_h_runtime_i_ptrs[i] = m_soa_tile.GetIntData(i + NArrayInt).dataPtr();
                copy_int = true;
            }
        }
        if (copy_int) {
            Gpu::htod_memcpy_async(m_runtime_i_ptrs.data(), m_h_runtime_i_ptrs.data(),
                                   m_h_runtime_i_ptrs.size()*sizeof(int*));
        }
#else
        for (std::size_t i = 0; i < m_runtime_r_ptrs.size(); ++i) {
            m_runtime_r_ptrs[i] = m_soa_tile.GetRealData(i + NArrayReal).dataPtr();
        }

        for (std::size_t i = 0; i < m_runtime_i_ptrs.size(); ++i) {
            m_runtime_i_ptrs[i] = m_soa_tile.GetIntData(i + NArrayInt).dataPtr();
        }
#endif

        ParticleTileDataType ptd;
        if constexpr (!ParticleType::is_soa_particle) {
            ptd.m_aos = m_aos_tile().dataPtr();
        } else {
            ptd.m_aos = nullptr;
        }
        if constexpr (ParticleType::is_soa_particle) {
            ptd.m_idcpu = m_soa_tile.GetIdCPUData().dataPtr();
        } else {
            ptd.m_idcpu = nullptr;
        }
        if constexpr(NArrayReal > 0) {
            for (int i = 0; i < NArrayReal; ++i) {
                ptd.m_rdata[i] = m_soa_tile.GetRealData(i).dataPtr();
            }
        }
        if constexpr(NArrayInt > 0) {
            for (int i = 0; i < NArrayInt; ++i) {
                ptd.m_idata[i] = m_soa_tile.GetIntData(i).dataPtr();
            }
        }
        ptd.m_size = size();
        ptd.m_num_runtime_real = m_runtime_r_ptrs.size();
        ptd.m_num_runtime_int = m_runtime_i_ptrs.size();
        ptd.m_runtime_rdata = m_runtime_r_ptrs.dataPtr();
        ptd.m_runtime_idata = m_runtime_i_ptrs.dataPtr();

#ifdef AMREX_USE_GPU
        if (copy_real || copy_int) {
            Gpu::streamSynchronize();
        }
#endif

        return ptd;
    }

    ConstParticleTileDataType getConstParticleTileData () const
    {
        m_runtime_r_cptrs.resize(m_soa_tile.NumRealComps() - NArrayReal);
        m_runtime_i_cptrs.resize(m_soa_tile.NumIntComps() - NArrayInt);
#ifdef AMREX_USE_GPU
        bool copy_real = false;
        m_h_runtime_r_cptrs.resize(m_soa_tile.NumRealComps() - NArrayReal, nullptr);
        for (std::size_t i = 0; i < m_h_runtime_r_cptrs.size(); ++i) {
            if (m_h_runtime_r_cptrs[i] != m_soa_tile.GetRealData(i + NArrayReal).dataPtr()) {
                m_h_runtime_r_cptrs[i] = m_soa_tile.GetRealData(i + NArrayReal).dataPtr();
                copy_real = true;
            }
        }
        if (copy_real) {
            Gpu::htod_memcpy_async(m_runtime_r_cptrs.data(), m_h_runtime_r_cptrs.data(),
                                   m_h_runtime_r_cptrs.size()*sizeof(ParticleReal*));
        }

        bool copy_int = false;
        m_h_runtime_i_cptrs.resize(m_soa_tile.NumIntComps() - NArrayInt, nullptr);
        for (std::size_t i = 0; i < m_h_runtime_i_cptrs.size(); ++i) {
            if (m_h_runtime_i_cptrs[i] != m_soa_tile.GetIntData(i + NArrayInt).dataPtr()) {
                m_h_runtime_i_cptrs[i] = m_soa_tile.GetIntData(i + NArrayInt).dataPtr();
                copy_int = true;
            }
        }
        if (copy_int) {
            Gpu::htod_memcpy_async(m_runtime_i_cptrs.data(), m_h_runtime_i_cptrs.data(),
                                   m_h_runtime_i_cptrs.size()*sizeof(int*));
        }
#else
        for (std::size_t i = 0; i < m_runtime_r_cptrs.size(); ++i) {
            m_runtime_r_cptrs[i] = m_soa_tile.GetRealData(i + NArrayReal).dataPtr();
        }

        for (std::size_t i = 0; i < m_runtime_i_cptrs.size(); ++i) {
            m_runtime_i_cptrs[i] = m_soa_tile.GetIntData(i + NArrayInt).dataPtr();
        }
#endif

        ConstParticleTileDataType ptd;
        if constexpr (!ParticleType::is_soa_particle) {
            ptd.m_aos = m_aos_tile().dataPtr();
        } else {
            ptd.m_aos = nullptr;
        }
        if constexpr (ParticleType::is_soa_particle) {
            ptd.m_idcpu = m_soa_tile.GetIdCPUData().dataPtr();
        } else {
            ptd.m_idcpu = nullptr;
        }
        if constexpr(NArrayReal > 0) {
            for (int i = 0; i < NArrayReal; ++i) {
                ptd.m_rdata[i] = m_soa_tile.GetRealData(i).dataPtr();
            }
        }
        if constexpr(NArrayInt > 0) {
            for (int i = 0; i < NArrayInt; ++i) {
                ptd.m_idata[i] = m_soa_tile.GetIntData(i).dataPtr();
            }
        }
        ptd.m_size = size();
        ptd.m_num_runtime_real = m_runtime_r_cptrs.size();
        ptd.m_num_runtime_int = m_runtime_i_cptrs.size();
        ptd.m_runtime_rdata = m_runtime_r_cptrs.dataPtr();
        ptd.m_runtime_idata = m_runtime_i_cptrs.dataPtr();

#ifdef AMREX_USE_GPU
        if (copy_real || copy_int) {
            Gpu::streamSynchronize();
        }
#endif

        return ptd;
    }

public:
    // ========== 新增: Tile内本地索引与分桶偏移表 ==========
    std::vector<int> m_local_index; // 底层存储，包含粒子ID或INVALID_PARTICLE_ID
    std::vector<int> m_bin_offsets;   // 标记每个bin在m_local_index中的起始物理索引 (size = numcell + 1)
    std::vector<int> m_bin_lengths;   // 存储每个bin中*有效*粒子的数量 (size = numcell)
    int m_num_particles;                       // 总粒子数
    int m_capacity;                            // m_local_index的实际容量
    int m_num_bins;
    bool m_was_rebuilt_this_step= false ;           // 标记该Tile是否在本步中被重建过
    
    std::vector<int> m_slot_offsets; // 和binoffset对其的指示空槽位置的，
    // slot_offsets[0]指向index[0]
    
    
    static constexpr int INVALID_PARTICLE_ID = -1;
    // static constexpr int INVALID_PARTICLE_ID = std::numeric_limits<int>::max();
    
    // --- 新增的反向查找表 ---
    std::vector<int> m_pid_to_bin_map; // Index = pid, Value = bin_id (0-based)
    // 使用 -1 表示无效或未映射
    std::vector<int> m_pid_to_index_map; // 其实就是perm，是一个紧凑的index数组，方便统计和访问
    
    // --- HBM版本的排序数组 ---
    HBM_Inner_Vector_Int m_local_index_hbm; 
    HBM_Inner_Vector_Int m_pid_to_bin_map_hbm; 
    // no need 
    HBM_Inner_Vector_Int m_pid_to_index_map_hbm;
    HBM_Inner_Vector_Int m_bin_offsets_hbm; 
    HBM_Inner_Vector_Int m_slot_offsets_hbm; 
    HBM_Inner_Vector_Int m_bin_lengths_hbm; 
    
    
    // 下面的已废弃2025.07.21
    std::vector<int> m_bin_max_ip; // 指示每个bin最大的ip，用于判断插入位置
    int m_num_empty_slots;                     // 当前空槽位数量
    // int m_max_bin_capacity=0;
    std::vector<std::vector<int>> m_free_slots; //双vector模拟栈
    // std::vector<std::stack<int>> m_free_slots_stack;
    // 单vector模拟栈
    // std::vector<int> m_all_free_slots;
    // std::vector<int> m_free_slots_offsets;
    // std::vector<int> m_free_slots_sizes;

    // ========== 新增: 方便外部写入/读取的接口 ==========
    // 当计算完Permutation & Offsets之后，可以调用这个set函数更新
    void setIndexAndOffset(const int* perm_ptr, std::size_t np,
                           const int* offset_ptr, int nbins)
    {
        // 1) 根据外部传入的 permutation array 填充 m_local_index
        m_local_index.assign(perm_ptr, perm_ptr + np);

        // 2) 根据外部传入的 offsets 填充 m_bin_offsets
        //    offsets 数组长度 = nbins + 1
        m_bin_offsets.assign(offset_ptr, offset_ptr + nbins + 1);
    }
# if 0
    /**
     * @brief 从给定的粒子排列（perm）和偏移量（offsets）为扩展瓦片（grown_tile_box）构建PMA（Packed Memory Array）结构。
     *
     * 此函数处理真实瓦片（real_tile_box）中的粒子数据，并为扩展瓦片中的所有单元格（包括真实瓦片和保护单元）分配空间。
     * 每个单元格（bin）至少保证一个空槽。真实瓦片中有粒子的bin会根据gap_ratio额外分配空槽。
     *
     * @param perm 指向已排序粒子ID数组的指针（仅包含真实瓦片内的粒子）。
     * @param np 真实瓦片中的总粒子数。
     * @param offsets 指向描述真实瓦片各bin在perm数组中偏移量的数组指针。
     * offsets[0]是第一个bin的起始，offsets[num_original_bins_in_real_tile]是最后一个bin的结束（等于np）。
     * 数组大小应为 num_original_bins_in_real_tile + 1。
     * @param num_original_bins_in_real_tile 真实瓦片中的原始bin数量（对应原函数中的numBins）。
     * @param real_tile_box AMReX Box对象，描述真实瓦片的几何范围。
     * @param grown_tile_box AMReX Box对象，描述扩展瓦片（真实瓦片 + 保护单元）的几何范围。
     * @param gap_ratio （默认0.25）指定为含有粒子的bin额外分配的空槽比例（相对于粒子数）。
     */
    void buildPMAForGrownTile( 
        const int* perm,
        const int np,
        const int* offsets, // 注意：原代码中 offsets 是 int*, 这里保持 const int* 以表示不修改输入
        int num_original_bins_in_real_tile,
        const amrex::Box& real_tile_box,
        const amrex::Box& grown_tile_box,
        double gap_ratio=0.25) // 默认值可以在类方法声明中指定，例如 gap_ratio = 0.25
    { 
        // 步骤 1: 计算扩展瓦片中的总bin数量
        // grown_tile_box 中的每个单元格都被视为一个新的bin
        const int num_total_bins_grown = grown_tile_box.numPts();
        m_num_bins=num_total_bins_grown;
        // 步骤 2: 估算PMA的总容量并初始化相关数据结构
        m_num_particles = np; // PMA中实际存储的“真实”粒子总数（来自real_tile_box）

        // 容量计算公式：
        // - np: 存储真实粒子的空间
        // - np * gap_ratio: 为真实粒子按比例额外分配的空槽
        // - num_total_bins_grown: 为扩展瓦片中的每一个bin（无论真实或保护）保证至少一个空槽
        m_capacity = static_cast<int>(std::ceil(static_cast<double>(np) * (1.0 + gap_ratio)) + 2*num_total_bins_grown);

        // 初始化PMA的底层存储区，所有位置填充无效粒子ID
        m_local_index.assign(m_capacity, INVALID_PARTICLE_ID);
        // m_pid_to_index_map.assign(perm,perm+np);

        // 调整bin偏移量数组和bin长度数组的大小以适应扩展瓦片
        m_bin_offsets.resize(num_total_bins_grown + 1); // +1 是为最后一个bin的结束偏移（哨兵）
        m_bin_lengths.assign(num_total_bins_grown, 0);  // 所有bin的初始粒子数设为0
        // m_free_slots.resize(num_total_bins_grown);
        // 创建一个模板内部vector，它已经预留了至少1个空间
        std::vector<int> inner_vector_template;
        inner_vector_template.reserve(1); // 或者使用上面更精确的 reserve_at_least 计算

        // 使用 assign 将外部vector resize 并用模板内部vector的拷贝来填充
        // m_free_slots.assign(num_total_bins_grown, inner_vector_template);
        m_free_slots_offsets.assign(num_total_bins_grown + 1, 0);
        m_free_slots_sizes.assign(num_total_bins_grown, 0);
        int total_gap_size=num_total_bins_grown+std::ceil(static_cast<double>(np) * (gap_ratio));
        m_all_free_slots.resize(total_gap_size);
        const int vlf=svcntw();

        // 初始化粒子ID到其所在新bin索引的反向映射表
        // 大小仍为np，因为只有这些粒子需要映射。用-1表示未映射或无效。
        if (np > 0) { // 仅当有粒子时才分配和初始化
            m_pid_to_bin_map.assign(np, -1);
            m_pid_to_index_map.assign(np, -1);
        } else {
            m_pid_to_bin_map.clear(); // 如果没有粒子，清空map
            m_pid_to_index_map.clear(); // 如果没有粒子，清空map
            amrex::Abort("Error: NO particle at m_pid_to_bin_map in buildPMAForGrownTile");
        }


        // 步骤 3: 逐个处理扩展瓦片中的所有bin，填充粒子和空槽
        int write_ptr = 0;                 // 当前在 m_local_index 中的写入位置指针
        int gap_ptr = 0;                 
        int current_grown_bin_idx = 0;     // 扩展瓦片中当前处理的bin的线性索引 (0-based)

        // 获取真实瓦片的左下角坐标和各维度长度，用于后续计算原始bin的索引
        const amrex::IntVect real_tile_lo = real_tile_box.smallEnd();
        const amrex::IntVect real_tile_lengths = real_tile_box.length(); // 获取各维度长度 (Lx, Ly, Lz)

        // 使用AMReX的BoxIterator遍历grown_tile_box中的所有单元格
        // BoxIterator通常采用Fortran顺序（列主序：x索引变化最快，然后是y，然后是z）
        for (amrex::BoxIterator bit(grown_tile_box); bit.ok(); ++bit) {
            const amrex::IntVect current_cell_iv = bit(); // 获取当前单元格在grown_tile_box中的三维坐标

            // 记录当前grown_bin在PMA (m_local_index)中的起始物理地址
            m_bin_offsets[current_grown_bin_idx] = write_ptr;
            m_free_slots_offsets[current_grown_bin_idx] = gap_ptr;
            int particles_in_this_grown_bin = 0; // 初始化当前grown_bin中的粒子数为0

            if (real_tile_box.contains(current_cell_iv)) {
                // --- 情况1：当前单元格位于真实瓦片区域 (real_tile_box) ---

                // 将当前单元格的三维坐标 (current_cell_iv, 相对于grown_tile_box)
                // 转换为其在 real_tile_box 中的原始局部一维bin索引 (original_real_bin_idx)。
                // 这个转换的线性化顺序必须与生成输入参数 `perm` 和 `offsets` 时所用的顺序完全一致。
                amrex::IntVect rel_iv = current_cell_iv - real_tile_lo; // 计算相对于real_tile_box左下角点的相对坐标(dx, dy, dz)
                int original_real_bin_idx = 0; // 0-based

                // --- 选择与您的 offsets 生成方式匹配的线性化方法 ---
                // 选项 A: Fortran顺序 (列主序: x变化最快, AMReX默认方式)
                // original_real_bin_idx = rel_iv[0]; // x分量
                // #if (AMREX_SPACEDIM >= 2)
                // original_real_bin_idx += rel_iv[1] * real_tile_lengths[0]; // y分量 * Lx
                // #endif
                // #if (AMREX_SPACEDIM == 3)
                // original_real_bin_idx += rel_iv[2] * (real_tile_lengths[0] * real_tile_lengths[1]); // z分量 * Lx * Ly
                // #endif

                // 选项 B: C语言顺序 (行主序: z变化最慢, x变化最快 - 假设是 (z,y,x) 的嵌套)
                // (即 idx = z_coord * (Lx * Ly) + y_coord * Lx + x_coord)
                original_real_bin_idx = rel_iv[0]; // x分量 (dx)
    #if (AMREX_SPACEDIM >= 2)
                original_real_bin_idx += rel_iv[1] * real_tile_lengths[0]; // y分量 * Lx (dy * Lx)
    #endif
    #if (AMREX_SPACEDIM == 3)
                // 确保这里的 real_tile_lengths 索引正确对应维度
                // 如果 real_tile_lengths[0]是x方向长度, real_tile_lengths[1]是y方向长度
                original_real_bin_idx += rel_iv[2] * (real_tile_lengths[0] * real_tile_lengths[1]); // z分量 * (Lx * Ly) (dz * Lx * Ly)
    #endif
                // --- 结束线性化方法选择 ---

                // 安全检查：确保计算出的original_real_bin_idx在有效范围内
                if (original_real_bin_idx < 0 || original_real_bin_idx >= num_original_bins_in_real_tile) {
                    fprintf(stderr, "错误 (buildPMAForGrownTile): 计算得到的 original_real_bin_idx %d 超出有效范围 [0, %d)。\n"
                                    "       grown_cell_iv: (%d,%d,%d), rel_iv: (%d,%d,%d)\n",
                            original_real_bin_idx, num_original_bins_in_real_tile,
                            current_cell_iv[0], current_cell_iv[1], (AMREX_SPACEDIM==3 ? current_cell_iv[2] : 0),
                            rel_iv[0], rel_iv[1], (AMREX_SPACEDIM==3 ? rel_iv[2] : 0));
                    // 发生此错误通常表示线性化逻辑不匹配或Box定义问题。
                    // 为简单起见，这里跳转到保护单元的处理逻辑，避免进一步错误。
                    amrex::Abort("ERROR original_real_bin_idx:线性化逻辑不匹配或Box定义问题");
                    // goto handle_as_guard_cell_on_error;
                }
                // 从 offsets 数组获取这个原始bin所包含的粒子在 perm 数组中的起止索引
                const int p_start_idx_in_perm = offsets[original_real_bin_idx];
                const int p_end_idx_in_perm   = offsets[original_real_bin_idx + 1];
                particles_in_this_grown_bin   = p_end_idx_in_perm - p_start_idx_in_perm; // 计算该原始bin中的粒子数
                
                if (particles_in_this_grown_bin > 0) {
                    const int* particle_ids_to_copy = perm + p_start_idx_in_perm; // 获取待拷贝粒子ID数据段的起始指针
    
                    // 再次检查是否有足够容量拷贝这些粒子 (主要用于调试，理论上容量计算应该避免此问题)
                    if (write_ptr + particles_in_this_grown_bin > m_capacity) {
                        fprintf(stderr, "错误 (buildPMAForGrownTile): 为真实瓦片区域的bin (grown_idx %d, orig_idx %d) 分配PMA时容量不足!\n"
                                        "       容量: %d, 当前写入点: %d, 需要空间: %d, 实际粒子数: %d\n",
                                current_grown_bin_idx, original_real_bin_idx, m_capacity, write_ptr,
                                write_ptr + particles_in_this_grown_bin, particles_in_this_grown_bin);
                        // 简化处理：如果容量不足，则不拷贝这些粒子，并将该bin的粒子数记为0。
                        // 实际应用中可能需要更复杂的错误处理机制，例如重新分配或抛出异常。
                        particles_in_this_grown_bin = 0;
                    } else {
                        // 将粒子ID从 perm 数组拷贝到 m_local_index 的当前写入位置
                        std::copy(particle_ids_to_copy, particle_ids_to_copy + particles_in_this_grown_bin, m_local_index.begin() + write_ptr);
    
                        // 更新 m_pid_to_bin_map：将每个成功拷贝的粒子ID映射到当前的 grown_bin_idx
                        if (!m_pid_to_bin_map.empty()) { // 仅当map有效时操作
                            for (int i = 0; i < particles_in_this_grown_bin; ++i) {
                                int pid = particle_ids_to_copy[i];
                                // 假设粒子ID是非负的，并且小于np (m_pid_to_bin_map的大小)
                                if (pid >= 0 && static_cast<size_t>(pid) < m_pid_to_bin_map.size()) {
                                    m_pid_to_bin_map[pid] = current_grown_bin_idx;
                                    m_pid_to_index_map[pid] = write_ptr+i;
                                } else {
                                    fprintf(stderr, "警告 (buildPMAForGrownTile): 粒子ID %d 对于 m_pid_to_bin_map (大小 %zu) 无效 (grown_bin_idx %d)。\n",
                                        pid, m_pid_to_bin_map.size(), current_grown_bin_idx);
                                }
                            }
                        }
                    }
                }
                m_bin_lengths[current_grown_bin_idx] = particles_in_this_grown_bin; // 更新当前grown_bin的有效粒子数
                write_ptr += particles_in_this_grown_bin;                           // 移动写入指针到这批粒子数据之后

                // 为当前真实瓦片区域的bin添加空槽：
                // 1. 基于实际粒子数的 gap_ratio 比例计算的空槽 (如果bin中无粒子，则此项为0)
                // 2. 再加上额外的1个保证空槽 (这是对所有grown_bin的要求)
                // const int particle_gap_slots = (particles_in_this_grown_bin > 0) ? static_cast<int>(std::ceil(static_cast<double>(particles_in_this_grown_bin) * gap_ratio)) : 0;
                // const int total_empty_slots_to_add = particle_gap_slots + 1; // 总共需要添加的空槽数
                const int total_empty_slots_to_add = static_cast<int>(std::ceil(particles_in_this_grown_bin * gap_ratio))+1;
                int end_of_bin_physical_space = write_ptr + total_empty_slots_to_add; // 计算此bin在本PMA段中应占据的结束物理位置
                // 确保写入指针在增加空槽后不超过PMA的总容量
                write_ptr = std::min(end_of_bin_physical_space, m_capacity);

                gap_ptr+=total_empty_slots_to_add;
                m_free_slots_sizes[current_grown_bin_idx]=total_empty_slots_to_add;
                for (int k = 0; k < total_empty_slots_to_add; k += vlf) {
                    // 創建謂詞，安全處理最後一個不滿的向量塊
                    svbool_t pg = svwhilelt_b32(k, total_empty_slots_to_add);
    
                    // a. 生成要寫入的數值向量
                    //    首先，建立一個索引向量 [k, k+1, k+2, ...]
                    svint32_t vec_k_indices = svindex_s32(k, 1);
                    //    然後，建立一個填滿 offset - 1 的向量
                    svint32_t vec_base = svdup_n_s32(write_ptr - 1);
                    //    計算出最終要寫入的值：(offset - 1) - k
                    svint32_t values_to_store = svsub_s32_z(pg, vec_base, vec_k_indices);
    
                    // b. 高效的連續寫入 (Store)
                    //    將產生的向量一次性寫入目標記憶體的正確區段
                    svst1_s32(pg, &m_all_free_slots.data()[m_free_slots_offsets[current_grown_bin_idx] + k], values_to_store);
                }
                // m_max_bin_capacity=std::max(m_max_bin_capacity,total_empty_slots_to_add);
                //暂时不需要这个单个bin最大容量指示器

                // 记录这些空槽的物理地址

                // 这些空槽位于 [write_ptr, write_ptr + total_empty_slots_to_add - 1]
                // 并且确保不超过 m_capacity
                // int actual_slots_added = 0;
                // for (int i = 0; i < total_empty_slots_to_add; ++i) {
                //     if (write_ptr + i < m_capacity) {
                //         // m_free_slots[current_grown_bin_idx].push_back(write_ptr +total_empty_slots_to_add-1- i);
                //         actual_slots_added++;
                //     } else {
                //         amrex::Abort("ERROR free slots:已达总容量上限");
                //         break; // 已达总容量上限
                //     }
                // }
                // write_ptr += actual_slots_added; // 更新写入指针，跳过已记录的空槽

                if(end_of_bin_physical_space>m_capacity){
                    amrex::Abort("ERROR write_ptrs:已达总容量上限");
                }
            } else {
                // 标签，用于处理保护单元，或在上述真实单元映射计算出错时跳转到这里进行统一处理
                handle_as_guard_cell_on_error:
                // --- 情况2：当前单元格位于保护单元区域 (guard cell area) ---
                // (或者之前真实瓦片区域的索引计算出错，也按保护单元处理以保证流程继续)
                m_bin_lengths[current_grown_bin_idx] = 0; // 保护单元没有粒子，长度为0
                particles_in_this_grown_bin = 0;          // 显式设为0
    
                // 为保护单元的bin添加1个保证的空槽 (此空间已在m_capacity的总量中预留)
                const int total_empty_slots_to_add = 1;
                int end_of_bin_physical_space = write_ptr + total_empty_slots_to_add;
                // 确保写入指针在增加空槽后不超过PMA的总容量
                write_ptr = std::min(end_of_bin_physical_space, m_capacity);
                if(end_of_bin_physical_space>m_capacity){
                    amrex::Abort("ERROR write_ptrs in guard_cell:已达总容量上限");
                }
            }
            current_grown_bin_idx++; // 处理下一个grown_bin
        }
        // 迭代结束后进行一次性检查，确保所有grown_bin都被正确处理了
        if (current_grown_bin_idx != num_total_bins_grown) {
            fprintf(stderr, "错误 (buildPMAForGrownTile): 迭代处理的 grown_bin 数量 (%d) 与预期的总 grown_bin 数量 (%d) 不符。\n"
                            "这可能表示BoxIterator的遍历与numPts()的计数不一致，或者循环逻辑有误。\n",
                current_grown_bin_idx, num_total_bins_grown);
            // 这里可能需要更健壮的错误处理，例如抛出异常
            amrex::Abort("ERROR current_grown_bin_idxs");
        }

        // 设置最后一个bin的结束偏移量（作为哨兵值），该值也代表了PMA中当前已使用的总长度
        m_bin_offsets[num_total_bins_grown] = write_ptr;
        m_capacity = write_ptr;
        // 更新PMA中总的空槽位数
        // m_num_particles 存储的是真实粒子数 np
        m_num_empty_slots = m_capacity - m_num_particles;

        m_was_rebuilt_this_step = true; // 标记PMA结构在本步骤中已被重建

        // 如果有 m_num_particles_in_pma 这样的成员变量，也需要更新
        // m_num_particles_in_pma = m_num_particles;
    }
#endif

    // gap_ratio=0.25 → 每个 bin 在末尾额外留 25% 空位
    void buildPMAfromDenseBins_org(const int* perm,
                                const int np,
                                int* offsets,
                                int numBins,
                                double gap_ratio=0.25)
    {
        // 1) 预估容量并一次性分配
        m_num_particles = np;
        // m_num_particles = static_cast<int>(np);
        m_capacity      = static_cast<int>( std::ceil(np * (1.0 + gap_ratio))+numBins );
        // printf("m_capacity = %d, np %d, numBins %d\n", m_capacity, np, numBins);
        m_local_index.assign(m_capacity, INVALID_PARTICLE_ID);

        m_bin_offsets.resize(numBins + 1);
        m_bin_lengths.resize(numBins);

        // 2) 逐 bin 拷贝 + 插空
        int write_ptr = 0;
        for (int b = 1; b <= numBins; ++b)
        {
            const int begin = static_cast<int>(offsets[b-1]);
            const int end   = static_cast<int>(offsets[b]);
            const int n_in_bin = end - begin;

            // 起始物理地址
            m_bin_offsets[b-1] = write_ptr;

            // 拷贝有效粒子
            std::copy( perm + begin, perm + end, m_local_index.begin() + write_ptr );
            m_bin_lengths[b-1] = n_in_bin;
            // write_ptr       += n_in_bin;
            if (write_ptr > m_capacity) {
                printf("错误：write_ptr (%d) 超出 m_capacity (%d)\n", write_ptr, m_capacity);
                // 处理错误情况
            }
            // 3) 给本 bin 末尾插入空槽
            const int gap   = static_cast<int>( std::round(n_in_bin * gap_ratio) )+1;
            write_ptr      += (gap+n_in_bin);                           
            // 直接跳过 gap 个 INVALID 槽
            // printf("pushing %d particles into bin %d, gap = %d\n", n_in_bin, b, gap);
        }
        m_bin_offsets[numBins] = write_ptr;                 // 终止哨兵
        m_num_empty_slots      = m_capacity - m_num_particles;
    }
# if 0
    // gap_ratio=0.25 → 每个 bin 在末尾额外留 25% 空位
    // have a stack to identify empty slot
    void buildPMAfromDenseBins(const int* perm,
                                const int np,
                                int* offsets,
                                int numBins,
                                double gap_ratio=0.25)
    {
        // 1) 预估容量并一次性分配
        m_num_particles = np;
        m_num_bins=numBins;
        // m_num_particles = static_cast<int>(np);
        m_capacity      = static_cast<int>( std::ceil(np * (1 + gap_ratio))+numBins*2 );
        // printf("m_capacity = %d, np %d, numBins %d\n", m_capacity, np, numBins);
        m_local_index.assign(m_capacity, INVALID_PARTICLE_ID);

        m_bin_offsets.resize(numBins + 1);
        m_bin_lengths.resize(numBins);
        // m_free_slots.resize(numBins);
        m_free_slots_offsets.assign(numBins + 1, 0);
        m_free_slots_sizes.assign(numBins, 0);
        int total_gap_size=numBins+std::ceil(static_cast<double>(np) * (gap_ratio));
        m_all_free_slots.resize(total_gap_size);

        // --- 新增：调整并初始化反向映射表 ---
        // 大小应该基于当前的粒子数 np
        m_pid_to_bin_map.assign(np, -1); // 初始化为无效 bin

        // 2) 逐 bin 拷贝 + 插空
        int write_ptr = 0;
        int gap_ptr = 0;
        const int vlf=svcntw();
        for (int b = 1; b <= numBins; ++b)
        {
            const int begin = static_cast<int>(offsets[b-1]);
            const int end   = static_cast<int>(offsets[b]);
            const int n_in_bin = end - begin;
            const int current_bin_id = b - 1; // 转换为 0-based bin index

            // 起始物理地址
            m_bin_offsets[b-1] = write_ptr;
            m_free_slots_offsets[b-1] = gap_ptr;

            // 拷贝有效粒子并更新 Map
            const int* read_ptr = perm + begin;
            // 检查边界，确保不会写入超出 m_local_index 容量的区域
            if (write_ptr + n_in_bin > m_capacity) {
                printf("Error: Not enough capacity during PMA build for bin %d! Cap: %d, Need: %d\n",
                        current_bin_id, m_capacity, write_ptr + n_in_bin);
                // 需要错误处理或重新计算容量
                // 这里简化：仅拷贝能放下的部分
                // int can_copy = m_capacity - write_ptr;
                // if (can_copy < 0) can_copy = 0;
                // std::copy(read_ptr, read_ptr + can_copy, m_local_index.begin() + write_ptr);
                // for (int i = 0; i < can_copy; ++i) {
                //     int ip = read_ptr[i];
                //     if (ip >= 0 && ip < m_pid_to_bin_map.size()) {
                //         m_pid_to_bin_map[ip] = current_bin_id;
                //     } else { /* Error */ }
                // }
                // m_bin_lengths[current_bin_id] = can_copy;
                // write_ptr += can_copy; // 更新写指针

            } else {
                // 容量足够
                std::copy(read_ptr, read_ptr + n_in_bin, m_local_index.begin() + write_ptr);
                // 更新 pid_to_bin_map
                for (int i = 0; i < n_in_bin; ++i) {
                    int ip = read_ptr[i];
                    if (ip >= 0 && ip < m_pid_to_bin_map.size()) {
                        m_pid_to_bin_map[ip] = current_bin_id;
                    } else {
                        printf("Error: Particle ID %d out of bounds for pid_to_bin_map (size %zu) during build.\n",
                                ip, m_pid_to_bin_map.size());
                    }
                }
                
                m_bin_lengths[current_bin_id] = n_in_bin;
                write_ptr += n_in_bin; // 更新写指针
            }

           // 3) 给本 bin 末尾插入空槽
            const int gap = static_cast<int>(std::round(n_in_bin * gap_ratio))+1;
            gap_ptr+=gap;
            m_free_slots_sizes[current_bin_id]=gap;

            // 确保增加 gap 后不超过容量
            int end_of_block = write_ptr + gap;
            // 跳过 gap 或直到容量末尾
            // write_ptr = std::min(end_of_block, m_capacity);
            if(end_of_block>=m_capacity){
                printf("end_of_block %d m_capacity %d \n",end_of_block,m_capacity);
                amrex::Abort("ERROR:end_of_block != m_capacity!!!");
            }
            write_ptr = end_of_block;
            // for(int i = write_ptr-1; i >= write_ptr-gap; --i){
                // m_free_slots[b-1].push_back(i);
            // }
            for (int k = 0; k < gap; k += vlf) {
                // 創建謂詞，安全處理最後一個不滿的向量塊
                svbool_t pg = svwhilelt_b32(k, gap);

                // a. 生成要寫入的數值向量
                //    首先，建立一個索引向量 [k, k+1, k+2, ...]
                svint32_t vec_k_indices = svindex_s32(k, 1);
                //    然後，建立一個填滿 offset - 1 的向量
                svint32_t vec_base = svdup_n_s32(write_ptr - 1);
                //    計算出最終要寫入的值：(offset - 1) - k
                svint32_t values_to_store = svsub_s32_z(pg, vec_base, vec_k_indices);

                // b. 高效的連續寫入 (Store)
                //    將產生的向量一次性寫入目標記憶體的正確區段
                svst1_s32(pg, &m_all_free_slots.data()[m_free_slots_offsets[b-1] + k], values_to_store);
            }
        }
        m_bin_offsets[numBins] = write_ptr; // 终止哨兵
        m_capacity=write_ptr;
        m_num_empty_slots = m_capacity - m_num_particles;
        m_local_index.resize(m_capacity, INVALID_PARTICLE_ID);

        // 在构建结束时，记录下与此PMA结构对应的粒子数
        // m_num_particles_in_pma = m_num_particles; // 使用 build 时传入的 np
    }
#endif

# if 1
/**
     * @brief 从给定的粒子排列（perm）和偏移量（offsets）为扩展瓦片（grown_tile_box）构建PMA（Packed Memory Array）结构。
     *
     * 此函数处理真实瓦片（real_tile_box）中的粒子数据，并为扩展瓦片中的所有单元格（包括真实瓦片和保护单元）分配空间。
     * 每个单元格（bin）至少保证一个空槽。真实瓦片中有粒子的bin会根据gap_ratio额外分配空槽。
     *
     * @param perm 指向已排序粒子ID数组的指针（仅包含真实瓦片内的粒子）。
     * @param np 真实瓦片中的总粒子数。
     * @param offsets 指向描述真实瓦片各bin在perm数组中偏移量的数组指针。
     * offsets[0]是第一个bin的起始，offsets[num_original_bins_in_real_tile]是最后一个bin的结束（等于np）。
     * 数组大小应为 num_original_bins_in_real_tile + 1。
     * @param num_original_bins_in_real_tile 真实瓦片中的原始bin数量（对应原函数中的numBins）。
     * @param real_tile_box AMReX Box对象，描述真实瓦片的几何范围。
     * @param grown_tile_box AMReX Box对象，描述扩展瓦片（真实瓦片 + 保护单元）的几何范围。
     * @param gap_ratio （默认0.25）指定为含有粒子的bin额外分配的空槽比例（相对于粒子数）。
     */
    void buildPMAForGrownTile( 
        const int* perm,
        const int np,
        const int* offsets, // 注意：原代码中 offsets 是 int*, 这里保持 const int* 以表示不修改输入
        int num_original_bins_in_real_tile,
        const amrex::Box& real_tile_box,
        const amrex::Box& grown_tile_box,
        double gap_ratio=0.25) // 默认值可以在类方法声明中指定，例如 gap_ratio = 0.25
    { 
        // 步骤 1: 计算扩展瓦片中的总bin数量
        // grown_tile_box 中的每个单元格都被视为一个新的bin
        const int num_total_bins_grown = grown_tile_box.numPts();
        // printf("num bins %d \n",num_total_bins_grown);
        m_num_bins=num_total_bins_grown;
        // 步骤 2: 估算PMA的总容量并初始化相关数据结构
        m_num_particles = np; // PMA中实际存储的“真实”粒子总数（来自real_tile_box）

        // 容量计算公式：
        // - np: 存储真实粒子的空间
        // - np * gap_ratio: 为真实粒子按比例额外分配的空槽
        // - num_total_bins_grown: 为扩展瓦片中的每一个bin（无论真实或保护）保证至少一个空槽
        m_capacity = static_cast<int>(std::ceil(static_cast<double>(np) * (1.0 + gap_ratio)) + 2*num_total_bins_grown);

        // 初始化PMA的底层存储区，所有位置填充无效粒子ID
        m_local_index.reserve(m_capacity+1);
        m_local_index.resize(m_capacity);
        std::fill_n(m_local_index.data(), m_capacity, INVALID_PARTICLE_ID);
        // m_local_index.assign(m_capacity, INVALID_PARTICLE_ID);
        // m_pid_to_index_map.assign(perm,perm+np);

        // 调整bin偏移量数组和bin长度数组的大小以适应扩展瓦片
        m_bin_offsets.resize(num_total_bins_grown + 1); // +1 是为最后一个bin的结束偏移（哨兵）
        m_slot_offsets.resize(num_total_bins_grown + 1); // +1 是为最后一个bin的结束偏移（哨兵）
        m_bin_max_ip.resize(num_total_bins_grown); // +1 是为最后一个bin的结束偏移（哨兵）
        m_bin_lengths.clear();
        m_bin_lengths.resize(num_total_bins_grown);  // 所有bin的初始粒子数设为0
        // m_free_slots.resize(num_total_bins_grown);
        // 创建一个模板内部vector，它已经预留了至少1个空间
        // std::vector<int> inner_vector_template;
        // inner_vector_template.reserve(1); // 或者使用上面更精确的 reserve_at_least 计算

        // 使用 assign 将外部vector resize 并用模板内部vector的拷贝来填充
        // m_free_slots.assign(num_total_bins_grown, inner_vector_template);

        // 初始化粒子ID到其所在新bin索引的反向映射表
        // 大小仍为np，因为只有这些粒子需要映射。用-1表示未映射或无效。
        if (np > 0) { // 仅当有粒子时才分配和初始化
            m_pid_to_bin_map.resize(np);
            m_pid_to_index_map.resize(np);
        } else {
            m_pid_to_bin_map.clear(); // 如果没有粒子，清空map
            m_pid_to_index_map.clear(); // 如果没有粒子，清空map
            amrex::Abort("Error: NO particle at m_pid_to_bin_map in buildPMAForGrownTile");
        }


        // 步骤 3: 逐个处理扩展瓦片中的所有bin，填充粒子和空槽
        int write_ptr = 0;                 // 当前在 m_local_index 中的写入位置指针
        int current_grown_bin_idx = 0;     // 扩展瓦片中当前处理的bin的线性索引 (0-based)
        m_slot_offsets[0] = 0;

        // 获取真实瓦片的左下角坐标和各维度长度，用于后续计算原始bin的索引
        const amrex::IntVect real_tile_lo = real_tile_box.smallEnd();
        const amrex::IntVect real_tile_lengths = real_tile_box.length(); // 获取各维度长度 (Lx, Ly, Lz)

        // 使用AMReX的BoxIterator遍历grown_tile_box中的所有单元格
        // BoxIterator通常采用Fortran顺序（列主序：x索引变化最快，然后是y，然后是z）
        for (amrex::BoxIterator bit(grown_tile_box); bit.ok(); ++bit) {
            const amrex::IntVect current_cell_iv = bit(); // 获取当前单元格在grown_tile_box中的三维坐标

            // 记录当前grown_bin在PMA (m_local_index)中的起始物理地址
            m_bin_offsets[current_grown_bin_idx] = write_ptr;
            int particles_in_this_grown_bin = 0; // 初始化当前grown_bin中的粒子数为0

            if (real_tile_box.contains(current_cell_iv)) {
                // --- 情况1：当前单元格位于真实瓦片区域 (real_tile_box) ---

                // 将当前单元格的三维坐标 (current_cell_iv, 相对于grown_tile_box)
                // 转换为其在 real_tile_box 中的原始局部一维bin索引 (original_real_bin_idx)。
                // 这个转换的线性化顺序必须与生成输入参数 `perm` 和 `offsets` 时所用的顺序完全一致。
                amrex::IntVect rel_iv = current_cell_iv - real_tile_lo; // 计算相对于real_tile_box左下角点的相对坐标(dx, dy, dz)
                int original_real_bin_idx = 0; // 0-based

                // --- 选择与您的 offsets 生成方式匹配的线性化方法 ---
                // 选项 A: Fortran顺序 (列主序: x变化最快, AMReX默认方式)
                // original_real_bin_idx = rel_iv[0]; // x分量
                // #if (AMREX_SPACEDIM >= 2)
                // original_real_bin_idx += rel_iv[1] * real_tile_lengths[0]; // y分量 * Lx
                // #endif
                // #if (AMREX_SPACEDIM == 3)
                // original_real_bin_idx += rel_iv[2] * (real_tile_lengths[0] * real_tile_lengths[1]); // z分量 * Lx * Ly
                // #endif

                // 选项 B: C语言顺序 (行主序: z变化最慢, x变化最快 - 假设是 (z,y,x) 的嵌套)
                // (即 idx = z_coord * (Lx * Ly) + y_coord * Lx + x_coord)
                original_real_bin_idx = rel_iv[0]; // x分量 (dx)
                original_real_bin_idx += rel_iv[1] * real_tile_lengths[0]; // y分量 * Lx (dy * Lx)
                // 确保这里的 real_tile_lengths 索引正确对应维度
                // 如果 real_tile_lengths[0]是x方向长度, real_tile_lengths[1]是y方向长度
                original_real_bin_idx += rel_iv[2] * (real_tile_lengths[0] * real_tile_lengths[1]); // z分量 * (Lx * Ly) (dz * Lx * Ly)
                // --- 结束线性化方法选择 ---

                // 安全检查：确保计算出的original_real_bin_idx在有效范围内
                if (original_real_bin_idx < 0 || original_real_bin_idx >= num_original_bins_in_real_tile) {
                    fprintf(stderr, "错误 (buildPMAForGrownTile): 计算得到的 original_real_bin_idx %d 超出有效范围 [0, %d)。\n"
                                    "       grown_cell_iv: (%d,%d,%d), rel_iv: (%d,%d,%d)\n",
                            original_real_bin_idx, num_original_bins_in_real_tile,
                            current_cell_iv[0], current_cell_iv[1], (AMREX_SPACEDIM==3 ? current_cell_iv[2] : 0),
                            rel_iv[0], rel_iv[1], (AMREX_SPACEDIM==3 ? rel_iv[2] : 0));
                    // 发生此错误通常表示线性化逻辑不匹配或Box定义问题。
                    // 为简单起见，这里跳转到保护单元的处理逻辑，避免进一步错误。
                    amrex::Abort("ERROR original_real_bin_idx:线性化逻辑不匹配或Box定义问题");
                    // goto handle_as_guard_cell_on_error;
                }
                // 从 offsets 数组获取这个原始bin所包含的粒子在 perm 数组中的起止索引
                const int p_start_idx_in_perm = offsets[original_real_bin_idx];
                const int p_end_idx_in_perm   = offsets[original_real_bin_idx + 1];
                particles_in_this_grown_bin   = p_end_idx_in_perm - p_start_idx_in_perm; // 计算该原始bin中的粒子数
                
                if (particles_in_this_grown_bin > 0) {
                    const int* particle_ids_to_copy = perm + p_start_idx_in_perm; // 获取待拷贝粒子ID数据段的起始指针
    
                    // 再次检查是否有足够容量拷贝这些粒子 (主要用于调试，理论上容量计算应该避免此问题)
                    if (write_ptr + particles_in_this_grown_bin > m_capacity) {
                        fprintf(stderr, "错误 (buildPMAForGrownTile): 为真实瓦片区域的bin (grown_idx %d, orig_idx %d) 分配PMA时容量不足!\n"
                                        "       容量: %d, 当前写入点: %d, 需要空间: %d, 实际粒子数: %d\n",
                                current_grown_bin_idx, original_real_bin_idx, m_capacity, write_ptr,
                                write_ptr + particles_in_this_grown_bin, particles_in_this_grown_bin);
                        // 简化处理：如果容量不足，则不拷贝这些粒子，并将该bin的粒子数记为0。
                        // 实际应用中可能需要更复杂的错误处理机制，例如重新分配或抛出异常。
                        particles_in_this_grown_bin = 0;
                    } else {
                        // 将粒子ID从 perm 数组拷贝到 m_local_index 的当前写入位置
                        std::copy(particle_ids_to_copy, 
                            particle_ids_to_copy + particles_in_this_grown_bin, 
                            m_local_index.begin() + write_ptr);
    
                        // 更新 m_pid_to_bin_map：将每个成功拷贝的粒子ID映射到当前的 grown_bin_idx
                        if (!m_pid_to_bin_map.empty()) { // 仅当map有效时操作
                            for (int i = 0; i < particles_in_this_grown_bin; ++i) {
                                int pid = particle_ids_to_copy[i];
                                // 假设粒子ID是非负的，并且小于np (m_pid_to_bin_map的大小)
                                if (pid >= 0 && static_cast<size_t>(pid) < m_pid_to_bin_map.size()) {
                                    m_pid_to_bin_map[pid] = current_grown_bin_idx;
                                    m_pid_to_index_map[pid] = write_ptr+i;
                                } else {
                                    fprintf(stderr, "警告 (buildPMAForGrownTile): 粒子ID %d 对于 m_pid_to_bin_map (大小 %zu) 无效 (grown_bin_idx %d)。\n",
                                        pid, m_pid_to_bin_map.size(), current_grown_bin_idx);
                                }
                            }
                        }
                    }
                }
                m_bin_lengths[current_grown_bin_idx] = particles_in_this_grown_bin; // 更新当前grown_bin的有效粒子数
                write_ptr += particles_in_this_grown_bin;                           // 移动写入指针到这批粒子数据之后
                // 更新空槽的位置到尾巴
                m_slot_offsets[current_grown_bin_idx] = write_ptr;

                // 为当前真实瓦片区域的bin添加空槽：
                // 1. 基于实际粒子数的 gap_ratio 比例计算的空槽 (如果bin中无粒子，则此项为0)
                // 2. 再加上额外的1个保证空槽 (这是对所有grown_bin的要求)
                const int particle_gap_slots = (particles_in_this_grown_bin > 0) ? static_cast<int>(std::ceil(static_cast<double>(particles_in_this_grown_bin) * gap_ratio)) : 0;
                const int total_empty_slots_to_add = particle_gap_slots + 1; // 总共需要添加的空槽数
                // m_max_bin_capacity=std::max(m_max_bin_capacity,total_empty_slots_to_add);
                //暂时不需要这个单个bin最大容量指示器

                // 计算此bin在本PMA段中应占据的结束物理位置
                write_ptr += total_empty_slots_to_add; 
                // 确保写入指针在增加空槽后不超过PMA的总容量
                write_ptr = std::min(write_ptr, m_capacity);
                if(write_ptr>m_capacity){
                    amrex::Abort("ERROR write_ptrs:已达总容量上限");
                }
                
            } else {
                // 标签，用于处理保护单元，或在上述真实单元映射计算出错时跳转到这里进行统一处理
                handle_as_guard_cell_on_error:
                // --- 情况2：当前单元格位于保护单元区域 (guard cell area) ---
                // (或者之前真实瓦片区域的索引计算出错，也按保护单元处理以保证流程继续)
                m_bin_lengths[current_grown_bin_idx] = 0; // 保护单元没有粒子，长度为0
                particles_in_this_grown_bin = 0;          // 显式设为0
                m_slot_offsets[current_grown_bin_idx] = write_ptr;
    
                // 为保护单元的bin添加1个保证的空槽 (此空间已在m_capacity的总量中预留)
                const int total_empty_slots_to_add = 1;
                int end_of_bin_physical_space = write_ptr + total_empty_slots_to_add;
                // 确保写入指针在增加空槽后不超过PMA的总容量
                write_ptr = std::min(end_of_bin_physical_space, m_capacity);
                if(end_of_bin_physical_space>m_capacity){
                    amrex::Abort("ERROR write_ptrs in guard_cell:已达总容量上限");
                }
            }
            current_grown_bin_idx++; // 处理下一个grown_bin
        }
        // 迭代结束后进行一次性检查，确保所有grown_bin都被正确处理了
        if (current_grown_bin_idx != num_total_bins_grown) {
            fprintf(stderr, "错误 (buildPMAForGrownTile): 迭代处理的 grown_bin 数量 (%d) 与预期的总 grown_bin 数量 (%d) 不符。\n"
                            "这可能表示BoxIterator的遍历与numPts()的计数不一致，或者循环逻辑有误。\n",
                current_grown_bin_idx, num_total_bins_grown);
            // 这里可能需要更健壮的错误处理，例如抛出异常
            amrex::Abort("ERROR current_grown_bin_idxs");
        }

        // 设置最后一个bin的结束偏移量（作为哨兵值），该值也代表了PMA中当前已使用的总长度
        m_bin_offsets[num_total_bins_grown] = write_ptr;
        m_slot_offsets[num_total_bins_grown] = m_bin_offsets[num_total_bins_grown-1]+
                            m_bin_lengths[num_total_bins_grown-1];
        m_capacity = write_ptr;
        // 更新PMA中总的空槽位数
        // m_num_particles 存储的是真实粒子数 np
        // m_num_empty_slots = m_capacity - m_num_particles;

        m_was_rebuilt_this_step = true; // 标记PMA结构在本步骤中已被重建

        // 如果有 m_num_particles_in_pma 这样的成员变量，也需要更新
        // m_num_particles_in_pma = m_num_particles;
        int vlf = svcntw();
        #pragma unroll
        for (int binID = 0; binID < m_num_bins; binID+=vlf) {
            svbool_t p_bin = svwhilelt_b32(binID,m_num_bins);
            svint32_t bin_max_ip_index=svld1_s32(p_bin,&m_slot_offsets[binID+1]);
            bin_max_ip_index=svsub_n_s32_x(p_bin,bin_max_ip_index,1);
            svint32_t bin_max_ip_v=svld1_gather_s32index_s32(p_bin,
                &m_local_index[0],bin_max_ip_index);
            svst1_s32(p_bin,&m_bin_max_ip[binID],bin_max_ip_v);
        }
    }

    void buildPMAForGrownTile_HBM( 
        const int* perm,
        const int np,
        const int* offsets, // 注意：原代码中 offsets 是 int*, 这里保持 const int* 以表示不修改输入
        int num_original_bins_in_real_tile,
        const amrex::Box& real_tile_box,
        const amrex::Box& grown_tile_box,
        double gap_ratio=0.25) // 默认值可以在类方法声明中指定，例如 gap_ratio = 0.25
    { 
        // 步骤 1: 计算扩展瓦片中的总bin数量
        // grown_tile_box 中的每个单元格都被视为一个新的bin
        const int num_total_bins_grown = grown_tile_box.numPts();
        // printf("num bins %d \n",num_total_bins_grown);
        m_num_bins=num_total_bins_grown;
        // 步骤 2: 估算PMA的总容量并初始化相关数据结构
        m_num_particles = np; // PMA中实际存储的“真实”粒子总数（来自real_tile_box）

        // 容量计算公式：
        // - np: 存储真实粒子的空间
        // - np * gap_ratio: 为真实粒子按比例额外分配的空槽
        // - num_total_bins_grown: 为扩展瓦片中的每一个bin（无论真实或保护）保证至少一个空槽
        m_capacity = static_cast<int>(std::ceil(static_cast<double>(np) * (1.0 + gap_ratio)) + 2*num_total_bins_grown);

        // 初始化PMA的底层存储区，所有位置填充无效粒子ID
        // m_local_index_hbm.reserve(m_capacity+1);
        // m_local_index_hbm.resize(m_capacity);
        m_local_index_hbm.assign(m_capacity, INVALID_PARTICLE_ID);
        // shrink_vector_capacity(m_local_index_hbm);

        // m_local_index_hbm.shrink_to_fit();
        // std::fill_n(m_local_index_hbm.data(), m_capacity, INVALID_PARTICLE_ID);
        // m_local_index_hbm.assign(m_capacity, INVALID_PARTICLE_ID);
        // m_pid_to_index_map.assign(perm,perm+np);

        // 调整bin偏移量数组和bin长度数组的大小以适应扩展瓦片
        m_bin_offsets.resize(num_total_bins_grown + 1); // +1 是为最后一个bin的结束偏移（哨兵）
        m_slot_offsets.resize(num_total_bins_grown + 1); // +1 是为最后一个bin的结束偏移（哨兵）
        m_bin_lengths.clear();
        m_bin_lengths.resize(num_total_bins_grown);  // 所有bin的初始粒子数设为0
        // m_free_slots.resize(num_total_bins_grown);
        // 创建一个模板内部vector，它已经预留了至少1个空间
        // std::vector<int> inner_vector_template;
        // inner_vector_template.reserve(1); // 或者使用上面更精确的 reserve_at_least 计算

        // 使用 assign 将外部vector resize 并用模板内部vector的拷贝来填充
        // m_free_slots.assign(num_total_bins_grown, inner_vector_template);

        // 初始化粒子ID到其所在新bin索引的反向映射表
        // 大小仍为np，因为只有这些粒子需要映射。用-1表示未映射或无效。
        if (np > 0) { // 仅当有粒子时才分配和初始化
            m_pid_to_bin_map_hbm.resize(np);
            m_pid_to_index_map.resize(np);
        } else {
            m_pid_to_bin_map_hbm.clear(); // 如果没有粒子，清空map
            m_pid_to_index_map.clear(); // 如果没有粒子，清空map
            amrex::Abort("Error: NO particle at m_pid_to_bin_map_hbm in buildPMAForGrownTile");
        }


        // 步骤 3: 逐个处理扩展瓦片中的所有bin，填充粒子和空槽
        int write_ptr = 0;                 // 当前在 m_local_index_hbm 中的写入位置指针
        int current_grown_bin_idx = 0;     // 扩展瓦片中当前处理的bin的线性索引 (0-based)
        m_slot_offsets[0] = 0;

        // 获取真实瓦片的左下角坐标和各维度长度，用于后续计算原始bin的索引
        const amrex::IntVect real_tile_lo = real_tile_box.smallEnd();
        const amrex::IntVect real_tile_lengths = real_tile_box.length(); // 获取各维度长度 (Lx, Ly, Lz)

        // 使用AMReX的BoxIterator遍历grown_tile_box中的所有单元格
        // BoxIterator通常采用Fortran顺序（列主序：x索引变化最快，然后是y，然后是z）
        for (amrex::BoxIterator bit(grown_tile_box); bit.ok(); ++bit) {
            const amrex::IntVect current_cell_iv = bit(); // 获取当前单元格在grown_tile_box中的三维坐标

            // 记录当前grown_bin在PMA (m_local_index_hbm)中的起始物理地址
            m_bin_offsets[current_grown_bin_idx] = write_ptr;
            int particles_in_this_grown_bin = 0; // 初始化当前grown_bin中的粒子数为0

            if (real_tile_box.contains(current_cell_iv)) {
                // --- 情况1：当前单元格位于真实瓦片区域 (real_tile_box) ---

                // 将当前单元格的三维坐标 (current_cell_iv, 相对于grown_tile_box)
                // 转换为其在 real_tile_box 中的原始局部一维bin索引 (original_real_bin_idx)。
                // 这个转换的线性化顺序必须与生成输入参数 `perm` 和 `offsets` 时所用的顺序完全一致。
                amrex::IntVect rel_iv = current_cell_iv - real_tile_lo; // 计算相对于real_tile_box左下角点的相对坐标(dx, dy, dz)
                int original_real_bin_idx = 0; // 0-based

                // --- 选择与您的 offsets 生成方式匹配的线性化方法 ---
                // 选项 A: Fortran顺序 (列主序: x变化最快, AMReX默认方式)
                // original_real_bin_idx = rel_iv[0]; // x分量
                // #if (AMREX_SPACEDIM >= 2)
                // original_real_bin_idx += rel_iv[1] * real_tile_lengths[0]; // y分量 * Lx
                // #endif
                // #if (AMREX_SPACEDIM == 3)
                // original_real_bin_idx += rel_iv[2] * (real_tile_lengths[0] * real_tile_lengths[1]); // z分量 * Lx * Ly
                // #endif

                // 选项 B: C语言顺序 (行主序: z变化最慢, x变化最快 - 假设是 (z,y,x) 的嵌套)
                // (即 idx = z_coord * (Lx * Ly) + y_coord * Lx + x_coord)
                original_real_bin_idx = rel_iv[0]; // x分量 (dx)
                original_real_bin_idx += rel_iv[1] * real_tile_lengths[0]; // y分量 * Lx (dy * Lx)
                // 确保这里的 real_tile_lengths 索引正确对应维度
                // 如果 real_tile_lengths[0]是x方向长度, real_tile_lengths[1]是y方向长度
                original_real_bin_idx += rel_iv[2] * (real_tile_lengths[0] * real_tile_lengths[1]); // z分量 * (Lx * Ly) (dz * Lx * Ly)
                // --- 结束线性化方法选择 ---

                // 安全检查：确保计算出的original_real_bin_idx在有效范围内
                if (original_real_bin_idx < 0 || original_real_bin_idx >= num_original_bins_in_real_tile) {
                    fprintf(stderr, "错误 (buildPMAForGrownTile): 计算得到的 original_real_bin_idx %d 超出有效范围 [0, %d)。\n"
                                    "       grown_cell_iv: (%d,%d,%d), rel_iv: (%d,%d,%d)\n",
                            original_real_bin_idx, num_original_bins_in_real_tile,
                            current_cell_iv[0], current_cell_iv[1], (AMREX_SPACEDIM==3 ? current_cell_iv[2] : 0),
                            rel_iv[0], rel_iv[1], (AMREX_SPACEDIM==3 ? rel_iv[2] : 0));
                    // 发生此错误通常表示线性化逻辑不匹配或Box定义问题。
                    // 为简单起见，这里跳转到保护单元的处理逻辑，避免进一步错误。
                    amrex::Abort("ERROR original_real_bin_idx:线性化逻辑不匹配或Box定义问题");
                    // goto handle_as_guard_cell_on_error;
                }
                // 从 offsets 数组获取这个原始bin所包含的粒子在 perm 数组中的起止索引
                const int p_start_idx_in_perm = offsets[original_real_bin_idx];
                const int p_end_idx_in_perm   = offsets[original_real_bin_idx + 1];
                particles_in_this_grown_bin   = p_end_idx_in_perm - p_start_idx_in_perm; // 计算该原始bin中的粒子数
                
                if (particles_in_this_grown_bin > 0) {
                    const int* particle_ids_to_copy = perm + p_start_idx_in_perm; // 获取待拷贝粒子ID数据段的起始指针
    
                    // 再次检查是否有足够容量拷贝这些粒子 (主要用于调试，理论上容量计算应该避免此问题)
                    if (write_ptr + particles_in_this_grown_bin > m_capacity) {
                        fprintf(stderr, "错误 (buildPMAForGrownTile): 为真实瓦片区域的bin (grown_idx %d, orig_idx %d) 分配PMA时容量不足!\n"
                                        "       容量: %d, 当前写入点: %d, 需要空间: %d, 实际粒子数: %d\n",
                                current_grown_bin_idx, original_real_bin_idx, m_capacity, write_ptr,
                                write_ptr + particles_in_this_grown_bin, particles_in_this_grown_bin);
                        // 简化处理：如果容量不足，则不拷贝这些粒子，并将该bin的粒子数记为0。
                        // 实际应用中可能需要更复杂的错误处理机制，例如重新分配或抛出异常。
                        particles_in_this_grown_bin = 0;
                    } else {
                        // 将粒子ID从 perm 数组拷贝到 m_local_index_hbm 的当前写入位置
                        std::copy(particle_ids_to_copy, 
                            particle_ids_to_copy + particles_in_this_grown_bin, 
                            m_local_index_hbm.begin() + write_ptr);
    
                        // 更新 m_pid_to_bin_map_hbm：将每个成功拷贝的粒子ID映射到当前的 grown_bin_idx
                        if (!m_pid_to_bin_map_hbm.empty()) { // 仅当map有效时操作
                            for (int i = 0; i < particles_in_this_grown_bin; ++i) {
                                int pid = particle_ids_to_copy[i];
                                // 假设粒子ID是非负的，并且小于np (m_pid_to_bin_map_hbm的大小)
                                if (pid >= 0 && static_cast<size_t>(pid) < m_pid_to_bin_map_hbm.size()) {
                                    m_pid_to_bin_map_hbm[pid] = current_grown_bin_idx;
                                    m_pid_to_index_map[pid] = write_ptr+i;
                                } else {
                                    fprintf(stderr, "警告 (buildPMAForGrownTile): 粒子ID %d 对于 m_pid_to_bin_map_hbm (大小 %zu) 无效 (grown_bin_idx %d)。\n",
                                        pid, m_pid_to_bin_map_hbm.size(), current_grown_bin_idx);
                                }
                            }
                        }
                    }
                }
                m_bin_lengths[current_grown_bin_idx] = particles_in_this_grown_bin; // 更新当前grown_bin的有效粒子数
                write_ptr += particles_in_this_grown_bin;                           // 移动写入指针到这批粒子数据之后
                // 更新空槽的位置到尾巴
                m_slot_offsets[current_grown_bin_idx] = write_ptr;

                // 为当前真实瓦片区域的bin添加空槽：
                // 1. 基于实际粒子数的 gap_ratio 比例计算的空槽 (如果bin中无粒子，则此项为0)
                // 2. 再加上额外的1个保证空槽 (这是对所有grown_bin的要求)
                const int particle_gap_slots = (particles_in_this_grown_bin > 0) ? static_cast<int>(std::ceil(static_cast<double>(particles_in_this_grown_bin) * gap_ratio)) : 0;
                const int total_empty_slots_to_add = particle_gap_slots + 1; // 总共需要添加的空槽数
                // m_max_bin_capacity=std::max(m_max_bin_capacity,total_empty_slots_to_add);
                //暂时不需要这个单个bin最大容量指示器

                // 计算此bin在本PMA段中应占据的结束物理位置
                write_ptr += total_empty_slots_to_add; 
                // 确保写入指针在增加空槽后不超过PMA的总容量
                write_ptr = std::min(write_ptr, m_capacity);
                if(write_ptr>m_capacity){
                    amrex::Abort("ERROR write_ptrs:已达总容量上限");
                }
                
            } else {
                // 标签，用于处理保护单元，或在上述真实单元映射计算出错时跳转到这里进行统一处理
                handle_as_guard_cell_on_error:
                // --- 情况2：当前单元格位于保护单元区域 (guard cell area) ---
                // (或者之前真实瓦片区域的索引计算出错，也按保护单元处理以保证流程继续)
                m_bin_lengths[current_grown_bin_idx] = 0; // 保护单元没有粒子，长度为0
                particles_in_this_grown_bin = 0;          // 显式设为0
                m_slot_offsets[current_grown_bin_idx] = write_ptr;
    
                // 为保护单元的bin添加1个保证的空槽 (此空间已在m_capacity的总量中预留)
                const int total_empty_slots_to_add = 1;
                int end_of_bin_physical_space = write_ptr + total_empty_slots_to_add;
                // 确保写入指针在增加空槽后不超过PMA的总容量
                write_ptr = std::min(end_of_bin_physical_space, m_capacity);
                if(end_of_bin_physical_space>m_capacity){
                    amrex::Abort("ERROR write_ptrs in guard_cell:已达总容量上限");
                }
            }
            current_grown_bin_idx++; // 处理下一个grown_bin
        }
        // 迭代结束后进行一次性检查，确保所有grown_bin都被正确处理了
        if (current_grown_bin_idx != num_total_bins_grown) {
            fprintf(stderr, "错误 (buildPMAForGrownTile): 迭代处理的 grown_bin 数量 (%d) 与预期的总 grown_bin 数量 (%d) 不符。\n"
                            "这可能表示BoxIterator的遍历与numPts()的计数不一致，或者循环逻辑有误。\n",
                current_grown_bin_idx, num_total_bins_grown);
            // 这里可能需要更健壮的错误处理，例如抛出异常
            amrex::Abort("ERROR current_grown_bin_idxs");
        }

        // 设置最后一个bin的结束偏移量（作为哨兵值），该值也代表了PMA中当前已使用的总长度
        m_bin_offsets[num_total_bins_grown] = write_ptr;
        m_slot_offsets[num_total_bins_grown] = m_bin_offsets[num_total_bins_grown-1]+
                            m_bin_lengths[num_total_bins_grown-1];
        m_capacity = write_ptr;
        // 更新PMA中总的空槽位数
        // m_num_particles 存储的是真实粒子数 np
        // m_num_empty_slots = m_capacity - m_num_particles;

        m_was_rebuilt_this_step = true; // 标记PMA结构在本步骤中已被重建

    }

    void buildPMAForGrownTile_HBM_order( 
        const int* perm,
        const int np,
        const int* offsets, // 注意：原代码中 offsets 是 int*, 这里保持 const int* 以表示不修改输入
        int num_original_bins_in_real_tile,
        const amrex::Box& real_tile_box,
        const amrex::Box& grown_tile_box,
        double gap_ratio=0.25, // 默认值可以在类方法声明中指定，例如 gap_ratio = 0.25
        int order = 1) 
    { 
        // 步骤 1: 计算扩展瓦片中的总bin数量
        // grown_tile_box 中的每个单元格都被视为一个新的bin
        const int num_total_bins_grown = grown_tile_box.numPts();
        // printf("num bins %d \n",num_total_bins_grown);
        m_num_bins=num_total_bins_grown;
        // 步骤 2: 估算PMA的总容量并初始化相关数据结构
        m_num_particles = np; // PMA中实际存储的“真实”粒子总数（来自real_tile_box）

        // 容量计算公式：
        // - np: 存储真实粒子的空间
        // - np * gap_ratio: 为真实粒子按比例额外分配的空槽
        // - num_total_bins_grown: 为扩展瓦片中的每一个bin（无论真实或保护）保证至少一个空槽
        m_capacity = static_cast<int>(std::ceil(static_cast<double>(np) * (1.0 + gap_ratio)) + 2*num_total_bins_grown);

        // 初始化PMA的底层存储区，所有位置填充无效粒子ID
        m_local_index_hbm.assign(m_capacity, INVALID_PARTICLE_ID);
        // shrink_vector_capacity(m_local_index_hbm);

        // 调整bin偏移量数组和bin长度数组的大小以适应扩展瓦片
        m_bin_offsets.resize(num_total_bins_grown + 1); // +1 是为最后一个bin的结束偏移（哨兵）
        m_slot_offsets.resize(num_total_bins_grown + 1); // +1 是为最后一个bin的结束偏移（哨兵）
        m_bin_lengths.clear();
        m_bin_lengths.resize(num_total_bins_grown);  // 所有bin的初始粒子数设为0

        // 初始化粒子ID到其所在新bin索引的反向映射表
        // 大小仍为np，因为只有这些粒子需要映射。用-1表示未映射或无效。
        if (np > 0) { // 仅当有粒子时才分配和初始化
            m_pid_to_bin_map_hbm.resize(np);
            m_pid_to_index_map.resize(np);
        } else {
            m_pid_to_bin_map_hbm.clear(); // 如果没有粒子，清空map
            m_pid_to_index_map.clear(); // 如果没有粒子，清空map
            amrex::Abort("Error: NO particle at m_pid_to_bin_map_hbm in buildPMAForGrownTile");
        }


        // 步骤 3: 逐个处理扩展瓦片中的所有bin，填充粒子和空槽
        int write_ptr = 0;                 // 当前在 m_local_index_hbm 中的写入位置指针
        int current_grown_bin_idx = 0;     // 扩展瓦片中当前处理的bin的线性索引 (0-based)
        m_slot_offsets[0] = 0;

        // 获取真实瓦片的左下角坐标和各维度长度，用于后续计算原始bin的索引
        const amrex::IntVect real_tile_lo = real_tile_box.smallEnd();
        const amrex::IntVect real_tile_lengths = real_tile_box.length(); // 获取各维度长度 (Lx, Ly, Lz)

        const int lo = amrex::Math::floor(0.4*order);   // order=1→0, order=3→1
        const amrex::IntVect lo_shift(AMREX_D_DECL(lo,lo,lo));

        // 使用AMReX的BoxIterator遍历grown_tile_box中的所有单元格
        // BoxIterator通常采用Fortran顺序（列主序：x索引变化最快，然后是y，然后是z）
        for (amrex::BoxIterator bit(grown_tile_box); bit.ok(); ++bit) {
            const amrex::IntVect current_cell_iv = bit(); // 获取当前单元格在grown_tile_box中的三维坐标

            // 记录当前grown_bin在PMA (m_local_index_hbm)中的起始物理地址
            m_bin_offsets[current_grown_bin_idx] = write_ptr;
            int particles_in_this_grown_bin = 0; // 初始化当前grown_bin中的粒子数为0
            // 对于三阶映射，当前目标单元格(current_cell_iv) 应该接收那些以它为“左下角”的源单元格中的粒子。
            // 这意味着源单元格的索引是 source_cell_iv = current_cell_iv + (1,1,1)。
            // 我们需要反向计算出这个源单元格的索引。
            const amrex::IntVect source_cell_iv = current_cell_iv + lo_shift;

            if (real_tile_box.contains(source_cell_iv)) {
                // --- 情况1：当前单元格位于真实瓦片区域 (real_tile_box) ---
                // 只对“平移后仍在 real 内”的 bin，才允许去取 offsets/perm

                // 将当前单元格的三维坐标 (current_cell_iv, 相对于grown_tile_box)
                // 转换为其在 real_tile_box 中的原始局部一维bin索引 (original_real_bin_idx)。
                // 这个转换的线性化顺序必须与生成输入参数 `perm` 和 `offsets` 时所用的顺序完全一致。
                amrex::IntVect rel_iv = source_cell_iv - real_tile_lo; // 计算相对于real_tile_box左下角点的相对坐标(dx, dy, dz)
                
                // C语言顺序 (行主序: z变化最慢, x变化最快 - 假设是 (z,y,x) 的嵌套)
                // (即 idx = z_coord * (Lx * Ly) + y_coord * Lx + x_coord)
                int original_real_bin_idx = rel_iv[0];
                original_real_bin_idx += rel_iv[1] * real_tile_lengths[0];
                #if (AMREX_SPACEDIM > 2)
                    original_real_bin_idx += rel_iv[2] * (real_tile_lengths[0] * real_tile_lengths[1]);
                #endif

                // 安全检查：确保计算出的original_real_bin_idx在有效范围内
                if (original_real_bin_idx < 0 || original_real_bin_idx >= num_original_bins_in_real_tile) {
                    fprintf(stderr, "错误 (buildPMAForGrownTile): 计算得到的 original_real_bin_idx %d 超出有效范围 [0, %d)。\n"
                                    "       grown_cell_iv: (%d,%d,%d), rel_iv: (%d,%d,%d)\n",
                            original_real_bin_idx, num_original_bins_in_real_tile,
                            current_cell_iv[0], current_cell_iv[1], (AMREX_SPACEDIM==3 ? current_cell_iv[2] : 0),
                            rel_iv[0], rel_iv[1], (AMREX_SPACEDIM==3 ? rel_iv[2] : 0));
                    // 发生此错误通常表示线性化逻辑不匹配或Box定义问题。
                    // 为简单起见，这里跳转到保护单元的处理逻辑，避免进一步错误。
                    amrex::Abort("ERROR original_real_bin_idx:线性化逻辑不匹配或Box定义问题");
                    // goto handle_as_guard_cell_on_error;
                }
                // 从 offsets 数组获取这个原始bin所包含的粒子在 perm 数组中的起止索引
                const int p_start_idx_in_perm = offsets[original_real_bin_idx];
                const int p_end_idx_in_perm   = offsets[original_real_bin_idx + 1];
                particles_in_this_grown_bin   = p_end_idx_in_perm - p_start_idx_in_perm; // 计算该原始bin中的粒子数
                
                if (particles_in_this_grown_bin > 0) {
                    const int* particle_ids_to_copy = perm + p_start_idx_in_perm; // 获取待拷贝粒子ID数据段的起始指针
    
                    // 再次检查是否有足够容量拷贝这些粒子 (主要用于调试，理论上容量计算应该避免此问题)
                    if (write_ptr + particles_in_this_grown_bin > m_capacity) {
                        fprintf(stderr, "错误 (buildPMAForGrownTile): 为真实瓦片区域的bin (grown_idx %d, orig_idx %d) 分配PMA时容量不足!\n"
                                        "       容量: %d, 当前写入点: %d, 需要空间: %d, 实际粒子数: %d\n",
                                current_grown_bin_idx, original_real_bin_idx, m_capacity, write_ptr,
                                write_ptr + particles_in_this_grown_bin, particles_in_this_grown_bin);
                        // 简化处理：如果容量不足，则不拷贝这些粒子，并将该bin的粒子数记为0。
                        // 实际应用中可能需要更复杂的错误处理机制，例如重新分配或抛出异常。
                        // particles_in_this_grown_bin = 0;
                        amrex::Abort("PMA capacity exceeded while copying particles.");
                    } 
                    // 将粒子ID从 perm 数组拷贝到 m_local_index_hbm 的当前写入位置
                    std::copy(particle_ids_to_copy, 
                        particle_ids_to_copy + particles_in_this_grown_bin, 
                        m_local_index_hbm.begin() + write_ptr);

                    // 更新 m_pid_to_bin_map_hbm：将每个成功拷贝的粒子ID映射到当前的 grown_bin_idx
                    if (!m_pid_to_bin_map_hbm.empty()) { // 仅当map有效时操作
                        for (int i = 0; i < particles_in_this_grown_bin; ++i) {
                            int pid = particle_ids_to_copy[i];
                            // 假设粒子ID是非负的，并且小于np (m_pid_to_bin_map_hbm的大小)
                            if (pid >= 0 && static_cast<size_t>(pid) < m_pid_to_bin_map_hbm.size()) {
                                m_pid_to_bin_map_hbm[pid] = current_grown_bin_idx;
                                m_pid_to_index_map[pid] = write_ptr+i;
                            } else {
                                fprintf(stderr, "警告 (buildPMAForGrownTile): 粒子ID %d 对于 m_pid_to_bin_map_hbm (大小 %zu) 无效 (grown_bin_idx %d)。\n",
                                    pid, m_pid_to_bin_map_hbm.size(), current_grown_bin_idx);
                            }
                        }
                    }
                    
                }
                m_bin_lengths[current_grown_bin_idx] = particles_in_this_grown_bin; // 更新当前grown_bin的有效粒子数
                write_ptr += particles_in_this_grown_bin;                           // 移动写入指针到这批粒子数据之后
                // 更新空槽的位置到尾巴
                m_slot_offsets[current_grown_bin_idx] = write_ptr;

                // 为当前真实瓦片区域的bin添加空槽：
                // 1. 基于实际粒子数的 gap_ratio 比例计算的空槽 (如果bin中无粒子，则此项为0)
                // 2. 再加上额外的1个保证空槽 (这是对所有grown_bin的要求)
                const int particle_gap_slots = (particles_in_this_grown_bin > 0) ? static_cast<int>(std::ceil(static_cast<double>(particles_in_this_grown_bin) * gap_ratio)) : 0;
                const int total_empty_slots_to_add = particle_gap_slots + 1; // 总共需要添加的空槽数


                // 计算此bin在本PMA段中应占据的结束物理位置
                write_ptr += total_empty_slots_to_add; 
                // 确保写入指针在增加空槽后不超过PMA的总容量
                write_ptr = std::min(write_ptr, m_capacity);
                if(write_ptr>m_capacity){
                    amrex::Abort("ERROR write_ptrs:已达总容量上限");
                }
                
            } else {
                // 标签，用于处理保护单元，或在上述真实单元映射计算出错时跳转到这里进行统一处理
                handle_as_guard_cell_on_error:
                // --- 情况2：当前单元格位于保护单元区域 (guard cell area) ---
                // (或者之前真实瓦片区域的索引计算出错，也按保护单元处理以保证流程继续)
                m_bin_lengths[current_grown_bin_idx] = 0; // 保护单元没有粒子，长度为0
                particles_in_this_grown_bin = 0;          // 显式设为0
                m_slot_offsets[current_grown_bin_idx] = write_ptr;
    
                // 为保护单元的bin添加1个保证的空槽 (此空间已在m_capacity的总量中预留)
                const int total_empty_slots_to_add = 1;
                int end_of_bin_physical_space = write_ptr + total_empty_slots_to_add;
                // 确保写入指针在增加空槽后不超过PMA的总容量
                write_ptr = std::min(end_of_bin_physical_space, m_capacity);
                if(end_of_bin_physical_space>m_capacity){
                    amrex::Abort("ERROR write_ptrs in guard_cell:已达总容量上限");
                }
            }
            current_grown_bin_idx++; // 处理下一个grown_bin
        }
        // 迭代结束后进行一次性检查，确保所有grown_bin都被正确处理了
        if (current_grown_bin_idx != num_total_bins_grown) {
            fprintf(stderr, "错误 (buildPMAForGrownTile): 迭代处理的 grown_bin 数量 (%d) 与预期的总 grown_bin 数量 (%d) 不符。\n"
                            "这可能表示BoxIterator的遍历与numPts()的计数不一致，或者循环逻辑有误。\n",
                current_grown_bin_idx, num_total_bins_grown);
            // 这里可能需要更健壮的错误处理，例如抛出异常
            amrex::Abort("ERROR current_grown_bin_idxs");
        }

        // 设置最后一个bin的结束偏移量（作为哨兵值），该值也代表了PMA中当前已使用的总长度
        m_bin_offsets[num_total_bins_grown] = write_ptr;
        m_slot_offsets[num_total_bins_grown] = m_bin_offsets[num_total_bins_grown-1]+
                            m_bin_lengths[num_total_bins_grown-1];
        m_capacity = write_ptr;
        // 更新PMA中总的空槽位数
        // m_num_particles 存储的是真实粒子数 np
        // m_num_empty_slots = m_capacity - m_num_particles;

        m_was_rebuilt_this_step = true; // 标记PMA结构在本步骤中已被重建

    }

    void buildPMAForGrownTile_order3( 
        const int* perm,
        const int np,
        const int* offsets, // 注意：原代码中 offsets 是 int*, 这里保持 const int* 以表示不修改输入
        int num_original_bins_in_real_tile,
        const amrex::Box& real_tile_box,
        const amrex::Box& grown_tile_box,
        double gap_ratio=0.25) // 默认值可以在类方法声明中指定，例如 gap_ratio = 0.25
    { 
        // 步骤 1: 计算扩展瓦片中的总bin数量
        // grown_tile_box 中的每个单元格都被视为一个新的bin
        const int num_total_bins_grown = grown_tile_box.numPts();
        // printf("num bins %d \n",num_total_bins_grown);
        m_num_bins=num_total_bins_grown;
        // 步骤 2: 估算PMA的总容量并初始化相关数据结构
        m_num_particles = np; // PMA中实际存储的“真实”粒子总数（来自real_tile_box）

        // 容量计算公式：
        // - np: 存储真实粒子的空间
        // - np * gap_ratio: 为真实粒子按比例额外分配的空槽
        // - num_total_bins_grown: 为扩展瓦片中的每一个bin（无论真实或保护）保证至少一个空槽
        m_capacity = static_cast<int>(std::ceil(static_cast<double>(np) * (1.0 + gap_ratio)) + 2*num_total_bins_grown);

        // 初始化PMA的底层存储区，所有位置填充无效粒子ID
        m_local_index.reserve(m_capacity+1);
        std::fill_n(m_local_index.data(), m_capacity, INVALID_PARTICLE_ID);
        // m_local_index.assign(m_capacity, INVALID_PARTICLE_ID);
        // m_pid_to_index_map.assign(perm,perm+np);

        // 调整bin偏移量数组和bin长度数组的大小以适应扩展瓦片
        m_bin_offsets.resize(num_total_bins_grown + 1); // +1 是为最后一个bin的结束偏移（哨兵）
        m_slot_offsets.resize(num_total_bins_grown + 1); // +1 是为最后一个bin的结束偏移（哨兵）
        m_bin_lengths.clear();
        m_bin_lengths.resize(num_total_bins_grown);  // 所有bin的初始粒子数设为0
        // m_free_slots.resize(num_total_bins_grown);
        // 创建一个模板内部vector，它已经预留了至少1个空间
        // std::vector<int> inner_vector_template;
        // inner_vector_template.reserve(1); // 或者使用上面更精确的 reserve_at_least 计算

        // 使用 assign 将外部vector resize 并用模板内部vector的拷贝来填充
        // m_free_slots.assign(num_total_bins_grown, inner_vector_template);

        // 初始化粒子ID到其所在新bin索引的反向映射表
        // 大小仍为np，因为只有这些粒子需要映射。用-1表示未映射或无效。
        if (np > 0) { // 仅当有粒子时才分配和初始化
            m_pid_to_bin_map.resize(np);
            m_pid_to_index_map.resize(np);
        } else {
            m_pid_to_bin_map.clear(); // 如果没有粒子，清空map
            m_pid_to_index_map.clear(); // 如果没有粒子，清空map
            amrex::Abort("Error: NO particle at m_pid_to_bin_map in buildPMAForGrownTile");
        }


        // 步骤 3: 逐个处理扩展瓦片中的所有bin，填充粒子和空槽
        int write_ptr = 0;                 // 当前在 m_local_index 中的写入位置指针
        int current_grown_bin_idx = 0;     // 扩展瓦片中当前处理的bin的线性索引 (0-based)
        m_slot_offsets[0] = 0;
        m_bin_offsets[0] = 0;

        // 获取真实瓦片的左下角坐标和各维度长度，用于后续计算原始bin的索引
        const amrex::IntVect real_tile_lo = real_tile_box.smallEnd();
        const amrex::IntVect real_tile_lengths = real_tile_box.length(); // 获取各维度长度 (Lx, Ly, Lz)
        const amrex::IntVect grow_tile_lengths = grown_tile_box.length(); // 获取各维度长度 (Lx, Ly, Lz)

        // 使用AMReX的BoxIterator遍历grown_tile_box中的所有单元格
        // BoxIterator通常采用Fortran顺序（列主序：x索引变化最快，然后是y，然后是z）
        for (amrex::BoxIterator bit(grown_tile_box); bit.ok(); ++bit) {
            const amrex::IntVect current_cell_iv = bit(); // 获取当前单元格在grown_tile_box中的三维坐标
            int order3_bin_idx = (current_cell_iv[0]-1) + 
                                (current_cell_iv[1]-1) * grow_tile_lengths[0]+
                                (current_cell_iv[2]-1) * (grow_tile_lengths[0] * grow_tile_lengths[1]); 
            int current_grown_bin_idx = (current_cell_iv[0]) + 
                                        (current_cell_iv[1]) * grow_tile_lengths[0]+
                                        (current_cell_iv[2]) * (grow_tile_lengths[0] * grow_tile_lengths[1]); 

            
            int particles_in_this_grown_bin = 0; // 初始化当前grown_bin中的粒子数为0

            if (real_tile_box.contains(current_cell_iv)) {
                // --- 情况1：当前单元格位于真实瓦片区域 (real_tile_box) ---

                // 将当前单元格的三维坐标 (current_cell_iv, 相对于grown_tile_box)
                // 转换为其在 real_tile_box 中的原始局部一维bin索引 (original_real_bin_idx)。
                // 这个转换的线性化顺序必须与生成输入参数 `perm` 和 `offsets` 时所用的顺序完全一致。
                amrex::IntVect rel_iv = current_cell_iv - real_tile_lo; // 计算相对于real_tile_box左下角点的相对坐标(dx, dy, dz)
                int original_real_bin_idx_org = 0; // 0-based

                // --- 选择与您的 offsets 生成方式匹配的线性化方法 ---
                // 选项 A: Fortran顺序 (列主序: x变化最快, AMReX默认方式)
                // original_real_bin_idx = rel_iv[0]; // x分量
                // #if (AMREX_SPACEDIM >= 2)
                // original_real_bin_idx += rel_iv[1] * real_tile_lengths[0]; // y分量 * Lx
                // #endif
                // #if (AMREX_SPACEDIM == 3)
                // original_real_bin_idx += rel_iv[2] * (real_tile_lengths[0] * real_tile_lengths[1]); // z分量 * Lx * Ly
                // #endif

                // 选项 B: C语言顺序 (行主序: z变化最慢, x变化最快 - 假设是 (z,y,x) 的嵌套)
                // (即 idx = z_coord * (Lx * Ly) + y_coord * Lx + x_coord)
                original_real_bin_idx_org = rel_iv[0]; // x分量 (dx)
                original_real_bin_idx_org += rel_iv[1] * real_tile_lengths[0]; // y分量 * Lx (dy * Lx)
                // 确保这里的 real_tile_lengths 索引正确对应维度
                // 如果 real_tile_lengths[0]是x方向长度, real_tile_lengths[1]是y方向长度
                original_real_bin_idx_org += rel_iv[2] * (real_tile_lengths[0] * real_tile_lengths[1]); // z分量 * (Lx * Ly) (dz * Lx * Ly)

                // --- 结束线性化方法选择 ---

                // 安全检查：确保计算出的original_real_bin_idx_org在有效范围内
                if (original_real_bin_idx_org < 0 || original_real_bin_idx_org >= num_original_bins_in_real_tile) {
                    fprintf(stderr, "错误 (buildPMAForGrownTile): 计算得到的 original_real_bin_idx_org %d 超出有效范围 [0, %d)。\n"
                                    "       grown_cell_iv: (%d,%d,%d), rel_iv: (%d,%d,%d)\n",
                            original_real_bin_idx_org, num_original_bins_in_real_tile,
                            current_cell_iv[0], current_cell_iv[1], (AMREX_SPACEDIM==3 ? current_cell_iv[2] : 0),
                            rel_iv[0], rel_iv[1], (AMREX_SPACEDIM==3 ? rel_iv[2] : 0));
                    // 发生此错误通常表示线性化逻辑不匹配或Box定义问题。
                    // 为简单起见，这里跳转到保护单元的处理逻辑，避免进一步错误。
                    amrex::Abort("ERROR original_real_bin_idx_org:线性化逻辑不匹配或Box定义问题");
                    // goto handle_as_guard_cell_on_error;
                }
                // 从 offsets 数组获取这个原始bin所包含的粒子在 perm 数组中的起止索引
                const int p_start_idx_in_perm = offsets[original_real_bin_idx_org];
                const int p_end_idx_in_perm   = offsets[original_real_bin_idx_org + 1];
                particles_in_this_grown_bin   = p_end_idx_in_perm - p_start_idx_in_perm; // 计算该原始bin中的粒子数
                
                if (particles_in_this_grown_bin > 0) {
                    const int* particle_ids_to_copy = perm + p_start_idx_in_perm; // 获取待拷贝粒子ID数据段的起始指针
    
                    // 再次检查是否有足够容量拷贝这些粒子 (主要用于调试，理论上容量计算应该避免此问题)
                    if (write_ptr + particles_in_this_grown_bin > m_capacity) {
                        fprintf(stderr, "错误 (buildPMAForGrownTile): 为真实瓦片区域的bin (grown_idx %d, orig_idx %d) 分配PMA时容量不足!\n"
                                        "       容量: %d, 当前写入点: %d, 需要空间: %d, 实际粒子数: %d\n",
                                current_grown_bin_idx, original_real_bin_idx_org, m_capacity, write_ptr,
                                write_ptr + particles_in_this_grown_bin, particles_in_this_grown_bin);
                        // 简化处理：如果容量不足，则不拷贝这些粒子，并将该bin的粒子数记为0。
                        // 实际应用中可能需要更复杂的错误处理机制，例如重新分配或抛出异常。
                        particles_in_this_grown_bin = 0;
                    } else {
                        // 将粒子ID从 perm 数组拷贝到 m_local_index 的当前写入位置
                        std::copy(particle_ids_to_copy, 
                            particle_ids_to_copy + particles_in_this_grown_bin, 
                            m_local_index.begin() + write_ptr);
    
                        // 更新 m_pid_to_bin_map：将每个成功拷贝的粒子ID映射到当前的 grown_bin_idx
                        if (!m_pid_to_bin_map.empty()) { // 仅当map有效时操作
                            for (int i = 0; i < particles_in_this_grown_bin; ++i) {
                                int pid = particle_ids_to_copy[i];
                                // 假设粒子ID是非负的，并且小于np (m_pid_to_bin_map的大小)
                                if (pid >= 0 && static_cast<size_t>(pid) < m_pid_to_bin_map.size()) {
                                    m_pid_to_bin_map[pid] = current_grown_bin_idx;
                                    m_pid_to_index_map[pid] = write_ptr+i;
                                } else {
                                    fprintf(stderr, "警告 (buildPMAForGrownTile): 粒子ID %d 对于 m_pid_to_bin_map (大小 %zu) 无效 (grown_bin_idx %d)。\n",
                                        pid, m_pid_to_bin_map.size(), current_grown_bin_idx);
                                }
                            }
                        }
                    }
                }
                m_bin_lengths[current_grown_bin_idx] = particles_in_this_grown_bin; // 更新当前grown_bin的有效粒子数
                write_ptr += particles_in_this_grown_bin;                           // 移动写入指针到这批粒子数据之后
                // 更新空槽的位置到尾巴
                m_slot_offsets[current_grown_bin_idx] = write_ptr;

                // 为当前真实瓦片区域的bin添加空槽：
                // 1. 基于实际粒子数的 gap_ratio 比例计算的空槽 (如果bin中无粒子，则此项为0)
                // 2. 再加上额外的1个保证空槽 (这是对所有grown_bin的要求)
                const int particle_gap_slots = (particles_in_this_grown_bin > 0) ? static_cast<int>(std::ceil(static_cast<double>(particles_in_this_grown_bin) * gap_ratio)) : 0;
                const int total_empty_slots_to_add = particle_gap_slots + 1; // 总共需要添加的空槽数
                // m_max_bin_capacity=std::max(m_max_bin_capacity,total_empty_slots_to_add);
                //暂时不需要这个单个bin最大容量指示器

                // 计算此bin在本PMA段中应占据的结束物理位置
                write_ptr += total_empty_slots_to_add; 
                // 确保写入指针在增加空槽后不超过PMA的总容量
                write_ptr = std::min(write_ptr, m_capacity);
                if(write_ptr>m_capacity){
                    amrex::Abort("ERROR write_ptrs:已达总容量上限");
                }
                
            } else {
                // 标签，用于处理保护单元，或在上述真实单元映射计算出错时跳转到这里进行统一处理
                handle_as_guard_cell_on_error:
                // --- 情况2：当前单元格位于保护单元区域 (guard cell area) ---
                // (或者之前真实瓦片区域的索引计算出错，也按保护单元处理以保证流程继续)
                m_bin_lengths[current_grown_bin_idx] = 0; // 保护单元没有粒子，长度为0
                particles_in_this_grown_bin = 0;          // 显式设为0
                m_slot_offsets[current_grown_bin_idx] = write_ptr;
    
                // 为保护单元的bin添加1个保证的空槽 (此空间已在m_capacity的总量中预留)
                const int total_empty_slots_to_add = 1;
                int end_of_bin_physical_space = write_ptr + total_empty_slots_to_add;
                // 确保写入指针在增加空槽后不超过PMA的总容量
                write_ptr = std::min(end_of_bin_physical_space, m_capacity);
                if(end_of_bin_physical_space>m_capacity){
                    amrex::Abort("ERROR write_ptrs in guard_cell:已达总容量上限");
                }
            }
            m_bin_offsets[current_grown_bin_idx] = write_ptr;
            // current_grown_bin_idx++; // 处理下一个grown_bin
            // 记录当前grown_bin在PMA (m_local_index)中的起始物理地址
        }
        // 迭代结束后进行一次性检查，确保所有grown_bin都被正确处理了
        if (current_grown_bin_idx != num_total_bins_grown) {
            fprintf(stderr, "错误 (buildPMAForGrownTile): 迭代处理的 grown_bin 数量 (%d) 与预期的总 grown_bin 数量 (%d) 不符。\n"
                            "这可能表示BoxIterator的遍历与numPts()的计数不一致，或者循环逻辑有误。\n",
                current_grown_bin_idx, num_total_bins_grown);
            // 这里可能需要更健壮的错误处理，例如抛出异常
            amrex::Abort("ERROR current_grown_bin_idxs");
        }

        // 设置最后一个bin的结束偏移量（作为哨兵值），该值也代表了PMA中当前已使用的总长度
        m_bin_offsets[num_total_bins_grown] = write_ptr;
        m_slot_offsets[num_total_bins_grown] = m_bin_offsets[num_total_bins_grown-1]+
                            m_bin_lengths[num_total_bins_grown];
        m_capacity = write_ptr;
        // 更新PMA中总的空槽位数
        // m_num_particles 存储的是真实粒子数 np
        // m_num_empty_slots = m_capacity - m_num_particles;

        m_was_rebuilt_this_step = true; // 标记PMA结构在本步骤中已被重建

        // 如果有 m_num_particles_in_pma 这样的成员变量，也需要更新
        // m_num_particles_in_pma = m_num_particles;
        int vlf = svcntw();
        #pragma unroll
        for (int binID = 0; binID < m_num_bins; binID+=vlf) {
            svbool_t p_bin = svwhilelt_b32(binID,m_num_bins);
            svint32_t bin_max_ip_index=svld1_s32(p_bin,&m_slot_offsets[binID+1]);
            bin_max_ip_index=svsub_n_s32_x(p_bin,bin_max_ip_index,1);
            svint32_t bin_max_ip_v=svld1_gather_s32index_s32(p_bin,
                &m_local_index[0],bin_max_ip_index);
            svst1_s32(p_bin,&m_bin_max_ip[binID],bin_max_ip_v);
        }
    }


    
#endif

#if 0
    void buildPMAForGrownTile( 
        const int* perm,
        const int np,
        const int* offsets, // 注意：原代码中 offsets 是 int*, 这里保持 const int* 以表示不修改输入
        int num_original_bins_in_real_tile,
        const amrex::Box& real_tile_box,
        const amrex::Box& grown_tile_box,
        double gap_ratio=0.25) // 默认值可以在类方法声明中指定，例如 gap_ratio = 0.25
    { 
        // 步骤 1: 计算扩展瓦片中的总bin数量
        // grown_tile_box 中的每个单元格都被视为一个新的bin
        const int num_total_bins_grown = grown_tile_box.numPts();
        // printf("num bins %d \n",num_total_bins_grown);
        m_num_bins=num_total_bins_grown;
        // 步骤 2: 估算PMA的总容量并初始化相关数据结构
        m_num_particles = np; // PMA中实际存储的“真实”粒子总数（来自real_tile_box）

        // 容量计算公式：
        // - np: 存储真实粒子的空间
        // - np * gap_ratio: 为真实粒子按比例额外分配的空槽
        // - num_total_bins_grown: 为扩展瓦片中的每一个bin（无论真实或保护）保证至少一个空槽
        m_capacity = static_cast<int>(std::ceil(static_cast<double>(np) * (1.0 + gap_ratio)) + 2*num_total_bins_grown);

        // 初始化PMA的底层存储区，所有位置填充无效粒子ID
        m_local_index.assign(m_capacity, INVALID_PARTICLE_ID);
        // m_pid_to_index_map.assign(perm,perm+np);

        // 调整bin偏移量数组和bin长度数组的大小以适应扩展瓦片
        m_bin_offsets.resize(num_total_bins_grown + 1); // +1 是为最后一个bin的结束偏移（哨兵）
        m_bin_lengths.assign(num_total_bins_grown, 0);  // 所有bin的初始粒子数设为0
        // m_free_slots.resize(num_total_bins_grown);
        // 创建一个模板内部vector，它已经预留了至少1个空间
        std::vector<int> inner_vector_template;
        inner_vector_template.reserve(1); // 或者使用上面更精确的 reserve_at_least 计算

        // 使用 assign 将外部vector resize 并用模板内部vector的拷贝来填充
        m_free_slots.assign(num_total_bins_grown, inner_vector_template);

        // 初始化粒子ID到其所在新bin索引的反向映射表
        // 大小仍为np，因为只有这些粒子需要映射。用-1表示未映射或无效。
        if (np > 0) { // 仅当有粒子时才分配和初始化
            m_pid_to_bin_map.assign(np, -1);
            m_pid_to_index_map.assign(np, -1);
        } else {
            m_pid_to_bin_map.clear(); // 如果没有粒子，清空map
            m_pid_to_index_map.clear(); // 如果没有粒子，清空map
            amrex::Abort("Error: NO particle at m_pid_to_bin_map in buildPMAForGrownTile");
        }


        // 步骤 3: 逐个处理扩展瓦片中的所有bin，填充粒子和空槽
        int write_ptr = 0;                 // 当前在 m_local_index 中的写入位置指针
        int current_grown_bin_idx = 0;     // 扩展瓦片中当前处理的bin的线性索引 (0-based)

        // 获取真实瓦片的左下角坐标和各维度长度，用于后续计算原始bin的索引
        const amrex::IntVect real_tile_lo = real_tile_box.smallEnd();
        const amrex::IntVect real_tile_lengths = real_tile_box.length(); // 获取各维度长度 (Lx, Ly, Lz)

        // 使用AMReX的BoxIterator遍历grown_tile_box中的所有单元格
        // BoxIterator通常采用Fortran顺序（列主序：x索引变化最快，然后是y，然后是z）
        for (amrex::BoxIterator bit(grown_tile_box); bit.ok(); ++bit) {
            const amrex::IntVect current_cell_iv = bit(); // 获取当前单元格在grown_tile_box中的三维坐标

            // 记录当前grown_bin在PMA (m_local_index)中的起始物理地址
            m_bin_offsets[current_grown_bin_idx] = write_ptr;
            int particles_in_this_grown_bin = 0; // 初始化当前grown_bin中的粒子数为0

            if (real_tile_box.contains(current_cell_iv)) {
                // --- 情况1：当前单元格位于真实瓦片区域 (real_tile_box) ---

                // 将当前单元格的三维坐标 (current_cell_iv, 相对于grown_tile_box)
                // 转换为其在 real_tile_box 中的原始局部一维bin索引 (original_real_bin_idx)。
                // 这个转换的线性化顺序必须与生成输入参数 `perm` 和 `offsets` 时所用的顺序完全一致。
                amrex::IntVect rel_iv = current_cell_iv - real_tile_lo; // 计算相对于real_tile_box左下角点的相对坐标(dx, dy, dz)
                int original_real_bin_idx = 0; // 0-based

                // --- 选择与您的 offsets 生成方式匹配的线性化方法 ---
                // 选项 A: Fortran顺序 (列主序: x变化最快, AMReX默认方式)
                // original_real_bin_idx = rel_iv[0]; // x分量
                // #if (AMREX_SPACEDIM >= 2)
                // original_real_bin_idx += rel_iv[1] * real_tile_lengths[0]; // y分量 * Lx
                // #endif
                // #if (AMREX_SPACEDIM == 3)
                // original_real_bin_idx += rel_iv[2] * (real_tile_lengths[0] * real_tile_lengths[1]); // z分量 * Lx * Ly
                // #endif

                // 选项 B: C语言顺序 (行主序: z变化最慢, x变化最快 - 假设是 (z,y,x) 的嵌套)
                // (即 idx = z_coord * (Lx * Ly) + y_coord * Lx + x_coord)
                original_real_bin_idx = rel_iv[0]; // x分量 (dx)
    #if (AMREX_SPACEDIM >= 2)
                original_real_bin_idx += rel_iv[1] * real_tile_lengths[0]; // y分量 * Lx (dy * Lx)
    #endif
    #if (AMREX_SPACEDIM == 3)
                // 确保这里的 real_tile_lengths 索引正确对应维度
                // 如果 real_tile_lengths[0]是x方向长度, real_tile_lengths[1]是y方向长度
                original_real_bin_idx += rel_iv[2] * (real_tile_lengths[0] * real_tile_lengths[1]); // z分量 * (Lx * Ly) (dz * Lx * Ly)
    #endif
                // --- 结束线性化方法选择 ---

                // 安全检查：确保计算出的original_real_bin_idx在有效范围内
                if (original_real_bin_idx < 0 || original_real_bin_idx >= num_original_bins_in_real_tile) {
                    fprintf(stderr, "错误 (buildPMAForGrownTile): 计算得到的 original_real_bin_idx %d 超出有效范围 [0, %d)。\n"
                                    "       grown_cell_iv: (%d,%d,%d), rel_iv: (%d,%d,%d)\n",
                            original_real_bin_idx, num_original_bins_in_real_tile,
                            current_cell_iv[0], current_cell_iv[1], (AMREX_SPACEDIM==3 ? current_cell_iv[2] : 0),
                            rel_iv[0], rel_iv[1], (AMREX_SPACEDIM==3 ? rel_iv[2] : 0));
                    // 发生此错误通常表示线性化逻辑不匹配或Box定义问题。
                    // 为简单起见，这里跳转到保护单元的处理逻辑，避免进一步错误。
                    amrex::Abort("ERROR original_real_bin_idx:线性化逻辑不匹配或Box定义问题");
                    // goto handle_as_guard_cell_on_error;
                }
                // 从 offsets 数组获取这个原始bin所包含的粒子在 perm 数组中的起止索引
                const int p_start_idx_in_perm = offsets[original_real_bin_idx];
                const int p_end_idx_in_perm   = offsets[original_real_bin_idx + 1];
                particles_in_this_grown_bin   = p_end_idx_in_perm - p_start_idx_in_perm; // 计算该原始bin中的粒子数
                
                if (particles_in_this_grown_bin > 0) {
                    const int* particle_ids_to_copy = perm + p_start_idx_in_perm; // 获取待拷贝粒子ID数据段的起始指针
    
                    // 再次检查是否有足够容量拷贝这些粒子 (主要用于调试，理论上容量计算应该避免此问题)
                    if (write_ptr + particles_in_this_grown_bin > m_capacity) {
                        fprintf(stderr, "错误 (buildPMAForGrownTile): 为真实瓦片区域的bin (grown_idx %d, orig_idx %d) 分配PMA时容量不足!\n"
                                        "       容量: %d, 当前写入点: %d, 需要空间: %d, 实际粒子数: %d\n",
                                current_grown_bin_idx, original_real_bin_idx, m_capacity, write_ptr,
                                write_ptr + particles_in_this_grown_bin, particles_in_this_grown_bin);
                        // 简化处理：如果容量不足，则不拷贝这些粒子，并将该bin的粒子数记为0。
                        // 实际应用中可能需要更复杂的错误处理机制，例如重新分配或抛出异常。
                        particles_in_this_grown_bin = 0;
                    } else {
                        // 将粒子ID从 perm 数组拷贝到 m_local_index 的当前写入位置
                        std::copy(particle_ids_to_copy, particle_ids_to_copy + particles_in_this_grown_bin, m_local_index.begin() + write_ptr);
    
                        // 更新 m_pid_to_bin_map：将每个成功拷贝的粒子ID映射到当前的 grown_bin_idx
                        if (!m_pid_to_bin_map.empty()) { // 仅当map有效时操作
                            for (int i = 0; i < particles_in_this_grown_bin; ++i) {
                                int pid = particle_ids_to_copy[i];
                                // 假设粒子ID是非负的，并且小于np (m_pid_to_bin_map的大小)
                                if (pid >= 0 && static_cast<size_t>(pid) < m_pid_to_bin_map.size()) {
                                    m_pid_to_bin_map[pid] = current_grown_bin_idx;
                                    m_pid_to_index_map[pid] = write_ptr+i;
                                } else {
                                    fprintf(stderr, "警告 (buildPMAForGrownTile): 粒子ID %d 对于 m_pid_to_bin_map (大小 %zu) 无效 (grown_bin_idx %d)。\n",
                                        pid, m_pid_to_bin_map.size(), current_grown_bin_idx);
                                }
                            }
                        }
                    }
                }
                m_bin_lengths[current_grown_bin_idx] = particles_in_this_grown_bin; // 更新当前grown_bin的有效粒子数
                write_ptr += particles_in_this_grown_bin;                           // 移动写入指针到这批粒子数据之后

                // 为当前真实瓦片区域的bin添加空槽：
                // 1. 基于实际粒子数的 gap_ratio 比例计算的空槽 (如果bin中无粒子，则此项为0)
                // 2. 再加上额外的1个保证空槽 (这是对所有grown_bin的要求)
                const int particle_gap_slots = (particles_in_this_grown_bin > 0) ? static_cast<int>(std::ceil(static_cast<double>(particles_in_this_grown_bin) * gap_ratio)) : 0;
                const int total_empty_slots_to_add = particle_gap_slots + 1; // 总共需要添加的空槽数
                // m_max_bin_capacity=std::max(m_max_bin_capacity,total_empty_slots_to_add);
                //暂时不需要这个单个bin最大容量指示器

                // 记录这些空槽的物理地址
                // 这些空槽位于 [write_ptr, write_ptr + total_empty_slots_to_add - 1]
                // 并且确保不超过 m_capacity
                int actual_slots_added = 0;
                for (int i = 0; i < total_empty_slots_to_add; ++i) {
                    if (write_ptr + i < m_capacity) {
                        m_free_slots[current_grown_bin_idx].push_back(write_ptr +total_empty_slots_to_add-1- i);
                        actual_slots_added++;
                    } else {
                        amrex::Abort("ERROR free slots:已达总容量上限");
                        break; // 已达总容量上限
                    }
                }
                // write_ptr += actual_slots_added; // 更新写入指针，跳过已记录的空槽

                int end_of_bin_physical_space = write_ptr + total_empty_slots_to_add; // 计算此bin在本PMA段中应占据的结束物理位置
                // 确保写入指针在增加空槽后不超过PMA的总容量
                write_ptr = std::min(end_of_bin_physical_space, m_capacity);
                if(end_of_bin_physical_space>m_capacity){
                    amrex::Abort("ERROR write_ptrs:已达总容量上限");
                }
            } else {
                // 标签，用于处理保护单元，或在上述真实单元映射计算出错时跳转到这里进行统一处理
                handle_as_guard_cell_on_error:
                // --- 情况2：当前单元格位于保护单元区域 (guard cell area) ---
                // (或者之前真实瓦片区域的索引计算出错，也按保护单元处理以保证流程继续)
                m_bin_lengths[current_grown_bin_idx] = 0; // 保护单元没有粒子，长度为0
                particles_in_this_grown_bin = 0;          // 显式设为0
    
                // 为保护单元的bin添加1个保证的空槽 (此空间已在m_capacity的总量中预留)
                const int total_empty_slots_to_add = 1;
                int end_of_bin_physical_space = write_ptr + total_empty_slots_to_add;
                // 确保写入指针在增加空槽后不超过PMA的总容量
                write_ptr = std::min(end_of_bin_physical_space, m_capacity);
                if(end_of_bin_physical_space>m_capacity){
                    amrex::Abort("ERROR write_ptrs in guard_cell:已达总容量上限");
                }
            }
            current_grown_bin_idx++; // 处理下一个grown_bin
        }
        // 迭代结束后进行一次性检查，确保所有grown_bin都被正确处理了
        if (current_grown_bin_idx != num_total_bins_grown) {
            fprintf(stderr, "错误 (buildPMAForGrownTile): 迭代处理的 grown_bin 数量 (%d) 与预期的总 grown_bin 数量 (%d) 不符。\n"
                            "这可能表示BoxIterator的遍历与numPts()的计数不一致，或者循环逻辑有误。\n",
                current_grown_bin_idx, num_total_bins_grown);
            // 这里可能需要更健壮的错误处理，例如抛出异常
            amrex::Abort("ERROR current_grown_bin_idxs");
        }

        // 设置最后一个bin的结束偏移量（作为哨兵值），该值也代表了PMA中当前已使用的总长度
        m_bin_offsets[num_total_bins_grown] = write_ptr;
        m_capacity = write_ptr;
        // 更新PMA中总的空槽位数
        // m_num_particles 存储的是真实粒子数 np
        m_num_empty_slots = m_capacity - m_num_particles;

        m_was_rebuilt_this_step = true; // 标记PMA结构在本步骤中已被重建

        // 如果有 m_num_particles_in_pma 这样的成员变量，也需要更新
        // m_num_particles_in_pma = m_num_particles;
    }

    // gap_ratio=0.25 → 每个 bin 在末尾额外留 25% 空位
    // have a stack to identify empty slot
    void buildPMAfromDenseBins(const int* perm,
                                const int np,
                                int* offsets,
                                int numBins,
                                double gap_ratio=0.25)
    {
        // 1) 预估容量并一次性分配
        m_num_particles = np;
        m_num_bins=numBins;
        // m_num_particles = static_cast<int>(np);
        m_capacity      = static_cast<int>( std::ceil(np * (1 + gap_ratio))+numBins*2 );
        // printf("m_capacity = %d, np %d, numBins %d\n", m_capacity, np, numBins);
        m_local_index.assign(m_capacity, INVALID_PARTICLE_ID);

        m_bin_offsets.resize(numBins + 1);
        m_bin_lengths.resize(numBins);
        m_free_slots.resize(numBins);

        // --- 新增：调整并初始化反向映射表 ---
        // 大小应该基于当前的粒子数 np
        m_pid_to_bin_map.assign(np, -1); // 初始化为无效 bin

        // 2) 逐 bin 拷贝 + 插空
        int write_ptr = 0;
        for (int b = 1; b <= numBins; ++b)
        {
            const int begin = static_cast<int>(offsets[b-1]);
            const int end   = static_cast<int>(offsets[b]);
            const int n_in_bin = end - begin;
            const int current_bin_id = b - 1; // 转换为 0-based bin index

            // 起始物理地址
            m_bin_offsets[b-1] = write_ptr;

            // 拷贝有效粒子并更新 Map
            const int* read_ptr = perm + begin;
            // 检查边界，确保不会写入超出 m_local_index 容量的区域
            if (write_ptr + n_in_bin > m_capacity) {
                printf("Error: Not enough capacity during PMA build for bin %d! Cap: %d, Need: %d\n",
                        current_bin_id, m_capacity, write_ptr + n_in_bin);
                // 需要错误处理或重新计算容量
                // 这里简化：仅拷贝能放下的部分
                // int can_copy = m_capacity - write_ptr;
                // if (can_copy < 0) can_copy = 0;
                // std::copy(read_ptr, read_ptr + can_copy, m_local_index.begin() + write_ptr);
                // for (int i = 0; i < can_copy; ++i) {
                //     int ip = read_ptr[i];
                //     if (ip >= 0 && ip < m_pid_to_bin_map.size()) {
                //         m_pid_to_bin_map[ip] = current_bin_id;
                //     } else { /* Error */ }
                // }
                // m_bin_lengths[current_bin_id] = can_copy;
                // write_ptr += can_copy; // 更新写指针

            } else {
                // 容量足够
                std::copy(read_ptr, read_ptr + n_in_bin, m_local_index.begin() + write_ptr);
                // 更新 pid_to_bin_map
                for (int i = 0; i < n_in_bin; ++i) {
                    int ip = read_ptr[i];
                    if (ip >= 0 && ip < m_pid_to_bin_map.size()) {
                        m_pid_to_bin_map[ip] = current_bin_id;
                    } else {
                        printf("Error: Particle ID %d out of bounds for pid_to_bin_map (size %zu) during build.\n",
                                ip, m_pid_to_bin_map.size());
                    }
                }
                
                m_bin_lengths[current_bin_id] = n_in_bin;
                write_ptr += n_in_bin; // 更新写指针
            }

           // 3) 给本 bin 末尾插入空槽
            const int gap = static_cast<int>(std::round(n_in_bin * gap_ratio))+1;
            
            // 确保增加 gap 后不超过容量
            int end_of_block = write_ptr + gap;
            // 跳过 gap 或直到容量末尾
            // write_ptr = std::min(end_of_block, m_capacity);
            if(end_of_block>=m_capacity){
                printf("end_of_block %d m_capacity %d \n",end_of_block,m_capacity);
                amrex::Abort("ERROR:end_of_block != m_capacity!!!");
            }
            write_ptr = end_of_block;
            for(int i = write_ptr-1; i >= write_ptr-gap; --i){
                m_free_slots[b-1].push_back(i);
            }
        }
        m_bin_offsets[numBins] = write_ptr; // 终止哨兵
        m_capacity=write_ptr;
        m_num_empty_slots = m_capacity - m_num_particles;
        m_local_index.resize(m_capacity, INVALID_PARTICLE_ID);

        // 在构建结束时，记录下与此PMA结构对应的粒子数
        // m_num_particles_in_pma = m_num_particles; // 使用 build 时传入的 np
    }
    void buildPMAfromDenseBins_nogap(const int* perm,
                                const int np,
                                int* offsets,
                                int numBins,
                                double gap_ratio=0)
    {
        // 1) 预估容量并一次性分配
        m_num_particles = np;
        // m_num_particles = static_cast<int>(np);
        m_capacity      = np;
        // printf("m_capacity = %d, np %d, numBins %d\n", m_capacity, np, numBins);
        m_local_index.assign(m_capacity, INVALID_PARTICLE_ID);

        m_bin_offsets.resize(numBins + 1);
        m_bin_lengths.resize(numBins);
        m_free_slots.resize(numBins);

        // --- 新增：调整并初始化反向映射表 ---
        // 大小应该基于当前的粒子数 np
        m_pid_to_bin_map.assign(np, -1); // 初始化为无效 bin

        // 2) 逐 bin 拷贝 + 插空
        int write_ptr = 0;
        for (int b = 1; b <= numBins; ++b)
        {
            const int begin = static_cast<int>(offsets[b-1]);
            const int end   = static_cast<int>(offsets[b]);
            const int n_in_bin = end - begin;
            const int current_bin_id = b - 1; // 转换为 0-based bin index

            // 起始物理地址
            m_bin_offsets[b-1] = write_ptr;

            // 拷贝有效粒子并更新 Map
            const int* read_ptr = perm + begin;
            // 检查边界，确保不会写入超出 m_local_index 容量的区域
            if (0) {
                printf("Error: Not enough capacity during PMA build for bin %d! Cap: %d, Need: %d\n",
                        current_bin_id, m_capacity, write_ptr + n_in_bin);

            } else {
                // 容量足够
                std::copy(read_ptr, read_ptr + n_in_bin, m_local_index.begin() + write_ptr);
                // 更新 pid_to_bin_map
                for (int i = 0; i < n_in_bin; ++i) {
                    int ip = read_ptr[i];
                    if (ip >= 0 && ip < m_pid_to_bin_map.size()) {
                        m_pid_to_bin_map[ip] = current_bin_id;
                    } else {
                        printf("Error: Particle ID %d out of bounds for pid_to_bin_map (size %zu) during build.\n",
                                ip, m_pid_to_bin_map.size());
                    }
                }
                
                m_bin_lengths[current_bin_id] = n_in_bin;
                write_ptr += n_in_bin; // 更新写指针
            }

           // 3) 给本 bin 末尾插入空槽
            const int gap = 0;
            
            // 确保增加 gap 后不超过容量
            int end_of_block = write_ptr + gap;
            // 跳过 gap 或直到容量末尾
            // write_ptr = std::min(end_of_block, m_capacity);
            // if(end_of_block>=m_capacity){
            //     printf("end_of_block %d m_capacity %d \n",end_of_block,m_capacity);
            //     amrex::Abort("ERROR:end_of_block != m_capacity!!!");
            // }
            write_ptr = end_of_block;
            for(int i = write_ptr-1; i >= write_ptr-gap; --i){
                m_free_slots[b-1].push_back(i);
            }
        }
        m_bin_offsets[numBins] = write_ptr; // 终止哨兵
        m_capacity=write_ptr;
        m_num_empty_slots = m_capacity - m_num_particles;
        m_local_index.resize(m_capacity, INVALID_PARTICLE_ID);

        // 在构建结束时，记录下与此PMA结构对应的粒子数
        // m_num_particles_in_pma = m_num_particles; // 使用 build 时传入的 np
    }

    void buildPMAfromDenseBins_bak(const int* perm,
                                const int np,
                                int* offsets,
                                int numBins,
                                double gap_ratio=0.25)
    {
        // 1) 预估容量并一次性分配
        m_num_particles = np;
        // m_num_particles = static_cast<int>(np);
        m_capacity      = static_cast<int>( std::ceil(np * (1.0 + gap_ratio))+numBins );
        // printf("m_capacity = %d, np %d, numBins %d\n", m_capacity, np, numBins);
        m_local_index.assign(m_capacity, INVALID_PARTICLE_ID);

        m_bin_offsets.resize(numBins + 1);
        m_bin_lengths.resize(numBins);

        // --- 新增：调整并初始化反向映射表 ---
        // 大小应该基于当前的粒子数 np
        m_pid_to_bin_map.assign(np, -1); // 初始化为无效 bin

        // 2) 逐 bin 拷贝 + 插空
        int write_ptr = 0;
        for (int b = 1; b <= numBins; ++b)
        {
            const int begin = static_cast<int>(offsets[b-1]);
            const int end   = static_cast<int>(offsets[b]);
            const int n_in_bin = end - begin;
            const int current_bin_id = b - 1; // 转换为 0-based bin index

            // 起始物理地址
            m_bin_offsets[b-1] = write_ptr;

            // 拷贝有效粒子并更新 Map
            const int* read_ptr = perm + begin;
            // 检查边界，确保不会写入超出 m_local_index 容量的区域
            if (write_ptr + n_in_bin > m_capacity) {
                printf("Error: Not enough capacity during PMA build for bin %d! Cap: %d, Need: %d\n",
                        current_bin_id, m_capacity, write_ptr + n_in_bin);
                // 需要错误处理或重新计算容量
                // 这里简化：仅拷贝能放下的部分
                // int can_copy = m_capacity - write_ptr;
                // if (can_copy < 0) can_copy = 0;
                // std::copy(read_ptr, read_ptr + can_copy, m_local_index.begin() + write_ptr);
                // for (int i = 0; i < can_copy; ++i) {
                //     int ip = read_ptr[i];
                //     if (ip >= 0 && ip < m_pid_to_bin_map.size()) {
                //         m_pid_to_bin_map[ip] = current_bin_id;
                //     } else { /* Error */ }
                // }
                // m_bin_lengths[current_bin_id] = can_copy;
                // write_ptr += can_copy; // 更新写指针

            } else {
                // 容量足够
                std::copy(read_ptr, read_ptr + n_in_bin, m_local_index.begin() + write_ptr);
                // 更新 pid_to_bin_map
                for (int i = 0; i < n_in_bin; ++i) {
                    int ip = read_ptr[i];
                    if (ip >= 0 && ip < m_pid_to_bin_map.size()) {
                        m_pid_to_bin_map[ip] = current_bin_id;
                    } else {
                        printf("Error: Particle ID %d out of bounds for pid_to_bin_map (size %zu) during build.\n",
                                ip, m_pid_to_bin_map.size());
                    }
                }
                m_bin_lengths[current_bin_id] = n_in_bin;
                write_ptr += n_in_bin; // 更新写指针
            }

           // 3) 给本 bin 末尾插入空槽
            const int gap = static_cast<int>(std::round(n_in_bin * gap_ratio));
            // 确保增加 gap 后不超过容量
            int end_of_block = write_ptr + gap;
            write_ptr = std::min(end_of_block, m_capacity); // 跳过 gap 或直到容量末尾
        }
        m_bin_offsets[numBins] = write_ptr; // 终止哨兵
        m_num_empty_slots = m_capacity - m_num_particles;

        // 在构建结束时，记录下与此PMA结构对应的粒子数
        // m_num_particles_in_pma = m_num_particles; // 使用 build 时传入的 np
    }
# endif

    // ========== 新增: 用于后续增量修改时, 你可以写各种对 m_local_index / m_bin_offsets 的操作函数
    inline int computeBinId(int i, int j, int k, const Box& box, const int ngx=0, const int ngy=0, const int ngz=0)
    {
        // ngx, ngy, ngz 是网格的 ghost cell 数量
        const int nx = box.length(0);
        const int ny = box.length(1);

        // 按 x 方向连续排列，计算一维索引
        return (i - ngx) + (j - ngy) * nx + (k - ngz) * nx * ny;
    }
# if 0
    int findEmptySlotInBin(int bin_id) const {
        int current_size  = m_free_slots_sizes[bin_id];
        if (current_size == 0) {
            return -1;
        }
        // 1. 獲取該bin的數據在all_free_slots中的起始位置
        int start_pos = m_free_slots_offsets[bin_id];
        // 2. 計算棧頂元素（即最後一個有效元素）的確切索引
        int top_element_index = start_pos + current_size - 1;
        // 3. 從大向量中取得該值
        int insert_idx = m_all_free_slots[top_element_index];
        return insert_idx;
    }
    void slotPopBack(int bin_id){
        int current_size  = m_free_slots_sizes[bin_id];
        if (current_size == 0) {
            amrex::Abort("Error: binslot is EMPTY!");
            return;
        }
        m_free_slots_sizes[bin_id]--;
    }
#endif

# if 1
    // --- 配套的空槽管理函数示例 ---

    /**
     * @brief 从指定的bin获取一个空闲槽的物理索引。
     * 如果没有空闲槽，返回-1。
     * @param grown_bin_id 目标bin的索引 (0-based, 对应grown_tile_box的线性索引)。
     * @return 空闲槽在m_local_index中的物理索引，或-1（如果没有）。
     */
    int getFreeSlotFromBin(int grown_bin_id) {
        if (static_cast<size_t>(grown_bin_id) >= m_free_slots.size() || m_free_slots[grown_bin_id].empty()) {
            // 如果bin索引无效，或者该bin的空闲槽列表为空
            // 可以在这里触发一次 bin 扩容检查或者 PMA 重建检查的逻辑，如果需要动态调整
            // 或者简单返回-1，表示当前没有立即可用的空槽
            return -1;
        }
        int slot_idx = m_free_slots[grown_bin_id].back(); // 获取最后一个空闲槽索引 (LIFO)
        m_free_slots[grown_bin_id].pop_back();          // 从列表中移除该索引
        m_num_empty_slots--; // 总空槽数减少 (如果通过这个方式管理总数的话)
        return slot_idx;
    }
    /**
     * @brief 将一个物理索引标记为空闲槽并归还给指定的bin。
     * @param grown_bin_id 目标bin的索引。
     * @param slot_idx 要归还的空闲槽在m_local_index中的物理索引。
     */
    void returnSlotToBin(int grown_bin_id, int slot_idx) {
        if (static_cast<size_t>(grown_bin_id) < m_free_slots.size() && slot_idx >=0 && slot_idx < m_capacity) {
            m_local_index[slot_idx] = INVALID_PARTICLE_ID; // 确保该槽在底层数组中标记为无效
            m_free_slots[grown_bin_id].push_back(slot_idx); // 将索引添加回空闲列表
            m_num_empty_slots++; // 总空槽数增加 (如果通过这个方式管理总数的话)
        } else {
            fprintf(stderr, "警告 (returnSlotToBin): 无效的 grown_bin_id %d 或 slot_idx %d (容量 %d)。\n",
                    grown_bin_id, slot_idx, m_capacity);
        }
    }
    // 在指定bin的当前分配范围内查找第一个空槽
    int findEmptySlotInBin(int bin_id) const {
        // printf("binid %d size %d \n",bin_id,m_free_slots.size());
        if (bin_id < 0 || bin_id >= m_free_slots.size()) {
            printf("binid %d size %d \n",bin_id,m_free_slots.size());
            // Handle error: bin_id is out of bounds
            // e.g., print an error message, assert, or throw an exception
            amrex::Abort("Error: bin_id out of bounds in findEmptySlotInBin!");
        }
        if(m_free_slots[bin_id].empty()) {
            // amrex::Abort("ERRRRRRR!!!!!");
            return -1;
        }
        int slot_id=m_free_slots[bin_id].back();
        // m_free_slots[bin_id].pop();
        return slot_id;
    }

    // 在指定bin的当前分配范围内查找第一个空槽
    //stack version
    // int findEmptySlotInBin_stack(int bin_id) const {
    //     if(m_free_slots[bin_id].empty())return -1;
    //     int slot_id=m_free_slots[bin_id].top();
    //     // m_free_slots[bin_id].pop();
    //     return slot_id;
    // }
    int findEmptySlotInBin_bak(int bin_id) const {
        int start_idx = m_bin_offsets[bin_id];
        int end_idx = (bin_id + 1 < m_bin_offsets.size()) ? m_bin_offsets[bin_id + 1] : m_capacity;
        for (int i = start_idx; i < end_idx; ++i) {
            if (m_local_index[i] == INVALID_PARTICLE_ID) {
                return i;
            }
        }
        return -1; // 没有找到
    }

    // 查找第一个全局空槽（效率较低，破坏局部性）
    int findFirstEmptySlot(int start_search_idx = 0) const {
         for (int i = start_search_idx; i < m_capacity; ++i) {
             if (m_local_index[i] == INVALID_PARTICLE_ID) {
                 return i;
             }
         }
         // 如果从头搜索还没找到
         if (start_search_idx > 0) {
              for (int i = 0; i < start_search_idx; ++i) {
                 if (m_local_index[i] == INVALID_PARTICLE_ID) {
                     return i;
                 }
             }
         }
         return -1;
    }

    
    void insertParticleInBin(int real_idx, int i, int j, int k, const Box& box)
    {
        // int bin_id = computeBinId(i, j, k, box);
        // // 注意：建议在使用前确保 particleBins 的大小 >= 总桶数
        // particleBins[bin_id].push_back(real_idx);
    }
    
    // 删除粒子：输入粒子的 ip 以及其在网格中的下标 (i,j,k)
    // 直接根据 box 计算出桶号，从对应的桶中删除粒子的 ip
    void removeParticleFromBin(int real_idx, int i, int j, int k, const Box& box)
    {
        // int bin_id = computeBinId(i, j, k, box);
        // std::vector<int>& bin = particleBins[bin_id];
        // // 查找粒子的 ip 是否存在于桶中，并删除
        // auto it = std::find(bin.begin(), bin.end(), real_idx);
        // if (it != bin.end())
        // {
        //     bin.erase(it);
        // }
    }
# endif
    // ...更多增量更新操作...


private:

    AoS m_aos_tile;
    SoA m_soa_tile;

    bool m_defined = false;

    amrex::PODVector<ParticleReal*, Allocator<ParticleReal*> > m_runtime_r_ptrs;
    amrex::PODVector<int*, Allocator<int*> > m_runtime_i_ptrs;

    mutable amrex::PODVector<const ParticleReal*, Allocator<const ParticleReal*> > m_runtime_r_cptrs;
    mutable amrex::PODVector<const int*, Allocator<const int*> >m_runtime_i_cptrs;

    amrex::Gpu::HostVector<ParticleReal*> m_h_runtime_r_ptrs;
    amrex::Gpu::HostVector<int*> m_h_runtime_i_ptrs;

    mutable amrex::Gpu::HostVector<const ParticleReal*> m_h_runtime_r_cptrs;
    mutable amrex::Gpu::HostVector<const int*> m_h_runtime_i_cptrs;
};

} // namespace amrex

#endif // AMREX_PARTICLETILE_H_
