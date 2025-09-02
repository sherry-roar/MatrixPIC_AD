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
// #include <../OPM_Allocator.H>
#include "/pacific_ext/SYSU/sysu04/Test_install_warpx_hw/WarpX-development/Source/OPM_Allocator.H"

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
    // ========== ADDED: Local index and bin offset table within the Tile ==========
    std::vector<int> m_local_index;      // Underlying storage, contains particle ID or INVALID_PARTICLE_ID
    std::vector<int> m_bin_offsets;      // Marks the starting physical index of each bin in m_local_index (size = num_cells + 1)
    std::vector<int> m_bin_lengths;      // Stores the number of *valid* particles in each bin (size = num_cells)
    int m_num_particles;                 // Total number of particles
    int m_capacity;                      // The actual capacity of m_local_index
    int m_num_bins;
    bool m_was_rebuilt_this_step = false; // Flag to indicate if this Tile was rebuilt in the current step

    std::vector<int> m_slot_offsets;     // Aligned with bin_offsets to indicate the position of empty slots,
                                         // slot_offsets[0] points to index[0]

    static constexpr int INVALID_PARTICLE_ID = -1;

    // --- ADDED: Reverse lookup tables ---
    std::vector<int> m_pid_to_bin_map;   // Index = pid, Value = bin_id (0-based). Use -1 for invalid/unmapped.
    std::vector<int> m_pid_to_index_map; // This is essentially the permutation array (perm), a compact index array for easy statistics and access.

    // --- OPM versions of sorting arrays ---
    OPM_Inner_Vector_Int m_local_index_hbm;
    OPM_Inner_Vector_Int m_pid_to_bin_map_hbm;
    OPM_Inner_Vector_Int m_pid_to_index_map_hbm;
    OPM_Inner_Vector_Int m_bin_offsets_hbm;
    OPM_Inner_Vector_Int m_slot_offsets_hbm;
    OPM_Inner_Vector_Int m_bin_lengths_hbm;

    // The following are deprecated as of 2025.07.21
    std::vector<int> m_bin_max_ip;         // Indicates the largest particle index (ip) in each bin, used to determine insertion position
    int m_num_empty_slots;                 // Current number of empty slots
    std::vector<std::vector<int>> m_free_slots; // Simulating a stack with a vector of vectors

    // ========== ADDED: Interfaces for convenient external read/write access ==========
    // After calculating Permutation & Offsets, this set function can be called to update
    void setIndexAndOffset(const int* perm_ptr, std::size_t np,
                           const int* offset_ptr, int nbins)
    {
        // 1) Fill m_local_index based on the externally provided permutation array
        m_local_index.assign(perm_ptr, perm_ptr + np);

        // 2) Fill m_bin_offsets based on the externally provided offsets
        //    offsets array length = nbins + 1
        m_bin_offsets.assign(offset_ptr, offset_ptr + nbins + 1);
    }

    // gap_ratio=0.25 -> leave an extra 25% of empty space at the end of each bin
    void buildPMAfromDenseBins_org(const int* perm,
                                   const int np,
                                   int* offsets,
                                   int numBins,
                                   double gap_ratio=0.25)
    {
        // 1) Estimate capacity and allocate memory in one go
        m_num_particles = np;
        m_capacity      = static_cast<int>( std::ceil(np * (1.0 + gap_ratio))+numBins );
        m_local_index.assign(m_capacity, INVALID_PARTICLE_ID);

        m_bin_offsets.resize(numBins + 1);
        m_bin_lengths.resize(numBins);

        // 2) Copy bin by bin + insert gaps
        int write_ptr = 0;
        for (int b = 1; b <= numBins; ++b)
        {
            const int begin = static_cast<int>(offsets[b-1]);
            const int end   = static_cast<int>(offsets[b]);
            const int n_in_bin = end - begin;

            // Starting physical address
            m_bin_offsets[b-1] = write_ptr;

            // Copy valid particles
            std::copy( perm + begin, perm + end, m_local_index.begin() + write_ptr );
            m_bin_lengths[b-1] = n_in_bin;
            if (write_ptr > m_capacity) {
                printf("Error: write_ptr (%d) exceeds m_capacity (%d)\n", write_ptr, m_capacity);
                // Handle error case
            }
            // 3) Insert empty slots at the end of this bin
            const int gap   = static_cast<int>( std::round(n_in_bin * gap_ratio) )+1;
            write_ptr      += (gap+n_in_bin);
        }
        m_bin_offsets[numBins] = write_ptr;             // Sentinel value
        m_num_empty_slots      = m_capacity - m_num_particles;
    }

# if 1
/**
     * @brief Builds a Packed Memory Array (PMA) structure for a grown tile box from a given particle permutation (perm) and offsets.
     *
     * This function processes particle data within the real tile box and allocates space for all cells in the grown tile box (including real and guard cells).
     * Each cell (bin) is guaranteed at least one empty slot. Bins within the real tile that contain particles are allocated additional empty slots based on gap_ratio.
     *
     * @param perm Pointer to the sorted particle ID array (containing only particles within the real tile).
     * @param np Total number of particles in the real tile.
     * @param offsets Pointer to an array describing the offsets of each bin from the real tile within the perm array.
     * offsets[0] is the start of the first bin, and offsets[num_original_bins_in_real_tile] is the end of the last bin (equal to np).
     * The array size should be num_original_bins_in_real_tile + 1.
     * @param num_original_bins_in_real_tile The original number of bins in the real tile (corresponds to numBins in the original function).
     * @param real_tile_box AMReX Box object describing the geometric bounds of the real tile.
     * @param grown_tile_box AMReX Box object describing the geometric bounds of the grown tile (real tile + guard cells).
     * @param gap_ratio (default 0.25) Specifies the proportion of extra empty slots to allocate for bins containing particles, relative to the particle count.
     */
    void buildPMAForGrownTile(
        const int* perm,
        const int np,
        const int* offsets,
        int num_original_bins_in_real_tile,
        const amrex::Box& real_tile_box,
        const amrex::Box& grown_tile_box,
        double gap_ratio=0.25)
    {
        // Step 1: Calculate the total number of bins in the grown tile
        const int num_total_bins_grown = grown_tile_box.numPts();
        m_num_bins = num_total_bins_grown;

        // Step 2: Estimate PMA total capacity and initialize data structures
        m_num_particles = np; // Total number of "real" particles stored in the PMA (from real_tile_box)

        // Capacity formula:
        // - np: Space for real particles
        // - np * gap_ratio: Extra empty slots for real particles based on the ratio
        // - num_total_bins_grown: Guarantees at least one empty slot for every bin in the grown tile
        m_capacity = static_cast<int>(std::ceil(static_cast<double>(np) * (1.0 + gap_ratio)) + 2*num_total_bins_grown);

        // Initialize PMA's underlying storage, filling all positions with an invalid particle ID
        m_local_index.reserve(m_capacity+1);
        m_local_index.resize(m_capacity);
        std::fill_n(m_local_index.data(), m_capacity, INVALID_PARTICLE_ID);

        // Resize bin offset and length arrays to accommodate the grown tile
        m_bin_offsets.resize(num_total_bins_grown + 1); // +1 for the sentinel at the end
        m_slot_offsets.resize(num_total_bins_grown + 1); // +1 for the sentinel at the end
        m_bin_max_ip.resize(num_total_bins_grown);
        m_bin_lengths.clear();
        m_bin_lengths.resize(num_total_bins_grown);  // All bins initially have 0 particles

        // Initialize the reverse lookup map from particle ID to its new bin index
        if (np > 0) { // Only allocate and initialize if there are particles
            m_pid_to_bin_map.resize(np);
            m_pid_to_index_map.resize(np);
        } else {
            m_pid_to_bin_map.clear();
            m_pid_to_index_map.clear();
            amrex::Abort("Error: NO particle at m_pid_to_bin_map in buildPMAForGrownTile");
        }

        // Step 3: Process each bin in the grown tile, filling with particles and empty slots
        int write_ptr = 0;               // Current writing position pointer in m_local_index
        int current_grown_bin_idx = 0;   // Linear index of the current bin being processed in the grown tile (0-based)
        m_slot_offsets[0] = 0;

        // Get the lower-left corner and dimensions of the real tile for calculating original bin indices
        const amrex::IntVect real_tile_lo = real_tile_box.smallEnd();
        const amrex::IntVect real_tile_lengths = real_tile_box.length();

        // Iterate through all cells in the grown_tile_box using AMReX's BoxIterator
        // The iterator typically follows Fortran order (column-major: x index varies fastest, then y, then z)
        for (amrex::BoxIterator bit(grown_tile_box); bit.ok(); ++bit) {
            const amrex::IntVect current_cell_iv = bit(); // Get the 3D coordinates of the current cell

            // Record the starting physical address of the current grown_bin in the PMA (m_local_index)
            m_bin_offsets[current_grown_bin_idx] = write_ptr;
            int particles_in_this_grown_bin = 0;

            if (real_tile_box.contains(current_cell_iv)) {
                // --- Case 1: The current cell is within the real_tile_box region ---

                // Convert the 3D coordinates of the current cell to its original 1D bin index within the real_tile_box.
                // This linearization order must exactly match the one used to generate the input 'perm' and 'offsets' arrays.
                amrex::IntVect rel_iv = current_cell_iv - real_tile_lo; // Relative coordinates
                int original_real_bin_idx = 0;

                // C-style ordering (row-major: z varies slowest, x fastest, assuming (z,y,x) nesting)
                original_real_bin_idx = rel_iv[0];
                original_real_bin_idx += rel_iv[1] * real_tile_lengths[0];
                original_real_bin_idx += rel_iv[2] * (real_tile_lengths[0] * real_tile_lengths[1]);

                // Safety check: ensure the calculated original_real_bin_idx is within valid bounds
                if (original_real_bin_idx < 0 || original_real_bin_idx >= num_original_bins_in_real_tile) {
                    fprintf(stderr, "Error (buildPMAForGrownTile): Calculated original_real_bin_idx %d is out of valid range [0, %d).\n"
                                      "         grown_cell_iv: (%d,%d,%d), rel_iv: (%d,%d,%d)\n",
                            original_real_bin_idx, num_original_bins_in_real_tile,
                            current_cell_iv[0], current_cell_iv[1], (AMREX_SPACEDIM==3 ? current_cell_iv[2] : 0),
                            rel_iv[0], rel_iv[1], (AMREX_SPACEDIM==3 ? rel_iv[2] : 0));
                    amrex::Abort("ERROR original_real_bin_idx: Mismatch in linearization logic or Box definition.");
                }

                // Get the start and end indices for this original bin's particles in the perm array
                const int p_start_idx_in_perm = offsets[original_real_bin_idx];
                const int p_end_idx_in_perm   = offsets[original_real_bin_idx + 1];
                particles_in_this_grown_bin   = p_end_idx_in_perm - p_start_idx_in_perm;

                if (particles_in_this_grown_bin > 0) {
                    const int* particle_ids_to_copy = perm + p_start_idx_in_perm;

                    if (write_ptr + particles_in_this_grown_bin > m_capacity) {
                        fprintf(stderr, "Error (buildPMAForGrownTile): Insufficient capacity when assigning PMA for bin in real tile (grown_idx %d, orig_idx %d)!\n"
                                          "         Capacity: %d, Current write_ptr: %d, Required space: %d, Particle count: %d\n",
                                current_grown_bin_idx, original_real_bin_idx, m_capacity, write_ptr,
                                write_ptr + particles_in_this_grown_bin, particles_in_this_grown_bin);
                        particles_in_this_grown_bin = 0;
                    } else {
                        // Copy particle IDs from the perm array to the current write position in m_local_index
                        std::copy(particle_ids_to_copy,
                              particle_ids_to_copy + particles_in_this_grown_bin,
                              m_local_index.begin() + write_ptr);

                        // Update m_pid_to_bin_map: map each copied particle ID to the current grown_bin_idx
                        if (!m_pid_to_bin_map.empty()) {
                            for (int i = 0; i < particles_in_this_grown_bin; ++i) {
                                int pid = particle_ids_to_copy[i];
                                if (pid >= 0 && static_cast<size_t>(pid) < m_pid_to_bin_map.size()) {
                                    m_pid_to_bin_map[pid] = current_grown_bin_idx;
                                    m_pid_to_index_map[pid] = write_ptr+i;
                                } else {
                                    fprintf(stderr, "Warning (buildPMAForGrownTile): Particle ID %d is invalid for m_pid_to_bin_map (size %zu) in grown_bin_idx %d.\n",
                                            pid, m_pid_to_bin_map.size(), current_grown_bin_idx);
                                }
                            }
                        }
                    }
                }
                m_bin_lengths[current_grown_bin_idx] = particles_in_this_grown_bin;
                write_ptr += particles_in_this_grown_bin;
                // Update the position of the empty slots to the end
                m_slot_offsets[current_grown_bin_idx] = write_ptr;

                // Add empty slots for the current bin in the real tile region:
                // 1. Slots calculated from gap_ratio (0 if no particles in bin)
                // 2. Plus one guaranteed empty slot (required for all grown_bins)
                const int particle_gap_slots = (particles_in_this_grown_bin > 0) ? static_cast<int>(std::ceil(static_cast<double>(particles_in_this_grown_bin) * gap_ratio)) : 0;
                const int total_empty_slots_to_add = particle_gap_slots + 1;

                // Advance the write pointer by the number of empty slots, ensuring it does not exceed capacity
                write_ptr += total_empty_slots_to_add;
                write_ptr = std::min(write_ptr, m_capacity);
                if(write_ptr > m_capacity){
                    amrex::Abort("ERROR write_ptrs: Capacity limit reached.");
                }

            } else {
                // --- Case 2: The current cell is in the guard cell area ---
                m_bin_lengths[current_grown_bin_idx] = 0; // Guard cells have no particles
                particles_in_this_grown_bin = 0;
                m_slot_offsets[current_grown_bin_idx] = write_ptr;

                // Add one guaranteed empty slot for the guard cell bin
                const int total_empty_slots_to_add = 1;
                int end_of_bin_physical_space = write_ptr + total_empty_slots_to_add;
                write_ptr = std::min(end_of_bin_physical_space, m_capacity);
                if(end_of_bin_physical_space > m_capacity){
                    amrex::Abort("ERROR write_ptrs in guard_cell: Capacity limit reached.");
                }
            }
            current_grown_bin_idx++; // Move to the next grown_bin
        }

        // Final check after iteration to ensure all grown_bins were processed
        if (current_grown_bin_idx != num_total_bins_grown) {
            fprintf(stderr, "Error (buildPMAForGrownTile): Number of processed grown_bins (%d) does not match expected total (%d).\n"
                              "This may indicate an inconsistency between BoxIterator and numPts() or a loop logic error.\n",
                    current_grown_bin_idx, num_total_bins_grown);
            amrex::Abort("ERROR current_grown_bin_idxs");
        }

        // Set the final offset for the last bin (sentinel value), which also represents the total used length of the PMA
        m_bin_offsets[num_total_bins_grown] = write_ptr;
        m_slot_offsets[num_total_bins_grown] = m_bin_offsets[num_total_bins_grown-1] + m_bin_lengths[num_total_bins_grown-1];
        m_capacity = write_ptr;

        m_was_rebuilt_this_step = true; // Mark that the PMA structure has been rebuilt in this step

        int vlf = svcntw();
        #pragma unroll
        for (int binID = 0; binID < m_num_bins; binID+=vlf) {
            svbool_t p_bin = svwhilelt_b32(binID, m_num_bins);
            svint32_t bin_max_ip_index = svld1_s32(p_bin, &m_slot_offsets[binID+1]);
            bin_max_ip_index = svsub_n_s32_x(p_bin, bin_max_ip_index, 1);
            svint32_t bin_max_ip_v = svld1_gather_s32index_s32(p_bin, &m_local_index[0], bin_max_ip_index);
            svst1_s32(p_bin, &m_bin_max_ip[binID], bin_max_ip_v);
        }
    }

    void buildPMAForGrownTile_OPM(
        const int* perm,
        const int np,
        const int* offsets,
        int num_original_bins_in_real_tile,
        const amrex::Box& real_tile_box,
        const amrex::Box& grown_tile_box,
        double gap_ratio=0.25)
    {
        const int num_total_bins_grown = grown_tile_box.numPts();
        m_num_bins = num_total_bins_grown;
        m_num_particles = np;

        m_capacity = static_cast<int>(std::ceil(static_cast<double>(np) * (1.0 + gap_ratio)) + 2*num_total_bins_grown);

        m_local_index_hbm.assign(m_capacity, INVALID_PARTICLE_ID);

        m_bin_offsets.resize(num_total_bins_grown + 1);
        m_slot_offsets.resize(num_total_bins_grown + 1);
        m_bin_lengths.clear();
        m_bin_lengths.resize(num_total_bins_grown);

        if (np > 0) {
            m_pid_to_bin_map_hbm.resize(np);
            m_pid_to_index_map.resize(np);
        } else {
            m_pid_to_bin_map_hbm.clear();
            m_pid_to_index_map.clear();
            amrex::Abort("Error: NO particle at m_pid_to_bin_map_hbm in buildPMAForGrownTile");
        }

        int write_ptr = 0;
        int current_grown_bin_idx = 0;
        m_slot_offsets[0] = 0;

        const amrex::IntVect real_tile_lo = real_tile_box.smallEnd();
        const amrex::IntVect real_tile_lengths = real_tile_box.length();

        for (amrex::BoxIterator bit(grown_tile_box); bit.ok(); ++bit) {
            const amrex::IntVect current_cell_iv = bit();

            m_bin_offsets[current_grown_bin_idx] = write_ptr;
            int particles_in_this_grown_bin = 0;

            if (real_tile_box.contains(current_cell_iv)) {
                amrex::IntVect rel_iv = current_cell_iv - real_tile_lo;
                int original_real_bin_idx = 0;
                original_real_bin_idx = rel_iv[0];
                original_real_bin_idx += rel_iv[1] * real_tile_lengths[0];
                original_real_bin_idx += rel_iv[2] * (real_tile_lengths[0] * real_tile_lengths[1]);

                if (original_real_bin_idx < 0 || original_real_bin_idx >= num_original_bins_in_real_tile) {
                    amrex::Abort("ERROR original_real_bin_idx: Mismatch in linearization logic or Box definition.");
                }
                const int p_start_idx_in_perm = offsets[original_real_bin_idx];
                const int p_end_idx_in_perm   = offsets[original_real_bin_idx + 1];
                particles_in_this_grown_bin   = p_end_idx_in_perm - p_start_idx_in_perm;

                if (particles_in_this_grown_bin > 0) {
                    const int* particle_ids_to_copy = perm + p_start_idx_in_perm;

                    if (write_ptr + particles_in_this_grown_bin > m_capacity) {
                        particles_in_this_grown_bin = 0;
                    } else {
                        std::copy(particle_ids_to_copy,
                              particle_ids_to_copy + particles_in_this_grown_bin,
                              m_local_index_hbm.begin() + write_ptr);

                        if (!m_pid_to_bin_map_hbm.empty()) {
                            for (int i = 0; i < particles_in_this_grown_bin; ++i) {
                                int pid = particle_ids_to_copy[i];
                                if (pid >= 0 && static_cast<size_t>(pid) < m_pid_to_bin_map_hbm.size()) {
                                    m_pid_to_bin_map_hbm[pid] = current_grown_bin_idx;
                                    m_pid_to_index_map[pid] = write_ptr+i;
                                }
                            }
                        }
                    }
                }
                m_bin_lengths[current_grown_bin_idx] = particles_in_this_grown_bin;
                write_ptr += particles_in_this_grown_bin;
                m_slot_offsets[current_grown_bin_idx] = write_ptr;

                const int particle_gap_slots = (particles_in_this_grown_bin > 0) ? static_cast<int>(std::ceil(static_cast<double>(particles_in_this_grown_bin) * gap_ratio)) : 0;
                const int total_empty_slots_to_add = particle_gap_slots + 1;

                write_ptr += total_empty_slots_to_add;
                write_ptr = std::min(write_ptr, m_capacity);
                if(write_ptr>m_capacity){
                    amrex::Abort("ERROR write_ptrs: Capacity limit reached.");
                }

            } else {
                m_bin_lengths[current_grown_bin_idx] = 0;
                particles_in_this_grown_bin = 0;
                m_slot_offsets[current_grown_bin_idx] = write_ptr;

                const int total_empty_slots_to_add = 1;
                int end_of_bin_physical_space = write_ptr + total_empty_slots_to_add;
                write_ptr = std::min(end_of_bin_physical_space, m_capacity);
                if(end_of_bin_physical_space>m_capacity){
                    amrex::Abort("ERROR write_ptrs in guard_cell: Capacity limit reached.");
                }
            }
            current_grown_bin_idx++;
        }

        if (current_grown_bin_idx != num_total_bins_grown) {
            amrex::Abort("ERROR current_grown_bin_idxs");
        }

        m_bin_offsets[num_total_bins_grown] = write_ptr;
        m_slot_offsets[num_total_bins_grown] = m_bin_offsets[num_total_bins_grown-1] + m_bin_lengths[num_total_bins_grown-1];
        m_capacity = write_ptr;

        m_was_rebuilt_this_step = true;
    }

    void buildPMAForGrownTile_OPM_order(
        const int* perm,
        const int np,
        const int* offsets,
        int num_original_bins_in_real_tile,
        const amrex::Box& real_tile_box,
        const amrex::Box& grown_tile_box,
        double gap_ratio=0.25,
        int order = 1)
    {
        const int num_total_bins_grown = grown_tile_box.numPts();
        m_num_bins=num_total_bins_grown;
        m_num_particles = np;

        m_capacity = static_cast<int>(std::ceil(static_cast<double>(np) * (1.0 + gap_ratio)) + 2*num_total_bins_grown);

        m_local_index_hbm.assign(m_capacity, INVALID_PARTICLE_ID);

        m_bin_offsets.resize(num_total_bins_grown + 1);
        m_slot_offsets.resize(num_total_bins_grown + 1);
        m_bin_lengths.clear();
        m_bin_lengths.resize(num_total_bins_grown);

        if (np > 0) {
            m_pid_to_bin_map_hbm.resize(np);
            m_pid_to_index_map.resize(np);
        } else {
            m_pid_to_bin_map_hbm.clear();
            m_pid_to_index_map.clear();
            amrex::Abort("Error: NO particle at m_pid_to_bin_map_hbm in buildPMAForGrownTile");
        }

        int write_ptr = 0;
        int current_grown_bin_idx = 0;
        m_slot_offsets[0] = 0;

        const amrex::IntVect real_tile_lo = real_tile_box.smallEnd();
        const amrex::IntVect real_tile_lengths = real_tile_box.length();

        const int lo = amrex::Math::floor(0.4*order);   // order=1->0, order=3->1
        const amrex::IntVect lo_shift(AMREX_D_DECL(lo,lo,lo));

        for (amrex::BoxIterator bit(grown_tile_box); bit.ok(); ++bit) {
            const amrex::IntVect current_cell_iv = bit();
            m_bin_offsets[current_grown_bin_idx] = write_ptr;
            int particles_in_this_grown_bin = 0;
            const amrex::IntVect source_cell_iv = current_cell_iv + lo_shift;

            if (real_tile_box.contains(source_cell_iv)) {
                amrex::IntVect rel_iv = source_cell_iv - real_tile_lo;

                int original_real_bin_idx = rel_iv[0];
                original_real_bin_idx += rel_iv[1] * real_tile_lengths[0];
                #if (AMREX_SPACEDIM > 2)
                    original_real_bin_idx += rel_iv[2] * (real_tile_lengths[0] * real_tile_lengths[1]);
                #endif

                if (original_real_bin_idx < 0 || original_real_bin_idx >= num_original_bins_in_real_tile) {
                    amrex::Abort("ERROR original_real_bin_idx: Mismatch in linearization logic or Box definition.");
                }
                const int p_start_idx_in_perm = offsets[original_real_bin_idx];
                const int p_end_idx_in_perm   = offsets[original_real_bin_idx + 1];
                particles_in_this_grown_bin   = p_end_idx_in_perm - p_start_idx_in_perm;

                if (particles_in_this_grown_bin > 0) {
                    const int* particle_ids_to_copy = perm + p_start_idx_in_perm;

                    if (write_ptr + particles_in_this_grown_bin > m_capacity) {
                        amrex::Abort("PMA capacity exceeded while copying particles.");
                    }
                    std::copy(particle_ids_to_copy,
                          particle_ids_to_copy + particles_in_this_grown_bin,
                          m_local_index_hbm.begin() + write_ptr);

                    if (!m_pid_to_bin_map_hbm.empty()) {
                        for (int i = 0; i < particles_in_this_grown_bin; ++i) {
                            int pid = particle_ids_to_copy[i];
                            if (pid >= 0 && static_cast<size_t>(pid) < m_pid_to_bin_map_hbm.size()) {
                                m_pid_to_bin_map_hbm[pid] = current_grown_bin_idx;
                                m_pid_to_index_map[pid] = write_ptr+i;
                            }
                        }
                    }
                }
                m_bin_lengths[current_grown_bin_idx] = particles_in_this_grown_bin;
                write_ptr += particles_in_this_grown_bin;
                m_slot_offsets[current_grown_bin_idx] = write_ptr;

                const int particle_gap_slots = (particles_in_this_grown_bin > 0) ? static_cast<int>(std::ceil(static_cast<double>(particles_in_this_grown_bin) * gap_ratio)) : 0;
                const int total_empty_slots_to_add = particle_gap_slots + 1;

                write_ptr += total_empty_slots_to_add;
                write_ptr = std::min(write_ptr, m_capacity);
                if(write_ptr>m_capacity){
                    amrex::Abort("ERROR write_ptrs: Capacity limit reached.");
                }

            } else {
                m_bin_lengths[current_grown_bin_idx] = 0;
                particles_in_this_grown_bin = 0;
                m_slot_offsets[current_grown_bin_idx] = write_ptr;

                const int total_empty_slots_to_add = 1;
                int end_of_bin_physical_space = write_ptr + total_empty_slots_to_add;
                write_ptr = std::min(end_of_bin_physical_space, m_capacity);
                if(end_of_bin_physical_space>m_capacity){
                    amrex::Abort("ERROR write_ptrs in guard_cell: Capacity limit reached.");
                }
            }
            current_grown_bin_idx++;
        }
        if (current_grown_bin_idx != num_total_bins_grown) {
            amrex::Abort("ERROR current_grown_bin_idxs");
        }

        m_bin_offsets[num_total_bins_grown] = write_ptr;
        m_slot_offsets[num_total_bins_grown] = m_bin_offsets[num_total_bins_grown-1] + m_bin_lengths[num_total_bins_grown-1];
        m_capacity = write_ptr;
        m_was_rebuilt_this_step = true;
    }
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
