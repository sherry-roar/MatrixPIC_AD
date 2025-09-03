/* Copyright 2019 Axel Huebl, David Grote, Maxence Thevenet
 * Remi Lehe, Weiqun Zhang, Michael Rowan
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
*/
#ifndef WARPX_CURRENTDEPOSITION_H_
#define WARPX_CURRENTDEPOSITION_H_

#include "Particles/Deposition/SharedDepositionUtils.H"
#include "ablastr/parallelization/KernelTimer.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/ShapeFactors.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"
#ifdef WARPX_DIM_RZ
#   include "Utils/WarpX_Complex.H"
#endif

#include "WarpX.H" // todo: remove include and pass globals as args

#include <AMReX.H>
#include <AMReX_Arena.H>
#include <AMReX_Array4.H>
#include <AMReX_Dim3.H>
#include <AMReX_REAL.H>
#include <chrono>

#include <RankSortStats.H>
#include "Utils/WarpXProfilerWrapper.H"

#include <arm_sve.h>
#include <arm_sme.h>
 
// Typedefs for SVE vectors with a specific bit width
typedef svfloat64_t svec __attribute__((arm_sve_vector_bits(__ARM_FEATURE_SVE_BITS)));
typedef svint64_t svecint __attribute__((arm_sve_vector_bits(__ARM_FEATURE_SVE_BITS)));
typedef svuint64_t svecuint __attribute__((arm_sve_vector_bits(__ARM_FEATURE_SVE_BITS)));

// Inline function to read the virtual timer counter register
inline uint64_t rdtscv(void) {
    uint64_t val;
    asm volatile("mrs %0, cntvct_el0" : "=r" (val) : : "memory");
    return val;
}

// Inline function to read the virtual timer counter register for streaming SVE
inline uint64_t rdtscm(void) __arm_preserves("za") __arm_streaming {
    uint64_t val;
    asm volatile("mrs %0, cntvct_el0" : "=r" (val) : : "memory");
    return val;
}

// Inline function to read the virtual timer with instruction synchronization barriers
static inline uint64_t rdtscm1(void) __arm_preserves("za") __arm_streaming {
    uint64_t val;
    asm volatile("isb              \n\t"  /* Ensure all previous instructions are complete */
                 "mrs %0, cntvct_el0\n\t"  /* Read the counter */
                 "isb              \n\t"  /* Ensure the read is complete */
                 : "=r"(val)
                 :
                 : "memory");
    return val;
}

class Vec {
private:
    svec v_; // Assuming svec is an alias for a SVE float64 vector type

public:
    // Default constructor
    Vec() = default;

    // Construct from svfloat64_t
    Vec(svfloat64_t v) : v_(v) {}

    // Construct from double
    Vec(double v) : v_(svdup_f64(v)) {}

    // Convert to svfloat64_t
    operator svfloat64_t() const {
        return v_;
    }

    // Vector addition assignment
    void operator+=(const Vec& rhs) {
        v_ = svadd_f64_x(svptrue_b64(), v_, rhs.v_);
    }

    // Vector subtraction assignment
    void operator-=(const Vec& rhs) {
        v_ = svsub_f64_x(svptrue_b64(), v_, rhs.v_);
    }

    // Vector multiplication assignment
    void operator*=(const Vec& rhs) {
        v_ = svmul_f64_x(svptrue_b64(), v_, rhs.v_);
    }

    // Vector addition
    Vec operator+(const Vec &rhs) const {
        return Vec(svadd_f64_x(svptrue_b64(), v_, rhs.v_));
    }

    // Vector subtraction
    Vec operator-(const Vec &rhs) const {
        return Vec(svsub_f64_x(svptrue_b64(), v_, rhs.v_));
    }

    // Vector multiplication
    Vec operator*(const Vec &rhs) const {
        return Vec(svmul_f64_x(svptrue_b64(), v_, rhs.v_));
    }

    // Vector division
    Vec operator/(const Vec &rhs) const {
        return Vec(svdiv_f64_x(svptrue_b64(), v_, rhs.v_));
    }

    // Vector negation
    Vec operator-() const {
        return Vec(svneg_f64_x(svptrue_b64(), v_));
    }

    // Vector square root
    Vec Sqrt() const {
        return Vec(svsqrt_f64_x(svptrue_b64(), v_));
    }

    // Vector-scalar addition
    Vec operator+(double rhs) const {
        return *this + Vec(rhs);
    }

    // Vector-scalar subtraction
    Vec operator-(double rhs) const {
        return *this - Vec(rhs);
    }

    // Vector-scalar multiplication
    Vec operator*(double rhs) const {
        return *this * Vec(rhs);
    }

    // Vector-scalar division
    Vec operator/(double rhs) const {
        return *this / Vec(rhs);
    }

    // Friend function: scalar-vector addition
    friend Vec operator+(double lhs, const Vec &rhs) {
        return Vec(lhs) + rhs;
    }

    // Friend function: scalar-vector subtraction
    friend Vec operator-(double lhs, const Vec &rhs) {
        return Vec(lhs) - rhs;
    }

    // Friend function: scalar-vector multiplication
    friend Vec operator*(double lhs, const Vec &rhs) {
        return Vec(lhs) * rhs;
    }

    // Friend function: scalar-vector division
    friend Vec operator/(double lhs, const Vec &rhs) {
        return Vec(lhs) / rhs;
    }

    // Store vector to memory
    void Store(svbool_t p, double *mem) const {
        svst1_f64(p, mem, v_);
    }

    // Load vector from memory
    static Vec Load(svbool_t p, const double *mem) {
        return Vec(svld1_f64(p, mem));
    }
};

class MVec {
private:
    svec v_;

public:
    MVec() __arm_preserves("za") __arm_streaming = default;

    // Construct from svec
    MVec(svec v) __arm_preserves("za") __arm_streaming : v_(v) {}
    // Construct from double
    MVec(double v) __arm_preserves("za") __arm_streaming : v_(svdup_f64(v)) {}

    // Convert to svec
    operator svec() const __arm_preserves("za") __arm_streaming {
        return v_;
    }

    // Vector addition assignment
    void operator+=(const MVec &rhs) __arm_preserves("za") __arm_streaming {
        v_ = svadd_f64_x(svptrue_b64(), v_, rhs.v_);
    }

    // Vector subtraction assignment
    void operator-=(const MVec &rhs) __arm_preserves("za") __arm_streaming {
        v_ = svsub_f64_x(svptrue_b64(), v_, rhs.v_);
    }

    // Vector multiplication assignment
    void operator*=(const MVec &rhs) __arm_preserves("za") __arm_streaming {
        v_ = svmul_f64_x(svptrue_b64(), v_, rhs.v_);
    }

    // Vector addition
    MVec operator+(const MVec &rhs) const __arm_preserves("za") __arm_streaming {
        return svadd_f64_x(svptrue_b64(), v_, rhs.v_);
    }

    // Vector subtraction
    MVec operator-(const MVec &rhs) const __arm_preserves("za") __arm_streaming {
        return svsub_f64_x(svptrue_b64(), v_, rhs.v_);
    }

    // Vector multiplication
    MVec operator*(const MVec &rhs) const __arm_preserves("za") __arm_streaming {
        return svmul_f64_x(svptrue_b64(), v_, rhs.v_);
    }

    // Vector negation
    MVec operator-() const __arm_preserves("za") __arm_streaming {
        return svneg_f64_x(svptrue_b64(), v_);
    }

    // Vector-scalar addition
    MVec operator+(double rhs) const __arm_preserves("za") __arm_streaming {
        return *this + MVec(rhs);
    }

    // Vector-scalar subtraction
    MVec operator-(double rhs) const __arm_preserves("za") __arm_streaming {
        return *this - MVec(rhs);
    }

    // Vector-scalar multiplication
    MVec operator*(double rhs) const __arm_preserves("za") __arm_streaming {
        return *this * MVec(rhs);
    }

    // Vector division
    MVec operator/(const MVec &rhs) const __arm_preserves("za") __arm_streaming {
        return MVec(svdiv_f64_x(svptrue_b64(), v_, rhs.v_));
    }

    // Vector-scalar division
    MVec operator/(double rhs) const __arm_preserves("za") __arm_streaming {
        return *this / MVec(rhs);
    }
    
    // Friend function: scalar-vector division
    friend MVec operator/(double lhs, const MVec &rhs) __arm_preserves("za") __arm_streaming {
        return MVec(lhs) / rhs;
    }

    // Vector square root
    MVec Sqrt() const __arm_preserves("za") __arm_streaming {
        return MVec(svsqrt_f64_x(svptrue_b64(), v_));
    }

    // Friend function: scalar-vector addition
    friend MVec operator+(double lhs, const MVec &rhs) __arm_preserves("za") __arm_streaming {
        return MVec(lhs) + rhs;
    }

    // Friend function: scalar-vector subtraction
    friend MVec operator-(double lhs, const MVec &rhs) __arm_preserves("za") __arm_streaming {
        return MVec(lhs) - rhs;
    }

    // Friend function: scalar-vector multiplication
    friend MVec operator*(double lhs, const MVec &rhs) __arm_preserves("za") __arm_streaming {
        return MVec(lhs) * rhs;
    }

    // Store vector to memory
    void Store(svbool_t p, double *mem) const __arm_preserves("za") __arm_streaming {
        svst1_f64(p, mem, v_);
    }

    // Load vector from memory
    static MVec Load(svbool_t p, const double *mem) __arm_preserves("za") __arm_streaming {
        return svld1_f64(p, mem);
    }

    // Reduce vector elements to a sum
    double ReduceSum() const __arm_preserves("za") __arm_streaming {
        return svaddv_f64(svptrue_b64(), v_);
    }
};

class UintVec {
private:
    svecuint v_;

public:
    // Default constructor
    UintVec() = default;

    // Construct from svuint64_t
    UintVec(svuint64_t v) : v_(v) {}

    // Construct from uint64_t
    UintVec(uint64_t v) : v_(svdup_u64(v)) {}

    // Convert to svuint64_t
    operator svuint64_t() const {
        return v_;
    }

    // Vector addition assignment
    void operator+=(const UintVec& rhs) {
        v_ = svadd_u64_x(svptrue_b64(), v_, rhs.v_);
    }

    // Vector subtraction assignment
    void operator-=(const UintVec& rhs) {
        v_ = svsub_u64_x(svptrue_b64(), v_, rhs.v_);
    }

    // Vector multiplication assignment
    void operator*=(const UintVec& rhs) {
        v_ = svmul_u64_x(svptrue_b64(), v_, rhs.v_);
    }

    // Vector addition
    UintVec operator+(const UintVec &rhs) const {
        return UintVec(svadd_u64_x(svptrue_b64(), v_, rhs.v_));
    }

    // Vector subtraction
    UintVec operator-(const UintVec &rhs) const {
        return UintVec(svsub_u64_x(svptrue_b64(), v_, rhs.v_));
    }

    // Vector multiplication
    UintVec operator*(const UintVec &rhs) const {
        return UintVec(svmul_u64_x(svptrue_b64(), v_, rhs.v_));
    }

    // Vector division
    UintVec operator/(const UintVec &rhs) const {
        return UintVec(svdiv_u64_x(svptrue_b64(), v_, rhs.v_));
    }

    // Vector-scalar addition
    UintVec operator+(uint64_t rhs) const {
        return *this + UintVec(rhs);
    }

    // Vector-scalar subtraction
    UintVec operator-(uint64_t rhs) const {
        return *this - UintVec(rhs);
    }

    // Vector-scalar multiplication
    UintVec operator*(uint64_t rhs) const {
        return *this * UintVec(rhs);
    }

    // Vector-scalar division
    UintVec operator/(uint64_t rhs) const {
        return *this / UintVec(rhs);
    }

    // Friend function: scalar-vector addition
    friend UintVec operator+(uint64_t lhs, const UintVec &rhs) {
        return UintVec(lhs) + rhs;
    }

    // Friend function: scalar-vector subtraction
    friend UintVec operator-(uint64_t lhs, const UintVec &rhs) {
        return UintVec(lhs) - rhs;
    }

    // Friend function: scalar-vector multiplication
    friend UintVec operator*(uint64_t lhs, const UintVec &rhs) {
        return UintVec(lhs) * rhs;
    }

    // Friend function: scalar-vector division
    friend UintVec operator/(uint64_t lhs, const UintVec &rhs) {
        return UintVec(lhs) / rhs;
    }

    // Store vector to memory
    void Store(svbool_t p, uint64_t *mem) const {
        svst1_u64(p, mem, v_);
    }

    // Load vector from memory
    static UintVec Load(svbool_t p, const uint64_t *mem) {
        return UintVec(svld1_u64(p, mem));
    }
};

class intVec {
private:
    svecint  v_;

public:
    // Default constructor
    intVec() = default;

    // Construct from svint64_t
    intVec(svint64_t v) : v_(v) {}

    // Construct from int64_t
    intVec(int64_t v) : v_(svdup_s64(v)) {}

    // Convert to svint64_t
    operator svint64_t() const {
        return v_;
    }

    // Vector addition assignment
    void operator+=(const intVec& rhs) {
        v_ = svadd_s64_x(svptrue_b64(), v_, rhs.v_);
    }

    // Vector subtraction assignment
    void operator-=(const intVec& rhs) {
        v_ = svsub_s64_x(svptrue_b64(), v_, rhs.v_);
    }

    // Vector multiplication assignment
    void operator*=(const intVec& rhs) {
        v_ = svmul_s64_x(svptrue_b64(), v_, rhs.v_);
    }

    // Vector addition
    intVec operator+(const intVec &rhs) const {
        return intVec(svadd_s64_x(svptrue_b64(), v_, rhs.v_));
    }

    // Vector subtraction
    intVec operator-(const intVec &rhs) const {
        return intVec(svsub_s64_x(svptrue_b64(), v_, rhs.v_));
    }

    // Vector multiplication
    intVec operator*(const intVec &rhs) const {
        return intVec(svmul_s64_x(svptrue_b64(), v_, rhs.v_));
    }

    // Vector division
    intVec operator/(const intVec &rhs) const {
        return intVec(svdiv_s64_x(svptrue_b64(), v_, rhs.v_));
    }

    // Vector negation
    intVec operator-() const {
        return intVec(svneg_s64_x(svptrue_b64(), v_));
    }

    // Vector-scalar addition
    intVec operator+(int64_t rhs) const {
        return *this + intVec(rhs);
    }

    // Vector-scalar subtraction
    intVec operator-(int64_t rhs) const {
        return *this - intVec(rhs);
    }

    // Vector-scalar multiplication
    intVec operator*(int64_t rhs) const {
        return *this * intVec(rhs);
    }

    // Vector-scalar division
    intVec operator/(int64_t rhs) const {
        return *this / intVec(rhs);
    }

    // Friend function: scalar-vector addition
    friend intVec operator+(int64_t lhs, const intVec &rhs) {
        return intVec(lhs) + rhs;
    }

    // Friend function: scalar-vector subtraction
    friend intVec operator-(int64_t lhs, const intVec &rhs) {
        return intVec(lhs) - rhs;
    }

    // Friend function: scalar-vector multiplication
    friend intVec operator*(int64_t lhs, const intVec &rhs) {
        return intVec(lhs) * rhs;
    }

    // Friend function: scalar-vector division
    friend intVec operator/(int64_t lhs, const intVec &rhs) {
        return intVec(lhs) / rhs;
    }

    // Store vector to memory
    void Store(svbool_t p, int64_t *mem) const {
        svst1_s64(p, mem, v_);
    }

    // Load vector from memory
    static intVec Load(svbool_t p, const int64_t *mem) {
        return intVec(svld1_s64(p, mem));
    }
};

 
/**
 * \brief Kernel for the direct current deposition for thread thread_num
 * \tparam depos_order deposition order
 * \param xp, yp, zp    The particle positions.
 * \param wq            The charge of the macroparticle
 * \param vx,vy,vz      The particle velocities
 * \param jx_arr,jy_arr,jz_arr Array4 of current density, either full array or tile.
 * \param jx_type,jy_type,jz_type The grid types along each direction, either NODE or CELL
 * \param relative_time Time at which to deposit J, relative to the time of the
 *                      current positions of the particles. When different than 0,
 *                      the particle position will be temporarily modified to match
 *                      the time of the deposition.
 * \param dinv          3D cell size inverse
 * \param xyzmin        The lower bounds of the domain
 * \param invvol        The inverse volume of a grid cell
 * \param lo            Index lower bounds of domain.
 * \param n_rz_azimuthal_modes Number of azimuthal modes when using RZ geometry.
 */
template <int depos_order>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void doDepositionShapeNKernel([[maybe_unused]] const amrex::ParticleReal xp,
                              [[maybe_unused]] const amrex::ParticleReal yp,
                              [[maybe_unused]] const amrex::ParticleReal zp,
                              const amrex::ParticleReal wq,
                              const amrex::ParticleReal vx,
                              const amrex::ParticleReal vy,
                              const amrex::ParticleReal vz,
                              amrex::Array4<amrex::Real> const& jx_arr,
                              amrex::Array4<amrex::Real> const& jy_arr,
                              amrex::Array4<amrex::Real> const& jz_arr,
                              amrex::IntVect const& jx_type,
                              amrex::IntVect const& jy_type,
                              amrex::IntVect const& jz_type,
                              const amrex::Real relative_time,
                              const amrex::XDim3 & dinv,
                              const amrex::XDim3 & xyzmin,
                              const amrex::Real invvol,
                              const amrex::Dim3 lo,
                              [[maybe_unused]] const int n_rz_azimuthal_modes)
{
    using namespace amrex::literals;

    constexpr int NODE = amrex::IndexType::NODE;
    constexpr int CELL = amrex::IndexType::CELL;

    // wqx, wqy wqz are particle current in each direction
#if defined(WARPX_DIM_RZ)
    // In RZ, wqx is actually wqr, and wqy is wqtheta
    // Convert to cylindrical at the mid point
    const amrex::Real xpmid = xp + relative_time*vx;
    const amrex::Real ypmid = yp + relative_time*vy;
    const amrex::Real rpmid = std::sqrt(xpmid*xpmid + ypmid*ypmid);
    const amrex::Real costheta = (rpmid > 0._rt ? xpmid/rpmid : 1._rt);
    const amrex::Real sintheta = (rpmid > 0._rt ? ypmid/rpmid : 0._rt);
    const Complex xy0 = Complex{costheta, sintheta};
    const amrex::Real wqx = wq*invvol*(+vx*costheta + vy*sintheta);
    const amrex::Real wqy = wq*invvol*(-vx*sintheta + vy*costheta);
#else
    const amrex::Real wqx = wq*invvol*vx;
    const amrex::Real wqy = wq*invvol*vy;
#endif
    const amrex::Real wqz = wq*invvol*vz;

    // --- Compute shape factors
    Compute_shape_factor< depos_order > const compute_shape_factor;
#if (AMREX_SPACEDIM >= 2)
    // x direction
    // Get particle position after 1/2 push back in position
    // Keep these double to avoid bug in single precision
#if defined(WARPX_DIM_RZ)
    const double xmid = (rpmid - xyzmin.x)*dinv.x;
#else
    const double xmid = ((xp - xyzmin.x) + relative_time*vx)*dinv.x;
#endif

    // j_j[xyz] leftmost grid point in x that the particle touches for the centering of each current
    // sx_j[xyz] shape factor along x for the centering of each current
    // There are only two possible centerings, node or cell centered, so at most only two shape factor
    // arrays will be needed.
    // Keep these double to avoid bug in single precision
    double sx_node[depos_order + 1] = {0.};
    double sx_cell[depos_order + 1] = {0.};
    int j_node = 0;
    int j_cell = 0;
    if (jx_type[0] == NODE || jy_type[0] == NODE || jz_type[0] == NODE) {
        j_node = compute_shape_factor(sx_node, xmid);
    }
    if (jx_type[0] == CELL || jy_type[0] == CELL || jz_type[0] == CELL) {
        j_cell = compute_shape_factor(sx_cell, xmid - 0.5);
    }

    amrex::Real sx_jx[depos_order + 1] = {0._rt};
    amrex::Real sx_jy[depos_order + 1] = {0._rt};
    amrex::Real sx_jz[depos_order + 1] = {0._rt};
    for (int ix=0; ix<=depos_order; ix++)
    {
        sx_jx[ix] = ((jx_type[0] == NODE) ? amrex::Real(sx_node[ix]) : amrex::Real(sx_cell[ix]));
        sx_jy[ix] = ((jy_type[0] == NODE) ? amrex::Real(sx_node[ix]) : amrex::Real(sx_cell[ix]));
        sx_jz[ix] = ((jz_type[0] == NODE) ? amrex::Real(sx_node[ix]) : amrex::Real(sx_cell[ix]));
    }

    int const j_jx = ((jx_type[0] == NODE) ? j_node : j_cell);
    int const j_jy = ((jy_type[0] == NODE) ? j_node : j_cell);
    int const j_jz = ((jz_type[0] == NODE) ? j_node : j_cell);
#endif //AMREX_SPACEDIM >= 2

#if defined(WARPX_DIM_3D)
    // y direction
    // Keep these double to avoid bug in single precision
    const double ymid = ((yp - xyzmin.y) + relative_time*vy)*dinv.y;
    double sy_node[depos_order + 1] = {0.};
    double sy_cell[depos_order + 1] = {0.};
    int k_node = 0;
    int k_cell = 0;
    if (jx_type[1] == NODE || jy_type[1] == NODE || jz_type[1] == NODE) {
        k_node = compute_shape_factor(sy_node, ymid);
    }
    if (jx_type[1] == CELL || jy_type[1] == CELL || jz_type[1] == CELL) {
        k_cell = compute_shape_factor(sy_cell, ymid - 0.5);
    }
    amrex::Real sy_jx[depos_order + 1] = {0._rt};
    amrex::Real sy_jy[depos_order + 1] = {0._rt};
    amrex::Real sy_jz[depos_order + 1] = {0._rt};
    for (int iy=0; iy<=depos_order; iy++)
    {
        sy_jx[iy] = ((jx_type[1] == NODE) ? amrex::Real(sy_node[iy]) : amrex::Real(sy_cell[iy]));
        sy_jy[iy] = ((jy_type[1] == NODE) ? amrex::Real(sy_node[iy]) : amrex::Real(sy_cell[iy]));
        sy_jz[iy] = ((jz_type[1] == NODE) ? amrex::Real(sy_node[iy]) : amrex::Real(sy_cell[iy]));
    }
    int const k_jx = ((jx_type[1] == NODE) ? k_node : k_cell);
    int const k_jy = ((jy_type[1] == NODE) ? k_node : k_cell);
    int const k_jz = ((jz_type[1] == NODE) ? k_node : k_cell);
#endif

    // z direction
    // Keep these double to avoid bug in single precision
    constexpr int zdir = WARPX_ZINDEX;
    const double zmid = ((zp - xyzmin.z) + relative_time*vz)*dinv.z;
    double sz_node[depos_order + 1] = {0.};
    double sz_cell[depos_order + 1] = {0.};
    int l_node = 0;
    int l_cell = 0;
    if (jx_type[zdir] == NODE || jy_type[zdir] == NODE || jz_type[zdir] == NODE) {
        l_node = compute_shape_factor(sz_node, zmid);
    }
    if (jx_type[zdir] == CELL || jy_type[zdir] == CELL || jz_type[zdir] == CELL) {
        l_cell = compute_shape_factor(sz_cell, zmid - 0.5);
    }
    amrex::Real sz_jx[depos_order + 1] = {0._rt};
    amrex::Real sz_jy[depos_order + 1] = {0._rt};
    amrex::Real sz_jz[depos_order + 1] = {0._rt};
    for (int iz=0; iz<=depos_order; iz++)
    {
        sz_jx[iz] = ((jx_type[zdir] == NODE) ? amrex::Real(sz_node[iz]) : amrex::Real(sz_cell[iz]));
        sz_jy[iz] = ((jy_type[zdir] == NODE) ? amrex::Real(sz_node[iz]) : amrex::Real(sz_cell[iz]));
        sz_jz[iz] = ((jz_type[zdir] == NODE) ? amrex::Real(sz_node[iz]) : amrex::Real(sz_cell[iz]));
    }
    int const l_jx = ((jx_type[zdir] == NODE) ? l_node : l_cell);
    int const l_jy = ((jy_type[zdir] == NODE) ? l_node : l_cell);
    int const l_jz = ((jz_type[zdir] == NODE) ? l_node : l_cell);

    // Deposit current into jx_arr, jy_arr and jz_arr
#if defined(WARPX_DIM_1D_Z)
    for (int iz=0; iz<=depos_order; iz++){
        amrex::Gpu::Atomic::AddNoRet(
            &jx_arr(lo.x+l_jx+iz, 0, 0, 0),
            sz_jx[iz]*wqx);
        amrex::Gpu::Atomic::AddNoRet(
            &jy_arr(lo.x+l_jy+iz, 0, 0, 0),
            sz_jy[iz]*wqy);
        amrex::Gpu::Atomic::AddNoRet(
            &jz_arr(lo.x+l_jz+iz, 0, 0, 0),
            sz_jz[iz]*wqz);
    }
#endif
#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    for (int iz=0; iz<=depos_order; iz++){
        for (int ix=0; ix<=depos_order; ix++){
            amrex::Gpu::Atomic::AddNoRet(
                &jx_arr(lo.x+j_jx+ix, lo.y+l_jx+iz, 0, 0),
                sx_jx[ix]*sz_jx[iz]*wqx);
            amrex::Gpu::Atomic::AddNoRet(
                &jy_arr(lo.x+j_jy+ix, lo.y+l_jy+iz, 0, 0),
                sx_jy[ix]*sz_jy[iz]*wqy);
            amrex::Gpu::Atomic::AddNoRet(
                &jz_arr(lo.x+j_jz+ix, lo.y+l_jz+iz, 0, 0),
                sx_jz[ix]*sz_jz[iz]*wqz);
#if defined(WARPX_DIM_RZ)
            Complex xy = xy0; // Note that xy is equal to e^{i m theta}
            for (int imode=1 ; imode < n_rz_azimuthal_modes ; imode++) {
                // The factor 2 on the weighting comes from the normalization of the modes
                amrex::Gpu::Atomic::AddNoRet( &jx_arr(lo.x+j_jx+ix, lo.y+l_jx+iz, 0, 2*imode-1), 2._rt*sx_jx[ix]*sz_jx[iz]*wqx*xy.real());
                amrex::Gpu::Atomic::AddNoRet( &jx_arr(lo.x+j_jx+ix, lo.y+l_jx+iz, 0, 2*imode  ), 2._rt*sx_jx[ix]*sz_jx[iz]*wqx*xy.imag());
                amrex::Gpu::Atomic::AddNoRet( &jy_arr(lo.x+j_jy+ix, lo.y+l_jy+iz, 0, 2*imode-1), 2._rt*sx_jy[ix]*sz_jy[iz]*wqy*xy.real());
                amrex::Gpu::Atomic::AddNoRet( &jy_arr(lo.x+j_jy+ix, lo.y+l_jy+iz, 0, 2*imode  ), 2._rt*sx_jy[ix]*sz_jy[iz]*wqy*xy.imag());
                amrex::Gpu::Atomic::AddNoRet( &jz_arr(lo.x+j_jz+ix, lo.y+l_jz+iz, 0, 2*imode-1), 2._rt*sx_jz[ix]*sz_jz[iz]*wqz*xy.real());
                amrex::Gpu::Atomic::AddNoRet( &jz_arr(lo.x+j_jz+ix, lo.y+l_jz+iz, 0, 2*imode  ), 2._rt*sx_jz[ix]*sz_jz[iz]*wqz*xy.imag());
                xy = xy*xy0;
            }
#endif
        }
    }
#elif defined(WARPX_DIM_3D)
    for (int iz=0; iz<=depos_order; iz++){
        for (int iy=0; iy<=depos_order; iy++){
            for (int ix=0; ix<=depos_order; ix++){
                amrex::Gpu::Atomic::AddNoRet(
                    &jx_arr(lo.x+j_jx+ix, lo.y+k_jx+iy, lo.z+l_jx+iz),
                    sx_jx[ix]*sy_jx[iy]*sz_jx[iz]*wqx);
                amrex::Gpu::Atomic::AddNoRet(
                    &jy_arr(lo.x+j_jy+ix, lo.y+k_jy+iy, lo.z+l_jy+iz),
                    sx_jy[ix]*sy_jy[iy]*sz_jy[iz]*wqy);
                amrex::Gpu::Atomic::AddNoRet(
                    &jz_arr(lo.x+j_jz+ix, lo.y+k_jz+iy, lo.z+l_jz+iz),
                    sx_jz[ix]*sy_jz[iy]*sz_jz[iz]*wqz);
            }
        }
    }
#endif
}

/**
 * \brief Current Deposition for thread thread_num
 * \tparam depos_order deposition order
 * \param GetPosition  A functor for returning the particle position.
 * \param wp           Pointer to array of particle weights.
 * \param uxp,uyp,uzp  Pointer to arrays of particle momentum.
 * \param ion_lev      Pointer to array of particle ionization level. This is
                         required to have the charge of each macroparticle
                         since q is a scalar. For non-ionizable species,
                         ion_lev is a null pointer.
 * \param jx_fab,jy_fab,jz_fab FArrayBox of current density, either full array or tile.
 * \param np_to_deposit Number of particles for which current is deposited.
 * \param relative_time Time at which to deposit J, relative to the time of the
 *                      current positions of the particles. When different than 0,
 *                      the particle position will be temporarily modified to match
 *                      the time of the deposition.
 * \param dinv         3D cell size inverse
 * \param xyzmin       Physical lower bounds of domain.
 * \param lo           Index lower bounds of domain.
 * \param q            species charge.
 * \param n_rz_azimuthal_modes Number of azimuthal modes when using RZ geometry.
 */
template <int depos_order>
void doDepositionShapeN (const GetParticlePosition<PIdx>& GetPosition,
                         const amrex::ParticleReal * const wp,
                         const amrex::ParticleReal * const uxp,
                         const amrex::ParticleReal * const uyp,
                         const amrex::ParticleReal * const uzp,
                         const int* ion_lev,
                         amrex::FArrayBox& jx_fab,
                         amrex::FArrayBox& jy_fab,
                         amrex::FArrayBox& jz_fab,
                         long np_to_deposit,
                         amrex::Real relative_time,
                         const amrex::XDim3 & dinv,
                         const amrex::XDim3 & xyzmin,
                         amrex::Dim3 lo,
                         amrex::Real q,
                         [[maybe_unused]]int n_rz_azimuthal_modes)
{
    using namespace amrex::literals;

    // Whether ion_lev is a null pointer (do_ionization=0) or a real pointer
    // (do_ionization=1)
    const bool do_ionization = ion_lev;

    const amrex::Real invvol = dinv.x*dinv.y*dinv.z;

    const amrex::Real clightsq = 1.0_rt/PhysConst::c/PhysConst::c;

    amrex::Array4<amrex::Real> const& jx_arr = jx_fab.array();
    amrex::Array4<amrex::Real> const& jy_arr = jy_fab.array();
    amrex::Array4<amrex::Real> const& jz_arr = jz_fab.array();
    amrex::IntVect const jx_type = jx_fab.box().type();
    amrex::IntVect const jy_type = jy_fab.box().type();
    amrex::IntVect const jz_type = jz_fab.box().type();

    // Loop over particles and deposit into jx_fab, jy_fab and jz_fab
    amrex::ParallelFor(
        np_to_deposit,
        [=] AMREX_GPU_DEVICE (long ip) {
            amrex::ParticleReal xp, yp, zp;
            GetPosition(ip, xp, yp, zp);

            // --- Get particle quantities
            const amrex::Real gaminv = 1.0_rt/std::sqrt(1.0_rt + uxp[ip]*uxp[ip]*clightsq
                                                        + uyp[ip]*uyp[ip]*clightsq
                                                        + uzp[ip]*uzp[ip]*clightsq);
            const amrex::Real vx  = uxp[ip]*gaminv;
            const amrex::Real vy  = uyp[ip]*gaminv;
            const amrex::Real vz  = uzp[ip]*gaminv;

            amrex::Real wq  = q*wp[ip];
            if (do_ionization){
                wq *= ion_lev[ip];
            }

            doDepositionShapeNKernel<depos_order>(xp, yp, zp, wq, vx, vy, vz, jx_arr, jy_arr, jz_arr,
                                                  jx_type, jy_type, jz_type,
                                                  relative_time, dinv, xyzmin,
                                                  invvol, lo, n_rz_azimuthal_modes);

        }
    );
}

// =================Start Functions================

inline void increment_sort_particles (
    int np_to_deposit, ParticleTileType& ptile,
    std::vector<int>& newbin, int numcell
)
{
    auto& local_index = ptile.m_local_index;
    auto& slot_offsets = ptile.m_slot_offsets; 
    auto& bin_length = ptile.m_bin_lengths;
    
    auto& num_particles = ptile.m_num_particles;
    auto& bin_offsets = ptile.m_bin_offsets;
    auto& pid_to_bin_map = ptile.m_pid_to_bin_map;
    auto& pid_to_index_map = ptile.m_pid_to_index_map; 
    auto& was_rebuilt_this_step =  ptile.m_was_rebuilt_this_step;
    was_rebuilt_this_step = false ;
    
    std::unordered_map<int, std::vector<int>> m_pending_moves;
    m_pending_moves.clear();

    bool rebuild=false;
    int vlf = svcntw();

    // Stage 1: Prepare data for incremental update.
    // Pre-process particle updates, organizing them into contiguous blocks.
    if(np_to_deposit<ptile.m_capacity && !rebuild){
        
        // 1. Handle particle deletion for particles that are no longer present.
        // move outside
        std::unordered_map<int, int> bin_decrement_counts;
        for(int ip=np_to_deposit;ip<num_particles;ip++){
            int old_bin=pid_to_bin_map[ip];
            int idx=pid_to_index_map[ip]; 
            // Invalidate the particle's index, which will be compacted later.
            ptile.m_local_index[idx] = INVALID_PARTICLE_ID;
            // This is handled later in a batch operation.
            // bin_length[old_bin]--; 
            // This must be recorded to be included in the compaction step.
            bin_decrement_counts[old_bin]++; 
        }

        // 2. For particles that have changed bins, treat it as a deletion from their old bin.
        // The loop iterates up to the minimum of the old and new particle counts,
        // as newly added particles (ip > num_particles) and removed particles (ip > np_to_deposit) are handled separately.
        int loop_np=std::min(num_particles,np_to_deposit); 
        int block_size=256; 
        float max_np_rate=0.15; 
        int MOVED_PARTICLES_MAX=loop_np/4+block_size;

        int moved_ips[MOVED_PARTICLES_MAX];
        int moved_old_bins[MOVED_PARTICLES_MAX];
        int moved_new_bins[MOVED_PARTICLES_MAX];
        int moved_count = 0;

        svint32_t vec_invalid_id = svdup_n_s32(INVALID_PARTICLE_ID);

        for (int i = 0; i < loop_np; i += block_size) {
            int current_block_size = std::min(block_size, loop_np - i);
            size_t bytes_to_compare = current_block_size * sizeof(int);

            // --- Fast path: Use memcmp to compare the entire block ---
            // memcmp compares memory content. A return value of 0 means the two blocks are identical.
            if (memcmp(&newbin[i], &pid_to_bin_map[i], bytes_to_compare) == 0) {
                // If no particles in the block have moved, skip it. This is key for performance.
                continue; 
            } 
            else {
                for (int j = 0; j < current_block_size; j += vlf) {
                    int ip=i+j;
                    // Create a predicate covering the current vector lane.
                    svbool_t pg32 = svwhilelt_b32(j, current_block_size);
                    svint32_t vec_newbin = svld1_s32(pg32, &newbin[ip]);
                    svint32_t vec_oldbin   = svld1_s32(pg32, &pid_to_bin_map[ip]);
                    svbool_t moved_mask = svcmpne(pg32, vec_newbin, vec_oldbin);
                    if (!svptest_any(pg32, moved_mask)) {
                        continue;
                    }
                    // --- Start processing moved particles ---
                    // 0) Create a vector of particle indices (ip) for the current block.
                    svint32_t all_ips_in_block = svindex_s32(ip, 1);

                    // 1) Get the storage indices (idx) of the moved particles.
                    svint32_t moved_idxs_vec = svld1_s32(moved_mask, &pid_to_index_map[ip]);

                    // 4) SCATTER operation: Set the corresponding positions in local_index to invalid.
                    svst1_scatter_s32index_s32(moved_mask, &local_index[0], moved_idxs_vec, vec_invalid_id);
                    
                    // 5) PACK operation: Pack all required information.
                    int32_t count_in_this_block = svcntp_b32(pg32, moved_mask);
                    svbool_t pg_compact = svwhilelt_b32(0, count_in_this_block);

                    // a. Compact ip, new_bin, and old_bin.
                    svint32_t moved_ips_vec      = svcompact(moved_mask, all_ips_in_block);
                    svint32_t moved_new_bins_vec = svcompact(moved_mask, vec_newbin);
                    svint32_t moved_old_bins_vec = svcompact(moved_mask, vec_oldbin);

                    // b. Store the compacted vectors into the output arrays.
                    svst1_s32(pg_compact, &moved_ips[moved_count], moved_ips_vec);
                    svst1_s32(pg_compact, &moved_new_bins[moved_count], moved_new_bins_vec);
                    svst1_s32(pg_compact, &moved_old_bins[moved_count], moved_old_bins_vec);

                    // 6) Update the total count of moved particles.
                    moved_count += count_in_this_block;
                }
                
                if (moved_count > loop_np*max_np_rate) {
                    rebuild=true;
                    break;
                }
            }
        }
        
        // To efficiently update bin_length, use a temporary map for counting.
        if (!rebuild) {
            for (int k = 0; k < moved_count; ++k) {
                int ip = moved_ips[k];
                int new_bin = moved_new_bins[k];
                int old_bin = moved_old_bins[k];

                // Add the particle to the list of pending moves for its new bin.
                m_pending_moves[new_bin].push_back(ip);

                // Count the number of particles leaving each old_bin.
                bin_decrement_counts[old_bin]++;
            }
            // Update bin_length in one batch operation.
            for (auto const& [bin, count] : bin_decrement_counts) {
                bin_length[bin] -= count;
            }
            
            // 3. Compact the local_index to remove invalid entries.
            sorted_index_pma_compact(bin_decrement_counts,
                local_index,slot_offsets,bin_offsets,pid_to_index_map);

            // 4. Add newly created particles. They are appended to the pending moves list.
            for(int ip=num_particles;ip<np_to_deposit;ip++){
                int new_bin=newbin[ip];
                m_pending_moves[new_bin].push_back(ip);
            }
        }
    } else {
        rebuild=true;
    }
    
    // At this point, bin_length and slot_offset have been decremented for particles that left bins.
    // m_local_index has been compacted, and pid_to_index_map has been updated accordingly.
    

    // Stage 2: Perform the incremental update.
    // Next, update m_pid_to_bin_map, pid_to_index_map, and num_particles.
    // As particles are inserted, slot_offset and bin_offsets will also need to be updated.
    // The moved particles will be inserted into their new locations in m_local_index.
    pid_to_index_map.reserve(np_to_deposit+1);  
    pid_to_index_map.resize(np_to_deposit);

    pid_to_bin_map.reserve(np_to_deposit+1);
    pid_to_bin_map.resize(np_to_deposit);
    std::memcpy(pid_to_bin_map.data(),
                newbin.data(),
                np_to_deposit * sizeof(int));
    num_particles=np_to_deposit;

    // Stage 2.1: Perform insertions into bins.
    // Insert at the tail first, as empty slots are typically located there.
    // If there's not enough space at the tail, borrow from the front. If the front also lacks space, break and trigger a rebuild.
    if(!rebuild){ 
        for (const auto& pair : m_pending_moves) {
            int new_bin=pair.first;
            const std::vector<int>& moved_ips = pair.second;
            int ip_size = moved_ips.size();
            
            int bin_end = bin_offsets[new_bin+1];
            int tail_bin_slot = slot_offsets[new_bin+1];
            int bin_start = bin_offsets[new_bin];
            int front_bin_slot = slot_offsets[new_bin];
            
            // Case 1: Enough space at the tail of the bin.
            if(bin_end-tail_bin_slot>=ip_size){
                std::memcpy(local_index.data()+tail_bin_slot,
                            moved_ips.data(),
                            ip_size * sizeof(int)); 
                
                for(int i=0;i<ip_size;i+=vlf){
                    svbool_t p=svwhilelt_b32(i,ip_size);
                    svint32_t ip_v = svld1_s32(p,&moved_ips[i]);
                    svint32_t pid_to_index_map_v = svindex_s32(slot_offsets[new_bin+1]+i,1);
                    svst1_scatter_s32index_s32(p,&pid_to_index_map[0],ip_v,pid_to_index_map_v);
                } 
                slot_offsets[new_bin+1]+=ip_size;
            // Case 2: Not enough space at the tail, but combined front and tail space is sufficient.
            }else if((bin_end-tail_bin_slot)+(bin_start-front_bin_slot)>=ip_size){
                int borrow_front=ip_size-(bin_end-tail_bin_slot);
                // Shift the entire bin's existing data forward to make space at the end.
                std::memmove(local_index.data()+bin_start-borrow_front,
                            local_index.data()+bin_start,
                            bin_length[new_bin] * sizeof(int));

                bin_offsets[new_bin]-=borrow_front;
                slot_offsets[new_bin+1]-=borrow_front;
                // Copy the new particle data into the newly created space.
                std::memcpy(local_index.data()+slot_offsets[new_bin+1],
                            moved_ips.data(),
                            ip_size * sizeof(int));
                // Update pid_to_index_map for all particles in the shifted bin.
                for(int i=0;i<ip_size+bin_length[new_bin];i+=vlf){
                    svbool_t p=svwhilelt_b32(i,ip_size+bin_length[new_bin]);
                    svint32_t ip_v = svld1_s32(p,&local_index[bin_offsets[new_bin]+i]);
                    svint32_t pid_to_index_map_v = svindex_s32(bin_offsets[new_bin]+i,1);
                    svst1_scatter_s32index_s32(p,&pid_to_index_map[0],ip_v,pid_to_index_map_v);
                }
                slot_offsets[new_bin+1]+=ip_size;
            // Case 3: Not enough space, trigger a rebuild.
            }else{
                rebuild=true;
                break;
            }
            bin_length[new_bin]+=ip_size;
        }
    }
    m_pending_moves.clear();

    // Stage 3: If incremental update failed or was deemed inefficient, perform a full rebuild.
    if(rebuild)
    { 
        ptile.m_was_rebuilt_this_step = true;

        float gap_ratio=WarpX::GetInstance().m_gap_ratio;
        ptile.m_bin_lengths.clear();
        ptile.m_bin_lengths.resize(numcell);

        // Stage 3.1: Recalculate m_bin_lengths from scratch.
        for(int ip=0;ip<np_to_deposit;ip++){
            ptile.m_bin_lengths[ptile.m_pid_to_bin_map[ip]]++;
        }   
        // Estimate the required capacity with a gap for future growth.
        ptile.m_capacity = static_cast<int>(
                std::ceil(static_cast<double>(
                np_to_deposit) * (1.0 + gap_ratio)) + 2*numcell);
        // Initialize offset arrays.
        int current_offset = 0;
        ptile.m_bin_offsets.clear();
        ptile.m_bin_offsets.resize(numcell+1);
        ptile.m_slot_offsets.clear();
        ptile.m_slot_offsets.resize(numcell+1);
        ptile.m_bin_offsets[0] = 0;
        ptile.m_slot_offsets[0] = 0;
        
        // Stage 3.2: Calculate m_bin_offsets and m_slot_offsets using SVE for parallel prefix sum.
        #pragma unroll
        for (int binID = 0; binID < numcell; binID+=vlf) {
            svbool_t p_bin = svwhilelt_b32(binID,numcell);
            svint32_t np_in_bin=svld1_s32(p_bin,&ptile.m_bin_lengths[binID]);
            svint32_t gap_v=svadd_n_s32_x(p_bin,
                                svcvt_s32_f32_x(p_bin,svmul_n_f32_x(p_bin,
                                svcvt_f32_s32_x(p_bin,np_in_bin),gap_ratio)),
                                1);
            svint32_t block_size_v=svadd_s32_x(p_bin,gap_v,np_in_bin);
            // 1. Vectorized local accumulation (Intra-Vector Exclusive Scan).
            //    This is the core of converting the serial dependency of prefix sum into parallel computation.
            svint32_t local_scan_v = block_size_v;
            local_scan_v = svadd_s32_m(p_bin, local_scan_v, svext_s32(svdup_s32(0), local_scan_v, 16-1));
            local_scan_v = svadd_s32_m(p_bin, local_scan_v, svext_s32(svdup_s32(0), local_scan_v, 16-2));
            local_scan_v = svadd_s32_m(p_bin, local_scan_v, svext_s32(svdup_s32(0), local_scan_v, 16-4));
            local_scan_v = svadd_s32_m(p_bin, local_scan_v, svext_s32(svdup_s32(0), local_scan_v, 16-8));
            local_scan_v = svadd_s32_m(p_bin, local_scan_v, svext_s32(svdup_s32(0), local_scan_v, 16-16));
            
            local_scan_v   = svadd_n_s32_x(p_bin, local_scan_v, current_offset);

            // 2. Store the final calculated offset results.
            //    Note: We start storing from the (binID + 1)-th position of m_bin_offsets.
            svst1_s32(p_bin, &ptile.m_bin_offsets[binID + 1], local_scan_v);
            // 2.2 Store the final m_slot_offsets results.
            svst1_s32(p_bin, &ptile.m_slot_offsets[binID + 1], local_scan_v-gap_v);
            
            // 3. Update the global offset for the next block.
            //    Use svaddv to efficiently calculate the sum of all block_sizes in the current chunk.
            current_offset += svaddv_s32(p_bin, block_size_v);
        }

        // Stage 3.3: Update m_capacity and initialize m_local_index.
        ptile.m_capacity=ptile.m_bin_offsets[numcell];
        ptile.m_local_index.clear();
        ptile.m_local_index.assign(ptile.m_capacity, INVALID_PARTICLE_ID);
        // Initialize with the starting position of each bin.
        std::vector<int> next_write_slot = ptile.m_bin_offsets; 

        // Stage 3.4: Populate m_local_index and m_pid_to_index_map with data.
        for(int ip=0;ip<np_to_deposit;ip++){
            int bin_id =ptile.m_pid_to_bin_map[ip];
            int insert_idx = next_write_slot[bin_id];
            ptile.m_local_index[insert_idx] = ip;
            ptile.m_pid_to_index_map[ip]=insert_idx;
            next_write_slot[bin_id]++;
        }
    }
}

template <int depos_order>
__arm_new("za") inline void sort_3d_sme_kernal (
    const long numcell,
    const std::vector<int>& local_index,
    const std::vector<int>& bin_length,
    const std::vector<int>& bin_offsets,
    std::vector<amrex::Real>& xrhocells,
    std::vector<amrex::Real>& yrhocells,
    std::vector<amrex::Real>& zrhocells,
    std::vector<amrex::Real>& sx_m,
    std::vector<amrex::Real>& sy_m,
    std::vector<amrex::Real>& sz_m,
    std::vector<amrex::Real>& wqx,
    std::vector<amrex::Real>& wqy,
    std::vector<amrex::Real>& wqz,
    amrex::Dim3 len, 
    std::vector<amrex::Real>& testsxwq,
    std::vector<amrex::Real>& test0, int np
)__arm_streaming
{
    using namespace amrex::literals;

    svbool_t p_4=svwhilelt_b64(0,4);
    svbool_t p_4_8=svbic_z(svptrue_b64(),svptrue_b64(),p_4);
    svbool_t p0=svwhilelt_b64(0,2);
    svbool_t p1=svbic_z(p_4,p_4, p0);
    svbool_t p = svptrue_b64();
    MVec vzero(0);
    const uint64_t x_indices[8]={0,2,0,2,1,3,1,3};
    MVec x_indices_v=svld1_u64(svptrue_b64(),x_indices);
    const uint64_t y_indices[8]={0,0,2,2,1,1,3,3};
    MVec y_indices_v=svld1_u64(svptrue_b64(),y_indices);
    const svuint64_t sx_index = svindex_u64(0, np);
    const uint64_t sx_idx1[8]={0,0,0,static_cast<uint64_t>(np),0,0,0,0};
    const svuint64_t sx_idx=svld1_u64(svptrue_b64(),sx_idx1);

    #pragma omp unroll 
    for (int old_bin = 0; old_bin < numcell; ++old_bin) {
        int binlength = bin_length[old_bin];
        int binoffset = bin_offsets[old_bin];
        int last=binlength%2;
        
        svzero_za();
        #pragma omp unroll 
        for(int i=0;i<binlength-last;i+=2){
            int ip1 = local_index[binoffset+i];
            int ip2 = local_index[binoffset+i+1];

            MVec vx1=svld1_gather_index(p0,&sx_m[ip1],sx_index);
            MVec vx2=svld1_gather_index(p0,&sx_m[ip2],sx_index);
            MVec vx_broadcast=svtbl(svzip1(vx1,vx2),x_indices_v);
            MVec vy1=svld1_gather_index(p0,&sy_m[ip1],sx_index);
            MVec vy2=svld1_gather_index(p0,&sy_m[ip2],sx_index);
            MVec vy_broadcast=svtbl(svzip1(vy1,vy2),y_indices_v);
            MVec sxsy_v=vx_broadcast*vy_broadcast; 
            
            MVec sz_x1=svld1_gather_index(p0,&sz_m[ip1],sx_index);
            MVec sz_x2=svld1_gather_index(p1,&sz_m[ip2],sx_idx);
            
            sz_x1*=wqx[ip1];
            MVec sz_xv1=svsel(p0,sz_x1,vzero);
            sz_x2*=wqx[ip2];
            MVec sz_xv2=svsel(p1,sz_x2,vzero);
            MVec sz_xv=svsel(p1,sz_xv2,sz_xv1);

            MVec sz_y1=sz_x1*wqy[ip1];
            MVec sz_yv1=svsel(p0,sz_y1,vzero);
            svmopa_za64_m(0, p_4, svptrue_b64(), sz_xv, sxsy_v);
            MVec sz_y2=sz_x2*wqy[ip2];
            MVec sz_yv2=svsel(p1,sz_y2,vzero);
            MVec sz_yv=svsel(p1,sz_yv2,sz_yv1);
            MVec sz_z1=sz_x1*wqz[ip1];
            MVec sz_zv1=svsel(p0,sz_z1,vzero);
            svmopa_za64_m(1, p_4, svptrue_b64(), sz_yv, sxsy_v);
            MVec sz_z2=sz_x1*wqz[ip2];
            MVec sz_zv2=svsel(p1,sz_z2,vzero);
            MVec sz_zv=svsel(p1,sz_zv2,sz_zv1);
            svmopa_za64_m(2, p_4, svptrue_b64(), sz_zv, sxsy_v);
        }        
        if(last>0){

            int ip1 = local_index[binoffset+binlength-1];

            MVec vx1=svld1_gather_index(p0,&sx_m[ip1],sx_index);
            MVec vy1=svld1_gather_index(p0,&sy_m[ip1],sx_index);
            MVec sxsy_v=svsplice(p0,vx1,vx1)*svzip1(vy1,vy1);

            MVec sz_mm=svld1_gather_index(p0,&sz_m[ip1],sx_index);

            MVec sz_1=sz_mm*wqx[ip1];
            svmopa_za64_m(0, p0, p_4, sz_1, sxsy_v);
            MVec sz_2=sz_mm*wqy[ip1];
            svmopa_za64_m(1, p0, p_4, sz_2, sxsy_v);
            MVec sz_3=sz_mm*wqz[ip1];
            svmopa_za64_m(2, p0, p_4, sz_3, sxsy_v);

        }


        MVec wwx = MVec::Load(p,&xrhocells[8*(old_bin)]);
        MVec wxsxsysz0 = svread_hor_za64_m(vzero,p_4,0,0);
        MVec wxsxsysz1 = svread_hor_za64_m(vzero,p_4,0,1);
        MVec s=svsplice(p_4,wxsxsysz0,wxsxsysz1);
        wwx+=s;
        wxsxsysz0 = svread_hor_za64_m(vzero,p_4_8,0,2);
        wxsxsysz1 = svread_hor_za64_m(vzero,p_4_8,0,3);
        MVec B=svext(wxsxsysz1,wxsxsysz1,4);
        s=svext(wxsxsysz0,B,4);
        wwx+=s;
        wwx.Store(p,&xrhocells[8*(old_bin)]);
        
        MVec wwy = MVec::Load(p,&yrhocells[8*(old_bin)]);
        wxsxsysz0 = svread_hor_za64_m(vzero,p_4,1,0);
        wxsxsysz1 = svread_hor_za64_m(vzero,p_4,1,1);
        MVec sy=svsplice(p_4,wxsxsysz0,wxsxsysz1);
        wwy+=sy;
        wxsxsysz0 = svread_hor_za64_m(vzero,p_4_8,1,2);
        wxsxsysz1 = svread_hor_za64_m(vzero,p_4_8,1,3);
        MVec By=svext(wxsxsysz1,wxsxsysz1,4);
        sy=svext(wxsxsysz0,By,4);
        wwy+=sy;
        wwy.Store(p,&yrhocells[8*(old_bin)]);
        
        MVec wwz = MVec::Load(p,&zrhocells[8*(old_bin)]);
        wxsxsysz0 = svread_hor_za64_m(vzero,p_4,2,0);
        wxsxsysz1 = svread_hor_za64_m(vzero,p_4,2,1);
        MVec sz=svsplice(p_4,wxsxsysz0,wxsxsysz1);
        wwz+=sz;
        wxsxsysz0 = svread_hor_za64_m(vzero,p_4_8,2,2);
        wxsxsysz1 = svread_hor_za64_m(vzero,p_4_8,2,3);
        MVec Bz=svext(wxsxsysz1,wxsxsysz1,4);
        sz=svext(wxsxsysz0,Bz,4);
        wwz+=sy;
        wwz.Store(p,&zrhocells[8*(old_bin)]); 
        // svetime[1]=rdtscm();
        // reducing+=(svetime[1]-svetime[0]); 

    }

}


template <int depos_order>
void doDepositionShapeN_3d_sme (const GetParticlePosition<PIdx>& GetPosition,
                        const amrex::ParticleReal * const wp,
                        const amrex::ParticleReal * const uxp,
                        const amrex::ParticleReal * const uyp,
                        const amrex::ParticleReal * const uzp,
                        const int* ion_lev,
                        amrex::FArrayBox& jx_fab,
                        amrex::FArrayBox& jy_fab,
                        amrex::FArrayBox& jz_fab,
                        long np_to_deposit,
                        amrex::Real relative_time,
                        const amrex::XDim3 & dinv,
                        const amrex::XDim3 & xyzmin,
                        amrex::Dim3 lo,
                        amrex::Dim3 hi,
                        amrex::Dim3 len,
                        amrex::Real q,
                        std::vector<int>& test,
                        std::vector<double>& test0,
                        std::vector<double>& test1,
                        ParticleTileType& ptile, const amrex::Box& box,
                        std::vector<amrex::Real>& test_rhocells,
                        std::vector<amrex::Real>& test_sxwq,
                        [[maybe_unused]]int n_rz_azimuthal_modes,
                        std::vector<uint64_t>& inner_timer
                    )
{
    using namespace amrex::literals;
    #ifdef BREAKDOWN
    uint64_t total_time[3]={0};
    using namespace std::chrono;
    uint64_t precompute=0;
    uint64_t cale_time=0;
    uint64_t sort_time=0;
        uint64_t update[2]={0};
        uint64_t update1[7]={0};
        uint64_t presort=0;
        uint64_t insert=0;
        uint64_t borrow=0;
        uint64_t rebuildtime=0;
        uint64_t rebuild1[7]={0};
    uint64_t reduce_time=0;
    uint64_t svetime[3]={0};
    uint64_t svetime1[3]={0};
    uint64_t branch_time=0;
    
    total_time[0]=rdtscv();
    #endif
    

    const bool do_ionization = ion_lev;

    const amrex::Real invvol = dinv.x*dinv.y*dinv.z; 

    const amrex::Real clightsq = 1.0_rt/PhysConst::c/PhysConst::c;

    amrex::Array4<amrex::Real> const& jx_arr = jx_fab.array();
    amrex::Array4<amrex::Real> const& jy_arr = jy_fab.array();
    amrex::Array4<amrex::Real> const& jz_arr = jz_fab.array();

    constexpr int nshapes = depos_order + 1;
    int vl=svcntd();

    const amrex::ParticleReal* mx = GetPosition.m_x;
    const amrex::ParticleReal* my = GetPosition.m_y;
    const amrex::ParticleReal* mz = GetPosition.m_z;

    auto compute_shape_factor_v2 = [](double* sx, Vec xmid, svbool_t p, int np) {
        intVec i_newv = svcvt_s64_f64_z(p, xmid);
        Vec j = svcvt_f64_s64_z(p, i_newv);
        Vec xint = xmid - j;
        Vec sx0 = 1.0 - xint;
        Vec sx1 = xint;
        sx0.Store(p,&sx[0*np]);
        sx1.Store(p,&sx[1*np]);
        return i_newv;
    };

    // 预计算网格步长
    const int xjstride = jx_arr.jstride;
    const int xkstride = jx_arr.kstride;
    const int yjstride = jy_arr.jstride;
    const int ykstride = jy_arr.kstride;
    const int zjstride = jz_arr.jstride;
    const int zkstride = jz_arr.kstride;

    struct alignas(64) AlignedReal { amrex::Real v; };
    struct alignas(64) Alignedlong { long v; };

    std::vector<amrex::Real> wqx(np_to_deposit, 0.0);
    std::vector<amrex::Real> wqy(np_to_deposit, 0.0);
    std::vector<amrex::Real> wqz(np_to_deposit, 0.0);

    std::vector<amrex::Real> sx_m(np_to_deposit*nshapes, 0.);
    std::vector<amrex::Real> sy_m(np_to_deposit*nshapes, 0.);
    std::vector<amrex::Real> sz_m(np_to_deposit*nshapes, 0.);
    std::vector<long> xoffsets(np_to_deposit);
    amrex::IntVect box_shape = box.length();
    long nnx = len.x;
    long nny = len.y;
    long nnxny = nnx*nny;
    alignas(64) std::vector<int> newbin(np_to_deposit,0);
    amrex::IntVect const jx_type = jx_fab.box().type();
    amrex::IntVect const jy_type = jy_fab.box().type();
    amrex::IntVect const jz_type = jz_fab.box().type();
    amrex::Real jm=0;
    amrex::Real km=0;
    amrex::Real lm=0;
    #ifdef BREAKDOWN
    svetime[0]=rdtscv();
    #endif
    #pragma omp unroll
    for(long ip=0;ip<np_to_deposit;ip+=vl){


        svbool_t p=svwhilelt_b64(ip,np_to_deposit);
        // Inverse of Lorentz factor gamma
        Vec uxp_v=Vec::Load(p,&uxp[ip]);
        Vec uyp_v=Vec::Load(p,&uyp[ip]);
        Vec uzp_v=Vec::Load(p,&uzp[ip]);
        Vec gaminv = 1._rt / (1._rt + uxp_v * uxp_v * clightsq
                                    + uyp_v * uyp_v * clightsq
                                    + uzp_v * uzp_v * clightsq).Sqrt();
        // Product of particle charges and weights
        Vec wp_v=Vec::Load(p,&wp[ip]);
        Vec wq_v = q * wp_v;

        // Current particle positions (in physical units)
        Vec xp = Vec::Load(p, &mx[ip]);
        Vec yp = Vec::Load(p, &my[ip]);
        Vec zp = Vec::Load(p, &mz[ip]);

        // Particle velocities
        Vec vx_v = uxp_v * gaminv;
        Vec vy_v = uyp_v * gaminv;
        Vec vz_v = uzp_v * gaminv;

        // if (do_ionization) wq *= ion_lev[ip];
        Vec wqx_v = wq_v * invvol * vx_v;
        Vec wqy_v = wq_v * invvol * vy_v;
        Vec wqz_v = wq_v * invvol * vz_v;

        wqx_v.Store(p, &wqx[ip]);
        wqy_v.Store(p, &wqy[ip]);
        wqz_v.Store(p, &wqz[ip]);

        Vec xmid = ((xp - xyzmin.x) + relative_time * vx_v) * dinv.x;
        Vec ymid = ((yp - xyzmin.y) + relative_time * vy_v) * dinv.y;
        Vec zmid = ((zp - xyzmin.z) + relative_time * vz_v) * dinv.z;
        intVec j_cellv = compute_shape_factor_v2( &sx_m[ip], xmid, p,np_to_deposit);
        intVec k_cellv = compute_shape_factor_v2( &sy_m[ip], ymid, p,np_to_deposit);
        intVec l_cellv = compute_shape_factor_v2( &sz_m[ip], zmid, p,np_to_deposit); 
        intVec new_bin_v = (j_cellv) + 
                            (k_cellv)* nnx + 
                            (l_cellv) * nnxny;
        
        svst1w_s64 (p, &newbin[ip], new_bin_v);
    }
    #ifdef BREAKDOWN
    svetime[1]=rdtscv();
    precompute+=(svetime[1]-svetime[0]); 
    #endif

    const int numcell=ptile.m_num_bins;


    std::vector<amrex::Real> xrhocells(8*numcell, 0.0);
    std::vector<amrex::Real> testsxwq(8*np_to_deposit, 0.0);
    std::vector<amrex::Real> yrhocells(8*numcell, 0.0);
    std::vector<amrex::Real> zrhocells(8*numcell, 0.0);
    #ifdef BREAKDOWN
    svetime1[0]=rdtscv();
    #endif
    increment_sort_particles (np_to_deposit, ptile,newbin, numcell);
    #ifdef BREAKDOWN
    svetime1[1]=rdtscv();
    sort_time+=(svetime1[1]-svetime1[0]); 
    #endif

 
    auto& local_index = ptile.m_local_index;
    auto& bin_offsets = ptile.m_bin_offsets;
    auto& bin_length = ptile.m_bin_lengths;
    #ifdef BREAKDOWN
    svetime[0]=rdtscv();
    #endif

    sort_3d_sme_kernal<1>(
        numcell,local_index,
        bin_length,bin_offsets,
        xrhocells,yrhocells,zrhocells,
        sx_m,sy_m,sz_m,
        wqx,wqy,wqz,len,testsxwq,test0,np_to_deposit);

    #ifdef BREAKDOWN
    svetime[1]=rdtscv();
    cale_time+=(svetime[1]-svetime[0]);
    #endif

    long nx=xjstride;
    long nxy=xkstride;
    long moff[8] = {0, 1, nx, nx+1, nxy, nxy+1, nx+nxy, nx+nxy+1};

    svetime[0]=rdtscv();
    #pragma omp simd
    for (int iz = 0; iz < len.z; ++iz) {
        #pragma omp simd
        for (int iy = 0; iy < len.y; ++iy) { 
            #pragma unroll      
            for (int ix = 0; ix < len.x; ix+=vl) {

                // vector
                svbool_t p=svwhilelt_b64(ix,(len.x));
                svuint64_t rho_index = svindex_u64(0ULL, 8ULL);
                long xoffset=ix + iy * xjstride + iz * xkstride;
                long cell_idx=ix + iy * nnx + iz * nnxny;
                long jxoffset=xoffset;
                long irhox=8*cell_idx;
                Vec jx0=Vec::Load(p,&jx_arr.p[jxoffset+moff[0]]);
                svfloat64_t rhox0=svld1_gather_u64index_f64(p,&xrhocells[0+irhox],rho_index);
                jx0+=rhox0;
                jx0.Store(p,&jx_arr.p[jxoffset+moff[0]]);

                Vec jx1=Vec::Load(p,&jx_arr.p[jxoffset+moff[1]]);
                svfloat64_t rhox1=svld1_gather_u64index_f64(p,&xrhocells[1+irhox],rho_index);
                jx1+=rhox1;
                jx1.Store(p,&jx_arr.p[jxoffset+moff[1]]);

                Vec jx2=Vec::Load(p,&jx_arr.p[jxoffset+moff[2]]);
                svfloat64_t rhox2=svld1_gather_u64index_f64(p,&xrhocells[2+irhox],rho_index);
                jx2+=rhox2;
                jx2.Store(p,&jx_arr.p[jxoffset+moff[2]]);

                Vec jx3=Vec::Load(p,&jx_arr.p[jxoffset+moff[3]]);
                svfloat64_t rhox3=svld1_gather_u64index_f64(p,&xrhocells[3+irhox],rho_index);
                jx3+=rhox3;
                jx3.Store(p,&jx_arr.p[jxoffset+moff[3]]);

                Vec jx4=Vec::Load(p,&jx_arr.p[jxoffset+moff[4]]);
                svfloat64_t rhox4=svld1_gather_u64index_f64(p,&xrhocells[4+irhox],rho_index);
                jx4+=rhox4;
                jx4.Store(p,&jx_arr.p[jxoffset+moff[4]]);

                Vec jx5=Vec::Load(p,&jx_arr.p[jxoffset+moff[5]]);
                svfloat64_t rhox5=svld1_gather_u64index_f64(p,&xrhocells[5+irhox],rho_index);
                jx5+=rhox5;
                jx5.Store(p,&jx_arr.p[jxoffset+moff[5]]);

                Vec jx6=Vec::Load(p,&jx_arr.p[jxoffset+moff[6]]);
                svfloat64_t rhox6=svld1_gather_u64index_f64(p,&xrhocells[6+irhox],rho_index);
                jx6+=rhox6;
                jx6.Store(p,&jx_arr.p[jxoffset+moff[6]]);

                Vec jx7=Vec::Load(p,&jx_arr.p[jxoffset+moff[7]]);
                svfloat64_t rhox7=svld1_gather_u64index_f64(p,&xrhocells[7+irhox],rho_index);
                jx7+=rhox7;
                jx7.Store(p,&jx_arr.p[jxoffset+moff[7]]);
                
                // y
                long jyoffset=xoffset;
                long irhoy=irhox;
                Vec jy0=Vec::Load(p,&jy_arr.p[jyoffset+moff[0]]);
                svfloat64_t rhoy0=svld1_gather_u64index_f64(p,&yrhocells[0+irhoy],rho_index);
                jy0+=rhoy0;
                jy0.Store(p,&jy_arr.p[jyoffset+moff[0]]);

                Vec jy1=Vec::Load(p,&jy_arr.p[jyoffset+moff[1]]);
                svfloat64_t rhoy1=svld1_gather_u64index_f64(p,&yrhocells[1+irhoy],rho_index);
                jy1+=rhoy1;
                jy1.Store(p,&jy_arr.p[jyoffset+moff[1]]);

                Vec jy2=Vec::Load(p,&jy_arr.p[jyoffset+moff[2]]);
                svfloat64_t rhoy2=svld1_gather_u64index_f64(p,&yrhocells[2+irhoy],rho_index);
                jy2+=rhoy2;
                jy2.Store(p,&jy_arr.p[jyoffset+moff[2]]);

                Vec jy3=Vec::Load(p,&jy_arr.p[jyoffset+moff[3]]);
                svfloat64_t rhoy3=svld1_gather_u64index_f64(p,&yrhocells[3+irhoy],rho_index);
                jy3+=rhoy3;
                jy3.Store(p,&jy_arr.p[jyoffset+moff[3]]);

                Vec jy4=Vec::Load(p,&jy_arr.p[jyoffset+moff[4]]);
                svfloat64_t rhoy4=svld1_gather_u64index_f64(p,&yrhocells[4+irhoy],rho_index);
                jy4+=rhoy4;
                jy4.Store(p,&jy_arr.p[jyoffset+moff[4]]);

                Vec jy5=Vec::Load(p,&jy_arr.p[jyoffset+moff[5]]);
                svfloat64_t rhoy5=svld1_gather_u64index_f64(p,&yrhocells[5+irhoy],rho_index);
                jy5+=rhoy5;
                jy5.Store(p,&jy_arr.p[jyoffset+moff[5]]);

                Vec jy6=Vec::Load(p,&jy_arr.p[jyoffset+moff[6]]);
                svfloat64_t rhoy6=svld1_gather_u64index_f64(p,&yrhocells[6+irhoy],rho_index);
                jy6+=rhoy6;
                jy6.Store(p,&jy_arr.p[jyoffset+moff[6]]);

                Vec jy7=Vec::Load(p,&jy_arr.p[jyoffset+moff[7]]);
                svfloat64_t rhoy7=svld1_gather_u64index_f64(p,&yrhocells[7+irhoy],rho_index);
                jy7+=rhoy7;
                jy7.Store(p,&jy_arr.p[jyoffset+moff[7]]);

                // z
                long jzoffset=xoffset;
                long irhoz=irhox;
                Vec jz0=Vec::Load(p,&jz_arr.p[jzoffset+moff[0]]);
                svfloat64_t rhoz0=svld1_gather_u64index_f64(p,&zrhocells[0+irhoz],rho_index);
                jz0+=rhoz0;
                jz0.Store(p,&jz_arr.p[jzoffset+moff[0]]);

                Vec jz1=Vec::Load(p,&jz_arr.p[jzoffset+moff[1]]);
                svfloat64_t rhoz1=svld1_gather_u64index_f64(p,&zrhocells[1+irhoz],rho_index);
                jz1+=rhoz1;
                jz1.Store(p,&jz_arr.p[jzoffset+moff[1]]);

                Vec jz2=Vec::Load(p,&jz_arr.p[jzoffset+moff[2]]);
                svfloat64_t rhoz2=svld1_gather_u64index_f64(p,&zrhocells[2+irhoz],rho_index);
                jz2+=rhoz2;
                jz2.Store(p,&jz_arr.p[jzoffset+moff[2]]);

                Vec jz3=Vec::Load(p,&jz_arr.p[jzoffset+moff[3]]);
                svfloat64_t rhoz3=svld1_gather_u64index_f64(p,&zrhocells[3+irhoz],rho_index);
                jz3+=rhoz3;
                jz3.Store(p,&jz_arr.p[jzoffset+moff[3]]);

                Vec jz4=Vec::Load(p,&jz_arr.p[jzoffset+moff[4]]);
                svfloat64_t rhoz4=svld1_gather_u64index_f64(p,&zrhocells[4+irhoz],rho_index);
                jz4+=rhoz4;
                jz4.Store(p,&jz_arr.p[jzoffset+moff[4]]);

                Vec jz5=Vec::Load(p,&jz_arr.p[jzoffset+moff[5]]);
                svfloat64_t rhoz5=svld1_gather_u64index_f64(p,&zrhocells[5+irhoz],rho_index);
                jz5+=rhoz5;
                jz5.Store(p,&jz_arr.p[jzoffset+moff[5]]);

                Vec jz6=Vec::Load(p,&jz_arr.p[jzoffset+moff[6]]);
                svfloat64_t rhoz6=svld1_gather_u64index_f64(p,&zrhocells[6+irhoz],rho_index);
                jz6+=rhoz6;
                jz6.Store(p,&jz_arr.p[jzoffset+moff[6]]);

                Vec jz7=Vec::Load(p,&jz_arr.p[jzoffset+moff[7]]);
                svfloat64_t rhoz7=svld1_gather_u64index_f64(p,&zrhocells[7+irhoz],rho_index);
                jz7+=rhoz7;
                jz7.Store(p,&jz_arr.p[jzoffset+moff[7]]);

            }
        }
    }
    #ifdef BREAKDOWN
    svetime[1]=rdtscv();
    reduce_time+=(svetime[1]-svetime[0]);

    total_time[1]=rdtscv();
    total_time[2]=total_time[1]-total_time[0];
    inner_timer[0]=precompute+cale_time+sort_time+reduce_time;
    inner_timer[1]=precompute;
    inner_timer[2]=cale_time;
    inner_timer[3]=sort_time;
    inner_timer[4]=reduce_time;
    #endif

}
 
template <int depos_order>
__arm_new("za") inline void sme_kernal_pure_sme(
    long np_to_deposit,
    std::vector<int>& newbin,
    std::vector<amrex::Real> &xrhocells,
    std::vector<amrex::Real> &yrhocells,
    std::vector<amrex::Real> &zrhocells,
    std::vector<amrex::Real> &sx_m,
    std::vector<amrex::Real> &sy_m,
    std::vector<amrex::Real> &sz_m,
    std::vector<amrex::Real> &wqx,
    std::vector<amrex::Real> &wqy,
    std::vector<amrex::Real> &wqz
    ) __arm_streaming
{
    using namespace amrex::literals;
    svbool_t p_4 = svwhilelt_b64(0, 4);
    svbool_t p_4_8 = svbic_z(svptrue_b64(), svptrue_b64(), p_4);
    svbool_t p0 = svwhilelt_b64(0, 2);
    svbool_t p1 = svbic_z(p_4, p_4, p0);
    svbool_t p = svptrue_b64();
    MVec vzero(0);

    
    int last=np_to_deposit%2;
    for (int idx = 0; idx < np_to_deposit-last; idx += 2)
    {
        svzero_za();
        int ip1 = idx;
        int ip2 = idx + 1;

        MVec sz_xv=svdup_n_f64(0);
        MVec sz_yv=svdup_n_f64(0);
        MVec sz_zv=svdup_n_f64(0);
        MVec sz_x1 = svld1(p0,&sz_m[ip1*2]);
        MVec sz_x2 = svld1(p1,&sz_m[ip1*2]);
        sz_xv=svsel(p0,sz_x1,sz_xv);
        sz_xv=svsel(p1,sz_x2,sz_xv);

        MVec vx1=MVec::Load(p0, &sx_m[ip1*2]);
        MVec vx2=MVec::Load(p0, &sx_m[ip2*2]);

        MVec vy1=MVec::Load(p0, &sy_m[ip1*2]);
        MVec vy2=MVec::Load(p0, &sy_m[ip2*2]);

        const uint64_t x_indices[8]={0,2,0,2,1,3,1,3};
        MVec vx_broadcast=svtbl(svzip1(vx1,vx2),svld1_u64(svptrue_b64(),x_indices));
        const uint64_t y_indices[8]={0,0,2,2,1,1,3,3};
        MVec vy_broadcast=svtbl(svzip1(vy1,vy2),svld1_u64(svptrue_b64(),y_indices));

        MVec sxsy_v=vx_broadcast*vy_broadcast;      

        svmopa_za64_m(0, p_4, svptrue_b64(), sz_xv, sxsy_v);

        int idx1=newbin[ip1];
        MVec wwx = MVec::Load(p,&xrhocells[8*(idx1)]);
        MVec wwy = MVec::Load(p,&yrhocells[8*(idx1)]);
        MVec wwz = MVec::Load(p,&zrhocells[8*(idx1)]);
        MVec wxsxsysz0 = svread_hor_za64_m(vzero,p_4,0,0);
        MVec wxsxsysz1 = svread_hor_za64_m(vzero,p_4,0,1);
        MVec s=svsplice(p_4,wxsxsysz0,wxsxsysz1);
        wwx+=(s*wqx[ip1]);
        wwy+=(s*wqy[ip1]);
        wwz+=(s*wqz[ip1]);
        wwx.Store(p,&xrhocells[8*(idx1)]);
        wwy.Store(p,&yrhocells[8*(idx1)]);
        wwz.Store(p,&zrhocells[8*(idx1)]);

        int idx2=newbin[ip2];
        wwx = MVec::Load(p,&xrhocells[8*(idx2)]);
        wwy = MVec::Load(p,&yrhocells[8*(idx2)]);
        wwz = MVec::Load(p,&zrhocells[8*(idx2)]);
        wxsxsysz0 = svread_hor_za64_m(vzero,p_4_8,0,2);
        wxsxsysz1 = svread_hor_za64_m(vzero,p_4_8,0,3);
        MVec B=svext(wxsxsysz1,wxsxsysz1,4);
        MVec s2=svext(wxsxsysz0,B,4);
        wwx+=(s2*wqx[ip2]);
        wwy+=(s2*wqy[ip2]);
        wwz+=(s2*wqz[ip2]);
        wwx.Store(p,&xrhocells[8*(idx2)]);
        wwy.Store(p,&yrhocells[8*(idx2)]);
        wwz.Store(p,&zrhocells[8*(idx2)]); 
    }


    if(np_to_deposit%2){
        svzero_za();

        int ip1 = np_to_deposit-1;
        MVec sz_v=svdup_n_f64(0);
        MVec sz_1 = svld1(p0,&sz_m[ip1*2]);
        // sz_1*=wqx[ip1];
        sz_v=svsel(p0,sz_1,sz_v);
        // svbool_t p1=svwhilelt_b64(2,4);
        MVec vx1=MVec::Load(p0, &sx_m[ip1*2]);
        MVec vy1=MVec::Load(p0, &sy_m[ip1*2]);
        MVec sxsy_v=svsplice(p0,vx1,vy1);
        svmopa_za64_m(0, p0, p_4, sz_v, sxsy_v);
        int idx1=newbin[ip1];
        MVec wwx = MVec::Load(p,&xrhocells[8*(idx1)]);
        MVec wwy = MVec::Load(p,&yrhocells[8*(idx1)]);
        MVec wwz = MVec::Load(p,&zrhocells[8*(idx1)]);
        MVec wxsxsysz0 = svread_hor_za64_m(vzero,p_4,0,0);
        MVec wxsxsysz1 = svread_hor_za64_m(vzero,p_4,0,1);
        MVec s=svsplice(p_4,wxsxsysz0,wxsxsysz1);
        wwx+=(s*wqx[ip1]);
        wwy+=(s*wqy[ip1]);
        wwz+=(s*wqz[ip1]);
        wwx.Store(p,&xrhocells[8*(idx1)]);
        wwy.Store(p,&yrhocells[8*(idx1)]);
        wwz.Store(p,&zrhocells[8*(idx1)]);
    }

}

template <int depos_order>
void doDepositionShapeN_pure_sme(const GetParticlePosition<PIdx> &GetPosition,
                                        const amrex::ParticleReal *const wp,
                                        const amrex::ParticleReal *const uxp,
                                        const amrex::ParticleReal *const uyp,
                                        const amrex::ParticleReal *const uzp,
                                        const int *ion_lev,
                                        amrex::FArrayBox &jx_fab,
                                        amrex::FArrayBox &jy_fab,
                                        amrex::FArrayBox &jz_fab,
                                        long np_to_deposit,
                                        amrex::Real relative_time,
                                        const amrex::XDim3 &dinv,
                                        const amrex::XDim3 &xyzmin,
                                        amrex::Dim3 lo,
                                        amrex::Dim3 hi,
                                        amrex::Dim3 len,
                                        amrex::Real q,
                                        std::vector<int> &test,
                                        std::vector<int> &test0,
                                        WarpXParticleContainer::ParticleTileType &ptile, const amrex::Box &box,
                                        [[maybe_unused]] int n_rz_azimuthal_modes)
{
    using namespace std::chrono;
    using namespace amrex::literals;
    
    amrex::Array4<amrex::Real> const &jx_arr = jx_fab.array();
    amrex::Array4<amrex::Real> const &jy_arr = jy_fab.array();
    amrex::Array4<amrex::Real> const &jz_arr = jz_fab.array();

    const int xjstride = jx_arr.jstride;
    const int xkstride = jx_arr.kstride;
    const int yjstride = jy_arr.jstride;
    const int ykstride = jy_arr.kstride;
    const int zjstride = jz_arr.jstride;
    const int zkstride = jz_arr.kstride;

    long nx = xjstride;
    long nxy = xkstride;
    long moff[8] = {0, 1, nx, nx + 1, nxy, nxy + 1, nx + nxy, nx + nxy + 1};

    const bool do_ionization = ion_lev;

    const amrex::Real invvol = dinv.x * dinv.y * dinv.z;

    const amrex::Real clightsq = 1.0_rt / PhysConst::c / PhysConst::c;

    const long numcell = jx_fab.box().numPts();

    amrex::IntVect const jx_type = jx_fab.box().type();
    amrex::IntVect const jy_type = jy_fab.box().type();
    amrex::IntVect const jz_type = jz_fab.box().type();

    std::vector<amrex::Real> wqx(np_to_deposit, 0.0_rt), wqy(np_to_deposit, 0.0_rt), wqz(np_to_deposit, 0.0_rt); 
    
    std::vector<amrex::Real> sx_m(2 * np_to_deposit, 0);
    std::vector<amrex::Real> sy_m(2 * np_to_deposit, 0);
    std::vector<amrex::Real> sz_m(2 * np_to_deposit, 0);

    std::vector<amrex::Real> xrhocells(8 * numcell, 0.0);
    std::vector<amrex::Real> yrhocells(8 * numcell, 0.0);
    std::vector<amrex::Real> zrhocells(8 * numcell, 0.0);

    int j_m=-1;
    int k_m=-1;
    int l_m=-1;
    alignas(64) std::vector<int> newbin(np_to_deposit,-1);

    for (int ip = 0; ip < np_to_deposit; ip++)
    {
        amrex::ParticleReal xp, yp, zp;
        GetPosition(ip, xp, yp, zp);

        // --- Get particle quantities
        const amrex::Real gaminv = 1.0_rt / std::sqrt(1.0_rt + uxp[ip] * uxp[ip] * clightsq + uyp[ip] * uyp[ip] * clightsq + uzp[ip] * uzp[ip] * clightsq);
        const amrex::Real vx = uxp[ip] * gaminv;
        const amrex::Real vy = uyp[ip] * gaminv;
        const amrex::Real vz = uzp[ip] * gaminv;

        amrex::Real wq = q * wp[ip];
        if (do_ionization)
        {
            wq *= ion_lev[ip];
        }

        constexpr int NODE = amrex::IndexType::NODE;
        constexpr int CELL = amrex::IndexType::CELL;

        // wqx, wqy wqz are particle current in each direction
        wqx[ip] = wq * invvol * vx;
        wqy[ip] = wq * invvol * vy;
        wqz[ip] = wq * invvol * vz;

        // --- Compute shape factors
        Compute_shape_factor<depos_order> const compute_shape_factor;

        // const auto j = static_cast<int>(xmid);
        // x direction
        // Get particle position after 1/2 push back in position
        // Keep these double to avoid bug in single precision

        const double xmid = ((xp - xyzmin.x) + relative_time * vx) * dinv.x;

        double sx_node[depos_order + 1] = {0.};
        double sx_cell[depos_order + 1] = {0.};
        int j_node = 0;
        int j_cell = 0;
        if (jx_type[0] == NODE || jy_type[0] == NODE || jz_type[0] == NODE)
        {
            j_node = compute_shape_factor(sx_node, xmid);
        }
        if (jx_type[0] == CELL || jy_type[0] == CELL || jz_type[0] == CELL)
        {
            j_cell = compute_shape_factor(sx_cell, xmid - 0.5);
        }

        for (int ix = 0; ix <= depos_order; ix++)
        {
            sx_m[ip * 2 + ix] = ((jx_type[0] == NODE) ? amrex::Real(sx_node[ix]) : amrex::Real(sx_cell[ix]));
        }

        j_m = ((jz_type[0] == NODE) ? j_node : j_cell);

        // y direction
        // Keep these double to avoid bug in single precision
        const double ymid = ((yp - xyzmin.y) + relative_time * vy) * dinv.y;
        double sy_node[depos_order + 1] = {0.};
        double sy_cell[depos_order + 1] = {0.};
        int k_node = 0;
        int k_cell = 0;
        if (jx_type[1] == NODE || jy_type[1] == NODE || jz_type[1] == NODE)
        {
            k_node = compute_shape_factor(sy_node, ymid);
        }
        if (jx_type[1] == CELL || jy_type[1] == CELL || jz_type[1] == CELL)
        {
            k_cell = compute_shape_factor(sy_cell, ymid - 0.5);
        }
        for (int iy = 0; iy <= depos_order; iy++)
        {
            sy_m[ip * 2 + iy] = ((jx_type[1] == NODE) ? amrex::Real(sy_node[iy]) : amrex::Real(sy_cell[iy]));
        }
        k_m = ((jx_type[1] == NODE) ? k_node : k_cell);

        // z direction
        // Keep these double to avoid bug in single precision
        constexpr int zdir = WARPX_ZINDEX;
        const double zmid = ((zp - xyzmin.z) + relative_time * vz) * dinv.z;
        double sz_node[depos_order + 1] = {0.};
        double sz_cell[depos_order + 1] = {0.};
        int l_node = 0;
        int l_cell = 0;
        if (jx_type[zdir] == NODE || jy_type[zdir] == NODE || jz_type[zdir] == NODE)
        {
            l_node = compute_shape_factor(sz_node, zmid);
        }
        if (jx_type[zdir] == CELL || jy_type[zdir] == CELL || jz_type[zdir] == CELL)
        {
            l_cell = compute_shape_factor(sz_cell, zmid - 0.5);
        }
        for (int iz = 0; iz <= depos_order; iz++)
        {
            sz_m[ip * 2 + iz] = ((jx_type[zdir] == NODE) ? amrex::Real(sz_node[iz]) : amrex::Real(sz_cell[iz]));
        }
        l_m = ((jz_type[zdir] == NODE) ? l_node : l_cell);
        newbin[ip]=j_m+k_m*xjstride+l_m*xkstride;
    }

    amrex::IntVect box_shape = box.length();

    int nx_box = box_shape[0];
    int ny_box = box_shape[1];
    int nxny_box = nx_box * ny_box;

    sme_kernal_pure_sme<1>(np_to_deposit,  
                    newbin,
                    xrhocells, yrhocells, zrhocells,
                    sx_m, sy_m, sz_m,
                    wqx, wqy, wqz);

    int errcount = 0;
    for (int iz = 0; iz <= len.z; ++iz)
    {
        for (int iy = 0; iy <= len.y; ++iy)
        {
            // scalar mode
            for(int ix = 0; ix <= len.x; ix++){
                
                long jxoffset = ix + iy * xjstride + iz * xkstride;
                long xoffset = 8*jxoffset;
            
                jx_arr.p[jxoffset+moff[0]]+=xrhocells[0+xoffset];
                jx_arr.p[jxoffset+moff[1]]+=xrhocells[1+xoffset];
                jx_arr.p[jxoffset+moff[2]]+=xrhocells[2+xoffset];
                jx_arr.p[jxoffset+moff[3]]+=xrhocells[3+xoffset];
                jx_arr.p[jxoffset+moff[4]]+=xrhocells[4+xoffset];
                jx_arr.p[jxoffset+moff[5]]+=xrhocells[5+xoffset];
                jx_arr.p[jxoffset+moff[6]]+=xrhocells[6+xoffset];
                jx_arr.p[jxoffset+moff[7]]+=xrhocells[7+xoffset];
                
                long jyoffset = jxoffset;
                long yoffset = xoffset;
                jy_arr.p[jyoffset+moff[0]]+=yrhocells[0+yoffset];
                jy_arr.p[jyoffset+moff[1]]+=yrhocells[1+yoffset];
                jy_arr.p[jyoffset+moff[2]]+=yrhocells[2+yoffset];
                jy_arr.p[jyoffset+moff[3]]+=yrhocells[3+yoffset];
                jy_arr.p[jyoffset+moff[4]]+=yrhocells[4+yoffset];
                jy_arr.p[jyoffset+moff[5]]+=yrhocells[5+yoffset];
                jy_arr.p[jyoffset+moff[6]]+=yrhocells[6+yoffset];
                jy_arr.p[jyoffset+moff[7]]+=yrhocells[7+yoffset];

                long jzoffset = jxoffset;
                long zoffset = xoffset;
                jz_arr.p[jzoffset+moff[0]]+=zrhocells[0+zoffset];
                jz_arr.p[jzoffset+moff[1]]+=zrhocells[1+zoffset];
                jz_arr.p[jzoffset+moff[2]]+=zrhocells[2+zoffset];
                jz_arr.p[jzoffset+moff[3]]+=zrhocells[3+zoffset];
                jz_arr.p[jzoffset+moff[4]]+=zrhocells[4+zoffset];
                jz_arr.p[jzoffset+moff[5]]+=zrhocells[5+zoffset];
                jz_arr.p[jzoffset+moff[6]]+=zrhocells[6+zoffset];
                jz_arr.p[jzoffset+moff[7]]+=zrhocells[7+zoffset];
            }

        }
    }

}

template <int depos_order>
__arm_new("za") inline void sort_3d_sme_kernal_hybrid_nosort(
    const long np_to_deposit, const std::vector<long>& newbin,
    std::vector<amrex::Real>& xrhocells,
    std::vector<amrex::Real>& yrhocells,
    std::vector<amrex::Real>& zrhocells,
    std::vector<amrex::Real>& sx_m,
    std::vector<amrex::Real>& sy_m,
    std::vector<amrex::Real>& sz_m,
    std::vector<amrex::Real>& wqx,
    std::vector<amrex::Real>& wqy,
    std::vector<amrex::Real>& wqz
)__arm_streaming
{
    using namespace amrex::literals;

    svbool_t p_4=svwhilelt_b64(0,4);
    svbool_t p_4_8=svbic_z(svptrue_b64(),svptrue_b64(),p_4);
    svbool_t p0=svwhilelt_b64(0,2);
    svbool_t p1=svbic_z(p_4,p_4, p0);
    svbool_t p = svptrue_b64();
    MVec vzero(0);
    int last=np_to_deposit%2;
    for(long ip=0;ip<np_to_deposit-last;ip+=2){
        svzero_za();
        int ip1 = ip;
        int ip2 = ip+1;
        
        MVec sz_xv=svdup_n_f64(0);
        MVec sz_yv=svdup_n_f64(0);
        MVec sz_zv=svdup_n_f64(0);
        MVec sz_x1 = svld1(p0,&sz_m[ip1*2]);
        MVec sz_x2 = svld1(p1,&sz_m[ip1*2]);
        sz_xv=svsel(p0,sz_x1,sz_xv);
        sz_xv=svsel(p1,sz_x2,sz_xv);

        MVec vx1=MVec::Load(p0, &sx_m[ip1*2]);
        MVec vx2=MVec::Load(p0, &sx_m[ip2*2]);

        MVec vy1=MVec::Load(p0, &sy_m[ip1*2]);
        MVec vy2=MVec::Load(p0, &sy_m[ip2*2]);

        const uint64_t x_indices[8]={0,2,0,2,1,3,1,3};
        MVec vx_broadcast=svtbl(svzip1(vx1,vx2),svld1_u64(svptrue_b64(),x_indices));
        const uint64_t y_indices[8]={0,0,2,2,1,1,3,3};
        MVec vy_broadcast=svtbl(svzip1(vy1,vy2),svld1_u64(svptrue_b64(),y_indices));

        MVec sxsy_v=vx_broadcast*vy_broadcast;      

        svmopa_za64_m(0, p_4, svptrue_b64(), sz_xv, sxsy_v);

        int idx1=newbin[ip1];
        MVec wwx = MVec::Load(p,&xrhocells[8*(idx1)]);
        MVec wwy = MVec::Load(p,&yrhocells[8*(idx1)]);
        MVec wwz = MVec::Load(p,&zrhocells[8*(idx1)]);
        MVec wxsxsysz0 = svread_hor_za64_m(vzero,p_4,0,0);
        MVec wxsxsysz1 = svread_hor_za64_m(vzero,p_4,0,1);
        MVec s=svsplice(p_4,wxsxsysz0,wxsxsysz1);
        wwx+=(s*wqx[ip1]);
        wwy+=(s*wqy[ip1]);
        wwz+=(s*wqz[ip1]);
        wwx.Store(p,&xrhocells[8*(idx1)]);
        wwy.Store(p,&yrhocells[8*(idx1)]);
        wwz.Store(p,&zrhocells[8*(idx1)]);

        int idx2=newbin[ip2];
        wwx = MVec::Load(p,&xrhocells[8*(idx2)]);
        wwy = MVec::Load(p,&yrhocells[8*(idx2)]);
        wwz = MVec::Load(p,&zrhocells[8*(idx2)]);
        wxsxsysz0 = svread_hor_za64_m(vzero,p_4_8,0,2);
        wxsxsysz1 = svread_hor_za64_m(vzero,p_4_8,0,3);
        MVec B=svext(wxsxsysz1,wxsxsysz1,4);
        MVec s2=svext(wxsxsysz0,B,4);
        wwx+=(s2*wqx[ip2]);
        wwy+=(s2*wqy[ip2]);
        wwz+=(s2*wqz[ip2]);
        wwx.Store(p,&xrhocells[8*(idx2)]);
        wwy.Store(p,&yrhocells[8*(idx2)]);
        wwz.Store(p,&zrhocells[8*(idx2)]); 
            
    }

    if(np_to_deposit%2){
        svzero_za();

        int ip1 = np_to_deposit-1;
        MVec sz_v=svdup_n_f64(0);
        MVec sz_1 = svld1(p0,&sz_m[ip1*2]);
        // sz_1*=wqx[ip1];
        sz_v=svsel(p0,sz_1,sz_v);
        // svbool_t p1=svwhilelt_b64(2,4);
        MVec vx1=MVec::Load(p0, &sx_m[ip1*2]);
        MVec vy1=MVec::Load(p0, &sy_m[ip1*2]);
        MVec sxsy_v=svsplice(p0,vx1,vy1);
        svmopa_za64_m(0, p0, p_4, sz_v, sxsy_v);
        int idx1=newbin[ip1];
        MVec wwx = MVec::Load(p,&xrhocells[8*(idx1)]);
        MVec wwy = MVec::Load(p,&yrhocells[8*(idx1)]);
        MVec wwz = MVec::Load(p,&zrhocells[8*(idx1)]);
        MVec wxsxsysz0 = svread_hor_za64_m(vzero,p_4,0,0);
        MVec wxsxsysz1 = svread_hor_za64_m(vzero,p_4,0,1);
        MVec s=svsplice(p_4,wxsxsysz0,wxsxsysz1);
        wwx+=(s*wqx[ip1]);
        wwy+=(s*wqy[ip1]);
        wwz+=(s*wqz[ip1]);
        wwx.Store(p,&xrhocells[8*(idx1)]);
        wwy.Store(p,&yrhocells[8*(idx1)]);
        wwz.Store(p,&zrhocells[8*(idx1)]);

    }
}


template <int depos_order>
void doDepositionShapeN_3d_sme_hybrid_nosort (const GetParticlePosition<PIdx>& GetPosition,
                         const amrex::ParticleReal * const wp,
                         const amrex::ParticleReal * const uxp,
                         const amrex::ParticleReal * const uyp,
                         const amrex::ParticleReal * const uzp,
                         const int* ion_lev,
                         amrex::FArrayBox& jx_fab,
                         amrex::FArrayBox& jy_fab,
                         amrex::FArrayBox& jz_fab,
                         long np_to_deposit,
                         amrex::Real relative_time,
                         const amrex::XDim3 & dinv,
                         const amrex::XDim3 & xyzmin,
                         amrex::Dim3 lo,
                         amrex::Dim3 hi,
                         amrex::Dim3 len,
                         amrex::Real q,
                         std::vector<int>& test,
                         std::vector<int>& test0,
                         ParticleTileType& ptile, const amrex::Box& box,
                         [[maybe_unused]]int n_rz_azimuthal_modes)
{
    using namespace amrex::literals;
    
    const bool do_ionization = ion_lev;

    const amrex::Real invvol = dinv.x*dinv.y*dinv.z; 

    const amrex::Real clightsq = 1.0_rt/PhysConst::c/PhysConst::c;

    amrex::Array4<amrex::Real> const& jx_arr = jx_fab.array();
    amrex::Array4<amrex::Real> const& jy_arr = jy_fab.array();
    amrex::Array4<amrex::Real> const& jz_arr = jz_fab.array();

    constexpr int nshapes = depos_order + 1;
    int vl=svcntd();

    const amrex::ParticleReal* mx = GetPosition.m_x;
    const amrex::ParticleReal* my = GetPosition.m_y;
    const amrex::ParticleReal* mz = GetPosition.m_z;

    auto compute_shape_factor_v1 = [](double* sx, Vec xmid, svbool_t p) {
        const svuint64_t sx_index = svindex_u64(0, 2);
        intVec i_newv = svcvt_s64_f64_z(p, xmid);
        Vec j = svcvt_f64_s64_z(p, i_newv);
        Vec xint = xmid - j;
        Vec sx0 = 1.0 - xint;
        Vec sx1 = xint;
        svst1_scatter_index(p,&sx[0],sx_index,sx0);
        svst1_scatter_index(p,&sx[1],sx_index,sx1);
        return i_newv;
    };

    const int xjstride = jx_arr.jstride;
    const int xkstride = jx_arr.kstride;
    const int yjstride = jy_arr.jstride;
    const int ykstride = jy_arr.kstride;
    const int zjstride = jz_arr.jstride;
    const int zkstride = jz_arr.kstride;

    const long numcell = jx_fab.box().numPts();

    std::vector<amrex::Real> xrhocells(8*numcell, 0.0);
    std::vector<amrex::Real> yrhocells(8*numcell, 0.0);
    std::vector<amrex::Real> zrhocells(8*numcell, 0.0);
    // one roll x
    uint64_t rmx[8] = {0,1,0,1,0,1,0,1};
    uint64_t rmy[8] = {0,0,1,1,0,0,1,1};
    uint64_t rmz[8] = {0,0,0,0,1,1,1,1};
    long nx=xjstride;
    long nxy=xkstride;
    long moff[8] = {0, 1, nx, nx+1, nxy, nxy+1, nx+nxy, nx+nxy+1};
    svuint64_t rmx_v = svld1_u64(svptrue_b64(), rmx);
    svbool_t px = svcmpeq_n_u64(svptrue_b64(), rmx_v, 0);
    svuint64_t rmy_v = svld1_u64(svptrue_b64(), rmy);
    svbool_t py = svcmpeq_n_u64(svptrue_b64(), rmy_v, 0);
    svuint64_t rmz_v = svld1_u64(svptrue_b64(), rmz);
    svbool_t pz = svcmpeq_n_u64(svptrue_b64(), rmz_v, 0);

    std::vector<amrex::Real> wqx(np_to_deposit, 0.0);
    std::vector<amrex::Real> wqy(np_to_deposit, 0.0);
    std::vector<amrex::Real> wqz(np_to_deposit, 0.0);

    std::vector<amrex::Real> sx_m(np_to_deposit*nshapes, 0.);
    std::vector<amrex::Real> sy_m(np_to_deposit*nshapes, 0.);
    std::vector<amrex::Real> sz_m(np_to_deposit*nshapes, 0.);
    alignas(64) std::vector<long> xoffsets(np_to_deposit); 
    long bx = 2, by = 2, bz = 2;
    amrex::IntVect box_shape = box.length();
    long nnx = box_shape[0];
    long nny = box_shape[1];
    long nnxny = nnx*nny;
    alignas(64) std::vector<long> newbin(np_to_deposit,-1);
    

    for(long ip=0;ip<np_to_deposit;ip+=vl){

        svbool_t p=svwhilele_b64(ip,np_to_deposit);
        Vec uxp_v=Vec::Load(p,&uxp[ip]);
        Vec uyp_v=Vec::Load(p,&uyp[ip]);
        Vec uzp_v=Vec::Load(p,&uzp[ip]);
        Vec gaminv = 1._rt / (1._rt + uxp_v * uxp_v * clightsq
                                    + uyp_v * uyp_v * clightsq
                                    + uzp_v * uzp_v * clightsq).Sqrt();
        // Product of particle charges and weights
        Vec wp_v=Vec::Load(p,&wp[ip]);
        Vec wq_v = q * wp_v;

        // Current particle positions (in physical units)
        Vec xp = Vec::Load(p, &mx[ip]);
        Vec yp = Vec::Load(p, &my[ip]);
        Vec zp = Vec::Load(p, &mz[ip]);

        // Particle velocities
        Vec vx_v = uxp_v * gaminv;
        Vec vy_v = uyp_v * gaminv;
        Vec vz_v = uzp_v * gaminv;

        // if (do_ionization) wq *= ion_lev[ip];
        Vec wqx_v = wq_v * invvol * vx_v;
        Vec wqy_v = wq_v * invvol * vy_v;
        Vec wqz_v = wq_v * invvol * vz_v;

        wqx_v.Store(p, &wqx[ip]);
        wqy_v.Store(p, &wqy[ip]);
        wqz_v.Store(p, &wqz[ip]);

        Vec xmid = ((xp - xyzmin.x) + relative_time * vx_v) * dinv.x;
        Vec ymid = ((yp - xyzmin.y) + relative_time * vy_v) * dinv.y;
        Vec zmid = ((zp - xyzmin.z) + relative_time * vz_v) * dinv.z;
        intVec j_cellv = compute_shape_factor_v1( &sx_m[ip*2], xmid, p);
        intVec k_cellv = compute_shape_factor_v1( &sy_m[ip*2], ymid, p);
        intVec l_cellv = compute_shape_factor_v1( &sz_m[ip*2], zmid, p); 

        intVec xoffsets_v = j_cellv + k_cellv * xjstride + l_cellv * xkstride;

        xoffsets_v.Store(p,&xoffsets[ip]);
    }
    
    sort_3d_sme_kernal_hybrid_nosort<1>(
        np_to_deposit,xoffsets,
        xrhocells,yrhocells,zrhocells,
        sx_m,sy_m,sz_m,
        wqx,wqy,wqz);

    for (int iz = bx; iz < len.z-bx; ++iz) {
        for (int iy = bx; iy < len.y-bx; ++iy) {       
            for (int ix = bx; ix < len.x-bx; ix+=vl) {

                // vector
                svbool_t p=svwhilelt_b64(ix,(len.x-2));
                // x
                svuint64_t rho_index = svindex_u64(0ULL, 8ULL);
                long xoffset=ix + iy * xjstride + iz * xkstride;
                long jxoffset=xoffset;
                long irhox=8*xoffset;
                Vec jx0=Vec::Load(p,&jx_arr.p[jxoffset+moff[0]]);
                svfloat64_t rhox0=svld1_gather_u64index_f64(p,&xrhocells[0+irhox],rho_index);
                jx0+=rhox0;
                jx0.Store(p,&jx_arr.p[jxoffset+moff[0]]);

                Vec jx1=Vec::Load(p,&jx_arr.p[jxoffset+moff[1]]);
                svfloat64_t rhox1=svld1_gather_u64index_f64(p,&xrhocells[1+irhox],rho_index);
                jx1+=rhox1;
                jx1.Store(p,&jx_arr.p[jxoffset+moff[1]]);

                Vec jx2=Vec::Load(p,&jx_arr.p[jxoffset+moff[2]]);
                svfloat64_t rhox2=svld1_gather_u64index_f64(p,&xrhocells[2+irhox],rho_index);
                jx2+=rhox2;
                jx2.Store(p,&jx_arr.p[jxoffset+moff[2]]);

                Vec jx3=Vec::Load(p,&jx_arr.p[jxoffset+moff[3]]);
                svfloat64_t rhox3=svld1_gather_u64index_f64(p,&xrhocells[3+irhox],rho_index);
                jx3+=rhox3;
                jx3.Store(p,&jx_arr.p[jxoffset+moff[3]]);

                Vec jx4=Vec::Load(p,&jx_arr.p[jxoffset+moff[4]]);
                svfloat64_t rhox4=svld1_gather_u64index_f64(p,&xrhocells[4+irhox],rho_index);
                jx4+=rhox4;
                jx4.Store(p,&jx_arr.p[jxoffset+moff[4]]);

                Vec jx5=Vec::Load(p,&jx_arr.p[jxoffset+moff[5]]);
                svfloat64_t rhox5=svld1_gather_u64index_f64(p,&xrhocells[5+irhox],rho_index);
                jx5+=rhox5;
                jx5.Store(p,&jx_arr.p[jxoffset+moff[5]]);

                Vec jx6=Vec::Load(p,&jx_arr.p[jxoffset+moff[6]]);
                svfloat64_t rhox6=svld1_gather_u64index_f64(p,&xrhocells[6+irhox],rho_index);
                jx6+=rhox6;
                jx6.Store(p,&jx_arr.p[jxoffset+moff[6]]);

                Vec jx7=Vec::Load(p,&jx_arr.p[jxoffset+moff[7]]);
                svfloat64_t rhox7=svld1_gather_u64index_f64(p,&xrhocells[7+irhox],rho_index);
                jx7+=rhox7;
                jx7.Store(p,&jx_arr.p[jxoffset+moff[7]]);
                
                // y
                long jyoffset=xoffset;
                long irhoy=irhox;

                Vec jy0=Vec::Load(p,&jy_arr.p[jyoffset+moff[0]]);
                svfloat64_t rhoy0=svld1_gather_u64index_f64(p,&yrhocells[0+irhoy],rho_index);
                jy0+=rhoy0;
                jy0.Store(p,&jy_arr.p[jyoffset+moff[0]]);

                Vec jy1=Vec::Load(p,&jy_arr.p[jyoffset+moff[1]]);
                svfloat64_t rhoy1=svld1_gather_u64index_f64(p,&yrhocells[1+irhoy],rho_index);
                jy1+=rhoy1;
                jy1.Store(p,&jy_arr.p[jyoffset+moff[1]]);

                Vec jy2=Vec::Load(p,&jy_arr.p[jyoffset+moff[2]]);
                svfloat64_t rhoy2=svld1_gather_u64index_f64(p,&yrhocells[2+irhoy],rho_index);
                jy2+=rhoy2;
                jy2.Store(p,&jy_arr.p[jyoffset+moff[2]]);

                Vec jy3=Vec::Load(p,&jy_arr.p[jyoffset+moff[3]]);
                svfloat64_t rhoy3=svld1_gather_u64index_f64(p,&yrhocells[3+irhoy],rho_index);
                jy3+=rhoy3;
                jy3.Store(p,&jy_arr.p[jyoffset+moff[3]]);

                Vec jy4=Vec::Load(p,&jy_arr.p[jyoffset+moff[4]]);
                svfloat64_t rhoy4=svld1_gather_u64index_f64(p,&yrhocells[4+irhoy],rho_index);
                jy4+=rhoy4;
                jy4.Store(p,&jy_arr.p[jyoffset+moff[4]]);

                Vec jy5=Vec::Load(p,&jy_arr.p[jyoffset+moff[5]]);
                svfloat64_t rhoy5=svld1_gather_u64index_f64(p,&yrhocells[5+irhoy],rho_index);
                jy5+=rhoy5;
                jy5.Store(p,&jy_arr.p[jyoffset+moff[5]]);

                Vec jy6=Vec::Load(p,&jy_arr.p[jyoffset+moff[6]]);
                svfloat64_t rhoy6=svld1_gather_u64index_f64(p,&yrhocells[6+irhoy],rho_index);
                jy6+=rhoy6;
                jy6.Store(p,&jy_arr.p[jyoffset+moff[6]]);

                Vec jy7=Vec::Load(p,&jy_arr.p[jyoffset+moff[7]]);
                svfloat64_t rhoy7=svld1_gather_u64index_f64(p,&yrhocells[7+irhoy],rho_index);
                jy7+=rhoy7;
                jy7.Store(p,&jy_arr.p[jyoffset+moff[7]]);

                // z
                long jzoffset=xoffset;
                long irhoz=irhox;
                Vec jz0=Vec::Load(p,&jz_arr.p[jzoffset+moff[0]]);
                svfloat64_t rhoz0=svld1_gather_u64index_f64(p,&zrhocells[0+irhoz],rho_index);
                jz0+=rhoz0;
                jz0.Store(p,&jz_arr.p[jzoffset+moff[0]]);

                Vec jz1=Vec::Load(p,&jz_arr.p[jzoffset+moff[1]]);
                svfloat64_t rhoz1=svld1_gather_u64index_f64(p,&zrhocells[1+irhoz],rho_index);
                jz1+=rhoz1;
                jz1.Store(p,&jz_arr.p[jzoffset+moff[1]]);

                Vec jz2=Vec::Load(p,&jz_arr.p[jzoffset+moff[2]]);
                svfloat64_t rhoz2=svld1_gather_u64index_f64(p,&zrhocells[2+irhoz],rho_index);
                jz2+=rhoz2;
                jz2.Store(p,&jz_arr.p[jzoffset+moff[2]]);

                Vec jz3=Vec::Load(p,&jz_arr.p[jzoffset+moff[3]]);
                svfloat64_t rhoz3=svld1_gather_u64index_f64(p,&zrhocells[3+irhoz],rho_index);
                jz3+=rhoz3;
                jz3.Store(p,&jz_arr.p[jzoffset+moff[3]]);

                Vec jz4=Vec::Load(p,&jz_arr.p[jzoffset+moff[4]]);
                svfloat64_t rhoz4=svld1_gather_u64index_f64(p,&zrhocells[4+irhoz],rho_index);
                jz4+=rhoz4;
                jz4.Store(p,&jz_arr.p[jzoffset+moff[4]]);

                Vec jz5=Vec::Load(p,&jz_arr.p[jzoffset+moff[5]]);
                svfloat64_t rhoz5=svld1_gather_u64index_f64(p,&zrhocells[5+irhoz],rho_index);
                jz5+=rhoz5;
                jz5.Store(p,&jz_arr.p[jzoffset+moff[5]]);

                Vec jz6=Vec::Load(p,&jz_arr.p[jzoffset+moff[6]]);
                svfloat64_t rhoz6=svld1_gather_u64index_f64(p,&zrhocells[6+irhoz],rho_index);
                jz6+=rhoz6;
                jz6.Store(p,&jz_arr.p[jzoffset+moff[6]]);

                Vec jz7=Vec::Load(p,&jz_arr.p[jzoffset+moff[7]]);
                svfloat64_t rhoz7=svld1_gather_u64index_f64(p,&zrhocells[7+irhoz],rho_index);
                jz7+=rhoz7;
                jz7.Store(p,&jz_arr.p[jzoffset+moff[7]]);

            }
        }
    }
}

template <int depos_order>
__arm_new("za")  void sort_real_sme_kernal_real(
    const long numcell,
    const std::vector<int>& xpresum,
    const std::vector<int>& xbinList,
    std::vector<amrex::Real>& xrhocells,
    std::vector<amrex::Real>& yrhocells,
    std::vector<amrex::Real>& zrhocells,
    std::vector<amrex::Real>& sx_m,
    std::vector<amrex::Real>& sy_m,
    std::vector<amrex::Real>& sz_m,
    std::vector<amrex::Real>& wqx,
    std::vector<amrex::Real>& wqy,
    std::vector<amrex::Real>& wqz,
    std::vector<amrex::Real>& testsxwq
)__arm_streaming
{
   using namespace amrex::literals;
   svbool_t p_4=svwhilelt_b64(0,4);
   svbool_t p_4_8=svbic_z(svptrue_b64(),svptrue_b64(),p_4);
   svbool_t p0=svwhilelt_b64(0,2);
   svbool_t p1=svbic_z(p_4,p_4, p0);
   svbool_t p = svptrue_b64();
   MVec vzero(0);

   for(int c=1;c<=numcell;c++){
       
       int start = xpresum[c-1];
       int end = xpresum[c];
       if(start==end){
           continue;
       }
       svzero_za();
       const int end1 = end-(end - start)%2;
       for(int idx=start;idx<end1;idx+=2){
           int ip=xbinList[idx];
           int ip1=xbinList[idx+1];
            
           MVec sz_xv=svdup_n_f64(0);
           MVec sz_yv=svdup_n_f64(0);
           MVec sz_zv=svdup_n_f64(0);
           MVec sz_x1 = svld1(p0,&sz_m[ip*2]);
            MVec sz_x2 = svld1(p1,&sz_m[(ip1-1)*2]);
           MVec sz_y1 = sz_x1; 
           MVec sz_y2 = sz_x2; 
           MVec sz_z1 = sz_x1;
           MVec sz_z2 = sz_x2;
           MVec testsxwq_v=sz_x1*wqx[ip];
           MVec final_vector = svcompact_f64(p0, testsxwq_v);
           final_vector.Store(p,&testsxwq[8*(ip)]);
           sz_x1*=wqx[ip];
           sz_xv=svsel(p0,sz_x1,sz_xv);

           testsxwq_v = svdup_f64(0.0);
           testsxwq_v=sz_x2*wqx[ip1];
           final_vector = svcompact_f64(p1, testsxwq_v);
           final_vector.Store(svptrue_b64(),&testsxwq[8*(ip1)]);
           sz_x2*=wqx[ip1];
           sz_xv=svsel(p1,sz_x2,sz_xv);
           sz_y1*=wqy[ip];
           sz_yv=svsel(p0,sz_y1,sz_yv);
           sz_y2*=wqy[ip1];
           sz_yv=svsel(p1,sz_y2,sz_yv);
           sz_z1*=wqz[ip];
           sz_zv=svsel(p0,sz_z1,sz_zv);
           sz_z2*=wqz[ip1];
           sz_zv=svsel(p1,sz_z2,sz_zv);

           MVec vx1=MVec::Load(p0, &sx_m[ip*2]);
           MVec vx2=MVec::Load(p0, &sx_m[ip1*2]);

           MVec vy1=MVec::Load(p0, &sy_m[ip*2]);
           MVec vy2=MVec::Load(p0, &sy_m[ip1*2]);

           const uint64_t x_indices[8]={0,2,0,2,1,3,1,3};
           MVec vx_broadcast=svtbl(svzip1(vx1,vx2),svld1_u64(svptrue_b64(),x_indices));
           const uint64_t y_indices[8]={0,0,2,2,1,1,3,3};
           MVec vy_broadcast=svtbl(svzip1(vy1,vy2),svld1_u64(svptrue_b64(),y_indices));

           MVec sxsy_v=vx_broadcast*vy_broadcast;
           
           svmopa_za64_m(0, p_4, svptrue_b64(), sz_xv, sxsy_v);
           svmopa_za64_m(1, p_4, svptrue_b64(), sz_yv, sxsy_v);
           svmopa_za64_m(2, p_4, svptrue_b64(), sz_zv, sxsy_v);

       }

       if((end - start)%2==1){
           int ip=xbinList[end1];   
           MVec sz_v=svdup_n_f64(0);
           MVec sz_x1 = svld1(p0,&sz_m[ip*2]);

           MVec testsxwq_v = svdup_f64(0.0);
           testsxwq_v=sz_x1*wqx[ip];
           testsxwq_v.Store(p,&testsxwq[8*(ip)]);

           sz_x1*=wqx[ip];
           sz_v=svsel(p0,sz_x1,sz_v);
            MVec vx1=MVec::Load(p0, &sx_m[ip*2]);
            MVec vy1=MVec::Load(p0, &sy_m[ip*2]);
            MVec sxsy_v=svsplice(p0,vx1,vx1)*svzip1(vy1,vy1);
            svmopa_za64_m(0, p0, p_4, sz_v, sxsy_v);
        }

        MVec wwx = MVec::Load(p,&xrhocells[8*c]);
        MVec wwy = MVec::Load(p,&yrhocells[8*c]);
        MVec wwz = MVec::Load(p,&zrhocells[8*c]);
        MVec wxsxsysz0 = svread_hor_za64_m(vzero,p_4,0,0);
        MVec wxsxsysz1 = svread_hor_za64_m(vzero,p_4,0,1);
        MVec s=svsplice(p_4,wxsxsysz0,wxsxsysz1);
        wwx+=s;
        wxsxsysz0 = svread_hor_za64_m(vzero,p_4_8,0,2);
        wxsxsysz1 = svread_hor_za64_m(vzero,p_4_8,0,3);
        MVec B=svext(wxsxsysz1,wxsxsysz1,4);
        s=svext(wxsxsysz0,B,4);
        wwx+=s;
        wwx.Store(p,&xrhocells[8*c]);
        wwy.Store(p,&yrhocells[8*c]);
        wwz.Store(p,&zrhocells[8*c]); 
    }


}

template <int depos_order>
void doDepositionShapeN_sve_sort_real_sme (const GetParticlePosition<PIdx>& GetPosition,
                        const amrex::ParticleReal * const wp,
                        const amrex::ParticleReal * const uxp,
                        const amrex::ParticleReal * const uyp,
                        const amrex::ParticleReal * const uzp,
                        const int* ion_lev,
                        amrex::FArrayBox& jx_fab,
                        amrex::FArrayBox& jy_fab,
                        amrex::FArrayBox& jz_fab,
                        long np_to_deposit,
                        amrex::Real relative_time,
                        const amrex::XDim3 & dinv,
                        const amrex::XDim3 & xyzmin,
                        amrex::Dim3 lo,
                        amrex::Dim3 hi,
                        amrex::Dim3 len,
                        amrex::Real q,
                        std::vector<int>& test,
                        std::vector<double>& test0,std::vector<double>& test1,
                        ParticleTileType& ptile, const amrex::Box& box,
                        std::vector<amrex::Real>& test_rhocells,
                        std::vector<amrex::Real>& test_sxwq,
                        [[maybe_unused]]int n_rz_azimuthal_modes)
{
    using namespace amrex::literals;

    const bool do_ionization = ion_lev;

    const amrex::Real invvol = dinv.x*dinv.y*dinv.z; 

    const amrex::Real clightsq = 1.0_rt/PhysConst::c/PhysConst::c;

    amrex::Array4<amrex::Real> const& jx_arr = jx_fab.array();
    amrex::Array4<amrex::Real> const& jy_arr = jy_fab.array();
    amrex::Array4<amrex::Real> const& jz_arr = jz_fab.array();

    constexpr int nshapes = depos_order + 1;
    // constexpr int nshapes = 2;
    int vl=svcntd();

    const amrex::ParticleReal* mx = GetPosition.m_x;
    const amrex::ParticleReal* my = GetPosition.m_y;
    const amrex::ParticleReal* mz = GetPosition.m_z;

    auto compute_shape_factor_v1 = [](double* sx, Vec xmid, svbool_t p) {
        const svuint64_t sx_index = svindex_u64(0, 2);
        intVec i_newv = svcvt_s64_f64_z(p, xmid);
        Vec j = svcvt_f64_s64_z(p, i_newv);
        Vec xint = xmid - j;
        Vec sx0 = 1.0 - xint;
        Vec sx1 = xint;
        svst1_scatter_index(p,&sx[0],sx_index,sx0);
        svst1_scatter_index(p,&sx[1],sx_index,sx1);
        return i_newv;
    };

    const int xjstride = jx_arr.jstride;
    const int xkstride = jx_arr.kstride;
    const int yjstride = jy_arr.jstride;
    const int ykstride = jy_arr.kstride;
    const int zjstride = jz_arr.jstride;
    const int zkstride = jz_arr.kstride;

    const long numcell = jx_fab.box().numPts();
    const long xnumcell = jx_fab.box().numPts();

    std::vector<amrex::Real> xrhocells(8*numcell, 0.0);
    std::vector<amrex::Real> yrhocells(8*numcell, 0.0);
    std::vector<amrex::Real> zrhocells(8*numcell, 0.0);
    // one roll x
    uint64_t rmx[8] = {0,1,0,1,0,1,0,1};
    uint64_t rmy[8] = {0,0,1,1,0,0,1,1};
    uint64_t rmz[8] = {0,0,0,0,1,1,1,1};
    long nx=xjstride;
    long nxy=xkstride;
    long moff[8] = {0, 1, nx, nx+1, nxy, nxy+1, nx+nxy, nx+nxy+1};
    svuint64_t rmx_v = svld1_u64(svptrue_b64(), rmx);
    svbool_t px = svcmpeq_n_u64(svptrue_b64(), rmx_v, 0);
    svuint64_t rmy_v = svld1_u64(svptrue_b64(), rmy);
    svbool_t py = svcmpeq_n_u64(svptrue_b64(), rmy_v, 0);
    svuint64_t rmz_v = svld1_u64(svptrue_b64(), rmz);
    svbool_t pz = svcmpeq_n_u64(svptrue_b64(), rmz_v, 0);

    std::vector<amrex::Real> wqx(np_to_deposit, 0.0);
    std::vector<amrex::Real> wqy(np_to_deposit, 0.0);
    std::vector<amrex::Real> wqz(np_to_deposit, 0.0);

    std::vector<amrex::Real> sx_m(np_to_deposit*nshapes, 0.);
    std::vector<amrex::Real> sy_m(np_to_deposit*nshapes, 0.);
    std::vector<amrex::Real> sz_m(np_to_deposit*nshapes, 0.);
    alignas(64) long xoffsets[np_to_deposit]; 
    std::vector<long> xcount(numcell,0);
    

    for(long ip=0;ip<np_to_deposit;ip+=vl){

        svbool_t p=svwhilele_b64(ip,np_to_deposit);
        // Inverse of Lorentz factor gamma
        Vec uxp_v=Vec::Load(p,&uxp[ip]);
        Vec uyp_v=Vec::Load(p,&uyp[ip]);
        Vec uzp_v=Vec::Load(p,&uzp[ip]);
        Vec gaminv = 1._rt / (1._rt + uxp_v * uxp_v * clightsq
                                    + uyp_v * uyp_v * clightsq
                                    + uzp_v * uzp_v * clightsq).Sqrt();
        // Product of particle charges and weights
        Vec wp_v=Vec::Load(p,&wp[ip]);
        Vec wq_v = q * wp_v;

        // Current particle positions (in physical units)
        Vec xp = Vec::Load(p, &mx[ip]);
        Vec yp = Vec::Load(p, &my[ip]);
        Vec zp = Vec::Load(p, &mz[ip]);

        // Particle velocities
        Vec vx_v = uxp_v * gaminv;
        Vec vy_v = uyp_v * gaminv;
        Vec vz_v = uzp_v * gaminv;

        // if (do_ionization) wq *= ion_lev[ip];
        Vec wqx_v = wq_v * invvol * vx_v;
        Vec wqy_v = wq_v * invvol * vy_v;
        Vec wqz_v = wq_v * invvol * vz_v;

        wqx_v.Store(p, &wqx[ip]);
        wqy_v.Store(p, &wqy[ip]);
        wqz_v.Store(p, &wqz[ip]);

        Vec xmid = ((xp - xyzmin.x) + relative_time * vx_v) * dinv.x;
        Vec ymid = ((yp - xyzmin.y) + relative_time * vy_v) * dinv.y;
        Vec zmid = ((zp - xyzmin.z) + relative_time * vz_v) * dinv.z;
        intVec j_cellv = compute_shape_factor_v1( &sx_m[ip*2], xmid, p);
        intVec k_cellv = compute_shape_factor_v1( &sy_m[ip*2], ymid, p);
        intVec l_cellv = compute_shape_factor_v1( &sz_m[ip*2], zmid, p); 

        intVec xoffsets_v = j_cellv + k_cellv * xjstride + l_cellv * xkstride;

        xoffsets_v.Store(p,&xoffsets[ip]);
    }
    
    for(long ip=0;ip<np_to_deposit;ip++){
        xcount[xoffsets[ip]]++;
    }

    std::vector<int> xpresum(numcell+1,0);
    for(int c=1;c<=numcell;c++){
        xpresum[c]=xpresum[c-1]+xcount[c-1];
    }

    std::vector<int> xbinList(np_to_deposit,-1);
    for(int ip=0;ip<np_to_deposit;ip++){
        long xoffset = xoffsets[ip];
        xbinList[xpresum[xoffset]]=ip;
        xpresum[xoffset]++;
    }

    std::vector<amrex::Real> testsxwq(8*np_to_deposit, 0.0);
    sort_real_sme_kernal_real<1>(numcell,xpresum,xbinList,
                            xrhocells,yrhocells,zrhocells,
                            sx_m,sy_m,sz_m,
                            wqx,wqy,wqz,testsxwq);

    int errcount=0;
    for (int iz = 0; iz <= len.z; ++iz) {
        for (int iy = 0; iy <= len.y; ++iy) {             
            for (int ix = 0; ix <= len.x; ix+=vl) {

                // vector
                svbool_t p=svwhilele_b64(ix,len.x);
                // x
                svuint64_t rho_index = svindex_u64(0ULL, 8ULL);
                long xoffset=ix + iy * xjstride + iz * xkstride;
                long jxoffset=xoffset;
                long irhox=8*xoffset;
                Vec jx0=Vec::Load(p,&jx_arr.p[jxoffset+moff[0]]);
                svfloat64_t rhox0=svld1_gather_u64index_f64(p,&xrhocells[0+irhox],rho_index);
                jx0+=rhox0;
                jx0.Store(p,&jx_arr.p[jxoffset+moff[0]]);

                Vec jx1=Vec::Load(p,&jx_arr.p[jxoffset+moff[1]]);
                svfloat64_t rhox1=svld1_gather_u64index_f64(p,&xrhocells[1+irhox],rho_index);
                jx1+=rhox1;
                jx1.Store(p,&jx_arr.p[jxoffset+moff[1]]);

                Vec jx2=Vec::Load(p,&jx_arr.p[jxoffset+moff[2]]);
                svfloat64_t rhox2=svld1_gather_u64index_f64(p,&xrhocells[2+irhox],rho_index);
                jx2+=rhox2;
                jx2.Store(p,&jx_arr.p[jxoffset+moff[2]]);

                Vec jx3=Vec::Load(p,&jx_arr.p[jxoffset+moff[3]]);
                svfloat64_t rhox3=svld1_gather_u64index_f64(p,&xrhocells[3+irhox],rho_index);
                jx3+=rhox3;
                jx3.Store(p,&jx_arr.p[jxoffset+moff[3]]);

                Vec jx4=Vec::Load(p,&jx_arr.p[jxoffset+moff[4]]);
                svfloat64_t rhox4=svld1_gather_u64index_f64(p,&xrhocells[4+irhox],rho_index);
                jx4+=rhox4;
                jx4.Store(p,&jx_arr.p[jxoffset+moff[4]]);

                Vec jx5=Vec::Load(p,&jx_arr.p[jxoffset+moff[5]]);
                svfloat64_t rhox5=svld1_gather_u64index_f64(p,&xrhocells[5+irhox],rho_index);
                jx5+=rhox5;
                jx5.Store(p,&jx_arr.p[jxoffset+moff[5]]);

                Vec jx6=Vec::Load(p,&jx_arr.p[jxoffset+moff[6]]);
                svfloat64_t rhox6=svld1_gather_u64index_f64(p,&xrhocells[6+irhox],rho_index);
                jx6+=rhox6;
                jx6.Store(p,&jx_arr.p[jxoffset+moff[6]]);

                Vec jx7=Vec::Load(p,&jx_arr.p[jxoffset+moff[7]]);
                svfloat64_t rhox7=svld1_gather_u64index_f64(p,&xrhocells[7+irhox],rho_index);
                jx7+=rhox7;
                jx7.Store(p,&jx_arr.p[jxoffset+moff[7]]);
                
                // y
                long yoffset = ix + iy * yjstride + iz * ykstride;
                long jyoffset=yoffset;
                long irhoy=8*yoffset;
                Vec jy0=Vec::Load(p,&jy_arr.p[jyoffset+moff[0]]);
                svfloat64_t rhoy0=svld1_gather_u64index_f64(p,&yrhocells[0+irhoy],rho_index);
                jy0+=rhoy0;
                jy0.Store(p,&jy_arr.p[jyoffset+moff[0]]);

                Vec jy1=Vec::Load(p,&jy_arr.p[jyoffset+moff[1]]);
                svfloat64_t rhoy1=svld1_gather_u64index_f64(p,&yrhocells[1+irhoy],rho_index);
                jy1+=rhoy1;
                jy1.Store(p,&jy_arr.p[jyoffset+moff[1]]);

                Vec jy2=Vec::Load(p,&jy_arr.p[jyoffset+moff[2]]);
                svfloat64_t rhoy2=svld1_gather_u64index_f64(p,&yrhocells[2+irhoy],rho_index);
                jy2+=rhoy2;
                jy2.Store(p,&jy_arr.p[jyoffset+moff[2]]);

                Vec jy3=Vec::Load(p,&jy_arr.p[jyoffset+moff[3]]);
                svfloat64_t rhoy3=svld1_gather_u64index_f64(p,&yrhocells[3+irhoy],rho_index);
                jy3+=rhoy3;
                jy3.Store(p,&jy_arr.p[jyoffset+moff[3]]);

                Vec jy4=Vec::Load(p,&jy_arr.p[jyoffset+moff[4]]);
                svfloat64_t rhoy4=svld1_gather_u64index_f64(p,&yrhocells[4+irhoy],rho_index);
                jy4+=rhoy4;
                jy4.Store(p,&jy_arr.p[jyoffset+moff[4]]);

                Vec jy5=Vec::Load(p,&jy_arr.p[jyoffset+moff[5]]);
                svfloat64_t rhoy5=svld1_gather_u64index_f64(p,&yrhocells[5+irhoy],rho_index);
                jy5+=rhoy5;
                jy5.Store(p,&jy_arr.p[jyoffset+moff[5]]);

                Vec jy6=Vec::Load(p,&jy_arr.p[jyoffset+moff[6]]);
                svfloat64_t rhoy6=svld1_gather_u64index_f64(p,&yrhocells[6+irhoy],rho_index);
                jy6+=rhoy6;
                jy6.Store(p,&jy_arr.p[jyoffset+moff[6]]);

                Vec jy7=Vec::Load(p,&jy_arr.p[jyoffset+moff[7]]);
                svfloat64_t rhoy7=svld1_gather_u64index_f64(p,&yrhocells[7+irhoy],rho_index);
                jy7+=rhoy7;
                jy7.Store(p,&jy_arr.p[jyoffset+moff[7]]);

                // z
                long zoffset = ix + iy * zjstride + iz * zkstride;
                long jzoffset=zoffset;
                long irhoz=8*zoffset;
                Vec jz0=Vec::Load(p,&jz_arr.p[jzoffset+moff[0]]);
                svfloat64_t rhoz0=svld1_gather_u64index_f64(p,&zrhocells[0+irhoz],rho_index);
                jz0+=rhoz0;
                jz0.Store(p,&jz_arr.p[jzoffset+moff[0]]);

                Vec jz1=Vec::Load(p,&jz_arr.p[jzoffset+moff[1]]);
                svfloat64_t rhoz1=svld1_gather_u64index_f64(p,&zrhocells[1+irhoz],rho_index);
                jz1+=rhoz1;
                jz1.Store(p,&jz_arr.p[jzoffset+moff[1]]);

                Vec jz2=Vec::Load(p,&jz_arr.p[jzoffset+moff[2]]);
                svfloat64_t rhoz2=svld1_gather_u64index_f64(p,&zrhocells[2+irhoz],rho_index);
                jz2+=rhoz2;
                jz2.Store(p,&jz_arr.p[jzoffset+moff[2]]);

                Vec jz3=Vec::Load(p,&jz_arr.p[jzoffset+moff[3]]);
                svfloat64_t rhoz3=svld1_gather_u64index_f64(p,&zrhocells[3+irhoz],rho_index);
                jz3+=rhoz3;
                jz3.Store(p,&jz_arr.p[jzoffset+moff[3]]);

                Vec jz4=Vec::Load(p,&jz_arr.p[jzoffset+moff[4]]);
                svfloat64_t rhoz4=svld1_gather_u64index_f64(p,&zrhocells[4+irhoz],rho_index);
                jz4+=rhoz4;
                jz4.Store(p,&jz_arr.p[jzoffset+moff[4]]);

                Vec jz5=Vec::Load(p,&jz_arr.p[jzoffset+moff[5]]);
                svfloat64_t rhoz5=svld1_gather_u64index_f64(p,&zrhocells[5+irhoz],rho_index);
                jz5+=rhoz5;
                jz5.Store(p,&jz_arr.p[jzoffset+moff[5]]);

                Vec jz6=Vec::Load(p,&jz_arr.p[jzoffset+moff[6]]);
                svfloat64_t rhoz6=svld1_gather_u64index_f64(p,&zrhocells[6+irhoz],rho_index);
                jz6+=rhoz6;
                jz6.Store(p,&jz_arr.p[jzoffset+moff[6]]);

                Vec jz7=Vec::Load(p,&jz_arr.p[jzoffset+moff[7]]);
                svfloat64_t rhoz7=svld1_gather_u64index_f64(p,&zrhocells[7+irhoz],rho_index);
                jz7+=rhoz7;
                jz7.Store(p,&jz_arr.p[jzoffset+moff[7]]);

            }
        }
    }
}
 
template <int depos_order>
 void doDepositionShapeN_org_sp_GPMA(const GetParticlePosition<PIdx> &GetPosition,
                                           const amrex::ParticleReal *const wp,
                                           const amrex::ParticleReal *const uxp,
                                           const amrex::ParticleReal *const uyp,
                                           const amrex::ParticleReal *const uzp,
                                           const int *ion_lev,
                                           amrex::FArrayBox &jx_fab,
                                           amrex::FArrayBox &jy_fab,
                                           amrex::FArrayBox &jz_fab,
                                           long np_to_deposit,
                                           amrex::Real relative_time,
                                           const amrex::XDim3 &dinv,
                                           const amrex::XDim3 &xyzmin,
                                           amrex::Dim3 lo,
                                           amrex::Dim3 hi,
                                           amrex::Dim3 len,
                                           amrex::Real q,
                                           WarpXParticleContainer::ParticleTileType &ptile, const amrex::Box &box,
                                           [[maybe_unused]] int n_rz_azimuthal_modes,
                                           std::vector<uint64_t>& inner_timer
                                        )
 {
    using namespace amrex::literals;
    #ifdef BREAKDWON
    using namespace std::chrono;
    uint64_t total_time[3]={0};
    uint64_t precompute=0;
    uint64_t cal_time=0;
    uint64_t sort_time=0;
        uint64_t presort=0;
        uint64_t insert=0;
        uint64_t borrow=0;
        uint64_t rebuildtime=0;
    uint64_t reduce_time=0;
    uint64_t svetime[3]={0};
    uint64_t svetime1[3]={0};
 
    total_time[0]=rdtscv();
    #endif
    amrex::Array4<amrex::Real> const &jx_arr = jx_fab.array();
    amrex::Array4<amrex::Real> const &jy_arr = jy_fab.array();
    amrex::Array4<amrex::Real> const &jz_arr = jz_fab.array();

    const int xjstride = jx_arr.jstride;
    const int xkstride = jx_arr.kstride;
    const int yjstride = jy_arr.jstride;
    const int ykstride = jy_arr.kstride;
    const int zjstride = jz_arr.jstride;
    const int zkstride = jz_arr.kstride;

    long nx = xjstride;
    long nxy = xkstride;
    long moff[8] = {0, 1, nx, nx + 1, nxy, nxy + 1, nx + nxy, nx + nxy + 1};

    const bool do_ionization = ion_lev;

    const amrex::Real invvol = dinv.x * dinv.y * dinv.z;

    const amrex::Real clightsq = 1.0_rt / PhysConst::c / PhysConst::c;

    const long numcell = jx_fab.box().numPts();

    amrex::IntVect const jx_type = jx_fab.box().type();
    amrex::IntVect const jy_type = jy_fab.box().type();
    amrex::IntVect const jz_type = jz_fab.box().type();

    std::vector<amrex::Real> wqx(np_to_deposit, 0.0_rt), wqy(np_to_deposit, 0.0_rt), wqz(np_to_deposit, 0.0_rt);
    
    std::vector<amrex::Real> sx_m(2 * np_to_deposit, 0);
    std::vector<amrex::Real> sy_m(2 * np_to_deposit, 0);
    std::vector<amrex::Real> sz_m(2 * np_to_deposit, 0);
    alignas(64) std::vector<int> newbin(np_to_deposit,-1);
    long nnx = len.x;
    long nny = len.y;
    long nnxny = nnx*nny;
    #ifdef BREAKDWON
    svetime[0]=rdtscv();
    #endif
    #pragma omp simd
    for (int ip = 0; ip < np_to_deposit; ip++)
    {
        amrex::ParticleReal xp, yp, zp;
        GetPosition(ip, xp, yp, zp);

        // --- Get particle quantities
        const amrex::Real gaminv = 1.0_rt / std::sqrt(1.0_rt + uxp[ip] * uxp[ip] * clightsq + uyp[ip] * uyp[ip] * clightsq + uzp[ip] * uzp[ip] * clightsq);
        const amrex::Real vx = uxp[ip] * gaminv;
        const amrex::Real vy = uyp[ip] * gaminv;
        const amrex::Real vz = uzp[ip] * gaminv;

        amrex::Real wq = q * wp[ip];
        if (do_ionization)
        {
            wq *= ion_lev[ip];
        }

        constexpr int NODE = amrex::IndexType::NODE;
        constexpr int CELL = amrex::IndexType::CELL;

        // wqx, wqy wqz are particle current in each direction
        wqx[ip] = wq * invvol * vx;
        wqy[ip] = wq * invvol * vy;
        wqz[ip] = wq * invvol * vz;

        // --- Compute shape factors
        Compute_shape_factor<depos_order> const compute_shape_factor;

        // const auto j = static_cast<int>(xmid);
        // x direction
        // Get particle position after 1/2 push back in position
        // Keep these double to avoid bug in single precision

        const double xmid = ((xp - xyzmin.x) + relative_time * vx) * dinv.x;

        double sx_node[depos_order + 1] = {0.};
        double sx_cell[depos_order + 1] = {0.};
        int j_node = 0;
        int j_cell = 0;
        if (jx_type[0] == NODE || jy_type[0] == NODE || jz_type[0] == NODE)
        {
            j_node = compute_shape_factor(sx_node, xmid);
        }
        if (jx_type[0] == CELL || jy_type[0] == CELL || jz_type[0] == CELL)
        {
            j_cell = compute_shape_factor(sx_cell, xmid - 0.5);
        }

        for (int ix = 0; ix <= depos_order; ix++)
        {
            sx_m[ip * 2 + ix] = ((jx_type[0] == NODE) ? amrex::Real(sx_node[ix]) : amrex::Real(sx_cell[ix]));
        }

        int jm = ((jz_type[0] == NODE) ? j_node : j_cell);

        // y direction
        // Keep these double to avoid bug in single precision
        const double ymid = ((yp - xyzmin.y) + relative_time * vy) * dinv.y;
        double sy_node[depos_order + 1] = {0.};
        double sy_cell[depos_order + 1] = {0.};
        int k_node = 0;
        int k_cell = 0;
        if (jx_type[1] == NODE || jy_type[1] == NODE || jz_type[1] == NODE)
        {
            k_node = compute_shape_factor(sy_node, ymid);
        }
        if (jx_type[1] == CELL || jy_type[1] == CELL || jz_type[1] == CELL)
        {
            k_cell = compute_shape_factor(sy_cell, ymid - 0.5);
        }
        for (int iy = 0; iy <= depos_order; iy++)
        {
            sy_m[ip * 2 + iy] = ((jx_type[1] == NODE) ? amrex::Real(sy_node[iy]) : amrex::Real(sy_cell[iy]));
        }
        int km = ((jx_type[1] == NODE) ? k_node : k_cell);

        // z direction
        // Keep these double to avoid bug in single precision
        constexpr int zdir = WARPX_ZINDEX;
        const double zmid = ((zp - xyzmin.z) + relative_time * vz) * dinv.z;
        double sz_node[depos_order + 1] = {0.};
        double sz_cell[depos_order + 1] = {0.};
        int l_node = 0;
        int l_cell = 0;
        if (jx_type[zdir] == NODE || jy_type[zdir] == NODE || jz_type[zdir] == NODE)
        {
            l_node = compute_shape_factor(sz_node, zmid);
        }
        if (jx_type[zdir] == CELL || jy_type[zdir] == CELL || jz_type[zdir] == CELL)
        {
            l_cell = compute_shape_factor(sz_cell, zmid - 0.5);
        }
        for (int iz = 0; iz <= depos_order; iz++)
        {
            sz_m[ip * 2 + iz] = ((jx_type[zdir] == NODE) ? amrex::Real(sz_node[iz]) : amrex::Real(sz_cell[iz]));
        }
        int lm = ((jz_type[zdir] == NODE) ? l_node : l_cell);
        newbin[ip]=jm+km*nnx+lm*nnxny;
    }
    #ifdef BREAKDWON
    svetime[1]=rdtscv();
    precompute+=(svetime[1]-svetime[0]); 
    svetime[0]=rdtscv();
    #endif

    increment_sort_particles(np_to_deposit, ptile,newbin, numcell);

    #ifdef BREAKDWON
    svetime[1]=rdtscv();
    sort_time+=(svetime[1]-svetime[0]); 
    #endif
    
    double xtemp[8]={0.0};
    double ytemp[8]={0.0};
    double ztemp[8]={0.0};


    #ifdef BREAKDWON
    svetime[0]=rdtscv();
    #endif

    #pragma omp unroll 
    for (int iz = 0; iz < len.z; ++iz) {
        for (int iy = 0; iy < len.y; ++iy) {       
            for (int ix = 0; ix < len.x; ix++) {
                int old_bin=(ix) + (iy) * nnx + (iz) * nnxny;
                int binlength = ptile.m_bin_lengths[old_bin];
                int binoffset = ptile.m_bin_offsets[old_bin];
        
                #pragma omp simd
                for(int i=0;i<binlength;i++){
                    int ip = ptile.m_local_index[binoffset+i];
                    #pragma omp simd
                    for (int iz=0; iz<=depos_order; iz++){
                        #pragma omp simd
                        for (int iy=0; iy<=depos_order; iy++){
                            #pragma omp simd
                            for (int ix=0; ix<=depos_order; ix++){
                                int idx=ix+iy*2+iz*4;
                                double sxsysz=sx_m[ix+2*ip]*sy_m[iy+2*ip]*sz_m[iz+2*ip];
                                double jx=sxsysz*wqx[ip];
                                xtemp[idx]+=jx;
                                double jy=sxsysz*wqy[ip];
                                ytemp[idx]+=jy;
                                double jz=sxsysz*wqz[ip];
                                ztemp[idx]+=jy;
                            }
                        }
                    }
                }
                #pragma omp simd
                for (int iz=0; iz<=depos_order; iz++){
                    #pragma omp simd
                    for (int iy=0; iy<=depos_order; iy++){
                        #pragma omp simd
                        for (int ix=0; ix<=depos_order; ix++){
                            int jxoffset=ix + iy * xjstride + iz * xkstride;
                            int xidx=ix+iy*nnx+iz*nnxny;
                            int idx=ix+iy*2+iz*4;
                            jx_arr.p[jxoffset+xidx]+=xtemp[idx];
                            jy_arr.p[jxoffset+xidx]+=ytemp[idx];
                            jz_arr.p[jxoffset+xidx]+=ztemp[idx];
                        }
                    }
                }
            }
        }
    }
    
    #ifdef BREAKDWON
    svetime[1]=rdtscv();
    cal_time+=(svetime[1]-svetime[0]); 

    total_time[1]=rdtscv();
    total_time[2]+=(total_time[1]-total_time[0]);
    inner_timer[0]=cal_time+precompute+sort_time;
    inner_timer[1]=precompute;
    inner_timer[2]=cal_time;
    inner_timer[3]=sort_time;
    #endif
 
}

template <int depos_order>
void doDepositionShapeN_org_sp_rhocell(const GetParticlePosition<PIdx> &GetPosition,
                                        const amrex::ParticleReal *const wp,
                                        const amrex::ParticleReal *const uxp,
                                        const amrex::ParticleReal *const uyp,
                                        const amrex::ParticleReal *const uzp,
                                        const int *ion_lev,
                                        amrex::FArrayBox &jx_fab,
                                        amrex::FArrayBox &jy_fab,
                                        amrex::FArrayBox &jz_fab,
                                        long np_to_deposit,
                                        amrex::Real relative_time,
                                        const amrex::XDim3 &dinv,
                                        const amrex::XDim3 &xyzmin,
                                        amrex::Dim3 lo,
                                        amrex::Dim3 hi,
                                        amrex::Dim3 len,
                                        amrex::Real q,
                                        WarpXParticleContainer::ParticleTileType &ptile, const amrex::Box &box,
                                        [[maybe_unused]] int n_rz_azimuthal_modes,
                                        std::vector<uint64_t>& inner_timer
                                    )
{
    using namespace amrex::literals;
    #ifdef BREAKDWON
    using namespace std::chrono;
    uint64_t total_time[3]={0};
    uint64_t precompute=0;
    uint64_t cal_time=0;
    uint64_t sort_time=0;
        uint64_t presort=0;
        uint64_t insert=0;
        uint64_t borrow=0;
        uint64_t rebuildtime=0;
    uint64_t reduce_time=0;
    uint64_t svetime[3]={0};
    uint64_t svetime1[3]={0};
 
    total_time[0]=rdtscv();
    #endif
    amrex::Array4<amrex::Real> const &jx_arr = jx_fab.array();
    amrex::Array4<amrex::Real> const &jy_arr = jy_fab.array();
    amrex::Array4<amrex::Real> const &jz_arr = jz_fab.array();

    const int xjstride = jx_arr.jstride;
    const int xkstride = jx_arr.kstride;
    const int yjstride = jy_arr.jstride;
    const int ykstride = jy_arr.kstride;
    const int zjstride = jz_arr.jstride;
    const int zkstride = jz_arr.kstride;

    long nx = xjstride;
    long nxy = xkstride;

    const bool do_ionization = ion_lev;

    const amrex::Real invvol = dinv.x * dinv.y * dinv.z;

    const amrex::Real clightsq = 1.0_rt / PhysConst::c / PhysConst::c;

    const long numcell = jx_fab.box().numPts();

    amrex::IntVect const jx_type = jx_fab.box().type();
    amrex::IntVect const jy_type = jy_fab.box().type();
    amrex::IntVect const jz_type = jz_fab.box().type();

    std::vector<amrex::Real> wqx(np_to_deposit, 0.0_rt), wqy(np_to_deposit, 0.0_rt), wqz(np_to_deposit, 0.0_rt);
    
    std::vector<amrex::Real> sx_m(2 * np_to_deposit, 0);
    std::vector<amrex::Real> sy_m(2 * np_to_deposit, 0);
    std::vector<amrex::Real> sz_m(2 * np_to_deposit, 0);

    long nnx = len.x;
    long nny = len.y;
    long nnxny = nnx*nny;
    std::vector<int> newbin(np_to_deposit,0);

    #ifdef BREAKDWON
    svetime[0]=rdtscv();
    #endif

    #pragma omp simd
    for (int ip = 0; ip < np_to_deposit; ip++)
    {
        amrex::ParticleReal xp, yp, zp;
        GetPosition(ip, xp, yp, zp);

        // --- Get particle quantities
        const amrex::Real gaminv = 1.0_rt / std::sqrt(1.0_rt + uxp[ip] * uxp[ip] * clightsq + uyp[ip] * uyp[ip] * clightsq + uzp[ip] * uzp[ip] * clightsq);
        const amrex::Real vx = uxp[ip] * gaminv;
        const amrex::Real vy = uyp[ip] * gaminv;
        const amrex::Real vz = uzp[ip] * gaminv;

        amrex::Real wq = q * wp[ip];
        if (do_ionization)
        {
            wq *= ion_lev[ip];
        }

        constexpr int NODE = amrex::IndexType::NODE;
        constexpr int CELL = amrex::IndexType::CELL;

        // wqx, wqy wqz are particle current in each direction
        wqx[ip] = wq * invvol * vx;
        wqy[ip] = wq * invvol * vy;
        wqz[ip] = wq * invvol * vz;

        // --- Compute shape factors
        Compute_shape_factor<depos_order> const compute_shape_factor;

        // const auto j = static_cast<int>(xmid);
        // x direction
        // Get particle position after 1/2 push back in position
        // Keep these double to avoid bug in single precision

        const double xmid = ((xp - xyzmin.x) + relative_time * vx) * dinv.x;

        double sx_node[depos_order + 1] = {0.};
        double sx_cell[depos_order + 1] = {0.};
        int j_node = 0;
        int j_cell = 0;
        if (jx_type[0] == NODE || jy_type[0] == NODE || jz_type[0] == NODE)
        {
            j_node = compute_shape_factor(sx_node, xmid);
        }
        if (jx_type[0] == CELL || jy_type[0] == CELL || jz_type[0] == CELL)
        {
            j_cell = compute_shape_factor(sx_cell, xmid - 0.5);
        }

        for (int ix = 0; ix <= depos_order; ix++)
        {
            sx_m[ip * 2 + ix] = ((jx_type[0] == NODE) ? amrex::Real(sx_node[ix]) : amrex::Real(sx_cell[ix]));
        }

        int jm = ((jz_type[0] == NODE) ? j_node : j_cell);

        // y direction
        // Keep these double to avoid bug in single precision
        const double ymid = ((yp - xyzmin.y) + relative_time * vy) * dinv.y;
        double sy_node[depos_order + 1] = {0.};
        double sy_cell[depos_order + 1] = {0.};
        int k_node = 0;
        int k_cell = 0;
        if (jx_type[1] == NODE || jy_type[1] == NODE || jz_type[1] == NODE)
        {
            k_node = compute_shape_factor(sy_node, ymid);
        }
        if (jx_type[1] == CELL || jy_type[1] == CELL || jz_type[1] == CELL)
        {
            k_cell = compute_shape_factor(sy_cell, ymid - 0.5);
        }
        for (int iy = 0; iy <= depos_order; iy++)
        {
            sy_m[ip * 2 + iy] = ((jx_type[1] == NODE) ? amrex::Real(sy_node[iy]) : amrex::Real(sy_cell[iy]));
        }
        int km = ((jx_type[1] == NODE) ? k_node : k_cell);

        // z direction
        // Keep these double to avoid bug in single precision
        constexpr int zdir = WARPX_ZINDEX;
        const double zmid = ((zp - xyzmin.z) + relative_time * vz) * dinv.z;
        double sz_node[depos_order + 1] = {0.};
        double sz_cell[depos_order + 1] = {0.};
        int l_node = 0;
        int l_cell = 0;
        if (jx_type[zdir] == NODE || jy_type[zdir] == NODE || jz_type[zdir] == NODE)
        {
            l_node = compute_shape_factor(sz_node, zmid);
        }
        if (jx_type[zdir] == CELL || jy_type[zdir] == CELL || jz_type[zdir] == CELL)
        {
            l_cell = compute_shape_factor(sz_cell, zmid - 0.5);
        }
        for (int iz = 0; iz <= depos_order; iz++)
        {
            sz_m[ip * 2 + iz] = ((jx_type[zdir] == NODE) ? amrex::Real(sz_node[iz]) : amrex::Real(sz_cell[iz]));
        }
        int lm = ((jz_type[zdir] == NODE) ? l_node : l_cell);
        newbin[ip]=jm+km*nnx+lm*nnxny;
    }

    #ifdef BREAKDWON
    svetime[1]=rdtscv();
    precompute+=(svetime[1]-svetime[0]); 
    #endif
    amrex::IntVect box_shape = box.length();

    int nx_box = box_shape[0];
    int ny_box = box_shape[1];
    int nxny_box = nx_box * ny_box;
    std::vector<double> jx_t(numcell,0),jy_t(numcell,0),jz_t(numcell,0);

    std::vector<amrex::Real> xrhocells(8*numcell, 0.0);
    std::vector<amrex::Real> yrhocells(8*numcell, 0.0);
    std::vector<amrex::Real> zrhocells(8*numcell, 0.0);
    const uint64_t x_indices[8]={0,1,0,1,0,1,0,1};
    const uint64_t y_indices[8]={0,0,1,1,0,0,1,1};
    const uint64_t z_indices[8]={0,0,0,0,1,1,1,1};

    #ifdef BREAKDWON
    svetime[0]=rdtscv();
    #endif

    #pragma omp simd
    for (int ip = 0; ip < np_to_deposit; ip++){
        int bin=newbin[ip];
        #pragma omp simd
        for(int i=0;i<8;i++){
            int ix=x_indices[i];
            int iy=y_indices[i];
            int iz=z_indices[i];
            int binidx=8*bin+ix+iy*2+iz*4;
            double sxsysz=sx_m[ix+2*ip]*sy_m[iy+2*ip]*sz_m[iz+2*ip];
            double jx=sxsysz*wqx[ip];
            xrhocells[binidx]+=jx;
            double jy=sxsysz*wqy[ip];
            yrhocells[binidx]+=jy;
            double jz=sxsysz*wqz[ip];
            zrhocells[binidx]+=jz;
        }
    }

    #ifdef BREAKDWON
    svetime[1]=rdtscv();
    cal_time+=(svetime[1]-svetime[0]); 
    #endif

    long moff[8] = {0, 1, nx, nx+1, nxy, nxy+1, nx+nxy, nx+nxy+1};
    #ifdef BREAKDWON
    svetime[0]=rdtscv();
    #endif

    #pragma omp simd
    for (int iz = 0; iz < len.z; ++iz) {
        #pragma omp simd
        for (int iy = 0; iy < len.y; ++iy) { 
            #pragma omp simd      
            for (int ix = 0; ix < len.x; ix++) {

                long xoffset=ix + iy * xjstride + iz * xkstride;
                long cell_idx=ix + iy * nnx + iz * nnxny;
                long jxoffset=xoffset;
                long irhox=8*cell_idx;
                jx_arr.p[jxoffset+moff[0]]+=xrhocells[0+irhox];
                jx_arr.p[jxoffset+moff[1]]+=xrhocells[1+irhox];
                jx_arr.p[jxoffset+moff[2]]+=xrhocells[2+irhox];
                jx_arr.p[jxoffset+moff[3]]+=xrhocells[3+irhox];
                jx_arr.p[jxoffset+moff[4]]+=xrhocells[4+irhox];
                jx_arr.p[jxoffset+moff[5]]+=xrhocells[5+irhox];
                jx_arr.p[jxoffset+moff[6]]+=xrhocells[6+irhox];
                jx_arr.p[jxoffset+moff[7]]+=xrhocells[7+irhox];
                
                jy_arr.p[jxoffset+moff[0]]+=yrhocells[0+irhox];
                jy_arr.p[jxoffset+moff[1]]+=yrhocells[1+irhox];
                jy_arr.p[jxoffset+moff[2]]+=yrhocells[2+irhox];
                jy_arr.p[jxoffset+moff[3]]+=yrhocells[3+irhox];
                jy_arr.p[jxoffset+moff[4]]+=yrhocells[4+irhox];
                jy_arr.p[jxoffset+moff[5]]+=yrhocells[5+irhox];
                jy_arr.p[jxoffset+moff[6]]+=yrhocells[6+irhox];
                jy_arr.p[jxoffset+moff[7]]+=yrhocells[7+irhox];

                jz_arr.p[jxoffset+moff[0]]+=zrhocells[0+irhox];
                jz_arr.p[jxoffset+moff[1]]+=zrhocells[1+irhox];
                jz_arr.p[jxoffset+moff[2]]+=zrhocells[2+irhox];
                jz_arr.p[jxoffset+moff[3]]+=zrhocells[3+irhox];
                jz_arr.p[jxoffset+moff[4]]+=zrhocells[4+irhox];
                jz_arr.p[jxoffset+moff[5]]+=zrhocells[5+irhox];
                jz_arr.p[jxoffset+moff[6]]+=zrhocells[6+irhox];
                jz_arr.p[jxoffset+moff[7]]+=zrhocells[7+irhox];

            }
        }
    }
    
    #ifdef BREAKDWON
    svetime[1]=rdtscv();
    reduce_time+=(svetime[1]-svetime[0]);

    total_time[1]=rdtscv();
    total_time[2]+=(total_time[1]-total_time[0]);
    inner_timer[0]=cal_time+precompute+reduce_time;
    inner_timer[1]=precompute;
    inner_timer[2]=cal_time;
    inner_timer[4]=reduce_time; 
    #endif
 }
 
template <int depos_order>
void doDepositionShapeN_org_sp_rhocell_GPMA(const GetParticlePosition<PIdx> &GetPosition,
                                        const amrex::ParticleReal *const wp,
                                        const amrex::ParticleReal *const uxp,
                                        const amrex::ParticleReal *const uyp,
                                        const amrex::ParticleReal *const uzp,
                                        const int *ion_lev,
                                        amrex::FArrayBox &jx_fab,
                                        amrex::FArrayBox &jy_fab,
                                        amrex::FArrayBox &jz_fab,
                                        long np_to_deposit,
                                        amrex::Real relative_time,
                                        const amrex::XDim3 &dinv,
                                        const amrex::XDim3 &xyzmin,
                                        amrex::Dim3 lo,
                                        amrex::Dim3 hi,
                                        amrex::Dim3 len,
                                        amrex::Real q,
                                        WarpXParticleContainer::ParticleTileType &ptile, const amrex::Box &box,
                                        [[maybe_unused]] int n_rz_azimuthal_modes,
                                        std::vector<uint64_t>& inner_timer
                                    )
{
    using namespace amrex::literals;
    #ifdef BREAKDOWN
    using namespace std::chrono;
    uint64_t total_time[3]={0};
    uint64_t precompute=0;
    uint64_t cal_time=0;
    uint64_t sort_time=0;
        uint64_t presort=0;
        uint64_t insert=0;
        uint64_t borrow=0;
        uint64_t rebuildtime=0;
    uint64_t reduce_time=0;
    uint64_t svetime[3]={0};
    uint64_t svetime1[3]={0};
 
    total_time[0]=rdtscv();
    #endif
    amrex::Array4<amrex::Real> const &jx_arr = jx_fab.array();
    amrex::Array4<amrex::Real> const &jy_arr = jy_fab.array();
    amrex::Array4<amrex::Real> const &jz_arr = jz_fab.array();

    const int xjstride = jx_arr.jstride;
    const int xkstride = jx_arr.kstride;
    const int yjstride = jy_arr.jstride;
    const int ykstride = jy_arr.kstride;
    const int zjstride = jz_arr.jstride;
    const int zkstride = jz_arr.kstride;

    long nx = xjstride;
    long nxy = xkstride;
 
    const bool do_ionization = ion_lev;

    const amrex::Real invvol = dinv.x * dinv.y * dinv.z;

    const amrex::Real clightsq = 1.0_rt / PhysConst::c / PhysConst::c;

    const long numcell = jx_fab.box().numPts();

    amrex::IntVect const jx_type = jx_fab.box().type();
    amrex::IntVect const jy_type = jy_fab.box().type();
    amrex::IntVect const jz_type = jz_fab.box().type();

    std::vector<amrex::Real> wqx(np_to_deposit, 0.0_rt), wqy(np_to_deposit, 0.0_rt), wqz(np_to_deposit, 0.0_rt);
    
    std::vector<amrex::Real> sx_m(2 * np_to_deposit, 0);
    std::vector<amrex::Real> sy_m(2 * np_to_deposit, 0);
    std::vector<amrex::Real> sz_m(2 * np_to_deposit, 0);
    alignas(64) std::vector<int> newbin(np_to_deposit,-1);
    long nnx = len.x;
    long nny = len.y;
    long nnxny = nnx*nny;
    #ifdef BREAKDOWN
    svetime[0]=rdtscv();
    #endif
    #pragma omp simd
    for (int ip = 0; ip < np_to_deposit; ip++)
    {
        // for(int ip=ptile.m_num_particles;ip<np_to_deposit;ip++){
        amrex::ParticleReal xp, yp, zp;
        GetPosition(ip, xp, yp, zp);

        // --- Get particle quantities
        const amrex::Real gaminv = 1.0_rt / std::sqrt(1.0_rt + uxp[ip] * uxp[ip] * clightsq + uyp[ip] * uyp[ip] * clightsq + uzp[ip] * uzp[ip] * clightsq);
        const amrex::Real vx = uxp[ip] * gaminv;
        const amrex::Real vy = uyp[ip] * gaminv;
        const amrex::Real vz = uzp[ip] * gaminv;

        amrex::Real wq = q * wp[ip];
        if (do_ionization)
        {
            wq *= ion_lev[ip];
        }

        constexpr int NODE = amrex::IndexType::NODE;
        constexpr int CELL = amrex::IndexType::CELL;

        // wqx, wqy wqz are particle current in each direction
        wqx[ip] = wq * invvol * vx;
        wqy[ip] = wq * invvol * vy;
        wqz[ip] = wq * invvol * vz;

        // --- Compute shape factors
        Compute_shape_factor<depos_order> const compute_shape_factor;

        // const auto j = static_cast<int>(xmid);
        // x direction
        // Get particle position after 1/2 push back in position
        // Keep these double to avoid bug in single precision

        const double xmid = ((xp - xyzmin.x) + relative_time * vx) * dinv.x;

        double sx_node[depos_order + 1] = {0.};
        double sx_cell[depos_order + 1] = {0.};
        int j_node = 0;
        int j_cell = 0;
        if (jx_type[0] == NODE || jy_type[0] == NODE || jz_type[0] == NODE)
        {
            j_node = compute_shape_factor(sx_node, xmid);
        }
        if (jx_type[0] == CELL || jy_type[0] == CELL || jz_type[0] == CELL)
        {
            j_cell = compute_shape_factor(sx_cell, xmid - 0.5);
        }
        for (int ix = 0; ix <= depos_order; ix++)
        {
            sx_m[ip * 2 + ix] = ((jx_type[0] == NODE) ? amrex::Real(sx_node[ix]) : amrex::Real(sx_cell[ix]));
        }

        int jm = ((jz_type[0] == NODE) ? j_node : j_cell);

        // y direction
        // Keep these double to avoid bug in single precision
        const double ymid = ((yp - xyzmin.y) + relative_time * vy) * dinv.y;
        double sy_node[depos_order + 1] = {0.};
        double sy_cell[depos_order + 1] = {0.};
        int k_node = 0;
        int k_cell = 0;
        if (jx_type[1] == NODE || jy_type[1] == NODE || jz_type[1] == NODE)
        {
            k_node = compute_shape_factor(sy_node, ymid);
        }
        if (jx_type[1] == CELL || jy_type[1] == CELL || jz_type[1] == CELL)
        {
            k_cell = compute_shape_factor(sy_cell, ymid - 0.5);
        }
        for (int iy = 0; iy <= depos_order; iy++)
        {
            sy_m[ip * 2 + iy] = ((jx_type[1] == NODE) ? amrex::Real(sy_node[iy]) : amrex::Real(sy_cell[iy]));
        }
        int km = ((jx_type[1] == NODE) ? k_node : k_cell);

        // z direction
        // Keep these double to avoid bug in single precision
        constexpr int zdir = WARPX_ZINDEX;
        const double zmid = ((zp - xyzmin.z) + relative_time * vz) * dinv.z;
        double sz_node[depos_order + 1] = {0.};
        double sz_cell[depos_order + 1] = {0.};
        int l_node = 0;
        int l_cell = 0;
        if (jx_type[zdir] == NODE || jy_type[zdir] == NODE || jz_type[zdir] == NODE)
        {
            l_node = compute_shape_factor(sz_node, zmid);
        }
        if (jx_type[zdir] == CELL || jy_type[zdir] == CELL || jz_type[zdir] == CELL)
        {
            l_cell = compute_shape_factor(sz_cell, zmid - 0.5);
        }
        for (int iz = 0; iz <= depos_order; iz++)
        {
            sz_m[ip * 2 + iz] = ((jx_type[zdir] == NODE) ? amrex::Real(sz_node[iz]) : amrex::Real(sz_cell[iz]));
        }
        int lm = ((jz_type[zdir] == NODE) ? l_node : l_cell);
        newbin[ip]=jm+km*nnx+lm*nnxny;
    }
    #ifdef BREAKDOWN
    svetime[1]=rdtscv();
    precompute+=(svetime[1]-svetime[0]); 

    svetime1[0]=rdtscv();
    #endif
    increment_sort_particles (np_to_deposit, ptile,newbin, numcell);
    #ifdef BREAKDOWN
    svetime1[1]=rdtscv();
    sort_time+=(svetime1[1]-svetime1[0]); 
    #endif
 
    amrex::IntVect box_shape = box.length();

    int nx_box = box_shape[0];
    int ny_box = box_shape[1];
    int nxny_box = nx_box * ny_box;
    std::vector<double> jx_t(numcell,0),jy_t(numcell,0),jz_t(numcell,0);

    std::vector<amrex::Real> xrhocells(8*numcell, 0.0);
    std::vector<amrex::Real> yrhocells(8*numcell, 0.0);
    std::vector<amrex::Real> zrhocells(8*numcell, 0.0);
    const uint64_t x_indices[8]={0,1,0,1,0,1,0,1};
    const uint64_t y_indices[8]={0,0,1,1,0,0,1,1};
    const uint64_t z_indices[8]={0,0,0,0,1,1,1,1};
    #ifdef BREAKDOWN
    svetime[0]=rdtscv();
    #endif
    #pragma omp unroll 
    for (int iz = 0; iz < len.z; ++iz) {
        for (int iy = 0; iy < len.y; ++iy) {       
            for (int ix = 0; ix < len.x; ix++) {
                int old_bin=(ix) + (iy) * nnx + (iz) * nnxny;
                int binlength = ptile.m_bin_lengths[old_bin];
                int binoffset = ptile.m_bin_offsets[old_bin];
                double xtemp[8]={0.0};
                double ytemp[8]={0.0};
                double ztemp[8]={0.0};
                #pragma omp simd
                for(int j=0;j<binlength;j++){
                    int ip = ptile.m_local_index[binoffset+j];
                    #pragma omp simd
                    for(int i=0;i<8;i++){
                        int ix=x_indices[i];
                        int iy=y_indices[i];
                        int iz=z_indices[i];
                        double sxsysz=sx_m[ix+2*ip]*sy_m[iy+2*ip]*sz_m[iz+2*ip];
                        double jx=sxsysz*wqx[ip];
                        xtemp[i]+=jx;
                        double jy=sxsysz*wqy[ip];
                        ytemp[i]+=jy;
                        double jz=sxsysz*wqz[ip];
                        ztemp[i]+=jz;
                    }
                }
                #pragma omp simd
                for(int i=0;i<8;i++){
                    int ix=x_indices[i];
                    int iy=y_indices[i];
                    int iz=z_indices[i];
                    int binidx=8*old_bin+ix+iy*2+iz*4;
                    xrhocells[binidx]=xtemp[i];
                    yrhocells[binidx]=ytemp[i];
                    zrhocells[binidx]=ztemp[i];
                }
            }
        }
    }
    #ifdef BREAKDOWN
    svetime[1]=rdtscv();
    cal_time+=(svetime[1]-svetime[0]); 
    #endif

    long moff[8] = {0, 1, nx, nx+1, nxy, nxy+1, nx+nxy, nx+nxy+1};
    #ifdef BREAKDOWN
    svetime[0]=rdtscv();
    #endif
    #pragma omp simd
    for (int iz = 0; iz < len.z; ++iz) {
        #pragma omp simd
        for (int iy = 0; iy < len.y; ++iy) { 
            #pragma omp simd      
            for (int ix = 0; ix < len.x; ix++) {

                long xoffset=ix + iy * xjstride + iz * xkstride;
                long cell_idx=ix + iy * nnx + iz * nnxny;
                long jxoffset=xoffset;
                long irhox=8*cell_idx;
                jx_arr.p[jxoffset+moff[0]]+=xrhocells[0+irhox];
                jx_arr.p[jxoffset+moff[1]]+=xrhocells[1+irhox];
                jx_arr.p[jxoffset+moff[2]]+=xrhocells[2+irhox];
                jx_arr.p[jxoffset+moff[3]]+=xrhocells[3+irhox];
                jx_arr.p[jxoffset+moff[4]]+=xrhocells[4+irhox];
                jx_arr.p[jxoffset+moff[5]]+=xrhocells[5+irhox];
                jx_arr.p[jxoffset+moff[6]]+=xrhocells[6+irhox];
                jx_arr.p[jxoffset+moff[7]]+=xrhocells[7+irhox];
                
                jy_arr.p[jxoffset+moff[0]]+=yrhocells[0+irhox];
                jy_arr.p[jxoffset+moff[1]]+=yrhocells[1+irhox];
                jy_arr.p[jxoffset+moff[2]]+=yrhocells[2+irhox];
                jy_arr.p[jxoffset+moff[3]]+=yrhocells[3+irhox];
                jy_arr.p[jxoffset+moff[4]]+=yrhocells[4+irhox];
                jy_arr.p[jxoffset+moff[5]]+=yrhocells[5+irhox];
                jy_arr.p[jxoffset+moff[6]]+=yrhocells[6+irhox];
                jy_arr.p[jxoffset+moff[7]]+=yrhocells[7+irhox];

                jz_arr.p[jxoffset+moff[0]]+=zrhocells[0+irhox];
                jz_arr.p[jxoffset+moff[1]]+=zrhocells[1+irhox];
                jz_arr.p[jxoffset+moff[2]]+=zrhocells[2+irhox];
                jz_arr.p[jxoffset+moff[3]]+=zrhocells[3+irhox];
                jz_arr.p[jxoffset+moff[4]]+=zrhocells[4+irhox];
                jz_arr.p[jxoffset+moff[5]]+=zrhocells[5+irhox];
                jz_arr.p[jxoffset+moff[6]]+=zrhocells[6+irhox];
                jz_arr.p[jxoffset+moff[7]]+=zrhocells[7+irhox];

            }
        }
    }
    #ifdef BREAKDOWN
    svetime[1]=rdtscv();
    reduce_time+=(svetime[1]-svetime[0]);

    total_time[1]=rdtscv();
    total_time[2]+=(total_time[1]-total_time[0]);
    inner_timer[0]=cal_time+precompute+sort_time+reduce_time;
    inner_timer[1]=precompute;
    inner_timer[2]=cal_time;
    inner_timer[3]=sort_time;
    inner_timer[4]=reduce_time;
    #endif
}

template <int depos_order>
void doDepositionShapeN_org_sp_rhocell_GPMA_sve(const GetParticlePosition<PIdx> &GetPosition,
                                        const amrex::ParticleReal *const wp,
                                        const amrex::ParticleReal *const uxp,
                                        const amrex::ParticleReal *const uyp,
                                        const amrex::ParticleReal *const uzp,
                                        const int *ion_lev,
                                        amrex::FArrayBox &jx_fab,
                                        amrex::FArrayBox &jy_fab,
                                        amrex::FArrayBox &jz_fab,
                                        long np_to_deposit,
                                        amrex::Real relative_time,
                                        const amrex::XDim3 &dinv,
                                        const amrex::XDim3 &xyzmin,
                                        amrex::Dim3 lo,
                                        amrex::Dim3 hi,
                                        amrex::Dim3 len,
                                        amrex::Real q,
                                        WarpXParticleContainer::ParticleTileType &ptile, const amrex::Box &box,
                                        [[maybe_unused]] int n_rz_azimuthal_modes,
                                        std::vector<uint64_t>& inner_timer
                                    )
{
    using namespace amrex::literals;
    #ifdef BREAKDOWN
    using namespace std::chrono;
    uint64_t total_time[3]={0};
    uint64_t precompute=0;
    uint64_t cal_time=0;
    uint64_t sort_time=0;
        uint64_t presort=0;
        uint64_t insert=0;
        uint64_t borrow=0;
        uint64_t rebuildtime=0;
    uint64_t reduce_time=0;
    uint64_t svetime[3]={0};
    uint64_t svetime1[3]={0};

    total_time[0]=rdtscv();
    #endif
    amrex::Array4<amrex::Real> const &jx_arr = jx_fab.array();
    amrex::Array4<amrex::Real> const &jy_arr = jy_fab.array();
    amrex::Array4<amrex::Real> const &jz_arr = jz_fab.array();

    const int xjstride = jx_arr.jstride;
    const int xkstride = jx_arr.kstride;
    const int yjstride = jy_arr.jstride;
    const int ykstride = jy_arr.kstride;
    const int zjstride = jz_arr.jstride;
    const int zkstride = jz_arr.kstride;

    long nx = xjstride;
    long nxy = xkstride;

    const bool do_ionization = ion_lev;

    const amrex::Real invvol = dinv.x * dinv.y * dinv.z;

    const amrex::Real clightsq = 1.0_rt / PhysConst::c / PhysConst::c;

    const long numcell = ptile.m_num_bins;


    amrex::IntVect const jx_type = jx_fab.box().type();
    amrex::IntVect const jy_type = jy_fab.box().type();
    amrex::IntVect const jz_type = jz_fab.box().type();

    std::vector<amrex::Real> wqx(np_to_deposit, 0.0_rt), wqy(np_to_deposit, 0.0_rt), wqz(np_to_deposit, 0.0_rt);
    
    std::vector<amrex::Real> sx_m(2 * np_to_deposit, 0);
    std::vector<amrex::Real> sy_m(2 * np_to_deposit, 0);
    std::vector<amrex::Real> sz_m(2 * np_to_deposit, 0);
    alignas(64) std::vector<int> newbin(np_to_deposit,-1);
        
    auto compute_shape_factor_v2 = [](double* sx, Vec xmid, svbool_t p, int np) {
        intVec i_newv = svcvt_s64_f64_z(p, xmid);
        Vec j = svcvt_f64_s64_z(p, i_newv);
        Vec xint = xmid - j;
        Vec sx0 = 1.0 - xint;
        Vec sx1 = xint;
        sx0.Store(p,&sx[0*np]);
        sx1.Store(p,&sx[1*np]);
        return i_newv;
    };
    int vl=svcntd();
    const amrex::ParticleReal* mx = GetPosition.m_x;
    const amrex::ParticleReal* my = GetPosition.m_y;
    const amrex::ParticleReal* mz = GetPosition.m_z;
    long nnx = len.x;
    long nny = len.y;
    long nnxny = nnx*nny;

    #ifdef BREAKDOWN
    svetime[0]=rdtscv();
    #endif
    #pragma omp unroll
    for(long ip=0;ip<np_to_deposit;ip+=vl){
        svbool_t p=svwhilelt_b64(ip,np_to_deposit);
        // Inverse of Lorentz factor gamma
        Vec uxp_v=Vec::Load(p,&uxp[ip]);
        Vec uyp_v=Vec::Load(p,&uyp[ip]);
        Vec uzp_v=Vec::Load(p,&uzp[ip]);
        Vec gaminv = 1._rt / (1._rt + uxp_v * uxp_v * clightsq
                                    + uyp_v * uyp_v * clightsq
                                    + uzp_v * uzp_v * clightsq).Sqrt();
        // Product of particle charges and weights
        Vec wp_v=Vec::Load(p,&wp[ip]);
        Vec wq_v = q * wp_v;

        // Current particle positions (in physical units)
        Vec xp = Vec::Load(p, &mx[ip]);
        Vec yp = Vec::Load(p, &my[ip]);
        Vec zp = Vec::Load(p, &mz[ip]);

        // Particle velocities
        Vec vx_v = uxp_v * gaminv;
        Vec vy_v = uyp_v * gaminv;
        Vec vz_v = uzp_v * gaminv;

        // if (do_ionization) wq *= ion_lev[ip];
        Vec wqx_v = wq_v * invvol * vx_v;
        Vec wqy_v = wq_v * invvol * vy_v;
        Vec wqz_v = wq_v * invvol * vz_v;

        wqx_v.Store(p, &wqx[ip]);
        wqy_v.Store(p, &wqy[ip]);
        wqz_v.Store(p, &wqz[ip]);

        Vec xmid = ((xp - xyzmin.x) + relative_time * vx_v) * dinv.x;
        Vec ymid = ((yp - xyzmin.y) + relative_time * vy_v) * dinv.y;
        Vec zmid = ((zp - xyzmin.z) + relative_time * vz_v) * dinv.z;
        intVec j_cellv = compute_shape_factor_v2( &sx_m[ip], xmid, p,np_to_deposit);
        intVec k_cellv = compute_shape_factor_v2( &sy_m[ip], ymid, p,np_to_deposit);
        intVec l_cellv = compute_shape_factor_v2( &sz_m[ip], zmid, p,np_to_deposit); 
        intVec new_bin_v = (j_cellv) + 
                            (k_cellv)* nnx + 
                            (l_cellv) * nnxny;
        
        svst1w_s64 (p, &newbin[ip], new_bin_v);
    }
    #ifdef BREAKDOWN
    svetime[1]=rdtscv();
    precompute+=(svetime[1]-svetime[0]); 

    svetime1[0]=rdtscv();
    #endif
    increment_sort_particles (np_to_deposit, ptile,newbin, numcell);
    #ifdef BREAKDOWN
    svetime1[1]=rdtscv();
    sort_time+=(svetime1[1]-svetime1[0]); 
    #endif

    amrex::IntVect box_shape = box.length();

    int nx_box = box_shape[0];
    int ny_box = box_shape[1];
    int nxny_box = nx_box * ny_box;
    std::vector<double> jx_t(numcell,0),jy_t(numcell,0),jz_t(numcell,0);

    std::vector<amrex::Real> xrhocells(8*numcell, 0.0);
    std::vector<amrex::Real> yrhocells(8*numcell, 0.0);
    std::vector<amrex::Real> zrhocells(8*numcell, 0.0);
    uint64_t npp=static_cast<uint64_t>(np_to_deposit);
    const uint64_t x_indices[8]={0,npp,0,npp,0,npp,0,npp};
    const uint64_t y_indices[8]={0,0,npp,npp,0,0,npp,npp};
    const uint64_t z_indices[8]={0,0,0,0,npp,npp,npp,npp};
    svuint64_t xidx_v=svld1_u64(svptrue_b64(),x_indices);
    svuint64_t yidx_v=svld1_u64(svptrue_b64(),y_indices);
    svuint64_t zidx_v=svld1_u64(svptrue_b64(),z_indices);
    #ifdef BREAKDOWN
    svetime[0]=rdtscv();
    #endif
    #pragma omp unroll 
    for (int iz = 0; iz < len.z; ++iz) {
        for (int iy = 0; iy < len.y; ++iy) {       
            for (int ix = 0; ix < len.x; ix++) {
                int old_bin=(ix) + (iy) * nnx + (iz) * nnxny;
                int binlength = ptile.m_bin_lengths[old_bin];
                int binoffset = ptile.m_bin_offsets[old_bin];
                Vec jx(0),jy(0),jz(0);
                #pragma omp unroll 
                for(int j=0;j<binlength;j++){
                    int ip = ptile.m_local_index[binoffset+j];
                    
                    Vec sxm=svld1_gather_index(svptrue_b64(),&sx_m[ip],xidx_v);
                    Vec sym=svld1_gather_index(svptrue_b64(),&sy_m[ip],yidx_v);
                    Vec szm=svld1_gather_index(svptrue_b64(),&sz_m[ip],zidx_v);
                    Vec sxsysz=sxm*sym*szm;
                    jx+=sxsysz*wqx[ip];
                    jy+=sxsysz*wqy[ip];
                    jz+=sxsysz*wqz[ip];
                    
                }
                jx.Store(svptrue_b64(),&xrhocells[8*old_bin]);
                jy.Store(svptrue_b64(),&yrhocells[8*old_bin]);
                jz.Store(svptrue_b64(),&zrhocells[8*old_bin]);
            }
        }
    }
    #ifdef BREAKDOWN
    svetime[1]=rdtscv();
    cal_time+=(svetime[1]-svetime[0]); 
    
    svetime[0]=rdtscv();
    #endif
    long moff[8] = {0, 1, nx, nx+1, nxy, nxy+1, nx+nxy, nx+nxy+1};
    #pragma omp simd
    for (int iz = 0; iz < len.z; ++iz) {
        #pragma omp simd
        for (int iy = 0; iy < len.y; ++iy) { 
            #pragma unroll      
            for (int ix = 0; ix < len.x; ix+=vl) {
                // vector
                svbool_t p=svwhilelt_b64(ix,(len.x));
                // x
                svuint64_t rho_index = svindex_u64(0ULL, 8ULL);
                long xoffset=ix + iy * xjstride + iz * xkstride;
                long cell_idx=ix + iy * nnx + iz * nnxny;
                long jxoffset=xoffset;
                long irhox=8*cell_idx;
                Vec jx0=Vec::Load(p,&jx_arr.p[jxoffset+moff[0]]);
                svfloat64_t rhox0=svld1_gather_u64index_f64(p,&xrhocells[0+irhox],rho_index);
                jx0+=rhox0;
                jx0.Store(p,&jx_arr.p[jxoffset+moff[0]]);

                Vec jx1=Vec::Load(p,&jx_arr.p[jxoffset+moff[1]]);
                svfloat64_t rhox1=svld1_gather_u64index_f64(p,&xrhocells[1+irhox],rho_index);
                jx1+=rhox1;
                jx1.Store(p,&jx_arr.p[jxoffset+moff[1]]);

                Vec jx2=Vec::Load(p,&jx_arr.p[jxoffset+moff[2]]);
                svfloat64_t rhox2=svld1_gather_u64index_f64(p,&xrhocells[2+irhox],rho_index);
                jx2+=rhox2;
                jx2.Store(p,&jx_arr.p[jxoffset+moff[2]]);

                Vec jx3=Vec::Load(p,&jx_arr.p[jxoffset+moff[3]]);
                svfloat64_t rhox3=svld1_gather_u64index_f64(p,&xrhocells[3+irhox],rho_index);
                jx3+=rhox3;
                jx3.Store(p,&jx_arr.p[jxoffset+moff[3]]);

                Vec jx4=Vec::Load(p,&jx_arr.p[jxoffset+moff[4]]);
                svfloat64_t rhox4=svld1_gather_u64index_f64(p,&xrhocells[4+irhox],rho_index);
                jx4+=rhox4;
                jx4.Store(p,&jx_arr.p[jxoffset+moff[4]]);

                Vec jx5=Vec::Load(p,&jx_arr.p[jxoffset+moff[5]]);
                svfloat64_t rhox5=svld1_gather_u64index_f64(p,&xrhocells[5+irhox],rho_index);
                jx5+=rhox5;
                jx5.Store(p,&jx_arr.p[jxoffset+moff[5]]);

                Vec jx6=Vec::Load(p,&jx_arr.p[jxoffset+moff[6]]);
                svfloat64_t rhox6=svld1_gather_u64index_f64(p,&xrhocells[6+irhox],rho_index);
                jx6+=rhox6;
                jx6.Store(p,&jx_arr.p[jxoffset+moff[6]]);

                Vec jx7=Vec::Load(p,&jx_arr.p[jxoffset+moff[7]]);
                svfloat64_t rhox7=svld1_gather_u64index_f64(p,&xrhocells[7+irhox],rho_index);
                jx7+=rhox7;
                jx7.Store(p,&jx_arr.p[jxoffset+moff[7]]);
                
                // y
                long jyoffset=xoffset;
                long irhoy=irhox;
                Vec jy0=Vec::Load(p,&jy_arr.p[jyoffset+moff[0]]);
                svfloat64_t rhoy0=svld1_gather_u64index_f64(p,&yrhocells[0+irhoy],rho_index);
                jy0+=rhoy0;
                jy0.Store(p,&jy_arr.p[jyoffset+moff[0]]);

                Vec jy1=Vec::Load(p,&jy_arr.p[jyoffset+moff[1]]);
                svfloat64_t rhoy1=svld1_gather_u64index_f64(p,&yrhocells[1+irhoy],rho_index);
                jy1+=rhoy1;
                jy1.Store(p,&jy_arr.p[jyoffset+moff[1]]);

                Vec jy2=Vec::Load(p,&jy_arr.p[jyoffset+moff[2]]);
                svfloat64_t rhoy2=svld1_gather_u64index_f64(p,&yrhocells[2+irhoy],rho_index);
                jy2+=rhoy2;
                jy2.Store(p,&jy_arr.p[jyoffset+moff[2]]);

                Vec jy3=Vec::Load(p,&jy_arr.p[jyoffset+moff[3]]);
                svfloat64_t rhoy3=svld1_gather_u64index_f64(p,&yrhocells[3+irhoy],rho_index);
                jy3+=rhoy3;
                jy3.Store(p,&jy_arr.p[jyoffset+moff[3]]);

                Vec jy4=Vec::Load(p,&jy_arr.p[jyoffset+moff[4]]);
                svfloat64_t rhoy4=svld1_gather_u64index_f64(p,&yrhocells[4+irhoy],rho_index);
                jy4+=rhoy4;
                jy4.Store(p,&jy_arr.p[jyoffset+moff[4]]);

                Vec jy5=Vec::Load(p,&jy_arr.p[jyoffset+moff[5]]);
                svfloat64_t rhoy5=svld1_gather_u64index_f64(p,&yrhocells[5+irhoy],rho_index);
                jy5+=rhoy5;
                jy5.Store(p,&jy_arr.p[jyoffset+moff[5]]);

                Vec jy6=Vec::Load(p,&jy_arr.p[jyoffset+moff[6]]);
                svfloat64_t rhoy6=svld1_gather_u64index_f64(p,&yrhocells[6+irhoy],rho_index);
                jy6+=rhoy6;
                jy6.Store(p,&jy_arr.p[jyoffset+moff[6]]);

                Vec jy7=Vec::Load(p,&jy_arr.p[jyoffset+moff[7]]);
                svfloat64_t rhoy7=svld1_gather_u64index_f64(p,&yrhocells[7+irhoy],rho_index);
                jy7+=rhoy7;
                jy7.Store(p,&jy_arr.p[jyoffset+moff[7]]);

                // z
                long jzoffset=xoffset;
                long irhoz=irhox;
                Vec jz0=Vec::Load(p,&jz_arr.p[jzoffset+moff[0]]);
                svfloat64_t rhoz0=svld1_gather_u64index_f64(p,&zrhocells[0+irhoz],rho_index);
                jz0+=rhoz0;
                jz0.Store(p,&jz_arr.p[jzoffset+moff[0]]);

                Vec jz1=Vec::Load(p,&jz_arr.p[jzoffset+moff[1]]);
                svfloat64_t rhoz1=svld1_gather_u64index_f64(p,&zrhocells[1+irhoz],rho_index);
                jz1+=rhoz1;
                jz1.Store(p,&jz_arr.p[jzoffset+moff[1]]);

                Vec jz2=Vec::Load(p,&jz_arr.p[jzoffset+moff[2]]);
                svfloat64_t rhoz2=svld1_gather_u64index_f64(p,&zrhocells[2+irhoz],rho_index);
                jz2+=rhoz2;
                jz2.Store(p,&jz_arr.p[jzoffset+moff[2]]);

                Vec jz3=Vec::Load(p,&jz_arr.p[jzoffset+moff[3]]);
                svfloat64_t rhoz3=svld1_gather_u64index_f64(p,&zrhocells[3+irhoz],rho_index);
                jz3+=rhoz3;
                jz3.Store(p,&jz_arr.p[jzoffset+moff[3]]);

                Vec jz4=Vec::Load(p,&jz_arr.p[jzoffset+moff[4]]);
                svfloat64_t rhoz4=svld1_gather_u64index_f64(p,&zrhocells[4+irhoz],rho_index);
                jz4+=rhoz4;
                jz4.Store(p,&jz_arr.p[jzoffset+moff[4]]);

                Vec jz5=Vec::Load(p,&jz_arr.p[jzoffset+moff[5]]);
                svfloat64_t rhoz5=svld1_gather_u64index_f64(p,&zrhocells[5+irhoz],rho_index);
                jz5+=rhoz5;
                jz5.Store(p,&jz_arr.p[jzoffset+moff[5]]);

                Vec jz6=Vec::Load(p,&jz_arr.p[jzoffset+moff[6]]);
                svfloat64_t rhoz6=svld1_gather_u64index_f64(p,&zrhocells[6+irhoz],rho_index);
                jz6+=rhoz6;
                jz6.Store(p,&jz_arr.p[jzoffset+moff[6]]);

                Vec jz7=Vec::Load(p,&jz_arr.p[jzoffset+moff[7]]);
                svfloat64_t rhoz7=svld1_gather_u64index_f64(p,&zrhocells[7+irhoz],rho_index);
                jz7+=rhoz7;
                jz7.Store(p,&jz_arr.p[jzoffset+moff[7]]);

            }
        }
    }
    #ifdef BREAKDOWN
    svetime[1]=rdtscv();
    reduce_time+=(svetime[1]-svetime[0]);

    total_time[1]=rdtscv();
    total_time[2]+=(total_time[1]-total_time[0]);
    inner_timer[0]=cal_time+precompute+sort_time+reduce_time;
    inner_timer[1]=precompute;
    inner_timer[2]=cal_time;
    inner_timer[3]=sort_time;
    inner_timer[4]=reduce_time;
    #endif
}

inline void increment_sort_particles_order3_OPM (
    int np_to_deposit, ParticleTileType& ptile,
    OPM_Inner_Vector_Int& newbin, int numcell, amrex::Dim3 len
)
{
    auto& local_index = ptile.m_local_index_hbm;
    // Aligned with bin_offsets, length is ncell+1, [0] points to index 0.
    auto& slot_offsets = ptile.m_slot_offsets; 
    auto& bin_length = ptile.m_bin_lengths;
    
    auto& num_particles = ptile.m_num_particles;
    auto& bin_offsets = ptile.m_bin_offsets;
    auto& pid_to_bin_map = ptile.m_pid_to_bin_map_hbm;
    auto& pid_to_index_map = ptile.m_pid_to_index_map; 
    auto& was_rebuilt_this_step =  ptile.m_was_rebuilt_this_step;
    was_rebuilt_this_step = false ;
    
    // TODO: move ouside
    std::unordered_map<int, OPM_Inner_Vector_Int> m_pending_moves;
    m_pending_moves.clear();

    bool rebuild=false;
    int vlf = svcntw();

    // Stage 1: Prepare data for incremental update.
    // Pre-process particle updates by organizing them into contiguous blocks.
    if(np_to_deposit<ptile.m_capacity && !rebuild){
        
        // 1. Handle particle deletion for particles that are no longer present.
        // move outside
        std::unordered_map<int, int> bin_decrement_counts;
        for(int ip=np_to_deposit;ip<num_particles;ip++){
            int old_bin=pid_to_bin_map[ip];
            int idx=pid_to_index_map[ip]; 
            // Invalidate the particle's index, which will be compacted later.
            ptile.m_local_index_hbm[idx] = INVALID_PARTICLE_ID;
            // This is handled later in a batch operation.
            // bin_length[old_bin]--; 
            // This must be recorded to be included in the compaction step.
            bin_decrement_counts[old_bin]++; 
        }

        // 2. For particles that have changed bins, treat it as a deletion from their old bin.
        // The loop iterates up to the minimum of the old and new particle counts,
        // as newly added particles (ip > num_particles) and removed particles (ip > np_to_deposit) are handled separately.
        int loop_np=std::min(num_particles,np_to_deposit); 
        int block_size=WarpX::GetInstance().m_fast_cmp_block_size;; 
        float max_np_rate=WarpX::GetInstance().m_max_moved_np_rate; 
        int MOVED_PARTICLES_MAX=loop_np/4+block_size;

        int moved_ips[MOVED_PARTICLES_MAX];
        int moved_old_bins[MOVED_PARTICLES_MAX];
        int moved_new_bins[MOVED_PARTICLES_MAX];
        int moved_count = 0;

        svint32_t vec_invalid_id = svdup_n_s32(INVALID_PARTICLE_ID);

        for (int i = 0; i < loop_np; i += block_size) {
            int current_block_size = std::min(block_size, loop_np - i);
            // Calculate the number of bytes to compare.
            size_t bytes_to_compare = current_block_size * sizeof(int);

            // --- Fast path: Use memcmp to compare the entire block ---
            // memcmp compares memory content. A return value of 0 means the two blocks are identical.
            if (memcmp(&newbin[i], &pid_to_bin_map[i], bytes_to_compare) == 0) {
                // If no particles in the block have moved, skip it. This is key for performance.
                continue; 
            } 
            else {
                for (int j = 0; j < current_block_size; j += vlf) {
                    int ip=i+j;
                    // Create a predicate covering the current vector lane.
                    svbool_t pg32 = svwhilelt_b32(j, current_block_size);
                    svint32_t vec_newbin = svld1_s32(pg32, &newbin[ip]);
                    svint32_t vec_oldbin   = svld1_s32(pg32, &pid_to_bin_map[ip]);
                    svbool_t moved_mask = svcmpne(pg32, vec_newbin, vec_oldbin);
                    if (!svptest_any(pg32, moved_mask)) {
                        continue;
                    }
                    // --- Start processing moved particles ---
                    // 0) Create a vector of particle indices (ip) for the current block.
                    svint32_t all_ips_in_block = svindex_s32(ip, 1);

                    // 1) Get the storage indices (idx) of the moved particles.
                    svint32_t moved_idxs_vec = svld1_s32(moved_mask, &pid_to_index_map[ip]);

                    // 4) SCATTER operation: Set the corresponding positions in local_index to invalid.
                    svst1_scatter_s32index_s32(moved_mask, &local_index[0], moved_idxs_vec, vec_invalid_id);

                    
                    // 5) PACK operation: Pack all required information.
                    int32_t count_in_this_block = svcntp_b32(pg32, moved_mask);
                    svbool_t pg_compact = svwhilelt_b32(0, count_in_this_block);

                    // a. Compact ip, new_bin, and old_bin.
                    svint32_t moved_ips_vec      = svcompact(moved_mask, all_ips_in_block);
                    svint32_t moved_new_bins_vec = svcompact(moved_mask, vec_newbin);
                    svint32_t moved_old_bins_vec = svcompact(moved_mask, vec_oldbin);

                    // b. Store the compacted vectors into the output arrays.
                    svst1_s32(pg_compact, &moved_ips[moved_count], moved_ips_vec);
                    svst1_s32(pg_compact, &moved_new_bins[moved_count], moved_new_bins_vec);
                    svst1_s32(pg_compact, &moved_old_bins[moved_count], moved_old_bins_vec);

                    // 6) Update the total count of moved particles.
                    moved_count += count_in_this_block;
                }
                
                if (moved_count > loop_np*max_np_rate) {
                    rebuild=true;
                    break;
                }
            }
        }
        
        // To efficiently update bin_length, use a temporary map for counting.
        if (!rebuild) {
            for (int k = 0; k < moved_count; ++k) {
                int ip = moved_ips[k];
                int new_bin = moved_new_bins[k];
                int old_bin = moved_old_bins[k];

                // Add the particle to the list of pending moves for its new bin.
                m_pending_moves[new_bin].push_back(ip);

                // Count the number of particles leaving each old_bin.
                bin_decrement_counts[old_bin]++;
            }
            // Update bin_length in one batch operation.
            for (auto const& [bin, count] : bin_decrement_counts) {
                bin_length[bin] -= count;
            }
            
            // 3. Compact the local_index to remove invalid entries.
            sorted_index_pma_compact_hbm(bin_decrement_counts,
                local_index,slot_offsets,bin_offsets,pid_to_index_map);

            // 4. Add newly created particles. They are appended to the pending moves list.
            for(int ip=num_particles;ip<np_to_deposit;ip++){
                int new_bin=newbin[ip];
                m_pending_moves[new_bin].push_back(ip);
            }
        }
    } else {
        rebuild=true;
    }
    
    // At this point, bin_length and slot_offset have been decremented for particles that left bins.
    // m_local_index has been compacted, and pid_to_index_map has been updated accordingly.

    // Stage 2: Perform the incremental update.
    // Next, update m_pid_to_bin_map, pid_to_index_map, and num_particles.
    // As particles are inserted, slot_offset and bin_offsets will also need to be updated.
    // The moved particles will be inserted into their new locations in m_local_index.
    pid_to_index_map.reserve(np_to_deposit+1);  
    pid_to_index_map.resize(np_to_deposit);

    pid_to_bin_map.reserve(np_to_deposit+1);
    pid_to_bin_map.resize(np_to_deposit);
    std::memcpy(pid_to_bin_map.data(),
                newbin.data(),
                np_to_deposit * sizeof(int));
    num_particles=np_to_deposit;

    // Stage 2.1: Perform insertions into bins.
    // Insert at the tail first, as empty slots are typically located there.
    // If there's not enough space at the tail, borrow from the front. If the front also lacks space, break and trigger a rebuild.
    if(!rebuild){ 
        for (const auto& pair : m_pending_moves) {
            int new_bin=pair.first;
            const OPM_Inner_Vector_Int& moved_ips = pair.second;
            int ip_size = moved_ips.size();
            
            int bin_end = bin_offsets[new_bin+1];
            int tail_bin_slot = slot_offsets[new_bin+1];
            int bin_start = bin_offsets[new_bin];
            int front_bin_slot = slot_offsets[new_bin];

            // Case 1: Enough space at the tail of the bin.
            if(bin_end-tail_bin_slot>=ip_size){
                std::memcpy(local_index.data()+tail_bin_slot,
                            moved_ips.data(),
                            ip_size * sizeof(int)); 
                
                for(int i=0;i<ip_size;i+=vlf){
                    svbool_t p=svwhilelt_b32(i,ip_size);
                    svint32_t ip_v = svld1_s32(p,&moved_ips[i]);
                    svint32_t pid_to_index_map_v = svindex_s32(slot_offsets[new_bin+1]+i,1);
                    svst1_scatter_s32index_s32(p,&pid_to_index_map[0],ip_v,pid_to_index_map_v);
                } 
                slot_offsets[new_bin+1]+=ip_size;
            // Case 2: Not enough space at the tail, but combined front and tail space is sufficient.
            }else if((bin_end-tail_bin_slot)+(bin_start-front_bin_slot)>=ip_size){
                int borrow_front=ip_size-(bin_end-tail_bin_slot);
                // Shift the entire bin's existing data forward to make space at the end.
                std::memmove(local_index.data()+bin_start-borrow_front,
                            local_index.data()+bin_start,
                            bin_length[new_bin] * sizeof(int));

                bin_offsets[new_bin]-=borrow_front;
                slot_offsets[new_bin+1]-=borrow_front;
                // Copy the new particle data into the newly created space.
                std::memcpy(local_index.data()+slot_offsets[new_bin+1],
                            moved_ips.data(),
                            ip_size * sizeof(int));
                // Update pid_to_index_map for all particles in the shifted bin.
                for(int i=0;i<ip_size+bin_length[new_bin];i+=vlf){
                    svbool_t p=svwhilelt_b32(i,ip_size+bin_length[new_bin]);
                    svint32_t ip_v = svld1_s32(p,&local_index[bin_offsets[new_bin]+i]);
                    svint32_t pid_to_index_map_v = svindex_s32(bin_offsets[new_bin]+i,1);
                    svst1_scatter_s32index_s32(p,&pid_to_index_map[0],ip_v,pid_to_index_map_v);
                }
                slot_offsets[new_bin+1]+=ip_size;
            // Case 3: Not enough space, trigger a rebuild.
            }else{
                rebuild=true;
                break;
            }
            bin_length[new_bin]+=ip_size;
        }
    }
    m_pending_moves.clear();

    // Stage 3: If incremental update failed or was deemed inefficient, perform a full rebuild.
    if(rebuild)
    { 
        ptile.m_was_rebuilt_this_step = true;

        // TODO:move outside
        float gap_ratio=WarpX::GetInstance().m_gap_ratio;
        ptile.m_bin_lengths.clear();
        ptile.m_bin_lengths.resize(numcell);

        // Stage 3.1: Recalculate m_bin_lengths from scratch.
        for(int ip=0;ip<np_to_deposit;ip++){
            ptile.m_bin_lengths[ptile.m_pid_to_bin_map_hbm[ip]]++;
        }   
        // Estimate the required capacity with a gap for future growth.
        ptile.m_capacity = static_cast<int>(
                std::ceil(static_cast<double>(
                np_to_deposit) * (1.0 + gap_ratio)) + 2*numcell);
        // Initialize offset arrays.
        int current_offset = 0;
        ptile.m_bin_offsets.clear();
        ptile.m_bin_offsets.resize(numcell+1);
        ptile.m_slot_offsets.clear();
        ptile.m_slot_offsets.resize(numcell+1);
        ptile.m_bin_offsets[0] = 0;
        ptile.m_slot_offsets[0] = 0;
        
        // Stage 3.2: Calculate m_bin_offsets and m_slot_offsets using SVE for parallel prefix sum.
        #pragma unroll
        for (int binID = 0; binID < numcell; binID+=vlf) {
            svbool_t p_bin = svwhilelt_b32(binID,numcell);
            svint32_t np_in_bin=svld1_s32(p_bin,&ptile.m_bin_lengths[binID]);
            svint32_t gap_v=svadd_n_s32_x(p_bin,
                                svcvt_s32_f32_x(p_bin,svmul_n_f32_x(p_bin,
                                svcvt_f32_s32_x(p_bin,np_in_bin),gap_ratio)),
                                1);
            svint32_t block_size_v=svadd_s32_x(p_bin,gap_v,np_in_bin);
            // 1. Vectorized local accumulation (Intra-Vector Exclusive Scan).
            //    This is the core of converting the serial dependency of prefix sum into parallel computation.
            svint32_t local_scan_v = block_size_v;
            local_scan_v = svadd_s32_m(p_bin, local_scan_v, svext_s32(svdup_s32(0), local_scan_v, 16-1));
            local_scan_v = svadd_s32_m(p_bin, local_scan_v, svext_s32(svdup_s32(0), local_scan_v, 16-2));
            local_scan_v = svadd_s32_m(p_bin, local_scan_v, svext_s32(svdup_s32(0), local_scan_v, 16-4));
            local_scan_v = svadd_s32_m(p_bin, local_scan_v, svext_s32(svdup_s32(0), local_scan_v, 16-8));
            local_scan_v = svadd_s32_m(p_bin, local_scan_v, svext_s32(svdup_s32(0), local_scan_v, 16-16));
            
            local_scan_v   = svadd_n_s32_x(p_bin, local_scan_v, current_offset);

            // 2. Store the final calculated offset results.
            //    Note: We start storing from the (binID + 1)-th position of m_bin_offsets.
            svst1_s32(p_bin, &ptile.m_bin_offsets[binID + 1], local_scan_v);
            // 2.2 Store the final m_slot_offsets results.
            svst1_s32(p_bin, &ptile.m_slot_offsets[binID + 1], local_scan_v-gap_v);
            
            // 3. Update the global offset for the next block.
            //    Use svaddv to efficiently calculate the sum of all block_sizes in the current chunk.
            current_offset += svaddv_s32(p_bin, block_size_v);
        }

        // Stage 3.3: Update m_capacity and initialize m_local_index.
        ptile.m_capacity=ptile.m_bin_offsets[numcell];
        ptile.m_local_index_hbm.clear();
        ptile.m_local_index_hbm.assign(ptile.m_capacity, INVALID_PARTICLE_ID);
        // Initialize with the starting position of each bin.
        std::vector<int> next_write_slot = ptile.m_bin_offsets; 

        // Stage 3.4: Populate m_local_index and m_pid_to_index_map with data.
        for(int ip=0;ip<np_to_deposit;ip++){
            int bin_id =ptile.m_pid_to_bin_map_hbm[ip];
            int insert_idx = next_write_slot[bin_id];
            ptile.m_local_index_hbm[insert_idx] = ip;
            ptile.m_pid_to_index_map[ip]=insert_idx;
            next_write_slot[bin_id]++;
        }
    }
}

template <int depos_order>
__arm_new("za") inline void sve_rhocell_sme_order3_kernel_sort(
    OPM_Inner_Vector_Real& xrhocells,
    OPM_Inner_Vector_Real& yrhocells,
    OPM_Inner_Vector_Real& zrhocells,
    const OPM_Inner_Vector_Real& sx_m,
    const OPM_Inner_Vector_Real& sy_m,
    const OPM_Inner_Vector_Real& sz_m,
    const OPM_Inner_Vector_Real& wqx,
    const OPM_Inner_Vector_Real& wqy,
    const OPM_Inner_Vector_Real& wqz,
    const OPM_Inner_Vector_Int& local_index,
    const std::vector<int>& bin_length,
    const std::vector<int>& bin_offsets,
    int lenx,
    int leny,
    int lenz,
    int rhonx,
    int rhonxy,
    int lenxy,
    long np_to_deposit
) __arm_streaming {
    constexpr int block_size = 8;
    svbool_t p_true = svwhilelt_b64(0, 8);
    svbool_t p_0_7 = svwhilelt_b64(0, 8);
    svbool_t p_0_3 = svwhilelt_b64(0, 4);
    MVec vzero(0);

    int np = static_cast<int>(np_to_deposit);

    const svuint64_t sm_index = svindex_u64(0, np);

    for (int iz = 0; iz < lenz-2; ++iz) {
        for (int iy = 0; iy < leny-2; ++iy) {       
            for (int ix = 0; ix < lenx-2; ix++) {
                int old_bin=(ix) + (iy) * lenx + (iz) * lenxy;
                int rho_bin=ix+iy*rhonx+iz*rhonxy;
                
                int binlength = bin_length[old_bin];
                int binoffset = bin_offsets[old_bin];
                
                svzero_za();
                for(int i=0;i<binlength;++i){
                    int ip = local_index[binoffset+i];

                    MVec sx_v = svld1_gather_index(p_0_3, &sx_m[ip], sm_index);
                    MVec sx_y0_v = sx_v * sy_m[0 * np + ip];
                    MVec sx_y1_v = sx_v * sy_m[1 * np + ip];
                    MVec sx_y_01_v = svsplice(p_0_3, sx_y0_v, sx_y1_v);
                    MVec sz_v = svld1_gather_index(p_0_3, &sz_m[ip], sm_index);
                    
                    MVec wqxsz_v = sz_v*wqx[ip];
                    svmopa_za64_m(0, p_0_3, p_0_7, wqxsz_v, sx_y_01_v);

                    MVec sx_y2_v = sx_v * sy_m[2 * np + ip];
                    MVec sx_y3_v = sx_v * sy_m[3 * np + ip];
                    MVec sx_y_23_v = svsplice(p_0_3, sx_y2_v, sx_y3_v);
                    svmopa_za64_m(1, p_0_3, p_0_7, wqxsz_v, sx_y_23_v);

                    MVec wqysz_v = sz_v*wqy[ip];
                    svmopa_za64_m(2, p_0_3, p_0_7, wqysz_v, sx_y_01_v);
                    svmopa_za64_m(3, p_0_3, p_0_7, wqysz_v, sx_y_23_v);

                    MVec wqzsz_v = sz_v*wqy[ip];
                    svmopa_za64_m(4, p_0_3, p_0_7, wqzsz_v, sx_y_01_v);
                    svmopa_za64_m(5, p_0_3, p_0_7, wqzsz_v, sx_y_23_v); 
                }
                // Jx_arr
                int rho_offset_iy01_z0 = (rho_bin + 0 * rhonx + 0 * rhonxy) * 8;
                int rho_offset_iy01_z1 = (rho_bin + 0 * rhonx + 1 * rhonxy) * 8;
                int rho_offset_iy01_z2 = (rho_bin + 0 * rhonx + 2 * rhonxy) * 8;
                int rho_offset_iy01_z3 = (rho_bin + 0 * rhonx + 3 * rhonxy) * 8;

                MVec wqx_sx_y01_z0 = svread_hor_za64_m(vzero, p_0_7, 0, 0);
                MVec xrhocells_iy01_z0_v = MVec::Load(p_true, &xrhocells[rho_offset_iy01_z0]);
                xrhocells_iy01_z0_v += wqx_sx_y01_z0;
                xrhocells_iy01_z0_v.Store(p_true, &xrhocells[rho_offset_iy01_z0]);

                MVec wqx_sx_y01_z1 = svread_hor_za64_m(vzero, p_0_7, 0, 1);
                MVec xrhocells_iy01_z1_v = MVec::Load(p_true, &xrhocells[rho_offset_iy01_z1]);
                xrhocells_iy01_z1_v += wqx_sx_y01_z1;
                xrhocells_iy01_z1_v.Store(p_true, &xrhocells[rho_offset_iy01_z1]);

                MVec wqx_sx_y01_z2 = svread_hor_za64_m(vzero, p_0_7, 0, 2);
                MVec xrhocells_iy01_z2_v = MVec::Load(p_true, &xrhocells[rho_offset_iy01_z2]);
                xrhocells_iy01_z2_v += wqx_sx_y01_z2;
                xrhocells_iy01_z2_v.Store(p_true, &xrhocells[rho_offset_iy01_z2]);

                MVec wqx_sx_y01_z3 = svread_hor_za64_m(vzero, p_0_7, 0, 3);
                MVec xrhocells_iy01_z3_v = MVec::Load(p_true, &xrhocells[rho_offset_iy01_z3]);
                xrhocells_iy01_z3_v += wqx_sx_y01_z3;
                xrhocells_iy01_z3_v.Store(p_true, &xrhocells[rho_offset_iy01_z3]);

                int rho_offset_iy23_z0 = (rho_bin + 2 * rhonx + 0 * rhonxy) * 8;
                int rho_offset_iy23_z1 = (rho_bin + 2 * rhonx + 1 * rhonxy) * 8;
                int rho_offset_iy23_z2 = (rho_bin + 2 * rhonx + 2 * rhonxy) * 8;
                int rho_offset_iy23_z3 = (rho_bin + 2 * rhonx + 3 * rhonxy) * 8;

                MVec wqx_sx_y23_z0 = svread_hor_za64_m(vzero, p_0_7, 1, 0);
                MVec xrhocells_iy23_z0_v = MVec::Load(p_true, &xrhocells[rho_offset_iy23_z0]);
                xrhocells_iy23_z0_v += wqx_sx_y23_z0;
                xrhocells_iy23_z0_v.Store(p_true, &xrhocells[rho_offset_iy23_z0]);

                MVec wqx_sx_y23_z1 = svread_hor_za64_m(vzero, p_0_7, 1, 1);
                MVec xrhocells_iy23_z1_v = MVec::Load(p_true, &xrhocells[rho_offset_iy23_z1]);
                xrhocells_iy23_z1_v += wqx_sx_y23_z1;
                xrhocells_iy23_z1_v.Store(p_true, &xrhocells[rho_offset_iy23_z1]);

                MVec wqx_sx_y23_z2 = svread_hor_za64_m(vzero, p_0_7, 1, 2);
                MVec xrhocells_iy23_z2_v = MVec::Load(p_true, &xrhocells[rho_offset_iy23_z2]);
                xrhocells_iy23_z2_v += wqx_sx_y23_z2;
                xrhocells_iy23_z2_v.Store(p_true, &xrhocells[rho_offset_iy23_z2]);

                MVec wqx_sx_y23_z3 = svread_hor_za64_m(vzero, p_0_7, 1, 3);
                MVec xrhocells_iy23_z3_v = MVec::Load(p_true, &xrhocells[rho_offset_iy23_z3]);
                xrhocells_iy23_z3_v += wqx_sx_y23_z3;
                xrhocells_iy23_z3_v.Store(p_true, &xrhocells[rho_offset_iy23_z3]);
        
                // Jy_arr
                MVec wqy_sx_y01_z0 = svread_hor_za64_m(vzero, p_0_7, 2, 0);
                MVec yrhocells_iy01_z0_v = MVec::Load(p_true, &yrhocells[rho_offset_iy01_z0]);
                yrhocells_iy01_z0_v += wqy_sx_y01_z0;
                yrhocells_iy01_z0_v.Store(p_true, &yrhocells[rho_offset_iy01_z0]);

                MVec wqy_sx_y01_z1 = svread_hor_za64_m(vzero, p_0_7, 2, 1);
                MVec yrhocells_iy01_z1_v = MVec::Load(p_true, &yrhocells[rho_offset_iy01_z1]);
                yrhocells_iy01_z1_v += wqy_sx_y01_z1;
                yrhocells_iy01_z1_v.Store(p_true, &yrhocells[rho_offset_iy01_z1]);

                MVec wqy_sx_y01_z2 = svread_hor_za64_m(vzero, p_0_7, 2, 2);
                MVec yrhocells_iy01_z2_v = MVec::Load(p_true, &yrhocells[rho_offset_iy01_z2]);
                yrhocells_iy01_z2_v += wqy_sx_y01_z2;
                yrhocells_iy01_z2_v.Store(p_true, &yrhocells[rho_offset_iy01_z2]);

                MVec wqy_sx_y01_z3 = svread_hor_za64_m(vzero, p_0_7, 2, 3);
                MVec yrhocells_iy01_z3_v = MVec::Load(p_true, &yrhocells[rho_offset_iy01_z3]);
                yrhocells_iy01_z3_v += wqy_sx_y01_z3;
                yrhocells_iy01_z3_v.Store(p_true, &yrhocells[rho_offset_iy01_z3]);

                MVec wqy_sx_y23_z0 = svread_hor_za64_m(vzero, p_0_7, 3, 0);
                MVec yrhocells_iy23_z0_v = MVec::Load(p_true, &yrhocells[rho_offset_iy23_z0]);
                yrhocells_iy23_z0_v += wqy_sx_y23_z0;
                yrhocells_iy23_z0_v.Store(p_true, &yrhocells[rho_offset_iy23_z0]);

                MVec wqy_sx_y23_z1 = svread_hor_za64_m(vzero, p_0_7, 3, 1);
                MVec yrhocells_iy23_z1_v = MVec::Load(p_true, &yrhocells[rho_offset_iy23_z1]);
                yrhocells_iy23_z1_v += wqy_sx_y23_z1;
                yrhocells_iy23_z1_v.Store(p_true, &yrhocells[rho_offset_iy23_z1]);

                MVec wqy_sx_y23_z2 = svread_hor_za64_m(vzero, p_0_7, 3, 2);
                MVec yrhocells_iy23_z2_v = MVec::Load(p_true, &yrhocells[rho_offset_iy23_z2]);
                yrhocells_iy23_z2_v += wqy_sx_y23_z2;
                yrhocells_iy23_z2_v.Store(p_true, &yrhocells[rho_offset_iy23_z2]);

                MVec wqy_sx_y23_z3 = svread_hor_za64_m(vzero, p_0_7, 3, 3);
                MVec yrhocells_iy23_z3_v = MVec::Load(p_true, &yrhocells[rho_offset_iy23_z3]);
                yrhocells_iy23_z3_v += wqy_sx_y23_z3;
                yrhocells_iy23_z3_v.Store(p_true, &yrhocells[rho_offset_iy23_z3]);

                // Jz_arr
                MVec wqz_sx_y01_z0 = svread_hor_za64_m(vzero, p_0_7, 4, 0);
                MVec zrhocells_iy01_z0_v = MVec::Load(p_true, &zrhocells[rho_offset_iy01_z0]);
                zrhocells_iy01_z0_v += wqz_sx_y01_z0;
                zrhocells_iy01_z0_v.Store(p_true, &zrhocells[rho_offset_iy01_z0]);

                MVec wqz_sx_y01_z1 = svread_hor_za64_m(vzero, p_0_7, 4, 1);
                MVec zrhocells_iy01_z1_v = MVec::Load(p_true, &zrhocells[rho_offset_iy01_z1]);
                zrhocells_iy01_z1_v += wqz_sx_y01_z1;
                zrhocells_iy01_z1_v.Store(p_true, &zrhocells[rho_offset_iy01_z1]);

                MVec wqz_sx_y01_z2 = svread_hor_za64_m(vzero, p_0_7, 4, 2);
                MVec zrhocells_iy01_z2_v = MVec::Load(p_true, &zrhocells[rho_offset_iy01_z2]);
                zrhocells_iy01_z2_v += wqz_sx_y01_z2;
                zrhocells_iy01_z2_v.Store(p_true, &zrhocells[rho_offset_iy01_z2]);

                MVec wqz_sx_y01_z3 = svread_hor_za64_m(vzero, p_0_7, 4, 3);
                MVec zrhocells_iy01_z3_v = MVec::Load(p_true, &zrhocells[rho_offset_iy01_z3]);
                zrhocells_iy01_z3_v += wqz_sx_y01_z3;
                zrhocells_iy01_z3_v.Store(p_true, &zrhocells[rho_offset_iy01_z3]);

                MVec wqz_sx_y23_z0 = svread_hor_za64_m(vzero, p_0_7, 5, 0);
                MVec zrhocells_iy23_z0_v = MVec::Load(p_true, &zrhocells[rho_offset_iy23_z0]);
                zrhocells_iy23_z0_v += wqz_sx_y23_z0;
                zrhocells_iy23_z0_v.Store(p_true, &zrhocells[rho_offset_iy23_z0]);

                MVec wqz_sx_y23_z1 = svread_hor_za64_m(vzero, p_0_7, 5, 1);
                MVec zrhocells_iy23_z1_v = MVec::Load(p_true, &zrhocells[rho_offset_iy23_z1]);
                zrhocells_iy23_z1_v += wqz_sx_y23_z1;
                zrhocells_iy23_z1_v.Store(p_true, &zrhocells[rho_offset_iy23_z1]);

                MVec wqz_sx_y23_z2 = svread_hor_za64_m(vzero, p_0_7, 5, 2);
                MVec zrhocells_iy23_z2_v = MVec::Load(p_true, &zrhocells[rho_offset_iy23_z2]);
                zrhocells_iy23_z2_v += wqz_sx_y23_z2;
                zrhocells_iy23_z2_v.Store(p_true, &zrhocells[rho_offset_iy23_z2]);

                MVec wqz_sx_y23_z3 = svread_hor_za64_m(vzero, p_0_7, 5, 3);
                MVec zrhocells_iy23_z3_v = MVec::Load(p_true, &zrhocells[rho_offset_iy23_z3]);
                zrhocells_iy23_z3_v += wqz_sx_y23_z3;
                zrhocells_iy23_z3_v.Store(p_true, &zrhocells[rho_offset_iy23_z3]);
            }
        }
    }
}

template <int depos_order>
void doDepositionShapeN_sve_rhocell_sme_order3 (const GetParticlePosition<PIdx>& GetPosition,
                        const amrex::ParticleReal * const wp,
                        const amrex::ParticleReal * const uxp,
                        const amrex::ParticleReal * const uyp,
                        const amrex::ParticleReal * const uzp,
                        const int* ion_lev,
                        amrex::FArrayBox& jx_fab,
                        amrex::FArrayBox& jy_fab,
                        amrex::FArrayBox& jz_fab,
                        long np_to_deposit,
                        amrex::Real relative_time,
                        const amrex::XDim3 & dinv,
                        const amrex::XDim3 & xyzmin,
                        amrex::Dim3 len,
                        amrex::Real q,
                        [[maybe_unused]]int n_rz_azimuthal_modes,
                        OPM_Inner_Vector_Real& wqx,
                        OPM_Inner_Vector_Real& wqy,
                        OPM_Inner_Vector_Real& wqz,
                        OPM_Inner_Vector_Real& sx_m,
                        OPM_Inner_Vector_Real& sy_m,
                        OPM_Inner_Vector_Real& sz_m,
                        OPM_Inner_Vector_Real& xrhocells,
                        OPM_Inner_Vector_Real& yrhocells,
                        OPM_Inner_Vector_Real& zrhocells,
                        OPM_Inner_Vector_Int& newbin,
                        ParticleTileType& ptile, const amrex::Box& box
                    )
{

    using namespace amrex::literals;

    const bool do_ionization = ion_lev;

    const amrex::Real invvol = dinv.x*dinv.y*dinv.z;

    const amrex::Real clightsq = 1.0_rt/PhysConst::c/PhysConst::c;

    amrex::Array4<amrex::Real> const& jx_arr = jx_fab.array();
    amrex::Array4<amrex::Real> const& jy_arr = jy_fab.array();
    amrex::Array4<amrex::Real> const& jz_arr = jz_fab.array();
    
    constexpr int nshapes = 1 + 3;
    const int vl = svcntd();

    const amrex::ParticleReal* mx = GetPosition.m_x;
    const amrex::ParticleReal* my = GetPosition.m_y;
    const amrex::ParticleReal* mz = GetPosition.m_z;
    
    const int xjstride = jx_arr.jstride;
    const int xkstride = jx_arr.kstride;
    const int yjstride = jy_arr.jstride;
    const int ykstride = jy_arr.kstride;
    const int zjstride = jz_arr.jstride;
    const int zkstride = jz_arr.kstride;

    auto compute_shape_factor_sve_order3 = [](double* sx, Vec xmid, svbool_t p, long np) {
        intVec i_newv = svcvt_s64_f64_z(p, xmid);

        Vec j = svrintz_x(p, xmid);
        Vec xint = xmid - j;
        Vec one_minus_xint = 1.0 - xint;

        Vec sx0 = (1.0 / 6.0) * one_minus_xint * one_minus_xint * one_minus_xint;
        sx0.Store(p, &sx[0 * np]);
        Vec sx1 = (2.0 / 3.0) - xint * xint * (1.0 - xint * 0.5);
        sx1.Store(p, &sx[1 * np]);
        Vec sx2 = (2.0 / 3.0) - one_minus_xint * one_minus_xint * (1.0 - 0.5 *one_minus_xint);
        sx2.Store(p, &sx[2 * np]);
        Vec sx3 = (1.0 / 6.0) * xint * xint * xint;
        sx3.Store(p, &sx[3 * np]);

        return i_newv - 1;
    };

    std::vector<long> rho_bin(np_to_deposit);
    svbool_t p_ix = svwhilelt_b64(0, nshapes);

    const int lenx = len.x;
    const int leny = len.y;
    const int lenz = len.z;
    const int lenxy = lenx*leny;

    svbool_t p_true = svwhilelt_b64(0, 8);
    svbool_t p_0_3 = svwhilelt_b64(0, 4);

    for(long ip = 0; ip < np_to_deposit; ip += vl)
    {
        svbool_t p_ip = svwhilelt_b64(ip, np_to_deposit);
        
        Vec uxp_v = Vec::Load(p_ip, &uxp[ip]);
        Vec uyp_v = Vec::Load(p_ip, &uyp[ip]);
        Vec uzp_v = Vec::Load(p_ip, &uzp[ip]);
        Vec gaminv = 1._rt / (1._rt + uxp_v * uxp_v * clightsq
                                    + uyp_v * uyp_v * clightsq
                                    + uzp_v * uzp_v * clightsq).Sqrt();
        Vec vx_v = uxp_v * gaminv;
        Vec vy_v = uyp_v * gaminv;
        Vec vz_v = uzp_v * gaminv;

        Vec wp_v = Vec::Load(p_ip, &wp[ip]);
        
        Vec wq_v = q * wp_v;
        if (do_ionization) 
        {
            wq_v *= ion_lev[ip];
        }

        Vec wqx_v = wq_v * invvol * vx_v;
        Vec wqy_v = wq_v * invvol * vy_v;
        Vec wqz_v = wq_v * invvol * vz_v;

        wqx_v.Store(p_ip, &wqx[ip]);
        wqy_v.Store(p_ip, &wqy[ip]);
        wqz_v.Store(p_ip, &wqz[ip]);

        Vec xp = Vec::Load(p_ip, &mx[ip]);
        Vec yp = Vec::Load(p_ip, &my[ip]);
        Vec zp = Vec::Load(p_ip, &mz[ip]);

        Vec xmid = ((xp - xyzmin.x) + relative_time * vx_v) * dinv.x;
        Vec ymid = ((yp - xyzmin.y) + relative_time * vy_v) * dinv.y;
        Vec zmid = ((zp - xyzmin.z) + relative_time * vz_v) * dinv.z;
        
        intVec j_cellv = compute_shape_factor_sve_order3(&sx_m[ip], xmid, p_ip, np_to_deposit);
        intVec k_cellv = compute_shape_factor_sve_order3(&sy_m[ip], ymid, p_ip, np_to_deposit);
        intVec l_cellv = compute_shape_factor_sve_order3(&sz_m[ip], zmid, p_ip, np_to_deposit);

        intVec rho_bin_v = j_cellv + k_cellv * lenx + l_cellv * lenxy;
        // rho_bin_v.Store(p_ip, &rho_bin[ip]);
        svst1w_s64(p_ip, &newbin[ip], rho_bin_v);
    }

    const int numcell=ptile.m_num_bins;
    increment_sort_particles_order3_OPM(np_to_deposit,ptile,newbin,numcell,len);

    auto& local_index = ptile.m_local_index_hbm;
    auto& bin_offsets = ptile.m_bin_offsets;
    auto& bin_length = ptile.m_bin_lengths;
    int rhonx = lenx-3;
    int rhonxy = rhonx*(leny);
    
    sve_rhocell_sme_order3_kernel_sort<3>(
        xrhocells, yrhocells, zrhocells, 
        sx_m, sy_m, sz_m,
        wqx, wqy, wqz, 
        local_index,bin_length,bin_offsets,
        lenx, leny, lenz,
        rhonx, rhonxy, lenxy, 
        np_to_deposit);

    
    for (int iz = 0; iz <= len.z; ++iz)
    {
        for (int iy = 0; iy <= len.y - 1; ++iy)
        {
            for(int ix = 0; ix <= len.x - 4; ix++){
                int arr_offset = ix + iy * xjstride + iz * xkstride;
                int rho_offset = 8 * (ix + iy * rhonx + iz * rhonxy);

                Vec xrhocells_0_3_v = Vec::Load(p_0_3, &xrhocells[0 + rho_offset]);
                Vec xrhocells_4_7_v = Vec::Load(p_0_3, &xrhocells[4 + rho_offset]);
                Vec jx_arr_0_3_v = Vec::Load(p_0_3, &jx_arr.p[0 + arr_offset]);
                Vec jx_arr_4_7_v = Vec::Load(p_0_3, &jx_arr.p[xjstride + arr_offset]);

                Vec yrhocells_0_3_v = Vec::Load(p_0_3, &yrhocells[0 + rho_offset]);
                Vec yrhocells_4_7_v = Vec::Load(p_0_3, &yrhocells[4 + rho_offset]);
                Vec jy_arr_0_3_v = Vec::Load(p_0_3, &jy_arr.p[0 + arr_offset]);
                Vec jy_arr_4_7_v = Vec::Load(p_0_3, &jy_arr.p[xjstride + arr_offset]);
                
                Vec zrhocells_0_3_v = Vec::Load(p_0_3, &zrhocells[0 + rho_offset]);
                Vec zrhocells_4_7_v = Vec::Load(p_0_3, &zrhocells[4 + rho_offset]);
                Vec jz_arr_0_3_v = Vec::Load(p_0_3, &jz_arr.p[0 + arr_offset]);
                Vec jz_arr_4_7_v = Vec::Load(p_0_3, &jz_arr.p[xjstride + arr_offset]);

                jx_arr_0_3_v += xrhocells_0_3_v;
                jy_arr_0_3_v += yrhocells_0_3_v;
                jz_arr_0_3_v += zrhocells_0_3_v;
                jx_arr_4_7_v += xrhocells_4_7_v;
                jy_arr_4_7_v += yrhocells_4_7_v;
                jz_arr_4_7_v += zrhocells_4_7_v;

                jx_arr_0_3_v.Store(p_0_3, &jx_arr.p[0 + arr_offset]);
                jy_arr_0_3_v.Store(p_0_3, &jy_arr.p[0 + arr_offset]);
                jz_arr_0_3_v.Store(p_0_3, &jz_arr.p[0 + arr_offset]);

                jx_arr_4_7_v.Store(p_0_3, &jx_arr.p[xjstride + arr_offset]);
                jy_arr_4_7_v.Store(p_0_3, &jy_arr.p[xjstride + arr_offset]);
                jz_arr_4_7_v.Store(p_0_3, &jz_arr.p[xjstride + arr_offset]);
            }
        }
    }
}

template <int depos_order>
void doDepositionShapeN_sve_rhocell_order3_sort (const GetParticlePosition<PIdx>& GetPosition,
                                const amrex::ParticleReal * const uxp,
                                const amrex::ParticleReal * const uyp,
                                const amrex::ParticleReal * const uzp,
                                const int * ion_lev,
                                amrex::FArrayBox& jx_fab,
                                amrex::FArrayBox& jy_fab,
                                amrex::FArrayBox& jz_fab,
                                long np_to_deposit,
                                amrex::Real relative_time,
                                const amrex::XDim3& dinv,
                                const amrex::XDim3& xyzmin,
                                amrex::Dim3 len,
                                amrex::Real q,
                                [[maybe_unused]]int n_rz_azimuthal_modes,
                                OPM_Inner_Vector_Real& wqx,
                                OPM_Inner_Vector_Real& wqy,
                                OPM_Inner_Vector_Real& wqz,
                                OPM_Inner_Vector_Real& sx_m,
                                OPM_Inner_Vector_Real& sy_m,
                                OPM_Inner_Vector_Real& sz_m,
                                OPM_Inner_Vector_Real& xrhocells,
                                OPM_Inner_Vector_Real& yrhocells,
                                OPM_Inner_Vector_Real& zrhocells,
                                OPM_Inner_Vector_Int& newbin,
                                ParticleTileType& ptile, const amrex::Box& box)
{
#ifdef BREAKDOWN
    uint64_t precompute_area[3] = {0};
    uint64_t calculate_area[3] = {0};
    uint64_t reduce_area[3] = {0};
    uint64_t init_area[3] = {0};
    init_area[0] = rdtscv();
#endif  // BREAKDOWN
    using namespace amrex::literals;

    const bool do_ionization = ion_lev;

    const amrex::Real invvol = dinv.x*dinv.y*dinv.z;

    const amrex::Real clightsq = 1.0_rt/PhysConst::c/PhysConst::c;

    amrex::Array4<amrex::Real> const& jx_arr = jx_fab.array();
    amrex::Array4<amrex::Real> const& jy_arr = jy_fab.array();
    amrex::Array4<amrex::Real> const& jz_arr = jz_fab.array();
    
    constexpr int nshapes = 1 + 3;
    const int block_size = 8;
    long n_blocks = np_to_deposit / block_size;
    long remainder = np_to_deposit - n_blocks * block_size;
    const int vl = svcntd();

    const amrex::ParticleReal* mx = GetPosition.m_x;
    const amrex::ParticleReal* my = GetPosition.m_y;
    const amrex::ParticleReal* mz = GetPosition.m_z;
    
    // 预计算网格步长
    const int xjstride = jx_arr.jstride;
    const int xkstride = jx_arr.kstride;
    const int yjstride = jy_arr.jstride;
    const int ykstride = jy_arr.kstride;
    const int zjstride = jz_arr.jstride;
    const int zkstride = jz_arr.kstride;


    std::vector<long> rho_bin(np_to_deposit);

    svbool_t p_ix = svwhilelt_b64(0, nshapes);

    const int lenx = len.x - 3;
    const int leny = len.y;
    const int lenz = len.z + 1;
    const int nrho = lenx * leny * lenz;

    std::vector<amrex::Real> xrhocells(8 * nrho, 0.0);
    std::vector<amrex::Real> yrhocells(8 * nrho, 0.0);
    std::vector<amrex::Real> zrhocells(8 * nrho, 0.0);

    svbool_t p_true = svwhilelt_b64(0, 8);
    svbool_t p_0_3 = svwhilelt_b64(0, 4);

    int np = static_cast<int>(np_to_deposit);

    int sx_step[8] = {0, 1 * np, 2 * np, 3 * np, 0, 1 * np, 2 * np, 3 * np};
    int sy_step[8] = {0, 0, 0, 0, np, np, np, np};

#ifdef BREAKDOWN
    init_area[1] = rdtscv();
    init_area[2] += init_area[1] - init_area[0];
    precompute_area[0] = rdtscv();
#endif  // BREAKDOWN 

    auto compute_shape_factor_sve_order3 = [](double* sx, Vec xmid, svbool_t p, long np) {
        intVec i_newv = svcvt_s64_f64_z(p, xmid);

        Vec j = svrintz_x(p, xmid);
        Vec xint = xmid - j;
        Vec one_minus_xint = 1.0 - xint;

        Vec sx0 = (1.0 / 6.0) * one_minus_xint * one_minus_xint * one_minus_xint;
        sx0.Store(p, &sx[0 * np]);
        Vec sx1 = (2.0 / 3.0) - xint * xint * (1.0 - xint * 0.5);
        sx1.Store(p, &sx[1 * np]);
        Vec sx2 = (2.0 / 3.0) - one_minus_xint * one_minus_xint * (1.0 - 0.5 *one_minus_xint);
        sx2.Store(p, &sx[2 * np]);
        Vec sx3 = (1.0 / 6.0) * xint * xint * xint;
        sx3.Store(p, &sx[3 * np]);

        return i_newv - 1;
    };

    for(long ip = 0; ip < np_to_deposit; ip += vl){ 
        svbool_t p_ip = svwhilelt_b64(ip, np_to_deposit);
        
        Vec uxp_v = Vec::Load(p_ip, &uxp[ip]);
        Vec uyp_v = Vec::Load(p_ip, &uyp[ip]);
        Vec uzp_v = Vec::Load(p_ip, &uzp[ip]);
        Vec gaminv = 1._rt / (1._rt + uxp_v * uxp_v * clightsq
                                    + uyp_v * uyp_v * clightsq
                                    + uzp_v * uzp_v * clightsq).Sqrt();
        Vec vx_v = uxp_v * gaminv;
        Vec vy_v = uyp_v * gaminv;
        Vec vz_v = uzp_v * gaminv;

        Vec wp_v = Vec::Load(p_ip, &wp[ip]);
        
        Vec wq_v = q * wp_v;
        if (do_ionization) 
        {
            wq_v *= ion_lev[ip];
        }

        Vec wqx_v = wq_v * invvol * vx_v;
        Vec wqy_v = wq_v * invvol * vy_v;
        Vec wqz_v = wq_v * invvol * vz_v;

        wqx_v.Store(p_ip, &wqx[ip]);
        wqy_v.Store(p_ip, &wqy[ip]);
        wqz_v.Store(p_ip, &wqz[ip]);

        Vec xp = Vec::Load(p_ip, &mx[ip]);
        Vec yp = Vec::Load(p_ip, &my[ip]);
        Vec zp = Vec::Load(p_ip, &mz[ip]);

        Vec xmid = ((xp - xyzmin.x) + relative_time * vx_v) * dinv.x;
        Vec ymid = ((yp - xyzmin.y) + relative_time * vy_v) * dinv.y;
        Vec zmid = ((zp - xyzmin.z) + relative_time * vz_v) * dinv.z;
        
        intVec j_cellv = compute_shape_factor_sve_order3(&sx_m[ip], xmid, p_ip, np_to_deposit);
        intVec k_cellv = compute_shape_factor_sve_order3(&sy_m[ip], ymid, p_ip, np_to_deposit);
        intVec l_cellv = compute_shape_factor_sve_order3(&sz_m[ip], zmid, p_ip, np_to_deposit);

        intVec rho_bin_v = j_cellv + k_cellv * lenx + l_cellv * lenx * leny;
        // rho_bin_v.Store(p_ip, &rho_bin[ip]);
        svst1w_s64(p_ip, &new_bin[ip], rho_bin_v);
    }
#ifdef BREAKDOWN
    precompute_area[1] = rdtscv();
    precompute_area[2] += precompute_area[1] - precompute_area[0];
    calculate_area[0] = rdtscv();
#endif  // BREAKDOWN

    increment_sort_particles_order3_OPM(np_to_deposit,ptile,newbin,numcell,len);

    auto& local_index = ptile.m_local_index_hbm;
    auto& bin_offsets = ptile.m_bin_offsets;
    auto& bin_length = ptile.m_bin_lengths;
    int rhonx = len.x-3;
    int rhonxy = rhonx*(len.y);
    IntVec mx_index = svld1sw_s64(p_true, sx_step);
    IntVec my_index = svld1sw_s64(p_true, sy_step);

    for (int iz = 0; iz < len.z-2; ++iz) {
        for (int iy = 0; iy < len.y-2; ++iy) {
            for (int ix = 0; ix < len.x-2; ix++) {
                int old_bin=(ix) + (iy) * len.x + (iz) * len.x*len.y;
                int rho_bin=ix+iy*rhonx+iz*rhonxy;

                int binlength = bin_length[old_bin];
                int binoffset = bin_offsets[old_bin];

                Vec xrhocells_v00(0);
                Vec yrhocells_v00(0);
                Vec zrhocells_v00(0);

                Vec xrhocells_v02(0);
                Vec yrhocells_v02(0);
                Vec zrhocells_v02(0);

                Vec xrhocells_v10(0);
                Vec yrhocells_v10(0);
                Vec zrhocells_v10(0);

                Vec xrhocells_v12(0);
                Vec yrhocells_v12(0);
                Vec zrhocells_v12(0);

                Vec xrhocells_v20(0);
                Vec yrhocells_v20(0);
                Vec zrhocells_v20(0);
                
                Vec xrhocells_v22(0);
                Vec yrhocells_v22(0);
                Vec zrhocells_v22(0);

                Vec xrhocells_v30(0);
                Vec yrhocells_v30(0);
                Vec zrhocells_v30(0);

                Vec xrhocells_v32(0);
                Vec yrhocells_v32(0);
                Vec zrhocells_v32(0);

                for(int i=0; i<binlength; ++i) {
                    int iip = local_index[binoffset+i];
                    // iy={0, 2} iz=0
                    double sz_wqx_tmp0 = sz_m[0 * np_to_deposit + iip] * wqx[iip];
                    double sz_wqy_tmp0 = sz_m[0 * np_to_deposit + iip] * wqy[iip];
                    double sz_wqz_tmp0 = sz_m[0 * np_to_deposit + iip] * wqz[iip];

                    // int rho_offset00 = (rho_bin + 0 * len.x + 0 * len.x * len.y) * 8;

                    Vec sx_v00 = svldi_gather_s64index_f64(p_true, &sx_m[iip], mx_index);
                    Vec sy_v00 = svldi_gather_s64index_f64(p_true, &sy_m[0 * np_to_deposit + iip], my_index);
                    Vec sx_m_sy_v00 = sx_v00 * sy_v00;

                    xrhocells_v00 += sx_m_sy_v00 * sz_wqx_tmp0;
                    yrhocells_v00 += sx_m_sy_v00 * sz_wqy_tmp0;
                    zrhocells_v00 += sx_m_sy_v00 * sz_wqz_tmp0;

                    // int rho_offset32 = (rho_bin + 2 * len.x + 0 * len.x * len.y) * 8;

                    Vec sx_v02 = svldi_gather_s64index_f64(p_true, &sx_m[iip], mx_index);
                    Vec sy_v02 = svldi_gather_s64index_f64(p_true, &sy_m[2 * np_to_deposit + iip], my_index);
                    Vec sx_m_sy_v02 = sx_v02 * sy_v02;

                    xrhocells_v02 += sx_m_sy_v02 * sz_wqx_tmp0;
                    yrhocells_v02 += sx_m_sy_v02 * sz_wqy_tmp0;
                    zrhocells_v02 += sx_m_sy_v02 * sz_wqz_tmp0;
                    
                    // iy={0, 2} iz=1
                    double sz_wqx_tmp1 = sz_m[1 * np_to_deposit + iip] * wqx[iip];
                    double sz_wqy_tmp1 = sz_m[1 * np_to_deposit + iip] * wqy[iip];
                    double sz_wqz_tmp1 = sz_m[1 * np_to_deposit + iip] * wqz[iip];

                    // int rho_offset10 = (rho_bin + 0 * len.x + 1 * len.x * len.y) * 8;

                    Vec sx_v10 = svldi_gather_s64index_f64(p_true, &sx_m[iip], mx_index);
                    Vec sy_v10 = svldi_gather_s64index_f64(p_true, &sy_m[0 * np_to_deposit + iip], my_index);
                    Vec sx_m_sy_v10 = sx_v10 * sy_v10;

                    xrhocells_v10 += sx_m_sy_v10 * sz_wqx_tmp1;
                    yrhocells_v10 += sx_m_sy_v10 * sz_wqy_tmp1;
                    zrhocells_v10 += sx_m_sy_v10 * sz_wqz_tmp1;

                    // int rho_offset12 = (rho_bin + 2 * len.x + 1 * len.x * len.y) * 8;

                    Vec sx_v12 = svldi_gather_s64index_f64(p_true, &sx_m[iip], mx_index);
                    Vec sy_v12 = svldi_gather_s64index_f64(p_true, &sy_m[2 * np_to_deposit + iip], my_index);
                    Vec sx_m_sy_v12 = sx_v12 * sy_v12;

                    xrhocells_v12 += sx_m_sy_v12 * sz_wqx_tmp1;
                    yrhocells_v12 += sx_m_sy_v12 * sz_wqy_tmp1;
                    zrhocells_v12 += sx_m_sy_v12 * sz_wqz_tmp1;
                    
                    // iy={0, 2} iz=2
                    double sz_wqx_tmp2 = sz_m[2 * np_to_deposit + iip] * wqx[iip];
                    double sz_wqy_tmp2 = sz_m[2 * np_to_deposit + iip] * wqy[iip];
                    double sz_wqz_tmp2 = sz_m[2 * np_to_deposit + iip] * wqz[iip];

                    // int rho_offset20 = (rho_bin + 0 * len.x + 2 * len.x * len.y) * 8;

                    Vec sx_v20 = svldi_gather_s64index_f64(p_true, &sx_m[iip], mx_index);
                    Vec sy_v20 = svldi_gather_s64index_f64(p_true, &sy_m[0 * np_to_deposit + iip], my_index);
                    Vec sx_m_sy_v20 = sx_v20 * sy_v20;

                    xrhocells_v20 += sx_m_sy_v20 * sz_wqx_tmp2;
                    yrhocells_v20 += sx_m_sy_v20 * sz_wqy_tmp2;
                    zrhocells_v20 += sx_m_sy_v20 * sz_wqz_tmp2;

                    // int rho_offset22 = (rho_bin + 2 * len.x + 2 * len.x * len.y) * 8;
                    
                    Vec sx_v22 = svldi_gather_s64index_f64(p_true, &sx_m[iip], mx_index);
                    Vec sy_v22 = svldi_gather_s64index_f64(p_true, &sy_m[2 * np_to_deposit + iip], my_index);
                    Vec sx_m_sy_v22 = sx_v22 * sy_v22;

                    xrhocells_v22 += sx_m_sy_v22 * sz_wqx_tmp2;
                    yrhocells_v22 += sx_m_sy_v22 * sz_wqy_tmp2;
                    zrhocells_v22 += sx_m_sy_v22 * sz_wqz_tmp2;

                    // iy={0, 2} iz=3
                    double sz_wqx_tmp3 = sz_m[3 * np_to_deposit + iip] * wqx[iip];
                    double sz_wqy_tmp3 = sz_m[3 * np_to_deposit + iip] * wqy[iip];
                    double sz_wqz_tmp3 = sz_m[3 * np_to_deposit + iip] * wqz[iip];
                    
                    // int rho_offset30 = (rho_bin + 0 * len.x + 3 * len.x * len.y) * 8;

                    Vec sx_v30 = svldi_gather_s64index_f64(p_true, &sx_m[iip], mx_index);
                    Vec sy_v30 = svldi_gather_s64index_f64(p_true, &sy_m[0 * np_to_deposit + iip], my_index);
                    Vec sx_m_sy_v30 = sx_v30 * sy_v30;

                    xrhocells_v30 += sx_m_sy_v30 * sz_wqx_tmp3;
                    yrhocells_v30 += sx_m_sy_v30 * sz_wqy_tmp3;
                    zrhocells_v30 += sx_m_sy_v30 * sz_wqz_tmp3;
                    
                    // int rho_offset32 = (rho_bin + 2 * len.x + 3 * len.x * len.y) * 8;

                    Vec sx_v32 = svldi_gather_s64index_f64(p_true, &sx_m[iip], mx_index);
                    Vec sy_v32 = svldi_gather_s64index_f64(p_true, &sy_m[2 * np_to_deposit + iip], my_index);
                    Vec sx_m_sy_v32 = sx_v32 * sy_v32;
                    
                    xrhocells_v32 += sx_m_sy_v32 * sz_wqx_tmp3;
                    yrhocells_v32 += sx_m_sy_v32 * sz_wqy_tmp3;
                    zrhocells_v32 += sx_m_sy_v32 * sz_wqz_tmp3;
                }
                
                int rho_offset00 = (rho_bin + 0 * rhonx + 0 * rhonxy) * 8;
                int rho_offset02 = (rho_bin + 2 * rhonx + 0 * rhonxy) * 8;
                int rho_offset10 = (rho_bin + 0 * rhonx + 1 * rhonxy) * 8;
                int rho_offset12 = (rho_bin + 2 * rhonx + 1 * rhonxy) * 8;
                int rho_offset20 = (rho_bin + 0 * rhonx + 2 * rhonxy) * 8;
                int rho_offset22 = (rho_bin + 2 * rhonx + 2 * rhonxy) * 8;
                int rho_offset30 = (rho_bin + 0 * rhonx + 3 * rhonxy) * 8;
                int rho_offset32 = (rho_bin + 2 * rhonx + 3 * rhonxy) * 8;
                // 00
                Vec xrhocells_00 = Vec::Load(p_true, &xrhocells[rho_offset00]);
                Vec yrhocells_00 = Vec::Load(p_true, &yrhocells[rho_offset00]);
                Vec zrhocells_00 = Vec::Load(p_true, &zrhocells[rho_offset00]);

                xrhocells_00 += xrhocells_v00;
                yrhocells_00 += yrhocells_v00;
                zrhocells_00 += zrhocells_v00;

                xrhocells_00.Store(p_true, &xrhocells[rho_offset00]);
                yrhocells_00.Store(p_true, &yrhocells[rho_offset00]);
                zrhocells_00.Store(p_true, &zrhocells[rho_offset00]);
                // 02
                Vec xrhocells_02 = Vec::Load(p_true, &xrhocells[rho_offset02]);
                Vec yrhocells_02 = Vec::Load(p_true, &yrhocells[rho_offset02]);
                Vec zrhocells_02 = Vec::Load(p_true, &zrhocells[rho_offset02]);

                xrhocells_02 += xrhocells_v02;
                yrhocells_02 += yrhocells_v02;
                zrhocells_02 += zrhocells_v02;

                xrhocells_02.Store(p_true, &xrhocells[rho_offset02]);
                yrhocells_02.Store(p_true, &yrhocells[rho_offset02]);
                zrhocells_02.Store(p_true, &zrhocells[rho_offset02]);
                // 10
                Vec xrhocells_10 = Vec::Load(p_true, &xrhocells[rho_offset10]);
                Vec yrhocells_10 = Vec::Load(p_true, &yrhocells[rho_offset10]);
                Vec zrhocells_10 = Vec::Load(p_true, &zrhocells[rho_offset10]);

                xrhocells_10 += xrhocells_v10;
                yrhocells_10 += yrhocells_v10;
                zrhocells_10 += zrhocells_v10;

                xrhocells_10.Store(p_true, &xrhocells[rho_offset10]);
                yrhocells_10.Store(p_true, &yrhocells[rho_offset10]);
                zrhocells_10.Store(p_true, &zrhocells[rho_offset10]);
                // 12
                Vec xrhocells_12 = Vec::Load(p_true, &xrhocells[rho_offset12]);
                Vec yrhocells_12 = Vec::Load(p_true, &yrhocells[rho_offset12]);
                Vec zrhocells_12 = Vec::Load(p_true, &zrhocells[rho_offset12]);

                xrhocells_12 += xrhocells_v12;
                yrhocells_12 += yrhocells_v12;
                zrhocells_12 += zrhocells_v12;

                xrhocells_12.Store(p_true, &xrhocells[rho_offset12]);
                yrhocells_12.Store(p_true, &yrhocells[rho_offset12]);
                zrhocells_12.Store(p_true, &zrhocells[rho_offset12]);
                // 20
                Vec xrhocells_20 = Vec::Load(p_true, &xrhocells[rho_offset20]);
                Vec yrhocells_20 = Vec::Load(p_true, &yrhocells[rho_offset20]);
                Vec zrhocells_20 = Vec::Load(p_true, &zrhocells[rho_offset20]);
                
                xrhocells_20 += xrhocells_v20;
                yrhocells_20 += yrhocells_v20;
                zrhocells_20 += zrhocells_v20;
                
                xrhocells_20.Store(p_true, &xrhocells[rho_offset20]);
                yrhocells_20.Store(p_true, &yrhocells[rho_offset20]);
                zrhocells_20.Store(p_true, &zrhocells[rho_offset20]);
                // 22
                Vec xrhocells_22 = Vec::Load(p_true, &xrhocells[rho_offset22]);
                Vec yrhocells_22 = Vec::Load(p_true, &yrhocells[rho_offset22]);
                Vec zrhocells_22 = Vec::Load(p_true, &zrhocells[rho_offset22]);

                xrhocells_22 += xrhocells_v22;
                yrhocells_22 += yrhocells_v22;
                zrhocells_22 += zrhocells_v22;

                xrhocells_22.Store(p_true, &xrhocells[rho_offset22]);
                yrhocells_22.Store(p_true, &yrhocells[rho_offset22]);
                zrhocells_22.Store(p_true, &zrhocells[rho_offset22]);
                // 30
                Vec xrhocells_30 = Vec::Load(p_true, &xrhocells[rho_offset30]);
                Vec yrhocells_30 = Vec::Load(p_true, &yrhocells[rho_offset30]);
                Vec zrhocells_30 = Vec::Load(p_true, &zrhocells[rho_offset30]);

                xrhocells_30 += xrhocells_v30;
                yrhocells_30 += yrhocells_v30;
                zrhocells_30 += zrhocells_v30;

                xrhocells_30.Store(p_true, &xrhocells[rho_offset30]);
                yrhocells_30.Store(p_true, &yrhocells[rho_offset30]);
                zrhocells_30.Store(p_true, &zrhocells[rho_offset30]);
                // 32
                Vec xrhocells_32 = Vec::Load(p_true, &xrhocells[rho_offset32]);
                Vec yrhocells_32 = Vec::Load(p_true, &yrhocells[rho_offset32]);
                Vec zrhocells_32 = Vec::Load(p_true, &zrhocells[rho_offset32]);

                xrhocells_32 += xrhocells_v32;
                yrhocells_32 += yrhocells_v32;
                zrhocells_32 += zrhocells_v32;

                xrhocells_32.Store(p_true, &xrhocells[rho_offset32]);
                yrhocells_32.Store(p_true, &yrhocells[rho_offset32]);
                zrhocells_32.Store(p_true, &zrhocells[rho_offset32]);
            }
        }
    }
#ifdef BREAKDOWN
    calculate_area[1] = rdtscv();
    calculate_area[2] += calculate_area[1] - calculate_area[0];
    reduce_area[0] = rdtscv();
#endif  // BREAKDOWN
    for (int iz = 0; iz <= len.z; ++iz)
    {
        for (int iy = 0; iy <= len.y - 1; ++iy)
        {
            for(int ix = 0; ix <= len.x - 4; ix++){
                int arr_offset = ix + iy * xjstride + iz * xkstride;
                int rho_offset = 8 * (ix + iy * lenx + iz * lenx * leny);

                Vec xrhocells_0_3_v = Vec::Load(p_0_3, &xrhocells[0 + rho_offset]);
                Vec xrhocells_4_7_v = Vec::Load(p_0_3, &xrhocells[4 + rho_offset]);
                Vec jx_arr_0_3_v = Vec::Load(p_0_3, &jx_arr.p[0 + arr_offset]);
                Vec jx_arr_4_7_v = Vec::Load(p_0_3, &jx_arr.p[xjstride + arr_offset]);

                Vec yrhocells_0_3_v = Vec::Load(p_0_3, &yrhocells[0 + rho_offset]);
                Vec yrhocells_4_7_v = Vec::Load(p_0_3, &yrhocells[4 + rho_offset]);
                Vec jy_arr_0_3_v = Vec::Load(p_0_3, &jy_arr.p[0 + arr_offset]);
                Vec jy_arr_4_7_v = Vec::Load(p_0_3, &jy_arr.p[xjstride + arr_offset]);
                
                Vec zrhocells_0_3_v = Vec::Load(p_0_3, &zrhocells[0 + rho_offset]);
                Vec zrhocells_4_7_v = Vec::Load(p_0_3, &zrhocells[4 + rho_offset]);
                Vec jz_arr_0_3_v = Vec::Load(p_0_3, &jz_arr.p[0 + arr_offset]);
                Vec jz_arr_4_7_v = Vec::Load(p_0_3, &jz_arr.p[xjstride + arr_offset]);

                jx_arr_0_3_v += xrhocells_0_3_v;
                jy_arr_0_3_v += yrhocells_0_3_v;
                jz_arr_0_3_v += zrhocells_0_3_v;
                jx_arr_4_7_v += xrhocells_4_7_v;
                jy_arr_4_7_v += yrhocells_4_7_v;
                jz_arr_4_7_v += zrhocells_4_7_v;

                jx_arr_0_3_v.Store(p_0_3, &jx_arr.p[0 + arr_offset]);
                jy_arr_0_3_v.Store(p_0_3, &jy_arr.p[0 + arr_offset]);
                jz_arr_0_3_v.Store(p_0_3, &jz_arr.p[0 + arr_offset]);

                jx_arr_4_7_v.Store(p_0_3, &jx_arr.p[xjstride + arr_offset]);
                jy_arr_4_7_v.Store(p_0_3, &jy_arr.p[xjstride + arr_offset]);
                jz_arr_4_7_v.Store(p_0_3, &jz_arr.p[xjstride + arr_offset]);
            }
        }
    }
#ifdef BREAKDOWN
    reduce_area[1] = rdtscv();
    reduce_area[2] += reduce_area[1] - reduce_area[0];

    printf("total: %13lu\n", precompute_area[2] + calculate_area[2] + reduce_area[2]);
    printf("precompute: %8lu\n", precompute_area[2]);
    printf("calculate: %9lu\n", calculate_area[2]);
    printf("reduce: %12lu\n", reduce_area[2]);
#endif  // BREAKDOWN
}

template <int depos_order>
void doDepositionShapeN_org_sp_GPMA_order3(const GetParticlePosition<PIdx> &GetPosition,
                                        const amrex::ParticleReal *const wp,
                                        const amrex::ParticleReal *const uxp,
                                        const amrex::ParticleReal *const uyp,
                                        const amrex::ParticleReal *const uzp,
                                        const int *ion_lev,
                                        amrex::FArrayBox &jx_fab,
                                        amrex::FArrayBox &jy_fab,
                                        amrex::FArrayBox &jz_fab,
                                        long np_to_deposit,
                                        amrex::Real relative_time,
                                        const amrex::XDim3 &dinv,
                                        const amrex::XDim3 &xyzmin,
                                        amrex::Dim3 lo,
                                        amrex::Dim3 hi,
                                        amrex::Dim3 len,
                                        amrex::Real q,
                                        WarpXParticleContainer::ParticleTileType &ptile, const amrex::Box &box,
                                        [[maybe_unused]] int n_rz_azimuthal_modes,
                                        std::vector<uint64_t>& inner_timer
                                    )
{
    using namespace amrex::literals;
    #ifdef BREAKDWON
    using namespace std::chrono;
    uint64_t total_time[3]={0};
    uint64_t precompute=0;
    uint64_t cal_time=0;
    uint64_t sort_time=0;
        uint64_t presort=0;
        uint64_t insert=0;
        uint64_t borrow=0;
        uint64_t rebuildtime=0;
    uint64_t reduce_time=0;
    uint64_t svetime[3]={0};
    uint64_t svetime1[3]={0};
 
    total_time[0]=rdtscv();
    #endif
    amrex::Array4<amrex::Real> const &jx_arr = jx_fab.array();
    amrex::Array4<amrex::Real> const &jy_arr = jy_fab.array();
    amrex::Array4<amrex::Real> const &jz_arr = jz_fab.array();

    const int xjstride = jx_arr.jstride;
    const int xkstride = jx_arr.kstride;
    const int yjstride = jy_arr.jstride;
    const int ykstride = jy_arr.kstride;
    const int zjstride = jz_arr.jstride;
    const int zkstride = jz_arr.kstride;

    long nx = xjstride;
    long nxy = xkstride;
    long moff[8] = {0, 1, nx, nx + 1, nxy, nxy + 1, nx + nxy, nx + nxy + 1};

    const bool do_ionization = ion_lev;

    const amrex::Real invvol = dinv.x * dinv.y * dinv.z;

    const amrex::Real clightsq = 1.0_rt / PhysConst::c / PhysConst::c;

    const long numcell = jx_fab.box().numPts();

    amrex::IntVect const jx_type = jx_fab.box().type();
    amrex::IntVect const jy_type = jy_fab.box().type();
    amrex::IntVect const jz_type = jz_fab.box().type();

    std::vector<amrex::Real> wqx(np_to_deposit, 0.0_rt), wqy(np_to_deposit, 0.0_rt), wqz(np_to_deposit, 0.0_rt);
    
    std::vector<amrex::Real> sx_m(4 * np_to_deposit, 0);
    std::vector<amrex::Real> sy_m(4 * np_to_deposit, 0);
    std::vector<amrex::Real> sz_m(4 * np_to_deposit, 0);
    alignas(64) std::vector<int> newbin(np_to_deposit,-1);
    long nnx = len.x;
    long nny = len.y;
    long nnxny = nnx*nny;
    #ifdef BREAKDWON
    svetime[0]=rdtscv();
    #endif
    #pragma omp simd
    for (int ip = 0; ip < np_to_deposit; ip++)
    {
        amrex::ParticleReal xp, yp, zp;
        GetPosition(ip, xp, yp, zp);

        // --- Get particle quantities
        const amrex::Real gaminv = 1.0_rt / std::sqrt(1.0_rt + uxp[ip] * uxp[ip] * clightsq + uyp[ip] * uyp[ip] * clightsq + uzp[ip] * uzp[ip] * clightsq);
        const amrex::Real vx = uxp[ip] * gaminv;
        const amrex::Real vy = uyp[ip] * gaminv;
        const amrex::Real vz = uzp[ip] * gaminv;

        amrex::Real wq = q * wp[ip];
        if (do_ionization)
        {
            wq *= ion_lev[ip];
        }

        constexpr int NODE = amrex::IndexType::NODE;
        constexpr int CELL = amrex::IndexType::CELL;

        // wqx, wqy wqz are particle current in each direction
        wqx[ip] = wq * invvol * vx;
        wqy[ip] = wq * invvol * vy;
        wqz[ip] = wq * invvol * vz;

        // --- Compute shape factors
        Compute_shape_factor<depos_order> const compute_shape_factor;

        // const auto j = static_cast<int>(xmid);
        // x direction
        // Get particle position after 1/2 push back in position
        // Keep these double to avoid bug in single precision

        const double xmid = ((xp - xyzmin.x) + relative_time * vx) * dinv.x;

        double sx_node[depos_order + 1] = {0.};
        double sx_cell[depos_order + 1] = {0.};
        int j_node = 0;
        int j_cell = 0;
        if (jx_type[0] == NODE || jy_type[0] == NODE || jz_type[0] == NODE)
        {
            j_node = compute_shape_factor(sx_node, xmid);
        }
        if (jx_type[0] == CELL || jy_type[0] == CELL || jz_type[0] == CELL)
        {
            j_cell = compute_shape_factor(sx_cell, xmid - 0.5);
        }

        for (int ix = 0; ix <= depos_order; ix++)
        {
            sx_m[ip * 4 + ix] = ((jx_type[0] == NODE) ? amrex::Real(sx_node[ix]) : amrex::Real(sx_cell[ix]));
        }

        int jm = ((jz_type[0] == NODE) ? j_node : j_cell);

        // y direction
        // Keep these double to avoid bug in single precision
        const double ymid = ((yp - xyzmin.y) + relative_time * vy) * dinv.y;
        double sy_node[depos_order + 1] = {0.};
        double sy_cell[depos_order + 1] = {0.};
        int k_node = 0;
        int k_cell = 0;
        if (jx_type[1] == NODE || jy_type[1] == NODE || jz_type[1] == NODE)
        {
            k_node = compute_shape_factor(sy_node, ymid);
        }
        if (jx_type[1] == CELL || jy_type[1] == CELL || jz_type[1] == CELL)
        {
            k_cell = compute_shape_factor(sy_cell, ymid - 0.5);
        }
        for (int iy = 0; iy <= depos_order; iy++)
        {
            sy_m[ip * 4 + iy] = ((jx_type[1] == NODE) ? amrex::Real(sy_node[iy]) : amrex::Real(sy_cell[iy]));
        }
        int km = ((jx_type[1] == NODE) ? k_node : k_cell);

        // z direction
        // Keep these double to avoid bug in single precision
        constexpr int zdir = WARPX_ZINDEX;
        const double zmid = ((zp - xyzmin.z) + relative_time * vz) * dinv.z;
        double sz_node[depos_order + 1] = {0.};
        double sz_cell[depos_order + 1] = {0.};
        int l_node = 0;
        int l_cell = 0;
        if (jx_type[zdir] == NODE || jy_type[zdir] == NODE || jz_type[zdir] == NODE)
        {
            l_node = compute_shape_factor(sz_node, zmid);
        }
        if (jx_type[zdir] == CELL || jy_type[zdir] == CELL || jz_type[zdir] == CELL)
        {
            l_cell = compute_shape_factor(sz_cell, zmid - 0.5);
        }
        for (int iz = 0; iz <= depos_order; iz++)
        {
            sz_m[ip * 4 + iz] = ((jx_type[zdir] == NODE) ? amrex::Real(sz_node[iz]) : amrex::Real(sz_cell[iz]));
        }
        int lm = ((jz_type[zdir] == NODE) ? l_node : l_cell);
        newbin[ip]=jm+km*nnx+lm*nnxny;
    }
    #ifdef BREAKDWON
    svetime[1]=rdtscv();
    precompute+=(svetime[1]-svetime[0]); 
    svetime[0]=rdtscv();
    #endif

    increment_sort_particles(np_to_deposit, ptile,newbin, numcell);

    #ifdef BREAKDWON
    svetime[1]=rdtscv();
    sort_time+=(svetime[1]-svetime[0]); 
    #endif
    
    // For 3rd order deposition, we need temporary storage for a 4x4x4 = 64 grid point stencil.
    double xtemp[64] = {0.0};
    double ytemp[64] = {0.0};
    double ztemp[64] = {0.0};

    #ifdef BREAKDOWN // Note: Corrected typo from BREAKDWON to BREAKDOWN
    svetime[0] = rdtscv();
    #endif

    // The order of deposition is 3 for this version.
    // const int depos_order = 3; 

    // Loop over all bins in the tile
    #pragma omp unroll 
    for (int iz = 0; iz < len.z; ++iz) {
        for (int iy = 0; iy < len.y; ++iy) {
            for (int ix = 0; ix < len.x; ix++) {
                int old_bin = (ix) + (iy) * nnx + (iz) * nnxny;
                int binlength = ptile.m_bin_lengths[old_bin];
                int binoffset = ptile.m_bin_offsets[old_bin];
                
                // It's crucial to clear the temp arrays for each bin to avoid accumulating values from previous bins.
                // If these arrays were declared inside this loop, this would not be necessary.
                std::fill_n(xtemp, 64, 0.0);
                std::fill_n(ytemp, 64, 0.0);
                std::fill_n(ztemp, 64, 0.0);

                // --- Stage 1: Accumulate currents from all particles in the bin into the temp arrays ---
                #pragma omp simd
                for (int i = 0; i < binlength; i++) {
                    int ip = ptile.m_local_index[binoffset + i];
                    // Loop over the 4x4x4 stencil for each particle
                    #pragma omp simd
                    for (int iz_s = 0; iz_s <= depos_order; iz_s++) {
                        #pragma omp simd
                        for (int iy_s = 0; iy_s <= depos_order; iy_s++) {
                            #pragma omp simd
                            for (int ix_s = 0; ix_s <= depos_order; ix_s++) {
                                // Flattened index for the 4x4x4 stencil (64 points)
                                int idx = ix_s + iy_s * 4 + iz_s * 16;
                                
                                // For 3rd order, each particle has 4 shape function values per dimension.
                                // The stride is now 4*ip.
                                double sxsysz = sx_m[ix_s + 4 * ip] * sy_m[iy_s + 4 * ip] * sz_m[iz_s + 4 * ip];
                                
                                double jx = sxsysz * wqx[ip];
                                xtemp[idx] += jx;
                                
                                double jy = sxsysz * wqy[ip];
                                ytemp[idx] += jy;
                                
                                double jz = sxsysz * wqz[ip];
                                // Original code had a bug here: ztemp[idx] += jy. Corrected to jz.
                                ztemp[idx] += jz;
                            }
                        }
                    }
                }

                // --- Stage 2: Reduce the accumulated values from temp arrays to the global grid ---
                #pragma omp simd
                for (int iz_s = 0; iz_s <= depos_order; iz_s++) {
                    #pragma omp simd
                    for (int iy_s = 0; iy_s <= depos_order; iy_s++) {
                        #pragma omp simd
                        for (int ix_s = 0; ix_s <= depos_order; ix_s++) {
                            int jxoffset = ix_s + iy_s * xjstride + iz_s * xkstride;
                            int xidx = ix + iy * nnx + iz * nnxny; // Base index of the bin corner
                            int idx = ix_s + iy_s * 4 + iz_s * 16;
                            
                            // Atomically add the temp value to the corresponding global grid point.
                            // Note: Using direct addition assumes no race conditions. For multi-threading,
                            // this part might require atomic operations if different threads can write to the same location.
                            jx_arr.p[jxoffset + xidx] += xtemp[idx];
                            jy_arr.p[jxoffset + xidx] += ytemp[idx];
                            jz_arr.p[jxoffset + xidx] += ztemp[idx];
                        }
                    }
                }
            }
        }
    }
    
    #ifdef BREAKDWON
    svetime[1]=rdtscv();
    cal_time+=(svetime[1]-svetime[0]); 

    total_time[1]=rdtscv();
    total_time[2]+=(total_time[1]-total_time[0]);
    inner_timer[0]=cal_time+precompute+sort_time;
    inner_timer[1]=precompute;
    inner_timer[2]=cal_time;
    inner_timer[3]=sort_time;
    #endif
}


// =================END Functions================












/**
 * \brief Direct current deposition for thread thread_num for the implicit scheme
 *        The only difference from doDepositionShapeN is in how the particle gamma
 *        is calculated.
 * \tparam depos_order deposition order
 * \param GetPosition  A functor for returning the particle position.
 * \param wp           Pointer to array of particle weights.
 * \param uxp_n,uyp_n,uzp_n  Pointer to arrays of particle momentum at time n.
 * \param uxp,uyp,uzp  Pointer to arrays of particle momentum at time n+1/2.
 * \param ion_lev      Pointer to array of particle ionization level. This is
                         required to have the charge of each macroparticle
                         since q is a scalar. For non-ionizable species,
                         ion_lev is a null pointer.
 * \param jx_fab,jy_fab,jz_fab FArrayBox of current density, either full array or tile.
 * \param np_to_deposit Number of particles for which current is deposited.
 * \param dinv         3D cell size inverse
 * \param xyzmin       Physical lower bounds of domain.
 * \param lo           Index lower bounds of domain.
 * \param q            species charge.
 * \param n_rz_azimuthal_modes Number of azimuthal modes when using RZ geometry.
 */
template <int depos_order>
void doDepositionShapeNImplicit(const GetParticlePosition<PIdx>& GetPosition,
                                const amrex::ParticleReal * const wp,
                                const amrex::ParticleReal * const uxp_n,
                                const amrex::ParticleReal * const uyp_n,
                                const amrex::ParticleReal * const uzp_n,
                                const amrex::ParticleReal * const uxp,
                                const amrex::ParticleReal * const uyp,
                                const amrex::ParticleReal * const uzp,
                                const int * const ion_lev,
                                amrex::FArrayBox& jx_fab,
                                amrex::FArrayBox& jy_fab,
                                amrex::FArrayBox& jz_fab,
                                const long np_to_deposit,
                                const amrex::XDim3 & dinv,
                                const amrex::XDim3 & xyzmin,
                                const amrex::Dim3 lo,
                                const amrex::Real q,
                                [[maybe_unused]]const int n_rz_azimuthal_modes)
{
    using namespace amrex::literals;

    // Whether ion_lev is a null pointer (do_ionization=0) or a real pointer
    // (do_ionization=1)
    const bool do_ionization = ion_lev;

    const amrex::Real invvol = dinv.x*dinv.y*dinv.z;

    amrex::Array4<amrex::Real> const& jx_arr = jx_fab.array();
    amrex::Array4<amrex::Real> const& jy_arr = jy_fab.array();
    amrex::Array4<amrex::Real> const& jz_arr = jz_fab.array();
    amrex::IntVect const jx_type = jx_fab.box().type();
    amrex::IntVect const jy_type = jy_fab.box().type();
    amrex::IntVect const jz_type = jz_fab.box().type();

    // Loop over particles and deposit into jx_fab, jy_fab and jz_fab
    amrex::ParallelFor(
            np_to_deposit,
            [=] AMREX_GPU_DEVICE (long ip) {
            amrex::ParticleReal xp, yp, zp;
            GetPosition(ip, xp, yp, zp);

            constexpr amrex::ParticleReal inv_c2 = 1._prt/(PhysConst::c*PhysConst::c);

            // Compute inverse Lorentz factor, the average of gamma at time levels n and n+1
            // The uxp,uyp,uzp are the velocities at time level n+1/2
            const amrex::ParticleReal uxp_np1 = 2._prt*uxp[ip] - uxp_n[ip];
            const amrex::ParticleReal uyp_np1 = 2._prt*uyp[ip] - uyp_n[ip];
            const amrex::ParticleReal uzp_np1 = 2._prt*uzp[ip] - uzp_n[ip];
            const amrex::ParticleReal gamma_n = std::sqrt(1._prt + (uxp_n[ip]*uxp_n[ip] + uyp_n[ip]*uyp_n[ip] + uzp_n[ip]*uzp_n[ip])*inv_c2);
            const amrex::ParticleReal gamma_np1 = std::sqrt(1._prt + (uxp_np1*uxp_np1 + uyp_np1*uyp_np1 + uzp_np1*uzp_np1)*inv_c2);
            const amrex::ParticleReal gaminv = 2.0_prt/(gamma_n + gamma_np1);

            const amrex::Real vx  = uxp[ip]*gaminv;
            const amrex::Real vy  = uyp[ip]*gaminv;
            const amrex::Real vz  = uzp[ip]*gaminv;

            amrex::Real wq  = q*wp[ip];
            if (do_ionization){
                wq *= ion_lev[ip];
            }

            const amrex::Real relative_time = 0._rt;
            doDepositionShapeNKernel<depos_order>(xp, yp, zp, wq, vx, vy, vz, jx_arr, jy_arr, jz_arr,
                                                  jx_type, jy_type, jz_type,
                                                  relative_time, dinv, xyzmin,
                                                  invvol, lo, n_rz_azimuthal_modes);

        }
    );
}

/**
 * \brief Current Deposition for thread thread_num using shared memory
 * \tparam depos_order deposition order
 * \param GetPosition  A functor for returning the particle position.
 * \param wp           Pointer to array of particle weights.
 * \param uxp,uyp,uzp  Pointer to arrays of particle momentum.
 * \param ion_lev      Pointer to array of particle ionization level. This is
                         required to have the charge of each macroparticle
                         since q is a scalar. For non-ionizable species,
                         ion_lev is a null pointer.
 * \param jx_fab,jy_fab,jz_fab FArrayBox of current density, either full array or tile.
 * \param np_to_deposit Number of particles for which current is deposited.
 * \param dt           Time step for particle level
 * \param relative_time Time at which to deposit J, relative to the time of the
 *                      current positions of the particles. When different than 0,
 *                      the particle position will be temporarily modified to match
 *                      the time of the deposition.
 * \param dinv         3D cell size inverse
 * \param xyzmin       Physical lower bounds of domain.
 * \param lo           Index lower bounds of domain.
 * \param q            species charge.
 * \param n_rz_azimuthal_modes Number of azimuthal modes when using RZ geometry.
 */
template <int depos_order>
void doDepositionSharedShapeN (const GetParticlePosition<PIdx>& GetPosition,
                               const amrex::ParticleReal * const wp,
                               const amrex::ParticleReal * const uxp,
                               const amrex::ParticleReal * const uyp,
                               const amrex::ParticleReal * const uzp,
                               const int*  ion_lev,
                               amrex::FArrayBox& jx_fab,
                               amrex::FArrayBox& jy_fab,
                               amrex::FArrayBox& jz_fab,
                               long np_to_deposit,
                               const amrex::Real relative_time,
                               const amrex::XDim3 & dinv,
                               const amrex::XDim3 & xyzmin,
                               amrex::Dim3 lo,
                               amrex::Real q,
                               int n_rz_azimuthal_modes,
                               const amrex::DenseBins<WarpXParticleContainer::ParticleTileType::ParticleTileDataType>& a_bins,
                               const amrex::Box& box,
                               const amrex::Geometry& geom,
                               const amrex::IntVect& a_tbox_max_size)
{
    using namespace amrex::literals;

#if defined(AMREX_USE_HIP) || defined(AMREX_USE_CUDA)
    using namespace amrex;

    auto permutation = a_bins.permutationPtr();

    amrex::Array4<amrex::Real> const& jx_arr = jx_fab.array();
    amrex::Array4<amrex::Real> const& jy_arr = jy_fab.array();
    amrex::Array4<amrex::Real> const& jz_arr = jz_fab.array();
    amrex::IntVect const jx_type = jx_fab.box().type();
    amrex::IntVect const jy_type = jy_fab.box().type();
    amrex::IntVect const jz_type = jz_fab.box().type();

    constexpr int zdir = WARPX_ZINDEX;
    constexpr int NODE = amrex::IndexType::NODE;
    constexpr int CELL = amrex::IndexType::CELL;

    // Loop over particles and deposit into jx_fab, jy_fab and jz_fab
    const auto dxiarr = geom.InvCellSizeArray();
    const auto plo = geom.ProbLoArray();
    const auto domain = geom.Domain();

    amrex::Box sample_tbox(IntVect(AMREX_D_DECL(0,0,0)), a_tbox_max_size - 1);
    sample_tbox.grow(depos_order);

    amrex::Box sample_tbox_x = convert(sample_tbox, jx_type);
    amrex::Box sample_tbox_y = convert(sample_tbox, jy_type);
    amrex::Box sample_tbox_z = convert(sample_tbox, jz_type);

    const auto npts = amrex::max(sample_tbox_x.numPts(), sample_tbox_y.numPts(), sample_tbox_z.numPts());

    const int nblocks = a_bins.numBins();
    const int threads_per_block = WarpX::shared_mem_current_tpb;
    const auto offsets_ptr = a_bins.offsetsPtr();

    const std::size_t shared_mem_bytes = npts*sizeof(amrex::Real);
    const amrex::IntVect bin_size = WarpX::shared_tilesize;
    const std::size_t max_shared_mem_bytes = amrex::Gpu::Device::sharedMemPerBlock();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(shared_mem_bytes <= max_shared_mem_bytes,
                                     "Tile size too big for GPU shared memory current deposition");

    amrex::ignore_unused(np_to_deposit);
    // Launch one thread-block per bin
    amrex::launch(
            nblocks, threads_per_block, shared_mem_bytes, amrex::Gpu::gpuStream(),
            [=] AMREX_GPU_DEVICE () noexcept {
        const int bin_id = blockIdx.x;
        const unsigned int bin_start = offsets_ptr[bin_id];
        const unsigned int bin_stop = offsets_ptr[bin_id+1];

        if (bin_start == bin_stop) { return; /*this bin has no particles*/ }

        // These boxes define the index space for the shared memory buffers
        amrex::Box buffer_box;
        {
            ParticleReal xp, yp, zp;
            GetPosition(permutation[bin_start], xp, yp, zp);
#if defined(WARPX_DIM_3D)
            IntVect iv = IntVect(int( amrex::Math::floor((xp-plo[0]) * dxiarr[0]) ),
                                 int( amrex::Math::floor((yp-plo[1]) * dxiarr[1]) ),
                                 int( amrex::Math::floor((zp-plo[2]) * dxiarr[2]) ));
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
            IntVect iv = IntVect(int( amrex::Math::floor((xp-plo[0]) * dxiarr[0]) ),
                                 int( amrex::Math::floor((zp-plo[1]) * dxiarr[1]) ));
#elif defined(WARPX_DIM_1D_Z)
            IntVect iv = IntVect(int( amrex::Math::floor((zp-plo[0]) * dxiarr[0]) ));
#endif
            iv += domain.smallEnd();
            getTileIndex(iv, box, true, bin_size, buffer_box);
        }

        buffer_box.grow(depos_order);
        Box tbox_x = convert(buffer_box, jx_type);
        Box tbox_y = convert(buffer_box, jy_type);
        Box tbox_z = convert(buffer_box, jz_type);

        Gpu::SharedMemory<amrex::Real> gsm;
        amrex::Real* const shared = gsm.dataPtr();

        amrex::Array4<amrex::Real> const jx_buff(shared,
                amrex::begin(tbox_x), amrex::end(tbox_x), 1);
        amrex::Array4<amrex::Real> const jy_buff(shared,
                amrex::begin(tbox_y), amrex::end(tbox_y), 1);
        amrex::Array4<amrex::Real> const jz_buff(shared,
                amrex::begin(tbox_z), amrex::end(tbox_z), 1);

        // Zero-initialize the temporary array in shared memory
        volatile amrex::Real* vs = shared;
        for (int i = threadIdx.x; i < npts; i += blockDim.x){
            vs[i] = 0.0;
        }
        __syncthreads();
        for (unsigned int ip_orig = bin_start+threadIdx.x; ip_orig<bin_stop; ip_orig += blockDim.x)
        {
            const unsigned int ip = permutation[ip_orig];
            depositComponent<depos_order>(GetPosition, wp, uxp, uyp, uzp, ion_lev, jx_buff, jx_type,
                                          relative_time, dinv, xyzmin, lo, q, n_rz_azimuthal_modes,
                                          ip, zdir, NODE, CELL, 0);
        }

        __syncthreads();
        addLocalToGlobal(tbox_x, jx_arr, jx_buff);
        for (int i = threadIdx.x; i < npts; i += blockDim.x){
            vs[i] = 0.0;
        }

        __syncthreads();
        for (unsigned int ip_orig = bin_start+threadIdx.x; ip_orig<bin_stop; ip_orig += blockDim.x)
        {
            const unsigned int ip = permutation[ip_orig];
            depositComponent<depos_order>(GetPosition, wp, uxp, uyp, uzp, ion_lev, jy_buff, jy_type,
                                          relative_time, dinv, xyzmin, lo, q, n_rz_azimuthal_modes,
                                          ip, zdir, NODE, CELL, 1);
        }

        __syncthreads();
        addLocalToGlobal(tbox_y, jy_arr, jy_buff);
        for (int i = threadIdx.x; i < npts; i += blockDim.x){
            vs[i] = 0.0;
        }

        __syncthreads();
        for (unsigned int ip_orig = bin_start+threadIdx.x; ip_orig<bin_stop; ip_orig += blockDim.x)
        {
            const unsigned int ip = permutation[ip_orig];
            depositComponent<depos_order>(GetPosition, wp, uxp, uyp, uzp, ion_lev, jz_buff, jz_type,
                                          relative_time, dinv, xyzmin, lo, q, n_rz_azimuthal_modes,
                                          ip, zdir, NODE, CELL, 2);
        }

        __syncthreads();
        addLocalToGlobal(tbox_z, jz_arr, jz_buff);
    });
#else // not using hip/cuda
    // Note, you should never reach this part of the code. This funcion cannot be called unless
    // using HIP/CUDA, and those things are checked prior
    //don't use any args
    ignore_unused(GetPosition, wp, uxp, uyp, uzp, ion_lev, jx_fab, jy_fab, jz_fab, np_to_deposit, relative_time, dinv, xyzmin, lo, q, n_rz_azimuthal_modes, a_bins, box, geom, a_tbox_max_size);
    WARPX_ABORT_WITH_MESSAGE("Shared memory only implemented for HIP/CUDA");
#endif
}

/**
 * \brief Esirkepov Current Deposition for thread thread_num
 *
 * \tparam depos_order  deposition order
 * \param GetPosition  A functor for returning the particle position.
 * \param wp           Pointer to array of particle weights.
 * \param uxp,uyp,uzp  Pointer to arrays of particle momentum.
 * \param ion_lev      Pointer to array of particle ionization level. This is
                       required to have the charge of each macroparticle
                       since q is a scalar. For non-ionizable species,
                       ion_lev is a null pointer.
 * \param Jx_arr,Jy_arr,Jz_arr Array4 of current density, either full array or tile.
 * \param np_to_deposit Number of particles for which current is deposited.
 * \param dt           Time step for particle level
 * \param[in] relative_time Time at which to deposit J, relative to the time of the
 *                          current positions of the particles. When different than 0,
 *                          the particle position will be temporarily modified to match
 *                          the time of the deposition.
 * \param dinv         3D cell size inverse
 * \param xyzmin       Physical lower bounds of domain.
 * \param lo           Index lower bounds of domain.
 * \param q            species charge.
 * \param n_rz_azimuthal_modes Number of azimuthal modes when using RZ geometry.
 */
template <int depos_order>
void doEsirkepovDepositionShapeN (const GetParticlePosition<PIdx>& GetPosition,
                                  const amrex::ParticleReal * const wp,
                                  const amrex::ParticleReal * const uxp,
                                  const amrex::ParticleReal * const uyp,
                                  const amrex::ParticleReal * const uzp,
                                  const int* ion_lev,
                                  const amrex::Array4<amrex::Real>& Jx_arr,
                                  const amrex::Array4<amrex::Real>& Jy_arr,
                                  const amrex::Array4<amrex::Real>& Jz_arr,
                                  long np_to_deposit,
                                  amrex::Real dt,
                                  amrex::Real relative_time,
                                  const amrex::XDim3 & dinv,
                                  const amrex::XDim3 & xyzmin,
                                  amrex::Dim3 lo,
                                  amrex::Real q,
                                  [[maybe_unused]]int n_rz_azimuthal_modes)
{
    using namespace amrex;
    using namespace amrex::literals;

    // Whether ion_lev is a null pointer (do_ionization=0) or a real pointer
    // (do_ionization=1)
    bool const do_ionization = ion_lev;
#if !defined(WARPX_DIM_3D)
    const amrex::Real invvol = dinv.x*dinv.y*dinv.z;
#endif

    amrex::XDim3 const invdtd = amrex::XDim3{(1.0_rt/dt)*dinv.y*dinv.z,
                                             (1.0_rt/dt)*dinv.x*dinv.z,
                                             (1.0_rt/dt)*dinv.x*dinv.y};

    Real constexpr clightsq = 1.0_rt / ( PhysConst::c * PhysConst::c );

#if !defined(WARPX_DIM_1D_Z)
    Real constexpr one_third = 1.0_rt / 3.0_rt;
    Real constexpr one_sixth = 1.0_rt / 6.0_rt;
#endif

    // Loop over particles and deposit into Jx_arr, Jy_arr and Jz_arr
    amrex::ParallelFor(
        np_to_deposit,
        [=] AMREX_GPU_DEVICE (long const ip) {
            // --- Get particle quantities
            Real const gaminv = 1.0_rt/std::sqrt(1.0_rt + uxp[ip]*uxp[ip]*clightsq
                                                 + uyp[ip]*uyp[ip]*clightsq
                                                 + uzp[ip]*uzp[ip]*clightsq);

            Real wq = q*wp[ip];
            if (do_ionization){
                wq *= ion_lev[ip];
            }

            ParticleReal xp, yp, zp;
            GetPosition(ip, xp, yp, zp);

            // computes current and old position in grid units
#if defined(WARPX_DIM_RZ)
            Real const xp_new = xp + (relative_time + 0.5_rt*dt)*uxp[ip]*gaminv;
            Real const yp_new = yp + (relative_time + 0.5_rt*dt)*uyp[ip]*gaminv;
            Real const xp_mid = xp_new - 0.5_rt*dt*uxp[ip]*gaminv;
            Real const yp_mid = yp_new - 0.5_rt*dt*uyp[ip]*gaminv;
            Real const xp_old = xp_new - dt*uxp[ip]*gaminv;
            Real const yp_old = yp_new - dt*uyp[ip]*gaminv;
            Real const rp_new = std::sqrt(xp_new*xp_new + yp_new*yp_new);
            Real const rp_mid = std::sqrt(xp_mid*xp_mid + yp_mid*yp_mid);
            Real const rp_old = std::sqrt(xp_old*xp_old + yp_old*yp_old);
            const amrex::Real costheta_mid = (rp_mid > 0._rt ? xp_mid/rp_mid : 1._rt);
            const amrex::Real sintheta_mid = (rp_mid > 0._rt ? yp_mid/rp_mid : 0._rt);
            const amrex::Real costheta_new = (rp_new > 0._rt ? xp_new/rp_new : 1._rt);
            const amrex::Real sintheta_new = (rp_new > 0._rt ? yp_new/rp_new : 0._rt);
            const amrex::Real costheta_old = (rp_old > 0._rt ? xp_old/rp_old : 1._rt);
            const amrex::Real sintheta_old = (rp_old > 0._rt ? yp_old/rp_old : 0._rt);
            const Complex xy_new0 = Complex{costheta_new, sintheta_new};
            const Complex xy_mid0 = Complex{costheta_mid, sintheta_mid};
            const Complex xy_old0 = Complex{costheta_old, sintheta_old};
            // Keep these double to avoid bug in single precision
            double const x_new = (rp_new - xyzmin.x)*dinv.x;
            double const x_old = (rp_old - xyzmin.x)*dinv.x;
#else
#if !defined(WARPX_DIM_1D_Z)
            // Keep these double to avoid bug in single precision
            double const x_new = (xp - xyzmin.x + (relative_time + 0.5_rt*dt)*uxp[ip]*gaminv)*dinv.x;
            double const x_old = x_new - dt*dinv.x*uxp[ip]*gaminv;
#endif
#endif
#if defined(WARPX_DIM_3D)
            // Keep these double to avoid bug in single precision
            double const y_new = (yp - xyzmin.y + (relative_time + 0.5_rt*dt)*uyp[ip]*gaminv)*dinv.y;
            double const y_old = y_new - dt*dinv.y*uyp[ip]*gaminv;
#endif
            // Keep these double to avoid bug in single precision
            double const z_new = (zp - xyzmin.z + (relative_time + 0.5_rt*dt)*uzp[ip]*gaminv)*dinv.z;
            double const z_old = z_new - dt*dinv.z*uzp[ip]*gaminv;

#if defined(WARPX_DIM_RZ)
            Real const vy = (-uxp[ip]*sintheta_mid + uyp[ip]*costheta_mid)*gaminv;
#elif defined(WARPX_DIM_XZ)
            Real const vy = uyp[ip]*gaminv;
#elif defined(WARPX_DIM_1D_Z)
            Real const vx = uxp[ip]*gaminv;
            Real const vy = uyp[ip]*gaminv;
#endif

            // --- Compute shape factors
            // Compute shape factors for position as they are now and at old positions
            // [ijk]_new: leftmost grid point that the particle touches
            const Compute_shape_factor< depos_order > compute_shape_factor;
            const Compute_shifted_shape_factor< depos_order > compute_shifted_shape_factor;

            // Shape factor arrays
            // Note that there are extra values above and below
            // to possibly hold the factor for the old particle
            // which can be at a different grid location.
            // Keep these double to avoid bug in single precision
#if !defined(WARPX_DIM_1D_Z)
            double sx_new[depos_order + 3] = {0.};
            double sx_old[depos_order + 3] = {0.};
            const int i_new = compute_shape_factor(sx_new+1, x_new);
            const int i_old = compute_shifted_shape_factor(sx_old, x_old, i_new);
#endif
#if defined(WARPX_DIM_3D)
            double sy_new[depos_order + 3] = {0.};
            double sy_old[depos_order + 3] = {0.};
            const int j_new = compute_shape_factor(sy_new+1, y_new);
            const int j_old = compute_shifted_shape_factor(sy_old, y_old, j_new);
#endif
            double sz_new[depos_order + 3] = {0.};
            double sz_old[depos_order + 3] = {0.};
            const int k_new = compute_shape_factor(sz_new+1, z_new);
            const int k_old = compute_shifted_shape_factor(sz_old, z_old, k_new);

            // computes min/max positions of current contributions
#if !defined(WARPX_DIM_1D_Z)
            int dil = 1, diu = 1;
            if (i_old < i_new) { dil = 0; }
            if (i_old > i_new) { diu = 0; }
#endif
#if defined(WARPX_DIM_3D)
            int djl = 1, dju = 1;
            if (j_old < j_new) { djl = 0; }
            if (j_old > j_new) { dju = 0; }
#endif
            int dkl = 1, dku = 1;
            if (k_old < k_new) { dkl = 0; }
            if (k_old > k_new) { dku = 0; }

#if defined(WARPX_DIM_3D)

            for (int k=dkl; k<=depos_order+2-dku; k++) {
                for (int j=djl; j<=depos_order+2-dju; j++) {
                    amrex::Real sdxi = 0._rt;
                    for (int i=dil; i<=depos_order+1-diu; i++) {
                        sdxi += wq*invdtd.x*(sx_old[i] - sx_new[i])*(
                            one_third*(sy_new[j]*sz_new[k] + sy_old[j]*sz_old[k])
                           +one_sixth*(sy_new[j]*sz_old[k] + sy_old[j]*sz_new[k]));
                        amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i_new-1+i, lo.y+j_new-1+j, lo.z+k_new-1+k), sdxi);
                    }
                }
            }
            for (int k=dkl; k<=depos_order+2-dku; k++) {
                for (int i=dil; i<=depos_order+2-diu; i++) {
                    amrex::Real sdyj = 0._rt;
                    for (int j=djl; j<=depos_order+1-dju; j++) {
                        sdyj += wq*invdtd.y*(sy_old[j] - sy_new[j])*(
                            one_third*(sx_new[i]*sz_new[k] + sx_old[i]*sz_old[k])
                           +one_sixth*(sx_new[i]*sz_old[k] + sx_old[i]*sz_new[k]));
                        amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i_new-1+i, lo.y+j_new-1+j, lo.z+k_new-1+k), sdyj);
                    }
                }
            }
            for (int j=djl; j<=depos_order+2-dju; j++) {
                for (int i=dil; i<=depos_order+2-diu; i++) {
                    amrex::Real sdzk = 0._rt;
                    for (int k=dkl; k<=depos_order+1-dku; k++) {
                        sdzk += wq*invdtd.z*(sz_old[k] - sz_new[k])*(
                            one_third*(sx_new[i]*sy_new[j] + sx_old[i]*sy_old[j])
                           +one_sixth*(sx_new[i]*sy_old[j] + sx_old[i]*sy_new[j]));
                        amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i_new-1+i, lo.y+j_new-1+j, lo.z+k_new-1+k), sdzk);
                    }
                }
            }

#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)

            for (int k=dkl; k<=depos_order+2-dku; k++) {
                amrex::Real sdxi = 0._rt;
                for (int i=dil; i<=depos_order+1-diu; i++) {
                    sdxi += wq*invdtd.x*(sx_old[i] - sx_new[i])*0.5_rt*(sz_new[k] + sz_old[k]);
                    amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 0), sdxi);
#if defined(WARPX_DIM_RZ)
                    Complex xy_mid = xy_mid0; // Throughout the following loop, xy_mid takes the value e^{i m theta}
                    for (int imode=1 ; imode < n_rz_azimuthal_modes ; imode++) {
                        // The factor 2 comes from the normalization of the modes
                        const Complex djr_cmplx = 2._rt *sdxi*xy_mid;
                        amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode-1), djr_cmplx.real());
                        amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode), djr_cmplx.imag());
                        xy_mid = xy_mid*xy_mid0;
                    }
#endif
                }
            }
            for (int k=dkl; k<=depos_order+2-dku; k++) {
                for (int i=dil; i<=depos_order+2-diu; i++) {
                    Real const sdyj = wq*vy*invvol*(
                        one_third*(sx_new[i]*sz_new[k] + sx_old[i]*sz_old[k])
                       +one_sixth*(sx_new[i]*sz_old[k] + sx_old[i]*sz_new[k]));
                    amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 0), sdyj);
#if defined(WARPX_DIM_RZ)
                    Complex const I = Complex{0._rt, 1._rt};
                    Complex xy_new = xy_new0;
                    Complex xy_mid = xy_mid0;
                    Complex xy_old = xy_old0;
                    // Throughout the following loop, xy_ takes the value e^{i m theta_}
                    for (int imode=1 ; imode < n_rz_azimuthal_modes ; imode++) {
                        // The factor 2 comes from the normalization of the modes
                        // The minus sign comes from the different convention with respect to Davidson et al.
                        const Complex djt_cmplx = -2._rt * I*(i_new-1 + i + xyzmin.x*dinv.x)*wq*invdtd.x/(amrex::Real)imode
                                                  *(Complex(sx_new[i]*sz_new[k], 0._rt)*(xy_new - xy_mid)
                                                  + Complex(sx_old[i]*sz_old[k], 0._rt)*(xy_mid - xy_old));
                        amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode-1), djt_cmplx.real());
                        amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode), djt_cmplx.imag());
                        xy_new = xy_new*xy_new0;
                        xy_mid = xy_mid*xy_mid0;
                        xy_old = xy_old*xy_old0;
                    }
#endif
                }
            }
            for (int i=dil; i<=depos_order+2-diu; i++) {
                Real sdzk = 0._rt;
                for (int k=dkl; k<=depos_order+1-dku; k++) {
                    sdzk += wq*invdtd.z*(sz_old[k] - sz_new[k])*0.5_rt*(sx_new[i] + sx_old[i]);
                    amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 0), sdzk);
#if defined(WARPX_DIM_RZ)
                    Complex xy_mid = xy_mid0; // Throughout the following loop, xy_mid takes the value e^{i m theta}
                    for (int imode=1 ; imode < n_rz_azimuthal_modes ; imode++) {
                        // The factor 2 comes from the normalization of the modes
                        const Complex djz_cmplx = 2._rt * sdzk * xy_mid;
                        amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode-1), djz_cmplx.real());
                        amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode), djz_cmplx.imag());
                        xy_mid = xy_mid*xy_mid0;
                    }
#endif
                }
            }
#elif defined(WARPX_DIM_1D_Z)

            for (int k=dkl; k<=depos_order+2-dku; k++) {
                amrex::Real const sdxi = wq*vx*invvol*0.5_rt*(sz_old[k] + sz_new[k]);
                amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+k_new-1+k, 0, 0, 0), sdxi);
            }
            for (int k=dkl; k<=depos_order+2-dku; k++) {
                amrex::Real const sdyj = wq*vy*invvol*0.5_rt*(sz_old[k] + sz_new[k]);
                amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+k_new-1+k, 0, 0, 0), sdyj);
            }
            amrex::Real sdzk = 0._rt;
            for (int k=dkl; k<=depos_order+1-dku; k++) {
                sdzk += wq*invdtd.z*(sz_old[k] - sz_new[k]);
                amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+k_new-1+k, 0, 0, 0), sdzk);
            }
#endif
        }
    );
}

/**
 * \brief Esirkepov Current Deposition for thread thread_num for implicit scheme
 *        The difference from doEsirkepovDepositionShapeN is in how the old and new
 *        particles positions are determined and in how the particle gamma is calculated.
 *
 * \tparam depos_order  deposition order
 * \param xp_n,yp_n,zp_n  Pointer to arrays of particle position at time level n.
 * \param GetPosition  A functor for returning the particle position.
 * \param wp           Pointer to array of particle weights.
 * \param uxp_n,uyp_n,uzp_n  Pointer to arrays of particle momentum at time level n.
 * \param uxp_nph,uyp_nph,uzp_nph  Pointer to arrays of particle momentum at time level n + 1/2.
 * \param ion_lev      Pointer to array of particle ionization level. This is
                       required to have the charge of each macroparticle
                       since q is a scalar. For non-ionizable species,
                       ion_lev is a null pointer.
 * \param Jx_arr,Jy_arr,Jz_arr Array4 of current density, either full array or tile.
 * \param np_to_deposit Number of particles for which current is deposited.
 * \param dt           Time step for particle level
 * \param dinv         3D cell size inverse
 * \param xyzmin       Physical lower bounds of domain.
 * \param lo           Index lower bounds of domain.
 * \param q            species charge.
 * \param n_rz_azimuthal_modes Number of azimuthal modes when using RZ geometry.
 */
template <int depos_order>
void doChargeConservingDepositionShapeNImplicit ([[maybe_unused]]const amrex::ParticleReal * const xp_n,
                                                 [[maybe_unused]]const amrex::ParticleReal * const yp_n,
                                                 [[maybe_unused]]const amrex::ParticleReal * const zp_n,
                                                 const GetParticlePosition<PIdx>& GetPosition,
                                                 const amrex::ParticleReal * const wp,
                                                 [[maybe_unused]]const amrex::ParticleReal * const uxp_n,
                                                 [[maybe_unused]]const amrex::ParticleReal * const uyp_n,
                                                 [[maybe_unused]]const amrex::ParticleReal * const uzp_n,
                                                 [[maybe_unused]]const amrex::ParticleReal * const uxp_nph,
                                                 [[maybe_unused]]const amrex::ParticleReal * const uyp_nph,
                                                 [[maybe_unused]]const amrex::ParticleReal * const uzp_nph,
                                                 const int * const ion_lev,
                                                 const amrex::Array4<amrex::Real>& Jx_arr,
                                                 const amrex::Array4<amrex::Real>& Jy_arr,
                                                 const amrex::Array4<amrex::Real>& Jz_arr,
                                                 const long np_to_deposit,
                                                 const amrex::Real dt,
                                                 const amrex::XDim3 & dinv,
                                                 const amrex::XDim3 & xyzmin,
                                                 const amrex::Dim3 lo,
                                                 const amrex::Real q,
                                                 [[maybe_unused]] const int n_rz_azimuthal_modes)
{
    using namespace amrex;

    // Whether ion_lev is a null pointer (do_ionization=0) or a real pointer
    // (do_ionization=1)
    bool const do_ionization = ion_lev;

#if !defined(WARPX_DIM_3D)
    const amrex::Real invvol = dinv.x*dinv.y*dinv.z;
#endif

    amrex::XDim3 const invdtd = amrex::XDim3{(1.0_rt/dt)*dinv.y*dinv.z,
                                             (1.0_rt/dt)*dinv.x*dinv.z,
                                             (1.0_rt/dt)*dinv.x*dinv.y};

#if !defined(WARPX_DIM_1D_Z)
    Real constexpr one_third = 1.0_rt / 3.0_rt;
    Real constexpr one_sixth = 1.0_rt / 6.0_rt;
#endif

    // Loop over particles and deposit into Jx_arr, Jy_arr and Jz_arr
    amrex::ParallelFor(
        np_to_deposit,
        [=] AMREX_GPU_DEVICE (long const ip) {
#if !defined(WARPX_DIM_3D)
            constexpr amrex::ParticleReal inv_c2 = 1._prt/(PhysConst::c*PhysConst::c);

            // Compute inverse Lorentz factor, the average of gamma at time levels n and n+1
            // The uxp,uyp,uzp are the velocities at time level n+1/2
            const amrex::ParticleReal uxp_np1 = 2._prt*uxp_nph[ip] - uxp_n[ip];
            const amrex::ParticleReal uyp_np1 = 2._prt*uyp_nph[ip] - uyp_n[ip];
            const amrex::ParticleReal uzp_np1 = 2._prt*uzp_nph[ip] - uzp_n[ip];
            const amrex::ParticleReal gamma_n = std::sqrt(1._prt + (uxp_n[ip]*uxp_n[ip] + uyp_n[ip]*uyp_n[ip] + uzp_n[ip]*uzp_n[ip])*inv_c2);
            const amrex::ParticleReal gamma_np1 = std::sqrt(1._prt + (uxp_np1*uxp_np1 + uyp_np1*uyp_np1 + uzp_np1*uzp_np1)*inv_c2);
            const amrex::ParticleReal gaminv = 2.0_prt/(gamma_n + gamma_np1);
#endif

            Real wq = q*wp[ip];
            if (do_ionization){
                wq *= ion_lev[ip];
            }

            ParticleReal xp_nph, yp_nph, zp_nph;
            GetPosition(ip, xp_nph, yp_nph, zp_nph);

#if !defined(WARPX_DIM_1D_Z)
            ParticleReal const xp_np1 = 2._prt*xp_nph - xp_n[ip];
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ)
            ParticleReal const yp_np1 = 2._prt*yp_nph - yp_n[ip];
#endif
            ParticleReal const zp_np1 = 2._prt*zp_nph - zp_n[ip];

            // computes current and old position in grid units
#if defined(WARPX_DIM_RZ)
            amrex::Real const xp_new = xp_np1;
            amrex::Real const yp_new = yp_np1;
            amrex::Real const xp_mid = xp_nph;
            amrex::Real const yp_mid = yp_nph;
            amrex::Real const xp_old = xp_n[ip];
            amrex::Real const yp_old = yp_n[ip];
            amrex::Real const rp_new = std::sqrt(xp_new*xp_new + yp_new*yp_new);
            amrex::Real const rp_old = std::sqrt(xp_old*xp_old + yp_old*yp_old);
            amrex::Real const rp_mid = (rp_new + rp_old)/2._rt;
            const amrex::Real costheta_mid = (rp_mid > 0._rt ? xp_mid/rp_mid : 1._rt);
            const amrex::Real sintheta_mid = (rp_mid > 0._rt ? yp_mid/rp_mid : 0._rt);
            const amrex::Real costheta_new = (rp_new > 0._rt ? xp_new/rp_new : 1._rt);
            const amrex::Real sintheta_new = (rp_new > 0._rt ? yp_new/rp_new : 0._rt);
            const amrex::Real costheta_old = (rp_old > 0._rt ? xp_old/rp_old : 1._rt);
            const amrex::Real sintheta_old = (rp_old > 0._rt ? yp_old/rp_old : 0._rt);
            const Complex xy_new0 = Complex{costheta_new, sintheta_new};
            const Complex xy_mid0 = Complex{costheta_mid, sintheta_mid};
            const Complex xy_old0 = Complex{costheta_old, sintheta_old};
            // Keep these double to avoid bug in single precision
            double const x_new = (rp_new - xyzmin.x)*dinv.x;
            double const x_old = (rp_old - xyzmin.x)*dinv.x;
#else
#if !defined(WARPX_DIM_1D_Z)
            // Keep these double to avoid bug in single precision
            double const x_new = (xp_np1 - xyzmin.x)*dinv.x;
            double const x_old = (xp_n[ip] - xyzmin.x)*dinv.x;
#endif
#endif
#if defined(WARPX_DIM_3D)
            // Keep these double to avoid bug in single precision
            double const y_new = (yp_np1 - xyzmin.y)*dinv.y;
            double const y_old = (yp_n[ip] - xyzmin.y)*dinv.y;
#endif
            // Keep these double to avoid bug in single precision
            double const z_new = (zp_np1 - xyzmin.z)*dinv.z;
            double const z_old = (zp_n[ip] - xyzmin.z)*dinv.z;

#if defined(WARPX_DIM_RZ)
            amrex::Real const vy = (-uxp_nph[ip]*sintheta_mid + uyp_nph[ip]*costheta_mid)*gaminv;
#elif defined(WARPX_DIM_XZ)
            amrex::Real const vy = uyp_nph[ip]*gaminv;
#elif defined(WARPX_DIM_1D_Z)
            amrex::Real const vx = uxp_nph[ip]*gaminv;
            amrex::Real const vy = uyp_nph[ip]*gaminv;
#endif

            // --- Compute shape factors
            // Compute shape factors for position as they are now and at old positions
            // [ijk]_new: leftmost grid point that the particle touches
            const Compute_shape_factor< depos_order > compute_shape_factor;
            const Compute_shifted_shape_factor< depos_order > compute_shifted_shape_factor;

            // Shape factor arrays
            // Note that there are extra values above and below
            // to possibly hold the factor for the old particle
            // which can be at a different grid location.
            // Keep these double to avoid bug in single precision
#if !defined(WARPX_DIM_1D_Z)
            double sx_new[depos_order + 3] = {0.};
            double sx_old[depos_order + 3] = {0.};
            const int i_new = compute_shape_factor(sx_new+1, x_new);
            const int i_old = compute_shifted_shape_factor(sx_old, x_old, i_new);
#endif
#if defined(WARPX_DIM_3D)
            double sy_new[depos_order + 3] = {0.};
            double sy_old[depos_order + 3] = {0.};
            const int j_new = compute_shape_factor(sy_new+1, y_new);
            const int j_old = compute_shifted_shape_factor(sy_old, y_old, j_new);
#endif
            double sz_new[depos_order + 3] = {0.};
            double sz_old[depos_order + 3] = {0.};
            const int k_new = compute_shape_factor(sz_new+1, z_new);
            const int k_old = compute_shifted_shape_factor(sz_old, z_old, k_new);

            // computes min/max positions of current contributions
#if !defined(WARPX_DIM_1D_Z)
            int dil = 1, diu = 1;
            if (i_old < i_new) { dil = 0; }
            if (i_old > i_new) { diu = 0; }
#endif
#if defined(WARPX_DIM_3D)
            int djl = 1, dju = 1;
            if (j_old < j_new) { djl = 0; }
            if (j_old > j_new) { dju = 0; }
#endif
            int dkl = 1, dku = 1;
            if (k_old < k_new) { dkl = 0; }
            if (k_old > k_new) { dku = 0; }

#if defined(WARPX_DIM_3D)

            for (int k=dkl; k<=depos_order+2-dku; k++) {
                for (int j=djl; j<=depos_order+2-dju; j++) {
                    amrex::Real sdxi = 0._rt;
                    for (int i=dil; i<=depos_order+1-diu; i++) {
                        sdxi += wq*invdtd.x*(sx_old[i] - sx_new[i])*(
                            one_third*(sy_new[j]*sz_new[k] + sy_old[j]*sz_old[k])
                           +one_sixth*(sy_new[j]*sz_old[k] + sy_old[j]*sz_new[k]));
                        amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i_new-1+i, lo.y+j_new-1+j, lo.z+k_new-1+k), sdxi);
                    }
                }
            }
            for (int k=dkl; k<=depos_order+2-dku; k++) {
                for (int i=dil; i<=depos_order+2-diu; i++) {
                    amrex::Real sdyj = 0._rt;
                    for (int j=djl; j<=depos_order+1-dju; j++) {
                        sdyj += wq*invdtd.y*(sy_old[j] - sy_new[j])*(
                            one_third*(sx_new[i]*sz_new[k] + sx_old[i]*sz_old[k])
                           +one_sixth*(sx_new[i]*sz_old[k] + sx_old[i]*sz_new[k]));
                        amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i_new-1+i, lo.y+j_new-1+j, lo.z+k_new-1+k), sdyj);
                    }
                }
            }
            for (int j=djl; j<=depos_order+2-dju; j++) {
                for (int i=dil; i<=depos_order+2-diu; i++) {
                    amrex::Real sdzk = 0._rt;
                    for (int k=dkl; k<=depos_order+1-dku; k++) {
                        sdzk += wq*invdtd.z*(sz_old[k] - sz_new[k])*(
                            one_third*(sx_new[i]*sy_new[j] + sx_old[i]*sy_old[j])
                           +one_sixth*(sx_new[i]*sy_old[j] + sx_old[i]*sy_new[j]));
                        amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i_new-1+i, lo.y+j_new-1+j, lo.z+k_new-1+k), sdzk);
                    }
                }
            }

#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)

            for (int k=dkl; k<=depos_order+2-dku; k++) {
                amrex::Real sdxi = 0._rt;
                for (int i=dil; i<=depos_order+1-diu; i++) {
                    sdxi += wq*invdtd.x*(sx_old[i] - sx_new[i])*0.5_rt*(sz_new[k] + sz_old[k]);
                    amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 0), sdxi);
#if defined(WARPX_DIM_RZ)
                    Complex xy_mid = xy_mid0; // Throughout the following loop, xy_mid takes the value e^{i m theta}
                    for (int imode=1 ; imode < n_rz_azimuthal_modes ; imode++) {
                        // The factor 2 comes from the normalization of the modes
                        const Complex djr_cmplx = 2._rt *sdxi*xy_mid;
                        amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode-1), djr_cmplx.real());
                        amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode), djr_cmplx.imag());
                        xy_mid = xy_mid*xy_mid0;
                    }
#endif
                }
            }
            for (int k=dkl; k<=depos_order+2-dku; k++) {
                for (int i=dil; i<=depos_order+2-diu; i++) {
                    Real const sdyj = wq*vy*invvol*(
                        one_third*(sx_new[i]*sz_new[k] + sx_old[i]*sz_old[k])
                       +one_sixth*(sx_new[i]*sz_old[k] + sx_old[i]*sz_new[k]));
                    amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 0), sdyj);
#if defined(WARPX_DIM_RZ)
                    Complex const I = Complex{0._rt, 1._rt};
                    Complex xy_new = xy_new0;
                    Complex xy_mid = xy_mid0;
                    Complex xy_old = xy_old0;
                    // Throughout the following loop, xy_ takes the value e^{i m theta_}
                    for (int imode=1 ; imode < n_rz_azimuthal_modes ; imode++) {
                        // The factor 2 comes from the normalization of the modes
                        // The minus sign comes from the different convention with respect to Davidson et al.
                        const Complex djt_cmplx = -2._rt * I*(i_new-1 + i + xyzmin.x*dinv.x)*wq*invdtd.x/(amrex::Real)imode
                                                  *(Complex(sx_new[i]*sz_new[k], 0._rt)*(xy_new - xy_mid)
                                                  + Complex(sx_old[i]*sz_old[k], 0._rt)*(xy_mid - xy_old));
                        amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode-1), djt_cmplx.real());
                        amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode), djt_cmplx.imag());
                        xy_new = xy_new*xy_new0;
                        xy_mid = xy_mid*xy_mid0;
                        xy_old = xy_old*xy_old0;
                    }
#endif
                }
            }
            for (int i=dil; i<=depos_order+2-diu; i++) {
                Real sdzk = 0._rt;
                for (int k=dkl; k<=depos_order+1-dku; k++) {
                    sdzk += wq*invdtd.z*(sz_old[k] - sz_new[k])*0.5_rt*(sx_new[i] + sx_old[i]);
                    amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 0), sdzk);
#if defined(WARPX_DIM_RZ)
                    Complex xy_mid = xy_mid0; // Throughout the following loop, xy_mid takes the value e^{i m theta}
                    for (int imode=1 ; imode < n_rz_azimuthal_modes ; imode++) {
                        // The factor 2 comes from the normalization of the modes
                        const Complex djz_cmplx = 2._rt * sdzk * xy_mid;
                        amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode-1), djz_cmplx.real());
                        amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode), djz_cmplx.imag());
                        xy_mid = xy_mid*xy_mid0;
                    }
#endif
                }
            }
#elif defined(WARPX_DIM_1D_Z)

            for (int k=dkl; k<=depos_order+2-dku; k++) {
                amrex::Real const sdxi = wq*vx*invvol*0.5_rt*(sz_old[k] + sz_new[k]);
                amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+k_new-1+k, 0, 0, 0), sdxi);
            }
            for (int k=dkl; k<=depos_order+2-dku; k++) {
                amrex::Real const sdyj = wq*vy*invvol*0.5_rt*(sz_old[k] + sz_new[k]);
                amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+k_new-1+k, 0, 0, 0), sdyj);
            }
            amrex::Real sdzk = 0._rt;
            for (int k=dkl; k<=depos_order+1-dku; k++) {
                sdzk += wq*invdtd.z*(sz_old[k] - sz_new[k]);
                amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+k_new-1+k, 0, 0, 0), sdzk);
            }
#endif
        }
    );
}

/**
 * \brief Villasenor and Buneman Current Deposition for thread thread_num for implicit scheme.
 *        The specifics for the implicit scheme are in how gamma is determined. This is a charge-
 *        conserving deposition. The difference from Esirkepov is that the deposit is done segment
 *        by segment, where the segments are determined by cell crossings. In general, this results
 *        in a tighter stencil. The implementation is valid for an arbitrary number of cell crossings.
 *
 * \param depos_order  deposition order
 * \param xp_n,yp_n,zp_n  Pointer to arrays of particle position at time level n.
 * \param GetPosition  A functor for returning the particle position.
 * \param wp           Pointer to array of particle weights.
 * \param uxp_n,uyp_n,uzp_n  Pointer to arrays of particle momentum at time level n.
 * \param uxp_nph,uyp_nph,uzp_nph  Pointer to arrays of particle momentum at time level n + 1/2.
 * \param ion_lev      Pointer to array of particle ionization level. This is
                       required to have the charge of each macroparticle
                       since q is a scalar. For non-ionizable species,
                       ion_lev is a null pointer.
 * \param Jx_arr,Jy_arr,Jz_arr  Array4 of current density, either full array or tile.
 * \param np_to_deposit         Number of particles for which current is deposited.
 * \param dt                    Time step for particle level
 * \param dinv                  3D cell size inverse
 * \param xyzmin                Physical lower bounds of domain.
 * \param lo                    Index lower bounds of domain.
 * \param q                     species charge.
 * \param n_rz_azimuthal_modes  Number of azimuthal modes when using RZ geometry.
 */
template <int depos_order>
void doVillasenorDepositionShapeNImplicit ([[maybe_unused]]const amrex::ParticleReal * const xp_n,
                                           [[maybe_unused]]const amrex::ParticleReal * const yp_n,
                                           [[maybe_unused]]const amrex::ParticleReal * const zp_n,
                                           const GetParticlePosition<PIdx>& GetPosition,
                                           const amrex::ParticleReal * const wp,
                                           [[maybe_unused]]const amrex::ParticleReal * const uxp_n,
                                           [[maybe_unused]]const amrex::ParticleReal * const uyp_n,
                                           [[maybe_unused]]const amrex::ParticleReal * const uzp_n,
                                           [[maybe_unused]]const amrex::ParticleReal * const uxp_nph,
                                           [[maybe_unused]]const amrex::ParticleReal * const uyp_nph,
                                           [[maybe_unused]]const amrex::ParticleReal * const uzp_nph,
                                           const int * const ion_lev,
                                           const amrex::Array4<amrex::Real>& Jx_arr,
                                           const amrex::Array4<amrex::Real>& Jy_arr,
                                           const amrex::Array4<amrex::Real>& Jz_arr,
                                           const long np_to_deposit,
                                           const amrex::Real dt,
                                           const amrex::XDim3 & dinv,
                                           const amrex::XDim3 & xyzmin,
                                           const amrex::Dim3 lo,
                                           const amrex::Real q,
                                           [[maybe_unused]] const int n_rz_azimuthal_modes)
{
    using namespace amrex;

    // Whether ion_lev is a null pointer (do_ionization=0) or a real pointer
    // (do_ionization=1)
    bool const do_ionization = ion_lev;

    const amrex::Real invvol = dinv.x*dinv.y*dinv.z;

#if (AMREX_SPACEDIM > 1)
    Real constexpr one_third = 1.0_rt / 3.0_rt;
    Real constexpr one_sixth = 1.0_rt / 6.0_rt;
#endif

    // Loop over particles and deposit into Jx_arr, Jy_arr and Jz_arr
    amrex::ParallelFor(
        np_to_deposit,
        [=] AMREX_GPU_DEVICE (long const ip) {

#if !defined(WARPX_DIM_3D)
            constexpr amrex::ParticleReal inv_c2 = 1._prt/(PhysConst::c*PhysConst::c);

            // Compute inverse Lorentz factor, the average of gamma at time levels n and n+1
            // The uxp,uyp,uzp are the velocities at time level n+1/2
            const amrex::ParticleReal uxp_np1 = 2._prt*uxp_nph[ip] - uxp_n[ip];
            const amrex::ParticleReal uyp_np1 = 2._prt*uyp_nph[ip] - uyp_n[ip];
            const amrex::ParticleReal uzp_np1 = 2._prt*uzp_nph[ip] - uzp_n[ip];
            const amrex::ParticleReal gamma_n = std::sqrt(1._prt + (uxp_n[ip]*uxp_n[ip] + uyp_n[ip]*uyp_n[ip] + uzp_n[ip]*uzp_n[ip])*inv_c2);
            const amrex::ParticleReal gamma_np1 = std::sqrt(1._prt + (uxp_np1*uxp_np1 + uyp_np1*uyp_np1 + uzp_np1*uzp_np1)*inv_c2);
            const amrex::ParticleReal gaminv = 2.0_prt/(gamma_n + gamma_np1);
#endif

            Real wq = q*wp[ip];
            if (do_ionization){
                wq *= ion_lev[ip];
            }

            ParticleReal xp_nph, yp_nph, zp_nph;
            GetPosition(ip, xp_nph, yp_nph, zp_nph);

#if !defined(WARPX_DIM_1D_Z)
            ParticleReal const xp_np1 = 2._prt*xp_nph - xp_n[ip];
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ)
            ParticleReal const yp_np1 = 2._prt*yp_nph - yp_n[ip];
#endif
            ParticleReal const zp_np1 = 2._prt*zp_nph - zp_n[ip];

            // computes current and old position in grid units
#if defined(WARPX_DIM_RZ)
            amrex::Real const xp_new = xp_np1;
            amrex::Real const yp_new = yp_np1;
            amrex::Real const xp_mid = xp_nph;
            amrex::Real const yp_mid = yp_nph;
            amrex::Real const xp_old = xp_n[ip];
            amrex::Real const yp_old = yp_n[ip];
            amrex::Real const rp_new = std::sqrt(xp_new*xp_new + yp_new*yp_new);
            amrex::Real const rp_old = std::sqrt(xp_old*xp_old + yp_old*yp_old);
            amrex::Real const rp_mid = (rp_new + rp_old)/2._rt;
            amrex::Real costheta_mid, sintheta_mid;
            if (rp_mid > 0._rt) {
                costheta_mid = xp_mid/rp_mid;
                sintheta_mid = yp_mid/rp_mid;
            } else {
                costheta_mid = 1._rt;
                sintheta_mid = 0._rt;
            }
            const Complex xy_mid0 = Complex{costheta_mid, sintheta_mid};

            // Keep these double to avoid bug in single precision
            double const x_new = (rp_new - xyzmin.x)*dinv.x;
            double const x_old = (rp_old - xyzmin.x)*dinv.x;
            amrex::Real const vx = (rp_new - rp_old)/dt;
            amrex::Real const vy = (-uxp_nph[ip]*sintheta_mid + uyp_nph[ip]*costheta_mid)*gaminv;
#elif defined(WARPX_DIM_XZ)
            // Keep these double to avoid bug in single precision
            double const x_new = (xp_np1 - xyzmin.x)*dinv.x;
            double const x_old = (xp_n[ip] - xyzmin.x)*dinv.x;
            amrex::Real const vx = (xp_np1 - xp_n[ip])/dt;
            amrex::Real const vy = uyp_nph[ip]*gaminv;
#elif defined(WARPX_DIM_1D_Z)
            amrex::Real const vx = uxp_nph[ip]*gaminv;
            amrex::Real const vy = uyp_nph[ip]*gaminv;
#elif defined(WARPX_DIM_3D)
            // Keep these double to avoid bug in single precision
            double const x_new = (xp_np1 - xyzmin.x)*dinv.x;
            double const x_old = (xp_n[ip] - xyzmin.x)*dinv.x;
            double const y_new = (yp_np1 - xyzmin.y)*dinv.y;
            double const y_old = (yp_n[ip] - xyzmin.y)*dinv.y;
            amrex::Real const vx = (xp_np1 - xp_n[ip])/dt;
            amrex::Real const vy = (yp_np1 - yp_n[ip])/dt;
#endif

            // Keep these double to avoid bug in single precision
            double const z_new = (zp_np1 - xyzmin.z)*dinv.z;
            double const z_old = (zp_n[ip] - xyzmin.z)*dinv.z;
            amrex::Real const vz = (zp_np1 - zp_n[ip])/dt;

            // Define velocity kernals to deposit
            amrex::Real const wqx = wq*vx*invvol;
            amrex::Real const wqy = wq*vy*invvol;
            amrex::Real const wqz = wq*vz*invvol;

            // 1) Determine the number of segments.
            // 2) Loop over segments and deposit current.

            // cell crossings are defined at cell edges if depos_order is odd
            // cell crossings are defined at cell centers if depos_order is even

            int num_segments = 1;
            double shift = 0.0;
            if ( (depos_order % 2) == 0 ) { shift = 0.5; }

#if defined(WARPX_DIM_3D)

            // compute cell crossings in X-direction
            const auto i_old = static_cast<int>(x_old-shift);
            const auto i_new = static_cast<int>(x_new-shift);
            const int cell_crossings_x = std::abs(i_new-i_old);
            num_segments += cell_crossings_x;

            // compute cell crossings in Y-direction
            const auto j_old = static_cast<int>(y_old-shift);
            const auto j_new = static_cast<int>(y_new-shift);
            const int cell_crossings_y = std::abs(j_new-j_old);
            num_segments += cell_crossings_y;

            // compute cell crossings in Z-direction
            const auto k_old = static_cast<int>(z_old-shift);
            const auto k_new = static_cast<int>(z_new-shift);
            const int cell_crossings_z = std::abs(k_new-k_old);
            num_segments += cell_crossings_z;

            // need to assert that the number of cell crossings in each direction
            // is within the range permitted by the number of guard cells
            // e.g., if (num_segments > 7) ...

            // compute total change in particle position and the initial cell
            // locations in each direction used to find the position at cell crossings.
            const double dxp = x_new - x_old;
            const double dyp = y_new - y_old;
            const double dzp = z_new - z_old;
            const auto dirX_sign = static_cast<double>(dxp < 0. ? -1. : 1.);
            const auto dirY_sign = static_cast<double>(dyp < 0. ? -1. : 1.);
            const auto dirZ_sign = static_cast<double>(dzp < 0. ? -1. : 1.);
            double Xcell = 0., Ycell = 0., Zcell = 0.;
            if (num_segments > 1) {
                Xcell = static_cast<double>(i_old) + shift + 0.5*(1.-dirX_sign);
                Ycell = static_cast<double>(j_old) + shift + 0.5*(1.-dirY_sign);
                Zcell = static_cast<double>(k_old) + shift + 0.5*(1.-dirZ_sign);
            }

            // loop over the number of segments and deposit
            const Compute_shape_factor< depos_order-1 > compute_shape_factor_cell;
            const Compute_shape_factor_pair< depos_order > compute_shape_factors_node;
            double dxp_seg, dyp_seg, dzp_seg;
            double x0_new, y0_new, z0_new;
            double x0_old = x_old;
            double y0_old = y_old;
            double z0_old = z_old;

            for (int ns=0; ns<num_segments; ns++) {

                if (ns == num_segments-1) { // final segment

                    x0_new = x_new;
                    y0_new = y_new;
                    z0_new = z_new;
                    dxp_seg = x0_new - x0_old;
                    dyp_seg = y0_new - y0_old;
                    dzp_seg = z0_new - z0_old;

                }
                else {

                    x0_new = Xcell + dirX_sign;
                    y0_new = Ycell + dirY_sign;
                    z0_new = Zcell + dirZ_sign;
                    dxp_seg = x0_new - x0_old;
                    dyp_seg = y0_new - y0_old;
                    dzp_seg = z0_new - z0_old;

                    if ( (dyp == 0. || std::abs(dxp_seg) < std::abs(dxp/dyp*dyp_seg))
                      && (dzp == 0. || std::abs(dxp_seg) < std::abs(dxp/dzp*dzp_seg)) ) {
                        Xcell = x0_new;
                        dyp_seg = dyp/dxp*dxp_seg;
                        dzp_seg = dzp/dxp*dxp_seg;
                        y0_new = y0_old + dyp_seg;
                        z0_new = z0_old + dzp_seg;
                    }
                    else if (dzp == 0. || std::abs(dyp_seg) < std::abs(dyp/dzp*dzp_seg)) {
                        Ycell = y0_new;
                        dxp_seg = dxp/dyp*dyp_seg;
                        dzp_seg = dzp/dyp*dyp_seg;
                        x0_new = x0_old + dxp_seg;
                        z0_new = z0_old + dzp_seg;
                    }
                    else {
                        Zcell = z0_new;
                        dxp_seg = dxp/dzp*dzp_seg;
                        dyp_seg = dyp/dzp*dzp_seg;
                        x0_new = x0_old + dxp_seg;
                        y0_new = y0_old + dyp_seg;
                    }

                }

                // compute the segment factors (each equal to dt_seg/dt for nonzero dxp, dyp, or dzp)
                const auto seg_factor_x = static_cast<double>(dxp == 0. ? 1. : dxp_seg/dxp);
                const auto seg_factor_y = static_cast<double>(dyp == 0. ? 1. : dyp_seg/dyp);
                const auto seg_factor_z = static_cast<double>(dzp == 0. ? 1. : dzp_seg/dzp);

                // compute cell-based weights using the average segment position
                double sx_cell[depos_order] = {0.};
                double sy_cell[depos_order] = {0.};
                double sz_cell[depos_order] = {0.};
                double const x0_bar = (x0_new + x0_old)/2.0;
                double const y0_bar = (y0_new + y0_old)/2.0;
                double const z0_bar = (z0_new + z0_old)/2.0;
                const int i0_cell = compute_shape_factor_cell( sx_cell, x0_bar-0.5 );
                const int j0_cell = compute_shape_factor_cell( sy_cell, y0_bar-0.5 );
                const int k0_cell = compute_shape_factor_cell( sz_cell, z0_bar-0.5 );

                if constexpr (depos_order >= 3) { // higher-order correction to the cell-based weights
                    const Compute_shape_factor_pair<depos_order-1> compute_shape_factors_cell;
                    double sx_old_cell[depos_order] = {0.};
                    double sx_new_cell[depos_order] = {0.};
                    double sy_old_cell[depos_order] = {0.};
                    double sy_new_cell[depos_order] = {0.};
                    double sz_old_cell[depos_order] = {0.};
                    double sz_new_cell[depos_order] = {0.};
                    const int i0_cell_2 = compute_shape_factors_cell( sx_old_cell, sx_new_cell, x0_old-0.5, x0_new-0.5 );
                    const int j0_cell_2 = compute_shape_factors_cell( sy_old_cell, sy_new_cell, y0_old-0.5, y0_new-0.5 );
                    const int k0_cell_2 = compute_shape_factors_cell( sz_old_cell, sz_new_cell, z0_old-0.5, z0_new-0.5 );
                    ignore_unused(i0_cell_2, j0_cell_2, k0_cell_2);
                    for (int m=0; m<depos_order; m++) {
                        sx_cell[m] = (4.0*sx_cell[m] + sx_old_cell[m] + sx_new_cell[m])/6.0;
                        sy_cell[m] = (4.0*sy_cell[m] + sy_old_cell[m] + sy_new_cell[m])/6.0;
                        sz_cell[m] = (4.0*sz_cell[m] + sz_old_cell[m] + sz_new_cell[m])/6.0;
                    }
                }

                // compute node-based weights using the old and new segment positions
                double sx_old_node[depos_order+1] = {0.};
                double sx_new_node[depos_order+1] = {0.};
                double sy_old_node[depos_order+1] = {0.};
                double sy_new_node[depos_order+1] = {0.};
                double sz_old_node[depos_order+1] = {0.};
                double sz_new_node[depos_order+1] = {0.};
                const int i0_node = compute_shape_factors_node( sx_old_node, sx_new_node, x0_old, x0_new );
                const int j0_node = compute_shape_factors_node( sy_old_node, sy_new_node, y0_old, y0_new );
                const int k0_node = compute_shape_factors_node( sz_old_node, sz_new_node, z0_old, z0_new );

                // deposit Jx for this segment
                amrex::Real this_Jx;
                for (int i=0; i<=depos_order-1; i++) {
                    for (int j=0; j<=depos_order; j++) {
                        for (int k=0; k<=depos_order; k++) {
                            this_Jx = wqx*sx_cell[i]*( sy_old_node[j]*sz_old_node[k]*one_third
                                                     + sy_old_node[j]*sz_new_node[k]*one_sixth
                                                     + sy_new_node[j]*sz_old_node[k]*one_sixth
                                                     + sy_new_node[j]*sz_new_node[k]*one_third )*seg_factor_x;
                            amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i0_cell+i, lo.y+j0_node+j, lo.z+k0_node+k), this_Jx);
                        }
                    }
                }

                // deposit Jy for this segment
                amrex::Real this_Jy;
                for (int i=0; i<=depos_order; i++) {
                    for (int j=0; j<=depos_order-1; j++) {
                        for (int k=0; k<=depos_order; k++) {
                            this_Jy = wqy*sy_cell[j]*( sx_old_node[i]*sz_old_node[k]*one_third
                                                     + sx_old_node[i]*sz_new_node[k]*one_sixth
                                                     + sx_new_node[i]*sz_old_node[k]*one_sixth
                                                     + sx_new_node[i]*sz_new_node[k]*one_third )*seg_factor_y;
                            amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i0_node+i, lo.y+j0_cell+j, lo.z+k0_node+k), this_Jy);
                        }
                    }
                }

                // deposit Jz for this segment
                amrex::Real this_Jz;
                for (int i=0; i<=depos_order; i++) {
                    for (int j=0; j<=depos_order; j++) {
                        for (int k=0; k<=depos_order-1; k++) {
                            this_Jz = wqz*sz_cell[k]*( sx_old_node[i]*sy_old_node[j]*one_third
                                                     + sx_old_node[i]*sy_new_node[j]*one_sixth
                                                     + sx_new_node[i]*sy_old_node[j]*one_sixth
                                                     + sx_new_node[i]*sy_new_node[j]*one_third )*seg_factor_z;
                            amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i0_node+i, lo.y+j0_node+j, lo.z+k0_cell+k), this_Jz);
                        }
                    }
                }

                // update old segment values
                if (ns < num_segments-1) {
                    x0_old = x0_new;
                    y0_old = y0_new;
                    z0_old = z0_new;
                }

            } // end loop over segments

#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)

            // compute cell crossings in X-direction
            const auto i_old = static_cast<int>(x_old-shift);
            const auto i_new = static_cast<int>(x_new-shift);
            const int cell_crossings_x = std::abs(i_new-i_old);
            num_segments += cell_crossings_x;

            // compute cell crossings in Z-direction
            const auto k_old = static_cast<int>(z_old-shift);
            const auto k_new = static_cast<int>(z_new-shift);
            const int cell_crossings_z = std::abs(k_new-k_old);
            num_segments += cell_crossings_z;

            // need to assert that the number of cell crossings in each direction
            // is within the range permitted by the number of guard cells
            // e.g., if (num_segments > 5) ...

            // compute total change in particle position and the initial cell
            // locations in each direction used to find the position at cell crossings.
            const double dxp = x_new - x_old;
            const double dzp = z_new - z_old;
            const auto dirX_sign = static_cast<double>(dxp < 0. ? -1. : 1.);
            const auto dirZ_sign = static_cast<double>(dzp < 0. ? -1. : 1.);
            double Xcell = 0., Zcell = 0.;
            if (num_segments > 1) {
                Xcell = static_cast<double>(i_old) + shift + 0.5*(1.-dirX_sign);
                Zcell = static_cast<double>(k_old) + shift + 0.5*(1.-dirZ_sign);
            }

            // loop over the number of segments and deposit
            const Compute_shape_factor< depos_order-1 > compute_shape_factor_cell;
            const Compute_shape_factor_pair< depos_order > compute_shape_factors_node;
            double dxp_seg, dzp_seg;
            double x0_new, z0_new;
            double x0_old = x_old;
            double z0_old = z_old;

            for (int ns=0; ns<num_segments; ns++) {

                if (ns == num_segments-1) { // final segment

                    x0_new = x_new;
                    z0_new = z_new;
                    dxp_seg = x0_new - x0_old;
                    dzp_seg = z0_new - z0_old;

                }
                else {

                    x0_new = Xcell + dirX_sign;
                    z0_new = Zcell + dirZ_sign;
                    dxp_seg = x0_new - x0_old;
                    dzp_seg = z0_new - z0_old;

                    if (dzp == 0. || std::abs(dxp_seg) < std::abs(dxp/dzp*dzp_seg)) {
                        Xcell = x0_new;
                        dzp_seg = dzp/dxp*dxp_seg;
                        z0_new = z0_old + dzp_seg;
                    }
                    else {
                        Zcell = z0_new;
                        dxp_seg = dxp/dzp*dzp_seg;
                        x0_new = x0_old + dxp_seg;
                    }

                }

                // compute the segment factors (each equal to dt_seg/dt for nonzero dxp, or dzp)
                const auto seg_factor_x = static_cast<double>(dxp == 0. ? 1. : dxp_seg/dxp);
                const auto seg_factor_z = static_cast<double>(dzp == 0. ? 1. : dzp_seg/dzp);

                // compute cell-based weights using the average segment position
                double sx_cell[depos_order] = {0.};
                double sz_cell[depos_order] = {0.};
                double const x0_bar = (x0_new + x0_old)/2.0;
                double const z0_bar = (z0_new + z0_old)/2.0;
                const int i0_cell = compute_shape_factor_cell( sx_cell, x0_bar-0.5 );
                const int k0_cell = compute_shape_factor_cell( sz_cell, z0_bar-0.5 );

                if constexpr (depos_order >= 3) { // higher-order correction to the cell-based weights
                    const Compute_shape_factor_pair<depos_order-1> compute_shape_factors_cell;
                    double sx_old_cell[depos_order] = {0.};
                    double sx_new_cell[depos_order] = {0.};
                    double sz_old_cell[depos_order] = {0.};
                    double sz_new_cell[depos_order] = {0.};
                    const int i0_cell_2 = compute_shape_factors_cell( sx_old_cell, sx_new_cell, x0_old-0.5, x0_new-0.5 );
                    const int k0_cell_2 = compute_shape_factors_cell( sz_old_cell, sz_new_cell, z0_old-0.5, z0_new-0.5 );
                    ignore_unused(i0_cell_2, k0_cell_2);
                    for (int m=0; m<depos_order; m++) {
                        sx_cell[m] = (4.0*sx_cell[m] + sx_old_cell[m] + sx_new_cell[m])/6.0;
                        sz_cell[m] = (4.0*sz_cell[m] + sz_old_cell[m] + sz_new_cell[m])/6.0;
                    }
                }

                // compute node-based weights using the old and new segment positions
                double sx_old_node[depos_order+1] = {0.};
                double sx_new_node[depos_order+1] = {0.};
                double sz_old_node[depos_order+1] = {0.};
                double sz_new_node[depos_order+1] = {0.};
                const int i0_node = compute_shape_factors_node( sx_old_node, sx_new_node, x0_old, x0_new );
                const int k0_node = compute_shape_factors_node( sz_old_node, sz_new_node, z0_old, z0_new );

                // deposit Jx for this segment
                amrex::Real this_Jx;
                for (int i=0; i<=depos_order-1; i++) {
                    for (int k=0; k<=depos_order; k++) {
                        this_Jx = wqx*sx_cell[i]*(sz_old_node[k] + sz_new_node[k])/2.0_rt*seg_factor_x;
                        amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i0_cell+i, lo.y+k0_node+k, 0, 0), this_Jx);
#if defined(WARPX_DIM_RZ)
                        Complex xy_mid = xy_mid0; // Throughout the following loop, xy_mid takes the value e^{i m theta}
                        for (int imode=1 ; imode < n_rz_azimuthal_modes ; imode++) {
                            // The factor 2 comes from the normalization of the modes
                            const Complex djr_cmplx = 2._rt*this_Jx*xy_mid;
                            amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i0_cell+i, lo.y+k0_node+k, 0, 2*imode-1), djr_cmplx.real());
                            amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i0_cell+i, lo.y+k0_node+k, 0, 2*imode), djr_cmplx.imag());
                            xy_mid = xy_mid*xy_mid0;
                        }
#endif
                    }
                }

                // deposit out-of-plane Jy for this segment
                const auto seg_factor_y = std::min(seg_factor_x,seg_factor_z);
                amrex::Real this_Jy;
                for (int i=0; i<=depos_order; i++) {
                    for (int k=0; k<=depos_order; k++) {
                        this_Jy = wqy*( sx_old_node[i]*sz_old_node[k]*one_third
                                      + sx_old_node[i]*sz_new_node[k]*one_sixth
                                      + sx_new_node[i]*sz_old_node[k]*one_sixth
                                      + sx_new_node[i]*sz_new_node[k]*one_third )*seg_factor_y;
                        amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i0_node+i, lo.y+k0_node+k, 0, 0), this_Jy);
#if defined(WARPX_DIM_RZ)
                        Complex xy_mid = xy_mid0;
                        // Throughout the following loop, xy_ takes the value e^{i m theta_}
                        for (int imode=1 ; imode < n_rz_azimuthal_modes ; imode++) {
                            // The factor 2 comes from the normalization of the modes
                            const Complex djy_cmplx = 2._rt*this_Jy*xy_mid;
                            amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i0_node+i, lo.y+k0_node+k, 0, 2*imode-1), djy_cmplx.real());
                            amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i0_node+i, lo.y+k0_node+k, 0, 2*imode), djy_cmplx.imag());
                            xy_mid = xy_mid*xy_mid0;
                        }
#endif
                    }
                }

                // deposit Jz for this segment
                amrex::Real this_Jz;
                for (int i=0; i<=depos_order; i++) {
                    for (int k=0; k<=depos_order-1; k++) {
                        this_Jz = wqz*sz_cell[k]*(sx_old_node[i] + sx_new_node[i])/2.0_rt*seg_factor_z;
                        amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i0_node+i, lo.y+k0_cell+k, 0, 0), this_Jz);
#if defined(WARPX_DIM_RZ)
                        Complex xy_mid = xy_mid0; // Throughout the following loop, xy_mid takes the value e^{i m theta}
                        for (int imode=1 ; imode < n_rz_azimuthal_modes ; imode++) {
                            // The factor 2 comes from the normalization of the modes
                            const Complex djz_cmplx = 2._rt*this_Jz*xy_mid;
                            amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i0_node+i, lo.y+k0_cell+k, 0, 2*imode-1), djz_cmplx.real());
                            amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i0_node+i, lo.y+k0_cell+k, 0, 2*imode), djz_cmplx.imag());
                            xy_mid = xy_mid*xy_mid0;
                        }
#endif
                    }
                }

                // update old segment values
                if (ns < num_segments-1) {
                    x0_old = x0_new;
                    z0_old = z0_new;
                }

            } // end loop over segments

#elif defined(WARPX_DIM_1D_Z)

            // compute cell crossings in Z-direction
            const auto k_old = static_cast<int>(z_old-shift);
            const auto k_new = static_cast<int>(z_new-shift);
            const int cell_crossings_z = std::abs(k_new-k_old);
            num_segments += cell_crossings_z;

            // need to assert that the number of cell crossings in each direction
            // is within the range permitted by the number of guard cells
            // e.g., if (num_segments > 3) ...

            // compute dzp and the initial cell location used to find the cell crossings.
            double const dzp = z_new - z_old;
            const auto dirZ_sign = static_cast<double>(dzp < 0. ? -1. : 1.);
            double Zcell = static_cast<double>(k_old) + shift + 0.5*(1.-dirZ_sign);

            // loop over the number of segments and deposit
            const Compute_shape_factor< depos_order-1 > compute_shape_factor_cell;
            const Compute_shape_factor_pair< depos_order > compute_shape_factors_node;
            double dzp_seg;
            double z0_new;
            double z0_old = z_old;

            for (int ns=0; ns<num_segments; ns++) {

                if (ns == num_segments-1) { // final segment
                    z0_new = z_new;
                    dzp_seg = z0_new - z0_old;
                }
                else {
                    Zcell = Zcell + dirZ_sign;
                    z0_new = Zcell;
                    dzp_seg = z0_new - z0_old;
                }

                // compute the segment factor (equal to dt_seg/dt for nonzero dzp)
                const auto seg_factor = static_cast<double>(dzp == 0. ? 1. : dzp_seg/dzp);

                // compute cell-based weights using the average segment position
                double sz_cell[depos_order] = {0.};
                double const z0_bar = (z0_new + z0_old)/2.0;
                const int k0_cell = compute_shape_factor_cell( sz_cell, z0_bar-0.5 );

                if constexpr (depos_order >= 3) { // higher-order correction to the cell-based weights
                    const Compute_shape_factor_pair<depos_order-1> compute_shape_factors_cell;
                    double sz_old_cell[depos_order] = {0.};
                    double sz_new_cell[depos_order] = {0.};
                    const int k0_cell_2 = compute_shape_factors_cell( sz_old_cell, sz_new_cell, z0_old-0.5, z0_new-0.5 );
                    ignore_unused(k0_cell_2);
                    for (int m=0; m<depos_order; m++) {
                        sz_cell[m] = (4.0*sz_cell[m] + sz_old_cell[m] + sz_new_cell[m])/6.0;
                    }
                }

                // compute node-based weights using the old and new segment positions
                double sz_old_node[depos_order+1] = {0.};
                double sz_new_node[depos_order+1] = {0.};
                const int k0_node = compute_shape_factors_node( sz_old_node, sz_new_node, z0_old, z0_new );

                // deposit out-of-plane Jx and Jy for this segment
                for (int k=0; k<=depos_order; k++) {
                    const amrex::Real weight = 0.5_rt*(sz_old_node[k] + sz_new_node[k])*seg_factor;
                    amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+k0_node+k, 0, 0), wqx*weight);
                    amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+k0_node+k, 0, 0), wqy*weight);
                }

                // deposit Jz for this segment
                for (int k=0; k<=depos_order-1; k++) {
                    const amrex::Real this_Jz = wqz*sz_cell[k]*seg_factor;
                    amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+k0_cell+k, 0, 0), this_Jz);
                }

                // update old segment values
                if (ns < num_segments-1) {
                    z0_old = z0_new;
                }

            }

#endif
        }
    );
}

/**
 * \brief Vay current deposition
 * (<a href="https://doi.org/10.1016/j.jcp.2013.03.010"> Vay et al, 2013</a>)
 * for thread \c thread_num: deposit \c D in real space and store the result in
 * \c Dx_fab, \c Dy_fab, \c Dz_fab
 *
 * \tparam depos_order  deposition order
 * \param[in] GetPosition  Functor that returns the particle position
 * \param[in] wp           Pointer to array of particle weights
 * \param[in] uxp,uyp,uzp  Pointer to arrays of particle momentum along \c x
 * \param[in] ion_lev      Pointer to array of particle ionization level. This is
                           required to have the charge of each macroparticle since \c q
                           is a scalar. For non-ionizable species, \c ion_lev is \c null
 * \param[in,out] Dx_fab,Dy_fab,Dz_fab FArrayBox of Vay current density, either full array or tile
 * \param[in] np_to_deposit Number of particles for which current is deposited
 * \param[in] dt           Time step for particle level
 * \param[in] relative_time Time at which to deposit D, relative to the time of the
 *                          current positions of the particles. When different than 0,
 *                          the particle position will be temporarily modified to match
 *                          the time of the deposition.
 * \param[in] dinv         3D cell size inverse
 * \param[in] xyzmin       3D lower bounds of physical domain
 * \param[in] lo           Dimension-agnostic lower bounds of index domain
 * \param[in] q            Species charge
 * \param[in] n_rz_azimuthal_modes Number of azimuthal modes in RZ geometry
 */
template <int depos_order>
void doVayDepositionShapeN (const GetParticlePosition<PIdx>& GetPosition,
                            const amrex::ParticleReal* const wp,
                            const amrex::ParticleReal* const uxp,
                            const amrex::ParticleReal* const uyp,
                            const amrex::ParticleReal* const uzp,
                            const int* const ion_lev,
                            amrex::FArrayBox& Dx_fab,
                            amrex::FArrayBox& Dy_fab,
                            amrex::FArrayBox& Dz_fab,
                            long np_to_deposit,
                            amrex::Real dt,
                            amrex::Real relative_time,
                            const amrex::XDim3 & dinv,
                            const amrex::XDim3 & xyzmin,
                            amrex::Dim3 lo,
                            amrex::Real q,
                            [[maybe_unused]]int n_rz_azimuthal_modes)
{
    using namespace amrex::literals;

#if defined(WARPX_DIM_RZ)
    amrex::ignore_unused(GetPosition,
        wp, uxp, uyp, uzp, ion_lev, Dx_fab, Dy_fab, Dz_fab,
        np_to_deposit, dt, relative_time, dinv, xyzmin, lo, q);
    WARPX_ABORT_WITH_MESSAGE("Vay deposition not implemented in RZ geometry");
#endif

#if defined(WARPX_DIM_1D_Z)
    amrex::ignore_unused(GetPosition,
        wp, uxp, uyp, uzp, ion_lev, Dx_fab, Dy_fab, Dz_fab,
        np_to_deposit, dt, relative_time, dinv, xyzmin, lo, q);
    WARPX_ABORT_WITH_MESSAGE("Vay deposition not implemented in 1D geometry");
#endif

#if !(defined WARPX_DIM_RZ || defined WARPX_DIM_1D_Z)

    // If ion_lev is a null pointer, then do_ionization=0, else do_ionization=1
    const bool do_ionization = ion_lev;

    // Inverse of time step
    const amrex::Real invdt = 1._rt / dt;

    const amrex::Real invvol = dinv.x*dinv.y*dinv.z;

    // Allocate temporary arrays
#if defined(WARPX_DIM_3D)
    AMREX_ALWAYS_ASSERT(Dx_fab.box() == Dy_fab.box() && Dx_fab.box() == Dz_fab.box());
    amrex::FArrayBox temp_fab{Dx_fab.box(), 4};
#elif defined(WARPX_DIM_XZ)
    AMREX_ALWAYS_ASSERT(Dx_fab.box() == Dz_fab.box());
    amrex::FArrayBox temp_fab{Dx_fab.box(), 2};
#endif
    temp_fab.setVal<amrex::RunOn::Device>(0._rt);
    amrex::Array4<amrex::Real> const& temp_arr = temp_fab.array();

    // Inverse of light speed squared
    const amrex::Real invcsq = 1._rt / (PhysConst::c * PhysConst::c);

    // Arrays where D will be stored
    amrex::Array4<amrex::Real> const& Dx_arr = Dx_fab.array();
    amrex::Array4<amrex::Real> const& Dy_arr = Dy_fab.array();
    amrex::Array4<amrex::Real> const& Dz_arr = Dz_fab.array();

    // Loop over particles and deposit (Dx,Dy,Dz) into Dx_fab, Dy_fab and Dz_fab
    amrex::ParallelFor(np_to_deposit, [=] AMREX_GPU_DEVICE (long ip)
    {
        // Inverse of Lorentz factor gamma
        const amrex::Real invgam = 1._rt / std::sqrt(1._rt + uxp[ip] * uxp[ip] * invcsq
                                                           + uyp[ip] * uyp[ip] * invcsq
                                                           + uzp[ip] * uzp[ip] * invcsq);
        // Product of particle charges and weights
        amrex::Real wq = q * wp[ip];
        if (do_ionization) { wq *= ion_lev[ip]; }

        // Current particle positions (in physical units)
        amrex::ParticleReal xp, yp, zp;
        GetPosition(ip, xp, yp, zp);

        // Particle velocities
        const amrex::Real vx = uxp[ip] * invgam;
        const amrex::Real vy = uyp[ip] * invgam;
        const amrex::Real vz = uzp[ip] * invgam;

        // Modify the particle position to match the time of the deposition
        xp += relative_time * vx;
        yp += relative_time * vy;
        zp += relative_time * vz;

        // Current and old particle positions in grid units
        // Keep these double to avoid bug in single precision.
        double const x_new = (xp - xyzmin.x + 0.5_rt*dt*vx) * dinv.x;
        double const x_old = (xp - xyzmin.x - 0.5_rt*dt*vx) * dinv.x;
#if defined(WARPX_DIM_3D)
        // Keep these double to avoid bug in single precision.
        double const y_new = (yp - xyzmin.y + 0.5_rt*dt*vy) * dinv.y;
        double const y_old = (yp - xyzmin.y - 0.5_rt*dt*vy) * dinv.y;
#endif
        // Keep these double to avoid bug in single precision.
        double const z_new = (zp - xyzmin.z + 0.5_rt*dt*vz) * dinv.z;
        double const z_old = (zp - xyzmin.z - 0.5_rt*dt*vz) * dinv.z;

        // Shape factor arrays for current and old positions (nodal)
        // Keep these double to avoid bug in single precision.
        double sx_new[depos_order+1] = {0.};
        double sx_old[depos_order+1] = {0.};
#if defined(WARPX_DIM_3D)
        // Keep these double to avoid bug in single precision.
        double sy_new[depos_order+1] = {0.};
        double sy_old[depos_order+1] = {0.};
#endif
        // Keep these double to avoid bug in single precision.
        double sz_new[depos_order+1] = {0.};
        double sz_old[depos_order+1] = {0.};

        // Compute shape factors for current positions

        // i_new leftmost grid point in x that the particle touches
        // sx_new shape factor along x for the centering of each current
        Compute_shape_factor< depos_order > const compute_shape_factor;
        const int i_new = compute_shape_factor(sx_new, x_new);
#if defined(WARPX_DIM_3D)
        // j_new leftmost grid point in y that the particle touches
        // sy_new shape factor along y for the centering of each current
        const int j_new = compute_shape_factor(sy_new, y_new);
#endif
        // k_new leftmost grid point in z that the particle touches
        // sz_new shape factor along z for the centering of each current
        const int k_new = compute_shape_factor(sz_new, z_new);

        // Compute shape factors for old positions

        // i_old leftmost grid point in x that the particle touches
        // sx_old shape factor along x for the centering of each current
        const int i_old = compute_shape_factor(sx_old, x_old);
#if defined(WARPX_DIM_3D)
        // j_old leftmost grid point in y that the particle touches
        // sy_old shape factor along y for the centering of each current
        const int j_old = compute_shape_factor(sy_old, y_old);
#endif
        // k_old leftmost grid point in z that the particle touches
        // sz_old shape factor along z for the centering of each current
        const int k_old = compute_shape_factor(sz_old, z_old);

        // Deposit current into Dx_arr, Dy_arr and Dz_arr
#if defined(WARPX_DIM_XZ)

        const amrex::Real wqy = wq * vy * invvol;
        for (int k=0; k<=depos_order; k++) {
            for (int i=0; i<=depos_order; i++) {

                // Re-casting sx_new and sz_new from double to amrex::Real so that
                // Atomic::Add has consistent types in its argument
                auto const sxn_szn = static_cast<amrex::Real>(sx_new[i] * sz_new[k]);
                auto const sxo_szn = static_cast<amrex::Real>(sx_old[i] * sz_new[k]);
                auto const sxn_szo = static_cast<amrex::Real>(sx_new[i] * sz_old[k]);
                auto const sxo_szo = static_cast<amrex::Real>(sx_old[i] * sz_old[k]);

                if (i_new == i_old && k_new == k_old) {
                    // temp arrays for Dx and Dz
                    amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + k_new + k, 0, 0),
                        wq * invvol * invdt * (sxn_szn - sxo_szo));

                    amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + k_new + k, 0, 1),
                        wq * invvol * invdt * (sxn_szo - sxo_szn));

                    // Dy
                    amrex::Gpu::Atomic::AddNoRet(&Dy_arr(lo.x + i_new + i, lo.y + k_new + k, 0, 0),
                        wqy * 0.25_rt * (sxn_szn + sxn_szo + sxo_szn + sxo_szo));
                } else {
                    // temp arrays for Dx and Dz
                    amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + k_new + k, 0, 0),
                        wq * invvol * invdt * sxn_szn);

                    amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_old + i, lo.y + k_old + k, 0, 0),
                        - wq * invvol * invdt * sxo_szo);

                    amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + k_old + k, 0, 1),
                        wq * invvol * invdt * sxn_szo);

                    amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_old + i, lo.y + k_new + k, 0, 1),
                        - wq * invvol * invdt * sxo_szn);

                    // Dy
                    amrex::Gpu::Atomic::AddNoRet(&Dy_arr(lo.x + i_new + i, lo.y + k_new + k, 0, 0),
                        wqy * 0.25_rt * sxn_szn);

                    amrex::Gpu::Atomic::AddNoRet(&Dy_arr(lo.x + i_new + i, lo.y + k_old + k, 0, 0),
                        wqy * 0.25_rt * sxn_szo);

                    amrex::Gpu::Atomic::AddNoRet(&Dy_arr(lo.x + i_old + i, lo.y + k_new + k, 0, 0),
                        wqy * 0.25_rt * sxo_szn);

                    amrex::Gpu::Atomic::AddNoRet(&Dy_arr(lo.x + i_old + i, lo.y + k_old + k, 0, 0),
                        wqy * 0.25_rt * sxo_szo);
                }

            }
        }

#elif defined(WARPX_DIM_3D)

        for (int k=0; k<=depos_order; k++) {
            for (int j=0; j<=depos_order; j++) {

                auto const syn_szn = static_cast<amrex::Real>(sy_new[j] * sz_new[k]);
                auto const syo_szn = static_cast<amrex::Real>(sy_old[j] * sz_new[k]);
                auto const syn_szo = static_cast<amrex::Real>(sy_new[j] * sz_old[k]);
                auto const syo_szo = static_cast<amrex::Real>(sy_old[j] * sz_old[k]);

                for (int i=0; i<=depos_order; i++) {

                    auto const sxn_syn_szn = static_cast<amrex::Real>(sx_new[i]) * syn_szn;
                    auto const sxo_syn_szn = static_cast<amrex::Real>(sx_old[i]) * syn_szn;
                    auto const sxn_syo_szn = static_cast<amrex::Real>(sx_new[i]) * syo_szn;
                    auto const sxo_syo_szn = static_cast<amrex::Real>(sx_old[i]) * syo_szn;
                    auto const sxn_syn_szo = static_cast<amrex::Real>(sx_new[i]) * syn_szo;
                    auto const sxo_syn_szo = static_cast<amrex::Real>(sx_old[i]) * syn_szo;
                    auto const sxn_syo_szo = static_cast<amrex::Real>(sx_new[i]) * syo_szo;
                    auto const sxo_syo_szo = static_cast<amrex::Real>(sx_old[i]) * syo_szo;

                    if (i_new == i_old && j_new == j_old && k_new == k_old) {
                        // temp arrays for Dx, Dy and Dz
                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + j_new + j, lo.z + k_new + k, 0),
                            wq * invvol * invdt * (sxn_syn_szn - sxo_syo_szo));

                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + j_new + j, lo.z + k_new + k, 1),
                            wq * invvol * invdt * (sxn_syn_szo - sxo_syo_szn));

                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + j_new + j, lo.z + k_new + k, 2),
                            wq * invvol * invdt * (sxn_syo_szn - sxo_syn_szo));

                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + j_new + j, lo.z + k_new + k, 3),
                            wq * invvol * invdt * (sxo_syn_szn - sxn_syo_szo));
                    } else {
                        // temp arrays for Dx, Dy and Dz
                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + j_new + j, lo.z + k_new + k, 0),
                            wq * invvol * invdt * sxn_syn_szn);

                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_old + i, lo.y + j_old + j, lo.z + k_old + k, 0),
                            - wq * invvol * invdt * sxo_syo_szo);

                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + j_new + j, lo.z + k_old + k, 1),
                            wq * invvol * invdt * sxn_syn_szo);

                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_old + i, lo.y + j_old + j, lo.z + k_new + k, 1),
                            - wq * invvol * invdt * sxo_syo_szn);

                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + j_old + j, lo.z + k_new + k, 2),
                            wq * invvol * invdt * sxn_syo_szn);

                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_old + i, lo.y + j_new + j, lo.z + k_old + k, 2),
                            - wq * invvol * invdt * sxo_syn_szo);

                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_old + i, lo.y + j_new + j, lo.z + k_new + k, 3),
                            wq * invvol * invdt * sxo_syn_szn);

                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + j_old + j, lo.z + k_old + k, 3),
                            - wq * invvol * invdt * sxn_syo_szo);
                    }
                }
            }
        }
#endif
    } );

#if defined(WARPX_DIM_3D)
    amrex::ParallelFor(Dx_fab.box(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        const amrex::Real t_a = temp_arr(i,j,k,0);
        const amrex::Real t_b = temp_arr(i,j,k,1);
        const amrex::Real t_c = temp_arr(i,j,k,2);
        const amrex::Real t_d = temp_arr(i,j,k,3);
        Dx_arr(i,j,k) += (1._rt/6._rt)*(2_rt*t_a       + t_b       + t_c - 2._rt*t_d);
        Dy_arr(i,j,k) += (1._rt/6._rt)*(2_rt*t_a       + t_b - 2._rt*t_c       + t_d);
        Dz_arr(i,j,k) += (1._rt/6._rt)*(2_rt*t_a - 2._rt*t_b       + t_c       + t_d);
    });
#elif defined(WARPX_DIM_XZ)
    amrex::ParallelFor(Dx_fab.box(), [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
    {
        const amrex::Real t_a = temp_arr(i,j,0,0);
        const amrex::Real t_b = temp_arr(i,j,0,1);
        Dx_arr(i,j,0) += (0.5_rt)*(t_a + t_b);
        Dz_arr(i,j,0) += (0.5_rt)*(t_a - t_b);
    });
#endif
    // Synchronize so that temp_fab can be safely deallocated in its destructor
    amrex::Gpu::streamSynchronize();

#endif // #if !(defined WARPX_DIM_RZ || defined WARPX_DIM_1D_Z)
}
#endif // WARPX_CURRENTDEPOSITION_H_
