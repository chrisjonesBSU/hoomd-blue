// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __EVALUATOR_PAIR_GB_FULL_H__
#define __EVALUATOR_PAIR_GB_FULL_H__

#ifndef __HIPCC__
#include <string>
#endif

#ifdef ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

#define HOOMD_GB_MIN(i, j) ((i > j) ? j : i)
#define HOOMD_GB_MAX(i, j) ((i > j) ? i : j)

#include "hoomd/VectorMath.h"

/*! \file EvaluatorPairGBFull.h
    \brief Defines an evaluator class for the full Gay-Berne potential with
           orientation-dependent well depth (Brown et al. 2009).
*/

#ifdef __HIPCC__
#define HOSTDEVICE __host__ __device__
#define DEVICE __device__
#else
#define HOSTDEVICE
#define DEVICE
#endif

namespace hoomd
    {
namespace md
    {
/*!
 * Full Gay-Berne potential for identical uniaxial particles, with
 * orientation-dependent well depth (eta and chi factors) matching the
 * Brown et al. (2009) formulation.
 * When e_i_perp == e_i_par == e_j_perp == e_j_par == 1, chi reduces to a
 * constant.
 */

class EvaluatorPairGBFull
    {
    public:
    struct param_type
        {
        Scalar epsilon;   //!< Overall energy scale.
        Scalar lperp;     //!< Semiaxis length perpendicular to the particle orientation.
        Scalar lpar;      //!< Semiaxis length parallel to the particle orientation.
        Scalar e_i_perp;  //!< Well depth of particle i perpendicular to its axis.
        Scalar e_i_par;   //!< Well depth of particle i parallel to its axis.
        Scalar e_j_perp;  //!< Well depth of particle j perpendicular to its axis.
        Scalar e_j_par;   //!< Well depth of particle j parallel to its axis.
        Scalar mu;        //!< Exponent for orientational dependence of well depth.
        Scalar upsilon;   //!< Exponent for shape-anisotropy prefactor.

        DEVICE void load_shared(char*& ptr, unsigned int& available_bytes) { }

        HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const { }

#ifdef ENABLE_HIP
        void set_memory_hint() const { }
#endif

        HOSTDEVICE param_type()
            {
            epsilon  = 0;
            lperp    = 0;
            lpar     = 0;
            e_i_perp = 1;
            e_i_par  = 1;
            e_j_perp = 1;
            e_j_par  = 1;
            mu       = 1;
            upsilon  = Scalar(1.0);
            }

#ifndef __HIPCC__

        param_type(pybind11::dict v, bool managed = false)
            {
            epsilon  = v["epsilon"].cast<Scalar>();
            lperp    = v["lperp"].cast<Scalar>();
            lpar     = v["lpar"].cast<Scalar>();
            e_i_perp = v.contains("e_i_perp") ? v["e_i_perp"].cast<Scalar>() : Scalar(1.0);
            e_i_par  = v.contains("e_i_par")  ? v["e_i_par"].cast<Scalar>()  : Scalar(1.0);
            e_j_perp = v.contains("e_j_perp") ? v["e_j_perp"].cast<Scalar>() : Scalar(1.0);
            e_j_par  = v.contains("e_j_par")  ? v["e_j_par"].cast<Scalar>()  : Scalar(1.0);
            mu       = v.contains("mu")       ? v["mu"].cast<Scalar>()       : Scalar(1.0);
            upsilon  = v.contains("upsilon")  ? v["upsilon"].cast<Scalar>()  : Scalar(1.0);
            }

        pybind11::dict toPython()
            {
            pybind11::dict v;
            v["epsilon"]  = epsilon;
            v["lperp"]    = lperp;
            v["lpar"]     = lpar;
            v["e_i_perp"] = e_i_perp;
            v["e_i_par"]  = e_i_par;
            v["e_j_perp"] = e_j_perp;
            v["e_j_par"]  = e_j_par;
            v["mu"]       = mu;
            v["upsilon"]  = upsilon;
            return v;
            }

#endif
        }
#if HOOMD_LONGREAL_SIZE == 32
        __attribute__((aligned(8)));
#else
        __attribute__((aligned(16)));
#endif

    struct shape_type
        {
        DEVICE void load_shared(char*& ptr, unsigned int& available_bytes) { }

        HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const { }

        HOSTDEVICE shape_type() { }

#ifndef __HIPCC__

        shape_type(pybind11::object shape_params, bool managed) { }

        pybind11::object toPython()
            {
            return pybind11::none();
            }
#endif

#ifdef ENABLE_HIP
        void set_memory_hint() const { }
#endif
        };

    HOSTDEVICE EvaluatorPairGBFull(const Scalar3& _dr,
                                   const Scalar4& _qi,
                                   const Scalar4& _qj,
                                   const Scalar _rcutsq,
                                   const param_type& _params)
        : dr(_dr), rcutsq(_rcutsq), qi(_qi), qj(_qj), epsilon(_params.epsilon),
          lperp(_params.lperp), lpar(_params.lpar),
          e_i_perp(_params.e_i_perp), e_i_par(_params.e_i_par),
          e_j_perp(_params.e_j_perp), e_j_par(_params.e_j_par),
          mu(_params.mu), upsilon(_params.upsilon)
        {
        }

    HOSTDEVICE static bool needsShape()
        {
        return false;
        }

    HOSTDEVICE static bool needsTags()
        {
        return false;
        }

    HOSTDEVICE static bool needsCharge()
        {
        return false;
        }

    HOSTDEVICE static bool constexpr implementsEnergyShift()
        {
        return true;
        }

    HOSTDEVICE void setShape(const shape_type* shapei, const shape_type* shapej) { }

    HOSTDEVICE void setTags(unsigned int tagi, unsigned int tagj) { }

    HOSTDEVICE void setCharge(Scalar qi, Scalar qj) { }

    HOSTDEVICE bool evaluate(Scalar3& force,
                             Scalar& pair_eng,
                             bool energy_shift,
                             Scalar3& torque_i,
                             Scalar3& torque_j)
        {
        Scalar rsq = dot(dr, dr);
        Scalar r = fast::sqrt(rsq);
        vec3<Scalar> unitr = fast::rsqrt(dot(dr, dr)) * dr;

        // obtain rotation matrices (space->body); last row is the symmetry axis
        rotmat3<Scalar> rotA(conj(qi));
        rotmat3<Scalar> rotB(conj(qj));
        vec3<Scalar> a3 = rotA.row2;
        vec3<Scalar> b3 = rotB.row2;

        Scalar ca  = dot(a3, unitr);
        Scalar cb  = dot(b3, unitr);
        Scalar cab = dot(a3, b3);

        // ---------------------------------------------------------------
        // sigma: orientation-dependent contact distance (Allen & Germano).
        // ---------------------------------------------------------------
        Scalar lperpsq    = lperp * lperp;
        Scalar lparsq     = lpar  * lpar;
        Scalar shape_chi  = (lparsq - lperpsq) / (lparsq + lperpsq);
        Scalar shape_chic = shape_chi * cab;

        Scalar shape_chi_fact = shape_chi / (Scalar(1.0) - shape_chic * shape_chic);
        vec3<Scalar> kappa
            = Scalar(0.5) * r / lperpsq
              * (unitr
                 - shape_chi_fact
                       * ((ca - shape_chic * cb) * a3 + (cb - shape_chic * ca) * b3));

        Scalar phi   = Scalar(0.5) * dot(dr, kappa) / rsq;
        Scalar sigma = fast::rsqrt(phi);

        Scalar sigma_min = Scalar(2.0) * HOOMD_GB_MIN(lperp, lpar);
        Scalar zeta      = (r - sigma + sigma_min) / sigma_min;
        Scalar zetasq    = zeta * zeta;

        // Radial cutoff at r >= r_cut (Brown 2009 rc = 4*sigma_0; matches LAMMPS).
        if (rsq >= rcutsq || epsilon == Scalar(0.0))
            return false;

        // ---------------------------------------------------------------
        // chi_e: orientation-dependent well-depth factor (LAMMPS "chi").
        // B_k = w_k_perp*I + (w_k_par - w_k_perp)*e_k x e_k
        // w_k_* = epsilon_k_*^(-1/mu)
        // B12^{-1} via double Sherman-Morrison.
        // chi_e = (2 * r_hat . B12^{-1} . r_hat)^mu
        // ---------------------------------------------------------------
        Scalar mu_inv  = Scalar(1.0) / mu;
        Scalar wi_perp = fast::pow(e_i_perp, -mu_inv);
        Scalar wi_par  = fast::pow(e_i_par,  -mu_inv);
        Scalar wj_perp = fast::pow(e_j_perp, -mu_inv);
        Scalar wj_par  = fast::pow(e_j_par,  -mu_inv);

        Scalar W0  = wi_perp + wj_perp;
        Scalar dWi = wi_par  - wi_perp;
        Scalar dWj = wj_par  - wj_perp;

        // First SM step: M1 = W0*I + dWi*(a3 x a3)
        Scalar fac_i             = dWi / (W0 + dWi);
        vec3<Scalar> M1inv_unitr = (Scalar(1.0) / W0) * (unitr - fac_i * ca  * a3);
        vec3<Scalar> M1inv_b3    = (Scalar(1.0) / W0) * (b3    - fac_i * cab * a3);

        // Second SM step: B12^{-1} = (M1 + dWj*(b3 x b3))^{-1}
        Scalar b3_M1inv_b3    = dot(b3, M1inv_b3);
        Scalar denom_j        = Scalar(1.0) + dWj * b3_M1inv_b3;
        Scalar b3_M1inv_unitr = dot(b3, M1inv_unitr);

        Scalar rhat_B12inv_rhat = dot(unitr, M1inv_unitr)
                                  - dWj * b3_M1inv_unitr * b3_M1inv_unitr / denom_j;

        vec3<Scalar> B12inv_unitr = M1inv_unitr
                                    - (dWj / denom_j) * b3_M1inv_unitr * M1inv_b3;
        vec3<Scalar> iota = r * B12inv_unitr;

        Scalar chi_e = fast::pow(Scalar(2.0) * rhat_B12inv_rhat, mu);

        // ---------------------------------------------------------------
        // eta: shape-anisotropy prefactor (LAMMPS "eta").
        // eta = (2 * lshape_i * lshape_j / det(G12))^upsilon
        // lshape = (lperp^2 + lpar^2) * lperp  for uniaxial particles
        // G12 = 2*lperp^2*I + (lpar^2-lperp^2)*(a3 x a3 + b3 x b3)
        // ---------------------------------------------------------------
        Scalar lshape  = (lperpsq + lparsq) * lperp;
        Scalar eta_num = Scalar(2.0) * lshape * lshape;

        // upsilon enters the shape prefactor as eta = (.)^(upsilon/2)  (Brown 2009 Eq. 6).
        Scalar upsilon_eff = Scalar(0.5) * upsilon;

        Scalar L0 = Scalar(2.0) * lperpsq;
        Scalar dL = lparsq - lperpsq;

        Scalar g00 = L0 + dL * (a3.x * a3.x + b3.x * b3.x);
        Scalar g11 = L0 + dL * (a3.y * a3.y + b3.y * b3.y);
        Scalar g22 = L0 + dL * (a3.z * a3.z + b3.z * b3.z);
        Scalar g01 =      dL * (a3.x * a3.y + b3.x * b3.y);
        Scalar g02 =      dL * (a3.x * a3.z + b3.x * b3.z);
        Scalar g12 =      dL * (a3.y * a3.z + b3.y * b3.z);
        Scalar det_G12 = g00 * (g11 * g22 - g12 * g12)
                       - g01 * (g01 * g22 - g12 * g02)
                       + g02 * (g01 * g12 - g11 * g02);

        Scalar eta = fast::pow(eta_num / det_G12, upsilon_eff);

        Scalar P = eta * chi_e;

        // ---------------------------------------------------------------
        // Radial LJ terms
        // ---------------------------------------------------------------
        Scalar zeta2inv   = Scalar(1.0) / zetasq;
        Scalar zeta6inv   = zeta2inv * zeta2inv * zeta2inv;
        Scalar u_r        = Scalar(4.0) * zeta6inv * (zeta6inv - Scalar(1.0));
        Scalar du_r_dzeta = Scalar(24.0) * zeta6inv / zeta
                            * (Scalar(1.0) - Scalar(2.0) * zeta6inv);

        Scalar dUdr_sigma = epsilon * P * du_r_dzeta / sigma_min;
        Scalar dUdphi     = epsilon * P * du_r_dzeta
                            * Scalar(0.5) * sigma * sigma * sigma / sigma_min;

        // ---------------------------------------------------------------
        // Energy
        // ---------------------------------------------------------------
        pair_eng = epsilon * P * u_r;

        if (energy_shift)
            {
            // shift so U is continuous at r = r_cut for this orientation
            Scalar rcut        = fast::sqrt(rcutsq);
            Scalar zeta_rcut   = (rcut - sigma + sigma_min) / sigma_min;
            Scalar zeta_rcut2i = Scalar(1.0) / (zeta_rcut * zeta_rcut);
            Scalar zeta_rcut6i = zeta_rcut2i * zeta_rcut2i * zeta_rcut2i;
            pair_eng -= epsilon * P
                        * Scalar(4.0) * zeta_rcut6i * (zeta_rcut6i - Scalar(1.0));
            }

        // ---------------------------------------------------------------
        // Force
        // ---------------------------------------------------------------
        Scalar r2inv = Scalar(1.0) / rsq;

        vec3<Scalar> fK = -r2inv * dUdphi * kappa;

        Scalar iota_dot_unitr = dot(iota, unitr);
        Scalar dchi_e_fac     = Scalar(-4.0) * r2inv * mu
                                * fast::pow(chi_e, (mu - Scalar(1.0)) / mu);
        vec3<Scalar> dchi_e_dr = dchi_e_fac * (iota - iota_dot_unitr * unitr);

        vec3<Scalar> f = -dUdr_sigma * unitr
                         + fK
                         + r2inv * dUdphi * dot(kappa, unitr) * unitr
                         + epsilon * eta * u_r * dchi_e_dr;
        force = vec_to_scalar3(f);

        // ---------------------------------------------------------------
        // Torques
        // ---------------------------------------------------------------

        // sigma/kappa path torque
        vec3<Scalar> rca
            = Scalar(0.5)
              * (-dr
                 - r * shape_chi_fact
                       * ((ca - shape_chic * cb) * a3 - (cb - shape_chic * ca) * b3));
        vec3<Scalar> rcb = rca + dr;
        torque_i = vec_to_scalar3(cross(rca, fK));
        torque_j = -vec_to_scalar3(cross(rcb, fK));

        // chi_e torque
        Scalar iota_dot_a3 = dot(iota, a3);
        Scalar iota_dot_b3 = dot(iota, b3);

        vec3<Scalar> Bi_iota = wi_perp * iota + dWi * iota_dot_a3 * a3;
        vec3<Scalar> Bj_iota = wj_perp * iota + dWj * iota_dot_b3 * b3;

        Scalar chi_e_tor_scale = epsilon * eta * u_r * dchi_e_fac;
        torque_i -= vec_to_scalar3(chi_e_tor_scale * cross(Bi_iota, iota));
        torque_j -= vec_to_scalar3(chi_e_tor_scale * cross(Bj_iota, iota));

        // eta torque via G12^{-1} (double Sherman-Morrison)
        Scalar fac_si           = dL / (L0 + dL);
        vec3<Scalar> M1s_inv_a3 = (Scalar(1.0) / (L0 + dL)) * a3;
        vec3<Scalar> M1s_inv_b3 = (Scalar(1.0) / L0) * (b3 - fac_si * cab * a3);

        Scalar b3_M1s_inv_b3   = dot(b3, M1s_inv_b3);
        Scalar denom_js        = Scalar(1.0) + dL * b3_M1s_inv_b3;
        Scalar b3_M1s_inv_a3   = dot(b3, M1s_inv_a3);

        vec3<Scalar> G12inv_a3 = M1s_inv_a3 - (dL / denom_js) * b3_M1s_inv_a3 * M1s_inv_b3;
        vec3<Scalar> G12inv_b3 = M1s_inv_b3 - (dL / denom_js) * b3_M1s_inv_b3 * M1s_inv_b3;

        Scalar eta_fac = upsilon_eff * eta * Scalar(2.0) * dL * epsilon * u_r * chi_e;
        torque_i += vec_to_scalar3(cross(a3, eta_fac * G12inv_a3));
        torque_j += vec_to_scalar3(cross(b3, eta_fac * G12inv_b3));

        return true;
        }

    DEVICE Scalar evalPressureLRCIntegral()
        {
        return 0;
        }

    DEVICE Scalar evalEnergyLRCIntegral()
        {
        return 0;
        }

#ifndef __HIPCC__
    static std::string getName()
        {
        return "gbfull";
        }
    static std::string getShapeParamName()
        {
        return "shape";
        }
    std::string getShapeSpec() const
        {
        std::ostringstream shapedef;
        shapedef << "{\"type\": \"Ellipsoid\", \"a\": " << lperp << ", \"b\": " << lperp
                 << ", \"c\": " << lpar << "}";
        return shapedef.str();
        }
#endif

    protected:
    vec3<Scalar> dr;
    Scalar rcutsq;
    quat<Scalar> qi;
    quat<Scalar> qj;
    Scalar epsilon;
    Scalar lperp;
    Scalar lpar;
    Scalar e_i_perp;
    Scalar e_i_par;
    Scalar e_j_perp;
    Scalar e_j_par;
    Scalar mu;
    Scalar upsilon;
    };

    } // end namespace md
    } // end namespace hoomd

#undef HOOMD_GB_MIN
#undef HOOMD_GB_MAX
#endif // __EVALUATOR_PAIR_GB_FULL_H__
