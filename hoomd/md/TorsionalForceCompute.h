// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/BondedGroupData.h"
#include "hoomd/ForceCompute.h"
#include "hoomd/ParticleGroup.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include <memory>

#include <vector>

/*! \file HarmonicDihedralForceCompute.h
    \brief Declares a class for computing harmonic dihedrals
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __TORSIONALFORCECOMPUTE_H__
#define __TORSIONALFORCECOMPUTE_H__

namespace hoomd
    {
namespace md
    {
struct torsional_sin_params
    {
    Scalar k;
    Scalar d;
    int n;
    Scalar phi_0;
    Scalar t_qx;
    Scalar t_qy;
    Scalar t_qz;

#ifndef __HIPCC__
    torsional_sin_params() : k(0.), d(0.), n(0), phi_0(0.), t_qx(0.), t_qy(0.), t_qz(0.) { }

    torsional_sin_params(pybind11::dict v)
        : k(v["k"].cast<Scalar>()), d(v["d"].cast<Scalar>()), n(v["n"].cast<int>()),
          phi_0(v["phi0"].cast<Scalar>()), t_qx(v["tqx"].cast<Scalar>()),t_qy(v["tqy"].cast<Scalar>()),t_qz(v["tqz"].cast<Scalar>())
        {
        }

    pybind11::dict asDict()
        {
        pybind11::dict v;
        v["k"] = k;
        v["d"] = d;
        v["n"] = n;
        v["phi0"] = phi_0;
        v["tqx"] = t_qx;
        v["tqy"] = t_qy;
        v["tqz"] = t_qz;
        return v;
        }
#endif
    } __attribute__((aligned(32)));

//! Computes harmonic dihedral forces on each particle
/*! Harmonic dihedral forces are computed on every particle in the simulation.

    The dihedrals which forces are computed on are accessed from ParticleData::getDihedralData
    \ingroup computes
*/
class PYBIND11_EXPORT TorsionalForceCompute : public ForceCompute
    {
    public:
    //! Constructs the compute
    TorsionalForceCompute(std::shared_ptr<SystemDefinition> sysdef,std::shared_ptr<ParticleGroup> group1,std::shared_ptr<ParticleGroup> group2);

    //! Destructor
    virtual ~TorsionalForceCompute();

    //! Set the parameters
    virtual void
    setParams(unsigned int type, Scalar K, Scalar sign, int multiplicity, Scalar phi_0, Scalar t_qx, Scalar t_qy, Scalar t_qz);

    virtual void setParamsPython(std::string type, pybind11::dict params);

    /// Get the parameters for a particular type
    pybind11::dict getParams(std::string type);

    std::shared_ptr<ParticleGroup>& getGroup1()
        {
        return m_group1;
        }
    std::shared_ptr<ParticleGroup>& getGroup2()
        {
        return m_group2;
        }

    //Retund angle in the range 0 to Pi
    Scalar anglDiff(Scalar diff)
        {
          if (diff > M_PI)
              {
              diff -= 2 * M_PI;
              }
          else if (diff <= -M_PI)
              {
              diff += 2 * M_PI;
              }
          return diff;
        }

#ifdef ENABLE_MPI
    //! Get ghost particle fields requested by this pair potential
    /*! \param timestep Current time step
     */
    virtual CommFlags getRequestedCommFlags(uint64_t timestep)
        {
        CommFlags flags = CommFlags(0);
        flags[comm_flag::tag] = 1;
        flags |= ForceCompute::getRequestedCommFlags(timestep);
        return flags;
        }
#endif

    protected:

    Scalar* m_K;     //!< K parameter for multiple dihedral tyes
    Scalar* m_sign;  //!< sign parameter for multiple dihedral types
    int* m_multi;    //!< multiplicity parameter for multiple dihedral types
    Scalar* m_phi_0; //!< phi_0 parameter for multiple dihedral types
    Scalar* m_t_qx; //!< phi_0 parameter for multiple dihedral types
    Scalar* m_t_qy; //!< phi_0 parameter for multiple dihedral types
    Scalar* m_t_qz; //!< phi_0 parameter for multiple dihedral types
    GPUArray<Scalar> m_angles;
    GPUArray<Scalar2> m_oldnew_angles; //!< x component old and y component new angles
    Index2D m_oldnew_value;            //!< Index table helper

    unsigned int m_num_angles;

    std::shared_ptr<DihedralData> m_dihedral_data; //!< Dihedral data to use in computing dihedrals
    std::shared_ptr<ParticleGroup> m_group1; //!< Group of particles on which this force is applied
    std::shared_ptr<ParticleGroup> m_group2; //!< Group of particles on which this force is applied
    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

    } // end namespace md
    } // end namespace hoomd

#endif
