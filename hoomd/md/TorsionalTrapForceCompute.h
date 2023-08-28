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

#ifndef __TORSIONALTRAPFORCECOMPUTE_H__
#define __TORSIONALTRAPFORCECOMPUTE_H__

namespace hoomd
    {
namespace md
    {
struct torsionaltrap_sin_params
    {
    Scalar k;


#ifndef __HIPCC__
    torsionaltrap_sin_params() : k(0.) { }

    torsionaltrap_sin_params(pybind11::dict v)
        : k(v["k"].cast<Scalar>())
        {
        }

    pybind11::dict asDict()
        {
        pybind11::dict v;
        v["k"] = k;
        return v;
        }
#endif
    } __attribute__((aligned(32)));

//! Computes harmonic dihedral forces on each particle
/*! Harmonic dihedral forces are computed on every particle in the simulation.

    The dihedrals which forces are computed on are accessed from ParticleData::getDihedralData
    \ingroup computes
*/
class PYBIND11_EXPORT TorsionalTrapForceCompute : public ForceCompute
    {
    public:
    //! Constructs the compute
    TorsionalTrapForceCompute(std::shared_ptr<SystemDefinition> sysdef,std::shared_ptr<ParticleGroup> group1,std::shared_ptr<ParticleGroup> group2,std::shared_ptr<ParticleGroup> group3,std::shared_ptr<ParticleGroup> group4,unsigned int num_angles);

    //! Destructor
    virtual ~TorsionalTrapForceCompute();

    //! Set the parameters
    virtual void
    setParams(unsigned int type, Scalar K);

    virtual void setParamsPython(std::string type, pybind11::dict params);
    //virtual void TorsionalTrapForceCompute::setangles();

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

    std::shared_ptr<ParticleGroup>& getGroup3()
        {
        return m_group3;
        }

    std::shared_ptr<ParticleGroup>& getGroup4()
        {
        return m_group4;
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

    unsigned int getnumangles()
        {
          return m_num_angles;
        }

    // Get the angles after rotation for a particular type
    pybind11::array_t<Scalar> getangles(std::string type);

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
    GPUArray<Scalar> m_angles;
    GPUArray<Scalar> m_ref_angles;

    GPUArray<Scalar2> m_oldnew_angles; //!< x component old and y component new angles
    Index2D m_oldnew_value;            //!< Index table helper

    GPUArray<Scalar3> m_ref_vecp;
    GPUArray<Scalar3> m_ref_vecn;

    Index3D m_ref_vecp_valu;
    Index3D m_ref_vecn_value;

    unsigned int m_num_angles;

    std::shared_ptr<DihedralData> m_dihedral_data; //!< Dihedral data to use in computing dihedrals
    std::shared_ptr<ParticleGroup> m_group1; //!< Group of particles on which this force is applied
    std::shared_ptr<ParticleGroup> m_group2; //!< Group of particles on which this force is applied
    std::shared_ptr<ParticleGroup> m_group3; //!< Group of particles on which this force is applied
    std::shared_ptr<ParticleGroup> m_group4; //!< Group of particles on which this force is applied
    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

    } // end namespace md
    } // end namespace hoomd

#endif
