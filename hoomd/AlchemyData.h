// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: jproc

/*! \file AlchemyData.h
    \brief Contains declarations for AlchemyData.
 */

#ifndef __ALCHEMYDATA_H__
#define __ALCHEMYDATA_H__

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <memory>
#include <string>

#include "HOOMDMPI.h"
#include "hoomd/ForceCompute.h"
#include "hoomd/HOOMDMath.h"

class AlchemicalParticle
    {
    public:
    Scalar getValue()
        {
        return m_value;
        };

    protected:
    std::shared_ptr<const ExecutionConfiguration> m_exec_conf; //!< Stored shared ptr to the execution configuration
    Scalar m_value; //!< Alpha space dimensionless position of the particle
    // TODO: decide if velocity or momentum would typically be better for numerical stability
    std::shared_ptr<Compute> m_base; //!< the associated Alchemical Compute
    };
class AlchemicalMDParticle : public AlchemicalParticle
    {
    public:
    void zeroForces()
        {
    ArrayHandle<Scalar> h_forces(m_alchemical_forces,
                                     access_location::host,
                                     access_mode::overwrite);
        memset((void*)h_forces.data, 0, sizeof(Scalar) * m_alchemical_forces.getNumElements());
        }
    
    void resizeForces(unsigned int N)
        {
        GlobalArray<Scalar> new_forces(N,m_exec_conf);
        m_alchemical_forces.swap(new_forces);
        }

    protected:
    // TODO: decide if velocity or momentum would typically be better for numerical stability
    Scalar3 m_kinetic_values; //!< x=mass, y=velocity/momentum, z=netForce
    // std::shared_ptr<ForceCompute> m_force; //!< the associated Alchemical Force Compute
    GlobalArray<Scalar> m_alchemical_forces; //!< Per particle alchemical forces
    };

class AlchemicalPairParticle : public AlchemicalMDParticle
    {
    int2 m_type_pair; // TODO: make this more general to non-pair interactions
    };

#endif
