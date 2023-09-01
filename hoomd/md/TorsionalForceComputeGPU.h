// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "TorsionalForceCompute.h"
#include "TorsionalForceGPU.cuh"
#include "hoomd/Autotuner.h"

#include <hoomd/extern/nano-signal-slot/nano_signal_slot.hpp>
#include <memory>

/*! \file HarmonicDihedralForceComputeGPU.h
    \brief Declares the HarmonicDihedralForceGPU class
*/

#ifndef __HARMONICDIHEDRALFORCECOMPUTEGPU_H__
#define __HARMONICDIHEDRALFORCECOMPUTEGPU_H__

namespace hoomd
    {
namespace md
    {
//! Implements the harmonic dihedral force calculation on the GPU
/*! HarmonicDihedralForceComputeGPU implements the same calculations as
   HarmonicDihedralForceCompute, but executing on the GPU.

    Per-type parameters are stored in a simple global memory area pointed to by
    \a m_gpu_params. They are stored as Scalar2's with the \a x component being K and the
    \a y component being t_0.

    The GPU kernel can be found in dihedralforce_kernel.cu.

    \ingroup computes
*/
class PYBIND11_EXPORT TorsionalForceComputeGPU : public TorsionalForceCompute
    {
    public:
    //! Constructs the compute
    TorsionalForceComputeGPU(std::shared_ptr<SystemDefinition> system,std::shared_ptr<ParticleGroup> group1,std::shared_ptr<ParticleGroup> group2,std::shared_ptr<ParticleGroup> group3,std::shared_ptr<ParticleGroup> group4,unsigned int num_angles);
    //! Destructor
    ~TorsionalForceComputeGPU();

    //! Set autotuner parameters
    /*! \param enable Enable/disable autotuning
        \param period period (approximate) in time steps when returning occurs
    */
    virtual void setAutotunerParams(bool enable, unsigned int period)
        {
        TorsionalForceCompute::setAutotunerParams(enable, period);
        m_tuner->setPeriod(period);
        m_tuner->setEnabled(enable);
        }

    //! Set the parameters
    virtual void
    setParams(unsigned int type, Scalar K, Scalar sign, int multiplicity, Scalar phi_0);

    protected:
    std::unique_ptr<Autotuner> m_tuner; //!< Autotuner for block size
    GPUArray<Scalar4> m_params;         //!< Parameters stored on the GPU (k,t_qx,t_qy,t_qz)

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

    } // end namespace md
    } // end namespace hoomd

#endif
