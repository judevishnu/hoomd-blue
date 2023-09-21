// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hip/hip_runtime.h"
#include "hoomd/BondedGroupData.cuh"
#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"
#include "hoomd/Index1D.h"

/*! \file HarmonicDihedralForceGPU.cuh
    \brief Declares GPU kernel code for calculating the harmonic dihedral forces. Used by
   HarmonicDihedralForceComputeGPU.
*/

#ifndef __TORSIONALTRAPFORCEGPU_CUH__
#define __TORSIONALTRAPFORCEGPU_CUH__
namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Kernel driver that computes harmonic dihedral forces for HarmonicDihedralForceComputeGPU
hipError_t gpu_compute_torsionaltrap_sin_forces(const unsigned int group_size,const BoxDim& box,
                                                const Scalar4* d_pos,
                                                Scalar4* d_torque,
                                                unsigned int* d_rtag,
                                                unsigned int* d_tag_array1,
                                                unsigned int* d_tag_array2,
                                                unsigned int* d_tag_array3,
                                                unsigned int* d_tag_array4,
                                                Scalar* d_ref_angles,
                                                Scalar* d_angles,
                                                Scalar2* d_oldnew_angles,
                                                Scalar3* d_ref_vecp,
                                                Scalar3* d_ref_vecn,
                                                const Index2D& d_oldnew_value,
                                                const Index2D& d_ref_vecp_value,
                                                const Index2D& d_ref_vecn_value,
                                                const typeval_union* d_group_typeval,
                                                Scalar* d_params,
                                                long unsigned int timestep,
                                                unsigned int block_size);

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif
