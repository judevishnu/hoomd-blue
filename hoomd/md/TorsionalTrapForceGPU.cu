// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "TorsionalTrapForceGPU.cuh"
#include "hoomd/TextureTools.h"
#include <iostream>
#include <math.h>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <assert.h>

#ifdef SINGLE_PRECISION
#define __scalar2int_rn __float2int_rn
#else
#define __scalar2int_rn __double2int_rn
#endif

/*! \file HarmonicDihedralForceGPU.cu
    \brief Defines GPU kernel code for calculating the harmonic dihedral forces. Used by
   HarmonicDihedralForceComputeGPU.
*/

namespace hoomd
    {
namespace md
    {
namespace kernel
    {

//! GPU implementation of anglDiff
__device__ Scalar gpu_anglDiff(Scalar diff)
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
//! Kernel for calculating harmonic dihedral forces on the GPU
/*! \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch pitch of 2D virial array
    \param N number of particles
    \param d_pos particle positions on the device
    \param d_params Parameters for the angle force
    \param box Box dimensions for periodic boundary condition handling
    \param tlist Dihedral data to use in calculating the forces
    \param dihedral_ABCD List of relative atom positions in the dihedrals
    \param pitch Pitch of 2D dihedral list
    \param n_dihedrals_list List of numbers of dihedrals per atom
*/
__global__ void gpu_compute_torsionaltrap_sin_force_kernel(const unsigned int group_size,const BoxDim box,
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
                                                       const Index2D d_oldnew_value,
                                                       const Index2D d_ref_vecp_value,
                                                       const Index2D d_ref_vecn_value,
                                                       const typeval_union* d_group_typeval,
                                                       const Scalar* d_params,
                                                       long unsigned int timestep)
    {
    unsigned int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_idx >= group_size)
        return;
    unsigned int typval = d_group_typeval[group_idx].type;
    //printf("%u \n", typva);
    Scalar params = __ldg(d_params + typval);

    Scalar K = params;


    unsigned int tagp = d_tag_array1[group_idx];
    unsigned int tagn = d_tag_array2[group_idx];
    unsigned int tagpside = d_tag_array3[group_idx];
    unsigned int tagnside = d_tag_array4[group_idx];
    unsigned int rtagp = d_rtag[tagp];
    unsigned int rtagn = d_rtag[tagn];
    unsigned int rtagpside = d_rtag[tagpside];
    unsigned int rtagnside = d_rtag[tagnside];




    Scalar4 pos_b1 = d_pos[rtagp];
    Scalar4 pos_c1 = d_pos[rtagn];
    Scalar4 pos_a1 = d_pos[rtagpside];
    Scalar4 pos_d1 = d_pos[rtagnside];



    Scalar3 a_poss1 = make_scalar3(pos_a1.x,pos_a1.y,pos_a1.z);
    Scalar3 b_poss1 = make_scalar3(pos_b1.x,pos_b1.y,pos_b1.z);
    Scalar3 c_poss1 = make_scalar3(pos_c1.x,pos_c1.y,pos_c1.z);
    Scalar3 d_poss1 = make_scalar3(pos_d1.x,pos_d1.y,pos_d1.z);



    Scalar3 dab1;
    dab1 = a_poss1 - b_poss1;

    Scalar3 ddc1;
    ddc1 = d_poss1 - c_poss1;

    dab1 = box.minImage(dab1);

    ddc1 = box.minImage(ddc1);

    Scalar dab1mag = fast::sqrt(dot(dab1,dab1));
    Scalar ddc1mag = fast::sqrt(dot(ddc1,ddc1));
    Scalar3 unitddc = make_scalar3(ddc1.x/ddc1mag,ddc1.y/ddc1mag,ddc1.z/ddc1mag);
    Scalar3 unitdab = make_scalar3(dab1.x/dab1mag,dab1.y/dab1mag,dab1.z/dab1mag);

    Scalar3 refvecn =__ldg(d_ref_vecn+d_ref_vecn_value(group_idx, typval));
    Scalar3 refvecp =__ldg(d_ref_vecp+d_ref_vecp_value(group_idx, typval));
    Scalar dotp = dot(refvecp,unitdab) ;
    Scalar dotn = dot(refvecn,unitddc) ;

    Scalar3 crossp;
    Scalar3 crossn;
    Scalar x = unitdab.y*refvecp.z - unitdab.z*refvecp.y;
    Scalar y = unitdab.z*refvecp.x - unitdab.x*refvecp.z;
    Scalar z = unitdab.x*refvecp.y - unitdab.y*refvecp.x;
    crossp = make_scalar3(x,y,z);

    Scalar x1 = unitddc.y*refvecn.z - unitddc.z*refvecn.y;
    Scalar y1 = unitddc.z*refvecn.x - unitddc.x*refvecn.z;
    Scalar z1 = unitddc.x*refvecn.y - unitddc.y*refvecn.x;
    crossn = make_scalar3(x1,y1,z1);

    Scalar angl;
    Scalar diffangl;
    Scalar ref_angl;
    Scalar tmpangl;
    Scalar oldangl;
    Scalar3 torqp;
    Scalar3 torqn;
    Scalar3 constT;
    torqp = make_scalar3(0.0,0.0,0.0);
    torqn = make_scalar3(0.0,0.0,0.0);
    oldangl = 0;
    diffangl = 0;
    ref_angl = 0;
    angl = 0;




    tmpangl = atan2(dab1.y, dab1.x) - atan2(ddc1.y, ddc1.x);
    tmpangl = gpu_anglDiff(tmpangl);
    Scalar2 TMPoldnew_angles = __ldg(d_oldnew_angles+d_oldnew_value(group_idx, typval));
    oldangl = TMPoldnew_angles.x;
    diffangl = tmpangl - oldangl;
    diffangl = gpu_anglDiff(diffangl);
    TMPoldnew_angles.y = tmpangl;

    d_TMP_angles = d_angles[group_idx];
    angl = d_TMP_angles+diffangl;
    ref_angl = d_ref_angles[group_idx];
    //printf("%d %f \n",i,angl);
    d_angles[group_idx] = angl;

    TMPoldnew_angles.x = tmpangl;
    d_oldnew_angles[d_oldnew_value(group_idx, typval)] = TMPoldnew_angles;
    Scalar tmagp = 2*K*dotp;
    Scalar tmagn = 2*K*dotn;

    torqp = make_scalar3(tmagp*crossp.x,tmagp*crossp.y,tmagp*crossp.z);
    torqn = make_scalar3(tmagn*crossn.x,tmagn*crossn.y,tmagn*crossn.z);
    d_torque[rtagp] = make_scalar4(torqp.x,torqp.y,torqp.z,0);
    d_torque[rtagn] = make_scalar4(torqn.x,torqn.y,torqn.z,0);



    }

/*! \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch pitch of 2D virial array
    \param N number of particles
    \param d_pos particle positions on the GPU
    \param box Box dimensions (in GPU format) to use for periodic boundary conditions
    \param tlist Dihedral data to use in calculating the forces
    \param dihedral_ABCD List of relative atom positions in the dihedrals
    \param pitch Pitch of 2D dihedral list
    \param n_dihedrals_list List of numbers of dihedrals per atom
    \param d_params K, sign,multiplicity params packed as padded Scalar4 variables
    \param n_dihedral_types Number of dihedral types in d_params
    \param block_size Block size to use when performing calculations
    \param compute_capability Compute capability of the device (200, 300, 350, ...)

    \returns Any error code resulting from the kernel launch
    \note Always returns hipSuccess in release builds to avoid the hipDeviceSynchronize()

    \a d_params should include one Scalar4 element per dihedral type. The x component contains K the
   spring constant and the y component contains sign, and the z component the multiplicity.
*/
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
                                                unsigned int block_size)
                                                //,int warp_size)
    {
    assert(d_params);
    // setup the grid to run the kernel
    dim3 grid(group_size / block_size + 1, 1, 1);
    dim3 threads(block_size, 1, 1);
    // unsigned int max_block_size;
    // hipFuncAttributes attr;
    // hipFuncGetAttributes(&attr, (const void*)gpu_compute_harmonic_dihedral_forces_kernel);
    // max_block_size = attr.maxThreadsPerBlock;
    // if (max_block_size % warp_size)
    //     // handle non-sensical return values from hipFuncGetAttributes
    //     max_block_size = (max_block_size / warp_size - 1) * warp_size;
    //
    // unsigned int run_block_size = min(block_size, max_block_size);
    //
    // // setup the grid to run the kernel
    // dim3 grid(N / run_block_size + 1, 1, 1);
    // dim3 threads(run_block_size, 1, 1);

    // run the kernel
    hipLaunchKernelGGL((gpu_compute_torsionaltrap_sin_force_kernel),
                       dim3(grid),dim3(threads),0,0,group_size,box,
                       d_pos,
                       d_torque,
                       d_rtag,
                       d_tag_array1,
                       d_tag_array2,
                       d_tag_array3,
                       d_tag_array4,
                       d_ref_angles,
                       d_angles,
                       d_oldnew_angles,
                       d_ref_vecp,
                       d_ref_vecn,
                       d_oldnew_value,
                       d_ref_vecp_value,
                       d_ref_vecn_value,
                       d_group_typeval,
                       d_params,
                       timestep);


    return hipSuccess;
    }

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
