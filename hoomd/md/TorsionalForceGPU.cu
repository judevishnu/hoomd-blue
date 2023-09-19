// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "TorsionalForceGPU.cuh"
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
__global__ void gpu_compute_torsional_sin_force_kernel(const unsigned int group_size,const BoxDim box,
                                                       const Scalar4* d_pos,
                                                       Scalar4* d_torque,
                                                       unsigned int* rtag,
                                                       unsigned int* d_index_array1,
                                                       unsigned int* d_index_array2,
                                                       unsigned int* d_index_array3,
                                                       unsigned int* d_index_array4,
                                                       Scalar* d_ref_angles,
                                                       Scalar* d_angles,
                                                       Scalar2* d_oldnew_angles,
                                                       const Index2D d_oldnew_value,
                                                       const typeval_union* d_group_typeval,
                                                       const Scalar4* d_params,
                                                       long unsigned int timestep)
    {
    unsigned int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_idx >= group_size)
        return;
    unsigned int typval = d_group_typeval[group_idx].type;
    //printf("%u \n", typva);
    Scalar4 params = __ldg(d_params + typval);
    //printf("%u %u %f %f %f %f \n", group_idx,typval,params.x,params.y,params.z,params.w);

    Scalar K = params.x;
    Scalar tqx = params.y;
    Scalar tqy = params.z;
    Scalar tqz = params.w;

    unsigned int tagp = d_index_array1[group_idx];
    unsigned int tagn = d_index_array2[group_idx];
    unsigned int tagpside = d_index_array3[group_idx];
    unsigned int tagnside = d_index_array4[group_idx];
    unsigned int rtagp = d_r_tag[tagp];
    unsigned int rtagn = r_tag[tagn];
    unsigned int rtagpside = d_r_tag[tagpside];
    unsigned int rtagnside = d_r_tag[tagnside];


    Scalar4 pos_b = __ldg(d_pos + tagp);
    Scalar4 pos_c = __ldg(d_pos + tagn);
    Scalar4 pos_a = __ldg(d_pos + tagpside);
    Scalar4 pos_d = __ldg(d_pos + tagnside);

    // Scalar4 pos_b = d_pos[tagp];
    // Scalar4 pos_c = d_pos[tagn];
    // Scalar4 pos_a = d_pos[tagpside];
    // Scalar4 pos_d = d_pos[tagnside];

    Scalar3 a_poss = make_scalar3(pos_a.x,pos_a.y,pos_a.z);
    Scalar3 b_poss = make_scalar3(pos_b.x,pos_b.y,pos_b.z);
    Scalar3 c_poss = make_scalar3(pos_c.x,pos_c.y,pos_c.z);
    Scalar3 d_poss = make_scalar3(pos_d.x,pos_d.y,pos_d.z);


    Scalar3 dab;
    dab = a_poss - b_poss;


    // dab.x = pos_a.x - pos_b.x;
    // dab.y = pos_a.y - pos_b.y;
    // dab.z = pos_a.z - pos_b.z;

    Scalar3 ddc;
    // ddc.x = pos_d.x - pos_c.x;
    // ddc.y = pos_d.y - pos_c.y;
    // ddc.z = pos_d.z - pos_c.z;

    ddc = d_poss - c_poss;


    dab = box.minImage(dab);

    ddc = box.minImage(ddc);

    //####################################################################################################
    Scalar angl;
    Scalar diffangl;
    Scalar ref_angl;
    Scalar tmpangl;
    Scalar oldangl;
    Scalar3 torqp;
    Scalar3 torqn;
    Scalar3 constT;
    torqp = make_scalar3(0,0,0);
    torqn = make_scalar3(0,0,0);
    tmpangl = 0;
    angl = 0;
    diffangl=0;
    ref_angl=0;

    tmpangl = atan2(dab.y, dab.x) - atan2(ddc.y, ddc.x);
    //tmpangl = gpu_anglDiff(tmpangl);
    if (tmpangl > M_PI)
        {
        tmpangl -= 2 * M_PI;
        }
    else if (tmpangl <= -M_PI)
        {
        tmpangl += 2 * M_PI;
        }
    oldangl = d_oldnew_angles[d_oldnew_value(group_idx, typval)].x;
    diffangl = tmpangl - oldangl;
    //diffangl = gpu_anglDiff(diffangl);
    if (diffangl > M_PI)
        {
        diffangl -= 2 * M_PI;
        }
    else if (diffangl <= -M_PI)
        {
        diffangl += 2 * M_PI;
        }
    d_oldnew_angles[d_oldnew_value(group_idx, typval)].y = tmpangl;
    angl = d_angles[group_idx]+diffangl;
    //printf("%d %f \n",i,angl);
    d_angles[group_idx] = angl;
    Scalar cs = slow::cos(angl);
    Scalar ss = slow::sin(angl);
    d_oldnew_angles[d_oldnew_value(group_idx, typval)].x = tmpangl;
    // Scalar distone = sqrt(dab.x*dab.x+dab.y*dab.y+dab.z*dab.z);
    // Scalar disttwo = sqrt(ddc.x*ddc.x+ddc.y*ddc.y+ddc.z*ddc.z);
    Scalar distone =  slow::rsqrt(dot(dab, dab));
    Scalar disttwo =  slow::rsqrt(dot(ddc, ddc));
    if(group_idx==0)
      {
      //printf("%u %u %u %u %u %u %f %f %f %f %f %f\n",timestep,group_idx,tagp,tagn,tagpside,tagnside,diffangl,tmpangl,oldangl,angl,distone,disttwo);
      printf("GPU %lu %u %u %u %u %u \n",timestep,group_idx,tagp,tagn,tagpside,tagnside);
      printf("GPU %lu %u %u %u %u %u \n",timestep,group_idx,rtagp,rtagn,rtagpside,rtagnside);


      }
    if ((angl> M_PI)&&(angl<3*M_PI/2))
      {
      ss = slow::sin(angl- M_PI);
      cs = slow::cos(angl- M_PI);
      torqp = make_scalar3(0,0,-2*K*cs*ss);
      torqn = make_scalar3(0,0,2*K*cs*ss);
      }
    else if (angl < 0)
      {
      Scalar cs = slow::cos(angl);
      Scalar ss = slow::sin(angl);
      torqp = make_scalar3(0,0,-2*K*cs*ss);
      torqn = make_scalar3(0,0,2*K*cs*ss);
      }
    // else if (angl == 0)
    //   {
    //   if (timestep < 10)
    //     {
    //     torqp.x =  tqx;
    //     torqp.y =  tqy;
    //     torqp.z =  tqz;
    //     torqn.x =  tqx;
    //     torqn.y =  tqy;
    //     torqn.z = -tqz;
    //     }
    //   }
    d_torque[tagp] = make_scalar4(torqp.x,torqp.y,torqp.z,0);
    d_torque[tagn] = make_scalar4(torqn.x,torqn.y,torqn.z,0);
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
hipError_t gpu_compute_torsional_sin_forces(const unsigned int group_size,const BoxDim& box,
                                                const Scalar4* d_pos,
                                                Scalar4* d_torque,
                                                unsigned int* rtag,
                                                unsigned int* d_index_array1,
                                                unsigned int* d_index_array2,
                                                unsigned int* d_index_array3,
                                                unsigned int* d_index_array4,
                                                Scalar* d_ref_angles,
                                                Scalar* d_angles,
                                                Scalar2* d_oldnew_angles,
                                                const Index2D& d_oldnew_value,
                                                const typeval_union* d_group_typeval,
                                                // const size_t virial_pitch,
                                                // const unsigned int N,
                                                // const Scalar4* d_pos,
                                                // const BoxDim& box,
                                                // const group_storage<4>* tlist,
                                                // const unsigned int* dihedral_ABCD,
                                                // const unsigned int pitch,
                                                // const unsigned int* n_dihedrals_list,
                                                Scalar4* d_params,
                                                unsigned int timestep,
                                                // unsigned int n_dihedral_types,
                                                long unsigned int block_size)
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
    hipLaunchKernelGGL((gpu_compute_torsional_sin_force_kernel),
                       dim3(grid),dim3(threads),0,0,group_size,box,
                       d_pos,
                       d_torque,
                       rtag,
                       d_index_array1,
                       d_index_array2,
                       d_index_array3,
                       d_index_array4,
                       d_ref_angles,
                       d_angles,
                       d_oldnew_angles,
                       d_oldnew_value,
                       d_group_typeval,
                       d_params,
                       timestep);
                       // 0,
                       // 0,
                       // d_force,
                       // d_virial,
                       // virial_pitch,
                       // N,
                       // d_pos,
                       // d_params,
                       // box,
                       // tlist,
                       // dihedral_ABCD,
                       // pitch,
                       // n_dihedrals_list);

    return hipSuccess;
    }

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
