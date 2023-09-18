// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file HarmonicDihedralForceComputeGPU.cc
    \brief Defines HarmonicDihedralForceComputeGPU
*/

#include "TorsionalForceComputeGPU.h"
#include <vector>

using namespace std;

namespace hoomd
    {
namespace md
    {
/*! \param sysdef System to compute bond forces on
 */
TorsionalForceComputeGPU::TorsionalForceComputeGPU(
    std::shared_ptr<SystemDefinition> sysdef,std::shared_ptr<ParticleGroup> group1,std::shared_ptr<ParticleGroup> group2,std::shared_ptr<ParticleGroup> group3,std::shared_ptr<ParticleGroup> group4,unsigned int num_angles)
    : TorsionalForceCompute(sysdef,group1,group2,group3,group4,num_angles)
    {
    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error()
            << "Creating a TorsionalForceComputeGPU with no GPU in the execution configuration"
            << endl;
        throw std::runtime_error("Error initializing TorsionalForceComputeGPU");
        }

    // allocate and zero device memory
    GPUArray<Scalar4> params(m_dihedral_data->getNTypes(), m_exec_conf);
    m_params.swap(params);

    unsigned int warp_size = m_exec_conf->dev_prop.warpSize;
    m_tuner.reset(new Autotuner(warp_size,
                                1024,
                                warp_size,
                                5,
                                100000,
                                "torsional_sin",
                                this->m_exec_conf));
    }

TorsionalForceComputeGPU::~TorsionalForceComputeGPU() { }

/*! \param type Type of the dihedral to set parameters for
    \param K Stiffness parameter for the force computation
    \param sign the sign of the cosine term
        \param multiplicity the multiplicity of the cosine term
    \param phi_0 the phase offset

    Sets parameters for the potential of a particular dihedral type and updates the
    parameters on the GPU.
*/
void TorsionalForceComputeGPU::setParams(unsigned int type,Scalar K, Scalar t_qx, Scalar t_qy, Scalar t_qz)
    {
    TorsionalForceCompute::setParams(type, K, t_qx, t_qy, t_qz);

    ArrayHandle<Scalar4> h_params(m_params, access_location::host, access_mode::readwrite);
    // update the local copy of the memory
    h_params.data[type]
        = make_scalar4(Scalar(K), Scalar(t_qx), Scalar(t_qy), Scalar(t_qz));

    // ArrayHandle<Scalar> h_angles(m_angles, access_location::host, access_mode::readwrite);
    // ArrayHandle<Scalar> h_ref_angles(m_ref_angles, access_location::host, access_mode::readwrite);
    //
    // ArrayHandle<Scalar2> h_oldnew_angles(m_oldnew_angles, access_location::host, access_mode::readwrite);
    //
    // ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    // ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
    //
    //
    // // get a local copy of the simulation box too
    // const BoxDim& box = m_pdata->getBox();

    // for (unsigned int i = 0; i < m_num_angles; i++)
    //     {
    //
    //     // const ImproperData::members_t& dihedral = m_dihedral_data->getMembersByIndex(i);
    //     // assert(dihedral.tag[0] <= m_pdata->getMaximumTag());
    //     // assert(dihedral.tag[1] <= m_pdata->getMaximumTag());
    //     // assert(dihedral.tag[2] <= m_pdata->getMaximumTag());
    //     // assert(dihedral.tag[3] <= m_pdata->getMaximumTag());
    //
    //     // transform a, b, and c into indices into the particle data arrays
    //     // MEM TRANSFER: 6 ints
    //     // unsigned int idx_a = h_rtag.data[dihedral.tag[0]];
    //     // unsigned int idx_b = h_rtag.data[dihedral.tag[1]];
    //     // unsigned int idx_c = h_rtag.data[dihedral.tag[2]];
    //     // unsigned int idx_d = h_rtag.data[dihedral.tag[3]];
    //     // unsigned int idp = m_group1->getMemberIndex(i);
    //     // unsigned int idn = m_group2->getMemberIndex(i);
    //     unsigned int tagp = m_group1->getMemberTag(i);
    //     unsigned int tagn = m_group2->getMemberTag(i);
    //     unsigned int tagpside = m_group3->getMemberTag(i);
    //     unsigned int tagnside = m_group4->getMemberTag(i);
    //     unsigned int rtagp = h_rtag.data[tagp];
    //     unsigned int rtagn = h_rtag.data[tagn];
    //     unsigned int rtagpside = h_rtag.data[tagpside];
    //     unsigned int rtagnside = h_rtag.data[tagnside];
    //     unsigned int dihedral_type = m_dihedral_data->getTypeByIndex(i);
    //
    //     // assert(idx_a < m_pdata->getN() + m_pdata->getNGhosts());
    //     // assert(idx_b < m_pdata->getN() + m_pdata->getNGhosts());
    //     // assert(idx_c < m_pdata->getN() + m_pdata->getNGhosts());
    //     // assert(idx_d < m_pdata->getN() + m_pdata->getNGhosts());
    //
    //     // calculate d\vec{r}
    //     Scalar3 dab;
    //     dab.x = h_pos.data[rtagpside].x - h_pos.data[rtagp].x;
    //     dab.y = h_pos.data[rtagpside].y - h_pos.data[rtagp].y;
    //     dab.z = h_pos.data[rtagpside].z - h_pos.data[rtagp].z;
    //
    //
    //     Scalar3 ddc;
    //     ddc.x = h_pos.data[rtagnside].x - h_pos.data[rtagn].x;
    //     ddc.y = h_pos.data[rtagnside].y - h_pos.data[rtagn].y;
    //     ddc.z = h_pos.data[rtagnside].z - h_pos.data[rtagn].z;
    //
    //     dab = box.minImage(dab);
    //     ddc = box.minImage(ddc);
    //     //####################################################################################################
    //     Scalar angl;
    //     angl = atan2(dab.y, dab.x) - atan2(ddc.y, ddc.x);
    //     h_oldnew_angles.data[m_oldnew_value(i, type)].x = angl;
    //     h_oldnew_angles.data[m_oldnew_value(i, type)].y = angl;
    //     h_ref_angles.data[i] = angl;
    //     h_angles.data[i] = 0;
    //
    //     }


    }

/*! Internal method for computing the forces on the GPU.
    \post The force data on the GPU is written with the calculated forces

    \param timestep Current time step of the simulation

    Calls gpu_compute_harmonic_dihedral_forces to do the dirty work.
*/
void TorsionalForceComputeGPU::computeForces(uint64_t timestep)
    {
    ArrayHandle<DihedralData::members_t> d_gpu_dihedral_list(m_dihedral_data->getGPUTable(),
                                                             access_location::device,
                                                             access_mode::read);
    ArrayHandle<unsigned int> d_n_dihedrals(m_dihedral_data->getNGroupsArray(),
                                            access_location::device,
                                            access_mode::read);
    ArrayHandle<unsigned int> d_dihedrals_ABCD(m_dihedral_data->getGPUPosTable(),
                                               access_location::device,
                                               access_mode::read);
    ArrayHandle<typeval_t> d_group_typeval(m_dihedral_data->getTypeValArray(),
                                           access_location::device,
                                           access_mode::read);
    // the dihedral table is up to date: we are good to go. Call the kernel
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    BoxDim box = m_pdata->getGlobalBox();

    //ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar4> d_torque(m_torque, access_location::device, access_mode::overwrite);

    //ArrayHandle<Scalar> d_virial(m_virial, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar4> d_params(m_params, access_location::device, access_mode::read);


    ArrayHandle<Scalar> d_angles(m_angles, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> d_ref_angles(m_ref_angles, access_location::device, access_mode::readwrite);

    ArrayHandle<Scalar2> d_oldnew_angles(m_oldnew_angles, access_location::device, access_mode::readwrite);

    ArrayHandle<unsigned int> d_index_array1(m_group1->getIndexArray(),access_location::device,access_mode::read);
    ArrayHandle<unsigned int> d_index_array2(m_group2->getIndexArray(),access_location::device,access_mode::read);
    ArrayHandle<unsigned int> d_index_array3(m_group3->getIndexArray(),access_location::device,access_mode::read);
    ArrayHandle<unsigned int> d_index_array4(m_group4->getIndexArray(),access_location::device,access_mode::read);

    //sanity check
    assert(d_ref_angles.data != NULL);
    assert(d_oldnew_angles.data != NULL);
    assert(d_angles.data != NULL);
    assert(d_pos.data != NULL);
    assert(d_torque.data != NULL);

    assert(d_index_array1.data != NULL);
    assert(d_index_array2.data != NULL);
    assert(d_index_array3.data != NULL);
    assert(d_index_array4.data != NULL);


    unsigned int group_size = m_group1->getNumMembers();
    unsigned int N = m_pdata->getN();
    // run the kernel in parallel on all GPUs
    this->m_tuner->begin();
    kernel::gpu_compute_torsional_sin_forces(group_size,box,d_pos.data,d_torque.data,
                                                 d_index_array1.data,
                                                 d_index_array2.data,
                                                 d_index_array3.data,
                                                 d_index_array4.data,
                                                 d_ref_angles.data,
                                                 d_angles.data,
                                                 d_oldnew_angles.data,
                                                 m_oldnew_value,
                                                 d_group_typeval.data,
                                                 d_params.data,
                                                 timestep,
                                                 this->m_tuner->getParam());
                                                 //,this->m_exec_conf->dev_prop.warpSize);
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    this->m_tuner->end();
    }


// void TorsionalForceComputeGPU::computeForces(uint64_t timestep)
//     {
//     ArrayHandle<DihedralData::members_t> d_gpu_dihedral_list(m_dihedral_data->getGPUTable(),
//                                                              access_location::device,
//                                                              access_mode::read);
//     ArrayHandle<unsigned int> d_n_dihedrals(m_dihedral_data->getNGroupsArray(),
//                                             access_location::device,
//                                             access_mode::read);
//     ArrayHandle<unsigned int> d_dihedrals_ABCD(m_dihedral_data->getGPUPosTable(),
//                                                access_location::device,
//                                                access_mode::read);
//
//     // the dihedral table is up to date: we are good to go. Call the kernel
//     ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
//     BoxDim box = m_pdata->getGlobalBox();
//
//     ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::overwrite);
//     ArrayHandle<Scalar> d_virial(m_virial, access_location::device, access_mode::overwrite);
//     ArrayHandle<Scalar4> d_params(m_params, access_location::device, access_mode::read);
//
//     ArrayHandle<Scalar> d_angles(m_angles, access_location::device, access_mode::readwrite);
//     ArrayHandle<Scalar> d_ref_angles(m_ref_angles, access_location::device, access_mode::readwrite);
//     ArrayHandle<Scalar2> d_oldnew_angles(m_oldnew_angles, access_location::device, access_mode::readwrite);
//     ArrayHandle<unsigned int> d_index_array1(m_group1->getIndexArray(),access_location::device,access_mode::read);
//     ArrayHandle<unsigned int> d_index_array2(m_group2->getIndexArray(),access_location::device,access_mode::read);
//     ArrayHandle<unsigned int> d_index_array3(m_group3->getIndexArray(),access_location::device,access_mode::read);
//     ArrayHandle<unsigned int> d_index_array4(m_group4->getIndexArray(),access_location::device,access_mode::read);
//
//         //sanity check
//     assert(d_ref_angles.data != NULL);
//     assert(d_oldnew_angles.data != NULL);
//     assert(d_angles.data != NULL);
//     assert(d_pos.data != NULL);
//     assert(d_torque.data != NULL);
//
//     assert(d_index_array1.data != NULL);
//     assert(d_index_array2.data != NULL);
//     assert(d_index_array3.data != NULL);
//     assert(d_index_array4.data != NULL);
//
//     unsigned int group_size = m_group1->getNumMembers();
//
//     // run the kernel in parallel on all GPUs
//     this->m_tuner->begin();
//     kernel::gpu_compute_torsional_sin_forces(d_force.data,
//                                                  d_virial.data,
//                                                  m_virial.getPitch(),
//                                                  m_pdata->getN(),
//                                                  d_pos.data,
//                                                  box,
//                                                  d_gpu_dihedral_list.data,
//                                                  d_dihedrals_ABCD.data,
//                                                  m_dihedral_data->getGPUTableIndexer().getW(),
//                                                  d_n_dihedrals.data,
//                                                  d_params.data,
//                                                  m_dihedral_data->getNTypes(),
//                                                  this->m_tuner->getParam(),
//                                                  this->m_exec_conf->dev_prop.warpSize);
//     if (m_exec_conf->isCUDAErrorCheckingEnabled())
//         CHECK_CUDA_ERROR();
//     this->m_tuner->end();
//     }

namespace detail
    {
void export_TorsionalForceComputeGPU(pybind11::module& m)
    {
    pybind11::class_<TorsionalForceComputeGPU,
                     TorsionalForceCompute,
                     std::shared_ptr<TorsionalForceComputeGPU>>(
        m,
        "TorsionalForceComputeGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,std::shared_ptr<ParticleGroup>,std::shared_ptr<ParticleGroup>,std::shared_ptr<ParticleGroup>,std::shared_ptr<ParticleGroup>,unsigned int >());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
