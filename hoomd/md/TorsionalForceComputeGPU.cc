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

    //GPUArray<Scalar> angles(m_num_angles, m_dihedral_data->getNTypes(), m_exec_conf);
    //m_angles.swap(angles);

    //GPUArray<Scalar> ref_angles(m_num_angles, m_dihedral_data->getNTypes(), m_exec_conf);
    //m_ref_angles.swap(ref_angles);

    GPUArray<Scalar4> oldnew_angles(m_num_angles, m_dihedral_data->getNTypes(), m_exec_conf);
    m_oldnew_angles.swap(oldnew_angles);

    //assert(!m_angles.isNull());
    //assert(!m_ref_angles.isNull());
    assert(!m_oldnew_angles.isNull());
    Index2D oldnew_value((unsigned int)m_oldnew_angles.getPitch(),
                        (unsigned int)m_dihedral_data->getNTypes());
    m_oldnew_value = oldnew_value;

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

    //ArrayHandle<Scalar> h_angles(m_angles, access_location::host, access_mode::readwrite);
    //ArrayHandle<Scalar> h_ref_angles(m_ref_angles, access_location::host, access_mode::readwrite);

    ArrayHandle<Scalar4> h_oldnew_angles(m_oldnew_angles, access_location::host, access_mode::readwrite);

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);


    // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getBox();

    for (unsigned int i = 0; i < m_num_angles; i++)
        {


        unsigned int tagp = m_group1->getMemberTag(i);
        unsigned int tagn = m_group2->getMemberTag(i);
        unsigned int tagpside = m_group3->getMemberTag(i);
        unsigned int tagnside = m_group4->getMemberTag(i);
        unsigned int rtagp = h_rtag.data[tagp];
        unsigned int rtagn = h_rtag.data[tagn];
        unsigned int rtagpside = h_rtag.data[tagpside];
        unsigned int rtagnside = h_rtag.data[tagnside];
        unsigned int dihedral_type = m_dihedral_data->getTypeByIndex(i);


        Scalar3 dab;
        dab.x = h_pos.data[rtagpside].x - h_pos.data[rtagp].x;
        dab.y = h_pos.data[rtagpside].y - h_pos.data[rtagp].y;
        dab.z = h_pos.data[rtagpside].z - h_pos.data[rtagp].z;


        Scalar3 ddc;
        ddc.x = h_pos.data[rtagnside].x - h_pos.data[rtagn].x;
        ddc.y = h_pos.data[rtagnside].y - h_pos.data[rtagn].y;
        ddc.z = h_pos.data[rtagnside].z - h_pos.data[rtagn].z;

        dab = box.minImage(dab);
        ddc = box.minImage(ddc);
        //####################################################################################################
        Scalar angl;
        angl = atan2(dab.y, dab.x) - atan2(ddc.y, ddc.x);
        h_oldnew_angles.data[m_oldnew_value(i, type)].x = angl;
        h_oldnew_angles.data[m_oldnew_value(i, type)].y = angl;
        h_oldnew_angles.data[m_oldnew_value(i, type)].z = 0;
        h_oldnew_angles.data[m_oldnew_value(i, type)].w = angl;

        }


    }

/*! Internal method for computing the forces on the GPU.
    \post The force data on the GPU is written with the calculated forces

    \param timestep Current time step of the simulation

    Calls gpu_compute_harmonic_dihedral_forces to do the dirty work.
*/
void TorsionalForceComputeGPU::computeForces(uint64_t timestep)
    {

    ArrayHandle<typeval_t> d_group_typeval(m_dihedral_data->getTypeValArray(),
                                           access_location::device,
                                           access_mode::read);

    ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::read);





    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);

    BoxDim box = m_pdata->getGlobalBox();

    ArrayHandle<Scalar4> d_torque(m_torque, access_location::device, access_mode::overwrite);

    //ArrayHandle<Scalar> d_virial(m_virial, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar4> d_params(m_params, access_location::device, access_mode::read);


    //ArrayHandle<Scalar> d_angles(m_angles, access_location::device, access_mode::readwrite);
    //ArrayHandle<Scalar> d_ref_angles(m_ref_angles, access_location::device, access_mode::readwrite);

    ArrayHandle<Scalar4> d_oldnew_angles(m_oldnew_angles, access_location::device, access_mode::readwrite);



    ArrayHandle<unsigned int> d_tag_array1(m_group1->getMemberTagArray(),access_location::device,access_mode::read);
    ArrayHandle<unsigned int> d_tag_array2(m_group2->getMemberTagArray(),access_location::device,access_mode::read);
    ArrayHandle<unsigned int> d_tag_array3(m_group3->getMemberTagArray(),access_location::device,access_mode::read);
    ArrayHandle<unsigned int> d_tag_array4(m_group4->getMemberTagArray(),access_location::device,access_mode::read);

    //sanity check
    //assert(d_ref_angles.data != NULL);
    assert(d_oldnew_angles.data != NULL);
    //assert(d_angles.data != NULL);
    assert(d_pos.data != NULL);
    assert(d_torque.data != NULL);
    assert(d_rtag.data != NULL);

    assert(d_tag_array1.data != NULL);
    assert(d_tag_array2.data != NULL);
    assert(d_tag_array3.data != NULL);
    assert(d_tag_array4.data != NULL);



    unsigned int group_size = m_group1->getNumMembers();

    unsigned int N = m_pdata->getN();
    // run the kernel in parallel on all GPUs
    this->m_tuner->begin();
    kernel::gpu_compute_torsional_sin_forces(group_size,box,d_pos.data,d_torque.data,
                                                 d_rtag.data,
                                                 d_tag_array1.data,
                                                 d_tag_array2.data,
                                                 d_tag_array3.data,
                                                 d_tag_array4.data,
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
