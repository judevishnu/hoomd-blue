// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file HarmonicDihedralForceComputeGPU.cc
    \brief Defines HarmonicDihedralForceComputeGPU
*/

#include "TorsionalTrapForceComputeGPU.h"
#include <vector>

using namespace std;

namespace hoomd
    {
namespace md
    {
/*! \param sysdef System to compute bond forces on
 */
TorsionalTrapForceComputeGPU::TorsionalTrapForceComputeGPU(
    std::shared_ptr<SystemDefinition> sysdef,std::shared_ptr<ParticleGroup> group1,std::shared_ptr<ParticleGroup> group2,std::shared_ptr<ParticleGroup> group3,std::shared_ptr<ParticleGroup> group4,unsigned int num_angles)
    : TorsionalTrapForceCompute(sysdef,group1,group2,group3,group4,num_angles)
    {
    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error()
            << "Creating a TorsionalTrapForceComputeGPU with no GPU in the execution configuration"
            << endl;
        throw std::runtime_error("Error initializing TorsionalForceComputeGPU");
        }

    // allocate and zero device memory
    GPUArray<Scalar> params(m_dihedral_data->getNTypes(), m_exec_conf);
    m_params.swap(params);

    GPUArray<Scalar> angles(m_num_angles, m_dihedral_data->getNTypes(), m_exec_conf);
    m_angles.swap(angles);

    GPUArray<Scalar> ref_angles(m_num_angles, m_dihedral_data->getNTypes(), m_exec_conf);
    m_ref_angles.swap(ref_angles);

    GPUArray<Scalar2> oldnew_angles(m_num_angles, m_dihedral_data->getNTypes(), m_exec_conf);
    m_oldnew_angles.swap(oldnew_angles);

    GPUArray<Scalar3> ref_vecp(m_num_angles, m_dihedral_data->getNTypes(), m_exec_conf);
    m_ref_vecp.swap(ref_vecp);
    GPUArray<Scalar3> ref_vecn(m_num_angles, m_dihedral_data->getNTypes(), m_exec_conf);
    m_ref_vecn.swap(ref_vecn);

    assert(!m_angles.isNull());
    assert(!m_ref_angles.isNull());
    assert(!m_oldnew_angles.isNull());
    assert(!m_ref_vecp.isNull());
    assert(!m_ref_vecn.isNull());

    Index2D ref_vecp_value((unsigned int)m_ref_vecp.getPitch(),
                                            (unsigned int)m_dihedral_data->getNTypes());
    Index2D ref_vecn_value((unsigned int)m_ref_vecn.getPitch(),
                                            (unsigned int)m_dihedral_data->getNTypes());

    Index2D oldnew_value((unsigned int)m_oldnew_angles.getPitch(),
                        (unsigned int)m_dihedral_data->getNTypes());


    m_oldnew_value = oldnew_value;
    m_ref_vecp_value = ref_vecp_value;
    m_ref_vecn_value = ref_vecn_value;

    unsigned int warp_size = m_exec_conf->dev_prop.warpSize;
    m_tuner.reset(new Autotuner(warp_size,
                                1024,
                                warp_size,
                                5,
                                100000,
                                "torsional_sin",
                                this->m_exec_conf));


    }

TorsionalTrapForceComputeGPU::~TorsionalTrapForceComputeGPU() { }

/*! \param type Type of the dihedral to set parameters for
    \param K Stiffness parameter for the force computation
    \param sign the sign of the cosine term
        \param multiplicity the multiplicity of the cosine term
    \param phi_0 the phase offset

    Sets parameters for the potential of a particular dihedral type and updates the
    parameters on the GPU.
*/
void TorsionalForceComputeGPU::setParams(unsigned int type,Scalar K)
    {
    TorsionalTrapForceCompute::setParams(type, K);

    ArrayHandle<Scalar> h_params(m_params, access_location::host, access_mode::readwrite);
    // update the local copy of the memory
    h_params.data[type] = Scalar(K);

    ArrayHandle<Scalar> h_angles(m_angles, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_ref_angles(m_ref_angles, access_location::host, access_mode::readwrite);

    ArrayHandle<Scalar2> h_oldnew_angles(m_oldnew_angles, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_ref_vecp(m_ref_vecp, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_ref_vecn(m_ref_vecn, access_location::host, access_mode::readwrite);


    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getBox();

    const BoxDim& box = m_pdata->getBox();



    for (unsigned int i = 0; i < m_num_angles; i++)
        {

        const ImproperData::members_t& dihedral = m_dihedral_data->getMembersByIndex(i);
        assert(dihedral.tag[0] <= m_pdata->getMaximumTag());
        assert(dihedral.tag[1] <= m_pdata->getMaximumTag());
        assert(dihedral.tag[2] <= m_pdata->getMaximumTag());
        assert(dihedral.tag[3] <= m_pdata->getMaximumTag());

        // transform a, b, and c into indices into the particle data arrays
        // MEM TRANSFER: 6 ints
        unsigned int idx_a = h_rtag.data[dihedral.tag[0]];
        unsigned int idx_b = h_rtag.data[dihedral.tag[1]];
        unsigned int idx_c = h_rtag.data[dihedral.tag[2]];
        unsigned int idx_d = h_rtag.data[dihedral.tag[3]];
        // unsigned int idp = m_group1->getMemberIndex(i);
        // unsigned int idn = m_group2->getMemberIndex(i);
        unsigned int tagp = m_group1->getMemberTag(i);
        unsigned int tagn = m_group2->getMemberTag(i);
        unsigned int tagpside = m_group3->getMemberTag(i);
        unsigned int tagnside = m_group4->getMemberTag(i);
        unsigned int rtagp = h_rtag.data[tagp];
        unsigned int rtagn = h_rtag.data[tagn];
        unsigned int rtagpside = h_rtag.data[tagpside];
        unsigned int rtagnside = h_rtag.data[tagnside];
        unsigned int dihedral_type = m_dihedral_data->getTypeByIndex(i);

        assert(idx_a < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_b < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_c < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_d < m_pdata->getN() + m_pdata->getNGhosts());

        // calculate d\vec{r}
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
        h_ref_angles.data[i] = angl;
        h_angles.data[i] = 0;
        Scalar rsqdab = dab.x*dab.x + dab.y*dab.y + dab.z*dab.z;
        Scalar rsqrtdab = sqrt(rsqdab);
        Scalar rsqddc = ddc.x*ddc.x + ddc.y*ddc.y + ddc.z*ddc.z;
        Scalar rsqrtddc = sqrt(rsqddc);
        Scalar3 unitdab;
        unitdab.x = dab.x/rsqrtdab;
        unitdab.y = dab.y/rsqrtdab;
        unitdab.z = dab.z/rsqrtdab;
        Scalar3 unitddc;
        unitddc.x = ddc.x/rsqrtddc;
        unitddc.y = ddc.y/rsqrtddc;
        unitddc.z = ddc.z/rsqrtddc;

        h_ref_vecp.data[m_ref_vecp_value(i, type)].x = unitdab.x;
        h_ref_vecp.data[m_ref_vecp_value(i, type)].y = unitdab.y;
        h_ref_vecp.data[m_ref_vecp_value(i, type)].z = unitdab.z;

        h_ref_vecn.data[m_ref_vecn_value(i, type)].x = unitddc.x;
        h_ref_vecn.data[m_ref_vecn_value(i, type)].y = unitddc.y;
        h_ref_vecn.data[m_ref_vecn_value(i, type)].z = unitddc.z;

        }
    }

/*! Internal method for computing the forces on the GPU.
    \post The force data on the GPU is written with the calculated forces

    \param timestep Current time step of the simulation

    Calls gpu_compute_harmonic_dihedral_forces to do the dirty work.
*/
void TorsionalTrapForceComputeGPU::computeForces(uint64_t timestep)
    {

    ArrayHandle<typeval_t> d_group_typeval(m_dihedral_data->getTypeValArray(),
                                           access_location::device,
                                           access_mode::read);

    ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::read);

    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);

    BoxDim box = m_pdata->getGlobalBox();

    ArrayHandle<Scalar4> d_torque(m_torque, access_location::device, access_mode::overwrite);

    //ArrayHandle<Scalar> d_virial(m_virial, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_params(m_params, access_location::device, access_mode::read);



    ArrayHandle<Scalar> d_angles(m_angles, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> d_ref_angles(m_ref_angles, access_location::device, access_mode::readwrite);

    ArrayHandle<Scalar2> d_oldnew_angles(m_oldnew_angles, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar3> d_ref_vecp(m_ref_vecp, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar3> d_ref_vecn(m_ref_vecn, access_location::device, access_mode::readwrite);

    ArrayHandle<unsigned int> d_tag_array1(m_group1->getMemberTagArray(),access_location::device,access_mode::read);
    ArrayHandle<unsigned int> d_tag_array2(m_group2->getMemberTagArray(),access_location::device,access_mode::read);
    ArrayHandle<unsigned int> d_tag_array3(m_group3->getMemberTagArray(),access_location::device,access_mode::read);
    ArrayHandle<unsigned int> d_tag_array4(m_group4->getMemberTagArray(),access_location::device,access_mode::read);

    //sanity check
    assert(d_ref_angles.data != NULL);
    assert(d_oldnew_angles.data != NULL);
    assert(d_ref_vecp.data != NULL);
    assert(d_ref_vecn.data != NULL);
    assert(d_angles.data != NULL);
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
    kernel::gpu_compute_torsionaltrap_sin_forces(group_size,box,d_pos.data,d_torque.data,
                                                 d_rtag.data,
                                                 d_tag_array1.data,
                                                 d_tag_array2.data,
                                                 d_tag_array3.data,
                                                 d_tag_array4.data,
                                                 d_ref_angles.data,
                                                 d_angles.data,
                                                 d_oldnew_angles.data,
                                                 d_ref_vecp.data,
                                                 d_ref_vecn.data,
                                                 m_oldnew_value,
                                                 m_ref_vecp_value,
                                                 m_ref_vecn_value,
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
void export_TorsionalTrapForceComputeGPU(pybind11::module& m)
    {
    pybind11::class_<TorsionalTrapTrapForceComputeGPU,
                     TorsionalTrapForceCompute,
                     std::shared_ptr<TorsionalTrapForceComputeGPU>>(
        m,
        "TorsionalTrapForceComputeGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,std::shared_ptr<ParticleGroup>,std::shared_ptr<ParticleGroup>,std::shared_ptr<ParticleGroup>,std::shared_ptr<ParticleGroup>,unsigned int >());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
