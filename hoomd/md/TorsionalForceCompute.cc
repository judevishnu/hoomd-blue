// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "TorsionalForceCompute.h"

#include <iostream>
#include <math.h>
#include <sstream>
#include <stdexcept>
#include <vector>

using namespace std;

/*! \file TorsionalForceCompute.cc
    \brief Contains code for the HarmonicDihedralForceCompute class
*/

namespace hoomd
    {
namespace md
    {
/*! \param sysdef System to compute forces on
    \post Memory is allocated, and forces are zeroed.
*/
TorsionalForceCompute::TorsionalForceCompute(std::shared_ptr<SystemDefinition> sysdef,std::shared_ptr<ParticleGroup> group1,std::shared_ptr<ParticleGroup> group2,unsigned int num_angles)
    : ForceCompute(sysdef), m_group1(group1), m_group2(group2),m_num_angles(num_angles), m_K(NULL), m_sign(NULL), m_multi(NULL), m_phi_0(NULL), m_t_qx(NULL), m_t_qy(NULL), m_t_qz(NULL)
    {
    m_exec_conf->msg->notice(5) << "Constructing TorsionalForceCompute" << endl;

    // access the dihedral data for later use
    m_dihedral_data = m_sysdef->getDihedralData();
    const unsigned int size = (unsigned int)m_dihedral_data->getN();
    // check for some silly errors a user could make
    if (m_dihedral_data->getNTypes() == 0)
        {
        throw runtime_error("No dihedral types in the system.");
        }

    // allocate the parameters
    m_K = new Scalar[m_dihedral_data->getNTypes()];
    m_sign = new Scalar[m_dihedral_data->getNTypes()];
    m_multi = new int[m_dihedral_data->getNTypes()];
    m_phi_0 = new Scalar[m_dihedral_data->getNTypes()];
    m_t_qx = new Scalar[m_dihedral_data->getNTypes()];
    m_t_qy = new Scalar[m_dihedral_data->getNTypes()];
    m_t_qz = new Scalar[m_dihedral_data->getNTypes()];
    // allocate storage for the angles, newangles and old angles
    GPUArray<Scalar> angles(m_num_angles, m_dihedral_data->getNTypes(), m_exec_conf);
    m_angles.swap(angles);
    GPUArray<Scalar2> oldnew_angles(m_num_angles, m_dihedral_data->getNTypes(), m_exec_conf);
    m_oldnew_angles.swap(oldnew_angles);

    assert(!m_angles.isNull());
    assert(!m_oldnew_angles.isNull());
    Index2D oldnew_value((unsigned int)m_oldnew_angles.getPitch(),
                        (unsigned int)m_dihedral_data->getNTypes());
    m_oldnew_value = oldnew_value;

    }

TorsionalForceCompute::~TorsionalForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying TorsionalForceCompute" << endl;

    delete[] m_K;
    delete[] m_sign;
    delete[] m_multi;
    delete[] m_phi_0;
    delete[] m_t_qx;
    delete[] m_t_qy;
    delete[] m_t_qz;
    m_K = NULL;
    m_sign = NULL;
    m_multi = NULL;
    m_phi_0 = NULL;
    m_t_qx = NULL;//make_scalar3(0.,0.,0.);
    m_t_qy = NULL;
    m_t_qz = NULL;
    }

/*! \param type Type of the dihedral to set parameters for
    \param K Stiffness parameter for the force computation
    \param sign the sign of the cosign term
    \param multiplicity of the dihedral itself

    Sets parameters for the potential of a particular dihedral type
*/
void TorsionalForceCompute::setParams(unsigned int type,
                                             Scalar K,
                                             Scalar sign,
                                             int multiplicity,
                                             Scalar phi_0, Scalar t_qx, Scalar t_qy, Scalar t_qz, int nang)
    {
    // make sure the type is valid
    if (type >= m_dihedral_data->getNTypes())
        {
        throw runtime_error("Invalid dihedral type.");
        }

    m_K[type] = K;
    m_sign[type] = sign;
    m_multi[type] = multiplicity;
    m_phi_0[type] = phi_0;
    m_t_qx[type] = t_qx;
    m_t_qy[type] = t_qy;
    m_t_qz[type] = t_qz;

    // check for some silly errors a user could make
    if (K <= 0)
        m_exec_conf->msg->warning() << "torsional.sin: specified K <= 0" << endl;
    if (sign != 1 && sign != -1)
        m_exec_conf->msg->warning()
            << "torsional.sin: a non unitary sign was specified" << endl;
    if (phi_0 < 0 || phi_0 >= 2 * M_PI)
        m_exec_conf->msg->warning()
            << "torsional.sin: specified phi_0 outside [0, 2pi)" << endl;

    ArrayHandle<Scalar> h_angles(m_angles, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar2> h_oldnew_angles(m_oldnew_angles, access_location::host, access_mode::readwrite);

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);


    // get a local copy of the simulation box too
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
        unsigned int idp = m_group1->getMemberIndex(i);
        unsigned int idn = m_group2->getMemberIndex(i);
        unsigned int tagp = m_group1->getMemberTag(i);
        unsigned int tagn = m_group2->getMemberTag(i);
        unsigned int rtagp = h_rtag.data[tagp];
        unsigned int rtagn = h_rtag.data[tagn];
        unsigned int rtagpside;
        unsigned int rtagnside;
        unsigned int dihedral_type = m_dihedral_data->getTypeByIndex(i);
        if (rtagp == idx_b)
            {
              rtagpside = idx_a;
              rtagnside = idx_d;
              rtagn = idx_c;
            }
        else if (rtagp == idx_c)
            {
              rtagpside = idx_d;
              rtagnside = idx_a;
              rtagn = idx_b;

            }
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
        h_angles.data[i] = 0;

        }

    }

void TorsionalForceCompute::setParamsPython(std::string type, pybind11::dict params)
    {
    // make sure the type is valid
    auto typ = m_dihedral_data->getTypeByName(type);
    torsional_sin_params _params(params);
    printf("Iam set %f %f %d %f %f %f %f \n",_params.k, _params.d, _params.n, _params.phi_0, _params.t_qx,_params.t_qy,_params.t_qz);
    setParams(typ, _params.k, _params.d, _params.n, _params.phi_0, _params.t_qx, _params.t_qy, _params.t_qz);
    }

pybind11::dict TorsionalForceCompute::getParams(std::string type)
    {
    auto typ = m_dihedral_data->getTypeByName(type);
    ArrayHandle<Scalar> h_angles(m_angles, access_location::host, access_mode::read);

    pybind11::dict params;
    params["k"] = m_K[typ];
    params["d"] = m_sign[typ];
    params["n"] = m_multi[typ];
    params["phi0"] = m_phi_0[typ];
    params["tqx"] = m_t_qx[typ];
    params["tqy"] = m_t_qy[typ];
    params["tqz"] = m_t_qz[typ];
    params["nang"] = m_num_angles;
    auto angL = pybind11::array_t<Scalar>(m_num_angles);
    auto angL_unchecked = angL.mutable_unchecked<1>();

    for (unsigned int i = 0; i < m_num_angles; i++)
        {
        angL_unchecked(i) = h_angles.data[i];
        }

    params["angles"] = angL;

    //printf("I am get %f %f %d %f %f %f %f \n",params["k"], params["d"], params["n"], params["phi_0"], params["t_qx"],params["t_qy"],params["t_qz"]);

    return params;
    }

/*! Actually perform the force computation
    \param timestep Current time step
 */
void TorsionalForceCompute::computeForces(uint64_t timestep)
    {
    assert(m_pdata);

    // access the particle data arrays
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
    //access angles , old and new angles in readwrite mode
    ArrayHandle<Scalar> h_angles(m_angles, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar2> h_oldnew_angles(m_oldnew_angles, access_location::host, access_mode::readwrite);
    //Change force to Torque
    //ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar4> h_torque(m_torque, access_location::host, access_mode::overwrite);

    //ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);

    // Zero data for force calculation.
    //memset((void*)h_force.data, 0, sizeof(Scalar4) * m_force.getNumElements());
    memset((void*)h_torque.data, 0, sizeof(Scalar4) * m_torque.getNumElements());

    //memset((void*)h_virial.data, 0, sizeof(Scalar) * m_virial.getNumElements());

    // there are enough other checks on the input data: but it doesn't hurt to be safe
    //assert(h_force.data);
    assert(h_torque.data);

    //assert(h_virial.data);
    assert(h_pos.data);
    assert(h_rtag.data);

    size_t virial_pitch = m_virial.getPitch();

    // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getBox();

    // for each of the dihedrals
    const unsigned int size = (unsigned int)m_dihedral_data->getN();
    //const unsigned int num_p_n = (unsigned int)(m_group1->getNumMembersGlobal() + m_group2->getNumMembersGlobal())
    for (unsigned int i = 0; i < size; i++)
        {
        // lookup the tag of each of the particles participating in the dihedral
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
        unsigned int idp = m_group1->getMemberIndex(i);
        unsigned int idn = m_group2->getMemberIndex(i);
        unsigned int tagp = m_group1->getMemberTag(i);
        unsigned int tagn = m_group2->getMemberTag(i);
        unsigned int rtagp = h_rtag.data[tagp];
        unsigned int rtagn = h_rtag.data[tagn];
        unsigned int rtagpside;
        unsigned int rtagnside;
        unsigned int dihedral_type = m_dihedral_data->getTypeByIndex(i);
        printf("I am computeForces %f %f %d %f %f %f %f \n",m_K[dihedral_type], m_sign[dihedral_type], m_multi[dihedral_type], m_phi_0[dihedral_type], m_t_qx[dihedral_type], m_t_qy[dihedral_type], m_t_qz[dihedral_type]);

        if (rtagp == idx_b)
            {
              rtagpside = idx_a;
              rtagnside = idx_d;
              rtagn = idx_c;
            }
        else if (rtagp == idx_c)
            {
              rtagpside = idx_d;
              rtagnside = idx_a;
              rtagn = idx_b;

            }


        // printf("dihedral particle ids %u %u %u %u %u %u %u %u \n",dihedral.tag[0],dihedral.tag[1],dihedral.tag[2],dihedral.tag[3],h_rtag.data[dihedral.tag[0]],h_rtag.data[dihedral.tag[1]],h_rtag.data[dihedral.tag[2]],h_rtag.data[dihedral.tag[3]]);
        // printf("group particle ids %u %u %u %u \n",idp,idn,h_rtag.data[idp],h_rtag.data[idn]);
        // printf("tag  %u %u %u %u %u %u \n",idp,idn,taga,tagb,h_rtag.data[taga],h_rtag.data[tagb]);

        // throw an error if this angle is incomplete
        if (idx_a == NOT_LOCAL || idx_b == NOT_LOCAL || idx_c == NOT_LOCAL || idx_d == NOT_LOCAL)
            {
            this->m_exec_conf->msg->error()
                << "torsional.sin: dihedral " << dihedral.tag[0] << " " << dihedral.tag[1]
                << " " << dihedral.tag[2] << " " << dihedral.tag[3] << " incomplete." << endl
                << endl;
            throw std::runtime_error("Error in torsional calculation");
            }

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
        //
        // apply periodic boundary conditions
        dab = box.minImage(dab);
        // dcb = box.minImage(dcb);
        ddc = box.minImage(ddc);
        //####################################################################################################
        Scalar angl;
        Scalar diffangl;
        Scalar tmpangl;
        Scalar oldangl;
        Scalar3 torqp;
        Scalar3 torqn;
        Scalar3 constT;
        torqp.x = 0.0;
        torqp.y = 0.0;
        torqp.z = 0.0;

        tmpangl = atan2(dab.y, dab.x) - atan2(ddc.y, ddc.x);
        tmpangl = anglDiff(tmpangl);
        oldangl = h_oldnew_angles.data[m_oldnew_value(i, dihedral_type)].x;
        diffangl = tmpangl - oldangl;
        diffangl = anglDiff(diffangl);
        h_oldnew_angles.data[m_oldnew_value(i, dihedral_type)].y = tmpangl;
        angl = h_angles.data[i]+diffangl;
        h_angles.data[i] = angl;
        Scalar cs = fast::cos(angl);
        Scalar ss = fast::sin(angl);
        h_oldnew_angles.data[m_oldnew_value(i, dihedral_type)].x = tmpangl;

        if (angl> M_PI)
            {
            ss = fast::sin(angl- M_PI);
            cs = fast::sin(angl- M_PI);
            torqp.x =  0.0 ;
            torqp.y =  0.0 ;
            torqp.z =  -2*m_K[dihedral_type]*cs*ss;
            torqn.x =  0.0 ;
            torqn.y =  0.0 ;
            torqn.z =  2*m_K[dihedral_type]*cs*ss;
            }
        else if (angl < 0)
            {
            torqp.x =  0.0 ;
            torqp.y =  0.0 ;
            torqp.z =  -2*m_K[dihedral_type]*cs*ss;
            torqn.x =  0.0 ;
            torqn.y =  0.0 ;
            torqn.z =  2*m_K[dihedral_type]*cs*ss;

            }
        else if (angl == 0)
            {
            if (timestep < 1000)
                {
                  torqp.x =  m_t_qx[dihedral_type];
                  torqp.y =  m_t_qy[dihedral_type];
                  torqp.z =  m_t_qz[dihedral_type];
                  torqn.x =  m_t_qx[dihedral_type];
                  torqn.y =  m_t_qy[dihedral_type];
                  torqn.z = -m_t_qz[dihedral_type];
                }
            }
        h_torque.data[rtagp].x += torqp.x;
        h_torque.data[rtagp].y += torqp.y;
        h_torque.data[rtagp].z += torqp.z;
        h_torque.data[rtagp].w = 0;
        h_torque.data[rtagn].x += torqn.x;
        h_torque.data[rtagn].y += torqn.y;
        h_torque.data[rtagn].z += torqn.z;
        h_torque.data[rtagn].w = 0;


        }
    }

namespace detail
    {
void export_TorsionalForceCompute(pybind11::module& m)
    {
    pybind11::class_<TorsionalForceCompute,
                     ForceCompute,
                     std::shared_ptr<TorsionalForceCompute>>(m,
                                                                    "TorsionalForceCompute")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,std::shared_ptr<ParticleGroup>,std::shared_ptr<ParticleGroup>>())
        .def("setParams", &TorsionalForceCompute::setParamsPython)
        .def("getParams", &TorsionalForceCompute::getParams)
        .def_property_readonly("filter1",
                               [](TorsionalForceCompute& force)
                               { return force.getGroup1()->getFilter(); })
        .def_property_readonly("filter2",
                               [](TorsionalForceCompute& force)
                               { return force.getGroup2()->getFilter(); });
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
