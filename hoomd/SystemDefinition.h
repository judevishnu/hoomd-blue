// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: joaander

/*! \file SystemDefinition.h
    \brief Defines the SystemDefinition class
 */

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "BondedGroupData.h"
#include "MeshGroupData.h"
#include "IntegratorData.h"
#include "ParticleData.h"

#include <memory>
#include <pybind11/pybind11.h>

#ifndef __SYSTEM_DEFINITION_H__
#define __SYSTEM_DEFINITION_H__

#ifdef ENABLE_MPI
//! Forward declaration of Communicator
class Communicator;
#endif

//! Forward declaration of SnapshotSystemData
template<class Real> struct SnapshotSystemData;

//! Container class for all data needed to define the MD system
/*! SystemDefinition is a big bucket where all of the data defining the MD system goes.
    Everything is stored as a shared pointer for quick and easy access from within C++
    and python without worrying about data management.

    <b>Background and intended usage</b>

    The most fundamental data structure stored in SystemDefinition is the ParticleData.
    It stores essential data on a per particle basis (position, velocity, type, mass, etc...)
    as well as defining the number of particles in the system and the simulation box. Many other
    data structures in SystemDefinition also refer to particles and store other data related to
    them (i.e. BondData lists bonds between particles). These will need access to information such
    as the number of particles in the system or potentially some of the per-particle data stored
    in ParticleData. To facilitate this, ParticleData will always be initialized \b fist and its
    shared pointer can then be passed to any future data structure in SystemDefinition that needs
    such a reference.

    More generally, any data structure class in SystemDefinition can potentially reference any
   other, simply by giving the shared pointer to the referenced class to the constructor of the one
   that needs to refer to it. Note that using this setup, there can be no circular references. This
   is a \b good \b thing ^TM, as it promotes good separation and isolation of the various classes
   responsibilities.

    In rare circumstances, a references back really is required (i.e. notification of referring
   classes when ParticleData resorts particles). Any event based notifications of such should be
   managed with Nano::Signal. Any ongoing references where two data structure classes are so
   interwoven that they must constantly refer to each other should be avoided (consider merging them
   into one class).

    <b>Initializing</b>

    A default constructed SystemDefinition is full of NULL shared pointers. Such is intended to be
   assigned to by one created by a SystemInitializer.

    Several other default constructors are provided, mainly to provide backward compatibility to
   unit tests that relied on the simple initialization constructors provided by ParticleData.

    \ingroup data_structs
*/
class PYBIND11_EXPORT SystemDefinition
    {
    public:
    //! Constructs a NULL SystemDefinition
    SystemDefinition();
    //! Constructs a SystemDefinition with a simply initialized ParticleData
    SystemDefinition(unsigned int N,
                     const BoxDim& box,
                     unsigned int n_types = 1,
                     unsigned int n_bond_types = 0,
                     unsigned int n_angle_types = 0,
                     unsigned int n_dihedral_types = 0,
                     unsigned int n_improper_types = 0,
                     unsigned int n_triangle_types = 0,
                     std::shared_ptr<ExecutionConfiguration> exec_conf
                     = std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration()),
                     std::shared_ptr<DomainDecomposition> decomposition
                     = std::shared_ptr<DomainDecomposition>());

    //! Construct from a snapshot
    template<class Real>
    SystemDefinition(std::shared_ptr<SnapshotSystemData<Real>> snapshot,
                     std::shared_ptr<ExecutionConfiguration> exec_conf
                     = std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration()),
                     std::shared_ptr<DomainDecomposition> decomposition
                     = std::shared_ptr<DomainDecomposition>());

    //! Set the dimensionality of the system
    void setNDimensions(unsigned int);

    //! Get the dimensionality of the system
    unsigned int getNDimensions() const
        {
        return m_n_dimensions;
        }

    /// Set the random numbers seed
    void setSeed(uint16_t seed)
        {
        m_seed = seed;

#ifdef ENABLE_MPI
        // In case of MPI run, every rank should be initialized with the same seed.
        // Broadcast the seed of rank 0 to all ranks to correct cases where the user provides
        // different seeds

        if (this->m_particle_data->getDomainDecomposition())
            bcast(m_seed, 0, this->m_particle_data->getExecConf()->getMPICommunicator());
#endif
        }

    /// Get the random number seed
    uint16_t getSeed() const
        {
        return m_seed;
        }

    //! Get the particle data
    std::shared_ptr<ParticleData> getParticleData() const
        {
        return m_particle_data;
        }
    //! Get the bond data
    std::shared_ptr<BondData> getBondData() const
        {
        return m_bond_data;
        }
    //! Access the angle data defined for the simulation
    std::shared_ptr<AngleData> getAngleData()
        {
        return m_angle_data;
        }
    //! Access the dihedral data defined for the simulation
    std::shared_ptr<DihedralData> getDihedralData()
        {
        return m_dihedral_data;
        }
    //h Access the improper data defined for the simulation
    std::shared_ptr<ImproperData> getImproperData()
        {
        return m_improper_data;
        }
    //! Access the triangle data defined for the simulation
    std::shared_ptr<TriangleData> getTriangleData()
        {
	if(m_mesh_change){
	    TriangleData::Snapshot snapshot;
	    m_meshtriangle_data->takeSnapshot(snapshot);
            m_triangle_data = std::shared_ptr<TriangleData>(new TriangleData(m_particle_data, snapshot));
	    m_mesh_change = false;
	}
	m_triangle_change = true;
        return m_triangle_data;
        }
    //! Access the mesh triangle data defined for the simulation
    std::shared_ptr<MeshTriangleData> getMeshTriangleData()
        {
        return m_meshtriangle_data;
        }
    //! Access the mesh bond data defined for the simulation
    std::shared_ptr<MeshBondData> getMeshBondData()
        {
        return m_meshbond_data;
        }

    void checkMeshData();

    //! Access the constraint data defined for the simulation
    std::shared_ptr<ConstraintData> getConstraintData()
        {
        return m_constraint_data;
        }

    //! Returns the integrator variables (if applicable)
    std::shared_ptr<IntegratorData> getIntegratorData()
        {
        return m_integrator_data;
        }

    //! Get the pair data
    std::shared_ptr<PairData> getPairData() const
        {
        return m_pair_data;
        }

    //! Return a snapshot of the current system data
    template<class Real> std::shared_ptr<SnapshotSystemData<Real>> takeSnapshot();

    //! Re-initialize the system from a snapshot
    template<class Real>
    void initializeFromSnapshot(std::shared_ptr<SnapshotSystemData<Real>> snapshot);

    private:
    unsigned int m_n_dimensions;                       //!< Dimensionality of the system
    uint16_t m_seed = 0;                               //!< Random number seed
    std::shared_ptr<ParticleData> m_particle_data;     //!< Particle data for the system
    std::shared_ptr<BondData> m_bond_data;             //!< Bond data for the system
    std::shared_ptr<AngleData> m_angle_data;           //!< Angle data for the system
    std::shared_ptr<DihedralData> m_dihedral_data;     //!< Dihedral data for the system
    std::shared_ptr<TriangleData> m_triangle_data;           //!< Angle data for the system
    std::shared_ptr<ImproperData> m_improper_data;     //!< Improper data for the system
    std::shared_ptr<ConstraintData> m_constraint_data; //!< Improper data for the system
    std::shared_ptr<IntegratorData> m_integrator_data; //!< Integrator data for the system
    std::shared_ptr<PairData> m_pair_data;             //!< Special pairs data for the system
    std::shared_ptr<MeshBondData> m_meshbond_data;     //!< Bond data for the mesh
    std::shared_ptr<MeshTriangleData> m_meshtriangle_data; //!< Triangle data for the mesh
    bool m_triangle_change;
    bool m_mesh_change;
    };

//! Exports SystemDefinition to python
void export_SystemDefinition(pybind11::module& m);

#endif
