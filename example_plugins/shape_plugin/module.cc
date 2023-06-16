// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "ComputeFreeVolume.h"
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"
#include "IntegratorHPMCMonoNEC.h"

#include "ComputeSDF.h"
#include "ShapeMySphere.h"
#include "ShapeUnion.h"

#include "ExternalField.h"
#include "ExternalFieldComposite.h"
#include "ExternalFieldHarmonic.h"
#include "ExternalFieldWall.h"

#include "UpdaterClusters.h"
#include "UpdaterMuVT.h"

#ifdef ENABLE_HIP
#include "ComputeFreeVolumeGPU.h"
#include "IntegratorHPMCMonoGPU.h"
#include "UpdaterClustersGPU.h"
#endif

namespace hoomd
    {
namespace hpmc
    {
//! Export the base HPMCMono integrators
PYBIND11_MODULE(_shape_plugin, m)
    {
    export_IntegratorHPMCMono<ShapeMySphere>(m, "IntegratorHPMCMonoMySphere");
    export_IntegratorHPMCMonoNEC<ShapeMySphere>(m, "IntegratorHPMCMonoNECMySphere");
    export_ComputeFreeVolume<ShapeMySphere>(m, "ComputeFreeVolumeMySphere");
    export_ComputeSDF<ShapeMySphere>(m, "ComputeSDFMySphere");
    export_UpdaterMuVT<ShapeMySphere>(m, "UpdaterMuVTMySphere");
    export_UpdaterClusters<ShapeMySphere>(m, "UpdaterClustersMySphere");

    export_ExternalFieldInterface<ShapeMySphere>(m, "ExternalFieldMySphere");
    export_HarmonicField<ShapeMySphere>(m, "ExternalFieldHarmonicMySphere");
    export_ExternalFieldComposite<ShapeMySphere>(m, "ExternalFieldCompositeMySphere");
    export_ExternalFieldWall<ShapeMySphere>(m, "WallMySphere");

    pybind11::class_<MySphereParams, std::shared_ptr<MySphereParams>>(m, "MySphereParams")
        .def(pybind11::init<pybind11::dict>())
        .def("asDict", &MySphereParams::asDict);

#ifdef ENABLE_HIP
    export_IntegratorHPMCMonoGPU<ShapeMySphere>(m, "IntegratorHPMCMonoMySphereGPU");
    export_ComputeFreeVolumeGPU<ShapeMySphere>(m, "ComputeFreeVolumeMySphereGPU");
    export_UpdaterClustersGPU<ShapeMySphere>(m, "UpdaterClustersMySphereGPU");
#endif

    } // namespace hpmc
    } // namespace hoomd
