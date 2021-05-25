import hoomd
import pytest
from hoomd.conftest import pickling_check


def test_attributes():
    active = hoomd.md.force.Active(filter=hoomd.filter.All(),
                                   rotation_diff=0.01)

    assert active.rotation_diff == 0.01
    assert active.active_force['A'] == (1.0, 0.0, 0.0)
    assert active.active_torque['A'] == (0.0, 0.0, 0.0)
    assert active.manifold_constraint is None

    active.rotation_diff = 0.1
    assert active.rotation_diff == 0.1
    active.active_force['A'] = (0.5, 0.0, 0.0)
    assert active.active_force['A'] == (0.5, 0.0, 0.0)
    active.active_force['A'] = (0.0, 0.0, 1.0)
    assert active.active_force['A'] == (0.0, 0.0, 1.0)

    plane = hoomd.md.manifold.Plane()
    with pytest.raises(AttributeError):
        active.manifold_constraint = plane
    assert active.manifold_constraint is None


def test_attach(simulation_factory, two_particle_snapshot_factory):
    active = hoomd.md.force.Active(filter=hoomd.filter.All(),
                                   rotation_diff=0.01)

    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=8))
    integrator = hoomd.md.Integrator(.05)
    integrator.methods.append(
        hoomd.md.methods.Langevin(hoomd.filter.All(), kT=0))
    integrator.forces.append(active)
    sim.operations.integrator = integrator
    sim.run(0)

    assert active.rotation_diff == 0.01
    assert active.active_force['A'] == (1.0, 0.0, 0.0)
    assert active.active_torque['A'] == (0.0, 0.0, 0.0)

    active.rotation_diff = 0.1
    assert active.rotation_diff == 0.1
    active.active_force['A'] = (0.5, 0.0, 0.0)
    assert active.active_force['A'] == (0.5, 0.0, 0.0)
    active.active_force['A'] = (0.0, 0.0, 1.0)
    assert active.active_force['A'] == (0.0, 0.0, 1.0)


def test_attach_manifold(simulation_factory, two_particle_snapshot_factory):
    plane = hoomd.md.manifold.Plane()
    active = hoomd.md.force.Active(filter=hoomd.filter.All(),
                                   rotation_diff=0.01,
                                   manifold_constraint=plane)

    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=8))
    integrator = hoomd.md.Integrator(.05)
    integrator.methods.append(
        hoomd.md.methods.Langevin(hoomd.filter.All(), kT=0))
    integrator.forces.append(active)
    sim.operations.integrator = integrator
    sim.run(0)

    assert active.rotation_diff == 0.01
    assert active.active_force['A'] == (1.0, 0.0, 0.0)
    assert active.active_torque['A'] == (0.0, 0.0, 0.0)
    assert active.manifold_constraint == plane

    active.rotation_diff = 0.1
    assert active.rotation_diff == 0.1
    active.active_force['A'] = (0.5, 0.0, 0.0)
    assert active.active_force['A'] == (0.5, 0.0, 0.0)
    active.active_force['A'] = (0.0, 0.0, 1.0)
    assert active.active_force['A'] == (0.0, 0.0, 1.0)
    sphere = hoomd.md.manifold.Sphere(r=2)
    with pytest.raises(AttributeError):
        active.manifold_constraint = sphere
    assert active.manifold_constraint == plane


def test_pickling(simulation_factory, two_particle_snapshot_factory):
    sim = simulation_factory(two_particle_snapshot_factory())
    active = hoomd.md.force.Active(filter=hoomd.filter.All(),
                                   rotation_diff=0.01)
    pickling_check(active)
    integrator = hoomd.md.Integrator(
        .05,
        methods=[hoomd.md.methods.Langevin(hoomd.filter.All(), kT=0)],
        forces=[active])
    sim.operations.integrator = integrator
    sim.run(0)
    pickling_check(active)
