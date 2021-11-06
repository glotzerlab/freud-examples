import itertools
import math

import gsd.hoomd
import hoomd
import numpy

if __name__ == "__main__":
    # create a particle lattice
    m = 10
    N_particles = m ** 3
    spacing = 1.3
    K = math.ceil(N_particles ** (1 / 3))
    L = K * spacing
    x = numpy.linspace(-L / 2, L / 2, K, endpoint=False)
    position = list(itertools.product(x, repeat=3))

    # create a Snapshot from the particle configuration
    snapshot = gsd.hoomd.Snapshot()
    snapshot.particles.N = N_particles
    snapshot.particles.position = position[0:N_particles]
    snapshot.particles.typeid = [0] * N_particles
    snapshot.configuration.box = [L, L, L, 0, 0, 0]
    snapshot.particles.types = ["A"]

    # initialize a simulation from the snapshot
    cpu = hoomd.device.CPU()
    sim = hoomd.Simulation(device=cpu, seed=1)
    sim.create_state_from_snapshot(snapshot)

    # configure the integrator and LJ potential
    integrator = hoomd.md.Integrator(dt=0.005)
    cell = hoomd.md.nlist.Cell()
    lj = hoomd.md.pair.LJ(nlist=cell)
    lj.params[("A", "A")] = dict(epsilon=1, sigma=1)
    lj.r_cut[("A", "A")] = 2.5
    integrator.forces.append(lj)
    nvt = hoomd.md.methods.NVT(kT=1.5, filter=hoomd.filter.All(), tau=1.0)
    integrator.methods.append(nvt)

    # link the integrator to the simulation, assign initial particle momenta
    sim.operations.integrator = integrator
    sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=1.5)

    # add a gsd writer, this will write 4 frames from the trajectory
    gsd_writer = hoomd.write.GSD(filename="../data/LJsampleTraj.gsd",
                                trigger=hoomd.trigger.Periodic(2500),
                                mode="wb")
    sim.operations.writers.append(gsd_writer)

    sim.run(10000)
