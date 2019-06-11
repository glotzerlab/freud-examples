import hoomd
from hoomd import hpmc
import freud
import numpy as np

hoomd.context.initialize('--mode=cpu')
system = hoomd.init.create_lattice(
    hoomd.lattice.sc(a=1), n=10)
mc = hpmc.integrate.sphere(seed=42, d=0.1, a=0.1)
mc.shape_param.set('A', diameter=0.5)

rdf = freud.density.RDF(rmax=4, dr=0.1)

box = freud.box.Box.from_box(system.box)
w6 = freud.order.LocalWlNear(box, 4, 6, 12)

def calc_rdf(timestep):
    snap = system.take_snapshot()
    rdf.accumulate(box, snap.particles.position)

def calc_W6(timestep):
    snap = system.take_snapshot()
    w6.compute(snap.particles.position)
    return np.mean(np.real(w6.Wl))

# Equilibrate the system before accumulating the RDF.
hoomd.run(1e6)
hoomd.analyze.callback(calc_rdf, period=100)

logger = hoomd.analyze.log(filename='output.log',
                           quantities=['w6'],
                           period=100,
                           header_prefix='#')

logger.register_callback('w6', calc_W6)

hoomd.run(2e5)

# Store the computed RDF in a file
np.savetxt('rdf.csv', np.vstack((rdf.R, rdf.RDF)).T,
           delimiter=',', header='r, rdf')
