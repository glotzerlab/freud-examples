import numpy as np
import freud

# For the MSD we also need images, which can be dumped using
# the LAMMPS dump custom command as follows:
# dump 2 all custom 100 output_custom.xyz x y z ix iy iz

# We read the number of particles, the system box, and the
# particle positions into 3 separate arrays.
N = int(np.genfromtxt(
    'output_custom.xyz', skip_header=3, max_rows=1))
box_data = np.genfromtxt(
    'output_custom.xyz', skip_header=5, max_rows=3)
data = np.genfromtxt(
    'output_custom.xyz', skip_header=9,
    invalid_raise=False)

# Remove the unwanted text rows
data = data[~np.isnan(data).all(axis=1)].reshape(-1, N, 6)

box = freud.box.Box.from_box(
    box_data[:, 1] - box_data[:, 0])

# We shift the system by half the box lengths to match the
# freud coordinate system, which is centered at the origin.
# Since all methods support periodicity, this shift is simply
# for consistency but does not affect any analyses.
data[..., :3] -= box.L/2
rdf = freud.density.RDF(rmax=4, dr=0.03, rmin=1)
for frame in data:
    rdf.accumulate(box, frame[:, :3])

msd = freud.msd.MSD(box)
msd.compute(positions=data[:, :, :3], images=data[:, :, 3:])

# The object contains all the data we need to plot the RDF
from matplotlib import pyplot as plt
plt.plot(rdf.R, rdf.RDF)
plt.show()
