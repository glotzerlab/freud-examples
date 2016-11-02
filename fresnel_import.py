import numpy as np
import fresnel
import time
from freud import box, order

def cubeellipse(theta, lam=0.5, gamma=1., s=4.0, r=1., h=1.):
    """Create an RGB colormap from an input angle theta. Takes lam (a list of
    intensity values, from 0 to 1), gamma (a nonlinear weighting power),
    s (starting angle), r (number of revolutions around the circle), and
    h (a hue factor)."""
    import numpy
    lam = lam**gamma

    a = h*lam*(1 - lam)*.5
    v = numpy.array([[-.14861, 1.78277], [-.29227, -.90649], [1.97294, 0.]], dtype=numpy.float32)
    ctarray = numpy.array([numpy.cos(theta*r + s), numpy.sin(theta*r + s)], dtype=numpy.float32)
    return (lam + a*v.dot(ctarray)).T

# load the data
data_path = "ex_data/phi065"
box_data = np.load("{}/box_data.npy".format(data_path))
pos_data = np.load("{}/pos_data.npy".format(data_path))
quat_data = np.load("{}/quat_data.npy".format(data_path))
n_frames = pos_data.shape[0]

device = fresnel.Device(mode='cpu')
l_box = box_data[-1]
l_pos = pos_data[-1]
fbox = box.Box(Lx=l_box["Lx"], Ly=l_box["Ly"], is2D=True)
start_time = time.time()
hex_order = order.HexOrderParameter(1.2, 6, 6);
stop_time = time.time()
print("time to calc 1 frame = {}".format(stop_time-start_time))
hex_order.compute(fbox, l_pos);
psi_k = hex_order.getPsi();
avg_psi_k = np.mean(psi_k);
a = np.angle(psi_k) - np.angle(avg_psi_k) + 1.0
a_unshift = np.angle(psi_k) - np.angle(avg_psi_k)
scene = fresnel.Scene(device)
w = 1920
# this controls the scene units
# whitted = fresnel.tracer.Whitted(device, w, int(w*1.61))
whitted = fresnel.tracer.Whitted(device, w, w)
verts = [[0.537284965911771, 0.31020161970069976],
  [3.7988742065678664e-17, 0.6204032394013997],
  [-0.5372849659117709, 0.31020161970070004],
  [-0.5372849659117711, -0.31020161970069976],
  [-1.1396622619703597e-16, -0.6204032394013997],
  [0.5372849659117711, -0.3102016197006997]]
verts = np.array(verts)
l_quat = quat_data[-1]
l_ang = 2*np.arctan2(np.copy(l_quat[:,3]), np.copy(l_quat[:,0]))
color = [tuple(cubeellipse(x)) for x in a_unshift]
# color = [(0.5,0.5,0.5) for x in a_unshift]
height = np.ones(shape=a.shape)
g = fresnel.geometry.Prism(scene, verts, l_pos[:,0:2], l_ang, height, color)
g.material = fresnel.material.Material(solid=0.0, color=(0,0.8,1))
cam = fresnel.camera.Orthographic(position=(0, 0, 1000), look_at=(0,0,0), up=(0,1,0), height=40)
whitted.set_camera(cam)
a = whitted.render(scene)