import numpy as np
import freud


def make_polygon(sides, radius=1):
    thetas = np.linspace(0, 2*np.pi, sides+1)[:sides]
    vertices = np.array([[radius*np.sin(theta), radius*np.cos(theta)]
                         for theta in thetas])
    return vertices


def default_bokeh(plot):
    """Wrapper which takes the default bokeh outputs and changes them to more
    sensible values
    """
    plot.title.text_font_size = "18pt"
    plot.title.align = "center"

    plot.xaxis.axis_label_text_font_size = "14pt"
    plot.yaxis.axis_label_text_font_size = "14pt"

    plot.xaxis.major_tick_in = 10
    plot.xaxis.major_tick_out = 0
    plot.xaxis.minor_tick_in = 5
    plot.xaxis.minor_tick_out = 0

    plot.yaxis.major_tick_in = 10
    plot.yaxis.major_tick_out = 0
    plot.yaxis.minor_tick_in = 5
    plot.yaxis.minor_tick_out = 0

    plot.xaxis.major_label_text_font_size = "12pt"
    plot.yaxis.major_label_text_font_size = "12pt"


def cubeellipse(theta, lam=0.5, gamma=0.6, s=4.0, r=1., h=1.):
    """Create an RGB colormap from an input angle theta. Takes lam (a list of
    intensity values, from 0 to 1), gamma (a nonlinear weighting power),
    s (starting angle), r (number of revolutions around the circle), and
    h (a hue factor)."""
    lam = lam**gamma

    a = h*lam*(1 - lam)
    v = np.array([[-.14861, 1.78277], [-.29227, -.90649], [1.97294, 0.]],
                 dtype=np.float32)
    ctarray = np.array([np.cos(theta*r + s), np.sin(theta*r + s)],
                       dtype=np.float32)
    # convert to 255 rgb
    ctarray = 255*(lam + a*v.dot(ctarray)).T
    ctarray = np.clip(ctarray.astype(dtype=np.int32), 0, 255)
    return "#{0:02x}{1:02x}{2:02x}".format(*ctarray)


def local_to_global(verts, positions, orientations):
    """
    Take a list of shape vertices, positions, and orientations and create
    a list of vertices in the "global coordinate system" for plotting
    in bokeh
    """
    verts = np.asarray(verts)
    positions = np.asarray(positions)
    orientations = np.asarray(orientations)
    # create array of rotation matrices
    rot_mats = np.array([[[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]]
                         for theta in orientations])
    # rotate vertices
    r_verts = np.swapaxes(rot_mats @ verts.T, 1, 2)
    # now translate to global coordinates
    output_array = np.add(r_verts, np.tile(positions[:, np.newaxis, :],
                                           reps=(len(verts), 1)))
    return output_array
