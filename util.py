import numpy as np
import freud


def make_cubic(nx=1, ny=1, nz=1, fractions=np.array([[0, 0, 0]],
               dtype=np.float32), scale=1.0, noise=0.0):
    """Make a cubic crystal for testing.

    Args:
        nx: Number of repeats in the x direction, default is 1.
        ny: Number of repeats in the y direction, default is 1.
        nz: Number of repeats in the z direction, default is 1.
        fractions: The basis to replicate using the lattice.
        scale: Amount to scale the unit cell by (in distance units), default is 1.0.
        noise: Apply Gaussian noise with this width to particle positions (Default value = 0.0).

    Returns:
        tuple (py:class:`freud.box.Box`, :class:`np.ndarray`): freud Box, particle positions, shape=(nx*ny*nz, 3)
    """

    fractions = np.tile(fractions[np.newaxis, np.newaxis, np.newaxis],
                        (nx, ny, nz, 1, 1))
    fractions[..., 0] += np.arange(nx)[:, np.newaxis, np.newaxis, np.newaxis]
    fractions[..., 1] += np.arange(ny)[np.newaxis, :, np.newaxis, np.newaxis]
    fractions[..., 2] += np.arange(nz)[np.newaxis, np.newaxis, :, np.newaxis]
    fractions /= [nx, ny, nz]

    box = 2*scale*np.array([nx, ny, nz], dtype=np.float32)
    positions = ((fractions - .5)*box).reshape((-1, 3))

    if noise != 0:
        positions += np.random.normal(scale=noise, size=positions.shape)

    return freud.box.Box(*box), positions

def make_fcc(nx=1, ny=1, nz=1, scale=1.0, noise=0.0):
    """Make a FCC crystal for testing

    Args:
        nx: Number of repeats in the x direction, default is 1
        ny: Number of repeats in the y direction, default is 1
        nz: Number of repeats in the z direction, default is 1
        scale: Amount to scale the unit cell by (in distance units), default is 1.0
        noise: Apply Gaussian noise with this width to particle positions (Default value = 0.0)

    Returns:
        tuple (py:class:`freud.box.Box`, :class:`np.ndarray`): freud Box, particle positions, shape=(nx*ny*nz, 3)
    """
    fractions = np.array([[.5, .5, 0],
                          [.5, 0, .5],
                          [0, .5, .5],
                          [0, 0, 0]], dtype=np.float32)
    return make_cubic(nx, ny, nz, fractions, scale, noise)

def make_bcc(nx=1, ny=1, nz=1, scale=1.0, noise=0.0):
    """Make a BCC crystal for testing

    Args:
        nx: Number of repeats in the x direction, default is 1
        ny: Number of repeats in the y direction, default is 1
        nz: Number of repeats in the z direction, default is 1
        scale: Amount to scale the unit cell by (in distance units), default is 1.0
        noise: Apply Gaussian noise with this width to particle positions (Default value = 0.0)

    Returns:
        tuple (py:class:`freud.box.Box`, :class:`np.ndarray`): freud Box, particle positions, shape=(nx*ny*nz, 3)
    """
    fractions = np.array([[0.5, .5, .5],
                          [0, 0, 0]], dtype=np.float32)
    return make_cubic(nx, ny, nz, fractions, scale, noise)

def make_sc(nx=1, ny=1, nz=1, scale=1.0, noise=0.0):
    """Make an SC crystal for testing

    Args:
        nx: Number of repeats in the x direction, default is 1
        ny: Number of repeats in the y direction, default is 1
        nz: Number of repeats in the z direction, default is 1
        scale: Amount to scale the unit cell by (in distance units), default is 1.0
        noise: Apply Gaussian noise with this width to particle positions (Default value = 0.0)

    Returns:
        tuple (py:class:`freud.box.Box`, :class:`np.ndarray`): freud Box, particle positions, shape=(nx*ny*nz, 3)
    """
    fractions = np.array([[0, 0, 0]], dtype=np.float32)
    return make_cubic(nx, ny, nz, fractions, scale, noise)

def make_square(nx=1, ny=1, fractions=np.array([[0, 0, 0]], dtype=np.float32),
                scale=1.0, noise=0.0):
    """Make a square crystal for testing

    Args:
        nx: Number of repeats in the x direction, default is 1
        ny: Number of repeats in the y direction, default is 1
        fractions: The basis to replicate using the lattice.
        scale: Amount to scale the unit cell by (in distance units), default is 1.0
        noise: Apply Gaussian noise with this width to particle positions (Default value = 0.0)

    Returns:
        tuple (py:class:`freud.box.Box`, :class:`np.ndarray`): freud Box, particle positions, shape=(nx*ny*nz, 3)
    """
    fractions = np.tile(fractions[np.newaxis, np.newaxis, np.newaxis],
                        (nx, ny, 1, 1, 1))
    fractions[..., 0] += np.arange(nx)[:, np.newaxis, np.newaxis, np.newaxis]
    fractions[..., 1] += np.arange(ny)[np.newaxis, :, np.newaxis, np.newaxis]
    fractions /= [nx, ny, 1]

    box = 2*scale*np.array([nx, ny, 0], dtype=np.float32)
    positions = ((fractions - .5)*box).reshape((-1, 3))

    if noise != 0:
        positions += np.random.normal(scale=noise, size=positions.shape)

    return freud.box.Box(*box), positions

def box_2d_to_points(box):
    """Generate the points needed to plot a 2d box.
    
    Args:
        box (:py:class:`freud.box.Box`): The Box to plot.
     
    Returns:
        :py:class:`np.ndarray`: The set of points to plot.
    
    Example::
        box = freud.box.Box.cube(L=5):
        points = box_2d_to_points(box)
        ax.plot(points[:, 0], points[:, 1])
    """
    points = []
    points.append(box.makeCoordinates([0, 0, 0])[:2])
    points.append(box.makeCoordinates([0, 1, 0])[:2])
    points.append(box.makeCoordinates([1, 1, 0])[:2])
    points.append(box.makeCoordinates([1, 0, 0])[:2])
    points.append(points[0])  # Need to copy this so that the box is closed.
    points = np.array(points)
    
    return points

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
