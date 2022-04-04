import numpy as np
from bokeh.io import output_notebook
from bokeh.plotting import figure
from bokeh.resources import INLINE
from freud import box

output_notebook(resources=INLINE)

# define vertices for hexagons
verts = [
    [0.537284965911771, 0.31020161970069976],
    [3.7988742065678664e-17, 0.6204032394013997],
    [-0.5372849659117709, 0.31020161970070004],
    [-0.5372849659117711, -0.31020161970069976],
    [-1.1396622619703597e-16, -0.6204032394013997],
    [0.5372849659117711, -0.3102016197006997],
]
verts = np.array(verts)

# define colors for our system
c_list = [
    "#30A2DA",
    "#FC4F30",
    "#E5AE38",
    "#6D904F",
    "#9757DB",
    "#188487",
    "#FF7F00",
    "#9A2C66",
    "#626DDA",
    "#8B8B8B",
]
c_dict = dict()
c_dict[6] = c_list[0]
c_dict[5] = c_list[1]
c_dict[4] = c_list[2]
c_dict[3] = c_list[7]
c_dict[2] = c_list[3]
c_dict[1] = c_list[5]
c_dict[0] = c_list[6]
c_dict[7] = c_list[4]


class DemoData:
    """docstring for DemoData"""

    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.verts = verts
        # load data
        self.load_data()

    def load_data(self):
        self.box_data = np.copy(np.load(f"{self.data_path}/box_data.npy"))
        self.pos_data = np.copy(np.load(f"{self.data_path}/pos_data.npy"))
        self.quat_data = np.copy(np.load(f"{self.data_path}/quat_data.npy"))
        self.n_frames = self.pos_data.shape[0]

    def freud_box(self, frame):
        l_box = self.box_data[frame]
        fbox = box.Box(Lx=l_box["Lx"], Ly=l_box["Ly"], is2D=True)
        return fbox

    def plot_frame(self, frame_idx, title="System Visualization", linked_plot=None):
        l_box = self.box_data[frame_idx]
        l_pos = self.pos_data[frame_idx]
        l_quat = self.quat_data[frame_idx]
        l_ang = 2 * np.arctan2(np.copy(l_quat[:, 3]), np.copy(l_quat[:, 0]))
        fbox = box.Box(Lx=l_box["Lx"], Ly=l_box["Ly"], is2D=True)
        side_length = max(fbox.Lx, fbox.Ly)
        l_min = -side_length / 2.0
        l_min *= 1.1
        l_max = -l_min

        # take local vertices and rotate, translate into
        # system coordinates
        patches = local_to_global(verts, l_pos[:, 0:2], l_ang)

        if linked_plot is not None:
            x_range = linked_plot.x_range
            y_range = linked_plot.y_range
        else:
            x_range = (l_min, l_max)
            y_range = (l_min, l_max)

        # plot
        p = figure(title=title, x_range=x_range, y_range=y_range, height=300, width=300)
        p.patches(
            xs=patches[:, :, 0].tolist(),
            ys=patches[:, :, 1].tolist(),
            fill_color=(42, 126, 187),
            line_color="black",
            line_width=1.5,
        )  # ,
        # legend="hexagons")
        # box display
        p.patches(
            xs=[[-fbox.Lx / 2, fbox.Lx / 2, fbox.Lx / 2, -fbox.Lx / 2]],
            ys=[[-fbox.Ly / 2, -fbox.Ly / 2, fbox.Ly / 2, fbox.Ly / 2]],
            fill_color=(0, 0, 0, 0),
            line_color="black",
            line_width=2,
        )
        # p.legend.location='bottom_center'
        # p.legend.orientation='horizontal'
        default_bokeh(p)
        # show(p)
        self.p = p
        return p

    def plot_single_neighbor(
        self,
        frame_idx,
        pidx,
        n_list,
        num_particles,
        title="Nearest Neighbor Visualization",
        linked_plot=None,
    ):

        l_box = self.box_data[frame_idx]
        l_pos = self.pos_data[frame_idx]
        l_quat = self.quat_data[frame_idx]
        l_ang = 2 * np.arctan2(np.copy(l_quat[:, 3]), np.copy(l_quat[:, 0]))
        fbox = box.Box(Lx=l_box["Lx"], Ly=l_box["Ly"], is2D=True)
        side_length = max(fbox.Lx, fbox.Ly)
        l_min = -side_length / 2.0
        l_min *= 1.1
        l_max = -l_min

        if linked_plot is not None:
            x_range = linked_plot.x_range
            y_range = linked_plot.y_range
        else:
            x_range = (l_min, l_max)
            y_range = (l_min, l_max)

        n_idxs = n_list[pidx]
        # clip padded values
        n_idxs = n_idxs[np.where(n_idxs < num_particles)]
        n_neigh = len(n_idxs)

        # get position, orientation for the central particle
        center_pos = np.zeros(shape=(1, 3), dtype=np.float32)
        center_ang = np.zeros(shape=(1), dtype=np.float32)
        center_pos[:] = l_pos[pidx]
        center_ang[:] = l_ang[pidx]

        # get the positions, orientations for the neighbor particles
        neigh_pos = np.zeros(shape=(n_neigh, 3), dtype=np.float32)
        neigh_ang = np.zeros(shape=(n_neigh), dtype=np.float32)
        neigh_pos[:] = l_pos[n_idxs]
        neigh_ang[:] = l_ang[n_idxs]

        # render in bokeh
        # create array of transformed positions
        # all particles
        patches = local_to_global(verts, l_pos[:, 0:2], l_ang)
        # center particle
        c_patches = local_to_global(verts, center_pos[:, 0:2], center_ang)
        # neighbor particles
        n_patches = local_to_global(verts, neigh_pos[:, 0:2], neigh_ang)
        # turn into list of colors
        # bokeh (as of this version) requires hex colors, so convert rgb to hex
        center_color = np.array([c_list[0] for _ in range(center_pos.shape[0])])
        neigh_color = np.array([c_list[1] for _ in range(neigh_pos.shape[0])])

        # plot
        p = figure(title=title, x_range=x_range, y_range=y_range, height=300, width=300)
        p.patches(
            xs=patches[:, :, 0].tolist(),
            ys=patches[:, :, 1].tolist(),
            fill_color=(0, 0, 0, 0.1),
            line_color="black",
        )
        p.patches(
            xs=n_patches[:, :, 0].tolist(),
            ys=n_patches[:, :, 1].tolist(),
            fill_color=neigh_color.tolist(),
            line_color="black",
            legend="neighbors",
        )
        p.patches(
            xs=c_patches[:, :, 0].tolist(),
            ys=c_patches[:, :, 1].tolist(),
            fill_color=center_color.tolist(),
            line_color="black",
            legend="centers",
        )
        # box display
        p.patches(
            xs=[[-fbox.Lx / 2, fbox.Lx / 2, fbox.Lx / 2, -fbox.Lx / 2]],
            ys=[[-fbox.Ly / 2, -fbox.Ly / 2, fbox.Ly / 2, fbox.Ly / 2]],
            fill_color=(0, 0, 0, 0),
            line_color="black",
            line_width=2,
        )
        p.legend.location = "bottom_center"
        p.legend.orientation = "horizontal"
        default_bokeh(p)
        self.p = p
        return p

    def plot_neighbors(
        self,
        frame_idx,
        n_list,
        num_particles,
        n_neigh,
        title="Nearest Neighbor Visualization",
        linked_plot=None,
    ):

        l_box = self.box_data[frame_idx]
        l_pos = self.pos_data[frame_idx]
        l_quat = self.quat_data[frame_idx]
        l_ang = 2 * np.arctan2(np.copy(l_quat[:, 3]), np.copy(l_quat[:, 0]))
        fbox = box.Box(Lx=l_box["Lx"], Ly=l_box["Ly"], is2D=True)
        side_length = max(fbox.Lx, fbox.Ly)
        l_min = -side_length / 2.0
        l_min *= 1.1
        l_max = -l_min

        if linked_plot is not None:
            x_range = linked_plot.x_range
            y_range = linked_plot.y_range
        else:
            x_range = (l_min, l_max)
            y_range = (l_min, l_max)

        # now for array manipulation magic
        # create an integer array of the same shape as the neighbor list array
        int_arr = np.ones(shape=n_list.shape, dtype=np.int32)
        # "search" for non-indexed particles (missing neighbors)
        # while it would be most accurate to use the UINTMAX value
        # provided by nn.getUINTMAX(), but this works just as well
        int_arr[n_list > (num_particles - 1)] = 0
        # sum along particle index axis to
        # determine the number of neighbors per particle
        n_neighbors = np.sum(int_arr, axis=1)
        # find the complement (if desired) to
        # find number of missing neighbors per particle
        # n_deficits = n_neigh - n_neighbors

        p = figure(title=title, x_range=x_range, y_range=y_range, height=300, width=300)
        for k in np.unique(n_neighbors):
            # find particles with k neighbors
            c_idxs = np.copy(np.where(n_neighbors == k)[0])
            center_pos = np.zeros(shape=(len(c_idxs), 3), dtype=np.float32)
            center_ang = np.zeros(shape=(len(c_idxs)), dtype=np.float32)
            center_pos = l_pos[c_idxs]
            center_ang = l_ang[c_idxs]
            c_patches = local_to_global(verts, center_pos[:, 0:2], center_ang)
            center_color = np.array([c_dict[k] for _ in range(center_pos.shape[0])])
            p.patches(
                xs=c_patches[:, :, 0].tolist(),
                ys=c_patches[:, :, 1].tolist(),
                fill_color=center_color.tolist(),
                line_color="black",
                legend=f"k={k}",
            )
        p.patches(
            xs=[[-fbox.Lx / 2, fbox.Lx / 2, fbox.Lx / 2, -fbox.Lx / 2]],
            ys=[[-fbox.Ly / 2, -fbox.Ly / 2, fbox.Ly / 2, fbox.Ly / 2]],
            fill_color=(0, 0, 0, 0),
            line_color="black",
            line_width=2,
        )
        p.legend.location = "bottom_center"
        p.legend.orientation = "horizontal"
        default_bokeh(p)
        self.p = p
        return p

    def plot_hexatic(
        self,
        frame_idx,
        psi_k,
        avg_psi_k,
        title="Hexatic Visualization",
        linked_plot=None,
    ):

        l_box = self.box_data[frame_idx]
        l_pos = self.pos_data[frame_idx]
        l_quat = self.quat_data[frame_idx]
        l_ang = 2 * np.arctan2(np.copy(l_quat[:, 3]), np.copy(l_quat[:, 0]))
        fbox = box.Box(Lx=l_box["Lx"], Ly=l_box["Ly"], is2D=True)
        side_length = max(fbox.Lx, fbox.Ly)
        l_min = -side_length / 2.0
        l_min *= 1.1
        l_max = -l_min

        if linked_plot is not None:
            x_range = linked_plot.x_range
            y_range = linked_plot.y_range
        else:
            x_range = (l_min, l_max)
            y_range = (l_min, l_max)

        # create array of transformed positions
        patches = local_to_global(verts, l_pos[:, 0:2], l_ang)
        # create an array of angles relative to the average
        a = np.angle(psi_k) - np.angle(avg_psi_k)
        # turn into an rgb array of tuples
        color = [tuple(cubeellipse(x)) for x in a]
        # bokeh (as of this version) requires hex colors, so convert rgb to hex
        hex_color = [
            f"#{clamp(r):02x}{clamp(g):02x}{clamp(b):02x}" for (r, g, b) in color
        ]
        # plot
        p = figure(title=title, x_range=x_range, y_range=y_range, height=300, width=300)
        p.patches(
            xs=patches[:, :, 0].tolist(),
            ys=patches[:, :, 1].tolist(),
            fill_color=hex_color,
            line_color="black",
        )
        default_bokeh(p)
        self.p = p
        return p

    def plot_orientation(
        self, frame_idx, title="Orientation Visualization", linked_plot=None
    ):

        l_box = self.box_data[frame_idx]
        l_pos = self.pos_data[frame_idx]
        l_quat = self.quat_data[frame_idx]
        l_ang = 2 * np.arctan2(np.copy(l_quat[:, 3]), np.copy(l_quat[:, 0]))
        fbox = box.Box(Lx=l_box["Lx"], Ly=l_box["Ly"], is2D=True)
        side_length = max(fbox.Lx, fbox.Ly)
        l_min = -side_length / 2.0
        l_min *= 1.1
        l_max = -l_min

        if linked_plot is not None:
            x_range = linked_plot.x_range
            y_range = linked_plot.y_range
        else:
            x_range = (l_min, l_max)
            y_range = (l_min, l_max)

        # create array of transformed positions
        patches = local_to_global(verts, l_pos[:, 0:2], l_ang)
        # turn into an rgb array of tuples
        theta = l_ang * 6.0
        color = [tuple(cubeellipse(x, lam=0.5, h=2.0)) for x in theta]
        # bokeh (as of this version) requires hex colors, so convert rgb to hex
        hex_color = [
            f"#{clamp(r):02x}{clamp(g):02x}{clamp(b):02x}" for (r, g, b) in color
        ]
        # plot
        p = figure(title=title, x_range=x_range, y_range=y_range, height=300, width=300)
        p.patches(
            xs=patches[:, :, 0].tolist(),
            ys=patches[:, :, 1].tolist(),
            fill_color=hex_color,
            line_color="black",
        )
        default_bokeh(p)
        self.p = p
        return p


def default_bokeh(p):
    """
    wrapper which takes the default bokeh outputs and changes them to more sensible values
    """
    p.title.text_font_size = "18pt"
    p.title.align = "center"

    p.xaxis.axis_label_text_font_size = "14pt"
    p.yaxis.axis_label_text_font_size = "14pt"

    p.xaxis.major_tick_in = 10
    p.xaxis.major_tick_out = 0
    p.xaxis.minor_tick_in = 5
    p.xaxis.minor_tick_out = 0

    p.yaxis.major_tick_in = 10
    p.yaxis.major_tick_out = 0
    p.yaxis.minor_tick_in = 5
    p.yaxis.minor_tick_out = 0

    p.xaxis.major_label_text_font_size = "12pt"
    p.yaxis.major_label_text_font_size = "12pt"


def cubeellipse(theta, lam=0.6, gamma=1.0, s=4.0, r=1.0, h=1.2):
    """Create an RGB colormap from an input angle theta. Takes lam (a list of
    intensity values, from 0 to 1), gamma (a nonlinear weighting power),
    s (starting angle), r (number of revolutions around the circle), and
    h (a hue factor)."""
    import numpy

    lam = lam**gamma

    a = h * lam * (1 - lam) * 0.5
    v = numpy.array(
        [[-0.14861, 1.78277], [-0.29227, -0.90649], [1.97294, 0.0]], dtype=numpy.float32
    )
    ctarray = numpy.array(
        [numpy.cos(theta * r + s), numpy.sin(theta * r + s)], dtype=numpy.float32
    )
    # convert to 255 rgb
    ctarray = (lam + a * v.dot(ctarray)).T
    ctarray *= 255
    ctarray = ctarray.astype(dtype=np.int32)
    return ctarray


def local_to_global(verts, positions, orientations):
    """
    Take a list of vertices, positions, and orientations and create
    a list of vertices in the "global coordinate system" for plotting
    in bokeh
    """
    num_particles = len(positions)
    num_verts = len(verts)
    # create list of vertices in the "local reference frame" i.e.
    # centered at (0,0)
    l_verts = np.zeros(shape=(num_particles, num_verts, 2), dtype=np.float32)
    l_verts[:] = verts
    # create array of rotation matrices
    rot_mat = np.zeros(shape=(num_particles, 2, 2), dtype=np.float32)
    for i, theta in enumerate(orientations):
        rot_mat[i] = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    # rotate; uses einsum for speed; please see numpy documentation
    # for more information
    r_verts = np.einsum("lij,lkj->lki", rot_mat, l_verts)
    # now translate to global coordinates
    # need to create a position array with same shape as vertex array
    l_pos = np.zeros(shape=(num_particles, num_verts, 2), dtype=np.float32)
    for i in range(num_particles):
        for j in range(len(verts)):
            l_pos[i, j] = positions[i]
    # translate
    output_array = np.add(r_verts, l_pos)
    return output_array


def clamp(x):
    """
    limit values between 0 and 255
    http://stackoverflow.com/questions/3380726/converting-a-rgb-color-tuple-to-a-six-digit-code-in-python
    """
    return max(0, min(x, 255))


def demo1(l_box, l_pos, l_ang, verts, title="System Visualization"):
    # create box
    fbox = box.Box(Lx=l_box["Lx"], Ly=l_box["Ly"], is2D=True)
    side_length = max(fbox.Lx, fbox.Ly)
    l_min = -side_length / 2.0
    l_min *= 1.1
    l_max = -l_min

    # take local vertices and rotate, translate into
    # system coordinates
    patches = local_to_global(verts, l_pos[:, 0:2], l_ang)

    # plot
    p = figure(
        title=title,
        x_range=(l_min, l_max),
        y_range=(l_min, l_max),
        height=300,
        width=300,
    )
    p.patches(
        xs=patches[:, :, 0].tolist(),
        ys=patches[:, :, 1].tolist(),
        fill_color=(42, 126, 187),
        line_color="black",
        line_width=1.5,
    )  # ,
    # legend="hexagons")
    # box display
    p.patches(
        xs=[[-fbox.Lx / 2, fbox.Lx / 2, fbox.Lx / 2, -fbox.Lx / 2]],
        ys=[[-fbox.Ly / 2, -fbox.Ly / 2, fbox.Ly / 2, fbox.Ly / 2]],
        fill_color=(0, 0, 0, 0),
        line_color="black",
        line_width=2,
    )
    # p.legend.location='bottom_center'
    # p.legend.orientation='horizontal'
    default_bokeh(p)
    # show(p)
    return p
