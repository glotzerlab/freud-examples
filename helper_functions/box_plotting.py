import numpy as np

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