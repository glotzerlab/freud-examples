import numpy as np

def make_polygon(sides, radius=1):
    thetas = np.linspace(0, 2*np.pi, sides+1)[:sides]
    vertices = np.array([[radius*np.sin(theta), radius*np.cos(theta)]
                         for theta in thetas])
    return vertices
