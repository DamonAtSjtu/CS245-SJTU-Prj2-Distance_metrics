"""
Custom distance metrics
"""
import numpy as np


def cosine_distance(x, y):

    return np.dot(x / np.linalg.norm(x), y / np.linalg.norm(y))



# TODO(Yifeng Gao): discuss the weights in EMD
def earth_mover_distance(x, y):

    return 1
