# GWR kernel function specifications

__author__ = "Taylor Oshan tayoshan@gmail.com"

import numpy as np
#adaptive specifications should be parameterized with nn-1 to match original gwr
#implementation. That is, pysal counts self neighbors with knn automatically.

#Soft dependency of numba's njit
try:
    from numba import njit
except ImportError:

    def njit(func):
        return func


@njit
def local_cdist(coords_i, coords, spherical):
    """
    Compute Haversine (spherical=True) or Euclidean (spherical=False) distance for a local kernel.
    """
    coords=np.array(coords)
    if spherical:
        dLat_1 = np.radians(coords[:, 1] - coords_i[1])
        dLon_1 = np.radians(coords[:, 0] - coords_i[0])
        lat1_1 = np.radians(coords[:, 1])
        lat2_1 = np.radians(coords_i[1])
        a_1 = np.sin(
            dLat_1 / 2)**2 + np.cos(lat1_1) * np.cos(lat2_1) * np.sin(dLon_1 / 2)**2
        c_1 = 2 * np.arcsin(np.sqrt(a_1))
        
        dLat_2 = np.radians(coords[:, 3] - coords_i[3])
        dLon_2 = np.radians(coords[:, 2] - coords_i[2])
        lat1_2 = np.radians(coords[:, 3])
        lat2_2 = np.radians(coords_i[3])
        a_2 = np.sin(
            dLat_2 / 2)**2 + np.cos(lat1_2) * np.cos(lat2_2) * np.sin(dLon_2 / 2)**2
        c_2 = 2 * np.arcsin(np.sqrt(a_2))
        R = 6371.0
        
        return R*(c_1+c_2)/1.60934
    else:
        return (np.sqrt(np.sum((coords_i[0:2] - coords[:, 0:2])**2, axis=1)) +\
    np.sqrt(np.sum((coords_i[2:] - coords[:, 2:])**2, axis=1)))/1609.34


class Kernel(object):
    """
    GWR kernel function specifications.
    
    """

    def __init__(self, i, data, bw=None, fixed=True, function='triangular',
                 eps=1.0000001, ids=None, points=None, spherical=False):

        if points is None:
            self.dvec = local_cdist(data[i], data, spherical).reshape(-1)
        else:
            self.dvec = local_cdist(points[i], data, spherical).reshape(-1)

        self.function = function.lower()

        if fixed:
            self.bandwidth = float(bw)
        else:
            self.bandwidth = np.partition(
                self.dvec,
                int(bw) - 1)[int(bw) - 1] * eps  #partial sort in O(n) Time

        self.kernel = self._kernel_funcs(self.dvec / self.bandwidth)

        if self.function == "bisquare":  #Truncate for bisquare
            self.kernel[(self.dvec >= self.bandwidth)] = 0

    def _kernel_funcs(self, zs):
        # functions follow Anselin and Rey (2010) table 5.4
        if self.function == 'triangular':
            return 1 - zs
        elif self.function == 'uniform':
            return np.ones(zi.shape) * 0.5
        elif self.function == 'quadratic':
            return (3. / 4) * (1 - zs**2)
        elif self.function == 'quartic':
            return (15. / 16) * (1 - zs**2)**2
        elif self.function == 'gaussian':
            return np.exp(-0.5 * (zs)**2)
        elif self.function == 'bisquare':
            return (1 - (zs)**2)**2
        elif self.function == 'exponential':
            return np.exp(-zs)
        else:
            print('Unsupported kernel function', self.function)
