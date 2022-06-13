import numpy as np
import open3d as o3d
import os
import argparse
import yaml
import itertools
from utils.config import read_yaml

VAL = 1000 # value that is necessary for voxelization. It is used to subset pointcloud. Since we do not want the pointcloud to be subset, it is larger than any coordinate value
"""
Takes a plot of forest [plot_id].npy and generates samples from it
"""

class SampleGenerator:
    def __init__(self, config):
        self.points = np.load(config.plot_path)
        self.label = None
        if self.points.shape[1] == 4:
            self.label = self.points[:, 3]
            self.points = self.points[:, :3]
        self.config = config
        self.x_range, self.y_range = get_ranges(self.points)
        self.occupancy = None

    def regular_generate(self):
        config = self.config
        # Preprocessing
        if config.remove_background:
            filter = self.label != 999
            self.points, self.label = self.points[filter], self.label[filter]
        if config.voxelize: 
            self.points, self.label = voxelize(self.points, self.label, config)
        
        # Definition of rectangles 
        x_range, y_range = self.x_range, self.y_range
        x_steps = np.arange(x_range[0], x_range[1], step=config.inner_edge)
        y_steps = np.arange(y_range[0], y_range[1], step=config.inner_edge)
        x_pairs = consecutive_elements(x_steps)
        y_pairs = consecutive_elements(y_steps)
        final_rectangles = combine(x_pairs, y_pairs)

        # add outer area
        final_rectangles[:, [0, 2]] = final_rectangles[:, [0, 2]] - config.outer_edge 
        final_rectangles[:, [1, 3]] = final_rectangles[:, [1, 3]] + config.outer_edge 

        # save as class element
        self.rectangles = final_rectangles

    def occupancy_grid(self):
        """
        occupancy grid mxnx3
        0 x mean
        1 y mean
        2 occupied or not
        """
        res = self.config.occupancy_resolution
        n = self.config.number_occupancy_points
        x_min, x_max = self.x_range[0], self.x_range[1]
        (x_res, x_dim), (y_res, y_dim) = adjust_res(self.x_range, res), adjust_res(self.y_range, res)
        y_steps = np.arange(self.y_range[0], self.y_range[1]+1e-5, step=y_res)
        x_steps = np.arange(self.x_range[0], self.x_range[1]+1e-5, step=x_res)
        grid = np.ones((x_dim, y_dim, 3))
        idx = np.random.randint(0, len(self.points), size=n)
        points = self.points[idx]
        x_coord, y_coord = points[:,0], points[:,1]

        for i in range(x_dim):
            for j in range(y_dim):
                point_exist = np.any((x_coord > x_steps[i]) & (x_coord <= x_steps[i + 1]) & (y_coord > y_steps[j]) & (y_coord <= y_steps[j + 1]))
                grid[i,j, 2] = point_exist
                grid[i,j, 0:2] = [np.mean(x_steps[i:i+2]), np.mean(y_steps[j:j+2])]
        grid = fill_holes(grid, how_far=self.config.fill_holes_how_far)
        return grid
    
    def random_generate(self, n_samples):
        assert isinstance(self.occupancy, np.ndarray)
        width = self.config.inner_edge + self.config.outer_edge

        # generate candidates
        x_low = np.random.uniform(self.x_range[0], self.x_range[1] - width, size=n_samples)
        y_low = np.random.uniform(self.y_range[0], self.y_range[1] - width, size=n_samples)
        rotations = np.random.uniform(0, 2*np.pi, size=n_samples)

        # generate vertices for candidates
        vertices_unrotated, centers = get_vertices(x_low, y_low, width)
        vertices = rotate_2d(vertices_unrotated, rotations, centers)

        # check occupancy in candidates and discard unoccupied regions
        ranges_x, ranges_y = get_ranges(vertices)
        # first step: check without rotations
        boolean = self.check_occupancy(ranges_x, ranges_y, vertices_unrotated, rotations, centers)
        
        return vertices, boolean, rotations, centers, vertices_unrotated
    
    def check_occupancy(self, ranges_x, ranges_y, vertices_unrotated, rotations, centers):
        min_occupied = self.config.min_occupied
        grid = self.occupancy  
        grid = grid.reshape(-1, grid.shape[-1])
        # We first select a rectangular area A from the occupancy grid
        As = generate_views(grid, ranges_x, ranges_y)
        # We apply the inverse rotation to A. receiving A_inv
        A_invs = [rotate_2d(A[:, :2], rotation[np.newaxis], centers[np.newaxis, :], inverse=True)[0] for A, rotation, centers in zip(As, rotations, centers)]
        A_invs = [np.hstack((A_inv, A[:,2][:, np.newaxis])) for A_inv, A in zip(A_invs, As)]
        # We subset A_inv by the unrotated vertices 
        xr_unrtd, yr_unrtd = get_ranges(vertices_unrotated)
        A_subset = [generate_views(A_inv, xr[np.newaxis,:], yr[np.newaxis,:]) for A_inv, xr, yr in zip(A_invs, xr_unrtd, yr_unrtd)] # [nx3]
        A_subset = list(itertools.chain.from_iterable(A_subset)) # unlist list of lists
        # we calculate statistics for the occupancy in the subsets. 
        occ = [(occ[:, 2].sum(), len(occ)) for occ in A_subset]
        occ = np.array(occ)
        measure = occ[:, 0] / occ[:,1]
        print(measure)
        filter = measure > min_occupied

        return filter

    def save(self, vertices, rotations, centers, vertices_unrotated):
        path = self.config.sample_path
        normalize = self.config.normalize
        pts = self.points
        if self.config.save_labels:
            pts = np.hstack((pts, self.label[:, np.newaxis]))
        # We first select a rectangular area A from the point cloud, that is larger than the rotated rectangle
        ranges_x, ranges_y = get_ranges(vertices)
        As = generate_views(pts, ranges_x, ranges_y)
        # We apply the inverse rotation to A. receiving A_inv
        A_invs = [rotate_2d(A[:, :2], rotation[np.newaxis], centers[np.newaxis, :], inverse=True)[0] for A, rotation, centers in zip(As, rotations, centers)]
        A_invs = [np.hstack((A_inv, A[:,2:])) for A_inv, A in zip(A_invs, As)]
        # We subset A_inv by the unrotated vertices to receive the points
        xr_unrtd, yr_unrtd = get_ranges(vertices_unrotated)
        A_subset = [generate_views(A_inv, xr[np.newaxis,:], yr[np.newaxis,:]) for A_inv, xr, yr in zip(A_invs, xr_unrtd, yr_unrtd)] # [nx3/4]
        A_subset = list(itertools.chain.from_iterable(A_subset)) # unlist list of lists
        # We rerotate A_subset into the normal space, reatttach the z axis and save the files
        A_xy = [rotate_2d(A[:, :2], rotation[np.newaxis], centers[np.newaxis, :], inverse=False)[0] for A, rotation, centers in zip(A_subset, rotations, centers)]
        A_subset = [np.hstack((a1, a2[:,2:])) for a1, a2 in zip(A_xy, A_subset)]

        paths = []
        for subset, center, rotation in zip(A_subset, centers, rotations):
            name = "".join([str(item) for item in np.round(center,2)])
            whole_path = os.path.join(path, "-c" + name + "-r" + str(np.round(rotation, 2)) + ".npy")
            if normalize:
                subset[:,:2] = subset[:,:2] - center
            np.save(whole_path, subset)
            paths += [whole_path]
        return paths

def generate_views(arr, ranges_x, ranges_y):
    """
    takes nx4, nx3 or nx2 array and filter by the first two dimensions
    """
    filters = (arr[:, 0][:, np.newaxis] > ranges_x[:,0][np.newaxis]) & \
                (arr[:, 0][:, np.newaxis] < ranges_x[:,1][np.newaxis]) & \
                (arr[:, 1][:, np.newaxis] > ranges_y[:,0][np.newaxis]) & \
                (arr[:, 1][:, np.newaxis] < ranges_y[:,1][np.newaxis]) # p x dim_ranges

    views = [arr[filter] for filter in filters.T]
    return views

def get_vertices(x, y, width):
    # vertices: nx4x2 np array
    # centers: nx2 np array
    edges = np.empty((len(x), 2,  2))
    edges[:,0,0] = x
    edges[:,0,1] = x + width
    edges[:,1,0] = y
    edges[:,1,1] = y + width
    vertices = np.empty((len(x), 4,  2))
    for i, edge in enumerate(edges):
        vertices[i, :, :] = cartesian(edge)
    centers = np.array([x + width / 2, y + width / 2])
    return vertices, centers.T


def adjust_res(range, res):
    dif = np.abs(range[0] - range[1])
    times_fit = np.floor(dif / res)
    adj_res = dif / times_fit
    print(f"Adjusted resolution from {res} to {adj_res}")
    return adj_res, times_fit.astype("int") 


def combine(a1, a2):
    """ from https://stackoverflow.com/questions/47143712/combination-of-all-rows-in-two-numpy-arrays
    Produces combinations of all rows in two numpy arrays. 
    In: m1xn1, m2xn2
    Out: cx(n1+n2)
    """
    m1,n1 = a1.shape
    m2,n2 = a2.shape
    out = np.zeros((m1,m2,n1+n2),dtype=int)
    out[:,:,:n1] = a1[:,None,:]
    out[:,:,n1:] = a2
    out.shape = (m1*m2,-1)
    return out


def consecutive_elements(steps: np.ndarray):
    """
    given a 1 dimensional np array steps nx1 returns a 2dim array (n-1)x2 with every row the consecutive elements in steps
    """
    elements = np.empty((len(steps)-1, 2))
    for i in range(len(steps)-1):
        elements[i] = [steps[i], steps[i+1]]
    return elements


def get_ranges(points):
    """
    points: either nx2/3 or bxnx2/3
    """
    x = points[...,0]
    y = points[...,1]
    ax = None if len(points.shape) == 2 else 1
    rng = np.array([x.min(axis=ax), x.max(axis=ax)]), np.array([y.min(axis=ax), y.max(axis=ax)])
    if len(points.shape) == 3:
        rng = (rng[0].T, rng[1].T)
    return rng


def voxelize(points, label, config):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    print(f"Voxelize the point cloud with a voxel size of {config.voxel_size}")
    min_bound, max_bound = np.array([-VAL, -VAL, -VAL]), np.array([VAL, VAL, VAL])
    downpcd, _, idx = pcd.voxel_down_sample_and_trace(config.voxel_size, min_bound, max_bound)
    print(f"Previous size: {points.shape[0]}, new size {np.asarray(downpcd.points).shape[0]}")
    idx = [item[0] for item in idx]
    label = label[idx]

    return np.asarray(downpcd.points), label


def read_yaml(path):
    with open(path, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    config = argparse.ArgumentParser()
    for k, v in parsed_yaml.items():
        config.add_argument('--' + k, default=v)
    return config.parse_args("")


def fill_holes(grid, how_far):
    min_neighbours =  ((how_far*2)**2 / 2)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not grid[i, j, 2]:
                enclosed = np.sum(grid[i-how_far:i+how_far, j-how_far:j+how_far, 2]) 
                grid[i, j, 2] = enclosed > min_neighbours
                
    return grid


def rotate_2d(points, rotations, centers=None, inverse=False):
    """
    nx2
    m
    centers as offset to rotate inplace 
    out: mxnx2
    """
    if inverse:
        rotations = -1 * rotations
    if np.any(centers):
        centers = centers[:, np.newaxis, :]
        points = points - centers
    R = np.empty((len(rotations), 2, 2))
    R[:,0,0] = np.cos(rotations)
    R[:,0,1] = -1 * np.sin(rotations)
    R[:,1,0] = -1 * R[:, 0, 1]
    R[:,1,1] = R[:,0,0]
    points = points @ R 
    if np.any(centers):
        points = points + centers
    return points
   

def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    #m = n / arrays[0].size
    m = int(n / arrays[0].size) 
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
        #for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out




if __name__ == '__main__':
    import sys
    config = read_yaml("datasets/test_generator.yaml")
    print(config)
    if not os.path.exists(config.sample_path):
        os.mkdir(config.sample_path)
    obj = SampleGenerator(config)
    obj.regular_generate()
    obj.occupancy = obj.occupancy_grid()
    vertices, boolean, rotations, centers, vertices_unrotated = obj.random_generate(8)
    import matplotlib.pyplot as plt
    plt.imshow(obj.occupancy[:,:, 2], cmap='hot', interpolation='nearest')
    plt.show()
    obj.save(vertices[boolean], rotations[boolean], centers[boolean], vertices_unrotated[boolean])
