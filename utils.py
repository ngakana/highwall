from typing import Tuple

import numpy as np
from tqdm import tqdm
from loguru import logger


class VoxelGrid:
    """
    class to represent a voxel grid
    """
    def __init__(self, size=None):
        """
        instantiates a voxel grid object\n
        :param size: size of cubes in voxel grid
        """
        if size is None:
            size = 0.25
        self.index: np.array = None
        self.density: np.array = None
        self.size = size
        self.shape: Tuple = tuple()


class OBB:
    """class to represent point cloud oriented bounding box"""
    def __init__(self):
        self.box: np.array = None
        self.vertices: np.array = None
        self.centroid: np.array = None
        self.R: np.array = None
        self.min: np.array = None
        self.max: np.array = None

    def build2(self, covariance_matrix, points):
        """
        builds oriented bounding box from covariance matrix\n
        :param: covariance matrix: covariance matrix of the point cloud
        :param: points: xyz coordinates of the point cloud
        """
        obb = OBB()
        _, eigv_t = np.linalg.eigh(covariance_matrix)
        eigv = eigv_t.T

        def try_to_normalize(v):
            n = np.linalg.norm(v)
            if n < np.finfo(float).resolution:
                raise ZeroDivisionError
            return v / n

        r = try_to_normalize(eigv[:, 0])
        u = try_to_normalize(eigv[:, 1])
        f = try_to_normalize(eigv[:, 2])
        obb.R = np.linalg.inv(np.array((r, u, f)))
        points_prime = [obb.R @ xyz for xyz in tqdm(points, desc='Rotating points: ', unit=' points')]
        obb.min = np.min(points_prime, axis=0)
        obb.max = np.max(points_prime, axis=0)
        diff = np.array((obb.max - obb.min) / 2)
        obb.centroid = obb.min + diff
        dx, dy, dz = diff
        obb.box = np.array([
            # 0, 0, 0
            obb.centroid + [-dx, -dy, -dz],
            # 0, 0, 1
            obb.centroid + [-dx, -dy, dz],
            # 0, 1, 1
            obb.centroid + [-dx,  dy, dz],
            # 1, 1, 1
            obb.centroid + [dx, dy, dz],
            # 1, 1, 0
            obb.centroid + [dx, dy, -dz],
            # 0, 1, 0
            obb.centroid + [-dx, dy, -dz],
            # 0, 0, 0
            obb.centroid + [-dx, -dy, -dz],
            # 1, 0, 0
            obb.centroid + [dx, -dy, -dz],
            # 1, 0, 1
            obb.centroid + [dx, -dy, dz],
            # 0, 0, 1
            obb.centroid + [-dx, -dy, dz],
            # 0, 1, 1
            obb.centroid + [-dx, dy, dz],
            # 0, 1, 0
            obb.centroid + [-dx, dy, -dz],
            # 1, 1, 0
            obb.centroid + [dx, dy, -dz],
            # 1, 0, 0
            obb.centroid + [dx, -dy, -dz],
            # 1, 0, 1
            obb.centroid + [dx, -dy, dz],
            # 1, 1, 1
            obb.centroid + [dx, dy, dz],
        ])
        obb.vertices = np.array((
            # back, left, bottom
            obb.centroid + [-dx, -dy, -dz],
            # back, right, bottom
            obb.centroid + [-dx, dy, -dz],
            # front, right, bottom
            obb.centroid + [dx, dy, -dz],
            # front, left, bottom
            obb.centroid + [dx, -dy, -dz],
            # front, left, top
            obb.centroid + [dx, -dy, dz],
            # front, right, top
            obb.centroid + [dx, dy, dz],
            # back, right, top
            obb.centroid + [-dx, dy, dz],
            # back, left, top
            obb.centroid + [-dx, -dy, dz],
        ))

        self.centroid = eigv_t @ obb.centroid
        self.box = np.array([eigv_t @ corner for corner in obb.box])
        self.vertices = np.array([eigv_t @ vertex for vertex in obb.vertices])

    def build(self, points: np.ndarray):
        """
        builds oriented bounding box from a point cloud\n
        :param: points: xyz coordinates of the point cloud
        """
        # no need to store the covariance matrix
        self.build2(covariance_matrix=np.cov(points, y=None, rowvar=False, bias=True), points=points)


def encode_(idx_arr: np.array) -> np.array:
    """
    encodes voxel coordinate indices into compact sums stored in min memory address
    :param idx_arr: array of voxel indices
    :return:
    """
    logger.debug(f'Encoding  indices...')
    idx_max = np.max(idx_arr, axis=0)

    # get minimum num of required bits to represent max xyz values in binary
    req_bits = [int(num) for num in np.ceil(np.log2(idx_max))]

    # get required shifts to compute compact row-wise sums
    shifts = [a + b for a, b in zip(req_bits, req_bits[1:])]
    shifts.append(0)
    shifts = np.array(shifts, dtype=np.uint8)
    compact_idx = np.sum(np.left_shift(idx_arr, shifts), axis=1)
    logger.info(f'Finished encoding {len(compact_idx)} indices.')
    return compact_idx

