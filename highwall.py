import numpy as np
import pandas as pd
import laspy
import pclpy
import open3d as o3d
from loguru import logger
from scipy.spatial import cKDTree
from typing import AnyStr

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import VoxelGrid, OBB


class HighwallPointCloud:
    """
    Class to represent a highwall point cloud
    """
    def __init__(self, src_file: AnyStr):
        """instantiates a highwall point cloud object from a LAS standard supported file"""
        try:
            logger.info(f'Creating a point cloud...')
            logger.debug(f'Reading file {src_file} ...')
            with laspy.file.File(src_file) as las:
                self.file = src_file
                self.las = las
                self.points: np.array = np.vstack((las.x, las.y, las.z)).T
                self.count = las.header.count
                self.dimensions = las.point_format
                logger.success(f'Point cloud created. {"{:,}".format(self.count)} points loaded.')
        except OSError() as e:
            logger.exception(e)
        self.origin: np.array = None
        self.voxelgrid = VoxelGrid()
        self.kdt: cKDTree
        self.octree = None
        self.bounding_box = OBB()

    def voxelize(self, size=None) -> None:
        """builds a voxel grid from a point cloud"""
        if size is None:
            size = 0.25
        else:
            self.voxelgrid.size = size
        logger.info('Building voxel grid...')
        self.origin = np.min(self.points, axis=0)
        voxels = np.array(np.round((self.points - self.origin) / size, 0), dtype=np.int32)
        self.voxelgrid.index, self.voxelgrid.density = np.unique(voxels, return_counts=True, axis=0)
        voxel_max = np.max(self.voxelgrid.index, axis=0)
        self.voxelgrid.shape = tuple(np.divide(voxel_max, size))
        logger.success(f'Voxel grid built. Voxel size = {self.voxelgrid.size}m')
        logger.info(f'{"{:.0f}".format(np.sum(self.voxelgrid.density)*100/self.count)}% of all points binned within'
                    f' {self.voxelgrid.shape[0]} x {self.voxelgrid.shape[1]} x {self.voxelgrid.shape[2]} voxels')

    def visualize(self):
        pc = pclpy.read(path=self.file, point_type="PointXYZ")
        writer = pclpy.pcl.io.PCDWriter()
        writer.writeBinary("PointCloud.pcd", pc)
        pcd = o3d.io.read_point_cloud("PointCloud.pcd")
        o3d.visualization.draw_geometries([pcd])

    def boxize(self):
        if self.points is None:
            self.points = self.las.points
        logger.info('  Building bounding box...')
        self.bounding_box.build(points=self.points)
        df = pd.DataFrame(self.bounding_box.vertices, columns=['X', 'Y', 'Z'])
        df['TYPE'] = 'vertex'
        df.append([self.bounding_box.centroid[0], self.bounding_box.centroid[1], self.bounding_box.centroid[2],
                   'centroid'])
        df.to_csv(r'data/Raw/boundingbox.csv')
        logger.success('  Finished building bounding box.')
        fig = plt.figure()
        ax: Axes3D = fig.add_subplot(121, projection='3d')
        ax.scatter3D(self.bounding_box.centroid[0], self.bounding_box.centroid[1], self.bounding_box.centroid[2])
        ax.plot3D(self.bounding_box.box[:, 0], self.bounding_box.box[:, 1], self.bounding_box.box[:, 2], '-')
        plt.show()


if __name__ == '__main__':
    in_filename = r'data/Raw/Jwaneng_Highwall_1-Point_Cloud/project_group1_densified_point_cloud.las'
    wall = HighwallPointCloud(src_file=in_filename)
    # wall.voxelize()
    # wall.visualize()
    wall.boxize()
