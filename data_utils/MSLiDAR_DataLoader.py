import os
import os.path as osp
import numpy as np
import sys
import warnings
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
import pickle
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.dirname(BASE_DIR)
# sys.path.append(BASE_DIR)

from hilbertcurve.hilbertcurve import HilbertCurve
from sklearn.cluster import OPTICS
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

import sys
cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../"))
warnings.filterwarnings('ignore')

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def interleave_bits(x, y, z, depth):
    """Interleave the bits of x, y, and z to form a Morton code."""
    morton_code = 0
    for i in range(depth):
        morton_code |= ((x >> i) & 1) << (3 * i + 2)
        morton_code |= ((y >> i) & 1) << (3 * i + 1)
        morton_code |= ((z >> i) & 1) << (3 * i)
    return morton_code

def sort_points_by_zorder(points, depth):
    """Sort 3D points based on their Morton codes (Z-order curve)."""
    def get_morton_code(point):
        x, y, z = point
        return interleave_bits(x, y, z, depth)

    # Compute Morton codes and sort points
    morton_codes = [get_morton_code(point) for point in points]

    sorted_ids = np.argsort(morton_codes)
    trans_indices = sorted_ids[::-1]

    return sorted_ids, trans_indices

def fps_zorder_func(points, samplepoints_list, flat_):
    pad_width = points.shape[0]
    points = torch.Tensor(points).float().unsqueeze(0)
    if torch.cuda.is_available():
        points = points.cuda()
    fps_index_list = []
    series_idx_lists = []

    grid_size = 500
    depth = 5  # Depth of the octree

    for i in range(len(samplepoints_list)):
        series_idx_list = []
        S = samplepoints_list[i]
        xyz = points[:, :,:3]

        if flat_ == True:
            fps_index = torch.arange(pad_width).repeat(1, 1)

        else:
            fps_index=farthest_point_sample(xyz, S)

        points = index_points(points, fps_index)

        fps_index=fps_index.cpu().data.numpy()
        # padded_fps_index = np.pad(fps_index, ((0, 0), (0, pad_width - fps_index.shape[1])), mode='constant')
        fps_index_list.append(fps_index)

        raw_points = np.squeeze(points.cpu().data.numpy(), axis=0)[:,:3]
        min_coords = np.min(raw_points, axis=0)

        normalized_points = raw_points - min_coords

        grid_points = np.floor(normalized_points*grid_size).astype(int)

        sorted_indices, trans_indices = sort_points_by_zorder(grid_points, depth)

        sorted_indices = np.expand_dims(sorted_indices, axis=0)
        trans_indices = np.expand_dims(trans_indices, axis=0)

        sorted_indices = np.expand_dims(sorted_indices, axis=0)
        trans_indices = np.expand_dims(trans_indices, axis=0)
        series_idx_list.append(sorted_indices)
        series_idx_list.append(trans_indices)

        series_idx_array = np.concatenate(series_idx_list, axis=1) # 1 8 N (padding 0)_
        series_idx_lists.append(series_idx_array)


    series_idx_arrays = np.concatenate(series_idx_lists, axis=0) # 3 8 N 
    fps_index_array = np.vstack(fps_index_list) # 3 N (padding 0)_

    return fps_index_array, series_idx_arrays

def radial_sort(point_cloud, center):
    # 计算点到中心的距离
    distances = np.sqrt(np.sum((point_cloud - center) ** 2, axis=1))
    
    # 根据距离进行排序
    sorted_indices = np.argsort(distances)
    trans_indices = sorted_indices[::-1]

    return sorted_indices, trans_indices

def random_sort(point_cloud):
    sorted_indices = np.arange(point_cloud.shape[0])
    
    # 根据距离进行排序
    np.random.shuffle(sorted_indices)
    trans_indices = sorted_indices[::-1]

    return sorted_indices, trans_indices


def fps_radial_func(points, samplepoints_list, flat_):
    pad_width = points.shape[0]
    points = torch.Tensor(points).float().unsqueeze(0)
    if torch.cuda.is_available():
        points = points.cuda()
    fps_index_list = []
    series_idx_lists = []

    grid_size = 500


    for i in range(len(samplepoints_list)):
        series_idx_list = []
        S = samplepoints_list[i]
        xyz = points[:, :,:3]

        if flat_ == True:
            fps_index = torch.arange(pad_width).repeat(1, 1)

        else:
            fps_index=farthest_point_sample(xyz, S)

        points = index_points(points, fps_index)

        fps_index=fps_index.cpu().data.numpy()
        # padded_fps_index = np.pad(fps_index, ((0, 0), (0, pad_width - fps_index.shape[1])), mode='constant')
        fps_index_list.append(fps_index)


        raw_points = np.squeeze(points.cpu().data.numpy(), axis=0)[:,:3]
        center_coords = np.mean(raw_points, axis=0)

        sorted_indices, trans_indices = radial_sort(raw_points, center_coords)

        sorted_indices = np.expand_dims(sorted_indices, axis=0)
        trans_indices = np.expand_dims(trans_indices, axis=0)

        sorted_indices = np.expand_dims(sorted_indices, axis=0)
        trans_indices = np.expand_dims(trans_indices, axis=0)
        series_idx_list.append(sorted_indices)
        series_idx_list.append(trans_indices)

        series_idx_array = np.concatenate(series_idx_list, axis=1) # 1 8 N (padding 0)_
        series_idx_lists.append(series_idx_array)


    series_idx_arrays = np.concatenate(series_idx_lists, axis=0) # 3 8 N 
    fps_index_array = np.vstack(fps_index_list) # 3 N (padding 0)_

    return fps_index_array, series_idx_arrays
    

def fps_random_func(points, samplepoints_list, flat_):
    pad_width = points.shape[0]
    points = torch.Tensor(points).float().unsqueeze(0)
    if torch.cuda.is_available():
        points = points.cuda()
    fps_index_list = []
    series_idx_lists = []

    grid_size = 500


    for i in range(len(samplepoints_list)):
        series_idx_list = []
        S = samplepoints_list[i]
        xyz = points[:, :,:3]

        if flat_ == True:
            fps_index = torch.arange(pad_width).repeat(1, 1)

        else:
            fps_index=farthest_point_sample(xyz, S)

        points = index_points(points, fps_index)

        fps_index=fps_index.cpu().data.numpy()
        # padded_fps_index = np.pad(fps_index, ((0, 0), (0, pad_width - fps_index.shape[1])), mode='constant')
        fps_index_list.append(fps_index)


        raw_points = np.squeeze(points.cpu().data.numpy(), axis=0)[:,:3]
        center_coords = np.mean(raw_points, axis=0)

        sorted_indices, trans_indices = random_sort(raw_points)

        sorted_indices = np.expand_dims(sorted_indices, axis=0)
        trans_indices = np.expand_dims(trans_indices, axis=0)

        sorted_indices = np.expand_dims(sorted_indices, axis=0)
        trans_indices = np.expand_dims(trans_indices, axis=0)
        series_idx_list.append(sorted_indices)
        series_idx_list.append(trans_indices)

        series_idx_array = np.concatenate(series_idx_list, axis=1) # 1 8 N (padding 0)_
        series_idx_lists.append(series_idx_array)


    series_idx_arrays = np.concatenate(series_idx_lists, axis=0) # 3 8 N 
    fps_index_array = np.vstack(fps_index_list) # 3 N (padding 0)_

    return fps_index_array, series_idx_arrays

def PL_sort(point_cloud):
    # 计算点到中心的距离
    sorted_indices = np.lexsort((point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]))
    trans_indices = sorted_indices[::-1]

    return sorted_indices, trans_indices


def fps_pl_func(points, samplepoints_list, flat_):
    pad_width = points.shape[0]
    points = torch.Tensor(points).float().unsqueeze(0)
    if torch.cuda.is_available():
        points = points.cuda()
    fps_index_list = []
    series_idx_lists = []

    grid_size = 500


    for i in range(len(samplepoints_list)):
        series_idx_list = []
        S = samplepoints_list[i]
        xyz = points[:, :,:3]

        if flat_ == True:
            fps_index = torch.arange(pad_width).repeat(1, 1)

        else:
            fps_index=farthest_point_sample(xyz, S)

        points = index_points(points, fps_index)

        fps_index=fps_index.cpu().data.numpy()
        # padded_fps_index = np.pad(fps_index, ((0, 0), (0, pad_width - fps_index.shape[1])), mode='constant')
        fps_index_list.append(fps_index)


        raw_points = np.squeeze(points.cpu().data.numpy(), axis=0)[:,:3]
        sorted_indices, trans_indices = PL_sort(raw_points)


        sorted_indices = np.expand_dims(sorted_indices, axis=0)
        trans_indices = np.expand_dims(trans_indices, axis=0)

        sorted_indices = np.expand_dims(sorted_indices, axis=0)
        trans_indices = np.expand_dims(trans_indices, axis=0)
        series_idx_list.append(sorted_indices)
        series_idx_list.append(trans_indices)

        series_idx_array = np.concatenate(series_idx_list, axis=1) # 1 8 N (padding 0)_
        series_idx_lists.append(series_idx_array)


    series_idx_arrays = np.concatenate(series_idx_lists, axis=0) # 3 8 N 
    fps_index_array = np.vstack(fps_index_list) # 3 N (padding 0)_

    return fps_index_array, series_idx_arrays

def fps_optics_func(points, samplepoints_list, flat_):
    pad_width = points.shape[0]
    points = torch.Tensor(points).float().unsqueeze(0)
    if torch.cuda.is_available():
        points = points.cuda()
    fps_index_list = []
    series_idx_lists = []



    for i in range(len(samplepoints_list)):
        series_idx_list = []
        S = samplepoints_list[i]
        xyz = points[:, :,:3]

        if flat_ == True:
            fps_index = torch.arange(pad_width).repeat(1, 1)

        else:
            fps_index=farthest_point_sample(xyz, S)

        points = index_points(points, fps_index)

        fps_index=fps_index.cpu().data.numpy()
        # padded_fps_index = np.pad(fps_index, ((0, 0), (0, pad_width - fps_index.shape[1])), mode='constant')
        fps_index_list.append(fps_index)


        raw_points = np.squeeze(points.cpu().data.numpy(), axis=0)[:,:3]
        optics_model = OPTICS(min_samples=16, cluster_method='dbscan')
        optics_model.fit(raw_points)

        sorted_indices = optics_model.ordering_
        trans_indices = sorted_indices[::-1]

        sorted_indices = np.expand_dims(sorted_indices, axis=0)
        trans_indices = np.expand_dims(trans_indices, axis=0)

        sorted_indices = np.expand_dims(sorted_indices, axis=0)
        trans_indices = np.expand_dims(trans_indices, axis=0)
        series_idx_list.append(sorted_indices)
        series_idx_list.append(trans_indices)

        series_idx_array = np.concatenate(series_idx_list, axis=1) # 1 8 N (padding 0)_
        series_idx_lists.append(series_idx_array)


    series_idx_arrays = np.concatenate(series_idx_lists, axis=0) # 3 8 N 
    fps_index_array = np.vstack(fps_index_list) # 3 N (padding 0)_

    return fps_index_array, series_idx_arrays


def compute_curvature_pca(point_cloud, k=16):
    """
    使用PCA计算点云中的曲率。
    
    参数:
    - point_cloud: 形状为(N, 3)的numpy数组，表示点云数据。
    - k: 近邻点的数量，用于计算曲率。
    
    返回:
    - 每个点的曲率。
    """
    nbrs = NearestNeighbors(n_neighbors=k).fit(point_cloud)
    distances, indices = nbrs.kneighbors(point_cloud)
    
    curvatures = np.zeros(len(point_cloud))
    
    for i in range(len(point_cloud)):
        neighbors = point_cloud[indices[i, 1:]] 
        pca = PCA(n_components=3)
        pca.fit(neighbors)

        variance_ratios = pca.explained_variance_ratio_
        curvatures[i] = variance_ratios[2] / (variance_ratios[0] + variance_ratios[1] + variance_ratios[2])
    
    sorted_indices = np.argsort(curvatures)
    trans_indices = sorted_indices[::-1]
    return sorted_indices, trans_indices


def fps_curvature_func(points, samplepoints_list, flat_):
    pad_width = points.shape[0]

    points = torch.Tensor(points).float().unsqueeze(0)
    if torch.cuda.is_available():
        points = points.cuda()
    fps_index_list = []
    series_idx_lists = []


    for i in range(len(samplepoints_list)):
        series_idx_list = []
        S = samplepoints_list[i]
        xyz = points[:, :,:3]

        if flat_ == True:
            fps_index = torch.arange(pad_width).repeat(1, 1)

        else:
            fps_index=farthest_point_sample(xyz, S)

        points = index_points(points, fps_index)

        fps_index=fps_index.cpu().data.numpy()
        # padded_fps_index = np.pad(fps_index, ((0, 0), (0, pad_width - fps_index.shape[1])), mode='constant')
        fps_index_list.append(fps_index)


        raw_points = np.squeeze(points.cpu().data.numpy(), axis=0)[:,:3]
        sorted_indices, trans_indices = compute_curvature_pca(raw_points)

        sorted_indices = np.expand_dims(sorted_indices, axis=0)
        trans_indices = np.expand_dims(trans_indices, axis=0)

        sorted_indices = np.expand_dims(sorted_indices, axis=0)
        trans_indices = np.expand_dims(trans_indices, axis=0)
        series_idx_list.append(sorted_indices)
        series_idx_list.append(trans_indices)

        series_idx_array = np.concatenate(series_idx_list, axis=1) # 1 8 N (padding 0)_
        series_idx_lists.append(series_idx_array)


    series_idx_arrays = np.concatenate(series_idx_lists, axis=0) # 3 8 N 
    fps_index_array = np.vstack(fps_index_list) # 3 N (padding 0)_

    return fps_index_array, series_idx_arrays

def fps_hilbert_func(points, samplepoints_list, flat_):
    pad_width = points.shape[0]
    points = torch.Tensor(points).float().unsqueeze(0)
    if torch.cuda.is_available():
        points = points.cuda()
    fps_index_list = []
    series_idx_lists = []

    grid_size = 500
    p = 10  
    n = 3   
    hilbert_curve = HilbertCurve(p, n)

    for i in range(len(samplepoints_list)):
        series_idx_list = []
        S = samplepoints_list[i]
        xyz = points[:, :,:3]

        if flat_ == True:
            fps_index = torch.arange(pad_width).repeat(1, 1)

        else:
            fps_index=farthest_point_sample(xyz, S)

        points = index_points(points, fps_index)

        fps_index=fps_index.cpu().data.numpy()
        # padded_fps_index = np.pad(fps_index, ((0, 0), (0, pad_width - fps_index.shape[1])), mode='constant')
        fps_index_list.append(fps_index)


        raw_points = np.squeeze(points.cpu().data.numpy(), axis=0)[:,:3]
        min_coords = np.min(raw_points, axis=0)

        normalized_points = raw_points - min_coords

        grid_points = np.floor(normalized_points*grid_size).astype(int)
        hilbert_indices = hilbert_curve.distances_from_points(grid_points)

        hilbert_indices = np.array(hilbert_indices)

        sorted_indices = np.argsort(hilbert_indices)
        trans_indices = sorted_indices[::-1]

        sorted_indices = np.expand_dims(sorted_indices, axis=0)
        trans_indices = np.expand_dims(trans_indices, axis=0)

        sorted_indices = np.expand_dims(sorted_indices, axis=0)
        trans_indices = np.expand_dims(trans_indices, axis=0)
        series_idx_list.append(sorted_indices)
        series_idx_list.append(trans_indices)

        series_idx_array = np.concatenate(series_idx_list, axis=1) # 1 8 N (padding 0)_
        series_idx_lists.append(series_idx_array)


    series_idx_arrays = np.concatenate(series_idx_lists, axis=0) # 3 8 N 
    fps_index_array = np.vstack(fps_index_list) # 3 N (padding 0)_

    return fps_index_array, series_idx_arrays

def fps_series_func(points, voxel_indices, samplepoints_list, flat_):
    pad_width = points.shape[0]

    points = torch.Tensor(points).float().unsqueeze(0)
    voxel_indices = torch.Tensor(voxel_indices).float().unsqueeze(0)
    if torch.cuda.is_available():
        points = points.cuda()
        voxel_indices = voxel_indices.cuda()

    fps_index_list = []
    series_idx_lists = []

    x1y1z1 = [1, 1, 1]
    x0y1z1 = [-1, 1, 1]
    x1y0z1 = [1, -1, 1]
    x0y0z1 = [-1, -1, 1]
    x1y1z0 = [1, 1, -1]
    x0y1z0 = [-1, 1, -1]
    x1y0z0 = [1, -1, -1]
    x0y0z0 = [-1, -1, -1]

    series_list = []
    series_list.append(x1y1z1)
    #series_list.append(x0y1z1)
    #series_list.append(x1y0z1)
    #series_list.append(x0y0z1)
    #series_list.append(x1y1z0)
    #series_list.append(x0y1z0)
    #series_list.append(x1y0z0)
    series_list.append(x0y0z0)

    for i in range(len(samplepoints_list)):
        S = samplepoints_list[i]
        xyz = points[:, :,:3]

        if flat_ == True:
            fps_index = torch.arange(pad_width).repeat(1, 1)

        else:
            fps_index=farthest_point_sample(xyz, S)

        points = index_points(points, fps_index)
        new_voxel_indices = index_points(voxel_indices, fps_index).squeeze(0).cpu().data.numpy()
        voxel_indices = index_points(voxel_indices, fps_index)

        fps_index=fps_index.cpu().data.numpy()
        # padded_fps_index = np.pad(fps_index, ((0, 0), (0, pad_width - fps_index.shape[1])), mode='constant')
        fps_index_list.append(fps_index)
        


        series_idx_list = []
        for j in range(len(series_list)):
            
            series = series_list[j]
            new_voxel_indices_ForSeries = new_voxel_indices*series
            sorting_indices = np.expand_dims(np.lexsort((new_voxel_indices_ForSeries[:, 0], new_voxel_indices_ForSeries[:, 1], new_voxel_indices_ForSeries[:, 2])), axis=0)
            # print(sorting_indices.shape)
            sorting_indices = np.expand_dims(sorting_indices, axis=0)
            series_idx_list.append(sorting_indices)

        series_idx_array = np.concatenate(series_idx_list, axis=1) # 1 8 N (padding 0)_
        series_idx_lists.append(series_idx_array)

    series_idx_arrays = np.concatenate(series_idx_lists, axis=0) # 3 8 N 
    fps_index_array = np.vstack(fps_index_list) # 3 N (padding 0)_

    return fps_index_array, series_idx_arrays

def voxelization(points, voxel_size):
        """
        Perform voxelization on a given point cloud.
        
        Parameters:
        points (numpy.ndarray): Nx3 array of points (x, y, z).
        voxel_size (float): Size of the voxel grid.
        
        Returns:
        numpy.ndarray: Nx3 array of voxelized coordinates.
        """
        # Calculate the voxel indices
        voxel_indices = np.floor(points[:,:3] / voxel_size).astype(np.int32)

        coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]

        bounding_box = coord_max - coord_min

        voxel_total = np.ceil(bounding_box[0]*bounding_box[1]*bounding_box[2] / voxel_size**3).astype(np.int32) # 25*25*25
        voxel_valid = np.unique(voxel_indices, axis=0)

        
        return points, voxel_indices, voxel_total, voxel_valid

class RSDataset(Dataset):
    def __init__(self, root,  pre_root, args,  split='train'):
        super().__init__() 

        self.npoints = args.npoint
        self.root = root
        self.series_type = args.series_type
        self.fps_n_list = [args.spoints]
        self.flat = False

        
        self.save_path = os.path.join(pre_root, 'rs_%s_%dpts_%s_11.dat' % (split, self.npoints, self.series_type))
        
        if not os.path.exists(self.save_path):
            print('Processing data %s (only running in the first time)...' % self.save_path)

            rooms = sorted(os.listdir(root))
            rooms = [room for room in rooms if '_data_' in room] # 
            if split == 'train':
                rooms_split = [room for room in rooms if 'test_data_11' not in room]
            else:
                rooms_split = [room for room in rooms if 'test_data_11' in room]

            self.sample_points, self.sample_labels = [], []
            self.fps_index_array_list, self.series_idx_arrays_list = [], []
            voxel_size = 0.04

            for room_name in tqdm(rooms_split, total=len(rooms_split)):
                room_path = os.path.join(root, room_name)
                room_data = np.load(room_path)  # xyzrgbl, N,4096,7
                #print('room_data shape:', room_data.shape)
                points, labels = room_data[:,:, :-1], room_data[:, :, -1]  #N,4096,6; N,4096,1 
                
                ######### svd feature #################
                #extra_feature = room_data[:, :, 7:] # N 4096 1
                #points = np.concatenate((points, extra_feature), axis=-1)
                ###########################################          
                        
                for N_idx in tqdm(range(points.shape[0])):
                    self.sample_points.append(points[N_idx]), self.sample_labels.append(labels[N_idx]) #4096,6; 4096,1

                    pc = points[N_idx]


                    if args.series_type == '3dumamba_s2':
                        pc, voxel_indices, voxel_total, voxel_valid = voxelization(pc, voxel_size)
                        fps_index_array, series_idx_arrays = fps_series_func(pc, voxel_indices, self.fps_n_list, self.flat) # (3, N) 和 （3, 8, N）。3：三层降采样，前面的N是降采样序列，后面的N是排序序列。8是有8个方向的排序

                    elif args.series_type == 'Hilbert':
                        fps_index_array, series_idx_arrays = fps_hilbert_func(pc, self.fps_n_list, self.flat)
                    
                    elif args.series_type == 'Zorder':
                        fps_index_array, series_idx_arrays = fps_zorder_func(pc, self.fps_n_list, self.flat)
                    
                    elif args.series_type == 'Radial':
                        fps_index_array, series_idx_arrays = fps_radial_func(pc, self.fps_n_list, self.flat)

                    elif args.series_type == 'PL':
                        fps_index_array, series_idx_arrays = fps_pl_func(pc, self.fps_n_list, self.flat)
                    
                    elif args.series_type == 'Optics':
                        fps_index_array, series_idx_arrays = fps_optics_func(pc, self.fps_n_list, self.flat)
                    
                    elif args.series_type == 'Random':
                        fps_index_array, series_idx_arrays = fps_random_func(pc, self.fps_n_list, self.flat)

                    elif args.series_type == 'Curvature':
                        fps_index_array, series_idx_arrays = fps_curvature_func(pc, self.fps_n_list, self.flat)
                    
                    else:
                        raise ValueError("No corresponding serialization method !")  
                    
                    self.fps_index_array_list.append(fps_index_array), self.series_idx_arrays_list.append(series_idx_arrays)
            
            # self.room_idxs = np.array(room_idxs)
            print("Totally {} samples in {} set.".format(len(self.sample_points), split))
            with open(self.save_path, 'wb') as f:
                pickle.dump([self.sample_points, self.sample_labels, self.fps_index_array_list, self.series_idx_arrays_list], f)
        else:
            print('Load processed data from %s...' % self.save_path)
            with open(self.save_path, 'rb') as f:
                self.sample_points, self.sample_labels, self.fps_index_array_list, self.series_idx_arrays_list = pickle.load(f)

    
    def __getitem__(self, idx):
        
        points, labels, fps_index_array, series_idx_arrays = self.sample_points[idx], self.sample_labels[idx], self.fps_index_array_list[idx], self.series_idx_arrays_list[idx]
        return points, labels, fps_index_array, series_idx_arrays

    def __len__(self):
        return len(self.sample_points)




if __name__ == '__main__':
    import torch
    import argparse


    def parse_args(args_dict=None):
        '''PARAMETERS'''
        parser = argparse.ArgumentParser('training')
        parser.add_argument('--model', type=str, default='pointnet2_semseg_msg', help='model name [default: pointnet_sem_seg]')
        parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 16]')
        parser.add_argument('--epoch', default=150, type=int, help='Epoch to run [default: 32]')
        parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
        parser.add_argument('--gpu', type=str, default='2,3', help='GPU to use [default: GPU 0]')
        parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
        parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
        parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
        parser.add_argument('--npoint', type=int, default=4096, help='Point Number [default: 4096]')
        parser.add_argument('--step_size', type=int, default=5, help='Decay step for lr decay [default: every 10 epochs]')
        parser.add_argument('--lr_decay', type=float, default=0.8, help='Decay rate for lr decay [default: 0.7]')
        parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
        parser.add_argument('--num_category', type=int, default=6, help='num_category')
        parser.add_argument('--weighted_loss', type=bool, default=False, help='weighted loss')
        parser.add_argument('--series_type', type=str, default='3dumamba', help='serilization type of method')
        parser.add_argument('--flat', action='store_true', default=False, help='weighted loss')
        parser.add_argument('--spoints', type=int, default=512, help='num_category')

    
    
        if args_dict:
     
            args_list = []
            for key, value in args_dict.items():
                if isinstance(value, bool):
                    if value:
                        args_list.append(f'--{key}')
                else:
                    args_list.append(f'--{key}')
                    args_list.append(str(value))
            return parser.parse_args(args_list)
        else:
            return parser.parse_args()

    args = parse_args()
    data_path = 'data/'
    pre_data_path = 'pre_data/'

    param_combinations = [
   # {'series_type': '3dumamba_s2'},

    
    {'series_type': 'Random'},

    {'series_type': 'Hilbert'},


    {'series_type': 'Zorder'},


    {'series_type': 'Radial'},


    {'series_type': 'PL'},


    {'series_type': 'Optics'},


    {'series_type': 'Curvature'}

    ]
    

    for params in param_combinations:
        args = parse_args(params)  
        print(f"Running with parameters: {params}")
    
        train_dataset = RSDataset(root=data_path, pre_root = pre_data_path, args=args, split='train')
        test_dataset = RSDataset(root=data_path, pre_root = pre_data_path, args=args, split='test')