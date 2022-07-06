import math
import os.path as osp
import os

import numpy as np
import scipy.interpolate
import scipy.ndimage
import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ..ops import voxelization_idx


# perhaps the chunks are still too large and i need to downsample them by 1/4


class TreeDataset(Dataset):

    CLASSES = ('ground', 'tree')

    def __init__(self,
                 data_root,
                 voxel_cfg=None,
                 training=True,
                 logger=None):
        self.data_root = data_root
        self.voxel_cfg = voxel_cfg
        self.training = training
        self.logger = logger
        self.mode = 'train' if training else 'test'
        self.filenames = self.get_filenames()
        self.logger.info(f'Load {self.mode} dataset: {len(self.filenames)} scans')

    # GET LIST OF FILENAMES TO SAVE IN SELF.FILENAMES
    def get_filenames(self):
        filenames = os.listdir(self.data_root)
        return filenames

    # LOAD SPECIFIC FILE (TO USE IN GETITEM)
    def load(self, filename):
        return torch.load(osp.join(self.data_root, filename))

    # LEN FUNC
    def __len__(self):
        return len(self.filenames)

    # ELASTIC TRANSFORMATION OF EXAMPLE
    def elastic(self, x, gran, mag):
        blur0 = np.ones((3, 1, 1)).astype('float32') / 3
        blur1 = np.ones((1, 3, 1)).astype('float32') / 3
        blur2 = np.ones((1, 1, 3)).astype('float32') / 3

        bb = np.abs(x).max(0).astype(np.int32) // gran + 3
        noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        ax = [np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in bb]
        interp = [
            scipy.interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0)
            for n in noise
        ]

        def g(x_):
            return np.hstack([i(x_)[:, None] for i in interp])

        return x + g(x) * mag

    # GET INSTANCE RELATED INFO FOR ALL INSTANCES IN AN EXAMPLE (THIS INCLUDES OFFSET VECTOR, INSTANCE CLASS, INSTANCE NUM_POINTS)
    def getInstanceInfo(self, xyz, instance_label, semantic_label):
        pt_mean = np.ones((xyz.shape[0], 3), dtype=np.float32) * -100.0 # -100 IS CHOSEN AS THE MEAN OF THE OBJECT IF IT IS NO INSTANCE (BACKGROUND) THIS RESULTS IN NEGATIVE OFFSET VECTOR
        instance_pointnum = []
        instance_cls = []
        instance_num = int(instance_label.max()) + 1 # INSTANCE LABELS IN EXAMPLE ARE CODED AS -100, 1, 2, 3, 4, .., max (-100 are points not belonging to any instance)
        for i_ in range(instance_num): # ONLY COUNTS INSTANCES (NOT BACKGROUND)
            inst_idx_i = np.where(instance_label == i_)
            xyz_i = xyz[inst_idx_i]
            pt_mean[inst_idx_i] = xyz_i.mean(0) # CALCULATE INSTANCE MEAN
            instance_pointnum.append(inst_idx_i[0].size)
            cls_idx = inst_idx_i[0][0]
            instance_cls.append(semantic_label[cls_idx])
        pt_offset_label = pt_mean - xyz
        return instance_num, instance_pointnum, instance_cls, pt_offset_label

    # CLASSICAL DATA AUGMENTATIONS HERE IMPLEMENTED AS A SINGLE MATRIX MULTIPLICATION
    def dataAugment(self, xyz, jitter=False, flip=False, rot=False, prob=1.0):
        m = np.eye(3)
        if jitter and np.random.rand() < prob:
            m += np.random.randn(3, 3) * 0.1 # ADD NORMALLY DISTRIBUTED VALUES TO IDENTITY MATRIX IF RANDOM NUMBER IS SMALLER THAN PROB (ALWAYS IF PROB=1)
        if flip and np.random.rand() < prob:
            m[0][0] *= np.random.randint(0, 2) * 2 - 1 # RANDOM FLIP OF X COORDINATE (SAME CONDITION AS ABOVE)
        if rot and np.random.rand() < prob:
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], # RANDOM ROTATION (SAME CONDITION AS ABOVE)
                              [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])

        return np.matmul(xyz, m)


    # TRAINING TRANSFORMATIONS
    def transform_train(self, xyz, semantic_label, instance_label, aug_prob=1.0):
        # APPLY NORMAL DATA AUGMENTATIONS
        xyz_middle = self.dataAugment(xyz, True, True, True, aug_prob)
        # SCALE ORIGINAL VALUES WITH 50
        xyz = xyz_middle * self.voxel_cfg.scale
        # APPLY ELASTIC TRANSFORMATION
        # if np.random.rand() < aug_prob:
        #     xyz = self.elastic(xyz, 6 * self.voxel_cfg.scale // 50, 40 * self.voxel_cfg.scale / 50)
        #     xyz = self.elastic(xyz, 20 * self.voxel_cfg.scale // 50,
        #                        160 * self.voxel_cfg.scale / 50)
        # xyz_middle = xyz / self.voxel_cfg.scale

        # MAKE SCALED POINTS HAVE ALL POSITIVE VALUES
        xyz = xyz - xyz.min(0)

        return xyz, xyz_middle, semantic_label, instance_label

    # ALSO TRANSFORMATIONS BUT LESS THAN DURING TRAINING
    def transform_test(self, xyz, semantic_label, instance_label):
        xyz_middle = self.dataAugment(xyz, False, False, False)
        xyz = xyz_middle * self.voxel_cfg.scale
        xyz = xyz - xyz.min(0)
        return xyz, xyz_middle, semantic_label, instance_label

    def __getitem__(self, index):
        filename = self.filenames[index]
        data = self.load(filename)
        data = self.transform_train(*data) if self.training else self.transform_test(*data)
        if data is None:
            return None
        xyz, xyz_middle, semantic_label, instance_label = data # XYZ MIDDLE ARE THE POINTS ON WHICH STANDARD DATA TRANSFORMATIONS WERE APPLIED (but only select cropped indices); ON XYZ ALSO ELASTIC, CROPPING, SCALING AND SHIFTING ARE APPLIED SUCH THAT ALL VALUES ARE POSITIVE
        info = self.getInstanceInfo(xyz_middle, instance_label.astype(np.int32), semantic_label) # GET ALL INSTANCE RELATED INFO (SEE ABOVE)
        inst_num, inst_pointnum, inst_cls, pt_offset_label = info
        coord = torch.from_numpy(xyz).long()
        coord_float = torch.from_numpy(xyz_middle)

        semantic_label = torch.from_numpy(semantic_label)
        instance_label = torch.from_numpy(instance_label)
        pt_offset_label = torch.from_numpy(pt_offset_label)
        scan_id = [filename.replace(".pth", "")]

        return (scan_id, coord, coord_float, semantic_label, instance_label, inst_num,
                inst_pointnum, inst_cls, pt_offset_label)

        # coord: coordinates of fully augmented data (scaled and forced to non float values and elastically transformed compared to coord_float)
        # coord_float: basically contains the original values only with jitter random rotate and random flip
        # semantic_label: semantic label of augmented example (13 possibilities here i think)
        # instance_label: instance labels in the form -100, 1, 2, 3... of the example
        # instance_num: number of instances in example (background excluded)
        # inst_pointnum: list of point numbers for each instance in example (background is not included in this list)
        # ins_cls: list of semantic class for each instance (background excluded)
        # pt_offset_label: offset label for each point (three dimensional)

    def collate_fn(self, batch):
        scan_ids = []
        coords = []
        coords_float = []
        semantic_labels = []
        instance_labels = []

        instance_pointnum = []  # (total_nInst), int
        instance_cls = []  # (total_nInst), long
        pt_offset_labels = []

        total_inst_num = 0
        batch_id = 0
        for data in batch: # BATCH IS ITERABLE THAT CONTAINS OBJECTS RETURNED BY GETITEM FUNCTION

            if data is None:
                continue

            (scan_id, coord, coord_float, semantic_label, instance_label, inst_num,
             inst_pointnum, inst_cls, pt_offset_label) = data # GET RESULT FROM GETITEM AND SAVE AS TUPLE
            instance_label[np.where(instance_label != -100)] += total_inst_num # THIS RESULTS IN A CONSECUTIVE LABELING OF INSTANCES IN A BATCH. E.G. WHEN THERE ARE 30 INSTANCES IN THE WHOLE BATCH THEY WILL BE LABELED 1, 2, 3, 4, .., 30
            total_inst_num += inst_num # COUNT TOTAL INSTANCE NUMBER IN WHOLE BATCH
            coords.append(torch.cat([coord.new_full((coord.size(0), 1), batch_id), coord], 1)) # CONCATENATE COLUMN TO COORDS THAT INDICATES WHICH SAMPLE FROM BATCH A POINT BELONGS TO
            coords_float.append(coord_float)
            semantic_labels.append(semantic_label)
            instance_labels.append(instance_label)
            instance_pointnum.extend(inst_pointnum)
            instance_cls.extend(inst_cls)
            pt_offset_labels.append(pt_offset_label)
            scan_ids.extend(scan_id)
            batch_id += 1
        assert batch_id > 0, 'empty batch'
        if batch_id < len(batch):
            self.logger.info(f'batch is truncated from size {len(batch)} to {batch_id}')

        # MERGE ALL THE SCENES IN A BATCH
        coords = torch.cat(coords, 0)  # long (N, 1 + 3), the batch item idx is put in coords[:, 0]
        batch_idxs = coords[:, 0].int()
        coords_float = torch.cat(coords_float, 0).to(torch.float32)  # float (N, 3)
        semantic_labels = torch.cat(semantic_labels, 0).long()  # long (N)
        instance_labels = torch.cat(instance_labels, 0).long()  # long (N)
        instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int)  # int (total_nInst)
        instance_cls = torch.tensor(instance_cls, dtype=torch.long)  # long (total_nInst)
        pt_offset_labels = torch.cat(pt_offset_labels).float()

        # this is just for code compatibility, does not contain any information
        feats = torch.zeros(len(coords), 3)


        spatial_shape = np.clip(
            coords.max(0)[0][1:].numpy() + 1, self.voxel_cfg.spatial_shape[0], None)
        voxel_coords, v2p_map, p2v_map = voxelization_idx(coords, batch_id)
        return {
            'scan_ids': scan_ids,
            'coords': coords,
            'feats': feats,
            'batch_idxs': batch_idxs,
            'voxel_coords': voxel_coords,
            'p2v_map': p2v_map,
            'v2p_map': v2p_map,
            'coords_float': coords_float,
            'semantic_labels': semantic_labels,
            'instance_labels': instance_labels,
            'instance_pointnum': instance_pointnum,
            'instance_cls': instance_cls,
            'pt_offset_labels': pt_offset_labels,
            'spatial_shape': spatial_shape,
            'batch_size': batch_id,
        }

        # coords: TORCH TENSOR LENGTH OF TOTAL NUMBER OF POINTS IN BATCH. FIRST COLUMN INDICATES TO WHICH EXAMPLE IN THE BATCH A POINT BELONGS TO
        # batch_idxs: TORCH TENSOR OF LENGTH OF TOTAL NUMBER OF POINTS IN BATCH. SAME AS FIRST COLUMN OF COORDS
        # voxel_coords: VOXELIZED COORDINATES (TYPICALLY OF SMALLER LENGTH THAN COORDS OR COORDS_FLOAT)

        # p2v_map: UNFORTUNATELY I DONT KNOW WHAT EXACTLY THIS IS BUT I GUESS IT SOMEHOW CONNECTS THE VOXELS TO THE ORIGINAL POINTS
        # v2p_map: UNFORTUNATELY I DONT KNOW WHAT EXACTLY THIS IS BUT I GUESS IT SOMEHOW CONNECTS THE VOXELS TO THE ORIGINAL POINTS

        # coords_float: TORCH TENSOR OF LENGTH OF TOTAL NUMBER OF POINTS IN BATCH. NO ELASTIC TRANSFORMATIONS AND NO SCALING COMPARED TO COORDS
        # semantic_labels: TORCH TENSOR OF SEMANTIC LABELS OF LENGHT OF TOTAL NUMBER OF POINTS IN BATCH
        # instance_labels: TORCH TENSOR OF INSTANCE LABELS (LABELED CONSECUTIVELY OVER WHOLE BATCH) OF LENGTH OF TOTAL NUMBER OF POINTS IN BATCH
        # instance_pointnum: TORCH TENSOR OF LENGTH OF TOTAL NUMBER OF INSTANCES THAT CONTAINS NUMBER OF POINTS FOR EACH INSTANCE
        # instance_cls: TORCH TENSOR OF LENGTH OF TOTAL NUMBER OF INSTANCES THAT CONTAINS SEMANTIC CLASS FOR EACH INSTANCE
        # pt_offset_labels: TORCH TENSOR OF LENGTH OF TOTAL NUMBER OF POINTS IN BATCH WITH OFFSET VECTORS FOR EACH POINT
        # spatial_shape: 1D-ARRAY OF LENGTH 3 THAT GIVES EXTENT ALONG X Y AND Z DIMENSIONS OF ALL POINTS IN BATCH: THE RESPECTIVE VALUES ARE CLIPPED TO BE IN RANGE OF [128, infinity] BY DEFAULT
        # batch_size: BATCH SIZE


def build_dataloader(dataset, batch_size=1, num_workers=1, training=True, dist=False):
    shuffle = training
    sampler = DistributedSampler(dataset, shuffle=shuffle) if dist else None
    if sampler is not None:
        shuffle = False
    if training:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=dataset.collate_fn,
            shuffle=shuffle,
            sampler=sampler,
            drop_last=True,
            pin_memory=True)
    else:
        assert batch_size == 1
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=dataset.collate_fn,
            shuffle=False,
            sampler=sampler,
            drop_last=False,
            pin_memory=True)