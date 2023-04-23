import os
import random
from abc import ABC, abstractmethod
from glob import glob

import cv2
import numpy as np
from skimage.io import imread

from utils.base_utils import compute_dR_dt, load_h5, read_pickle, np_skew_symmetric, epipolar_distance_mean,hpts_to_pts,pts_to_hpts

#path of sun3d and yfcc100m
data_root='data'

class PoseSequenceDataset(ABC):
    def __init__(self):
        self.cache_dir=''
        self.seq_name=''
        self.image_ids=[]
        self.pair_ids=[]

    @abstractmethod
    def get_image(self,img_id):
        return np.zeros([0,0,3],np.uint8)

    @abstractmethod
    def get_K(self,img_id):
        return np.zeros([3,3],np.float32)

    @abstractmethod
    def get_pose(self,id0,id1):
        return np.zeros([3,3],np.float32), np.zeros([3],np.float32)

    def cache_fn(self,suffix):
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
        return f'{self.cache_dir}/{self.seq_name}-{suffix}.h5'

    def get_F(self,id0,id1):
        R, t = self.get_pose(id0, id1)
        K0, K1 = self.get_K(id0), self.get_K(id1)
        F = np.linalg.inv(K1).T @ np_skew_symmetric(t) @ R @ np.linalg.inv(K0)
        return F

    def get_mask_gt(self,id0,id1,kps0,kps1,matches,thresh):
        K0,K1=self.get_K(id0),self.get_K(id1)
        R,t=self.get_pose(id0,id1)
        F = np.linalg.inv(K1).T @ np_skew_symmetric(t) @ R @ np.linalg.inv(K0)
        dist=epipolar_distance_mean(kps0[matches[:,0]],kps1[matches[:,1]],F)
        gt=dist<thresh
        return gt
    def get_mask_gt_v2(self,id0,id1,kps0,kps1,matches,thresh):
        def get_episym(x1, x2, dR, dt):
            num_pts = len(x1)
            # Make homogeneous coordinates
            x1 = np.concatenate([
                x1, np.ones((num_pts, 1))
            ], axis=-1).reshape(-1, 3, 1)
            x2 = np.concatenate([
                x2, np.ones((num_pts, 1))
            ], axis=-1).reshape(-1, 3, 1)

            # Compute Fundamental matrix
            dR = dR.reshape(1, 3, 3)
            # dt = dt.reshape(1, 3)
            F = np.repeat(np.matmul(
                np.reshape(np_skew_symmetric(dt), (-1, 3, 3)),
                dR
            ).reshape(-1, 3, 3), num_pts, axis=0)

            x2Fx1 = np.matmul(x2.transpose(0, 2, 1), np.matmul(F, x1)).flatten()
            Fx1 = np.matmul(F, x1).reshape(-1, 3)
            Ftx2 = np.matmul(F.transpose(0, 2, 1), x2).reshape(-1, 3)

            ys = x2Fx1**2 * (
                1.0 / (Fx1[..., 0]**2 + Fx1[..., 1]**2) +
                1.0 / (Ftx2[..., 0]**2 + Ftx2[..., 1]**2))

            return ys.flatten()
        K0,K1=self.get_K(id0),self.get_K(id1)
        R,t=self.get_pose(id0,id1)
        kps0=hpts_to_pts(pts_to_hpts(kps0) @ np.linalg.inv(K0).T)
        kps1=hpts_to_pts(pts_to_hpts(kps1) @ np.linalg.inv(K1).T)

        geod_d = get_episym(kps0[matches[:,0]], kps1[matches[:,1]], R, t)
        gt=geod_d<thresh
        return gt
class OANetSplitDataset(PoseSequenceDataset):
    def __init__(self, seq_name, vis_thresh=50, dataset_type='test', random_seed=1234, prefix="yfcc"):
        super().__init__()
        if prefix=='yfcc':
            self.cache_dir='data/yfcc_eval_cache'
        elif prefix=='sun3d':
            self.cache_dir='data/sun3d_eval_cache'
        else:
            raise NotImplementedError

        if prefix=='yfcc':
            self.seq_name=f'{seq_name}-{dataset_type}'
        elif prefix=='sun3d':
            self.seq_name=f'{seq_name}-{dataset_type}'
        else:
            raise NotImplementedError

        self.vis_thresh=vis_thresh
        self.random_seed=random_seed
        self.dataset_type=dataset_type
        self.seq_name=seq_name

        if prefix=='yfcc':
            seq_dir=os.path.join(data_root,'yfcc100m',seq_name,dataset_type)
        elif prefix=='sun3d':
            seq_dir=os.path.join(data_root,f'sun3d_{dataset_type}',seq_name,dataset_type)
        else:
            raise NotImplementedError


        img_list_file = os.path.join(seq_dir, "images.txt")
        geo_list_file = os.path.join(seq_dir, "calibration.txt")
        vis_list_file = os.path.join(seq_dir, "visibility.txt")

        self.img_pths=[os.path.join(seq_dir,pth) for pth in np.loadtxt(img_list_file,dtype=str)]
        self.geo_list=[os.path.join(seq_dir,pth) for pth in np.loadtxt(geo_list_file,dtype=str)]
        self.vis_list=[os.path.join(seq_dir,pth) for pth in np.loadtxt(vis_list_file,dtype=str)]

        self.pair_ids=read_pickle(os.path.join(data_root,'pairs',f'{self.seq_name}-te-1000-pairs.pkl'))
        self.pair_ids=[(str(pair[0]),str(pair[1])) for pair in self.pair_ids]

        unique_ids = set()
        for pair in self.pair_ids:
            unique_ids.add(pair[0])
            unique_ids.add(pair[1])
        self.image_ids = list(unique_ids)

    @staticmethod
    def rectify_K(geom):
        img_size, K = geom['imsize'][0], geom['K']
        if (type(img_size)==tuple or type(img_size)==list or type(img_size)==np.ndarray) and len(img_size)==2:
            w, h = img_size[0], img_size[1]
        else:
            h=w=img_size

        cx = (w - 1.0) * 0.5
        cy = (h - 1.0) * 0.5
        K[0, 2] += cx
        K[1, 2] += cy
        return K

    def get_image(self,img_id,grey_model=False):
        if grey_model:
            img=cv2.imread(self.img_pths[int(img_id)],cv2.IMREAD_GRAYSCALE)
        else:
            img=imread(self.img_pths[int(img_id)])
        return img

    def get_K(self,img_id):
        geo = load_h5(self.geo_list[int(img_id)])
        K=self.rectify_K(geo)
        return K

    def get_pose_single(self, img_id):
        geo = load_h5(self.geo_list[int(img_id)])
        R, t = geo['R'], geo['T'][0]
        return R, t

    def get_pose(self,id0,id1):
        geo0 = load_h5(self.geo_list[int(id0)])
        geo1 = load_h5(self.geo_list[int(id1)])
        R0, t0 = geo0['R'], geo0['T'][0]
        R1, t1 = geo1['R'], geo1['T'][0]
        dR, dt = compute_dR_dt(R0,t0,R1,t1)
        return dR, dt




def name2datalist(name):
    dataset_list=[]
    if name=='yfcc':
        seq_names = ['buckingham_palace', 'notre_dame_front_facade', 'reichstag', 'sacre_coeur']
        for seq_name in seq_names:
            dataset_list.append(OANetSplitDataset(seq_name, prefix='yfcc'))
        return dataset_list
    elif name=='sun3d':
        seq_names = ['te-brown1', 'te-brown2', 'te-brown3', 'te-brown4', 'te-brown5', 'te-hotel1', 'te-harvard1',
                     'te-harvard2', 'te-harvard3', 'te-harvard4','te-mit1', 'te-mit2', 'te-mit3', 'te-mit4', 'te-mit5']
        for seq_name in seq_names:
            dataset_list.append(OANetSplitDataset(seq_name, prefix='sun3d'))
        return dataset_list
    else:
        raise NotImplementedError
