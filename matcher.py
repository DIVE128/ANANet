import torch
from pyflann import FLANN
import numpy as np

from utils.extend_utils.extend_utils_fn import \
    find_nearest_point_idx, find_first_and_second_nearest_point


class Matcher:
    def __init__(self, cfg):
        self.mutual_best=cfg['mutual_best']
        self.ratio_test=cfg['ratio_test']
        self.ratio=cfg['ratio']
        self.use_cuda=cfg['cuda']
        self.flann=FLANN()
        if self.use_cuda:
            self.match_fn_1=lambda desc0,desc1: find_nearest_point_idx(desc1, desc0)
            self.match_fn_2=lambda desc0,desc1: find_first_and_second_nearest_point(desc1, desc0)
        else:
            self.match_fn_1=lambda desc0,desc1: self.flann.nn(desc1, desc0, 1, algorithm='linear')
            self.match_fn_2=lambda desc0,desc1: self.flann.nn(desc1, desc0, 2, algorithm='linear')

    def match(self,desc0,desc1,*args,**kwargs):
        mask=np.ones(desc0.shape[0],dtype=np.bool)
        if self.ratio_test:
            idxs,dists = self.match_fn_2(desc0,desc1)

            dists=np.sqrt(dists) # note the distance is squared
            ratio_mask=dists[:,0]/dists[:,1]<self.ratio
            mask&=ratio_mask
            idxs=idxs[:,0]
        else:
            idxs,_=self.match_fn_1(desc0,desc1)

        if self.mutual_best:
            idxs_mutual,_=self.match_fn_1(desc1,desc0)
            mutual_mask = np.arange(desc0.shape[0]) == idxs_mutual[idxs]
            mask&=mutual_mask

        matches=np.concatenate([np.arange(desc0.shape[0])[:,None],idxs[:,None]],axis=1)
        matches=matches[mask]

        return matches


name2matcher={
    'default': Matcher,
}