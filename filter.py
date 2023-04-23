import numpy as np
import torch
from network.ANANet import ANANet
from utils.base_utils import pts_to_hpts, hpts_to_pts,load_cfg



class ANANetFilter:
    def __init__(self,cfg):
        
        path = cfg['weights_path']
        
        weight=torch.load(path)
        self.network=ANANet(**load_cfg(cfg['model_build_file_path']))  
        self.network.load_state_dict(weight['state_dict'])
        self.network.eval()
        print("Loaded ananet model")

        self.device=torch.device(cfg['device'])
        self.network.to(self.device)
    def prepare_call(self,kps0, kps1, matches_info, img0, img1, name, K0, K1, desc0, desc1, **kwargs):
        matches=matches_info[:,:2].astype(np.int32)
        x0=hpts_to_pts(pts_to_hpts(kps0[matches[:,0]][:,:2]) @ np.linalg.inv(K0).T)
        x1=hpts_to_pts(pts_to_hpts(kps1[matches[:,1]][:,:2]) @ np.linalg.inv(K1).T)
        xs=np.concatenate([x0,x1],1)
        return xs
    def prepare_input(self,xs):
        # all_name "dataset_name-seq_name-id0-id1-det_name-desc_name-match_name"
        xs=torch.from_numpy(xs.astype(np.float32)).unsqueeze(0).cuda()
        return xs
    def return_intermediate(self, kps0, kps1, matches_info, img0, img1, name, K0, K1, desc0, desc1,return_intermediate):
        xs= self.prepare_call(kps0, kps1, matches_info, img0, img1, name, K0, K1, desc0, desc1)
        xs=self.prepare_input(xs)
        with torch.no_grad():
            out,w,slf_attn_list,c_desc_list=self.network(xs,return_intermediate)
        return out,w,slf_attn_list,c_desc_list
    def __call__(self, kps0, kps1, matches_info, img0, img1, name, K0, K1, desc0, desc1, **kwargs):
        xs= self.prepare_call(kps0, kps1, matches_info, img0, img1, name, K0, K1, desc0, desc1, **kwargs)
        xs=self.prepare_input(xs)
        with torch.no_grad():
            _,w=self.network(xs)
        
        w=w.detach().cpu().squeeze().numpy()
        return w>0,w

class NoneFilter:
    def __init__(self,config):
        pass
    def __call__(self, kps0, kps1, matches_info, img0, img1, name, K0, K1, desc0, desc1, **kwargs):
        res_logits=np.ones(kps0.shape[0])
        return res_logits>0,res_logits


name2filter={
    'ananet':ANANetFilter,
    'none':NoneFilter,
}