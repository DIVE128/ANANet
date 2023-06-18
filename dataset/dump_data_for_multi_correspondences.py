import numpy as np 
import os 
from scipy.spatial import KDTree
import cv2
import torch
import pickle
def split_str(lines):
    returned_list=[]
    for line in lines:
        line=line.replace("\n","")
        items=line.split(" ")
        returned_list+=[[ float(item)for item in items]]
    return returned_list
def make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0,
                            mkpts1, color=None, text=None, path=None,
                            show_keypoints=False, margin=10,
                            opencv_display=False, opencv_title='',
                            small_text=[],axis=1):
    if axis==1:
        if len(image0.shape)==2:
            H0, W0 = image0.shape
            H1, W1 = image1.shape
            H, W = max(H0, H1), W0 + W1 + margin

            out = 255*np.ones((H, W), np.uint8)
            out[:H0, :W0] = image0
            out[:H1, W0+margin:] = image1
            out = np.stack([out]*3, -1)
        elif len(image0.shape)==3:
            H0, W0,_ = image0.shape
            H1, W1,_= image1.shape
            H, W = max(H0, H1), W0 + W1 + margin

            out = 255*np.ones((H, W,3), np.uint8)
            out[:H0, :W0] = image0
            out[:H1, W0+margin:] = image1

        if show_keypoints:
            kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
            white = (255, 255, 255)
            black = (0, 0, 0)
            for x, y in kpts0:
                cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
                cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
            for x, y in kpts1:
                cv2.circle(out, (x + margin + W0, y), 2, black, -1,
                        lineType=cv2.LINE_AA)
                cv2.circle(out, (x + margin + W0, y), 1, white, -1,
                        lineType=cv2.LINE_AA)

        mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
        if color is None:
            color = np.array([[0,255,0]]*mkpts0.shape[0])
        else:    
            color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]
        for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
            c = c.tolist()
            cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                    color=c, thickness=1, lineType=cv2.LINE_AA)
            # display line end-points as circles
            cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
                    lineType=cv2.LINE_AA)

        # Scale factor for consistent visualization across scales.
        sc = min(H / 640., 2.0)

        # Big text.
        Ht = int(30 * sc)  # text height
        txt_color_fg = (255, 255, 255)
        txt_color_bg = (0, 0, 0)
        if text is not None:
            for i, t in enumerate(text):
                cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                            1.0*sc, txt_color_bg, 2, cv2.LINE_AA)
                cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                            1.0*sc, txt_color_fg, 1, cv2.LINE_AA)

        # Small text.
        if small_text is not None:
            Ht = int(18 * sc)  # text height
            for i, t in enumerate(reversed(small_text)):
                cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                            0.5*sc, txt_color_bg, 2, cv2.LINE_AA)
                cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                            0.5*sc, txt_color_fg, 1, cv2.LINE_AA)
    elif axis==0:
        if len(image0.shape)==2:
            H0, W0 = image0.shape
            H1, W1 = image1.shape
            H, W = H0+H1+margin, max(W0,W1)

            out = 255*np.ones((H, W), np.uint8)
            out[:H0, :W0] = image0
            out[H0+margin:, :W1] = image1
            out = np.stack([out]*3, -1)
        elif len(image0.shape)==3:
            H0, W0,_ = image0.shape
            H1, W1,_= image1.shape
            H, W = H0+H1+margin, max(W0,W1)

            out = 255*np.ones((H, W,3), np.uint8)
            out[:H0, :W0] = image0
            out[H0+margin:, :W1] = image1

        if show_keypoints:
            kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
            white = (255, 255, 255)
            black = (0, 0, 0)
            for x, y in kpts0:
                cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
                cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
            for x, y in kpts1:
                cv2.circle(out, (x , y + margin + H0), 2, black, -1,
                        lineType=cv2.LINE_AA)
                cv2.circle(out, (x , y + margin + H0), 1, white, -1,
                        lineType=cv2.LINE_AA)

        mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
        if color is None:
            color = np.array([[0,255,0]]*mkpts0.shape[0])
        else:    
            color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]
        for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
            c = c.tolist()
            cv2.line(out, (x0, y0), (x1 , y1 + margin + H0),
                    color=c, thickness=1, lineType=cv2.LINE_AA)
            # display line end-points as circles
            cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x1, y1 + margin + H0), 2, c, -1,
                    lineType=cv2.LINE_AA)

        # Scale factor for consistent visualization across scales.
        sc = min( W / 480., 2.0)

        # Big text.
        Ht = int(30 * sc)  # text height
        txt_color_fg = (255, 255, 255)
        txt_color_bg = (0, 0, 0)
        if text is not None:
            for i, t in enumerate(text):
                cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                            1.0*sc, txt_color_bg, 2, cv2.LINE_AA)
                cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                            1.0*sc, txt_color_fg, 1, cv2.LINE_AA)

        # Small text.
        if small_text is not None:
            Ht = int(18 * sc)  # text height
            for i, t in enumerate(reversed(small_text)):
                cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                            0.5*sc, txt_color_bg, 2, cv2.LINE_AA)
                cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                            0.5*sc, txt_color_fg, 1, cv2.LINE_AA)
    if path is not None:
        cv2.imwrite(str(path), out)

    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.waitKey(1)

    return out
def generate_idx(dis,idx0,idx1):
    idxs=[]
    for d,i0,i1 in zip(dis,idx0,idx1):
        anchor=d[0]
        for i, sampe_d in enumerate(d):
            if np.abs(sampe_d-anchor)<1:
                idxs+=[[i0[i],i1]]
    return np.array(idxs)
def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    height, width,_ = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one*width, one*height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return np.array((kpts - center[:, None, :]) / scaling[:, None, :])[0]
def get(root ,save_img_path=None):
    file_list=os.listdir(root)
    imgf_list,kpsf_list=[],[]
    gtf_location,matchf_idx=None,None
    for file_ in file_list:
        if 'gt_location' in file_:gtf_location=os.path.join(root,file_)
        elif 'match_idx' in file_:matchf_idx=os.path.join(root,file_)
        elif 'key_location' in file_:kpsf_list+=[os.path.join(root,file_)]
        elif 'JPG' in file_:imgf_list+=[os.path.join(root,file_)]
    imgf_list.sort(),kpsf_list.sort()
    assert len(imgf_list)==len(kpsf_list)==2 and gtf_location is not None and matchf_idx is not None
    
    img_l,img_r=cv2.imread(imgf_list[0]),cv2.imread(imgf_list[1])
    with open(gtf_location,"r") as f:
        gt_location=np.array(split_str(f.readlines())[1:])
    with open(matchf_idx,"r") as f:
        match_idx=np.array(split_str(f.readlines()))
    with open(kpsf_list[0],"r") as f:
        tkps0=np.array(split_str(f.readlines())[1:])
    with open(kpsf_list[1],"r") as f:
        tkps1=np.array(split_str(f.readlines())[1:])

    
    mkps0,mkps1=gt_location[:,0:2],gt_location[:,2:]
    if save_img_path is not None:
        out_o=make_matching_plot_fast(img_l,img_r,tkps0,tkps1,mkps0,mkps1)

    tree_l,tree_r=KDTree(tkps0),KDTree(tkps1)
    seach_l,seach_r=tree_l.query(mkps0,k=5, eps=0, p=2),tree_r.query(mkps1,k=1, eps=0, p=2)
    
    #idxs=np.concatenate((seach_l[1][:,None],seach_r[1][:,None]),axis=1)
    idxs=generate_idx(seach_l[0],seach_l[1],seach_r[1])
    labels=[False]*len(match_idx)
    err=0
    for idx in idxs:
        temp=match_idx[idx[0]]
        #labels[idx[0]]=True
        if temp[0]==idx[0] and temp[1]==idx[1]:
            labels[idx[0]]=True
        elif np.power(tkps1[idx[1]]-tkps1[temp[1].astype(int)],2).sum()<1:
            labels[idx[0]]=True
        else:
            err+=1
    print("err:{}".format(err))
    labels=np.array(labels)


    kps_l=tkps0[match_idx[:,0].astype(int)]
    kps_r=tkps1[match_idx[:,1].astype(int)]
    xs_initial=np.concatenate((kps_l,kps_r),axis=1)
    xs=np.concatenate((normalize_keypoints(torch.Tensor(kps_l),img_l.shape),normalize_keypoints(torch.Tensor(kps_r),img_r.shape)),axis=1)


    match_idx=match_idx[labels]
    mkps0=tkps0[match_idx[:,0].astype(int)]
    mkps1=tkps1[match_idx[:,1].astype(int)]
    if save_img_path is not None:
        out_c=make_matching_plot_fast(img_l,img_r,tkps0,tkps1,mkps0,mkps1)
        cv2.imwrite("{}.jpg".format(root),np.concatenate((out_o,out_c)))
    

    
    return xs_initial,xs,labels,imgf_list

    

    

        
if __name__=="__main__":
    # download dataset from https://github.com/sailor-z/Grid-GTM
    root_list=['./part-a','./part-b','./part-c']
    save_path='./multi_correspondences.pkl'
    data_type = ['xs_initial','xs','label','pairs']
    xs_initial_list,xs_list,labels_list,pairs_list=[],[],[],[]
    data={}
    for root in root_list:
        sce_list=os.listdir(root)

        for sce in sce_list:
            if '.jpg'  not in sce: 
                sce_root=os.path.join(root,sce)
                xs_initial,xs,labels,pairs=get(sce_root,True)
                xs_initial_list +=[xs_initial]
                xs_list +=[xs]
                labels_list +=[labels]
                pairs_list +=[pairs]
    data['xs_initial']=xs_initial_list
    data['xs']=xs_list
    data['label']=labels_list
    data['pairs']=pairs_list
    with open(save_path, "wb") as ofp:
        pickle.dump(data, ofp)
    
    
    # img_l=cv2.imread("/data3/yexinyi/Datasets/Grid-GTM/part-a/case1/DSC00762.JPG")
    # img_r=cv2.imread("/data3/yexinyi/Datasets/Grid-GTM/part-a/case1/DSC00763.JPG")
    # with open("/data3/yexinyi/Datasets/Grid-GTM/part-a/case1/gt_location.txt","r") as f:
    #     gt_location=f.readlines()
    #     gt_location=np.array(split_str(gt_location)[1:])

    # with open("/data3/yexinyi/Datasets/Grid-GTM/part-a/case1/1key_location.txt","r") as f:
    #     tkps0=f.readlines()
    #     tkps0=np.array(split_str(tkps0)[1:])
    # with open("/data3/yexinyi/Datasets/Grid-GTM/part-a/case1/2key_location.txt","r") as f:
    #     tkps1=f.readlines()
    #     tkps1=np.array(split_str(tkps1)[1:])
    # with open("/data3/yexinyi/Datasets/Grid-GTM/part-a/case1/match_idx.txt","r") as f:
    #     match_idx=f.readlines()
    #     match_idx=np.array(split_str(match_idx))

    

    # mkps0,mkps1=gt_location[:,0:2],gt_location[:,2:]
    # out=make_matching_plot_fast(img_l,img_r,tkps0,tkps1,mkps0,mkps1)
    # cv2.imwrite("a.jpg",out)
    
    # #kps_l=tkps0[match_idx[:,0].astype(int)]
    # #kps_r=tkps1[match_idx[:,1].astype(int)]

    # tree_l,tree_r=KDTree(tkps0),KDTree(tkps1)
    # seach_l,seach_r=tree_l.query(mkps0,k=5, eps=0, p=2),tree_r.query(mkps1,k=1, eps=0, p=2)
    
    # #idxs=np.concatenate((seach_l[1][:,None],seach_r[1][:,None]),axis=1)
    # idxs=generate_idx(seach_l[0],seach_l[1],seach_r[1])
    # labels=[False]*len(match_idx)
    # err=0
    # for idx in idxs:
    #     temp=match_idx[idx[0]]
    #     #labels[idx[0]]=True
    #     if temp[0]==idx[0] and temp[1]==idx[1]:
    #         labels[idx[0]]=True
    #     elif np.power(tkps1[idx[1]]-tkps1[temp[1].astype(int)],2).sum()<1:
    #         labels[idx[0]]=True
    #     else:
    #         err+=1
    # print("err:{}".format(err))
    # labels=np.array(labels)
    # match_idx=match_idx[labels]
    # mkps0=tkps0[match_idx[:,0].astype(int)]
    # mkps1=tkps1[match_idx[:,1].astype(int)]
    # out=make_matching_plot_fast(img_l,img_r,tkps0,tkps1,mkps0,mkps1)
    # cv2.imwrite("b.jpg",out)
    