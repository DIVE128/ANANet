import cv2
import numpy as np
import torch

class SIFTDetector:
    def __init__(self, cfg):
        num_kp = 2000
        contrastThreshold = 1e-5
        self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=num_kp, contrastThreshold=contrastThreshold)

    def __call__(self, img):
        cv_kp, desc = self.sift.detectAndCompute(img, None)
        kp = np.array([[_kp.pt[0], _kp.pt[1]] for _kp in cv_kp])  # N*4
        sizes = np.asarray([_kp.size for _kp in cv_kp])
        responses = np.asarray([_kp.response for _kp in cv_kp])
        angles = np.asarray([_kp.angle for _kp in cv_kp])
        kp=np.concatenate([kp,responses[:,None],sizes[:,None],angles[:,None]],1)
        return kp, desc


name2det = {
    'sift': SIFTDetector,
}