import os
import sys
import cv2
import numpy as np
import torch
from torchvision import transforms
import torchvision.models as models

CONSTANT_HEIGHT = 224

def classify_cat():
    numI = cv2.imread('cat.jpeg')
    new_width = int(round(numI.shape[1] / numI.shape[0] * CONSTANT_HEIGHT))
    numI = cv2.resize(numI, (new_width, CONSTANT_HEIGHT))
    X = torch.from_numpy((numI[np.newaxis, :, :, ::-1] / 255.0).astype('float32')).permute([0,3,1,2])
    my_clf = models.vgg16(pretrained=True)
    clf_probs = my_clf(transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])(X))
    import pdb
    pdb.set_trace()

if __name__ == '__main__':
    classify_cat()
