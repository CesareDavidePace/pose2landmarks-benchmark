import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def get_angles_error(x):
    '''
        Input: (T, 17, 3)
        Output: (T, 16)
    '''
    limbs_id = [[0,1], [1,2], [2,3],
         [0,4], [4,5], [5,6],
         [0,7], [7,8], [8,9], [9,10],
         [8,11], [11,12], [12,13],
         [8,14], [14,15], [15,16]
        ]
    angle_id = [[ 0,  3],
                [ 0,  6],
                [ 3,  6],
                [ 0,  1],
                [ 1,  2],
                [ 3,  4],
                [ 4,  5],
                [ 6,  7],
                [ 7, 10],
                [ 7, 13],
                [ 8, 13],
                [10, 13],
                [ 7,  8],
                [ 8,  9],
                [10, 11],
                [11, 12],
                [13, 14],
                [14, 15] ]
    eps = 1e-7
    limbs = x[:,limbs_id,:]
    limbs = limbs[:,:,0,:]-limbs[:,:,1,:]
    angles = limbs[:,angle_id,:]

    angle_cos = F.cosine_similarity(angles[:,:,0,:], angles[:,:,1,:], dim=-1)
    return torch.acos(angle_cos.clamp(-1+eps, 1-eps))




def angle_error(x, gt):
    '''
        Input: (N, T, 17, 3), (N, T, 17, 3)
    '''
    limb_angles_x = get_angles(x)
    limb_angles_gt = get_angles(gt)
    return nn.L1Loss()(limb_angles_x, limb_angles_gt)

def calculate_angle_error(x, gt):
    '''
        Input: (T, 17, 3), (T, 17, 3)
    '''

    limb_angle_x = get_angles_error(x)
    limb_angle_gt = get_angles_error(gt)

    # calculate the mean angle error
    angle_error = torch.abs(limb_angle_x - limb_angle_gt)
    return angle_error

    
