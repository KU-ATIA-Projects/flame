from src.utils.eval_util import load_mclip_model


mclip_ckpt = '/home/pjr726/flame/flame_mclip_hml3d_bc.ckpt'
mclip = load_mclip_model(mclip_ckpt)
mclip.to('cuda:0')

import sys
sys.path.append('../HumanML3D/')

from os.path import join as pjoin
from common.skeleton import Skeleton
from common.quaternion import *
from paramUtil import *
import scipy.ndimage.filters as filters


n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
kinematic_chain = t2m_kinematic_chain
# Face direction, r_hip, l_hip, sdr_r, sdr_l
face_joint_indx = [2, 1, 17, 16]


def get_cont6d_params(positions):
    skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
    # (seq_len, joints_num, 4)
    quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)

    '''Quaternion to continuous 6D'''
    cont_6d_params = quaternion_to_cont6d_np(quat_params)
    # (seq_len, 4)
    r_rot = quat_params[:, 0].copy()
    #     print(r_rot[0])
    '''Root Linear Velocity'''
    # (seq_len - 1, 3)
    velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
    #     print(r_rot.shape, velocity.shape)
    velocity = qrot_np(r_rot[1:], velocity)
    '''Root Angular Velocity'''
    # (seq_len - 1, 4)
    r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
    # (seq_len, joints_num, 4)
    return cont_6d_params, r_velocity, velocity, r_rot

from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d
import torch
import torch.nn.functional as F
import numpy as np

def xyz_to_rot6d(xyz):
    assert isinstance(xyz, np.ndarray), "The input should be numpy array"
    assert len(xyz.shape) == 3, "The shape of xyz should be (seq_len, 22, 3)"
    # Motion length, number of joints, feature dimension
    L, J, D = xyz.shape

    assert J == 22 and D == 3, "The shape of xyz should be (seq_len, 22, 3)"
    # Pad to add the missing hand joints and make it 24 joints
    # axis_angles = np.pad(xyz, (0, 0, 0, 2, 0, 0), mode='constant', value=0)

    # Pad to add the missing hand joints and make it 24 joints using numpy
    xyz = np.pad(xyz, ((0, 0), (0, 2), (0, 0)), mode='constant', constant_values=0)
    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(xyz)
    return cont_6d_params


def rot6d_to_flame(rot6d):
    assert isinstance(rot6d, np.ndarray), "The input should be numpy array"
    assert len(rot6d.shape) == 3 and rot6d.shape[1] == 24 and rot6d.shape[2] == 6, "The shape of rot6d should be (seq_len, 24, 6)"
    # The input is (120, 24, 6), make it (120, 147, 1)
    flame = rot6d.reshape(rot6d.shape[0], -1)
    flame = np.lib.pad(flame, ((0, 0), (0, 3)), 'constant', constant_values=0)
    return flame


if __name__ == '__main__':
    xyz = np.load('/home/pjr726/flame/sample00_rep00.npy')
    rot6d = xyz_to_rot6d(xyz)
    flame = rot6d_to_flame(rot6d)

    motion = torch.from_numpy(flame).float().to('cuda:0')
    motion.unsqueeze(0).permute(0, 2, 1).shape

    text = ['A person is walking in the street.']
    mclip.get_features(motion=motion.unsqueeze(0).permute(0, 2, 1), texts=list(text) * 10, motion_length=torch.tensor([motion.shape[0]], device='cuda:0'))