from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from unittest import TestLoader
from matplotlib.pyplot import title
import numpy as np
import random
import time
import datetime
import pprint
import torch
import torch.utils.data
import matplotlib
matplotlib.use('Agg')
import cv2

# cv2.setNumThreads(0)
# pytorch issue 1355: possible deadlock in dataloader. OpenCL may be enabled by default in OpenCV3;
# disable it because it's not thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)
import _init_paths
import ref
import utils.fancy_logger as logger
from utils.io import load_ply_vtx, write_ply
from model import build_model, save_model
from datasets.lm import LM
from train import train
from test import test
from config import config
from tqdm import tqdm 
from utils.transform3d import prj_vtx_observed
import open3d as o3d

def pts3D_visualization(pts_path, title):
    pcd = o3d.io.read_point_cloud(pts_path)
    o3d.visualization.draw_geometries([pcd], window_name=title) # 快速显示

def main():
    cfg = config().parse()

    # 以sample为单位
    # testDataset.load_pose({idx}), testDataset.__len__(), testDataset.cam_K
    testDataset = LM(cfg, 'test')   # 13425 test samples
    trainDataset = LM(cfg, 'train') # 2375 real samples and 0 synthetic samples
    test_len = len(testDataset)
    train_len = len(trainDataset)
    logger.info("Processing {} test samples and {} train samples... ".format(test_len, train_len))

    # 以obj为单位
    obj_vtx = {}
    logger.info('load 3d object models...')
    for obj in tqdm(cfg.dataset.classes):
        obj_vtx[obj] = load_ply_vtx(os.path.join(ref.lm_model_dir, '{}/{}.ply'.format(obj, obj)))   # 点云数据
    obj_info = LM.load_lm_model_info(ref.lm_model_info_pth)

    '''
    # 遍历testData
    for i in tqdm(range(test_len)):
        sampleOBJ = testDataset.load_obj(i)     # 对象
        samplePOSE = testDataset.load_pose(i)   # (3, 4)
        obVTX_path = testDataset.load_obVTX_pth(i)
        pts3D = prj_vtx_observed(obj_vtx[sampleOBJ], samplePOSE)
        # save
        try:
            write_ply(obVTX_path, pts3D)
        except:
            logger.error("{obVTX_path} fail to store!!")
            break
    logger.info("[testData]Succeed in creating obVTXs!")
    '''

# 遍历testData
    for i in tqdm(range(train_len)):
        sampleOBJ = trainDataset.load_obj(i)     # 对象
        samplePOSE = trainDataset.load_pose(i)   # (3, 4)
        obVTX_path = trainDataset.load_obVTX_pth(i)
        pts3D = prj_vtx_observed(obj_vtx[sampleOBJ], samplePOSE)
        # save
        try:
            write_ply(obVTX_path, pts3D)
        except:
            logger.error("{obVTX_path} fail to store!!")
            break
    logger.info("[trainData]Succeed in creating obVTXs!")

'''
    # 对test和train的数据进行处理，试验
    sample0_obj = testDataset.load_obj(0)   # 对象
    sample0_pose = testDataset.load_pose(0) # (3, 4)
    # logger.info("[DEBUG]sample0_obj: {}".format(sample0_obj))
    # logger.info("[DEBUG]sample0_pose: {}".format(sample0_pose))
    pts3D = prj_vtx_observed(obj_vtx[sample0_obj], sample0_pose)
    # logger.info("[DEBUG]sample0_pts_observed: {}".format(pts3D))

    # save 3d points
    path_to_save = os.path.join(cfg.pytorch.save_path, 'sample0_observed.ply')
    write_ply(path_to_save, pts3D)
    logger.info("[DEBUG]Successful stored")
    # open3d渲染，对比
    original_pth = os.path.join(ref.lm_model_dir, '{}/{}.ply'.format(sample0_obj, sample0_obj))
    pts3D_visualization(path_to_save, "observed")     # observed
    pts3D_visualization(original_pth, "canonical")    # canonical
'''

if __name__ == '__main__':
    main()
