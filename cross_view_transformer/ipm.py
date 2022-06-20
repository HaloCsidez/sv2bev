#!/usr/bin/env python

# ==============================================================================
# MIT License
#
# Copyright 2020 Institute for Automotive Engineering of RWTH Aachen University.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================


# python -m debugpy --wait-for-client --listen 0.0.0.0:8782 ipm4nus.py --batch --output output/ --drone ./camera_configs/1_Surround_View/drone.yaml ./camera_configs/0_Surround_View/CAM_FRONT.yaml ./nuScenes/cam_front/ ./camera_configs/0_Surround_View/CAM_FRONT_LEFT.yaml ./nuScenes/cam_front_left/ ./camera_configs/0_Surround_View/CAM_FRONT_RIGHT.yaml ./nuScenes/cam_front_right ./camera_configs/0_Surround_View/CAM_BACK.yaml ./nuScenes/cam_back/ ./camera_configs/0_Surround_View/CAM_BACK_LEFT.yaml ./nuScenes/cam_back_left ./camera_configs/0_Surround_View/CAM_BACK_RIGHT.yaml ./nuScenes/cam_back_right

import imp
import os
import sys
import yaml
import numpy as np
import cv2
import argparse
from tqdm import tqdm
from .rotation2angle import rotationMatrixToEulerAngles
from scipy.spatial.transform import Rotation as R

class Camera:
    
  K = np.zeros([3, 3])
  R = np.zeros([3, 3])
  t = np.zeros([3, 1])
  P = np.zeros([3, 4])

  def setK(self, fx, fy, px, py):
    self.K = np.array([[fx, 0, px],
                      [0, fy, py],
                      [0, 0, 1]])

  def setR(self, y, p, r):

    Rz = np.array([[np.cos(-y), -np.sin(-y), 0.0], [np.sin(-y), np.cos(-y), 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[np.cos(-p), 0.0, np.sin(-p)], [0.0, 1.0, 0.0], [-np.sin(-p), 0.0, np.cos(-p)]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(-r), -np.sin(-r)], [0.0, np.sin(-r), np.cos(-r)]])
    Rs = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]) # switch axes (x = -y, y = -z, z = x)
    self.R = Rs.dot(Rz.dot(Ry.dot(Rx)))

  def setT(self, XCam, YCam, ZCam):
    X = np.array([XCam, YCam, ZCam])
    self.t = -self.R.dot(X)

  def updateP(self):
    Rt = np.zeros([3, 4])
    Rt[0:3, 0:3] = self.R
    Rt[0:3, 3] = self.t
    self.P = self.K.dot(Rt)

  def __init__(self, config):
    self.setK(config["fx"], config["fy"], config["px"], config["py"])
    self.setR(np.deg2rad(config["yaw"]), np.deg2rad(config["pitch"]), np.deg2rad(config["roll"]))
    self.setT(config["XCam"], config["YCam"], config["ZCam"])
    self.updateP()



class ipm():
  def __init__(self, view_num, image_paths, camera_configs, drone_config, save_path, dataset_dir):
    self.image_paths = image_paths
    self.camera_configs = camera_configs
    self.drone_config = drone_config
    self.dataset_dir = dataset_dir
    # 加载相机参数和图像
    self.cams = []
    for config in camera_configs:
      self.cams.append(Camera(config))
    self.drone = Camera(drone_config)
    self.save_path = save_path
    self.processor()

    

  def processor(self):
    print("ipm___开始进行IPM处理", os.path.basename(self.save_path))
    # calculate output shape; adjust to match drone image, if specified
    # 通过内参设置输出图片的尺寸
    outputRes = (int(2 * self.drone_config["py"]), int(2 * self.drone_config["px"]))
    # 得到画面的真实尺寸
    dx = outputRes[1] / self.drone_config["fx"] * self.drone_config["ZCam"]
    dy = outputRes[0] / self.drone_config["fy"] * self.drone_config["ZCam"]
    # 通过输出图片尺寸和真实尺寸得到每一个像素所代表的真实距离
    pxPerM = (outputRes[0] / dy, outputRes[1] / dx)
      

    # setup mapping from street/top-image plane to world coords
    shift = (outputRes[0] / 2.0, outputRes[1] / 2.0)
    shift = shift[0] + self.drone_config["YCam"] * pxPerM[0], shift[1] - self.drone_config["XCam"] * pxPerM[1]
    M = np.array([[1.0 / pxPerM[1], 0.0, -shift[1] / pxPerM[1]], [0.0, -1.0 / pxPerM[0], shift[0] / pxPerM[0]], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    # find IPM as inverse of P*M
    IPMs = []
    for cam in self.cams:
      ipm_matrix = cam.P.dot(M)
      ipm_matrix_inv = np.linalg.inv(ipm_matrix)
      IPMs.append(ipm_matrix_inv)
      assert np.allclose(np.dot(ipm_matrix, ipm_matrix_inv), np.eye(3))

    # setup masks to later clip invalid parts from transformed images (inefficient, but constant runtime)
    masks = []
    for config in self.camera_configs:
      mask = np.zeros((outputRes[0], outputRes[1], 3), dtype=bool)
      for i in range(outputRes[1]):
        for j in range(outputRes[0]):
          theta = np.rad2deg(np.arctan2(-j + outputRes[0] / 2 - self.drone_config["YCam"] * pxPerM[0], i - outputRes[1] / 2 + self.drone_config["XCam"] * pxPerM[1]))
          if abs(theta - config["yaw"]) > 90 and abs(theta - config["yaw"]) < 270:
            mask[j,i,:] = True
    masks.append(mask)
    
    images = []
    for imgPath in self.image_paths:
      # load images
      images.append(cv2.imread(os.path.join(self.dataset_dir, imgPath)))

    # warp input images
    interpMode = cv2.INTER_NEAREST # cv2.INTER_LINEAR
    warpedImages = []
    for img, IPM in zip(images, IPMs):
      warpedImages.append(cv2.warpPerspective(img, IPM, (outputRes[1], outputRes[0]), flags=interpMode))

    # remove invalid areas (behind the camera) from warped images
    for warpedImg, mask in zip(warpedImages, masks):
      warpedImg[mask] = 0

    # stitch separate images to total bird's-eye-view
    birdsEyeView = np.zeros(warpedImages[0].shape, dtype=np.uint8)
    for warpedImg in warpedImages:
      mask = np.any(warpedImg != (0,0,0), axis=-1)
      birdsEyeView[mask] = warpedImg[mask]

    # display or export bird's-eye-view
    cv2.imwrite(self.save_path, birdsEyeView)
    print('ipm___save file', self.save_path)
