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
    self.K[0, 0] = fx
    self.K[1, 1] = fy
    self.K[0, 2] = px
    self.K[1, 2] = py
    self.K[2, 2] = 1.0

  def setR(self,
      r_00, r_01, r_02,
      r_10, r_11, r_12,
      r_20, r_21, r_22):
    self.R[0, 0] = r_00
    self.R[0, 1] = r_01
    self.R[0, 2] = r_02
    self.R[1, 0] = r_10
    self.R[1, 1] = r_11
    self.R[1, 2] = r_12
    self.R[2, 0] = r_20
    self.R[2, 1] = r_21
    self.R[2, 2] = r_22

  def setT(self, t_03, t_13, t_23):
    # self.t[0, 0] = t_03
    # self.t[1, 0] = t_13
    # self.t[2, 0] = t_23
    self.t = np.array([t_03, t_13, t_23])

  def updateP(self):
    Rt = np.zeros([3, 4])
    Rt[0:3, 0:3] = self.R
    Rt[0:3, 3] = self.t
    self.P = self.K.dot(Rt)

  def __init__(self, config):
    # intrinsic 相机内参
    # extrinsic 相机外参
    intrinsic = config[0]
    extrinsic = config[1]
    self.setK(intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2])
    self.setR(extrinsic[0][0], extrinsic[0][1], extrinsic[0][2],
              extrinsic[1][0], extrinsic[1][1], extrinsic[1][2],
              extrinsic[2][0], extrinsic[2][1], extrinsic[2][2])
    self.setT(extrinsic[0][3], extrinsic[1][3], extrinsic[2][3])
    self.updateP()

class Camera1:

  K = np.zeros([3, 3])
  R = np.zeros([3, 3])
  t = np.zeros([3, 1])
  P = np.zeros([3, 4])

  def setK(self, fx, fy, px, py):
    self.K[0, 0] = fx
    self.K[1, 1] = fy
    self.K[0, 2] = px
    self.K[1, 2] = py
    self.K[2, 2] = 1.0

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


# # parse command line arguments
# parser = argparse.ArgumentParser(description="Warps camera images to the plane z=0 in the world frame.")
# parser.add_argument("camera_img_pair", metavar="CAM IMG", nargs='*', help="camera config file and image file")
# parser.add_argument("-wm", type=float, help="output image width in [m]", default=30)
# parser.add_argument("-hm", type=float, help="output image height in [m]", default=40)
# parser.add_argument("-r", type=float, help="output image resolution in [px/m]", default=20)
# parser.add_argument("--drone", type=str, help="camera config file of drone to map to")
# parser.add_argument("--batch", help="process folders of images instead of single images", action="store_true")
# parser.add_argument("--output", help="output directory to write transformed images to")
# parser.add_argument("--cc", help="use with color-coded images to enable NN-interpolation", action="store_true")
# parser.add_argument("-v", help="only print homography matrices", action="store_true")
# args = parser.parse_args()


# load camera configurations and image paths
# cameraConfigs = []
# imagePathArgs = []
# for aIdx in range(int(len(args.camera_img_pair) / 2.0)):
#   with open(os.path.abspath(args.camera_img_pair[2*aIdx])) as stream:
#     cameraConfigs.append(yaml.safe_load(stream))
#   imagePathArgs.append(args.camera_img_pair[2*aIdx+1])
# toDrone = False
# if args.drone:
#   toDrone = True
#   with open(os.path.abspath(args.drone)) as stream:
#     droneConfig = yaml.safe_load(stream)

class ipm():
  def __init__(self, view_num, image_paths, camera_configs, drone_config, save_path):
    self.image_paths = image_paths
    self.camera_configs = camera_configs
    self.drone_config = drone_config
    # 加载相机参数和图像
    self.cams = []
    for i in range(view_num):
      self.cams.append(Camera(camera_configs[i]))
    self.drone = Camera1(drone_config)
    self.save_path = save_path
    self.processor()

    

  def processor(self):
    print("_____开始进行IPM处理_____")
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
      IPMs.append(np.linalg.inv(cam.P.dot(M)))

    # setup masks to later clip invalid parts from transformed images (inefficient, but constant runtime)
    masks = []
    for cam in self.cams:
      mask = np.zeros((outputRes[0], outputRes[1], 3), dtype=bool)
      # rotate_matrix = np.array([
      #   [config['r_00'], config['r_01'], config['r_02']],
      #   [config['r_10'], config['r_11'], config['r_12']],
      #   [config['r_20'], config['r_21'], config['r_22']]
      #   ], dtype='float32')
      rotate_matrix = R.from_matrix(cam.R)
      rotate_degrees = rotate_matrix.as_euler('xyz', degrees=True)
      print(rotate_degrees)
      yaw = rotate_degrees[0]
      pitch = rotate_degrees[1]
      roll = rotate_degrees[2]
      for i in range(outputRes[1]):
        for j in range(outputRes[0]):
          theta = np.rad2deg(np.arctan2(-j + outputRes[0] / 2 - self.drone_config["YCam"] * pxPerM[0], i - outputRes[1] / 2 + self.drone_config["XCam"] * pxPerM[1]))
          if abs(theta - yaw) > 90 and abs(theta - yaw) < 270:
            mask[j,i,:] = True
      masks.append(mask)

    # process images
    for imgPath in tqdm(self.image_paths):
      # load images
      images = []
      images.append(cv2.imread(imgPath))

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
