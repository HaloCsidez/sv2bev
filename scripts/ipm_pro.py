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

# python -m debugpy --listen 0.0.0.0:8787 --wait-for-client ipm.py --batch --out ../datasets/test --cc --drone ../datasets/0_Surround_View_test/drone.yaml ../datasets/0_Surround_View_test/CAM_FRONT_LEFT.yaml ../datasets/nuScenes/samples/CAM_FRONT_LEFT/ ../datasets/0_Surround_View_test/CAM_FRONT.yaml ../datasets/nuScenes/samples/CAM_FRONT ../datasets/0_Surround_View_test/CAM_BACK_LEFT.yaml ../datasets/nuScenes/samples/CAM_BACK_LEFT ../datasets/0_Surround_View_test/CAM_BACK.yaml ../datasets/nuScenes/samples/CAM_BACK ../datasets/0_Surround_View_test/CAM_BACK_RIGHT.yaml ../datasets/nuScenes/samples/CAM_BACK_RIGHT

# python -m debugpy --listen 0.0.0.0:8787 --wait-for-client ipm_pro.py --batch --out ../datasets/test --cc --drone ../datasets/0_Surround_View_test/drone.yaml ../datasets/0_Surround_View_test/CAM_FRONT_LEFT.yaml ../datasets/0_Surround_View_test/CAM_FRONT.yaml ../datasets/0_Surround_View_test/CAM_BACK_LEFT.yaml ../datasets/0_Surround_View_test/CAM_BACK.yaml ../datasets/0_Surround_View_test/CAM_BACK_RIGHT.yaml

from fileinput import filename
import torch
import os
import sys
import yaml
import numpy as np
import cv2
import argparse
from tqdm import tqdm
from pathlib import Path
from functools import lru_cache
from omegaconf import DictConfig
from pyquaternion import Quaternion
from shapely.geometry import MultiPolygon

from cross_view_transformer.data.common import INTERPOLATION, get_view_matrix, get_pose, get_split
from cross_view_transformer.data.transforms import Sample, SaveDataTransform

class NuScenesSingleton:
    """
    Wraps both nuScenes and nuScenes map API

    This was an attempt to sidestep the 30 second loading time in a "clean" manner
    """
    def __init__(self, dataset_dir, version):
        """
        dataset_dir: /path/to/nuscenes/
        version: v1.0-trainval
        """
        self.dataroot = str(dataset_dir)
        self.nusc = self.lazy_nusc(version, self.dataroot)

    @classmethod
    def lazy_nusc(cls, version, dataroot):
        # Import here so we don't require nuscenes-devkit unless regenerating labels
        from nuscenes.nuscenes import NuScenes

        if not hasattr(cls, '_lazy_nusc'):
            cls._lazy_nusc = NuScenes(version=version, dataroot=dataroot)

        return cls._lazy_nusc

    def get_scenes(self):
        for scene_record in self.nusc.scene:
            yield scene_record['name'], scene_record

    @lru_cache(maxsize=16)
    def get_map(self, log_token):
        # Import here so we don't require nuscenes-devkit unless regenerating labels
        from nuscenes.map_expansion.map_api import NuScenesMap

        map_name = self.nusc.get('log', log_token)['location']
        nusc_map = NuScenesMap(dataroot=self.dataroot, map_name=map_name)

        return nusc_map

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_singleton'):
            obj = super(NuScenesSingleton, cls).__new__(cls)
            obj.__init__(*args, **kwargs)

            cls._singleton = obj

        return cls._singleton


class NuScenesDataset(torch.utils.data.Dataset):
    CAMERAS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
               'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

    def __init__(
        self,
        scene_name: str,
        scene_record: dict,
        helper: NuScenesSingleton,
        camera_config: DictConfig,
        cameras=[[0, 1, 2, 3, 4, 5]],
        bev={'h': 200, 'w': 200, 'h_meters': 100, 'w_meters': 100, 'offset': 0.0},
    ):
        self.scene_name = scene_name
        self.nusc = helper.nusc
        self.nusc_map = helper.get_map(scene_record['log_token'])
        self.bev_shape = (bev['h'], bev['w'])
        self.samples = self.parse_scene(scene_record, cameras, camera_config)

    def parse_scene(self, scene_record, camera_rigs, camera_config):
        data = []
        sample_token = scene_record['first_sample_token']

        while sample_token:
            sample_record = self.nusc.get('sample', sample_token)

            for camera_rig in camera_rigs:
                data.append(self.parse_sample_record(sample_record, camera_rig, camera_config))

            sample_token = sample_record['next']

        return data

    def parse_pose(self, record, *args, **kwargs):
        return get_pose(record['rotation'], record['translation'], *args, **kwargs)

    def parse_sample_record(self, sample_record, camera_rig, camera_config):
        lidar_record = self.nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
        egolidar = self.nusc.get('ego_pose', lidar_record['ego_pose_token'])

        world_from_egolidarflat = self.parse_pose(egolidar, flat=True)
        egolidarflat_from_world = self.parse_pose(egolidar, flat=True, inv=True)

        cam_channels = []
        images = []
        intrinsics = []
        extrinsics = []
        translation = []
        rotation = []
        euler = []

        for cam_idx in camera_rig:
            cam_channel = self.CAMERAS[cam_idx]
            cam_token = sample_record['data'][cam_channel]
            
            cam_record = self.nusc.get('sample_data', cam_token)
            egocam = self.nusc.get('ego_pose', cam_record['ego_pose_token'])
            cam = self.nusc.get('calibrated_sensor', cam_record['calibrated_sensor_token'])

            cam_from_egocam = self.parse_pose(cam, inv=True)
            egocam_from_world = self.parse_pose(egocam, inv=True)

            E = cam_from_egocam @ egocam_from_world @ world_from_egolidarflat
            I = cam['camera_intrinsic']

            full_path = Path(self.nusc.get_sample_data_path(cam_token))
            image_path = str(full_path.relative_to(self.nusc.dataroot))

            cam_channels.append(cam_channel)
            intrinsics.append(I)
            extrinsics.append(E.tolist())
            images.append(image_path)
            translation.append(cam['translation'])
            rotation.append(cam['rotation'])
            c = dict(camera_config[cam_channel.lower()])
            c['fx'] = I[0][0]
            c['fy'] = I[1][1]
            c['px'] = I[0][2]
            c['py'] = I[1][2]
            c['XCam'] = cam['translation'][0]
            c['YCam'] = cam['translation'][1]
            c['ZCam'] = cam['translation'][2]
            euler.append(c)

        return {
            'scene': self.scene_name,
            'token': sample_record['token'],

            # 'pose': world_from_egolidarflat.tolist(),
            # 'pose_inverse': egolidarflat_from_world.tolist(),

            'cam_ids': list(camera_rig),
            'cam_channels': cam_channels,
            'intrinsics': intrinsics,
            # 'extrinsics': extrinsics,
            'images': images,
            'translation': translation,
            # 'rotation': rotation,
            'euler': euler
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = Sample(
            view=None,
            bev=None,
            extrinsics=None,
            **sample
        )

        return data


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



# parse command line arguments
# parser = argparse.ArgumentParser(description="Warps camera images to the plane z=0 in the world frame.")
# parser.add_argument("camera_img_pair", metavar="CAM IMG", nargs='*', help="camera config file and image file")
# parser.add_argument("-wm", type=float, help="output image width in [m]", default=20)
# parser.add_argument("-hm", type=float, help="output image height in [m]", default=40)
# parser.add_argument("-r", type=float, help="output image resolution in [px/m]", default=20)
# parser.add_argument("--drone", type=str, help="camera config file of drone to map to")
# parser.add_argument("--batch", help="process folders of images instead of single images", action="store_true")
# parser.add_argument("--output", help="output directory to write transformed images to")
# parser.add_argument("--cc", help="use with color-coded images to enable NN-interpolation", action="store_true")
# parser.add_argument("-v", help="only print homography matrices", action="store_true")
# args = parser.parse_args()

cameras=[0, 1, 2, 3, 4, 5]
cameras_euler = {
    'CAM_FRONT_LEFT': {'yaw':55.0, 'pitch':0.0, 'roll':0.0},
    'CAM_FRONT': {'yaw':0.0, 'pitch':0.0, 'roll':0.0},
    'CAM_FRONT_RIGHT': {'yaw':-55.0, 'pitch':0.0, 'roll':0.0},
    'CAM_BACK_LEFT': {'yaw':110.0, 'pitch':0.0, 'roll':0.0},
    'CAM_BACK': {'yaw':180.0, 'pitch':0.0, 'roll':0.0},
    'CAM_BACK_RIGHT': {'yaw':-110.0, 'pitch':0.0, 'roll':0.0},
}
drone_config = {
    'fx': 1266.417203046554, 'fy': 1266.417203046554, 
    'px': 816.2670197447984, 'py': 491.50706579294757, 
    'yaw': 0.0, 'pitch': 90.0, 'roll': 90.0, 
    'XCam': 0, 'YCam': 0, 'ZCam': 80
}
bev={'h': 200, 'w': 200}
CAMERAS = [
    'CAM_FRONT_LEFT', 
    'CAM_FRONT', 
    'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT', 
    'CAM_BACK', 
    'CAM_BACK_RIGHT'
]
helper = NuScenesSingleton('/media/wit/HDD_0/zhouhb/cvpr2022/sv2bev/datasets/nuScenes', 'v1.0-trainval')
output_path = '/media/wit/HDD_0/zhouhb/cvpr2022/sv2bev/datasets/ipm_gt'
# split = f'mini_{split}' if version == 'v1.0-mini' else split
split_scenes = get_split('total', 'nuscenes')

drone = Camera(drone_config)
# 场景
for scene_name, scene_record in helper.get_scenes():
    sample_index = 1
    scene_mask = []
    if scene_name not in split_scenes:
        continue
    
    scene_path = os.path.join(output_path, scene_name)
    # 取场景里面的第一个token，作为mask布局；每一个场景定定义一个布局
    nusc_map = helper.get_map(scene_record['log_token'])
    sample_token = scene_record['first_sample_token']
    # 一个场景下面包含多个token, 即 多个图片
    while sample_token:
        sample_record = helper.nusc.get('sample', sample_token)

        
        cameraConfigs = []
        image_paths=[]
        for cam_idx in cameras:
            cam_channel = CAMERAS[cam_idx]
            cam_token = sample_record['data'][cam_channel]
            cam_record = helper.nusc.get('sample_data', cam_token)
            cam = helper.nusc.get('calibrated_sensor', cam_record['calibrated_sensor_token'])
            # full_path = Path(helper.nusc.get_sample_data_path(cam_token))
            image_path = helper.nusc.get_sample_data_path(cam_token)


            image_paths.append(image_path)
            cameraConfig = {
                'fx':cam['camera_intrinsic'][0][0], 
                'fy':cam['camera_intrinsic'][1][1], 
                'px':cam['camera_intrinsic'][0][2], 
                'py':cam['camera_intrinsic'][1][2],
                'yaw':cameras_euler[cam_channel]['yaw'],
                'pitch':cameras_euler[cam_channel]['pitch'],
                'roll':cameras_euler[cam_channel]['roll'],
                'XCam':cam['translation'][0],
                'YCam':cam['translation'][1],
                'ZCam':cam['translation'][2]
            }
      
            # 配置cameraConfigs的格式
            cameraConfigs.append(cameraConfig)
            
            
        cams = []
        for config in cameraConfigs:
            cams.append(Camera(config))
      
      
            
    
        if sample_index == 1:
            # 生成该场景下的mask
            print('生成', scene_name, '场景下的mask')
            outputRes = (int(2 * drone_config["py"]), int(2 * drone_config["px"]))
            dx = outputRes[1] / drone_config["fx"] * drone_config["ZCam"]
            dy = outputRes[0] / drone_config["fy"] * drone_config["ZCam"]
            pxPerM = (outputRes[0] / dy, outputRes[1] / dx)
            # setup mapping from street/top-image plane to world coords
            shift = (outputRes[0] / 2.0, outputRes[1] / 2.0)
            shift = shift[0] + drone_config["YCam"] * pxPerM[0], shift[1] - drone_config["XCam"] * pxPerM[1]
            M = np.array([[1.0 / pxPerM[1], 0.0, -shift[1] / pxPerM[1]], [0.0, -1.0 / pxPerM[0], shift[0] / pxPerM[0]], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
                # find IPM as inverse of P*M
            IPMs = []
            for cam in cams:
                IPMs.append(np.linalg.inv(cam.P.dot(M)))
            # setup masks to later clip invalid parts from transformed images (inefficient, but constant runtime)
            for config in cameraConfigs:
                mask = np.zeros((outputRes[0], outputRes[1], 3), dtype=bool)
                for i in range(outputRes[1]):
                    for j in range(outputRes[0]):
                        theta = np.rad2deg(np.arctan2(-j + outputRes[0] / 2 - drone_config["YCam"] * pxPerM[0], i - outputRes[1] / 2 + drone_config["XCam"] * pxPerM[1]))
                        if abs(theta - config["yaw"]) > 90 and abs(theta - config["yaw"]) < 270:
                            mask[j,i,:] = True
                scene_mask.append(mask)
                
        # process images
        progBarWrapper = tqdm(image_paths)
        images = []
        for imgPath in progBarWrapper:
            # load images
            images.append(cv2.imread(imgPath))

        # warp input images
        interpMode = cv2.INTER_NEAREST # cv2.INTER_LINEAR
        warpedImages = []
        for img, IPM in zip(images, IPMs):
            warpedImages.append(cv2.warpPerspective(img, IPM, (outputRes[1], outputRes[0]), flags=interpMode))

        # remove invalid areas (behind the camera) from warped images
        for warpedImg, mask in zip(warpedImages, scene_mask):
            warpedImg[mask] = 0

        # stitch separate images to total bird's-eye-view
        birdsEyeView = np.zeros(warpedImages[0].shape, dtype=np.uint8)
        for warpedImg in warpedImages:
            mask = np.any(warpedImg != (0,0,0), axis=-1)
            birdsEyeView[mask] = warpedImg[mask]
        
        
        Path(scene_path).mkdir(exist_ok=True, parents=False)
        filename = 'ipm_' + str(sample_token) + '.png'
        # display or export bird's-eye-view
        birdsEyeView = cv2.resize(birdsEyeView, (bev['w'], bev['h']))
        # 将图片逆时针旋转90, 使得车头向前
        birdsEyeView = cv2.rotate(birdsEyeView, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(os.path.join(scene_path, filename), birdsEyeView)
        sample_index = sample_index + 1
        sample_token = sample_record['next']
