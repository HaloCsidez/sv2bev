import torch
import numpy as np
import cv2

from pathlib import Path
from functools import lru_cache

from pyquaternion import Quaternion
from shapely.geometry import MultiPolygon

from .common import INTERPOLATION, get_view_matrix, get_pose, get_split
from .transforms import Sample, SaveDataTransform
import pdb

STATIC = ['lane', 'road_segment']
DIVIDER = ['road_divider', 'lane_divider']
DYNAMIC = [
    'car', 'truck', 'bus',
    'trailer', 'construction',
    'pedestrian',
    'motorcycle', 'bicycle',
]

CLASSES = STATIC + DIVIDER + DYNAMIC
NUM_CLASSES = len(CLASSES)


def get_data(
    dataset_dir,
    labels_dir,
    split,
    version,
    dataset='unused',                   # ignore
    augment='unused',                   # ignore
    image='unused',                     # ignore
    label_indices='unused',             # ignore
    num_classes=NUM_CLASSES,            # in here to make config consistent
    **dataset_kwargs
):
    assert num_classes == NUM_CLASSES

    helper = NuScenesSingleton(dataset_dir, version)
    # transform = SaveDataTransform(labels_dir)

    # Format the split name
    split = f'mini_{split}' if version == 'v1.0-mini' else split
    split_scenes = get_split(split, 'nuscenes')

    result = list()

    for scene_name, scene_record in helper.get_scenes():
        if scene_name not in split_scenes:
            continue

        data = NuScenesDataset(scene_name, scene_record, helper, **dataset_kwargs)
        result.append(data)

    return result


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
        cameras=[[0, 1, 2, 3, 4, 5]],
        bev={'h': 200, 'w': 200, 'h_meters': 100, 'w_meters': 100, 'offset': 0.0},
    ):
        self.scene_name = scene_name
        self.nusc = helper.nusc
        self.nusc_map = helper.get_map(scene_record['log_token'])
        self.bev_shape = (bev['h'], bev['w'])
        self.samples = self.parse_scene(scene_record, cameras)

    def parse_scene(self, scene_record, camera_rigs):
        data = []
        sample_token = scene_record['first_sample_token']

        while sample_token:
            sample_record = self.nusc.get('sample', sample_token)

            for camera_rig in camera_rigs:
                data.append(self.parse_sample_record(sample_record, camera_rig))

            sample_token = sample_record['next']

        return data

    def parse_pose(self, record, *args, **kwargs):
        return get_pose(record['rotation'], record['translation'], *args, **kwargs)

    def parse_sample_record(self, sample_record, camera_rig):
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

        return {
            'scene': self.scene_name,
            'token': sample_record['token'],

            'pose': world_from_egolidarflat.tolist(),
            'pose_inverse': egolidarflat_from_world.tolist(),

            'cam_ids': list(camera_rig),
            'cam_channels': cam_channels,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'images': images,
            'translation': translation,
            'rotation': rotation,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = Sample(
            view=None,
            bev=None,
            **sample
        )

        return data
