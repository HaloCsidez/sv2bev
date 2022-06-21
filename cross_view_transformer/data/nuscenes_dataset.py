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
    transform = SaveDataTransform(labels_dir)

    # Format the split name
    split = f'mini_{split}' if version == 'v1.0-mini' else split
    split_scenes = get_split(split, 'nuscenes')

    result = list()

    for scene_name, scene_record in helper.get_scenes():
        if scene_name not in split_scenes:
            continue

        data = NuScenesDataset(scene_name, scene_record, helper,
                               transform=transform, **dataset_kwargs)
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
        transform=None,
        cameras=[[0, 1, 2, 3, 4, 5]],
        bev={'h': 200, 'w': 200, 'h_meters': 100, 'w_meters': 100, 'offset': 0.0},
    ):
        self.scene_name = scene_name
        self.transform = transform

        self.nusc = helper.nusc
        self.nusc_map = helper.get_map(scene_record['log_token'])

        self.view = get_view_matrix(**bev)
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

        for i in range(len(data)):
            data[i]['extrinsics'] = data[2]['extrinsics']
            
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

        for cam_idx in camera_rig:
            cam_channel = self.CAMERAS[cam_idx]
            cam_token = sample_record['data'][cam_channel]

            cam_record = self.nusc.get('sample_data', cam_token)
            egocam = self.nusc.get('ego_pose', cam_record['ego_pose_token'])
            cam = self.nusc.get('calibrated_sensor', cam_record['calibrated_sensor_token'])

            cam_from_egocam = self.parse_pose(cam, inv=True)
            egocam_from_world = self.parse_pose(egocam, inv=True)

            ## a @ b等同于a.mm(b)或a.matmul(b)
            # E = egocam_from_world
            E = cam_from_egocam @ egocam_from_world @ world_from_egolidarflat
            # E = cam_from_egocam

            I = cam['camera_intrinsic']

            full_path = Path(self.nusc.get_sample_data_path(cam_token))
            image_path = str(full_path.relative_to(self.nusc.dataroot))

            cam_channels.append(cam_channel)
            intrinsics.append(I)
            extrinsics.append(E.tolist())
            images.append(image_path)
        
        # extrinsics = np.array([
        #     [[0.8114373683929443, -0.5841529965400696, 0.018295178189873695, -0.8153076171875], [0.023528752848505974, 0.0013729481725022197, -0.9997221827507019, 1.5125293731689453], [0.5839656591415405, 0.8116422891616821, 0.014858454465866089, -1.176422119140625], [0.0, 0.0, 0.0, 1.0]],
        #     [[-0.007513053715229034, -0.999968945980072, 0.0023763971403241158, 0.0064697265625], [0.018220040947198868, -0.002512961393222213, -0.9998309016227722, 1.523299217224121], [0.9998056888580322, -0.007468481548130512, 0.018238354474306107, -1.5408935546875], [0.0, 0.0, 0.0, 1.0]],
        #     [[-0.838204026222229, -0.5446264743804932, -0.028213750571012497, 0.933013916015625], [0.024072373285889626, 0.014734776690602303, -0.9996016025543213, 1.5064961910247803], [0.5448251962661743, -0.8385492563247681, 0.0007597040385007858, -1.1817626953125], [0.0, 0.0, 0.0, 1.0]],
        #     [[0.9476759433746338, 0.3181700110435486, 0.026040121912956238, -1.1329498291015625], [0.032617829740047455, -0.015362664125859737, -0.99934983253479, 1.5888748168945312], [-0.31756311655044556, 0.9479091763496399, -0.024936843663454056, -0.10693359375], [0.0, 0.0, 0.0, 1.0]],
        #     [[0.00690278597176075, 0.9999622106552124, -0.005292232614010572, 0.00244140625], [0.0071794032119214535, -0.005341780371963978, -0.9999599456787109, 1.579721450805664], [-0.9999503493309021, 0.006864512339234352, -0.007216004654765129, -0.04815673828125], [0.0, 0.0, 0.0, 1.0]],
        #     [[-0.9314492344856262, 0.362310528755188, -0.033664435148239136, 1.052734375], [0.03988364338874817, 0.009697549045085907, -0.9991572499275208, 1.5549945831298828], [-0.36167874932289124, -0.9320070743560791, -0.023483041673898697, -0.09478759765625], [0.0, 0.0, 0.0, 1.0]]
        # ])
            
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
        }

    def get_dynamic_objects(self, sample, annotations):
        h, w = self.bev_shape[:2]

        segmentation = np.zeros((h, w), dtype=np.uint8)
        center_score = np.zeros((h, w), dtype=np.float32)
        center_offset = np.zeros((h, w, 2), dtype=np.float32)
        center_ohw = np.zeros((h, w, 4), dtype=np.float32)
        buf = np.zeros((h, w), dtype=np.uint8)

        visibility = np.full((h, w), 255, dtype=np.uint8)

        coords = np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1).astype(np.float32)

        for ann, p in zip(annotations, self.convert_to_box(sample, annotations)):
            box = p[:2, :4]
            center = p[:2, 4]
            front = p[:2, 5]
            left = p[:2, 6]

            buf.fill(0)
            cv2.fillPoly(buf, [box.round().astype(np.int32).T], 1, INTERPOLATION)
            mask = buf > 0

            if not np.count_nonzero(mask):
                continue

            sigma = 1
            segmentation[mask] = 255
            center_offset[mask] = center[None] - coords[mask]
            center_score[mask] = np.exp(-(center_offset[mask] ** 2).sum(-1) / (sigma ** 2))

            # orientation, h/2, w/2
            center_ohw[mask, 0:2] = ((front - center) / (np.linalg.norm(front - center) + 1e-6))[None]
            center_ohw[mask, 2:3] = np.linalg.norm(front - center)
            center_ohw[mask, 3:4] = np.linalg.norm(left - center)

            visibility[mask] = ann['visibility_token']

        segmentation = np.float32(segmentation[..., None])
        center_score = center_score[..., None]

        result = np.concatenate((segmentation, center_score, center_offset, center_ohw), 2)

        # (h, w, 1 + 1 + 2 + 2)
        return result, visibility

    def convert_to_box(self, sample, annotations):
        # Import here so we don't require nuscenes-devkit unless regenerating labels
        from nuscenes.utils import data_classes

        V = self.view
        M_inv = np.array(sample['pose_inverse'])
        S = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ])

        for a in annotations:
            box = data_classes.Box(a['translation'], a['size'], Quaternion(a['rotation']))

            corners = box.bottom_corners()                                              # 3 4
            center = corners.mean(-1)                                                   # 3
            front = (corners[:, 0] + corners[:, 1]) / 2.0                               # 3
            left = (corners[:, 0] + corners[:, 3]) / 2.0                                # 3

            p = np.concatenate((corners, np.stack((center, front, left), -1)), -1)      # 3 7
            p = np.pad(p, ((0, 1), (0, 0)), constant_values=1.0)                        # 4 7
            p = V @ S @ M_inv @ p                                                       # 3 7

            yield p                                                                     # 3 7

    def get_category_index(self, name, categories):
        """
        human.pedestrian.adult
        """
        tokens = name.split('.')

        for i, category in enumerate(categories):
            if category in tokens:
                return i

        return None

    def get_annotations_by_category(self, sample, categories):
        result = [[] for _ in categories]

        for ann_token in self.nusc.get('sample', sample['token'])['anns']:
            a = self.nusc.get('sample_annotation', ann_token)
            idx = self.get_category_index(a['category_name'], categories)

            if idx is not None:
                result[idx].append(a)

        return result

    def get_line_layers(self, sample, layers, patch_radius=150, thickness=1):
        h, w = self.bev_shape[:2]
        V = self.view
        M_inv = np.array(sample['pose_inverse'])
        S = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ])

        box_coords = (sample['pose'][0][-1] - patch_radius, sample['pose'][1][-1] - patch_radius,
                      sample['pose'][0][-1] + patch_radius, sample['pose'][1][-1] + patch_radius)
        records_in_patch = self.nusc_map.get_records_in_patch(box_coords, layers, 'intersect')

        result = list()

        for layer in layers:
            render = np.zeros((h, w), dtype=np.uint8)

            for r in records_in_patch[layer]:
                polygon_token = self.nusc_map.get(layer, r)
                line = self.nusc_map.extract_line(polygon_token['line_token'])

                p = np.float32(line.xy)                                     # 2 n
                p = np.pad(p, ((0, 1), (0, 0)), constant_values=0.0)        # 3 n
                p = np.pad(p, ((0, 1), (0, 0)), constant_values=1.0)        # 4 n
                p = V @ S @ M_inv @ p                                       # 3 n
                p = p[:2].round().astype(np.int32).T                        # n 2

                cv2.polylines(render, [p], False, 1, thickness=thickness)

            result.append(render)

        return 255 * np.stack(result, -1)

    def get_static_layers(self, sample, layers, patch_radius=150):
        h, w = self.bev_shape[:2]
        V = self.view
        M_inv = np.array(sample['pose_inverse'])
        S = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ])

        box_coords = (sample['pose'][0][-1] - patch_radius, sample['pose'][1][-1] - patch_radius,
                      sample['pose'][0][-1] + patch_radius, sample['pose'][1][-1] + patch_radius)
        records_in_patch = self.nusc_map.get_records_in_patch(box_coords, layers, 'intersect')

        result = list()

        for layer in layers:
            render = np.zeros((h, w), dtype=np.uint8)

            for r in records_in_patch[layer]:
                polygon_token = self.nusc_map.get(layer, r)

                if layer == 'drivable_area': polygon_tokens = polygon_token['polygon_tokens']
                else: polygon_tokens = [polygon_token['polygon_token']]

                for p in polygon_tokens:
                    polygon = self.nusc_map.extract_polygon(p)
                    polygon = MultiPolygon([polygon])

                    exteriors = [np.array(poly.exterior.coords).T for poly in polygon.geoms]
                    exteriors = [np.pad(p, ((0, 1), (0, 0)), constant_values=0.0) for p in exteriors]
                    exteriors = [np.pad(p, ((0, 1), (0, 0)), constant_values=1.0) for p in exteriors]
                    exteriors = [V @ S @ M_inv @ p for p in exteriors]
                    exteriors = [p[:2].round().astype(np.int32).T for p in exteriors]

                    cv2.fillPoly(render, exteriors, 1, INTERPOLATION)

                    interiors = [np.array(pi.coords).T for poly in polygon.geoms for pi in poly.interiors]
                    interiors = [np.pad(p, ((0, 1), (0, 0)), constant_values=0.0) for p in interiors]
                    interiors = [np.pad(p, ((0, 1), (0, 0)), constant_values=1.0) for p in interiors]
                    interiors = [V @ S @ M_inv @ p for p in interiors]
                    interiors = [p[:2].round().astype(np.int32).T for p in interiors]

                    cv2.fillPoly(render, interiors, 0, INTERPOLATION)

            result.append(render)

        return 255 * np.stack(result, -1)

    def get_dynamic_layers(self, sample, anns_by_category):
        h, w = self.bev_shape[:2]
        result = list()

        for anns in anns_by_category:
            render = np.zeros((h, w), dtype=np.uint8)

            for p in self.convert_to_box(sample, anns):
                p = p[:2, :4]

                cv2.fillPoly(render, [p.round().astype(np.int32).T], 1, INTERPOLATION)

            result.append(render)

        return 255 * np.stack(result, -1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Raw annotations
        anns_dynamic = self.get_annotations_by_category(sample, DYNAMIC)
        anns_vehicle = self.get_annotations_by_category(sample, ['vehicle'])[0]

        static = self.get_static_layers(sample, STATIC)                             # 200 200 2
        dividers = self.get_line_layers(sample, DIVIDER)                            # 200 200 2
        dynamic = self.get_dynamic_layers(sample, anns_dynamic)                     # 200 200 8
        bev = np.concatenate((static, dividers, dynamic), -1)                       # 200 200 12

        assert bev.shape[2] == NUM_CLASSES

        # Additional labels for vehicles only.
        aux, visibility = self.get_dynamic_objects(sample, anns_vehicle)

        # Package the data.
        data = Sample(
            view=self.view.tolist(),
            # bev 数据GT
            bev=bev,
            aux=aux,
            visibility=visibility,
            **sample
        )

        if self.transform is not None:
            data = self.transform(data)

        return data
