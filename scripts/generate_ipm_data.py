import torch
import json
import hydra
import numpy as np
import cv2

from pathlib import Path
from tqdm import tqdm

from cross_view_transformer.common import setup_config, setup_data_module, setup_viz
from cross_view_transformer.ipm import ipm
import os

def ipm_processor(batch, cfg_data):
    view_num = len(batch['images'])
    tmp = [batch['intrinsics'], batch['extrinsics']]
    camera_config = list(map(list, zip(*tmp))) # transpose ([[i1,i2],[e1,e2]] -> [[i1,e1],[i2,e2]])
    drone_config = {
        'fx': 1266.417203046554, 'fy': 1266.417203046554, 
        'px': 816.2670197447984, 'py': 491.50706579294757, 
        'yaw': 0.0, 'pitch': 90.0, 'roll': 90.0, 
        'XCam': 0, 'YCam': 0, 'ZCam': 80
        }
    ipm(view_num, batch['images'], camera_config, drone_config, 
        os.path.join(str(cfg_data['labels_dir']).replace('cvt_labels_nuscenes', 'ipm_gt_nuscenes'), 
                     str(batch['bev']).replace('bev', 'ipm')),
        cfg_data['dataset_dir']
        )

def setup(cfg):
    print('修改config中的配置项,改用nuscenes_ipm_generated进行数据生成')
    cfg.data.dataset = cfg.data.dataset.replace('nuscenes_generated', 'nuscenes_ipm')

    cfg.data.augment = 'none'
    cfg.loader.batch_size = 1
    cfg.loader.persistent_workers = True
    cfg.loader.drop_last = False
    cfg.loader.shuffle = False


@hydra.main(config_path=Path.cwd() / 'config', config_name='config.yaml')
def main(cfg):
    setup_config(cfg, setup)
    data = setup_data_module(cfg)

    labels_dir = Path(cfg.data.labels_dir)
    labels_dir.mkdir(parents=False, exist_ok=True)

    for split in ['train', 'val']:
        print(f'Generating split: {split}')

        for episode in tqdm(data.get_split(split, loader=False), position=0, leave=False):
            scene_dir = labels_dir / episode.scene_name
            scene_dir.mkdir(exist_ok=True, parents=False)

            loader = torch.utils.data.DataLoader(episode, collate_fn=list, **cfg.loader)
            info = []

            for i, batchs in enumerate(tqdm(loader, position=1, leave=False)):
                # batch中包含所需的视角信息,能够用于ipm合成,且尺寸可以为原石尺寸
                info.extend(batchs)
                
                for batch in batchs:
                    # 处理一组图片(6个视角为一组)
                    ipm_processor(batch, cfg['data'])
                        

                # Load data from disk to test if it was saved correctly
                # if i == 0 and viz_fn is not None:
                #     unbatched = [load_xform(s) for s in batchs]
                #     rebatched = torch.utils.data.dataloader.default_collate(unbatched)

                #     viz = np.vstack(viz_fn(rebatched))

                #     cv2.imshow('debug', cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
                #     cv2.waitKey(1)

            # Write all info for loading to json
            scene_json = labels_dir / f'{episode.scene_name}.json'
            scene_json.write_text(json.dumps(info))


if __name__ == '__main__':
    main()
