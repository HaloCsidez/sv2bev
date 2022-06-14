import torch
import json
import hydra
import cv2
import numpy as np

from pathlib import Path
from tqdm import tqdm

from cross_view_transformer.data.transforms import LoadDataTransform
from cross_view_transformer.common import setup_config, setup_data_module, setup_viz
from cross_view_transformer.ipm import ipm


def setup(cfg):
    # Don't change these
    cfg.data.dataset = cfg.data.dataset.replace('_generated', '')
    cfg.data.augment = 'none'
    cfg.loader.batch_size = 1
    cfg.loader.persistent_workers = True
    cfg.loader.drop_last = False
    cfg.loader.shuffle = False

    # Uncomment to debug errors hidden by multiprocessing
    # cfg.loader.num_workers = 0
    # cfg.loader.prefetch_factor = 2
    # cfg.loader.persistent_workers = False

def ipm_processor(batch, label_dir):
    view_num = len(batch['images'])
    tmp = [batch['intrinsics'], batch['extrinsics']]
    camera_config = list(map(list, zip(*tmp))) # transpose ([[i1,i2],[e1,e2]] -> [[i1,e1],[i2,e2]])
    drone_config = {
        'fx': 1266.417203046554, 'fy': 1266.417203046554, 
        'px': 816.2670197447984, 'py': 491.50706579294757, 
        'yaw': 0.0, 'pitch': 90.0, 'roll': 90.0, 
        'XCam': 0, 'YCam': 0, 'ZCam': 8
        }
    ipm(view_num, batch['images'], camera_config, drone_config, str(label_dir).replace('cvt_labels_nuscenes', 'ipm_gt_nuscenes'))


@hydra.main(config_path=Path.cwd() / 'config', config_name='config.yaml')
def main(cfg):
    """
    Creates the following dataset structure

    cfg.data.labels_dir/
        01234.json
        01234/
            bev_0001.png
            bev_0002.png
            ...

    If the 'visualization' flag is passed in,
    the generated data will be loaded from disk and shown on screen
    """
    setup_config(cfg, setup)

    data = setup_data_module(cfg)
    viz_fn = None

    if 'visualization' in cfg:
        viz_fn = setup_viz(cfg)
        load_xform = LoadDataTransform(cfg.data.dataset_dir, cfg.data.labels_dir,
                                       cfg.data.image, cfg.data.num_classes)

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
                    ipm_processor(batch)
                        

                # Load data from disk to test if it was saved correctly
                if i == 0 and viz_fn is not None:
                    unbatched = [load_xform(s) for s in batch]
                    rebatched = torch.utils.data.dataloader.default_collate(unbatched)

                    viz = np.vstack(viz_fn(rebatched))

                    cv2.imshow('debug', cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)

            # Write all info for loading to json
            scene_json = labels_dir / f'{episode.scene_name}.json'
            scene_json.write_text(json.dumps(info))


if __name__ == '__main__':
    main()
