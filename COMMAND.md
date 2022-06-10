##  view data
```cmd
python3 -m debugpy --wait-for-client --listen 0.0.0.0:8787 scripts/view_data.py data=nuscenes data.dataset_dir=/media/wit/9CACF0B70A8CAE3B/Code/cross_view_transformers/datasets/nuscenes data.labels_dir=/media/wit/9CACF0B70A8CAE3B/Code/cross_view_transformers/datasets/cvt_labels_nuscenes_v2 data.version=v1.0-trainval visualization=nuscenes_viz +split=val
```

## data generate
- visualiz
```cmd
python3 -m debugpy --wait-for-client --listen 0.0.0.0:8787 scripts/generate_data.py \
    data=nuscenes \
    data.version=v1.0-trainval \
    data.dataset_dir=/media/wit/9CACF0B70A8CAE3B/Code/cross_view_transformers/datasets/nuscenes \
    data.labels_dir=/media/wit/9CACF0B70A8CAE3B/Code/cross_view_transformers/datasets \
    visualization=nuscenes_viz
```

- Disable visualizations by omitting the "visualization" flag
```cmd
python3 -m debugpy --wait-for-client --listen 0.0.0.0:8787 scripts/generate_data.py \
    data=nuscenes \
    data.version=v1.0-trainval \
    data.dataset_dir=/media/wit/9CACF0B70A8CAE3B/Code/cross_view_transformers/datasets/nuscenes \
    data.labels_dir=/media/wit/9CACF0B70A8CAE3B/Code/cross_view_transformers/datasets
```