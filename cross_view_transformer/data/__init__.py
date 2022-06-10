from . import nuscenes_dataset
from . import nuscenes_dataset_generated
from . import nuscenes_ipm_generated


MODULES = {
    'nuscenes': nuscenes_dataset,
    'nuscenes_generated': nuscenes_dataset_generated,
    'nuscenes_ipm': nuscenes_ipm_generated
}


def get_dataset_module_by_name(name):
    return MODULES[name]
