# %%writefile /kaggle/working/MaskFreeVIS/mask2former_video/data_video/datasets/builtin.py
# %load /kaggle/working/MaskFreeVIS/mask2former_video/data_video/datasets/builtin.py
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

import os

from .ytvis import (
    register_ytvis_instances,
    _get_ytvis_2019_instances_meta,
    _get_ytvis_2021_instances_meta,
)

from detectron2.data.datasets.coco import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata

_PREDEFINED_SPLITS_COCO = {}
_PREDEFINED_SPLITS_COCO["coco"] = {
    "coco_2017_train_fake": ("/kaggle/input/coco-2017-dataset/coco2017/train2017", "/kaggle/input/jsonfarnoosh/jsonfarnoosh/_home_user01_MaskFreeVIS_coco2ytvis2019_train.json"),
}

# ==== Predefined splits for YTVIS 2019 ===========
_PREDEFINED_SPLITS_YTVIS_2019 = {
    "ytvis_2019_train": ("/kaggle/input/train-vis/train/JPEGImages",
                         "/kaggle/input/jsonfarnoosh/jsonfarnoosh/train2019.json"), 
    "ytvis_2019_val": ("/kaggle/input/vis2021-new-mob-45/vis2021_45",
                       "/kaggle/input/valid21-794-m2f/_home_user01_MaskFreeVIS_train_instances_vis2021_forvalid_794.json"),
    "ytvis_2019_test": ("ytvis_2019/test/JPEGImages",
                        "ytvis_2019/test.json"),
}


# ==== Predefined splits for YTVIS 2021 ===========
_PREDEFINED_SPLITS_YTVIS_2021 = {
    "ytvis_2021_train": ("/kaggle/input/vis2021/train/train/JPEGImages",
                         "/kaggle/input/vis2021/train/train/instances.json"),
    "ytvis_2021_val": ("/kaggle/input/vis2021/valid/valid/JPEGImages",
                       "/kaggle/input/vis2021/valid/valid/instances.json"),
    "ytvis_2021_test": ("ytvis_2021/test/JPEGImages",
                        "ytvis_2021/test.json"),
}


def register_all_ytvis_2019(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2019.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2019_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_ytvis_2021(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2021.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2021_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

def register_all_coco(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_ytvis_2019(_root)
    register_all_ytvis_2021(_root)
    register_all_coco(_root)
