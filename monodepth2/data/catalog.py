import os

class DatasetCatalog(object):
    DATA_DIR = "datasets"
    DATASETS = {
        "kitti_train": {
            "root": "kitti/kitti_data/",
            "fpath": "kitti/splits/eigen_zhou/train_files.txt"
        },
        "kitti_val": {
            "root": "kitti/kitti_data/",
            "fpath": "kitti/splits/eigen_zhou/val_files.txt"
        },
        "kitti_odom": {

        },
        "kitti_odom": {

        },
        "ts": {
            
        },
    }

    @staticmethod
    def get(name):
        data_dir = DatasetCatalog.DATA_DIR

        if name in ["kitti_train", "kitti_val"]:
            attrs = DatasetCatalog.DATASETS[name]
            attrs["root"] = os.path.join(data_dir, attrs["root"])
            attrs["fpath"] = os.path.join(data_dir, attrs["fpath"])
            return dict(factory="KITTIDataset", args=attrs)

        elif name in ["kitti_odom"]:
            attrs = DatasetCatalog.DATASETS[name]
            return dict(factory="KITTIOdomDataset", args=attrs)
            
        else:
            raise RuntimeError("Dataset not available: {}".format(name))