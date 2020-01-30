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
        "ts_train": {
            "bag_info": (
                "2019-12-17-13-24-03",
                "feature=base&ver=2019121700&base_pt=(32.75707,-111.55757)&end_pt=(32.092537212,-110.7892506)",
                "0:13:00",
                "0:43:00",
            )
        },
        "ts_val": {
            "bag_info": (
                "2019-12-17-13-24-03",
                "feature=base&ver=2019121700&base_pt=(32.75707,-111.55757)&end_pt=(32.092537212,-110.7892506)",
                "0:43:00",
                "0:46:00",
            )
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

        if name in ["ts_train", "ts_val"]:
            attrs = DatasetCatalog.DATASETS[name]
            return dict(factory="TSDataset", args=attrs)
            
        else:
            raise RuntimeError("Dataset not available: {}".format(name))