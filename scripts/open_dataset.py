import argparse
import os

from monodepth2.config import cfg
from monodepth2.data import build_dataset


def main(args):
    cfg.merge_from_file(args.config_file)

    dataset_names = cfg.DATASETS.TRAIN
    data_ids = cfg.INPUT.FRAME_IDS + ['stereo']
    dataset = build_dataset(dataset_names, data_ids, transform=None)

    for _, data in enumerate(dataset):
        for k, v in data.items():
            print(k)
            v.show()
        break



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training")
    parser.add_argument('--config-file',
                        type=str,help='path to a test image or folder of images',
                        default='configs/first.yaml')
    args = parser.parse_args()

    main(args)