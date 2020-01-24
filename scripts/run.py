import argparse
import os

from monodepth2.config import cfg
from monodepth2.runner import LocalizationModel
from monodepth2.data.datasets import TSDataset


def main(args):
    cfg.merge_from_file(args.config_file)

    bag_name = '2019-12-17-13-24-03'
    begin = '0:36:00'
    end = '0:36:10'
    data_ids = [0,-1,1]
    dataset = TSDataset(bag_name, begin, end, data_ids)

    
    
    map_name = 'feature=base&ver=2019121700&base_pt=(32.75707,-111.55757)&end_pt=(32.092537212,-110.7892506)'
    model = LocalizationModel(cfg)
    model.load_models(args.save_folder)

    for _, data in enumerate(dataset):
        model.step(data)
        break



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training")
    parser.add_argument('--config-file',
                        type=str, help='path to a test image or folder of images',
                        default='configs/first.yaml')
    parser.add_argument('--save-folder',
                        type=str, help='',
                        required=True)
    args = parser.parse_args()

    main(args)