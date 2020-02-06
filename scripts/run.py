import argparse
import os

from monodepth2.config import cfg
from monodepth2.utils.logger import setup_logger
from monodepth2.utils.miscellaneous import mkdir, save_config
from monodepth2.data.datasets.bag_reader import CameraBagReader

from localization import LocalizationModel

def setup(args):
    cfg.merge_from_file(args.config_file)

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("monodepth2", output_dir)
    logger.info("Running with config:\n{}".format(cfg))

    # Save overloaded model config in the output directory
    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    save_config(cfg, output_config_path)
    return cfg




def main(args):
    cfg = setup(args)
    bag_name = '2019-12-17-13-24-03'
    map_name = 'feature=base&ver=2019121700&base_pt=(32.75707,-111.55757)&end_pt=(32.092537212,-110.7892506)'
    begin = '0:43:00'
    end = '0:45:00'
    version = None

    # bag_name = '2020-01-13-09-35-05'
    # map_name = 'feature=firebird&ver=2020011600&base_pt=(32.75707,-111.55757)&end_pt=(32.092537,-110.7892506)'
    # begin, end = '02:10:20', '02:14:50'
    # # begin, end = '02:12:00', '02:14:50'
    # version = 'R20200120234009'
    bag_info = bag_name, map_name, begin, end, version

    gps_noise = cfg.INPUT.GPS_NOISE
    bag_reader = CameraBagReader(bag_info, gps_noise)

    localization_model = LocalizationModel(cfg)
    localization_model.setup(bag_info)

    for _, data in enumerate(bag_reader):
        localization_model.step(data)

    localization_model.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training")
    parser.add_argument('--config-file',
                        type=str, help='path to a test image or folder of images',
                        default='configs/ts.yaml')
    args = parser.parse_args()

    main(args)