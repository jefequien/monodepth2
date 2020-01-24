from __future__ import absolute_import, division, print_function

import argparse
import os

from monodepth2.config import cfg
from monodepth2.utils.logger import setup_logger
from monodepth2.utils.miscellaneous import mkdir, save_config
from monodepth2.trainer import Trainer



def main(args):
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

    trainer = Trainer(cfg)
    if args.resume:
        trainer.load_model()
    trainer.train()



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training")
    parser.add_argument('--config-file',
                        type=str,help='path to a test image or folder of images',
                        default='configs/first.yaml')
    parser.add_argument("--resume",
                        help="resume training",
                        action="store_true")
    args = parser.parse_args()

    main(args)
