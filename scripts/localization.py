import os

from monodepth2.model import MonodepthModel
from monodepth2.utils.visualize import vis_depth


class LocalizationModel:

    def __init__(self, cfg):
        self.output_dir = cfg.OUTPUT_DIR

        # Model
        save_folder = os.path.join(self.output_dir, "models", "latest_weights")
        self.model = MonodepthModel(cfg)
        self.model.load_model(save_folder)
        self.model.set_eval()

        # Localization params
        self.last_obs = {}
        self.initialized = False
    
    def setup(self, tsmap, bag_name):
        self.tsmap = tsmap
    
    def initialize(self, obs):
        self.initialized = True

    def step(self, obs):
        if not self.initialized:
            self.last_obs = obs
            self.initialized = True

        data = {}
        data[0] = obs['cam1']
        data[-1] = self.last_obs['cam1']

        pred = self.model.predict(data)

        T = pred['T']
        depth_img = vis_depth(pred['depth'])
        depth_img.show()
        raise

        self.last_obs = obs
        return pred
