import os

from monodepth2.model import MonodepthModel


class LocalizationModel:

    def __init__(self, cfg):

        # Model
        self.model = MonodepthModel(cfg)
        self.model.load_model()
        self.model.set_eval()

        # Localization params
        self.tsmap = None
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
        data[1] = obs['cam1']
        data[-1] = obs['cam1']

        pred = self.model.predict(data)
        T = pred['T']

        return pred
