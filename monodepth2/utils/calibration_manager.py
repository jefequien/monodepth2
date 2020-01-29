import pickle

class CalibrationManager:

    def __init__(self, dataset):
        self.dataset = dataset

        with open('datasets/ts/calibrations.pkl', 'rb') as f:
            self.all_calibs = pickle.load(f, encoding='latin1')
    
    def get_cameras(self):
        return self.all_calibs[self.dataset]