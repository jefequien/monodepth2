import os

import torch.utils.data as data

class SyncedDataset(data.Dataset):
    """ A dataset that loads timestamp synced data.
    """

    def __init__(self, data_ids, transform=None):
        self.data_ids = data_ids
        self.transform = transform

    def __getitem__(self, index):
        data = {}
        
        for data_id in self.data_ids:
            
            if isinstance(data_id, int):
                # Get time shifted image from cam1
                cam_name = 'cam1'
                data[data_id] = self.get_image(cam_name, index, shift=data_id)
                data[(data_id, 'calib')] = self.get_calibration(cam_name)

            elif 'cam' in data_id:
                # Get image for other cameras
                cam_name = data_id
                data[data_id] = self.get_image(cam_name, index, shift=0)
                data[(data_id, 'calib')] = self.get_calibration(cam_name)
            
            elif data_id == "gps":
                data[data_id] = self.get_gps(index)

            elif data_id == "depth":
                data[data_id] = self.get_depth(index)

            else:
                raise Exception("Data id not recognized: {}".format(data_id))
        
        if self.transform:
            data = self.transform(data)
        return data
    
    def __len__(self):
        raise NotImplementedError
    
    def get_image(self, cam_id, index, shift=0):
        raise NotImplementedError

    def get_calibration(self, cam_id):
        raise NotImplementedError

    def get_depth(self, index):
        raise NotImplementedError

    def get_gps(self, index):
        raise NotImplementedError


