


from __future__ import absolute_import, division, print_function

import os
import time
import datetime
import skimage.transform
import numpy as np
from PIL import Image
from io import BytesIO

import torch
import torch.utils.data as data
from torchvision import transforms

from .mono_dataset import MonoDataset

from dataset_store import Dataset

class TSDataset(data.Dataset):
    """Superclass for loading different types of TuSimple datasets
    """
    def __init__(self, bag_name, begin, end):
        super(TSDataset, self).__init__()

        self.bag_name = bag_name
        self.begin = parse_time(begin)
        self.end = parse_time(end)

        self.ds = Dataset.open(self.bag_name, ver=None)

        self.sample_freq = 10 # frames per sec
    
    def __len__(self):
        duration = self.end - self.begin
        return int(duration.total_seconds() * self.sample_freq)

    def __getitem__(self, index):
        if index >= self.__len__():
            raise IndexError

        img = self.get_color(index)
        return img


    def get_color(self, frame_index):
        td = self.begin + datetime.timedelta(seconds=frame_index / self.sample_freq)
        ts, camera = self.ds.fetch_near('/camera1/image_color/compressed', str(td), limit=1)[0]
        img = Image.open((BytesIO(camera.data)))
        return img

def parse_time(ts):
    t = time.strptime(ts, "%H:%M:%S")
    td = datetime.timedelta(hours=t.tm_hour, minutes=t.tm_min, seconds=t.tm_sec)
    return td
