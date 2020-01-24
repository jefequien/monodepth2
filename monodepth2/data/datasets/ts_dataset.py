
import time
import datetime
from PIL import Image
from io import BytesIO

from dataset_store import Dataset

from .synced_dataset import SyncedDataset

class TSDataset(SyncedDataset):
    
    def __init__(self, bag_name, begin, end, data_ids=None, transform=None):
        super(TSDataset, self).__init__(data_ids=data_ids, transform=transform)
        self.bag_name = bag_name
        self.begin = parse_time(begin)
        self.end = parse_time(end)
        self.sample_freq = 10

        self.ds = Dataset.open(self.bag_name, ver=None)

        self.topics = {
            'main': '/camera1/image_color/compressed',
            'stereo': '/camera3/image_color/compressed',
            'gps': ''
        }
        
    def __len__(self):
        duration = self.end - self.begin
        return int(duration.total_seconds() * self.sample_freq)
    
    def get_image(self, cam_id, index, shift=0):
        """ 
        Arguments:
            index: index in dataset
            cam_id: camera id
        """
        td = self.begin + datetime.timedelta(seconds=index + shift / self.sample_freq)
        _, camera = self.ds.fetch_near(self.topics[cam_id], str(td), limit=1)[0]

        img = Image.open((BytesIO(camera.data)))
        return img
        
    def get_depth(self, index):
        raise NotImplementedError

    def get_gps(self, index):
        raise NotImplementedError

def parse_time(ts):
    t = time.strptime(ts, "%H:%M:%S")
    td = datetime.timedelta(hours=t.tm_hour, minutes=t.tm_min, seconds=t.tm_sec)
    return td