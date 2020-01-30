import datetime
import numpy as np
from PIL import Image
from io import BytesIO

from dataset_store import Dataset
from tsmap3 import TSMap, GNSSTransformer
from localization_service import OfflineNovAtelService

class BagReader(object):

    def __init__(self, bag_info):
        bag_name, map_name, begin, end = bag_info
        version = None
        self.bag_info = bag_info
        self.ds = Dataset.open(bag_name, ver=version)

        self.topics = {
            'pvax': '/novatel_data/inspvax',
            'spd': '/novatel_data/insspd',
            'twist': '/vehicle/twist',
            'cam1': '/camera1/image_color/compressed',
        }

        # Setup GPS service for parsing
        tsmap = TSMap(map_name)
        map_base = GNSSTransformer.get_instance().get_base()
        novatel_srv_cfg = {
            'bag_name': bag_name, 
            'trans': {
                'enu': {
                    'base_lat': map_base[0],
                    'base_lon': map_base[1],
                }
            }
        }
        self.novatel_service = OfflineNovAtelService(novatel_srv_cfg)

        # Parse time at the end
        self.begin = self.parse_time(0 if begin is None else begin)
        self.end = self.parse_time('-0' if end is None else end)
        self.sample_freq = 10 # in FPS
    
    def __iter__(self):
        keys, topic_list = zip(*self.topics.items())

        last_idx = -1
        n = self.__len__()
        for all_topic_data in self.ds.fetch_aligned(*topic_list, ts_begin=self.begin, ts_end=self.end):
            ts = all_topic_data[0][0]
            idx = self.ts_to_idx(ts)

            if idx == last_idx: # Skip
                continue
            last_idx = idx

            if idx == n: # Done
                break
            
            data = {k: t_data for k, t_data in zip(keys, all_topic_data)}
            data = self.postprocess(data)
            yield data
    
    def postprocess(self, raw_data):
        # Parse GPS signal
        loc_ts = raw_data['cam1'][1].header
        novatel_loc, result_status = self.novatel_service.get_exact(loc_ts.stamp.to_nsec())
        if result_status == 0:
            raise Exception("GPS parsing failed")

        gps_data = np.array([
            novatel_loc.pose.position.x,
            novatel_loc.pose.position.y,
            novatel_loc.pose.position.z,
            novatel_loc.pose.orientation.x,
            novatel_loc.pose.orientation.y,
            novatel_loc.pose.orientation.z,
        ])
        
        data = {}
        data['gps_data'] = gps_data
        return data

    def add_topic(self, name, topic):
        if self.topic_exists(topic):
            self.topics[name] = topic

    def topic_exists(self, topic):
        try:
            self.ds.get_topic(topic)
            return True
        except KeyError as e:
            # print('Topic not found: {}'.format(topic))
            return False
    
    def __len__(self):
        return self.ts_to_idx(self.end)

    def idx_to_ts(self, idx):
        shift = 1. * idx / self.sample_freq * 1e9
        ts = int(self.begin + shift)
        return ts
    
    def ts_to_idx(self, ts):
        idx = (ts - self.begin) / 1e9 * self.sample_freq
        return int(idx)

    def parse_time(self, t):
        ts, _ = self.ds.fetch_near(self.topics['cam1'], t, limit=1)[0]
        return ts


class CameraBagReader(BagReader):

    def __init__(self, bag_info):
        super(CameraBagReader, self).__init__(bag_info)
        self.cam_ids = [1,3]
        for cam_id in self.cam_ids:
            self.add_topic('cam{}'.format(cam_id), '/camera{}/image_color/compressed'.format(cam_id))
    
    def postprocess(self, raw_data):
        data = super(CameraBagReader, self).postprocess(raw_data)
        for cam_id in self.cam_ids:
            cam_name = 'cam{}'.format(cam_id)
            _, camera = raw_data[cam_name]
            img = Image.open(BytesIO(camera.data))
            data[cam_name] = img
        return data

