import datetime
import numpy as np
from PIL import Image
from io import BytesIO

from dataset_store import Dataset

class BagPlayer(object):

    def __init__(self, bag_info):
        bag_name, map_name, begin, end = bag_info
        version = None

        self.bag_name = bag_name
        self.map_name = map_name
        self.ds = Dataset.open(bag_name, ver=version)

        self.topics = {'cam1': '/camera1/image_color/compressed'}

        # Parse time at the end
        self.begin = self.parse_time(0 if begin is None else begin)
        self.end = self.parse_time('-0' if end is None else end)
        self.sample_freq = 20 # in FPS
        
    def __getitem__(self, index):
        ts = self.index_to_ts(index)

        keys, topic_list = zip(*self.topics.items())
        all_topic_data = list(self.ds.fetch_aligned(
            *topic_list,
            ts_begin=ts,
            limit=1
        ))[0]

        data = {}
        for k, topic_data in zip(keys, all_topic_data):
            data[k] = topic_data
        data = self.postprocess(data)
        return data
    
    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __len__(self):
        duration = self.end - self.begin
        return int(duration / 1e9 * self.sample_freq)
    
    def postprocess(self, raw_data):
        return raw_data

    def topic_exists(self, topic):
        try:
            self.ds.get_topic(topic)
            return True
        except KeyError as e:
            # print('Topic not found: {}'.format(topic))
            return False
    
    def index_to_ts(self, index):
        shift = 1. * index / self.sample_freq * 1e9
        ts = int(self.begin + shift)
        return ts

    def parse_time(self, t):
        ts, _ = self.ds.fetch_near(self.topics['cam1'], t, limit=1)[0]
        return ts


class CameraBagPlayer(BagPlayer):

    def __init__(self, bag_info):
        super(CameraBagPlayer, self).__init__(bag_info)

        self.cam_ids = [1,3]
        self.topics = {
            'cam{}'.format(cam_id): '/camera{}/image_color/compressed'.format(cam_id)
            for cam_id in self.cam_ids
        }
    
    def postprocess(self, raw_data):
        data = {}
        for k,v in raw_data.items():
            _, camera = v
            data[k] = Image.open(BytesIO(camera.data))
        return data


# class PFBagDataset(BagDataset):

#     def __init__(self, bag_info):
#         super(PFBagDataset, self).__init__(bag_info)

#         self.cam_ids = [1,3]
#         self.sample_freq = 20 # in FPS

#         self.topics = {
#             'pvax': '/novatel_data/inspvax',
#             'spd': '/novatel_data/insspd',
#             'twist': '/vehicle/twist',
#         }
#         for cam_id in self.cam_ids:
#             self.topics['camera{}'.format(cam_id)] = '/camera{}/image_color/compressed'.format(cam_id)
#             self.topics['deep{}-0'.format(cam_id)] = '/lane_detection/camera{}'.format(cam_id)
#             self.topics['deep{}-1'.format(cam_id)] = '/lane_detection/camera{}/pip1'.format(cam_id)
#             self.topics['pip{}-0'.format(cam_id)] = '/pip/cam{}/pip0'.format(cam_id)
#             self.topics['pip{}-1'.format(cam_id)] = '/pip/cam{}/pip1'.format(cam_id)
#         self.topics = {k: v for k,v in self.topics.items() if self.topic_exists(v)}
        
#         tsmap = TSMap(self.map_name)
#         map_base = GNSSTransformer.get_instance().get_base()
#         # Setup GPS service for parsing
#         novatel_srv_cfg = {
#             'bag_name': self.bag_name, 
#             'trans': {
#                 'enu': {
#                     'base_lat': map_base[0],
#                     'base_lon': map_base[1],
#                 }
#             }
#         }
#         self.novatel_service = OfflineNovAtelService(novatel_srv_cfg)
    
#     def postprocess(self, data):
#         from lps_test.common import get_vision_info
#         # Parse vision data
#         lane_detector = None
#         deep_version = 2
#         vision_info, loc_ts = get_vision_info(data, self.cam_ids,
#                                                 pip_pairs=[], lane_detector=lane_detector, version=deep_version)

#         # Parse GPS signal
#         novatel_loc, result_status = self.novatel_service.get_exact(loc_ts.stamp.to_nsec())
#         if result_status == 0:
#             raise Exception("GPS parsing failed")

#         gps_data = np.array([novatel_loc.pose.position.x,
#                                 novatel_loc.pose.position.y,
#                                 novatel_loc.pose.position.z,
#                                 novatel_loc.pose.orientation.x,
#                                 novatel_loc.pose.orientation.y,
#                                 novatel_loc.pose.orientation.z,
#                                 1.]).reshape(1, -1)

#         obs = {}
#         obs['gps_data'] = gps_data
#         obs['vision_info'] = vision_info
#         return obs

        
    