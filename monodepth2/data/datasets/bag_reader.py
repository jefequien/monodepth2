import datetime
import numpy as np
from PIL import Image
from io import BytesIO

from dataset_store import Dataset
from tsmap3 import TSMap, GNSSTransformer
from localization_service import OfflineNovAtelService

from monodepth2.utils.calibration_manager import CalibrationManager
from ..maps.map_viewer import MapViewer, MapCamera

class BagReader(object):

    def __init__(self, bag_info):
        bag_name, map_name, begin, end, version = bag_info
        self.bag_name = bag_name
        self.map_name = map_name
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
            print('Topic not found: {}'.format(topic))
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
            self.add_topic('map_pred{}'.format(cam_id), '/lane_detection/camera{}'.format(cam_id))

        
        # Map View
        self.calib_manager = CalibrationManager(dataset=self.bag_name)
        self.camera_calibs = self.calib_manager.get_cameras()
        self.map_viewer = MapViewer(self.map_name)
        self.map_cameras = {
            'cam{}'.format(cam_id): MapCamera(self.camera_calibs[cam_id]) 
            for cam_id in self.camera_calibs.keys()
        }
    
    def postprocess(self, raw_data):
        data = super(CameraBagReader, self).postprocess(raw_data)
        for cam_id in self.cam_ids:
            cam_name = 'cam{}'.format(cam_id)
            _, camera = raw_data[cam_name]
            img = Image.open(BytesIO(camera.data))
            data[cam_name] = img.resize((512, 288))

            # Parse lanes            
            _, r_map_data = raw_data['map_pred{}'.format(cam_id)]
            rmap_list = translate_rmap_data(r_map_data)
            r_map = rmap_list[0]

            lane_dets = np.zeros((r_map.shape[0], r_map.shape[1], 3), dtype=np.uint8)
            lane_dets[r_map == 1] = [0, 255, 0] # white marker
            lane_dets[r_map == 2] = [0, 0, 255] # transparent
            lane_dets[r_map == 5] = [255, 0, 0] # curb
            lane_dets = Image.fromarray(lane_dets)
            data['map_pred/cam{}'.format(cam_id)] = lane_dets.resize((512, 288))
            data['map_view/cam{}'.format(cam_id)] = self.create_map_view(data, cam_id)
        return data

    def create_map_view(self, data, cam_id):
        cam_name = 'cam{}'.format(cam_id)
        gps_data = data['gps_data']
        self.map_cameras[cam_name].set_position(gps_data)
        map_view, depth = self.map_viewer.get_view(self.map_cameras[cam_name])
        map_view = map_view.resize((512, 288))
        return map_view

def translate_rmap_data(lane_det_msg):
    rmap_list = []

    rmap_pips = np.frombuffer(lane_det_msg.data, dtype=np.uint8)
    num_pips = int(rmap_pips.shape[0] / lane_det_msg.height / lane_det_msg.width)
    rmap_pips = rmap_pips.reshape((num_pips, lane_det_msg.height, lane_det_msg.width))
    for i in range(num_pips):
        rmap = rmap_pips[i, :, :]
        rmap = rmap.copy()
        rmap_list.append(rmap)
    return rmap_list