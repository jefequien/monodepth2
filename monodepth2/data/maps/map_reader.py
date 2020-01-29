import numpy as np

from tsmap3 import TSMap, Point3d, LaneSequence, Lane, Roadmarker

SOLID_LINE_TYPES = {Roadmarker.Type.WHITE_SOLID_LINE, Roadmarker.Type.WHITE_DASHED_LINE,
                    Roadmarker.Type.YELLOW_SOLID_LINE, Roadmarker.Type.YELLOW_DASHED_LINE}
DASH_BLOB_TYPES = {Roadmarker.Type.WHITE_DASHED_LINE_BLOB, Roadmarker.Type.YELLOW_DASHED_LINE_BLOB}

class MapReader:

    def __init__(self, map_name):
        self.tsmap = TSMap(map_name)
        self.front_limit = 200

    def get_landmarks(self, enu_coords):
        """ Returns a Nx3 array of ENU coordinates.
        """
        landmarks = []
        
        base = Point3d(*enu_coords[:3])
        self.tsmap.update_car_pos(*enu_coords[:2], async=False)

        # Get lanes
        roadmarkers = self.tsmap.get_visible_lines_by_range(base, self.front_limit, False)
        for marker in roadmarkers:
            if marker.type in SOLID_LINE_TYPES:
                lane = [list(p) for p in marker.pts[0]]
                landmarks.extend(lane)
            elif marker.type in DASH_BLOB_TYPES:
                lane = [list(p) for dash in marker.pts for p in dash]
                landmarks.extend(lane)
            else:
                continue
        
        if len(landmarks) == 0:
            print("Warning: No landmarks!")
        
        return np.array(landmarks)
