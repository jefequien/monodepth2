import numpy as np
import time

from tsmap3 import TSMap, Point3d, LaneSequence, Lane, Roadmarker

SOLID_LINE_TYPES = {Roadmarker.Type.WHITE_SOLID_LINE, Roadmarker.Type.WHITE_DASHED_LINE,
                    Roadmarker.Type.YELLOW_SOLID_LINE, Roadmarker.Type.YELLOW_DASHED_LINE}
DASH_BLOB_TYPES = {Roadmarker.Type.WHITE_DASHED_LINE_BLOB, Roadmarker.Type.YELLOW_DASHED_LINE_BLOB}

class MapReader:

    def __init__(self, map_name):
        self.tsmap = TSMap(map_name)
        self.front_limit = 50
        self.no_oncoming = True
    
    def get_solid_lines(self, enu_coords):
        base = Point3d(*enu_coords[:3])
        self.tsmap.update_car_pos(*enu_coords[:2], async=False)

        roadmarkers = self.tsmap.get_visible_lines_by_range(base, self.front_limit, self.no_oncoming)

        lines = []
        for marker in roadmarkers:
            if marker.type in SOLID_LINE_TYPES:
                line = [list(p) for p in marker.pts[0]]
                line = np.array(line)
                lines.append(line)
        return lines
    
    def get_dash_lines(self, enu_coords):
        base = Point3d(*enu_coords[:3])
        self.tsmap.update_car_pos(*enu_coords[:2], async=False)

        roadmarkers = self.tsmap.get_visible_lines_by_range(base, self.front_limit, self.no_oncoming)

        dashes = []
        for marker in roadmarkers:
            if marker.type in DASH_BLOB_TYPES:
                for pts in marker.pts:
                    dash = [list(p) for p in pts]
                    dash[0], dash[1] = dash[1], dash[0] # Flip to be convex
                    dash = np.array(dash)
                    dashes.append(dash)
        return dashes


    # def get_landmarks(self, enu_coords):
    #     """ Returns a Nx3 array of ENU coordinates.
    #     """
    #     landmarks = []

    #     base = Point3d(*enu_coords[:3])
    #     self.tsmap.update_car_pos(*enu_coords[:2], async=False)

    #     roadmarkers = self.tsmap.get_visible_lines_by_range(base, self.front_limit, self.no_oncoming)

    #     for marker in roadmarkers:
    #         if marker.type in SOLID_LINE_TYPES:
    #             for p in marker.pts[0]:
    #                 landmarks.append(list(p))
    #         elif marker.type in DASH_BLOB_TYPES:
    #             for dash in marker.pts:
    #                 for p in dash:
    #                     landmarks.append(list(p))
    #         else:
    #             continue

    #     landmarks = np.array(landmarks)
    #     return landmarks
        