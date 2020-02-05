import numpy as np
import time

from tsmap3 import TSMap, Point3d, LaneSequence, Lane, Roadmarker

SOLID_LINE_TYPES = {Roadmarker.Type.WHITE_SOLID_LINE, Roadmarker.Type.YELLOW_SOLID_LINE}
DASH_BLOB_TYPES = {Roadmarker.Type.WHITE_DASHED_LINE_BLOB, Roadmarker.Type.YELLOW_DASHED_LINE_BLOB}
CURB_TYPES = {Roadmarker.Type.CURB}
SURFACE_MARKING_TYPES = {Roadmarker.Type.ARROW, Roadmarker.Type.CHARACTER}

class MapReader:

    def __init__(self, map_name):
        self.tsmap = TSMap(map_name)
        self.radius = 100
        self.no_oncoming = True

        self.line_thickness = 0.1
    
    def get_road_markers(self, enu_coords):
        base = Point3d(*enu_coords[:3])
        self.tsmap.update_car_pos(*enu_coords[:2], async=False)
        roadmarkers = self.tsmap.get_visible_lines_by_range(base, self.radius, self.no_oncoming)
        return roadmarkers

    def parse_solid_lines(self, roadmarkers):
        roadmarkers = [m for m in roadmarkers if m.type in SOLID_LINE_TYPES]
        lines = [np.array(m.pts[0]) for m in roadmarkers]

        blobs = [line2blob(line, self.line_thickness) for line in lines]
        return blobs

    def parse_curb(self, roadmarkers):
        roadmarkers = [m for m in roadmarkers if m.type in CURB_TYPES]
        lines = [np.array(m.pts[0]) for m in roadmarkers]

        blobs = [line2blob(line, self.line_thickness) for line in lines]
        return blobs

    def parse_dash_lines(self, roadmarkers):
        roadmarkers = [m for m in roadmarkers if m.type in DASH_BLOB_TYPES]
        lines = []
        for m in roadmarkers:
            line = []
            for blob in m.pts:
                blob_pts = np.array(blob)
                pt = np.mean(blob_pts, axis=0)
                line.append(pt)
            line = np.array(line)
            lines.append(line)
        
        blobs = [line2blob(line, self.line_thickness) for line in lines]
        return blobs

    def parse_dash_blobs(self, roadmarkers):
        roadmarkers = [m for m in roadmarkers if m.type in DASH_BLOB_TYPES]

        blobs = []
        for m in roadmarkers:
            for blob in m.pts:
                blob_pts = np.array(blob)
                 # Flip to be convex
                x = blob_pts[0].copy()
                blob_pts[0] = blob_pts[1]
                blob_pts[1] = x
                blobs.append(blob_pts)
        return blobs
    
    def parse_surface_markings(self, roadmarkers):
        roadmarkers = [m for m in roadmarkers if m.type in SURFACE_MARKING_TYPES]
        blobs = [np.array(m.pts[0]) for m in roadmarkers]
        return blobs

def line2blob(line, thickness):
    l0 = []
    l1 = []
    for p0, p1 in zip(line[:-1], line[1:]):
        v = p1 - p0
        v /= np.linalg.norm(v)
        v = np.array([-v[1], v[0], v[2]])
        l0.append(p0 + v * thickness)
        l1.append(p0 - v * thickness)
    blob = np.array(l0 + l1[::-1])
    return blob