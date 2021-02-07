import math

import numpy as np
from shapely.geometry import Polygon,  MultiPoint

def iou(bb, bbgt):
    bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
    iw = bi[2] - bi[0] + 1
    ih = bi[3] - bi[1] + 1
    if iw > 0 and ih > 0:
        # compute overlap (IoU) = area of intersection / area of union
        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                        + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
    
    return iw * ih / ua # ov

def angle2point(b):
    # b = (cx, cy, rw, rh,angle)
    bow_x = b[0] + b[2] / 2 * math.cos(float(b[4]))
    bow_y = b[1] - b[2] / 2 * math.sin(float(b[4]))
    tail_x = b[0] - b[2] / 2 * math.cos(float(b[4]))
    tail_y = b[1] + b[2] / 2 * math.sin(float(b[4]))
    x1 = int(round(bow_x + b[3] / 2 * math.sin(float(b[4]))))
    y1 = int(round(bow_y + b[3] / 2 * math.cos(float(b[4]))))
    x2 = int(round(tail_x + b[3] / 2 * math.sin(float(b[4]))))
    y2 = int(round(tail_y + b[3] / 2 * math.cos(float(b[4]))))
    x3 = int(round(tail_x - b[3] / 2 * math.sin(float(b[4]))))
    y3 = int(round(tail_y - b[3] / 2 * math.cos(float(b[4]))))
    x4 = int(round(bow_x - b[3] / 2 * math.sin(float(b[4]))))
    y4 = int(round(bow_y - b[3] / 2 * math.cos(float(b[4]))))
    return np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]],dtype='float32')

def rbox_iou(a,b):
    b = angle2point(b)
    a = angle2point(a)

    poly1 = Polygon(a).convex_hull  
    poly2 = Polygon(b).convex_hull
    union_poly = np.concatenate((a, b)) 

    if not poly1.intersects(poly2): 
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area  
            # print(inter_area)
            # union_area = poly1.area + poly2.area - inter_area
            union_area = MultiPoint(union_poly).convex_hull.area
            # print(union_area)
            if union_area == 0:
                iou = 0
            # iou = float(inter_area) / (union_area-inter_area)  
            iou = float(inter_area) / union_area
            # iou=float(inter_area) /(poly1.area+poly2.area-inter_area)
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou