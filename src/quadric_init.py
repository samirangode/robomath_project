import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

###Intrinsics Matrix
K = np.array([[0,0,0],[0,0,0],[0,0,0]])

def line_eqs(bounding box):
  x_min = bounding_box[0]
  y_min = bounding_box[1]
  x_max = bounding_box[2]
  y_max = bounding_box[3]
  lines = []
  lines.append(np.array([1,0,-x_min]).T)
  lines.append(np.array([1,0,-x_max]).T)
  lines.append(np.array([0,1,-y_min]).T)
  lines.append(np.array([0,1,-y_max]).T)
  return lines

### getting bounding boxes from manual annotation
path_to_bounding_boxes = "bounding_boxes_manually_annotated.npy"
bounding_boxes_lists_numpy = np.load(path_to_bounding_boxes)
