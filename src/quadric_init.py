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
  lines.append(np.array([1, 0, -x_min]).T)
  lines.append(np.array([1, 0, -x_max]).T)
  lines.append(np.array([0, 1, -y_min]).T)
  lines.append(np.array([0, 1, -y_max]).T)
  return lines

### getting bounding boxes from manual annotation
path_to_bounding_boxes = "bounding_boxes_manually_annotated.npy"
bounding_boxes_lists_numpy = np.load(path_to_bounding_boxes)

path_to_trajectories = "traj_vio.csv"
traj_vio = pd.read_csv(path_to_trajectroies)

### Views and Indexes associated
first_index = 128
last_index = 150
first_index_view_one = 69
last_index_view_one = 73

### Populate trajectories
traj = []
for i in range(first_index, last_index+1):
  traj.append(traj_vio.iloc[i])
for j in range(first_index_view_one, last_index_view_one+1):
  traj.append(traj_vio.iloc[i])
  
num_of_images = len(traj)

### get camera position (rotation and location) and camera projection matrices
### Get a Projection matrix P such that it can be used later

### Get lines from bounding boxes
lines_per_box = []
for i in range(last_index-first_index+1):
  lines_per_box.append(line_eqs(bounding_boxes_list_numpy[i]))
for i in range(last_index-first_index+1,num_of_images):
  lines_per_box.append(line_eqs(bounding_boxes_list_numpy[i]))

### Getting Planes
planes = []
for i in range(len(lines_per_box)):
  for j in range(len(lines_per_box[i])):
    planes.append(P[i].T @ lines_per_box[i][j])
planes_numpy = np.array(planes)

for i,p in enumerate(planes):
  temp_plane = p
  pi1 = temp_plane[0]
  pi2 = temp_plane[1]
  pi3 = temp_plane[2]
  pi4 = temp_plane[3]
  A[i] = np.array([pi1**2, 2*pi1*pi2, 2*pi1*pi3, 2*pi1*pi4, pi2**2, 2*pi2*pi3, 2*pi2*pi4, pi3**2, 2*pi3*pi4, pi4**2])
  
u,s,vh = np.linalg.svd(A, full_matrices = True)
b = np.zeros(A.shape[0])

q = vh[:,-1]
Q = np.zeros((4,4))
Q[0,0] = q[0]
Q[0,1] = q[1]
Q[0,2] = q[2]
Q[0,3] = q[3]
Q[1,1] = q[4]
Q[1,2] = q[5]
Q[1,3] = q[6]
Q[2,2] = q[7]
Q[2,3] = q[8]
Q[3,3] = q[9]

Q[1,0] = Q[0,1]
Q[2,0] = Q[0,2]
Q[3,0] = Q[0,3]
Q[2,1] = Q[1,2]
Q[3,1] = Q[1,3]
Q[3,2] = Q[2,3]

Q_primal = np.linalg.inv(Q)

detQ = np.linalg.det(Q_primal)
detQ3x3 = np.linalg.det(Q_primal[:3,:3])

w, v = np.linalg.eig(Q_primal[:3,:3])

w_inv = 1 / w
R_q = v
t_q = np.array([q[3], q[6], q[8]])/q[9]
t_q_new = Q_primal[:3,:3] @ Q_primal[:3,3]

s = np.sqrt(-(detQ/detQ3x3) * w_inv)   
s_new = np.sqrt(np.abs(-(detQ/detQ3x3) * w_inv))

print(s)
print(s_new, "s_new")
print(R_q)
print(t_q)
# print(t_q_new/Q_primal[3,3], "t_q_new")
