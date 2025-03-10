import numpy as np
import open3d as o3d

import os
from os import path as osp

import pickle
import glob
import matplotlib.pyplot as plt

from geometry import make_tf, apply_tf

# ---------------------------------------------
# ---------------------------------------------
# --------DO NOT MODIFY BELOW THIS-------------
# ---------------------------------------------
# ---------------------------------------------

CLASS_NAMES = ['car','truck','motorcycle', 'pedestrian']
CLASS_COLORS = plt.cm.rainbow(np.linspace(0, 1, len(CLASS_NAMES)))[:, :3]
CLASS_NAME_TO_COLOR = dict(zip(CLASS_NAMES, CLASS_COLORS))
CLASS_NAME_TO_INDEX = dict(zip(CLASS_NAMES, range(len(CLASS_NAMES))))

# Path extraction
# change the root path to the path of the scenario1 folder
root_path = r"C:/Users/ambra/Downloads/lab0_ierardi_trovatello/open3d_lab-main/scenario1"

scenario = "Town01_type001_subtype0001_scenario00003"  


file_list = glob.glob(osp.join(root_path,
                                        'ego_vehicle', 'label', scenario) + '/*')
frame_list = []

with open(osp.join(root_path, "meta" ,scenario+ '.txt'), 'r') as f:
            lines = f.readlines()
line = lines[2]
agents =  [int(agent) for agent in line.split()[2:]]

for file_path in file_list:
    frame_list.append(file_path.split('/')[-1].split('.')[0].split("\\")[-1])
frame_list.sort()
# ---------------------------------------------
# ---------------------------------------------
# --------DO NOT MODIFY ABOVE THIS-------------
# ---------------------------------------------
# ---------------------------------------------

def get_actor_T_world(actor, n_frame):
    """
    frame = frame_list[n_frame]
    
    with open(osp.join(root_path, actor ,'calib',scenario, frame + '.pkl'), 'rb') as f:
        calib_dict = pickle.load(f)
    actor_tf_world = np.array(calib_dict['ego_to_world'])
    lidar_tf_actor = np.array(calib_dict['lidar_to_ego'])
    
    tf =  lidar_tf_actor @ actor_tf_world 
    trans = tf[:3,3]
    if actor == 'infrastructure':
        trans[2] += 2.0
    rot = tf[:3,:3]

    return make_tf(trans,rot) 

    """
    frame = frame_list[n_frame]
    file_path = osp.normpath(osp.join(root_path, actor, 'calib', scenario, frame + '.pkl'))

    if not osp.exists(file_path):
        raise FileNotFoundError(f"File not found at: {file_path}")
    with open(file_path, 'rb') as f:
        calib_dict = pickle.load(f)
    actor_tf_world = np.array(calib_dict['ego_to_world'])
    return actor_tf_world

def get_sensor_T_actor(actor, n_frame):
    raw_frame = frame_list[n_frame]
    frame = os.path.basename(raw_frame)
    frame = os.path.basename(raw_frame)  # Extract only the file name

    # Construct the path for the calibration file
    calib_path = os.path.join(
        root_path,
        actor,
        'calib',
        scenario,
        frame + '.pkl'
    )

    # Normalize the path
    calib_path = os.path.normpath(calib_path)


    # Check if the file exists
    if not os.path.exists(calib_path):
        print(f"Calibration file does not exist: {calib_path}")
        raise FileNotFoundError(f"Calibration file not found at: {calib_path}")

    # Load the calibration file
    with open(calib_path, 'rb') as f:
        calib_dict = pickle.load(f)

    # Extract the transformation matrix
    lidar_tf_actor = np.array(calib_dict['lidar_to_ego'])

    # Extract translation and rotation components
    tf = lidar_tf_actor
    trans = tf[:3, 3]
    rot = tf[:3, :3]

    # Return the transformation matrix
    return make_tf(trans, rot) 

def get_point_cloud(n_frame, actor):

    raw_frame = frame_list[n_frame]
    frame = os.path.basename(raw_frame)
    # Construct the file path
    lidar_path = os.path.join(
        root_path,
        actor,
        "lidar01",
        scenario,
        frame + ".npz"
    )

    # Normalize the path to use consistent separators
    lidar_path = os.path.normpath(lidar_path)


    # Check if the file exists
    if not os.path.exists(lidar_path):
        print(f"File does not exist: {lidar_path}")
        raise FileNotFoundError(f"Lidar file not found at: {lidar_path}")

    # Load lidar data
    lidar_data = np.load(lidar_path)['data']

    lidar_T_actor = get_sensor_T_actor(actor, n_frame)
    lidar_data_actor = apply_tf(lidar_T_actor, lidar_data) #in actor frame
 
    return lidar_data_actor

def get_available_point_clouds(n_frame, actors):
    '''
    :param n_frame: 
    :param actors:
    :return: (N, 8) - x, y, z, l, w, h, yaw, class

    This function is used to get all point clouds available in ego frame
    '''
    ego_to_world = get_actor_T_world(actors[0], n_frame) # the transformation from ego frame to world frame
    merged_pc = get_point_cloud(n_frame, actors[0]) #in ego frame
    for actor in actors[1:]:
        lidar_data_actor = get_point_cloud(n_frame, actor) # get the point cloud in actor frame
        lidar_data_world = apply_tf(get_actor_T_world(actor, n_frame),lidar_data_actor) # transform them in world frame
        lidar_data_ego = apply_tf(np.linalg.inv(ego_to_world),lidar_data_world) # transform them in ego frame
        merged_pc = np.vstack((merged_pc, lidar_data_ego)) # Merge the point clouds
        pass 
        
        
    return merged_pc

def get_boxes_in_sensor_frame(n_frame, actor):
    # Extract the base file name from frame_list
    raw_frame = frame_list[n_frame]
    frame = os.path.basename(raw_frame)  # Extract only the file name

    # Construct the boxes file path
    boxes_path = os.path.join(
        root_path,
        actor,
        "label",
        scenario,
        frame + ".txt"
    )
    boxes_path = os.path.normpath(boxes_path)


    if not os.path.exists(boxes_path):
        raise FileNotFoundError(f"Boxes file not found at: {boxes_path}")

    # Open and read the boxes data
    with open(boxes_path, 'r') as f:
        lines = f.readlines()

    # Process the box data
    boxes = []
    for line in lines[1:]:  # Skipping the first line if necessary
        line = line.split()
        if line[-1] == 'False':
            continue
        # Convert box values to float and map class name to class index
        box = np.array([float(line[1]), float(line[2]), float(line[3]), float(line[4]), 
                        float(line[5]), float(line[6]), float(line[7]), 
                        CLASS_NAME_TO_INDEX[line[0]]])
        # Format: cx, cy, cz, l, w, h, yaw, class
        boxes.append(box)
    
    return boxes


def get_boxes_in_actor_frame(n_frame, actor):
    '''
    :param n_frame: 
    :param actor:
    :return: (N, 8) - x, y, z, l, w, h, yaw, class

    This function is used to get boxes detected by the actor in actor frame
    '''

    boxes = get_boxes_in_sensor_frame(n_frame, actor)
    boxes = np.array(boxes).reshape(-1,8)

    sensor_to_actor = get_sensor_T_actor(actor, n_frame) # transformation to actor frame
    boxes[:,:3] = apply_tf(sensor_to_actor, boxes[:,:3]) # apply transformation

    return boxes

def get_available_boxes_in_ego_frame(n_frame, actors):
    '''
    :param n_frame: 
    :param actors: a list of actors, the first one is ego vehicle
    :return: (N, 8) - x, y, z, l, w, h, yaw, class

    This function is used to get all available boxes by the actors in ego frame
    '''

    boxes = get_boxes_in_actor_frame(n_frame, actors[0]) #in ego frame
    boxes = np.array(boxes).reshape(-1,8)
    ego_to_world = get_actor_T_world(actors[0], n_frame) # in world frame

    for actor in actors[1:]:
        boxes = get_boxes_in_actor_frame(n_frame, actor) # boxes in actor frame
        boxes = np.array(boxes).reshape(-1,8) 
        actor_to_world = get_actor_T_world(actor, n_frame) # transformnation to world frame
        boxes[:,:3] = apply_tf(actor_to_world, boxes[:,:3]) # boxes transformed in world frame
        boxes[:,:3] = apply_tf(np.linalg.inv(ego_to_world), boxes[:,:3]) # from world to ego frame
    return boxes


def filter_points(points: np.ndarray, range: np.ndarray):
    '''
    points: (N, 3) - x, y, z
    range: (6,) - xmin, ymin, zmin, xmax, ymax, zmax

    return: (M, 3) - x, y, z
    This function is used to filter points within the range
    '''
    # Create a mask to filter the points
    mask = (points[:, 0] >= range[0]) & (points[:, 0] <= range[3]) & \
           (points[:, 1] >= range[1]) & (points[:, 1] <= range[4]) & \
           (points[:, 2] >= range[2]) & (points[:, 2] <= range[5])
    
    filtered_points = points[mask] # Apply the mask to the points

    return filtered_points

